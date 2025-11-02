import jax
import jax.numpy as jnp
from jax import random, grad
import equinox as eqx
import optax
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional, Sequence, Callable
from jax import lax

from _physics import u_total, grad_u_total_batch, lap_u_total_batch, u_total_batch
from _state import runtime
from _pyqsc_min import grad_u_forward_batch

def _fourier_embed_xyz(
    xyz: jnp.ndarray,
    bands: Sequence[float],
    scale: float,
    R0: float = 1.0,
    *,
    add_raw: bool = True,
) -> jnp.ndarray:
    """Positional encoding of xyz with optional raw coords."""
    if bands is None or len(bands) == 0:
        return xyz if add_raw else jnp.zeros((*xyz.shape[:-1], 0), dtype=xyz.dtype)
    freqs = jnp.asarray(bands, dtype=xyz.dtype) / (R0 + 1e-12)      # (B,)
    ang = xyz[..., None, :] * (scale * freqs[None, :, None])        # (..., B, 3)
    s, c = jnp.sin(ang), jnp.cos(ang)                               # (..., B, 3)
    parts = []
    if add_raw:
        parts.append(xyz)
    parts += [s.reshape(*xyz.shape[:-1], -1), c.reshape(*xyz.shape[:-1], -1)]
    return jnp.concatenate(parts, axis=-1)


def _full_objective_like_training(params, Pi, Pb, Nb):
    # physics terms
    lap = lap_u_total_batch(params, Pi)
    g_b = grad_u_total_batch(params, Pb)
    n_dot = jnp.sum(Nb * g_b, axis=-1)

    loss_in_raw = jnp.mean(lap * lap)
    loss_bc_raw = jnp.mean(n_dot * n_dot)
    total = loss_in_raw + runtime.lam_bc * loss_bc_raw

    # zero-mean reg on interior
    u_vals = u_total_batch(params, Pi)
    mean_u = jnp.mean(u_vals)
    total = total + float(getattr(runtime, "zero_mean_weight", 0.1)) * (mean_u * mean_u)

    # AL term if enabled
    if runtime.al_enabled:
        c_bc_mean = jnp.mean(n_dot)
        total = total + runtime.al_lambda * c_bc_mean + 0.5 * runtime.al_rho * (c_bc_mean * c_bc_mean)

    return total

@eqx.filter_jit
def eval_total(params, pts_interior, pts_bdry, normals_bdry):
    """Full objective you actually train: physics + zero-mean + AL."""
    # reuse the exact same calc as in _full_objective_like_training
    lap = lap_u_total_batch(params, pts_interior)
    g_b = grad_u_total_batch(params, pts_bdry)
    n_dot = jnp.sum(normals_bdry * g_b, axis=-1)

    loss_in_raw = jnp.mean(lap * lap)
    loss_bc_raw = jnp.mean(n_dot * n_dot)
    total = loss_in_raw + runtime.lam_bc * loss_bc_raw

    # zero-mean reg
    u_vals = u_total_batch(params, pts_interior)
    mean_u = jnp.mean(u_vals)
    total = total + float(getattr(runtime, "zero_mean_weight", 0.1)) * (mean_u * mean_u)

    # AL term (mean constraint)
    if runtime.al_enabled:
        c_bc_mean = jnp.mean(n_dot)
        total = total + runtime.al_lambda * c_bc_mean + 0.5 * runtime.al_rho * (c_bc_mean * c_bc_mean)

    return total


def make_lr_schedule(
    base_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_ratio: float = 0.0,
):
    """
    Linear warmup to base_lr, then cosine decay to base_lr*min_ratio.
    All args are Python numbers (not JAX arrays).
    Returns an Optax schedule function: step -> lr (float).
    """
    warmup_steps = int(max(0, warmup_steps))
    total_steps  = int(max(1, total_steps))
    min_ratio    = float(max(0.0, min(1.0, min_ratio)))

    # Piecewise schedule: warmup then cosine
    warm = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=max(1, warmup_steps),
    )
    # Protect against warmup >= total
    decay_steps = max(1, total_steps - warmup_steps)
    decay = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=decay_steps,
        alpha=min_ratio,  # final fraction of init_value
    )

    def schedule(step):
        # step is an integer jnp.array; route to warmup or decay
        return jax.lax.cond(
            step < warmup_steps,
            lambda _: warm(step),
            lambda _: decay(step - warmup_steps),
            operand=None,
        )
    return schedule

def build_optimizer(
    *,
    base_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_ratio: float = 0.0,
    weight_decay: float = 0.0,
    grad_clip_norm: float = 1.0,
):
    # --- DTYPE harmonization -------------------------------------------------
    # Use the lr's dtype as the authority and cast everything else to it.
    lr_dtype = jnp.asarray(base_lr).dtype
    base_lr_a   = jnp.asarray(base_lr,   dtype=lr_dtype)
    min_ratio_a = jnp.asarray(min_ratio, dtype=lr_dtype)
    one_a       = jnp.asarray(1.0,       dtype=lr_dtype)
    half_a      = jnp.asarray(0.5,       dtype=lr_dtype)
    ten_pct_a   = jnp.asarray(0.1,       dtype=lr_dtype)
    pi_a        = jnp.asarray(jnp.pi,    dtype=lr_dtype)

    total_steps_i = jnp.asarray(total_steps, dtype=jnp.int32)
    warmup_steps_i = jnp.asarray(max(0, int(warmup_steps)), dtype=jnp.int32)

    def schedule(count):
        # count is int32 (Optax); build scalar time as lr_dtype
        count_i = jnp.asarray(count, dtype=jnp.int32)
        t = jnp.asarray(count_i, dtype=lr_dtype)
        T = jnp.asarray(jnp.maximum(total_steps_i, 1), dtype=lr_dtype)
        W = jnp.asarray(jnp.maximum(warmup_steps_i, 1), dtype=lr_dtype)

        # Cosine decay part (with floor alpha = min_ratio)
        phase  = jnp.clip(t / T, 0, 1)
        cosdec = half_a * (one_a + jnp.cos(pi_a * phase))
        lr_cos = base_lr_a * (min_ratio_a + (one_a - min_ratio_a) * cosdec)

        # Linear warmup from 0.1*lr -> 1.0*lr over W steps
        warm_phase = jnp.clip(t / W, 0, 1)
        lr_warm = (ten_pct_a * base_lr_a) + (base_lr_a - ten_pct_a * base_lr_a) * warm_phase

        # Branch-free selection (dtype-safe)
        use_warm = (count_i < warmup_steps_i)
        return jnp.where(use_warm, lr_warm, lr_cos)

    # Clip → AdamW chain
    tx = optax.chain(
        optax.clip_by_global_norm(float(grad_clip_norm)),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=float(weight_decay),
        ),
    )
    return tx, schedule

def _bdry_presample_mult() -> int:
    return int(getattr(runtime, "bdry_presample_mult", 16))

def _zero_mean_w():
    # pull from runtime if present, else default
    return float(getattr(runtime, "zero_mean_weight", 0.1))

def _identity(x): return x

def debug_stats(params, grads, lap_res, nres, s_in=None, s_bc=None):
    """Return small dict of scalars for printing/logging."""
    # param/grads norms (filtered already in train_step; re-filter here if needed)
    p_f, _ = eqx.partition(params, eqx.is_inexact_array)
    g_f, _ = eqx.partition(grads,  eqx.is_inexact_array)

    # Flatten tree norms
    def _l2(tree):
        return np.sqrt(sum([np.asarray((x**2).sum()).item() for x in jax.tree_util.tree_leaves(tree)]))
    def _inf(tree):
        return max([float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(tree)])

    # Helpers that accept either vectors or already-aggregated scalars:
    def _rms_and_max(x):
        if x is None:
            return float("nan"), float("nan")
        if jnp.ndim(x) == 0:   # already an rms-like scalar
            v = float(jnp.abs(x))
            # We don't have a true max when only rms is provided; echo v.
            return v, v
        v_rms = float(jnp.sqrt(jnp.mean(x**2)))
        v_max = float(jnp.max(jnp.abs(x)))
        return v_rms, v_max

    lap_rms, lap_max = _rms_and_max(lap_res)
    nbc_rms, nbc_max = _rms_and_max(nres)

    stats = {
        "param_L2": _l2(p_f),
        "grad_L2": _l2(g_f),
        "grad_Linf": _inf(g_f),
        "lap_rms": lap_rms,
        "lap_max": lap_max,
        "nbc_rms": nbc_rms,
        "nbc_max": nbc_max,
    }
    if s_in is not None: stats["s_in"] = float(s_in)
    if s_bc is not None: stats["s_bc"] = float(s_bc)
    return stats

class PotentialMLP(eqx.Module):
    layers: List[eqx.nn.Linear]
    act: Callable
    use_fourier: bool = False
    fourier_bands: Tuple[float, ...] = ()
    fourier_scale: float = 2 * jnp.pi
    R0_for_fourier: float = 1.0  # optional length scale for posenc

    def __init__(
        self,
        key,
        hidden_sizes: Tuple[int, ...] = (32, 32),
        act: Callable = jax.nn.tanh,
        *,
        use_fourier: bool = False,
        fourier_bands: Sequence[float] = (),
        fourier_scale: float = 2 * jnp.pi,
        R0_for_fourier: float = 1.0,
    ):
        self.use_fourier = bool(use_fourier)
        self.fourier_bands = tuple(float(b) for b in fourier_bands)
        self.fourier_scale = float(fourier_scale)
        self.R0_for_fourier = float(R0_for_fourier)
        self.act = act

        # Compute input dimension with optional Fourier embedding
        in_dim = 3
        if self.use_fourier and len(self.fourier_bands) > 0:
            B = len(self.fourier_bands)
            in_dim = 3 + 2 * 3 * B  # xyz + (sin,cos) for each coord per band

        # Build linear layers for arbitrary widths
        sizes = [in_dim] + list(map(int, hidden_sizes)) + [1]
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [
            eqx.nn.Linear(in_features=sizes[i], out_features=sizes[i + 1], key=keys[i])
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, xyz: jnp.ndarray) -> jnp.ndarray:
        # xyz: (..., 3)
        if self.use_fourier and len(self.fourier_bands) > 0:
            h = _fourier_embed_xyz(xyz, self.fourier_bands, self.fourier_scale, self.R0_for_fourier, add_raw=True)
        else:
            h = xyz

        # Keep original leading batch shape (could be scalar point or batched)
        orig_shape = h.shape[:-1]          # e.g., () or (N,)
        h = h.reshape(-1, h.shape[-1])     # (N, Din) row-major

        # Hidden layers: h = act(h @ W^T + b)
        for layer in self.layers[:-1]:
            W, b = layer.weight, layer.bias            # W: (Dout, Din), b: (Dout,)
            h = h @ W.T + b                            # (N, Dout)
            h = self.act(h)

        # Final linear to scalar: (N, 1)
        Wf, bf = self.layers[-1].weight, self.layers[-1].bias   # (1, Din_last), (1,)
        # Optional sanity check during development:
        # assert Wf.shape[0] == 1, "Final layer must output a scalar."
        h = h @ Wf.T + bf                                       # (N, 1)

        # Restore original batch shape and squeeze scalar
        h = h.reshape(orig_shape + (1,))
        return h.squeeze(-1)            # (...,)

def loss_fn_batch(params, pts_interior, pts_bdry, normals_bdry, key: jax.Array):
    ni, nb = pts_interior.shape[0], pts_bdry.shape[0]
    key_i, key_b = random.split(key)

    # ---- Interior: uniform random subset (Python ints only) ----
    n_in_take = int(min(runtime.BATCH_IN, ni))
    idx_i = random.choice(key_i, ni, (n_in_take,), replace=False)
    Pi = pts_interior[idx_i]

    # ---- Forward gradient matching (interior) ----
    if runtime.lam_grad > 0.0:
        # choose what to match:
        g_model = grad_u_total_batch(params, Pi)         # total model grad
        # If matching NN-only: g_model = grad_u_nn_batch(params, Pi)

        g_target = grad_u_forward_batch(Pi, surf_id=0)   # <= choose how to pass ID; 0 for single-surface
        if not runtime.grad_target_backprop:
            g_target = jax.lax.stop_gradient(g_target)
        loss_grad = jnp.mean(jnp.sum((g_model - g_target)**2, axis=-1))
    else:
        loss_grad = 0.0

    # ---- Boundary: importance sampling with Python ints only ----
    # pre-sample size m and final size K must be Python ints (not JAX arrays)
    # m = int(min(_bdry_presample_mult() * runtime.BATCH_BDRY, nb))
    # m = max(m, 1)  # avoid zero-size
    M = int(_bdry_presample_mult() * runtime.BATCH_BDRY)  # fixed upper bound
    M = max(M, 1)
    key_b_pre, key_b_top = random.split(key_b)

    # pre_idx = random.choice(key_b_pre, nb, (m,), replace=False)
    # Pb_pre, Nb_pre = pts_bdry[pre_idx], normals_bdry[pre_idx]
    replace_flag = (nb < M)
    pre_idx = random.choice(key_b_pre, nb, (M,), replace=replace_flag)
    Pb_pre, Nb_pre = pts_bdry[pre_idx], normals_bdry[pre_idx]

    # estimate residuals on the pre-sample
    gb_pre = grad_u_total_batch(params, Pb_pre)
    r_pre = jnp.sum(Nb_pre * gb_pre, axis=-1) ** 2

    # K = int(min(runtime.BATCH_BDRY, m))
    K = int(min(runtime.BATCH_BDRY, M))
    K = max(K, 1)
    _, topk_idx = lax.top_k(r_pre, K)
    # topk_idx = jnp.argsort(-r_pre)[:K]
    Pb, Nb_hat = Pb_pre[topk_idx], Nb_pre[topk_idx]

    # ---- Zero-mean regularizer ----
    u_vals = u_total_batch(params, Pi)
    mean_u = jnp.mean(u_vals)
    loss_zero_mean = mean_u * mean_u

    # ---- Physics residuals ----
    lap = lap_u_total_batch(params, Pi)
    g_b = gb_pre[topk_idx]
    n_dot = jnp.sum(Nb_hat * g_b, axis=-1)

    loss_in_raw = jnp.mean(lap * lap)
    loss_bc_raw = jnp.mean(n_dot * n_dot)
    
    # --- Augmented Lagrangian for Neumann (mean constraint) ---
    # constraint c := mean(n·∇u) on the boundary batch
    c_bc_mean = jnp.mean(n_dot)

    total = loss_in_raw + runtime.lam_bc * loss_bc_raw + _zero_mean_w() * loss_zero_mean

    if runtime.al_enabled:
        # L_AL = λ * c + 0.5 * ρ * c^2  (added to total loss)
        total = total + runtime.al_lambda * c_bc_mean + 0.5 * runtime.al_rho * (c_bc_mean * c_bc_mean)

    total = total + float(runtime.lam_grad) * loss_grad

    # Return c_bc_mean in aux so the outer loop can update λ
    return total, (loss_in_raw, loss_bc_raw, lap, n_dot, mean_u, c_bc_mean, loss_grad)


# Filtered versions keep non-JAX parts (if any) out of JIT
loss_value_and_grad = eqx.filter_value_and_grad(loss_fn_batch, has_aux=True)

@eqx.filter_jit(donate="all")
def train_step(params, opt_state, optimizer, pts_interior, pts_bdry, normals_bdry, key):
    (loss_val, aux), grads = loss_value_and_grad(params, pts_interior, pts_bdry, normals_bdry, key)
    params_f, params_s = eqx.partition(params, eqx.is_inexact_array)
    grads_f,  _        = eqx.partition(grads,  eqx.is_inexact_array)
    gnorm = optax.global_norm(grads_f)
    updates, opt_state = optimizer.update(grads_f, opt_state, params_f)
    params_f = optax.apply_updates(params_f, updates)
    params = eqx.combine(params_f, params_s)
    return params, opt_state, loss_val, aux, gnorm, grads

@eqx.filter_jit
def eval_full(params, pts_interior, pts_bdry, normals_bdry):
    lap = lap_u_total_batch(params, pts_interior)
    g_b = grad_u_total_batch(params, pts_bdry)
    n_dot = jnp.sum(normals_bdry * g_b, axis=-1)
    loss_in_raw = jnp.mean(lap * lap)
    loss_bc_raw = jnp.mean(n_dot * n_dot)
    total = loss_in_raw + runtime.lam_bc * loss_bc_raw
    return total, (loss_in_raw, loss_bc_raw, lap, n_dot)

def save_model(model, path: Path | str = "pinn_torus_model.eqx"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(path), model)
    print(f"[CKPT] Saved model to: {path}")

def load_model_if_exists(template: eqx.Module, path: Path | str = "pinn_torus_model.eqx") -> eqx.Module:
    path = Path(path)
    if not path.exists():
        print(f"[CKPT] No checkpoint found at {path}. Starting fresh.")
        return template
    try:
        loaded = eqx.tree_deserialise_leaves(str(path), template)

        # Compare only array (inexact) leaves to avoid floats like omega0
        t_arr, _ = eqx.partition(template, eqx.is_inexact_array)
        l_arr, _ = eqx.partition(loaded,   eqx.is_inexact_array)
        t_shapes = [x.shape for x in jax.tree_util.tree_leaves(t_arr)]
        l_shapes = [x.shape for x in jax.tree_util.tree_leaves(l_arr)]

        if (type(loaded) is not type(template)) or (t_shapes != l_shapes):
            print(f"[CKPT] Structure/shape mismatch; ignoring checkpoint {path}.")
            return template

        print(f"[CKPT] Loaded model from: {path}")
        return loaded
    except Exception as e:
        print(f"[CKPT] Failed to load checkpoint ({path}). Using fresh model. Reason: {e}")
        return template

# --- SIREN (minimal) ---------------------------------------------------------
class SirenMLP(eqx.Module):
    layers: List[Tuple[jnp.ndarray, jnp.ndarray]]
    omega0: float
    use_fourier: bool = False
    fourier_bands: Tuple[float, ...] = ()
    fourier_scale: float = 2 * jnp.pi
    R0_for_fourier: float = 1.0
    add_raw_xyz: bool = True

    def __init__(self,
                 key,
                 in_size: int = 3,
                 out_size: int = 1,
                 widths: Tuple[int, ...] = (64, 64, 64),
                 omega0: float = 30.0,
                 *,
                 use_fourier: bool = False,
                 fourier_bands: Sequence[float] = (),
                 fourier_scale: float = 2 * jnp.pi,
                 R0_for_fourier: float = 1.0,
                 add_raw_xyz=True):
        keys = jax.random.split(key, len(widths) + 1)
        self.omega0 = float(omega0)
        self.use_fourier = bool(use_fourier)
        self.fourier_bands = tuple(float(b) for b in fourier_bands)
        self.fourier_scale = float(fourier_scale)
        self.R0_for_fourier = float(R0_for_fourier)
        self.add_raw_xyz = bool(add_raw_xyz)
        in_dim = in_size
        if self.use_fourier and len(self.fourier_bands) > 0:
            B = len(self.fourier_bands)
            in_dim = (3 if self.add_raw_xyz else 0) + 2 * 3 * B

        self.layers = []

        # First layer init: U[-1/in, 1/in]
        W = jax.random.uniform(keys[0], (widths[0], in_dim), minval=-1.0/in_dim, maxval=1.0/in_dim)
        b = jnp.zeros((widths[0],))
        self.layers.append((W, b))

        # Hidden layers init: U[-sqrt(6/in)/omega0, +sqrt(6/in)/omega0]
        for li, width in enumerate(widths[1:], start=1):
            prev = widths[li-1]
            bound = jnp.sqrt(6.0/prev)# / self.omega0
            W = jax.random.uniform(keys[li], (width, prev), minval=-bound, maxval=bound)
            b = jnp.zeros((width,))
            self.layers.append((W, b))

        # Final linear layer (small init)
        last = widths[-1] if len(widths) else in_dim
        W = jax.random.uniform(keys[-1], (out_size, last), minval=-1e-4, maxval=1e-4)
        b = jnp.zeros((out_size,))
        self.layers.append((W, b))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., 3)
        orig_shape = x.shape[:-1]  # batch dims
        if self.use_fourier and len(self.fourier_bands) > 0:
            x = _fourier_embed_xyz(x, self.fourier_bands, self.fourier_scale, self.R0_for_fourier, add_raw=self.add_raw_xyz)

        x = x.reshape(-1, x.shape[-1])  # (N, Din)

        # First layer: sin(omega0 * (W x + b))
        W0, b0 = self.layers[0]
        h = jnp.sin(self.omega0 * (x @ W0.T + b0))

        # Hidden SIREN layers: sin(W h + b)
        for W, b in self.layers[1:-1]:
            h = jnp.sin(h @ W.T + b)

        # Final linear
        Wf, bf = self.layers[-1]
        y = h @ Wf.T + bf  # (N, out_size)
        y = y.reshape(orig_shape + (y.shape[-1],))
        return y.squeeze(-1)
    
def _diagnostics(params, pts_interior, pts_bdry, normals_bdry):
    """Return scalar diagnostics for printing (no JIT)."""
    total, (Lin, Lbc, lap, n_dot) = eval_full(params, pts_interior, pts_bdry, normals_bdry)
    lap_rms = float(jnp.sqrt(jnp.mean(lap**2)))
    lap_max = float(jnp.max(jnp.abs(lap)))
    nbc_rms = float(jnp.sqrt(jnp.mean(n_dot**2)))
    nbc_max = float(jnp.max(jnp.abs(n_dot)))
    return float(total), float(Lin), float(Lbc), lap_rms, lap_max, nbc_rms, nbc_max

# ======================= Many-surfaces (train all at once) ===================

def _pick_interior_batch(P_in_all: jnp.ndarray, K: int, key: jax.Array) -> jnp.ndarray:
    """Uniform (w/o replacement) interior sampling from fixed per-surface pool."""
    n = int(P_in_all.shape[0])
    if n == 0:
        return P_in_all  # empty
    k = int(min(K, n))
    idx = random.choice(key, n, (k,), replace=False)
    return P_in_all[idx]

def _param_l2(tree) -> jnp.ndarray:
    return sum([jnp.sum(x*x) for x in jax.tree_util.tree_leaves(tree)])

def _loss_many_single_surface(params,
                              Pi_all: jnp.ndarray,
                              Pb_all: jnp.ndarray,
                              Nb_all: jnp.ndarray,
                              surf_id: int,
                              key: jax.Array) -> tuple[jnp.ndarray, tuple]:
    """One-surface contribution (sampling + residuals)."""
    ki, kb = random.split(key)
    # --- interior
    Pi = _pick_interior_batch(Pi_all, runtime.BATCH_IN, ki)

    if runtime.lam_grad > 0.0:
        g_model = grad_u_total_batch(params, Pi)             # or grad_u_nn_batch
        g_target = grad_u_forward_batch(Pi, surf_id=surf_id)
        if not runtime.grad_target_backprop:
            g_target = jax.lax.stop_gradient(g_target)
        loss_grad = jnp.mean(jnp.sum((g_model - g_target)**2, axis=-1))
    else:
        loss_grad = 0.0

    # --- boundary (importance)
    nb = int(Pb_all.shape[0])
    if nb > 0:
        k_pre, k_top = random.split(kb)
        M = int(_bdry_presample_mult() * runtime.BATCH_BDRY)  # fixed upper bound
        M = max(M, 1)
        replace_flag = (nb < M)  # keeps shape static if nb<M
        pre_idx = random.choice(k_pre, nb, (M,), replace=replace_flag)
        Pb_pre, Nb_pre = Pb_all[pre_idx], Nb_all[pre_idx]

        gb_pre = grad_u_total_batch(params, Pb_pre)
        r_pre = jnp.sum(Nb_pre * gb_pre, axis=-1) ** 2
        K = int(min(runtime.BATCH_BDRY, M))
        K = max(K, 1)
        _, topk_idx = lax.top_k(r_pre, K)

        Pb, Nb_hat = Pb_pre[topk_idx], Nb_pre[topk_idx]
        g_b = gb_pre[topk_idx]  # <— reuse, NO extra grad pass
    else:
        # No boundary points: fall back (array may be empty; this is fine)
        Pb, Nb_hat = Pb_all, Nb_all
        g_b = grad_u_total_batch(params, Pb)  # harmless if size 0

    # --- zero-mean reg on interior
    u_vals = u_total_batch(params, Pi)
    mean_u = jnp.mean(u_vals)
    loss_zero = mean_u * mean_u

    # --- physics residuals
    lap = lap_u_total_batch(params, Pi)
    n_dot = jnp.sum(Nb_hat * g_b, axis=-1)

    loss_in_raw = jnp.mean(lap * lap)
    loss_bc_raw = jnp.mean(n_dot * n_dot)
    c_bc_mean = jnp.mean(n_dot)

    total = loss_in_raw + runtime.lam_bc * loss_bc_raw + _zero_mean_w() * loss_zero
    
    if runtime.al_enabled:
        total = total + runtime.al_lambda * c_bc_mean + 0.5 * runtime.al_rho * (c_bc_mean * c_bc_mean)

    total = total + float(runtime.lam_grad) * loss_grad

    return total, (loss_in_raw, loss_bc_raw, lap, n_dot, mean_u, c_bc_mean, loss_grad)

def loss_many(params,
              packs: tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int], ...],
              key: jax.Array):
    """
    packs: tuple of (P_in_all, P_b_all, N_b_all) for each surface (fixed arrays).
    Returns mean over surfaces to keep scale stable, with aux aggregated (means).
    """
    keys = random.split(key, len(packs))
    totals, auxes = [], []
    for (Pi_all, Pb_all, Nb_all, surf_id), k in zip(packs, keys):
        t, aux = _loss_many_single_surface(params, Pi_all, Pb_all, Nb_all, int(surf_id), k)
        totals.append(t); auxes.append(aux)

    total = sum(totals) / float(len(packs))

    def _rms_from_chunks(chunks):
        if len(chunks) == 0:
            return jnp.array(0.0)
        ss = sum(jnp.sum(x*x) for x in chunks)
        n  = sum(x.size for x in chunks)
        return jnp.sqrt(ss / jnp.maximum(1, n))

    Lin_mean  = sum(a[0] for a in auxes) / float(len(auxes))
    Lbc_mean  = sum(a[1] for a in auxes) / float(len(auxes))
    lap_rms   = _rms_from_chunks([a[2] for a in auxes])
    nbc_rms   = _rms_from_chunks([a[3] for a in auxes])
    mean_u    = sum(a[4] for a in auxes) / float(len(auxes))
    c_bc_mean = sum(a[5] for a in auxes) / float(len(auxes))
    Lg_mean   = sum(a[6] for a in auxes) / float(len(auxes))  # now valid
    return total, (Lin_mean, Lbc_mean, lap_rms, nbc_rms, mean_u, c_bc_mean, Lg_mean)

loss_many_value_and_grad = eqx.filter_value_and_grad(loss_many, has_aux=True)

@eqx.filter_jit(donate="all")
def train_step_many(params,
                    opt_state,
                    optimizer,
                    packs: tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int], ...],
                    key: jax.Array):
    (loss_val, aux), grads = loss_many_value_and_grad(params, packs, key)
    params_f, params_s = eqx.partition(params, eqx.is_inexact_array)
    grads_f, _         = eqx.partition(grads,  eqx.is_inexact_array)
    gnorm = optax.global_norm(grads_f)
    updates, opt_state = optimizer.update(grads_f, opt_state, params_f)
    params_f = optax.apply_updates(params_f, updates)
    params = eqx.combine(params_f, params_s)
    return params, opt_state, loss_val, aux, gnorm, grads
