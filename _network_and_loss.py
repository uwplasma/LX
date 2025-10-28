import jax
import jax.numpy as jnp
from jax import random, grad
import equinox as eqx
import optax
from pathlib import Path
from _state import runtime 
import numpy as np
from typing import List, Tuple

from _physics import u_total, grad_u_total_batch, lap_u_total_batch, u_total_batch

ZERO_MEAN_WEIGHT = 0.1  # small regularization

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

    stats = {
        "param_L2": _l2(p_f),
        "grad_L2": _l2(g_f),
        "grad_Linf": _inf(g_f),
        "lap_rms": float(jnp.sqrt(jnp.mean(lap_res**2)) ),
        "lap_max": float(jnp.max(jnp.abs(lap_res)) ),
        "nbc_rms": float(jnp.sqrt(jnp.mean(nres**2)) ),
        "nbc_max": float(jnp.max(jnp.abs(nres)) ),
    }
    if s_in is not None: stats["s_in"] = float(s_in)
    if s_bc is not None: stats["s_bc"] = float(s_bc)
    return stats

class PotentialMLP(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, key, hidden_sizes=(32, 32), act=jax.nn.tanh):
        depth = len(hidden_sizes) if hidden_sizes else 1
        width_val = (hidden_sizes[0] if hidden_sizes else 64)
        self.mlp = eqx.nn.MLP(
            in_size=3, out_size=1,
            width_size=width_val,
            depth=depth,
            key=key,
            activation=act,
            final_activation=_identity,
        )
    def __call__(self, xyz: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(xyz).squeeze(-1)

def loss_fn_batch(params, pts_interior, pts_bdry, normals_bdry, key: jax.Array):
    ni, nb = pts_interior.shape[0], pts_bdry.shape[0]
    key_i, key_b = random.split(key)

    # ---- Interior: uniform random subset (Python ints only) ----
    n_in_take = int(min(runtime.BATCH_IN, ni))
    idx_i = random.choice(key_i, ni, (n_in_take,), replace=False)
    Pi = pts_interior[idx_i]

    # ---- Boundary: importance sampling with Python ints only ----
    # pre-sample size m and final size K must be Python ints (not JAX arrays)
    m = int(min(16 * runtime.BATCH_BDRY, nb))
    m = max(m, 1)  # avoid zero-size
    key_b_pre, key_b_top = random.split(key_b)

    pre_idx = random.choice(key_b_pre, nb, (m,), replace=False)
    Pb_pre, Nb_pre = pts_bdry[pre_idx], normals_bdry[pre_idx]

    # estimate residuals on the pre-sample
    gb_pre = grad_u_total_batch(params, Pb_pre)
    r_pre = jnp.sum(Nb_pre * gb_pre, axis=-1) ** 2

    K = int(min(runtime.BATCH_BDRY, m))
    K = max(K, 1)
    topk_idx = jnp.argsort(-r_pre)[:K]
    Pb, Nb_hat = Pb_pre[topk_idx], Nb_pre[topk_idx]

    # ---- Zero-mean regularizer ----
    u_vals = u_total_batch(params, Pi)
    mean_u = jnp.mean(u_vals)
    loss_zero_mean = mean_u * mean_u

    # ---- Physics residuals ----
    lap = lap_u_total_batch(params, Pi)
    g_b = grad_u_total_batch(params, Pb)
    n_dot = jnp.sum(Nb_hat * g_b, axis=-1)

    loss_in_raw = jnp.mean(lap * lap)
    loss_bc_raw = jnp.mean(n_dot * n_dot)

    loss_in = loss_in_raw
    loss_bc = loss_bc_raw

    total = loss_in + runtime.lam_bc * loss_bc + ZERO_MEAN_WEIGHT * loss_zero_mean
    return total, (loss_in_raw, loss_bc_raw, lap, n_dot, mean_u)


# Filtered versions keep non-JAX parts (if any) out of JIT
loss_value_and_grad = eqx.filter_value_and_grad(loss_fn_batch, has_aux=True)

@eqx.filter_jit
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
    if path.exists():
        try:
            loaded = eqx.tree_deserialise_leaves(str(path), template)
            print(f"[CKPT] Loaded model from: {path}")
            return loaded
        except Exception as e:
            print(f"[CKPT] Failed to load checkpoint ({path}). Using fresh model. Reason: {e}")
    else:
        print(f"[CKPT] No checkpoint found at {path}. Starting fresh.")
    return template

# --- SIREN (minimal) ---------------------------------------------------------
class SirenMLP(eqx.Module):
    layers: List[Tuple[jnp.ndarray, jnp.ndarray]]
    omega0: float

    def __init__(self, key, in_size=3, out_size=1, widths=(64, 64, 64), omega0=30.0):
        keys = jax.random.split(key, len(widths) + 1)
        self.omega0 = float(omega0)
        self.layers = []

        # First layer init: U[-1/in, 1/in]
        in_dim = in_size
        w_key = keys[0]
        W = jax.random.uniform(w_key, (widths[0], in_dim), minval=-1.0/in_dim, maxval=1.0/in_dim)
        b = jnp.zeros((widths[0],))
        self.layers.append((W, b))

        # Hidden layers init: U[-sqrt(6/in)/omega0, +sqrt(6/in)/omega0]
        for li, width in enumerate(widths[1:], start=1):
            prev = widths[li-1]
            w_key = keys[li]
            bound = jnp.sqrt(6.0/prev) / self.omega0
            W = jax.random.uniform(w_key, (width, prev), minval=-bound, maxval=bound)
            b = jnp.zeros((width,))
            self.layers.append((W, b))

        # Final linear layer (small init)
        w_key = keys[-1]
        last = widths[-1] if len(widths) else in_size
        W = jax.random.uniform(w_key, (out_size, last), minval=-1e-4, maxval=1e-4)
        b = jnp.zeros((out_size,))
        self.layers.append((W, b))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., 3)
        # First layer: sin(omega0 * (W x + b))
        W0, b0 = self.layers[0]
        h = jnp.sin(self.omega0 * (x @ W0.T + b0))

        # Hidden SIREN layers: sin(W h + b)
        for W, b in self.layers[1:-1]:
            h = jnp.sin(h @ W.T + b)

        # Final linear
        Wf, bf = self.layers[-1]
        y = h @ Wf.T + bf
        return y.squeeze(-1)