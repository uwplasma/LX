#!/usr/bin/env python3
"""
PINN for Laplace's equation ∇²u = 0 inside a (possibly non-axisymmetric) torus
in Cartesian coordinates, with a soft Neumann boundary condition n·∇u ≈ 0.

u_total(x,y,z) = u_mv(x,y,z) + u_nn(x,y,z)
  - u_mv is a multi-valued piece: kappa * atan2(y,x) / R0  (analytic grad, zero Laplacian away from axis)
  - u_nn is a single-valued network (Equinox MLP) fed with geometry-aware + Fourier features.

Training:
  - Least-squares objective: ⟨(∇²u)^2⟩_interior + λ_bc ⟨(n·∇u)^2⟩_boundary
  - AdamW (warmup→cosine decay) + Lookahead + gradient clipping
  - EMA parameters for evaluation
  - Minibatching, residual-based adaptive resampling (RBAR), loss balancing, λ_bc curriculum

End:
  - Deterministic evaluation on full sets
  - Plot |∇u| on boundary (θ×φ grid)
"""

from __future__ import annotations
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad
import optax
import equinox as eqx
import matplotlib.pyplot as plt

# =============================================================================
# ============================= USER PARAMETERS ===============================
# =============================================================================

# ----- Geometry (torus example) -----
R0      = 1.0           # Major radius (centerline in xy-plane)
a0      = 0.35          # Minor-radius base
a1      = 0.10          # Minor-radius modulation amplitude
N_harm  = 3             # Mode in a(φ) = a0 + a1 cos(N_harm φ)

# ----- Multi-valued component -----
kappa   = 1.0           # u_mv = kappa * atan2(y,x) / R0; set 0 for single-valued

# ----- Sampling -----
N_in            = 10_000   # interior sample count (total pool; minibatches are drawn from here)
N_bdry_theta    = 64       # boundary θ resolution
N_bdry_phi      = 64      # boundary φ resolution
rng_seed        = 0

# ----- Batching -----
BATCH_IN   = 2048    # interior points per step
BATCH_BDRY = 2048    # boundary points per step

# ----- Features -----
FOURIER_M  = 3       # number of φ harmonics (adds 2*M features: sin mφ, cos mφ)
USE_INPUT_NORM = True

# ----- Network (Equinox) -----
MLP_HIDDEN_SIZES = (32, 32)   # width/depth
MLP_ACT          = jax.nn.silu  # jax.nn.tanh also fine

# ----- Optimization -----
steps       = 4000
lr          = 3e-3
lam_bc_base = 5.0               # baseline BC weight
warmup_frac = 0.1               # warmup portion for LR schedule
clip_norm   = 1.0               # global grad clip
ema_decay   = 0.999             # EMA decay for evaluation
lookahead_sync = 5
lookahead_slow  = 0.5

# ----- RBAR -----
RBAR_PERIOD = 200               # resample interior every this many steps
RBAR_FRAC   = 0.5               # fraction of interior points to replace

# ----- Plotting -----
PLOT_CMAP = "viridis"
FIGSIZE   = (8, 4.5)

# ----- Optional LBFGS polish -----
USE_LBFGS_FINAL = False   # set True to try LBFGS at the end (requires jaxopt)

# =============================================================================
# =============================== GEOMETRY ====================================
# =============================================================================

def _identity(x):
    return x

def a_of_phi(phi: jnp.ndarray) -> jnp.ndarray:
    return a0 + a1 * jnp.cos(N_harm * phi)

def cylindrical_phi(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(y, x)

def inside_torus_mask(x, y, z) -> jnp.ndarray:
    r   = jnp.sqrt(x*x + y*y)
    phi = cylindrical_phi(x, y)
    rho = jnp.sqrt((r - R0)**2 + z*z)        # distance to circular axis at angle φ
    return rho <= a_of_phi(phi)

def build_surface_torus(n_theta: int, n_phi: int):
    theta = jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False)
    phi   = jnp.linspace(0, 2*jnp.pi, n_phi,   endpoint=False)
    Θ, Φ  = jnp.meshgrid(theta, phi, indexing='ij')    # [nθ,nφ]
    aφ    = a_of_phi(Φ)
    Rring = R0 + aφ * jnp.cos(Θ)
    X = Rring * jnp.cos(Φ)
    Y = Rring * jnp.sin(Φ)
    Z = aφ * jnp.sin(Θ)
    return X, Y, Z

def normals_from_param_grid(X, Y, Z):
    def circ_diff(A, axis):
        return (jnp.roll(A, -1, axis=axis) - jnp.roll(A, 1, axis=axis)) * 0.5
    dX_dθ, dY_dθ, dZ_dθ = circ_diff(X,0), circ_diff(Y,0), circ_diff(Z,0)
    dX_dφ, dY_dφ, dZ_dφ = circ_diff(X,1), circ_diff(Y,1), circ_diff(Z,1)
    tθ = jnp.stack([dX_dθ, dY_dθ, dZ_dθ], axis=-1)  # [nθ,nφ,3]
    tφ = jnp.stack([dX_dφ, dY_dφ, dZ_dφ], axis=-1)
    n  = jnp.cross(tθ, tφ, axis=-1)
    n_norm = jnp.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    n_hat  = n / n_norm
    return n_hat

def surface_points_and_normals(nθ, nφ):
    X, Y, Z = build_surface_torus(nθ, nφ)
    Nhat    = normals_from_param_grid(X, Y, Z)
    P       = jnp.stack([X, Y, Z], axis=-1)
    return (P.reshape(-1,3), Nhat.reshape(-1,3), X, Y, Z)  # flattened + original grids

def sample_interior(key, n_points: int, oversample_factor: int = 8):
    """Single-shot JAX sampler: oversample once, keep first n_points valid points."""
    def _one_shot(key, n_points, factor):
        a_max = a0 + jnp.abs(a1)
        Lxy   = R0 + a_max
        M     = factor * n_points
        kx, ky, kz = random.split(key, 3)
        X = random.uniform(kx, (M,), minval=-Lxy,  maxval=Lxy)
        Y = random.uniform(ky, (M,), minval=-Lxy,  maxval=Lxy)
        Z = random.uniform(kz, (M,), minval=-a_max, maxval=a_max)
        mask = inside_torus_mask(X, Y, Z)                 # [M] boolean
        idx  = jnp.nonzero(mask, size=n_points, fill_value=0)[0]  # [n_points]
        got  = jnp.sum(mask)
        pts  = jnp.stack([X[idx], Y[idx], Z[idx]], axis=-1)       # [n_points,3]
        return pts, got
    pts, got = _one_shot(key, n_points, oversample_factor)
    if int(got) < n_points:
        pts, _ = _one_shot(key, n_points, oversample_factor * 2)
    return pts

# =============================================================================
# ============================ FEATURES & MODEL ===============================
# =============================================================================

def feature_map(xyz: jnp.ndarray, scale: float, Mphi: int = FOURIER_M) -> jnp.ndarray:
    """
    Geometry-aware + Fourier features (toroidal):
      base: [x/scale, y/scale, z/scale, (r-R0)/scale]
      fourier: [sin mφ, cos mφ] for m=1..Mphi
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = jnp.sqrt(x*x + y*y)
    phi = jnp.arctan2(y, x)
    base = [x/scale, y/scale, z/scale, (r - R0)/scale]
    if Mphi > 0:
        m = jnp.arange(1, Mphi + 1, dtype=phi.dtype)   # <-- use phi.dtype
        mphi = phi[..., None] * m                       # <-- no [None, ...] on m
        s = jnp.sin(mphi)                               # shapes (..., M)
        c = jnp.cos(mphi)
        fourier = jnp.concatenate([s, c], axis=-1)      # (..., 2M)
        base_stacked = jnp.stack(base, axis=-1)         # (..., 4)
        feats = jnp.concatenate([base_stacked, fourier], axis=-1)  # (..., 4+2M)
    else:
        feats = jnp.stack(base, axis=-1)
    return feats

class TorusNet(eqx.Module):
    mlp: eqx.nn.MLP
    scale: float
    Mphi: int

    def __init__(self, key, hidden_sizes=MLP_HIDDEN_SIZES, act=MLP_ACT, scale=1.0, Mphi=FOURIER_M):
        depth = len(hidden_sizes) if hidden_sizes else 1
        width_val = (hidden_sizes[0] if hidden_sizes else 64)
        # Determine feature dimension
        in_size = 4 + 2*Mphi  # [x,y,z,(r-R0)] + 2*Mphi
        # Equinox API differences: width vs width_size
        try:
            mlp = eqx.nn.MLP(
                in_size=in_size, out_size=1,
                width=width_val, depth=depth, key=key,
                activation=act, final_activation=_identity
            )
        except TypeError:
            mlp = eqx.nn.MLP(
                in_size=in_size, out_size=1,
                width_size=width_val, depth=depth, key=key,
                activation=act, final_activation=_identity
            )
        self.mlp = mlp
        self.scale = float(scale)
        self.Mphi = int(Mphi)

    def __call__(self, xyz: jnp.ndarray) -> jnp.ndarray:
        feats = feature_map(xyz, self.scale, self.Mphi)
        out = self.mlp(feats)
        return out.squeeze(-1)  # scalar

# =============================================================================
# ========================= ANALYTIC MULTI-VALUED PART ========================
# =============================================================================

def u_multivalued(xyz: jnp.ndarray) -> jnp.ndarray:
    x, y = xyz[..., 0], xyz[..., 1]
    return kappa * jnp.arctan2(y, x) / R0

def grad_u_mv(xyz: jnp.ndarray) -> jnp.ndarray:
    """∇(kappa * atan2(y,x)/R0) = kappa/R0 * (-y/(x^2+y^2), x/(x^2+y^2), 0)."""
    x, y = xyz[..., 0], xyz[..., 1]
    r2 = x*x + y*y
    inv = jnp.where(r2 > 1e-18, 1.0 / r2, 0.0)  # guard near axis
    gx = -y * inv
    gy =  x * inv
    gz = jnp.zeros_like(x)
    return (kappa / R0) * jnp.stack([gx, gy, gz], axis=-1)

@eqx.filter_jit
def lap_u_mv_zero(_params, _xyz):
    return jnp.array(0.0)

def u_total(params: TorusNet, xyz: jnp.ndarray) -> jnp.ndarray:
    return u_multivalued(xyz) + params(xyz)

# =============================================================================
# ============================ DERIVATIVES (FAST) =============================
# =============================================================================

@eqx.filter_jit
def grad_u_nn_scalar(params: TorusNet, xyz: jnp.ndarray) -> jnp.ndarray:
    return grad(lambda q: params(q))(xyz)

# Laplacian via three Hessian–vector products (no full Hessian materialization)
E1 = jnp.array([1.0, 0.0, 0.0])
E2 = jnp.array([0.0, 1.0, 0.0])
E3 = jnp.array([0.0, 0.0, 1.0])

@eqx.filter_jit
def lap_u_nn_scalar(params: TorusNet, xyz: jnp.ndarray) -> jnp.ndarray:
    def g(q): return grad_u_nn_scalar(params, q)
    e1 = jnp.array([1.0, 0.0, 0.0], dtype=xyz.dtype)
    e2 = jnp.array([0.0, 1.0, 0.0], dtype=xyz.dtype)
    e3 = jnp.array([0.0, 0.0, 1.0], dtype=xyz.dtype)
    _, He1 = jax.jvp(g, (xyz,), (e1,))
    _, He2 = jax.jvp(g, (xyz,), (e2,))
    _, He3 = jax.jvp(g, (xyz,), (e3,))
    return He1[0] + He2[1] + He3[2]

grad_u_nn_batch = jax.vmap(grad_u_nn_scalar, in_axes=(None, 0))
lap_u_nn_batch  = jax.vmap(lap_u_nn_scalar,  in_axes=(None, 0))

@eqx.filter_jit
def grad_u_total_batch(params: TorusNet, xyz_batch: jnp.ndarray) -> jnp.ndarray:
    return grad_u_mv(xyz_batch) + grad_u_nn_batch(params, xyz_batch)

@eqx.filter_jit
def lap_u_total_batch(params: TorusNet, xyz_batch: jnp.ndarray) -> jnp.ndarray:
    # Lap(u_mv)=0 away from axis ⇒ Lap(u_total)=Lap(u_nn)
    return lap_u_nn_batch(params, xyz_batch)

# =============================================================================
# ============================ LOSS & OPTIMIZER ===============================
# =============================================================================

def loss_fn_batch(params: TorusNet,
                  pts_interior: jnp.ndarray, pts_bdry: jnp.ndarray,
                  normals_bdry: jnp.ndarray,
                  key: jax.Array,
                  lam_eff: float,
                  bal_in: float, bal_bc: float):
    """Minibatch loss with balancing and dynamic λ_bc."""
    ni = pts_interior.shape[0]
    nb = pts_bdry.shape[0]
    key_i, key_b = random.split(key)
    idx_i = random.choice(key_i, ni, (min(BATCH_IN, ni),), replace=False)
    idx_b = random.choice(key_b, nb, (min(BATCH_BDRY, nb),), replace=False)

    Pi = pts_interior[idx_i]
    Pb = pts_bdry[idx_b]
    Nb_hat = normals_bdry[idx_b]

    lap = lap_u_total_batch(params, Pi)            # [Bi]
    g_b = grad_u_total_batch(params, Pb)           # [Bb,3]
    n_dot = jnp.sum(Nb_hat * g_b, axis=-1)

    loss_in = jnp.mean(lap * lap)
    loss_bc = jnp.mean(n_dot * n_dot)
    total   = bal_in * loss_in + lam_eff * bal_bc * loss_bc
    return total, (loss_in, loss_bc, lap, n_dot)

# Filtered value+grad
loss_value_and_grad = eqx.filter_value_and_grad(loss_fn_batch, has_aux=True)

@eqx.filter_jit
def train_step(params, opt_state, optimizer,
               pts_interior, pts_bdry, normals_bdry,
               key, lam_eff, bal_in, bal_bc):
    (loss_val, aux), grads = loss_value_and_grad(
        params, pts_interior, pts_bdry, normals_bdry, key, lam_eff, bal_in, bal_bc
    )
    
    params_f, params_s = eqx.partition(params, eqx.is_inexact_array)
    grads_f,  _        = eqx.partition(grads,  eqx.is_inexact_array)

    updates, opt_state = optimizer.update(grads_f, opt_state, params_f)
    params_f = optax.apply_updates(params_f, updates)

    params = eqx.combine(params_f, params_s)
    return params, opt_state, loss_val, aux

@eqx.filter_jit
def eval_full(params: TorusNet,
              pts_interior: jnp.ndarray, pts_bdry: jnp.ndarray,
              normals_bdry: jnp.ndarray):
    """Deterministic full-set evaluation."""
    lap = lap_u_total_batch(params, pts_interior)
    g_b = grad_u_total_batch(params, pts_bdry)
    n_dot = jnp.sum(normals_bdry * g_b, axis=-1)
    loss_in = jnp.mean(lap * lap)
    loss_bc = jnp.mean(n_dot * n_dot)
    total   = loss_in + lam_bc_base * loss_bc
    return total, (loss_in, loss_bc, lap, n_dot)

def lam_bc_schedule(it: int, total_steps: int, base: float) -> float:
    """Smooth curriculum for λ_bc: starts higher then decays toward base."""
    t = it / max(1, total_steps)
    return float(base * (0.3 + 0.7 * jnp.exp(-5.0 * (t - 0.2))))

class _FastSlowBox(eqx.Module):
    fast: object
    slow: object

def make_optimizer():
    warmup_steps = max(1, int(warmup_frac * steps))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=max(1, steps - warmup_steps),
        end_value=lr * 0.05,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(schedule, weight_decay=0.0),
    )
    return optimizer

def _slow_from_state(state):
    # Optax Lookahead changed field names across versions.
    # Try the common ones in order.
    if hasattr(state, "slow"):
        return state.slow
    if hasattr(state, "slow_params"):
        return state.slow_params
    # Some very old versions nested inside .params; keep as a safe fallback.
    if hasattr(state, "params") and hasattr(state.params, "slow"):
        return state.params.slow
    raise AttributeError("LookaheadState has no slow/slow_params attribute.")

def ema_update(ema_p, new_p, decay=ema_decay):
    # update array leaves; keep non-arrays
    arr_ema, non_ema = eqx.partition(ema_p, eqx.is_inexact_array)
    arr_new, _       = eqx.partition(new_p,  eqx.is_inexact_array)
    arr_upd = optax.incremental_update(arr_new, arr_ema, step_size=1.0 - decay)
    return eqx.combine(arr_upd, non_ema)

def resample_interior_by_residual(key, params: TorusNet, P_in: jnp.ndarray, frac=0.5):
    """RBAR: keep some uniform, replace rest with residual-weighted picks."""
    Ni = P_in.shape[0]
    lap = jnp.abs(lap_u_total_batch(params, P_in))
    w = lap / (jnp.mean(lap) + 1e-12)
    w = w / (jnp.sum(w) + 1e-12)
    k1, k2 = random.split(key)
    keep = int((1.0 - frac) * Ni)
    add  = Ni - keep
    idx_keep = random.choice(k1, Ni, (keep,), replace=False)
    idx_add  = random.choice(k2, Ni, (add,),  replace=True, p=w)
    return jnp.concatenate([P_in[idx_keep], P_in[idx_add]], axis=0)

# =============================================================================
# ================================== MAIN =====================================
# =============================================================================

def main():
    print("=== PINN Laplace on Torus ===")
    print(f"Major radius R0={R0:.3f}")
    print(f"Minor radius a(φ)=a0 + a1 cos(Nφ) with a0={a0:.3f}, a1={a1:.3f}, N={N_harm}")
    print(f"Multi-valued kappa={kappa:.3f}  (set 0 for single-valued potential)")
    print(f"Interior samples: {N_in}, Boundary grid: θ={N_bdry_theta}, φ={N_bdry_phi}")
    print(f"Network: hidden={MLP_HIDDEN_SIZES}, act={MLP_ACT.__name__}")
    print(f"Features: base+[sin/cos mφ for m=1..{FOURIER_M}], normalized={USE_INPUT_NORM}")
    print(f"Training steps: {steps}, lr={lr}, λ_bc(base)={lam_bc_base}")
    sys.stdout.flush()

    # RNG
    key = random.PRNGKey(rng_seed)
    key, k_model, k_in = random.split(key, 3)

    # Boundary points and normals
    P_bdry, N_bdry, Xg, Yg, Zg = surface_points_and_normals(N_bdry_theta, N_bdry_phi)
    Nb = P_bdry.shape[0]
    # Quick normal sanity + auto-flip if needed
    Rvec = jnp.stack([Xg, Yg, jnp.zeros_like(Zg)], axis=-1)
    Rhat = Rvec / (jnp.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-12)
    mean_out = jnp.mean(jnp.sum(N_bdry.reshape(Xg.shape + (3,)) * Rhat, axis=-1))
    mean_out_val = float(mean_out)
    print(f"[DEBUG] Mean outwardness of normals (dot with radial unit): {mean_out_val:+.4f} (≈positive expected)")
    if mean_out_val < 0:
        N_bdry = -N_bdry
        print("[DEBUG] Normals flipped to ensure outward orientation.")

    # Interior points
    P_in = sample_interior(k_in, N_in)
    Ni = P_in.shape[0]
    print(f"[DEBUG] Sampled {Ni} interior points, {Nb} boundary points.")

    print(f"[CHECK] P_in shape={P_in.shape}, P_bdry shape={P_bdry.shape}, N_bdry shape={N_bdry.shape}")
    print(f"[CHECK] Any NaNs? interior={bool(jnp.isnan(P_in).any())}, boundary={bool(jnp.isnan(P_bdry).any())}, normals={bool(jnp.isnan(N_bdry).any())}")
    nb_norms = jnp.linalg.norm(N_bdry, axis=-1)
    print(f"[CHECK] Normals |n| stats: min={float(nb_norms.min()):.3e}, max={float(nb_norms.max()):.3e}, mean={float(nb_norms.mean()):.3e}")

    # Input normalization scale
    scale = float(R0 + a0 + abs(a1)) if USE_INPUT_NORM else 1.0

    # Model + optimizer + EMA
    model = TorusNet(k_model, hidden_sizes=MLP_HIDDEN_SIZES, act=MLP_ACT, scale=scale, Mphi=FOURIER_M)
    optimizer = make_optimizer()
    fast_params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optimizer.init(fast_params)
    ema_params = jax.tree_util.tree_map(lambda x: x, model)  # copy

    # Initial loss check (raw + EMA)
    (L0, (Lin0, Lbc0, lap0, n0)) = eval_full(model, P_in, P_bdry, N_bdry)
    (L0e, (Lin0e, Lbc0e, lap0e, n0e)) = eval_full(ema_params, P_in, P_bdry, N_bdry)
    print(f"[INIT] raw={float(L0):.6e}  ema={float(L0e):.6e}")
    print(f"[INIT] lap stats (raw): mean={float(jnp.mean(lap0)):.3e}  rms={float(jnp.sqrt(jnp.mean(lap0**2))):.3e}")
    print(f"[INIT] n·∇u stats (raw): mean={float(jnp.mean(n0)):.3e}  rms={float(jnp.sqrt(jnp.mean(n0**2))):.3e}")
    sys.stdout.flush()

    # Training loop
    log_every = max(1, steps // 20)
    key_train = random.PRNGKey(1234)

    # Running loss balancing
    alpha = 0.99
    run_in, run_bc = 1.0, 1.0

    for it in range(1, steps + 1):
        key_train, subkey = random.split(key_train)
        # dynamic λ_bc schedule
        lam_eff = lam_bc_schedule(it, steps, lam_bc_base)

        # current balancing (based on running means)
        bal_in = 1.0 / (run_in + 1e-12)
        bal_bc = 1.0 / (run_bc + 1e-12)

        model, opt_state, L, (Lin, Lbc, lap_res, nres) = train_step(
            model, opt_state, optimizer, P_in, P_bdry, N_bdry, subkey,
            lam_eff, bal_in, bal_bc
        )

        # EMA update
        ema_params = ema_update(ema_params, model, decay=ema_decay)

        # Update running means (host-side floats)
        run_in = alpha * run_in + (1 - alpha) * float(Lin)
        run_bc = alpha * run_bc + (1 - alpha) * float(Lbc)

        # RBAR interior resampling
        if (it % RBAR_PERIOD) == 0:
            key_train, k_res = random.split(key_train)
            P_in = resample_interior_by_residual(k_res, model, P_in, frac=RBAR_FRAC)

        if (it % log_every) == 0 or it == 1:
            lap_rms = float(jnp.sqrt(jnp.mean(lap_res**2)))
            n_rms   = float(jnp.sqrt(jnp.mean(nres**2)))
            print(f"[{it:5d}] loss={float(L):.6e}  lap={float(Lin):.6e}  bc={float(Lbc):.6e}  "
                  f"|lap|_rms={lap_rms:.3e}  |n·∇u|_rms={n_rms:.3e}  "
                  f"λ_eff={lam_eff:.3f}  bal_in={bal_in:.3e} bal_bc={bal_bc:.3e}")
            sys.stdout.flush()

    # Optional LBFGS final polish (full-batch)
    if USE_LBFGS_FINAL:
        try:
            from jaxopt import LBFGS
            def full_objective(p):
                (L, _aux) = eval_full(p, P_in, P_bdry, N_bdry)
                return L
            solver = LBFGS(fun=full_objective, maxiter=300, tol=1e-6, value_and_grad=True)
            model = solver.run(model).params
            ema_params = ema_update(ema_params, model, decay=0.9)
            print("[LBFGS] Completed.")
        except Exception as e:
            print("[LBFGS] Skipped (jaxopt not available or error):", e)

    # Final diagnostics (raw + ema)
    (Lf, (Linf, Lbcf, lapf, nf)) = eval_full(model, P_in, P_bdry, N_bdry)
    (LfE, (LinfE, LbcfE, lapfE, nfE)) = eval_full(ema_params, P_in, P_bdry, N_bdry)
    print(f"[FINAL] raw: loss={float(Lf):.6e}  lap={float(Linf):.6e}  bc={float(Lbcf):.6e}")
    print(f"[FINAL] ema: loss={float(LfE):.6e}  lap={float(LinfE):.6e}  bc={float(LbcfE):.6e}")
    print(f"[FINAL] raw lap rms={float(jnp.sqrt(jnp.mean(lapf**2))):.3e}, max|lap|={float(jnp.max(jnp.abs(lapf))):.3e}")
    print(f"[FINAL] raw n·∇u rms={float(jnp.sqrt(jnp.mean(nf**2))):.3e}, max|n·∇u|={float(jnp.max(jnp.abs(nf))):.3e}")

    # Compute |∇u| on the boundary and plot on θ×φ grid (EMA params for smoother result)
    print("[PLOT] Computing |∇u| on the boundary grid (EMA params)...")
    gb = grad_u_total_batch(ema_params, P_bdry)   # [Nb,3]
    grad_norm = jnp.linalg.norm(gb, axis=-1)      # [Nb]
    GN = grad_norm.reshape(Xg.shape)              # [nθ,nφ]

    print(f"[PLOT] |∇u| stats on boundary: min={float(jnp.min(GN)):.3e}, "
          f"max={float(jnp.max(GN)):.3e}, mean={float(jnp.mean(GN)):.3e}")
    sys.stdout.flush()

    theta = jnp.linspace(0, 2*jnp.pi, N_bdry_theta, endpoint=False)
    phi   = jnp.linspace(0, 2*jnp.pi, N_bdry_phi,   endpoint=False)
    TH, PH = jnp.meshgrid(theta, phi, indexing='ij')

    plt.figure(figsize=FIGSIZE)
    im = plt.pcolormesh(PH, TH, GN, shading="auto", cmap=PLOT_CMAP)
    plt.colorbar(im, label=r"$|\nabla u|$ on boundary (EMA)")
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\theta$")
    plt.title(r"$|\nabla u|$ on torus boundary")
    plt.tight_layout()
    plt.show()

    return ema_params  # return EMA params by default

# =============================================================================
# ========================== RUNTIME SAFETY CHECKS ============================
# =============================================================================

if __name__ == "__main__":
    # Basic environment checks
    try:
        import jax  # noqa: F401
        _ = jnp.array([0.0]) + 1.0
    except Exception as e:
        print("[ERROR] JAX seems misconfigured:", e)
        sys.exit(1)

    try:
        import equinox as _eqx, optax as _optax, matplotlib as _mpl  # noqa: F401
    except Exception as e:
        print("[ERROR] Missing dependency. Please install: equinox, optax, matplotlib")
        print("       pip install equinox optax matplotlib")
        sys.exit(1)

    _ = main()
