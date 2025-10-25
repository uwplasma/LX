#!/usr/bin/env python3
"""
PINN for Laplace's equation ∇²u = 0 inside a (possibly non-axisymmetric) torus
in Cartesian coordinates, with a soft Neumann boundary condition n·∇u ≈ 0.

u_total(x,y,z) = u_mv(x,y,z) + u_nn(x,y,z)
  - u_mv is a multi-valued piece: kappa * atan2(y,x) / R0
  - u_nn is a single-valued MLP built with Equinox

Objective (least squares):
  L = ⟨(∇² u_total)^2⟩_interior  +  λ_bc ⟨(n·∇u_total)^2⟩_boundary

End of training:
  - Prints diagnostics (residuals, grads)
  - Plots |∇u| on the boundary surface (θ×φ grid)

You may replace the surface constructor with any surface provider that returns:
  - boundary points P ∈ ℝ^{Nb×3}
  - outward unit normals N ∈ ℝ^{Nb×3}
For the example we build a torus with a(φ) = a0 + a1 cos(N_harm φ).
"""

from __future__ import annotations
import sys
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, random
import optax
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =============================================================================
# ============================= USER PARAMETERS ===============================
# =============================================================================

# ----- Batching -----
BATCH_IN   = 2048    # interior points per step
BATCH_BDRY = 2048    # boundary points per step

# ----- Geometry (torus example) -----
R0      = 1.0           # Major radius (centerline in xy-plane)
a0      = 0.35          # Minor-radius base
a1      = 0.10          # Minor-radius modulation amplitude
N_harm  = 3             # Mode in a(φ) = a0 + a1 cos(N_harm φ)

# ----- Multi-valued potential component -----
kappa   = 1.0           # u_mv = kappa * atan2(y,x) / R0; set 0 for single-valued

# ----- Sampling -----
N_in            = 10_000   # interior sample count
N_bdry_theta    = 32       # boundary θ resolution
N_bdry_phi      = 64      # boundary φ resolution
rng_seed        = 0

# ----- Neural Network (Equinox MLP) -----
MLP_HIDDEN_SIZES = (64, 64)  # user controls width/depth here
MLP_ACT          = jax.nn.tanh
# output size is 1 (scalar potential), input size is 3 (x,y,z)

# ----- Optimization (Optax; least-squares objective) -----
steps   = 1000
lr      = 3e-3
lam_bc  = 5.0

# ----- Plotting -----
PLOT_CMAP = "viridis"
FIGSIZE   = (8, 4.5)

# =============================================================================
# =============================== GEOMETRY ====================================
# =============================================================================

def _identity(x):
    return x

def a_of_phi(phi: jnp.ndarray) -> jnp.ndarray:
    """Minor radius a(φ) = a0 + a1 cos(N_harm φ)."""
    return a0 + a1 * jnp.cos(N_harm * phi)

def cylindrical_phi(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(y, x)

def inside_torus_mask(x, y, z) -> jnp.ndarray:
    """Inside test for variable-radius torus."""
    r   = jnp.sqrt(x*x + y*y)
    phi = cylindrical_phi(x, y)
    rho = jnp.sqrt((r - R0)**2 + z*z)        # distance to circular axis at angle φ
    return rho <= a_of_phi(phi)

def build_surface_torus(n_theta: int, n_phi: int):
    """
    Parameterization:
      r(θ,φ) = [(R0 + a(φ) cosθ) cosφ, (R0 + a(φ) cosθ) sinφ, a(φ) sinθ]
    Returns X,Y,Z with shape [nθ, nφ].
    """
    theta = jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=True)
    phi   = jnp.linspace(0, 2*jnp.pi, n_phi,   endpoint=True)
    Θ, Φ  = jnp.meshgrid(theta, phi, indexing='ij')    # [nθ,nφ]
    aφ    = a_of_phi(Φ)
    Rring = R0 + aφ * jnp.cos(Θ)
    X = Rring * jnp.cos(Φ)
    Y = Rring * jnp.sin(Φ)
    Z = aφ * jnp.sin(Θ)
    return X, Y, Z

def normals_from_param_grid(X, Y, Z):
    """
    Estimate outward normals n̂ on a periodic (θ,φ) grid via:
      n ∝ ∂r/∂θ × ∂r/∂φ, then normalize.
    """
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
    """
    Single-shot JAX sampler: oversample once, keep first n_points valid points.
    Falls back to a second pass with larger oversampling if needed.
    """
    def _one_shot(key, n_points, factor):
        a_max = a0 + jnp.abs(a1)
        Lxy   = R0 + a_max
        M     = factor * n_points
        kx, ky, kz = random.split(key, 3)
        X = random.uniform(kx, (M,), minval=-Lxy,  maxval=Lxy)
        Y = random.uniform(ky, (M,), minval=-Lxy,  maxval=Lxy)
        Z = random.uniform(kz, (M,), minval=-a_max, maxval=a_max)
        mask = inside_torus_mask(X, Y, Z)                 # [M] boolean
        # Fixed-size index vector (pads with zeros):
        idx  = jnp.nonzero(mask, size=n_points, fill_value=0)[0]  # [n_points]
        # How many valid did we actually get?
        got  = jnp.sum(mask)
        pts  = jnp.stack([X[idx], Y[idx], Z[idx]], axis=-1)       # [n_points,3]
        return pts, got

    pts, got = _one_shot(key, n_points, oversample_factor)
    # If not enough points, try once more with a bigger factor (Python guard, no JIT).
    if int(got) < n_points:
        pts, _ = _one_shot(key, n_points, oversample_factor * 2)
    return pts

# ================== Boundary evaluation + plotting helpers (ADD) ============

def eval_on_boundary(params: PotentialMLP, P_bdry, N_bdry, Xg, Yg, Zg):
    """
    Returns:
      Gvec: (nθ, nφ, 3)  gradient vectors at boundary grid
      Gmag: (nθ, nφ)     |∇u| at boundary grid
    """
    Gvec_flat = grad_u_total_batch(params, P_bdry)     # (Nb,3)
    Gmag_flat = jnp.linalg.norm(Gvec_flat, axis=-1)    # (Nb,)

    Gvec = Gvec_flat.reshape(Xg.shape + (3,))          # (nθ,nφ,3)
    Gmag = Gmag_flat.reshape(Xg.shape)                 # (nθ,nφ)
    Nhat = N_bdry.reshape(Xg.shape + (3,))             # (nθ,nφ,3)
    return Gvec, Gmag, Nhat


def _downsample_grid_for_quiver(X, Y, Z, V, step_theta=6, step_phi=8):
    """
    Downsample θ,φ grid for quiver clarity.
      X,Y,Z: (nθ,nφ)
      V:     (nθ,nφ,3) vectors to plot
    Returns flattened (x,y,z,u,v,w) for ax.quiver.
    """
    Xs = X[::step_theta, ::step_phi]
    Ys = Y[::step_theta, ::step_phi]
    Zs = Z[::step_theta, ::step_phi]
    Vs = V[::step_theta, ::step_phi, :]

    x = Xs.ravel(); y = Ys.ravel(); z = Zs.ravel()
    u = Vs[...,0].ravel(); v = Vs[...,1].ravel(); w = Vs[...,2].ravel()
    return x, y, z, u, v, w

def fix_matplotlib_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_surface_with_vectors_ax(ax, X, Y, Z, Gmag, Nhat, Gvec=None,
                                 title="∇u on torus surface",
                                 cmap="viridis", quiver_len=0.15,
                                 step_theta=6, step_phi=8,
                                 vmin=None, vmax=None, plot_normals=True,
                                 grad_color="red", normals_color="black",
                                 surf_offset=0.0):
    """
    Plot a colored torus surface and vector overlays.
    NEW: 'surf_offset' (float, same units as X/Y/Z) shifts ONLY the surface
         inward along the outward normal (so quivers appear outside the surface).
         Quivers still originate at the true boundary (X,Y,Z).

    Returns a ScalarMappable for making a shared colorbar.
    """
    from matplotlib import colormaps as _cmaps
    import numpy as _np
    _cmap = _cmaps.get_cmap(cmap)

    # Shared color normalization
    gmin = _np.min(Gmag) if vmin is None else vmin
    gmax = _np.max(Gmag) if vmax is None else vmax
    normed = (Gmag - gmin) / (gmax - gmin + 1e-12)
    facecolors = _cmap(normed)

    # ---- Offset surface inward along outward normals (unit-length) ----
    if surf_offset != 0.0:
        Xs = X - surf_offset * Nhat[..., 0]
        Ys = Y - surf_offset * Nhat[..., 1]
        Zs = Z - surf_offset * Nhat[..., 2]
    else:
        Xs, Ys, Zs = X, Y, Z

    # Draw the (slightly smaller) surface
    ax.plot_surface(Xs, Ys, Zs, facecolors=facecolors, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)

    # Normals (optional), anchored at the true boundary (not the offset)
    if plot_normals:
        xn, yn, zn, un, vn, wn = _downsample_grid_for_quiver(X, Y, Z, Nhat, step_theta, step_phi)
        ax.quiver(xn, yn, zn, un, vn, wn,
                  length=quiver_len, normalize=True, linewidth=1.0,
                  color=normals_color, alpha=0.7)

    # ∇u vectors (optional), also anchored at the true boundary
    if Gvec is not None:
        xg, yg, zg, ug, vg, wg = _downsample_grid_for_quiver(X, Y, Z, Gvec, step_theta, step_phi)
        ax.quiver(xg, yg, zg, ug, vg, wg,
                  length=quiver_len, normalize=True, linewidth=1.2,
                  color=grad_color, alpha=0.95)

    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1,1,1))

    m = plt.cm.ScalarMappable(cmap=_cmap)
    m.set_array(Gmag)
    m.set_clim(gmin, gmax)
    return m


# =============================================================================
# ============================ MODEL (Equinox) ================================
# =============================================================================

class PotentialMLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key, hidden_sizes=MLP_HIDDEN_SIZES, act=MLP_ACT):
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
        out = self.mlp(xyz)
        return out.squeeze(-1)

def u_multivalued(xyz: jnp.ndarray) -> jnp.ndarray:
    x, y = xyz[..., 0], xyz[..., 1]
    return kappa * jnp.arctan2(y, x) / R0

def grad_u_mv(xyz: jnp.ndarray) -> jnp.ndarray:
    """∇(kappa * atan2(y,x)/R0) = kappa/R0 * (-y/(x^2+y^2), x/(x^2+y^2), 0)"""
    x, y = xyz[..., 0], xyz[..., 1]
    r2 = x*x + y*y
    inv = jnp.where(r2 > 1e-18, 1.0 / r2, 0.0)             # guard axis
    gx = -y * inv
    gy =  x * inv
    gz = jnp.zeros_like(x)
    return (kappa / R0) * jnp.stack([gx, gy, gz], axis=-1)

@eqx.filter_jit
def lap_u_mv_zero(_params, _xyz):
    # Laplacian of atan2(y,x) is 0 away from r=0 (singular on axis only).
    return jnp.array(0.0)

def u_total(params: PotentialMLP, xyz: jnp.ndarray) -> jnp.ndarray:
    return u_multivalued(xyz) + params(xyz)

# Gradient of the NN-only part
@eqx.filter_jit
def grad_u_nn_scalar(params: PotentialMLP, xyz: jnp.ndarray) -> jnp.ndarray:
    return grad(lambda q: params(q))(xyz)

# Laplacian via three Hessian–vector products (no full Hessian materialization)
# e_i are standard basis vectors in R^3
E1 = jnp.array([1.0, 0.0, 0.0])
E2 = jnp.array([0.0, 1.0, 0.0])
E3 = jnp.array([0.0, 0.0, 1.0])

@eqx.filter_jit
def lap_u_nn_scalar(params: PotentialMLP, xyz: jnp.ndarray) -> jnp.ndarray:
    # g(x) = ∇u_nn(x); jvp(g, (x,), (e_i,)) = H(x) e_i
    def g(q): return grad_u_nn_scalar(params, q)
    _, He1 = jax.jvp(g, (xyz,), (E1,))
    _, He2 = jax.jvp(g, (xyz,), (E2,))
    _, He3 = jax.jvp(g, (xyz,), (E3,))
    # e_i^T H e_i is the ith component of H e_i (since e_i picks that component)
    return He1[0] + He2[1] + He3[2]

# Batched wrappers
u_total_batch = jax.vmap(u_total, in_axes=(None, 0))
u_nn_batch = jax.vmap(lambda p, q: p(q), in_axes=(None, 0))
grad_u_nn_batch = jax.vmap(grad_u_nn_scalar, in_axes=(None, 0))
lap_u_nn_batch  = jax.vmap(lap_u_nn_scalar,  in_axes=(None, 0))

@eqx.filter_jit
def grad_u_total_batch(params: PotentialMLP, xyz_batch: jnp.ndarray) -> jnp.ndarray:
    return grad_u_mv(xyz_batch) + grad_u_nn_batch(params, xyz_batch)

@eqx.filter_jit
def lap_u_total_batch(params: PotentialMLP, xyz_batch: jnp.ndarray) -> jnp.ndarray:
    # Lap(u_mv)=0 away from r=0  ⇒ just Lap(u_nn)
    return lap_u_nn_batch(params, xyz_batch)

# =============================================================================
# ============================ LOSS & OPTIMIZER ===============================
# =============================================================================

def loss_fn_batch(params: PotentialMLP,
            pts_interior: jnp.ndarray, pts_bdry: jnp.ndarray,
            normals_bdry: jnp.ndarray,
            key: jax.Array):
    """Sample a random interior and boundary mini-batch and compute MSE losses."""
    ni = pts_interior.shape[0]
    nb = pts_bdry.shape[0]

    key_i, key_b = random.split(key)
    idx_i = random.choice(key_i, ni, (min(BATCH_IN, ni),), replace=False)
    idx_b = random.choice(key_b, nb, (min(BATCH_BDRY, nb),), replace=False)

    Pi = pts_interior[idx_i]
    Pb = pts_bdry[idx_b]
    Nb_hat = normals_bdry[idx_b]

    lap = lap_u_total_batch(params, Pi)                # [Bi]
    g_b = grad_u_total_batch(params, Pb)               # [Bb,3]
    n_dot = jnp.sum(Nb_hat * g_b, axis=-1)             # [Bb]

    loss_in = jnp.mean(lap * lap)
    loss_bc = jnp.mean(n_dot * n_dot)
    total   = loss_in + lam_bc * loss_bc
    return total, (loss_in, loss_bc, lap, n_dot)

# Filtered versions keep non-JAX parts (if any) out of JIT
loss_value_and_grad = eqx.filter_value_and_grad(loss_fn_batch, has_aux=True)

@eqx.filter_jit
def train_step(params, opt_state, optimizer,
               pts_interior, pts_bdry, normals_bdry, key):
    (loss_val, aux), grads = loss_value_and_grad(
        params, pts_interior, pts_bdry, normals_bdry, key
    )
    params_f, params_s = eqx.partition(params, eqx.is_inexact_array)
    grads_f,  _        = eqx.partition(grads,  eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads_f, opt_state, params_f)
    params_f = optax.apply_updates(params_f, updates)
    params = eqx.combine(params_f, params_s)
    return params, opt_state, loss_val, aux

@eqx.filter_jit
def eval_full(params: PotentialMLP,
              pts_interior: jnp.ndarray, pts_bdry: jnp.ndarray,
              normals_bdry: jnp.ndarray):
    """Evaluate losses and residuals on the full sets (deterministic)."""
    lap = lap_u_total_batch(params, pts_interior)      # [Ni]
    g_b = grad_u_total_batch(params, pts_bdry)         # [Nb,3]
    n_dot = jnp.sum(normals_bdry * g_b, axis=-1)       # [Nb]

    loss_in = jnp.mean(lap * lap)
    loss_bc = jnp.mean(n_dot * n_dot)
    total   = loss_in + lam_bc * loss_bc
    return total, (loss_in, loss_bc, lap, n_dot)

# =============================================================================
# ================================== MAIN =====================================
# =============================================================================

def main():
    print("=== PINN Laplace on Torus ===")
    print(f"Major radius R0={R0:.3f}")
    print(f"Minor radius a(φ)=a0 + a1 cos(Nφ) with a0={a0:.3f}, a1={a1:.3f}, N={N_harm}")
    print(f"Multi-valued kappa={kappa:.3f}  (set 0 for purely single-valued potential)")
    print(f"Interior samples: {N_in}, Boundary grid: θ={N_bdry_theta}, φ={N_bdry_phi}")
    print(f"Network: hidden={MLP_HIDDEN_SIZES}, act={MLP_ACT.__name__}, optimizer=Adam(lr={lr})")
    print(f"Training steps: {steps}, λ_bc={lam_bc}")
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
    # If average is negative, flip all normals.
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

    # Model + optimizer
    model = PotentialMLP(k_model, hidden_sizes=MLP_HIDDEN_SIZES, act=MLP_ACT)
    # optimizer = optax.adam(lr)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(optax.cosine_decay_schedule(init_value=lr, decay_steps=steps), weight_decay=0.0)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Initial loss check
    (L0, (Lin0, Lbc0, lap0, n0)) = eval_full(model, P_in, P_bdry, N_bdry)
    print(f"[INIT] loss={float(L0):.6e}  lap={float(Lin0):.6e}  bc={float(Lbc0):.6e}")
    print(f"[INIT] lap stats: mean={float(jnp.mean(lap0)):.3e}  rms={float(jnp.sqrt(jnp.mean(lap0**2))):.3e}")
    print(f"[INIT] n·∇u stats: mean={float(jnp.mean(n0)):.3e}  rms={float(jnp.sqrt(jnp.mean(n0**2))):.3e}")
    sys.stdout.flush()
    
    # ===== INITIAL boundary grad (store, don't plot now) =====
    Gvec_init, Gmag_init, Nhat_grid = eval_on_boundary(model, P_bdry, N_bdry, Xg, Yg, Zg)

    # Training loop
    log_every = max(1, steps // 20)
    key_train = random.PRNGKey(1234)
    for it in range(1, steps + 1):
        key_train, subkey = random.split(key_train)
        model, opt_state, L, (Lin, Lbc, lap_res, nres) = train_step(
            model, opt_state, optimizer, P_in, P_bdry, N_bdry, subkey
        )
        if (it % log_every) == 0 or it == 1:
            # these lap_res/nres are from the current mini-batches
            lap_rms = float(jnp.sqrt(jnp.mean(lap_res**2)))
            n_rms   = float(jnp.sqrt(jnp.mean(nres**2)))
            print(f"[{it:5d}] loss={float(L):.6e}  lap={float(Lin):.6e}  bc={float(Lbc):.6e}  "
                f"|lap|_rms={lap_rms:.3e}  |n·∇u|_rms={n_rms:.3e}")
            sys.stdout.flush()

    # Final diagnostics
    (Lf, (Linf, Lbcf, lapf, nf)) = eval_full(model, P_in, P_bdry, N_bdry)
    print(f"[FINAL] loss={float(Lf):.6e}  lap={float(Linf):.6e}  bc={float(Lbcf):.6e}")
    print(f"[FINAL] lap stats: mean={float(jnp.mean(lapf)):.3e}  rms={float(jnp.sqrt(jnp.mean(lapf**2))):.3e},  "
          f"max|lap|={float(jnp.max(jnp.abs(lapf))):.3e}")
    print(f"[FINAL] n·∇u stats: mean={float(jnp.mean(nf)):.3e}  rms={float(jnp.sqrt(jnp.mean(nf**2))):.3e},  "
          f"max|n·∇u|={float(jnp.max(jnp.abs(nf))):.3e}")

    # Compute |∇u| on the boundary and plot on θ×φ grid
    print("[PLOT] Computing |∇u| on the boundary grid...")
    gb = grad_u_total_batch(model, P_bdry)             # [Nb,3]
    grad_norm = jnp.linalg.norm(gb, axis=-1)           # [Nb]
    GN = grad_norm.reshape(Xg.shape)                   # [nθ,nφ]

    print(f"[PLOT] |∇u| stats on boundary: min={float(jnp.min(GN)):.3e}, "
          f"max={float(jnp.max(GN)):.3e}, mean={float(jnp.mean(GN)):.3e}")
    sys.stdout.flush()

    # ===== FINAL boundary grad (compute now) =====
    Gvec_final, Gmag_final, _ = eval_on_boundary(model, P_bdry, N_bdry, Xg, Yg, Zg)

    # ===== SIDE-BY-SIDE 3D comparison (initial vs final) =====
    # Shared color normalization
    vmin_shared = float(jnp.minimum(Gmag_init.min(), Gmag_final.min()))
    vmax_shared = float(jnp.maximum(Gmag_init.max(), Gmag_final.max()))

    fig3d = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig3d.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])  # reserve rightmost sliver for colorbar

    ax1 = fig3d.add_subplot(gs[0, 0], projection='3d')
    ax2 = fig3d.add_subplot(gs[0, 1], projection='3d')
    cax = fig3d.add_subplot(gs[0, 2])  # colorbar axis (skinny)

    offset = 0.03 * float(a0 + abs(a1))

    m1 = plot_surface_with_vectors_ax(ax1, Xg, Yg, Zg, Gmag_init, Nhat_grid, Gvec=Gvec_init,
                                    title="Initial |∇u| and vectors (boundary)",
                                    cmap=PLOT_CMAP, quiver_len=0.15,
                                    step_theta=6, step_phi=8, plot_normals=False,
                                    vmin=vmin_shared, vmax=vmax_shared,
                                    surf_offset=offset)

    m2 = plot_surface_with_vectors_ax(ax2, Xg, Yg, Zg, Gmag_final, Nhat_grid, Gvec=Gvec_final,
                                    title="Final |∇u| and vectors (boundary)",
                                    cmap=PLOT_CMAP, quiver_len=0.15,
                                    step_theta=6, step_phi=8, plot_normals=False,
                                    vmin=vmin_shared, vmax=vmax_shared,
                                    surf_offset=offset)

    cb = fig3d.colorbar(m2, cax=cax)   # one shared colorbar outside, using the right column
    cb.set_label(r"$|\nabla u|$")
    
    fix_matplotlib_3d(ax1)
    fix_matplotlib_3d(ax2)


    # ===== φ×θ heatmaps side-by-side (initial vs final) =====
    # Use the already-computed boundary grid
    theta = jnp.linspace(0, 2*jnp.pi, N_bdry_theta, endpoint=True)
    phi   = jnp.linspace(0, 2*jnp.pi, N_bdry_phi,   endpoint=True)
    TH, PH = jnp.meshgrid(theta, phi, indexing='ij')

    GN_init  = Gmag_init            # shape (nθ, nφ)
    GN_final = Gmag_final           # shape (nθ, nφ)

    vmin_hm = float(jnp.minimum(GN_init.min(), GN_final.min()))
    vmax_hm = float(jnp.maximum(GN_init.max(), GN_final.max()))

    figHM = plt.figure(figsize=(12, 4.5), constrained_layout=True)
    gs = figHM.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])

    axL = figHM.add_subplot(gs[0, 0])
    axR = figHM.add_subplot(gs[0, 1])
    cax = figHM.add_subplot(gs[0, 2])  # colorbar axis

    imL = axL.pcolormesh(PH, TH, GN_init, shading="auto", cmap=PLOT_CMAP,
                        vmin=vmin_hm, vmax=vmax_hm)
    axL.set_title("Initial  $|\\nabla u|$ on boundary")
    axL.set_xlabel(r"$\phi$")
    axL.set_ylabel(r"$\theta$")

    imR = axR.pcolormesh(PH, TH, GN_final, shading="auto", cmap=PLOT_CMAP,
                        vmin=vmin_hm, vmax=vmax_hm)
    axR.set_title("Final  $|\\nabla u|$ on boundary")
    axR.set_xlabel(r"$\phi$")

    cb = figHM.colorbar(imR, cax=cax)   # single shared bar outside
    cb.set_label(r"$|\nabla u|$")
    
    plt.show()


    return model

if __name__ == "__main__":
    _ = main()
