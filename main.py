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
from pathlib import Path
from typing import Any, Dict
import argparse

from _initialization import build_params_from_path
from _plotting import fix_matplotlib_3d, plot_surface_with_vectors_ax
from _geometry import surface_points_and_normals, sample_interior
from _network_and_loss import PotentialMLP, train_step, eval_full, load_model_if_exists, save_model
from _physics import eval_on_boundary, grad_u_total_batch
from _state import runtime

# =============================================================================
# ============================= CONFIG LOADING ================================
# =============================================================================

# Defaults are loaded at import-time so module-level functions see valid values.
_DEFAULT_PARAMS = build_params_from_path("input.toml")

# Globals populated from params; will be updated if a different config is passed.
CHECKPOINT_PATH: Path
BATCH_IN: int
BATCH_BDRY: int
R0: float
a0: float
a1: float
N_harm: int
kappa: float
N_in: int
N_bdry_theta: int
N_bdry_phi: int
rng_seed: int
MLP_HIDDEN_SIZES: tuple
MLP_ACT: Any
steps: int
lr: float
lam_bc: float
PLOT_CMAP: str
FIGSIZE: tuple


def _apply_params(params: Dict[str, Any]) -> None:
    global CHECKPOINT_PATH, BATCH_IN, BATCH_BDRY, R0, a0, a1, N_harm, kappa
    global N_in, N_bdry_theta, N_bdry_phi, rng_seed
    global MLP_HIDDEN_SIZES, MLP_ACT
    global steps, lr, lam_bc
    global PLOT_CMAP, FIGSIZE

    CHECKPOINT_PATH = Path(params["checkpoint_path"])  # type: ignore
    BATCH_IN   = int(params["batch_interior"])  # type: ignore
    BATCH_BDRY = int(params["batch_boundary"])  # type: ignore

    R0     = float(params["R0"])  # type: ignore
    a0     = float(params["a0"])  # type: ignore
    a1     = float(params["a1"])  # type: ignore
    N_harm = int(params["N_harm"])  # type: ignore

    kappa  = float(params["kappa"])  # type: ignore

    N_in          = int(params["N_in"])  # type: ignore
    N_bdry_theta  = int(params["N_bdry_theta"])  # type: ignore
    N_bdry_phi    = int(params["N_bdry_phi"])  # type: ignore
    rng_seed      = int(params["rng_seed"])  # type: ignore

    MLP_HIDDEN_SIZES = tuple(params["mlp_hidden_sizes"])  # type: ignore
    MLP_ACT          = params["mlp_activation"]  # type: ignore

    steps  = int(params["steps"])  # type: ignore
    lr     = float(params["lr"])  # type: ignore
    lam_bc = float(params["lam_bc"])  # type: ignore

    PLOT_CMAP = str(params["plot_cmap"])  # type: ignore
    FIGSIZE   = tuple(params["figsize"])  # type: ignore

# Apply defaults at import time
_apply_params(_DEFAULT_PARAMS)


def main(config_path: str = "input.toml"):
    if config_path and Path(config_path).exists():
        params = build_params_from_path(config_path)
        _apply_params(params)
        
    # >>> Set runtime state ONCE so all modules see it <<<
    runtime.R0 = R0
    runtime.a0 = a0
    runtime.a1 = a1
    runtime.N_harm = N_harm
    runtime.kappa = kappa
    runtime.BATCH_IN = BATCH_IN
    runtime.BATCH_BDRY = BATCH_BDRY
    runtime.lam_bc = lam_bc
        
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
    # Robust outwardness check: use centroid → point vector
    P_grid = jnp.stack([Xg, Yg, Zg], axis=-1)                 # (nθ, nφ, 3)
    centroid = jnp.mean(P_grid.reshape(-1, 3), axis=0, keepdims=True)  # (1,3)
    Rvec = P_grid - centroid                                   # (nθ, nφ, 3)
    Rhat = Rvec / (jnp.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-12)

    Nhat_grid = N_bdry.reshape(Xg.shape + (3,))                # (nθ, nφ, 3)
    mean_out = jnp.mean(jnp.sum(Nhat_grid * Rhat, axis=-1))
    mean_out_val = float(mean_out)
    print(f"[DEBUG] Mean outwardness of normals (centroid test): {mean_out_val:+.4f} (≈positive expected)")
    if mean_out_val < 0:
        N_bdry = -N_bdry
        Nhat_grid = -Nhat_grid
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
    # Try to resume from an existing checkpoint (architecture must match)
    model = load_model_if_exists(model, CHECKPOINT_PATH)

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
    
    save_model(model, CHECKPOINT_PATH)

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

    offset = 0.1 * float(a0 + abs(a1))

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
    parser = argparse.ArgumentParser(description="PINN for Laplace on a torus")
    parser.add_argument("--config", "-c", type=str, default="input.toml",
                        help="Path to TOML configuration file")
    args = parser.parse_args()
    _ = main(config_path=args.config)
