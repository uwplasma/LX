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

One may replace the surface constructor with any surface provider that returns:
  - boundary points P ∈ ℝ^{Nb×3}
  - outward unit normals N ∈ ℝ^{Nb×3}
For the example we build a torus with a(φ) = a0 + a1 cos(N_harm φ).

Change 10.27.2025: single-surface training OR multi-surface pretraining for
Laplace's equation ∇²u = 0 with Neumann BC (n·∇u = 0 on the surface).

Modes (from input.toml):
  [surfaces]
  mode = "single"  -> one surface (as before) using [geometry] params
  mode = "torus"   -> multi-surface pretraining over [[surfaces.torus_list]]

In all cases, interior points are sampled deterministically from a fixed box,
then masked to the current surface (never train on outside points).
"""


# C. Optional architecture toggles (one-line config changes)
# Activation: activation = "sine" (a SIREN-style MLP) often does better for PDEs with oscillatory/tangential structure. You already allow "sin" in _initialization.py mapping; use it for pretraining.
# Fourier features: prepend concat([xyz, sin(2πBx), cos(2πBx)]) with a small Gaussian B. This is ~15 lines in PotentialMLP and can drastically accelerate convergence on geometric families.
# LBFGS polish: After AdamW, run 200–500 LBFGS steps (scipy LBFGS on filtered params). PINN papers routinely report big gains.

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import optax
import equinox as eqx
import matplotlib.pyplot as plt

# ---------------- local imports ----------------
from _initialization import build_params_from_path
from _plotting import fix_matplotlib_3d, plot_surface_with_vectors_ax, draw_box_edges
from _geometry import (
    surface_points_and_normals,
    fixed_box_points,
    select_interior_from_fixed,
)
from _network_and_loss import (
    PotentialMLP,
    train_step,
    eval_full,
    load_model_if_exists,
    save_model,
)
from _physics import eval_on_boundary, grad_u_total_batch
from _state import runtime

# Multi-surface dataset helpers
from _multisurface import build_torus_family

# =============================================================================
# ========================== GLOBAL / PARAM BINDING ===========================
# =============================================================================
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
FIGSIZE: Tuple[float, float]

# box
box_zmin: float
box_zmax: float
box_points_total: int
box_seed: int

# surfaces config (raw dict from TOML; parsed here)
surfaces_cfg: dict

def _lambda_bc_schedule(it, T):
    # Start high, decay to ~your TOML lam_bc
    warm = 1000.0   # very high early
    end  = lam_bc   # target from TOML
    alpha = (it / max(1, T)) ** 0.5
    return end + (1.0 - alpha) * (warm - end)

def _apply_params(params: Dict[str, Any]) -> None:
    global CHECKPOINT_PATH, BATCH_IN, BATCH_BDRY, R0, a0, a1, N_harm, kappa
    global N_in, N_bdry_theta, N_bdry_phi, rng_seed
    global MLP_HIDDEN_SIZES, MLP_ACT
    global steps, lr, lam_bc
    global PLOT_CMAP, FIGSIZE
    global box_zmin, box_zmax, box_points_total, box_seed
    global surfaces_cfg

    CHECKPOINT_PATH = Path(params["checkpoint_path"])          # type: ignore
    BATCH_IN        = int(params["batch_interior"])            # type: ignore
    BATCH_BDRY      = int(params["batch_boundary"])            # type: ignore

    R0     = float(params["R0"])                               # type: ignore
    a0     = float(params["a0"])                               # type: ignore
    a1     = float(params["a1"])                               # type: ignore
    N_harm = int(params["N_harm"])                             # type: ignore

    kappa  = float(params["kappa"])                            # type: ignore

    N_in          = int(params["N_in"])                        # type: ignore
    N_bdry_theta  = int(params["N_bdry_theta"])                # type: ignore
    N_bdry_phi    = int(params["N_bdry_phi"])                  # type: ignore
    rng_seed      = int(params["rng_seed"])                    # type: ignore

    MLP_HIDDEN_SIZES = tuple(params["mlp_hidden_sizes"])       # type: ignore
    MLP_ACT          = params["mlp_activation"]                # type: ignore

    steps  = int(params["steps"])                              # type: ignore
    lr     = float(params["lr"])                               # type: ignore
    lam_bc = float(params["lam_bc"])                           # type: ignore

    PLOT_CMAP = str(params["plot_cmap"])                       # type: ignore
    FIGSIZE   = tuple(params["figsize"])                       # type: ignore

    # Box controls from TOML
    box_zmin        = float(params.get("box_zmin", -0.5))      # type: ignore
    box_zmax        = float(params.get("box_zmax",  0.5))      # type: ignore
    box_points_total= int(params.get("box_points_total", 200_000))  # type: ignore
    box_seed        = int(params.get("box_seed", 42))          # type: ignore

    # Raw surfaces config blob (parsed in main)
    surfaces_cfg    = params.get("surfaces_cfg", {})           # type: ignore


# =============================================================================
# ================================== MAIN =====================================
# =============================================================================
def main(config_path: str = "input.toml"):
    # Load config dict and bind globals
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

    print("=== PINN Laplace (single or multi-surface) ===")
    print(f"Network: hidden={MLP_HIDDEN_SIZES}, act={MLP_ACT.__name__}, optimizer=AdamW(lr={lr})")
    print(f"Training steps: {steps}, λ_bc={lam_bc}")
    sys.stdout.flush()

    # RNG (for model init and misc)
    key = random.PRNGKey(rng_seed)
    key, k_model = random.split(key)

    # ---------------------- Build fixed box + candidate points -----------------
    # x,y box computed from geometry envelope; z from TOML
    Lxy = float(R0 + (abs(a1) + a0))
    xmin, xmax = -Lxy, Lxy
    ymin, ymax = -Lxy, Lxy
    zmin, zmax = float(box_zmin), float(box_zmax)
    runtime.box_bounds = (xmin, xmax, ymin, ymax, zmin, zmax)

    print(f"[BOX] Using fixed sampling box:")
    print(f"      x∈[{xmin:+.3f},{xmax:+.3f}]  y∈[{ymin:+.3f},{ymax:+.3f}]  z∈[{zmin:+.3f},{zmax:+.3f}]")
    print(f"[BOX] Generating {box_points_total} deterministic points (seed={box_seed})")
    P_box = fixed_box_points(box_seed, box_points_total, runtime.box_bounds)   # [M,3]

    # ---------------------- Build model + optimizer ----------------------------
    model = PotentialMLP(k_model, hidden_sizes=MLP_HIDDEN_SIZES, act=MLP_ACT)
    model = load_model_if_exists(model, CHECKPOINT_PATH)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(optax.cosine_decay_schedule(init_value=lr, decay_steps=steps), weight_decay=0.0),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # ---------------------- Mode selection ------------------------------------
    mode = str(surfaces_cfg.get("mode", "single")).lower().strip()
    print(f"[MODE] {mode}")

    if mode == "single":
        # ===== SINGLE SURFACE (as before) =====
        # Build boundary grid for the current runtime geometry:
        P_bdry, N_bdry, Xg, Yg, Zg = surface_points_and_normals(N_bdry_theta, N_bdry_phi)

        # Robust outwardness check (centroid test)
        P_grid = jnp.stack([Xg, Yg, Zg], axis=-1)                              # (nθ,nφ,3)
        centroid = jnp.mean(P_grid.reshape(-1, 3), axis=0, keepdims=True)      # (1,3)
        Rvec = P_grid - centroid
        Rhat = Rvec / (jnp.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-12)
        Nhat_grid = N_bdry.reshape(Xg.shape + (3,))
        mean_out = float(jnp.mean(jnp.sum(Nhat_grid * Rhat, axis=-1)))
        print(f"[DEBUG] Mean outwardness of normals: {mean_out:+.4f} (positive expected)")
        if mean_out < 0:
            N_bdry = -N_bdry
            Nhat_grid = -Nhat_grid
            print("[DEBUG] Normals flipped to ensure outward orientation.")

        # Deterministic interior points from fixed box, masked to this surface
        P_in = select_interior_from_fixed(P_box, N_in)
        Ni = int(P_in.shape[0])
        if Ni < N_in:
            print(f"[WARN] Only {Ni} interior points found in box. Increase [box].points_total.")
        else:
            P_in = P_in[:N_in]; Ni = N_in

        # Initial diagnostics
        (L0, (Lin0, Lbc0, lap0, n0)) = eval_full(model, P_in, P_bdry, N_bdry)
        print(f"[INIT] loss={float(L0):.6e}  lap={float(Lin0):.6e}  bc={float(Lbc0):.6e}")

        # Pre-compute initial boundary grad (for side-by-side plot)
        Gvec_init, Gmag_init, Nhat_grid = eval_on_boundary(model, P_bdry, N_bdry, Xg, Yg, Zg)

        # Train
        key_train = random.PRNGKey(1234)
        log_every = max(1, steps // 20)
        for it in range(1, steps + 1):
            cur_lam = _lambda_bc_schedule(it, steps)
            runtime.lam_bc = float(cur_lam)   # update runtime so loss sees it
            key_train, subkey = random.split(key_train)
            model, opt_state, L, (Lin, Lbc, lap_res, nres, mean_u), gnorm = train_step(
                model, opt_state, optimizer, P_in, P_bdry, N_bdry, subkey
            )
            if (it % log_every) == 0 or it == 1:
                lap_rms = float(jnp.sqrt(jnp.mean(lap_res**2)))
                n_rms   = float(jnp.sqrt(jnp.mean(nres**2)))
                lap_max = float(jnp.max(jnp.abs(lap_res)))
                n_max   = float(jnp.max(jnp.abs(nres)))
                print(f"[{it:5d}] loss={float(L):.6e}  "
                    f"lin={float(Lin):.3e}  lbc={float(Lbc):.3e}  "
                    f"|lap|_rms={lap_rms:.3e} (max {lap_max:.2e})  "
                    f"|n·∇u|_rms={n_rms:.3e} (max {n_max:.2e})  "
                    f"|u|_mean={float(mean_u):.3e}  "
                    f"||g||={float(gnorm):.3e}  λ_bc={runtime.lam_bc:.2f}  surf={dataset[idx].name}")

        save_model(model, CHECKPOINT_PATH)

        # Final diagnostics
        (Lf, (Linf, Lbcf, lapf, nf)) = eval_full(model, P_in, P_bdry, N_bdry)
        print(f"[FINAL] loss={float(Lf):.6e}  lap={float(Linf):.6e}  bc={float(Lbcf):.6e}")
        print(f"[FINAL] |lap|_rms={float(jnp.sqrt(jnp.mean(lapf**2))):.3e}  "
              f"|n·∇u|_rms={float(jnp.sqrt(jnp.mean(nf**2))):.3e}")

        # Compute final boundary grad & plot side-by-side
        Gvec_final, Gmag_final, _ = eval_on_boundary(model, P_bdry, N_bdry, Xg, Yg, Zg)

        vmin_shared = float(jnp.minimum(Gmag_init.min(), Gmag_final.min()))
        vmax_shared = float(jnp.maximum(Gmag_init.max(), Gmag_final.max()))
        fig3d = plt.figure(figsize=(12, 6), constrained_layout=True)
        gs = fig3d.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
        ax1 = fig3d.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig3d.add_subplot(gs[0, 1], projection='3d')
        cax = fig3d.add_subplot(gs[0, 2])

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

        # draw the fixed box on both plots
        draw_box_edges(ax1, xmin, xmax, ymin, ymax, zmin, zmax, lw=1.0, alpha=0.6)
        draw_box_edges(ax2, xmin, xmax, ymin, ymax, zmin, zmax, lw=1.0, alpha=0.6)

        cb = fig3d.colorbar(m2, cax=cax); cb.set_label(r"$|\nabla u|$")
        fix_matplotlib_3d(ax1); fix_matplotlib_3d(ax2)

        # θ×φ heatmaps
        theta = jnp.linspace(0, 2*jnp.pi, N_bdry_theta, endpoint=True)
        phi   = jnp.linspace(0, 2*jnp.pi, N_bdry_phi,   endpoint=True)
        TH, PH = jnp.meshgrid(theta, phi, indexing='ij')
        GN_init  = Gmag_init
        GN_final = Gmag_final
        vmin_hm = float(jnp.minimum(GN_init.min(), GN_final.min()))
        vmax_hm = float(jnp.maximum(GN_init.max(), GN_final.max()))
        figHM = plt.figure(figsize=(12, 4.5), constrained_layout=True)
        gs = figHM.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
        axL = figHM.add_subplot(gs[0, 0]); axR = figHM.add_subplot(gs[0, 1]); cax = figHM.add_subplot(gs[0, 2])
        imL = axL.pcolormesh(PH, TH, GN_init, shading="auto", cmap=PLOT_CMAP, vmin=vmin_hm, vmax=vmax_hm)
        axL.set_title("Initial  $|\\nabla u|$ on boundary"); axL.set_xlabel(r"$\phi$"); axL.set_ylabel(r"$\theta$")
        imR = axR.pcolormesh(PH, TH, GN_final, shading="auto", cmap=PLOT_CMAP, vmin=vmin_hm, vmax=vmax_hm)
        axR.set_title("Final  $|\\nabla u|$ on boundary"); axR.set_xlabel(r"$\phi$")
        cb = figHM.colorbar(imR, cax=cax); cb.set_label(r"$|\nabla u|$")
        plt.show()

    elif mode == "torus":
        # ===== MULTI-SURFACE PRETRAINING =====
        torus_list = surfaces_cfg.get("torus_list", [])
        if len(torus_list) == 0:
            raise RuntimeError("No torus surfaces listed under [[surfaces.torus_list]] for mode='torus'.")

        # Build dataset of surfaces (each provides boundary points/normals + inside mask)
        dataset = build_torus_family(torus_list, N_bdry_theta, N_bdry_phi)
        print(f"[DATA] Loaded {len(dataset)} torus surfaces for pretraining.")

        # Optional: pick first surface for initial plotting/grids (for visualization later)
        surf0 = dataset[0]
        # Reconstruct Xg,Yg,Zg grids from flattened boundary points (shape nθ×nφ×3)
        Pgrid0 = surf0.P_bdry.reshape(N_bdry_theta, N_bdry_phi, 3)
        Xg0, Yg0, Zg0 = Pgrid0[..., 0], Pgrid0[..., 1], Pgrid0[..., 2]

        # Initial eval on surf0 (use masked interior from the common fixed box)
        mask0 = surf0.inside_mask_fn(P_box)
        ids0  = jnp.nonzero(mask0, size=min(N_in, P_box.shape[0]), fill_value=0)[0]
        P_in0 = P_box[ids0][:N_in]
        (L0, (Lin0, Lbc0, lap0, n0)) = eval_full(model, P_in0, surf0.P_bdry, surf0.N_bdry)
        print(f"[INIT:{surf0.name}] loss={float(L0):.6e}  lap={float(Lin0):.6e}  bc={float(Lbc0):.6e}")

        # Precompute initial boundary grad for surf0
        Gvec_init, Gmag_init, Nhat_grid0 = eval_on_boundary(model, surf0.P_bdry, surf0.N_bdry, Xg0, Yg0, Zg0)

        # Train: at each step pick a random surface and train on it
        key_train = random.PRNGKey(1234)
        log_every = max(1, steps // 20)
        for it in range(1, steps + 1):
            key_train, ks = random.split(key_train)
            # sample surface index
            idx = int(random.randint(ks, shape=(), minval=0, maxval=len(dataset)))
            surf = dataset[idx]
            # masked interior from common fixed set
            mask = surf.inside_mask_fn(P_box)
            ids  = jnp.nonzero(mask, size=min(N_in, P_box.shape[0]), fill_value=0)[0]
            P_in = P_box[ids][:N_in]

            model, opt_state, L, (Lin, Lbc, lap_res, nres, mean_u), gnorm = train_step(
                model, opt_state, optimizer, P_in, surf.P_bdry, surf.N_bdry, ks
            )
            if (it % log_every) == 0 or it == 1:
                lap_rms = float(jnp.sqrt(jnp.mean(lap_res**2)))
                n_rms   = float(jnp.sqrt(jnp.mean(nres**2)))
                lap_max = float(jnp.max(jnp.abs(lap_res)))
                n_max   = float(jnp.max(jnp.abs(nres)))
                print(f"[{it:5d}] loss={float(L):.6e}  "
                    f"lin={float(Lin):.3e}  lbc={float(Lbc):.3e}  "
                    f"|lap|_rms={lap_rms:.3e} (max {lap_max:.2e})  "
                    f"|n·∇u|_rms={n_rms:.3e} (max {n_max:.2e})  "
                    f"|u|_mean={float(mean_u):.3e}  "
                    f"||g||={float(gnorm):.3e}  λ_bc={runtime.lam_bc:.2f}  surf={dataset[idx].name}")

        save_model(model, CHECKPOINT_PATH)

        # Final diagnostics on a couple of surfaces
        for j in range(min(2, len(dataset))):
            surf = dataset[j]
            mask = surf.inside_mask_fn(P_box)
            ids  = jnp.nonzero(mask, size=min(N_in, P_box.shape[0]), fill_value=0)[0]
            P_in = P_box[ids][:N_in]
            (Lf, (Linf, Lbcf, lapf, nf)) = eval_full(model, P_in, surf.P_bdry, surf.N_bdry)
            print(f"[FINAL:{surf.name}] loss={float(Lf):.6e}  lap={float(Linf):.6e}  bc={float(Lbcf):.6e}  "
                  f"|lap|_rms={float(jnp.sqrt(jnp.mean(lapf**2))):.3e}  "
                  f"|n·∇u|_rms={float(jnp.sqrt(jnp.mean(nf**2))):.3e}")

        # Plot surf0 initial vs final
        Gvec_final, Gmag_final, _ = eval_on_boundary(model, surf0.P_bdry, surf0.N_bdry, Xg0, Yg0, Zg0)
        vmin_shared = float(jnp.minimum(Gmag_init.min(), Gmag_final.min()))
        vmax_shared = float(jnp.maximum(Gmag_init.max(), Gmag_final.max()))
        fig3d = plt.figure(figsize=(12, 6), constrained_layout=True)
        gs = fig3d.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
        ax1 = fig3d.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig3d.add_subplot(gs[0, 1], projection='3d')
        cax = fig3d.add_subplot(gs[0, 2])

        offset = 0.1 * float(a0 + abs(a1))
        m1 = plot_surface_with_vectors_ax(ax1, Xg0, Yg0, Zg0, Gmag_init, Nhat_grid0, Gvec=Gvec_init,
                                           title=f"Initial |∇u| ({surf0.name})",
                                           cmap=PLOT_CMAP, quiver_len=0.15,
                                           step_theta=6, step_phi=8, plot_normals=False,
                                           vmin=vmin_shared, vmax=vmax_shared,
                                           surf_offset=offset)
        m2 = plot_surface_with_vectors_ax(ax2, Xg0, Yg0, Zg0, Gmag_final, Nhat_grid0, Gvec=Gvec_final,
                                           title=f"Final |∇u| ({surf0.name})",
                                           cmap=PLOT_CMAP, quiver_len=0.15,
                                           step_theta=6, step_phi=8, plot_normals=False,
                                           vmin=vmin_shared, vmax=vmax_shared,
                                           surf_offset=offset)

        draw_box_edges(ax1, xmin, xmax, ymin, ymax, zmin, zmax, lw=1.0, alpha=0.6)
        draw_box_edges(ax2, xmin, xmax, ymin, ymax, zmin, zmax, lw=1.0, alpha=0.6)

        cb = fig3d.colorbar(m2, cax=cax); cb.set_label(r"$|\nabla u|$")
        fix_matplotlib_3d(ax1); fix_matplotlib_3d(ax2)
        plt.show()

    else:
        raise NotImplementedError(f"Unknown [surfaces].mode={mode!r}. Use 'single' or 'torus'.")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PINN for Laplace on a torus (single or multi-surface)")
    parser.add_argument("--config", "-c", type=str, default="input.toml",
                        help="Path to TOML configuration file")
    args = parser.parse_args()
    _ = main(config_path=args.config)