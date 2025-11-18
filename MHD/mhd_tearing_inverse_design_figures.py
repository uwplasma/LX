#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_inverse_design_figures.py

Paper figures for:

  - Reachable region in (f_kin, C_plasmoid) space for tearing/plasmoid reconnection.
  - Comparison of gradient-based inverse design vs. grid (black-box) search.
  - Robustness / differences across equilibria ("original" vs "forcefree").

This script relies on:

  * mhd_tearing_solve.py
  * mhd_tearing_inverse_design.py
  * mhd_tearing_postprocess.py   (for post-run analysis if desired)

It will:

  1) Perform a (log10_eta, log10_nu) grid scan for each equilibrium mode.
  2) Build publication-ready plots:
       - Heatmaps of f_kin and C_plasmoid vs (log10_eta, log10_nu).
       - Scatter of reachable region in (f_kin, C_plasmoid) for each mode.
       - Combined reachable region plot ("original" vs "forcefree").
  3) For a chosen target behaviour (f_kin*, C_plasmoid*), compare:
       - Best point from the grid scan (black-box search).
       - Gradient-based inverse design (differentiable).
     and show:
       - Cost vs number of simulations.
       - Location of the solutions in (f_kin, C_plasmoid) space.

All figures are saved as high-resolution PNGs with descriptive filenames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import equinox as eqx
import optax

from mhd_tearing_inverse_design import (
    InverseDesignConfig,
    DesignMLP,
    _simulate_metrics,
    build_training_step,
)


# -----------------------------------------------------------------------------#
# Global plotting style
# -----------------------------------------------------------------------------#

plt.rcParams.update({
    "font.size": 13,
    "text.usetex": False,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 220,
    "axes.linewidth": 1.1,
    "lines.linewidth": 1.8,
})


# -----------------------------------------------------------------------------#
# Figure / scan configuration
# -----------------------------------------------------------------------------#

@dataclass
class FigureScanConfig:
    # Grid for dissipation parameters (log10 space)
    n_eta: int = 4            # increase to 6–8 for final paper
    n_nu: int = 4
    log10_eta_min: float = -4.5
    log10_eta_max: float = -2.0
    log10_nu_min: float = -4.5
    log10_nu_max: float = -2.0

    # Time integration / grid (shared with InverseDesignConfig)
    Nx: int = 64
    Ny: int = 64
    Nz: int = 1
    Lx: float = 2.0 * math.pi
    Ly: float = 2.0 * math.pi
    Lz: float = 2.0 * math.pi
    B0: float = 1.0
    B_g: float = 0.2
    a: float = 0.25
    eps_B: float = 1e-3
    t0: float = 0.0
    t1: float = 60.0
    n_frames: int = 120
    dt0: float = 5e-4

    # Target behaviour for inverse design experiment
    target_f_kin: float = 0.03       # tweak after first scan
    target_complexity: float = 1e-5  # tweak after first scan
    lambda_complexity: float = 1.0

    # Inverse design training loop
    latent_dim: int = 1
    hidden_width: int = 32
    hidden_depth: int = 2
    learning_rate: float = 1e-2
    n_train_steps: int = 8   # each step = one sim; increase for final paper
    print_every: int = 1

    # Latent z (we can use z=0 for a single target)
    z_train: float = 0.0

    # Random seed
    seed: int = 2025


# -----------------------------------------------------------------------------#
# Helper: run one metric simulation
# -----------------------------------------------------------------------------#

def run_metrics_for_eq_mode(
    eta: float,
    nu: float,
    eq_mode: str,
    fig_cfg: FigureScanConfig,
) -> Tuple[float, float]:
    """
    Wrapper around _simulate_metrics using an InverseDesignConfig for a
    given equilibrium_mode and dissipation parameters (eta, nu).

    Returns
    -------
    f_kin, complexity
    """
    cfg = InverseDesignConfig(
        Nx=fig_cfg.Nx,
        Ny=fig_cfg.Ny,
        Nz=fig_cfg.Nz,
        Lx=fig_cfg.Lx,
        Ly=fig_cfg.Ly,
        Lz=fig_cfg.Lz,
        B0=fig_cfg.B0,
        B_g=fig_cfg.B_g,
        a=fig_cfg.a,
        eps_B=fig_cfg.eps_B,
        t0=fig_cfg.t0,
        t1=fig_cfg.t1,
        n_frames=fig_cfg.n_frames,
        dt0=fig_cfg.dt0,
        equilibrium_mode=eq_mode,
        target_f_kin=fig_cfg.target_f_kin,
        target_complexity=fig_cfg.target_complexity,
        lambda_complexity=fig_cfg.lambda_complexity,
        log10_eta_min=fig_cfg.log10_eta_min,
        log10_eta_max=fig_cfg.log10_eta_max,
        log10_nu_min=fig_cfg.log10_nu_min,
        log10_nu_max=fig_cfg.log10_nu_max,
        latent_dim=fig_cfg.latent_dim,
        hidden_width=fig_cfg.hidden_width,
        hidden_depth=fig_cfg.hidden_depth,
        learning_rate=fig_cfg.learning_rate,
        n_train_steps=fig_cfg.n_train_steps,
        print_every=fig_cfg.print_every,
        z_train=fig_cfg.z_train,
        seed=fig_cfg.seed,
    )

    f_kin, comp, _ = _simulate_metrics(jnp.asarray(eta), jnp.asarray(nu), cfg)
    return float(f_kin), float(comp)


# -----------------------------------------------------------------------------#
# 1) Parameter scans for reachable regions
# -----------------------------------------------------------------------------#

def parameter_scan(fig_cfg: FigureScanConfig, eq_mode: str):
    """
    Perform a (log10_eta, log10_nu) grid scan for a given equilibrium_mode.

    Returns
    -------
    scan_data : dict with fields:
        log10_eta_grid, log10_nu_grid, f_kin_grid, C_grid
    """
    log10_eta_vals = np.linspace(fig_cfg.log10_eta_min, fig_cfg.log10_eta_max, fig_cfg.n_eta)
    log10_nu_vals  = np.linspace(fig_cfg.log10_nu_min,  fig_cfg.log10_nu_max,  fig_cfg.n_nu)

    f_kin_grid = np.zeros((fig_cfg.n_eta, fig_cfg.n_nu))
    C_grid     = np.zeros((fig_cfg.n_eta, fig_cfg.n_nu))

    print(f"\n[SCAN] Starting parameter scan for equilibrium_mode='{eq_mode}'")
    print(f"       n_eta={fig_cfg.n_eta}, n_nu={fig_cfg.n_nu}")
    for i, log10_eta in enumerate(log10_eta_vals):
        for j, log10_nu in enumerate(log10_nu_vals):
            eta = 10.0**log10_eta
            nu  = 10.0**log10_nu
            print(
                f"[SCAN] eq={eq_mode:9s}, i={i}/{fig_cfg.n_eta-1}, j={j}/{fig_cfg.n_nu-1}, "
                f"log10_eta={log10_eta:.3f}, log10_nu={log10_nu:.3f}"
            )
            f_kin, comp = run_metrics_for_eq_mode(eta, nu, eq_mode, fig_cfg)
            f_kin_grid[i, j] = f_kin
            C_grid[i, j]     = comp

    scan_data = {
        "log10_eta_vals": log10_eta_vals,
        "log10_nu_vals": log10_nu_vals,
        "f_kin_grid": f_kin_grid,
        "C_grid": C_grid,
        "eq_mode": eq_mode,
    }

    np.savez(f"reachable_region_scan_{eq_mode}.npz", **scan_data)
    print(f"[SAVE] Saved scan data to reachable_region_scan_{eq_mode}.npz")

    return scan_data


def plot_scan_heatmaps(scan_data: Dict[str, Any]):
    log10_eta = scan_data["log10_eta_vals"]
    log10_nu  = scan_data["log10_nu_vals"]
    f_kin     = scan_data["f_kin_grid"]
    C_grid    = scan_data["C_grid"]
    eq_mode   = scan_data["eq_mode"]

    X, Y = np.meshgrid(log10_nu, log10_eta)  # columns=nu, rows=eta

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    im0 = axes[0].pcolormesh(
        X, Y, f_kin,
        shading="auto",
    )
    c0 = fig.colorbar(im0, ax=axes[0])
    c0.set_label(r"$f_{\mathrm{kin}}$")
    axes[0].set_xlabel(r"$\log_{10}\nu$")
    axes[0].set_ylabel(r"$\log_{10}\eta$")
    axes[0].set_title(rf"$f_{{\rm kin}}$ for '{eq_mode}'")

    im1 = axes[1].pcolormesh(
        X, Y, C_grid,
        shading="auto",
    )
    c1 = fig.colorbar(im1, ax=axes[1])
    c1.set_label(r"$C_{\mathrm{plasmoid}}$")
    axes[1].set_xlabel(r"$\log_{10}\nu$")
    axes[1].set_ylabel(r"$\log_{10}\eta$")
    axes[1].set_title(rf"$C_{{\rm plasmoid}}$ for '{eq_mode}'")

    fig.suptitle(rf"Reachable dissipation space: '{eq_mode}'", fontsize=14)
    outname = f"fig_reachable_heatmaps_{eq_mode}.png"
    fig.savefig(outname, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved {outname}")


def plot_reachable_region_plane(
    scan_orig: Dict[str, Any],
    scan_ff: Dict[str, Any],
    fig_cfg: FigureScanConfig,
):
    """
    Combined reachable region in (f_kin, C_plasmoid) plane for both
    equilibria, with target point marked.
    """
    f_o = scan_orig["f_kin_grid"].ravel()
    C_o = scan_orig["C_grid"].ravel()

    f_f = scan_ff["f_kin_grid"].ravel()
    C_f = scan_ff["C_grid"].ravel()

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)

    ax.scatter(
        f_o, C_o,
        s=40,
        marker="o",
        edgecolor="none",
        alpha=0.8,
        label="original eq.",
    )
    ax.scatter(
        f_f, C_f,
        s=40,
        marker="^",
        edgecolor="none",
        alpha=0.8,
        label="force-free eq.",
    )

    # Mark the target behaviour used for inverse design
    ax.scatter(
        [fig_cfg.target_f_kin],
        [fig_cfg.target_complexity],
        s=80,
        marker="*",
        color="k",
        label="target behaviour",
    )

    ax.set_xlabel(r"$f_{\mathrm{kin}}$ (late-time kinetic-energy fraction)")
    ax.set_ylabel(r"$C_{\mathrm{plasmoid}}$ (midplane complexity)")
    ax.set_title(r"Reachable $(f_{\rm kin}, C_{\rm plasmoid})$ region")
    ax.grid(True, alpha=0.3)
    ax.legend()

    outname = "fig_reachable_region_fkin_Cplasmoid.png"
    fig.savefig(outname, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved {outname}")


# -----------------------------------------------------------------------------#
# 2) Inverse design vs grid search
# -----------------------------------------------------------------------------#

def find_grid_best_for_target(
    scan_data: Dict[str, Any],
    f_target: float,
    C_target: float,
    lambda_complexity: float,
) -> Dict[str, float]:
    """
    Given a scan (for single equilibrium_mode), find the grid point that
    minimizes the cost

        (f - f_target)^2 + lambda * (C - C_target)^2.

    Returns a dict with:
        f_best, C_best, cost_best, log10_eta_best, log10_nu_best
    """
    f_grid = scan_data["f_kin_grid"]
    C_grid = scan_data["C_grid"]
    log10_eta = scan_data["log10_eta_vals"]
    log10_nu  = scan_data["log10_nu_vals"]

    n_eta, n_nu = f_grid.shape

    best_cost = np.inf
    best = {}

    eval_cost_history = []

    for i in range(n_eta):
        for j in range(n_nu):
            f = f_grid[i, j]
            C = C_grid[i, j]
            cost = (f - f_target)**2 + lambda_complexity * (C - C_target)**2
            eval_cost_history.append(cost)
            if cost < best_cost:
                best_cost = cost
                best = {
                    "f_best": f,
                    "C_best": C,
                    "cost_best": cost,
                    "log10_eta_best": float(log10_eta[i]),
                    "log10_nu_best": float(log10_nu[j]),
                }

    # best-so-far curve vs number of evals
    eval_cost_history = np.array(eval_cost_history)
    best_so_far = np.minimum.accumulate(eval_cost_history)

    best["eval_cost_history"] = eval_cost_history
    best["best_so_far"] = best_so_far

    return best


def run_inverse_design_for_mode(
    fig_cfg: FigureScanConfig,
    eq_mode: str,
) -> Dict[str, Any]:
    """
    Run gradient-based inverse design for a given equilibrium_mode,
    using mhd_tearing_inverse_design's infrastructure.

    Returns
    -------
    result : dict containing:
        history (dict), res_final, eta_final, nu_final, cost_history
    """
    cfg = InverseDesignConfig(
        Nx=fig_cfg.Nx,
        Ny=fig_cfg.Ny,
        Nz=fig_cfg.Nz,
        Lx=fig_cfg.Lx,
        Ly=fig_cfg.Ly,
        Lz=fig_cfg.Lz,
        B0=fig_cfg.B0,
        B_g=fig_cfg.B_g,
        a=fig_cfg.a,
        eps_B=fig_cfg.eps_B,
        t0=fig_cfg.t0,
        t1=fig_cfg.t1,
        n_frames=fig_cfg.n_frames,
        dt0=fig_cfg.dt0,
        equilibrium_mode=eq_mode,
        target_f_kin=fig_cfg.target_f_kin,
        target_complexity=fig_cfg.target_complexity,
        lambda_complexity=fig_cfg.lambda_complexity,
        log10_eta_min=fig_cfg.log10_eta_min,
        log10_eta_max=fig_cfg.log10_eta_max,
        log10_nu_min=fig_cfg.log10_nu_min,
        log10_nu_max=fig_cfg.log10_nu_max,
        latent_dim=fig_cfg.latent_dim,
        hidden_width=fig_cfg.hidden_width,
        hidden_depth=fig_cfg.hidden_depth,
        learning_rate=fig_cfg.learning_rate,
        n_train_steps=fig_cfg.n_train_steps,
        print_every=fig_cfg.print_every,
        z_train=fig_cfg.z_train,
        seed=fig_cfg.seed,
    )

    print(f"\n[INV] Starting inverse design for equilibrium_mode='{eq_mode}'")
    key = jax.random.PRNGKey(cfg.seed)
    key_model, key_train = jax.random.split(key)

    model = DesignMLP(
        in_dim=cfg.latent_dim,
        hidden_width=cfg.hidden_width,
        hidden_depth=cfg.hidden_depth,
        key=key_model,
    )

    optimizer = optax.adam(cfg.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    training_step = build_training_step(cfg)

    history = {
        "loss": [],
        "log10_eta": [],
        "log10_nu": [],
        "eta": [],
        "nu": [],
        "f_kin": [],
        "complexity": [],
    }

    last_aux = None
    cost_history = []

    for step in range(cfg.n_train_steps):
        key_train, key_step = jax.random.split(key_train)
        model, opt_state, loss_val, aux = training_step(
            model, opt_state, key_step, optimizer
        )

        loss_float = float(loss_val)
        log10_eta = float(aux["log10_eta"])
        log10_nu  = float(aux["log10_nu"])
        eta       = float(aux["eta"])
        nu        = float(aux["nu"])
        f_kin     = float(aux["f_kin"])
        comp      = float(aux["complexity"])

        history["loss"].append(loss_float)
        history["log10_eta"].append(log10_eta)
        history["log10_nu"].append(log10_nu)
        history["eta"].append(eta)
        history["nu"].append(nu)
        history["f_kin"].append(f_kin)
        history["complexity"].append(comp)

        # "Cost" in behaviour space (matching grid definition)
        cost_beh = (f_kin - cfg.target_f_kin)**2 + cfg.lambda_complexity * (comp - cfg.target_complexity)**2
        cost_history.append(cost_beh)

        if (step % cfg.print_every) == 0:
            print(
                f"[INV {eq_mode:9s} step {step:02d}] "
                f"L={loss_float:.3e}, "
                f"log10_eta={log10_eta:.3f}, log10_nu={log10_nu:.3f}, "
                f"eta={eta:.3e}, nu={nu:.3e}, "
                f"f_kin={f_kin:.4f}, complexity={comp:.3e}"
            )

        last_aux = aux

    assert last_aux is not None
    res_final = last_aux["res"]
    eta_final = float(last_aux["eta"])
    nu_final  = float(last_aux["nu"])

    print(
        f"[INV FINAL] eq={eq_mode}, "
        f"eta_final={eta_final:.3e}, nu_final={nu_final:.3e}, "
        f"f_kin_final={history['f_kin'][-1]:.4f}, "
        f"C_final={history['complexity'][-1]:.3e}"
    )

    result = {
        "history": history,
        "res_final": res_final,
        "eta_final": eta_final,
        "nu_final": nu_final,
        "cost_history": np.array(cost_history),
    }
    return result


def plot_inverse_vs_grid_for_mode(
    eq_mode: str,
    scan_data: Dict[str, Any],
    inv_result: Dict[str, Any],
    fig_cfg: FigureScanConfig,
):
    """
    For a single equilibrium_mode:

      - Plot cost vs number of simulations: grid search vs inverse design.
      - Plot (f_kin, C_plasmoid) reachable points, and mark:
          * grid-best
          * inverse-design final point
          * target

    Produces:
      fig_inverse_vs_grid_<eq_mode>.png
    """
    f_grid = scan_data["f_kin_grid"].ravel()
    C_grid = scan_data["C_grid"].ravel()

    # Grid-best info
    grid_best = find_grid_best_for_target(
        scan_data,
        fig_cfg.target_f_kin,
        fig_cfg.target_complexity,
        fig_cfg.lambda_complexity,
    )
    grid_eval_cost = grid_best["eval_cost_history"]
    grid_best_so_far = grid_best["best_so_far"]

    # Inverse-design info
    cost_inv = inv_result["cost_history"]
    hist = inv_result["history"]
    f_inv_final = hist["f_kin"][-1]
    C_inv_final = hist["complexity"][-1]

    # cost vs # simulations plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    n_grid_evals = np.arange(1, len(grid_best_so_far) + 1)
    n_inv_evals  = np.arange(1, len(cost_inv) + 1)

    axes[0].semilogy(n_grid_evals, grid_best_so_far, "o-", label="grid search")
    axes[0].semilogy(n_inv_evals, cost_inv, "s-", label="inverse design")
    axes[0].set_xlabel("number of simulations")
    axes[0].set_ylabel("behaviour-space cost")
    axes[0].set_title(rf"Cost vs simulations ('{eq_mode}')")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    # (f_kin, C_plasmoid) plane
    axes[1].scatter(
        f_grid, C_grid,
        s=30,
        alpha=0.5,
        edgecolor="none",
        label="grid samples",
    )
    axes[1].scatter(
        grid_best["f_best"], grid_best["C_best"],
        s=80,
        marker="D",
        color="tab:blue",
        label="grid best",
    )
    axes[1].scatter(
        f_inv_final, C_inv_final,
        s=80,
        marker="^",
        color="tab:orange",
        label="inverse design",
    )
    axes[1].scatter(
        [fig_cfg.target_f_kin],
        [fig_cfg.target_complexity],
        s=90,
        marker="*",
        color="k",
        label="target",
    )

    axes[1].set_xlabel(r"$f_{\mathrm{kin}}$")
    axes[1].set_ylabel(r"$C_{\mathrm{plasmoid}}$")
    axes[1].set_title(rf"Solutions in $(f_{{\rm kin}}, C_{{\rm plasmoid}})$ ('{eq_mode}')")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(rf"Inverse design vs grid search ('{eq_mode}')", fontsize=14)
    outname = f"fig_inverse_vs_grid_{eq_mode}.png"
    fig.savefig(outname, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved {outname}")


# -----------------------------------------------------------------------------#
# Main orchestration
# -----------------------------------------------------------------------------#

def main():
    fig_cfg = FigureScanConfig()

    print("========================================================")
    print(" MHD tearing/plasmoid: inverse design figure generator ")
    print("========================================================")
    print(fig_cfg)

    # 1) Parameter scans for both equilibria
    scan_orig = parameter_scan(fig_cfg, eq_mode="original")
    scan_ff   = parameter_scan(fig_cfg, eq_mode="forcefree")

    # Heatmaps for each
    plot_scan_heatmaps(scan_orig)
    plot_scan_heatmaps(scan_ff)

    # Combined reachable region in (f_kin, C_plasmoid) plane
    plot_reachable_region_plane(scan_orig, scan_ff, fig_cfg)

    # 2) Inverse design vs grid for each equilibrium_mode separately
    inv_orig = run_inverse_design_for_mode(fig_cfg, eq_mode="original")
    inv_ff   = run_inverse_design_for_mode(fig_cfg, eq_mode="forcefree")

    plot_inverse_vs_grid_for_mode("original", scan_orig, inv_orig, fig_cfg)
    plot_inverse_vs_grid_for_mode("forcefree", scan_ff, inv_ff, fig_cfg)

    print("\n[DONE] All figures generated. You can now:")
    print("  - Inspect reachable_region_scan_*.npz")
    print("  - Use mhd_tearing_postprocess.py on any saved NPZ from the inverse-design script")
    print("  - Drop the generated PNGs directly into the paper.")


if __name__ == "__main__":
    main()
