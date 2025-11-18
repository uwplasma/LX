#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_postprocess_ml_v2.py

Comprehensive postprocessing for the v2 data-driven reduced model of
Harris-sheet tearing (equilibrium_mode="original").

This script assumes that you have already run:

    python mhd_tearing_ml_v2.py

which creates:
  - output_ml/surrogate_model.pkl
  - output_ml/latent_ode_model.pkl
  - output_ml/tearing_ml_runs/tearing_ml_case_*.npz
  - training-time diagnostic plots (loss curves, parity, recon examples)

Here we:
  1) Rebuild the *full* dataset directly from the saved MHD runs:
       • tearing amplitude A(t)
       • fitted growth rate γ_fit from log A(t)
       • saturated amplitude A_sat
       • physics-aware feature vectors eq_params (S, Pm, k_y a, ε_B)
       • optional theoretical growth-rate estimate γ_th (stored as gamma_FKR)
       • Lundquist number S from the MHD runs

  2) Reload the trained models:
       • surrogate_model.pkl   (eq_params → γ_fit, A_sat)
       • latent_ode_model.pkl  (latent ODE autoencoder for A(t))

  3) Evaluate the models on all cases:
       • surrogate predictions (γ_pred, A_sat,pred)
       • latent ODE reconstructions Â(t) and reconstruction errors

  4) Reconstruct the same train/validation split used in training
     (same RNG seed and 75/25 split).

  5) Produce publication-ready figures summarizing the ML performance and
     the underlying tearing physics:

       (i)   Surrogate parity plots γ_pred vs γ_fit and A_sat,pred vs A_sat
             (train vs validation) with 1:1 lines and R^2.
       (ii)  Relative-error histograms for γ and A_sat.
       (iii) γ_fit and γ_pred versus S, with a Sweet–Parker-like γ ∝ S^{-1/2}
             reference scaling.
       (iv)  Latent-ODE reconstructions A(t) for representative cases
             (slow, intermediate, and fast / strongly nonlinear).
       (v)   Latent-ODE reconstruction error statistics.
       (vi)  Latent phase portraits z_1 vs z_2 illustrating the low-dimensional
             dynamical manifold learned by the model.

All outputs are written into the same directory `output_ml/` (or a directory
specified via --ml-dir).
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, Any, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 200,
        "text.usetex": False,
    }
)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Physics utilities from the main MHD solver
from mhd_tearing_solve import tearing_amplitude

# ML helpers and model I/O from the v2 training script
from mhd_tearing_ml_v2 import (
    safe_log,
    fit_growth_rate,
    compute_saturated_amplitude,
    GridConfig,
    PhysicalParams,
    build_feature_vector,
    load_surrogate_model,
    load_latent_ode_model,
    latent_ode_forward_single,
    surrogate_predict,
)

# -----------------------------------------------------------------------------#
# Arg parsing and small helpers
# -----------------------------------------------------------------------------#


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess v2 MHD tearing ML models using output_ml/"
    )
    parser.add_argument(
        "--ml-dir",
        type=str,
        default="output_ml",
        help="Directory containing surrogate_model.pkl, "
             "latent_ode_model.pkl, and tearing_ml_runs/ (default: output_ml)",
    )
    parser.add_argument(
        "--runs-subdir",
        type=str,
        default="tearing_ml_runs",
        help="Subdirectory inside ml-dir with tearing_ml_case_*.npz files "
             "(default: tearing_ml_runs)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=3,
        help="Number of representative cases to show in the A(t) recon plots "
             "(default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for the train/validation split and for "
             "choosing representative examples (default: 0)",
    )
    return parser.parse_args()


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num = np.sum((y_pred - y_true) ** 2)
    den = np.sum((y_true - y_true.mean()) ** 2) + 1e-16
    return float(1.0 - num / den)


# -----------------------------------------------------------------------------#
# Dataset reconstruction from tearing_ml_case_*.npz
# -----------------------------------------------------------------------------#


def load_dataset_from_runs(
    ml_dir: str,
    runs_subdir: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Rebuild the ML dataset from individual MHD tearing runs.

    Returns a dict with numpy arrays:
        ts        : (T,)
        amp       : (N, T)
        gamma_fit : (N,)
        A_sat     : (N,)
        gamma_th  : (N,)   (NaN if not present in file; stored as gamma_FKR)
        S         : (N,)   Lundquist number from the MHD runs
        eq_params : (N, P) physics-aware feature vectors used in training
        meta      : list of per-case dictionaries with raw parameters
    """
    runs_dir = os.path.join(ml_dir, runs_subdir)
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    pattern = os.path.join(runs_dir, "tearing_ml_case_*.npz")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No runs found matching {pattern}")

    if verbose:
        print(f"[DATASET] Found {len(files)} runs in {runs_dir}")

    ts_ref = None
    amp_all: List[np.ndarray] = []
    gamma_fit_all: List[float] = []
    A_sat_all: List[float] = []
    gamma_th_all: List[float] = []
    S_all: List[float] = []
    eq_params_all: List[np.ndarray] = []
    meta_all: List[Dict[str, Any]] = []

    for idx, fname in enumerate(files):
        data = np.load(fname, allow_pickle=True)
        ts = np.array(data["ts"], dtype=np.float64)

        if ts_ref is None:
            ts_ref = ts
        else:
            if ts.shape != ts_ref.shape or not np.allclose(ts, ts_ref):
                raise ValueError(
                    "All runs are expected to share the same time grid ts."
                )

        Nx = int(data["Nx"])
        Ny = int(data["Ny"])
        Nz = int(data["Nz"])
        Lx = float(data["Lx"])
        Ly = float(data["Ly"])
        Lz = float(data["Lz"])
        nu = float(data["nu"])
        eta = float(data["eta"])
        B0 = float(data["B0"])
        a = float(data["a"])
        B_g = float(data["B_g"])
        eps_B = float(data["eps_B"])
        # Theoretical estimate from the solver (FKR-like), kept as "gamma_th"
        gamma_th = float(data["gamma_FKR"]) if "gamma_FKR" in data else np.nan
        S_val = float(data["S"]) if "S" in data else np.nan

        B_hat_frames = jnp.array(data["B_hat"])  # (T, 3, Nx, Ny, Nz)

        # Recompute tearing amplitude A(t) in the *same way* as in training
        amp_case: List[float] = []
        for k in range(B_hat_frames.shape[0]):
            A_k = tearing_amplitude(B_hat_frames[k], Lx, Ly, Lz)
            amp_case.append(float(A_k))
        amp_case = np.array(amp_case, dtype=np.float64)

        # Growth rate and saturation from the simulation
        ts_j = jnp.asarray(ts_ref)
        amp_j = jnp.asarray(amp_case)
        gamma_fit = float(fit_growth_rate(ts_j, amp_j))
        A_sat = float(compute_saturated_amplitude(amp_j))

        # Physics-aware feature vector (same as in training)
        grid = GridConfig(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
        phys = PhysicalParams(
            nu=nu, eta=eta, B0=B0, a=a, B_g=B_g, eps_B=eps_B
        )
        eq_feat = np.array(build_feature_vector(grid, phys))

        amp_all.append(amp_case)
        gamma_fit_all.append(gamma_fit)
        A_sat_all.append(A_sat)
        gamma_th_all.append(gamma_th)
        S_all.append(S_val)
        eq_params_all.append(eq_feat)

        meta_all.append(
            dict(
                filename=fname,
                Nx=Nx,
                Ny=Ny,
                Nz=Nz,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                nu=nu,
                eta=eta,
                B0=B0,
                a=a,
                B_g=B_g,
                eps_B=eps_B,
                gamma_th=gamma_th,
                S=S_val,
            )
        )

        if verbose:
            print(
                f"[CASE {idx:02d}] "
                f"a={a:.3f}, eta={eta:.3e}, nu={nu:.3e}, "
                f"gamma_fit={gamma_fit:.3e}, A_sat={A_sat:.3f}, "
                f"S={S_val:.3e}, gamma_th={gamma_th:.3e}"
            )

    dataset = dict(
        ts=ts_ref,
        amp=np.stack(amp_all, axis=0),
        gamma_fit=np.array(gamma_fit_all),
        A_sat=np.array(A_sat_all),
        gamma_th=np.array(gamma_th_all),
        S=np.array(S_all),
        eq_params=np.stack(eq_params_all, axis=0),
        meta=meta_all,
    )

    if verbose:
        print("\n[DATASET] Reconstructed dataset:")
        print(f"  ts        : {dataset['ts'].shape}")
        print(f"  amp       : {dataset['amp'].shape}")
        print(f"  gamma_fit : {dataset['gamma_fit'].shape}")
        print(f"  A_sat     : {dataset['A_sat'].shape}")
        print(f"  S         : {dataset['S'].shape}")
        print(f"  eq_params : {dataset['eq_params'].shape}")

    return dataset


# -----------------------------------------------------------------------------#
# Plotting routines
# -----------------------------------------------------------------------------#


def plot_surrogate_parity_and_errors(
    ml_dir: str,
    gamma_true: np.ndarray,
    gamma_pred: np.ndarray,
    Asat_true: np.ndarray,
    Asat_pred: np.ndarray,
    seed: int = 0,
) -> None:
    """Parity plots + error histograms for the surrogate model."""

    N = gamma_true.shape[0]
    rng = np.random.default_rng(seed)
    idx_all = rng.permutation(N)
    n_train = max(2, int(0.75 * N))
    train_idx = idx_all[:n_train]
    val_idx = idx_all[n_train:]

    g_tr, g_v = gamma_true[train_idx], gamma_true[val_idx]
    gptr, gpv = gamma_pred[train_idx], gamma_pred[val_idx]
    A_tr, A_v = Asat_true[train_idx], Asat_true[val_idx]
    Aptr, Apv = Asat_pred[train_idx], Asat_pred[val_idx]

    r2_gamma = compute_r2(gamma_true, gamma_pred)
    r2_Asat = compute_r2(Asat_true, Asat_pred)

    # --- Parity plots ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.scatter(g_tr, gptr, marker="o", label="Train", alpha=0.9)
    ax.scatter(g_v, gpv, marker="s", label="Val", alpha=0.9)
    gmin, gmax = np.min(gamma_true), np.max(gamma_true)
    pad = 0.05 * (gmax - gmin + 1e-8)
    ax.plot(
        [gmin - pad, gmax + pad],
        [gmin - pad, gmax + pad],
        "k--",
        lw=1,
        label="Ideal",
    )
    ax.set_xlabel(r"$\gamma_{\mathrm{fit}}$ (simulation)")
    ax.set_ylabel(r"$\gamma_{\mathrm{pred}}$ (surrogate)")
    ax.set_title(r"Growth rate surrogate, $R^2 \approx %.3f$" % r2_gamma)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax2 = axes[1]
    ax2.scatter(A_tr, Aptr, marker="o", label="Train", alpha=0.9)
    ax2.scatter(A_v, Apv, marker="s", label="Val", alpha=0.9)
    Amin, Amax = np.min(Asat_true), np.max(Asat_true)
    padA = 0.05 * (Amax - Amin + 1e-8)
    ax2.plot(
        [Amin - padA, Amax + padA],
        [Amin - padA, Amax + padA],
        "k--",
        lw=1,
        label="Ideal",
    )
    ax2.set_xlabel(r"$A_{\mathrm{sat}}$ (simulation)")
    ax2.set_ylabel(r"$A_{\mathrm{sat,pred}}$ (surrogate)")
    ax2.set_title(r"Saturated amplitude surrogate, $R^2 \approx %.3f$" % r2_Asat)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False)

    fig.tight_layout()
    out_parity = os.path.join(ml_dir, "tearing_ml_v2_surrogate_parity.png")
    fig.savefig(out_parity, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_parity}")

    # --- Error histograms ---
    rel_err_gamma = (gamma_pred - gamma_true) / (gamma_true + 1e-16)
    rel_err_Asat = (Asat_pred - Asat_true) / (Asat_true + 1e-16)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.hist(rel_err_gamma, bins=15, edgecolor="k", alpha=0.85)
    ax.set_xlabel(
        r"Relative error in $\gamma$ "
        r"$\left(\gamma_{\mathrm{pred}}-\gamma_{\mathrm{fit}}\right)"
        r"/\gamma_{\mathrm{fit}}$"
    )
    ax.set_ylabel("Count")
    ax.set_title("Surrogate growth-rate errors")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(rel_err_Asat, bins=15, edgecolor="k", alpha=0.85)
    ax2.set_xlabel(
        r"Relative error in $A_{\mathrm{sat}}$ "
        r"$\left(A_{\mathrm{sat,pred}}-A_{\mathrm{sat}}\right)/A_{\mathrm{sat}}$"
    )
    ax2.set_ylabel("Count")
    ax2.set_title("Surrogate saturation-level errors")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_hist = os.path.join(ml_dir, "tearing_ml_v2_surrogate_error_hist.png")
    fig.savefig(out_hist, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_hist}")


def plot_gamma_vs_S(
    ml_dir: str,
    S_arr: np.ndarray,
    gamma_fit: np.ndarray,
    gamma_pred: np.ndarray,
) -> None:
    """
    Plot linear tearing growth rate versus Lundquist number S, comparing
    the full MHD simulations and the surrogate. A Sweet–Parker-like
    reference scaling γ ∝ S^{-1/2} is shown as a guide to the eye.
    """
    # Mask any non-finite S
    mask = np.isfinite(S_arr) & (S_arr > 0.0)
    if not np.any(mask):
        print("[WARN] No finite S values found; skipping γ vs S plot.")
        return

    S = S_arr[mask]
    g_fit = gamma_fit[mask]
    g_pred = gamma_pred[mask]

    fig, ax = plt.subplots(figsize=(5.8, 4.2))

    ax.scatter(S, g_fit, s=40, marker="o", label=r"Simulation $\gamma_{\mathrm{fit}}$")
    ax.scatter(S, g_pred, s=40, marker="s", label=r"Surrogate $\gamma_{\mathrm{pred}}$")

    # Sweet–Parker-like reference: γ ∝ S^{-1/2}, anchored at median point
    S_ref = np.linspace(0.9 * np.min(S), 1.1 * np.max(S), 200)
    S0 = np.median(S)
    g0 = np.median(g_fit)
    gamma_sp = g0 * (S_ref / S0) ** (-0.5)
    ax.plot(
        S_ref,
        gamma_sp,
        "k--",
        lw=1.2,
        label=r"$\propto S^{-1/2}$ (Sweet--Parker ref.)",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Lundquist number $S$")
    ax.set_ylabel(r"Growth rate $\gamma$")
    ax.set_title("Linear tearing growth versus resistive Lundquist number")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    out = os.path.join(ml_dir, "tearing_ml_v2_gamma_vs_S.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out}")


def plot_latent_recon_examples(
    ml_dir: str,
    ts: np.ndarray,
    amp_true: np.ndarray,
    amp_hat: np.ndarray,
    indices: List[int],
) -> None:
    """Show A(t) vs latent-ODE reconstruction for a few representative cases."""
    ts_np = np.array(ts)
    n_cases = len(indices)
    fig, axes = plt.subplots(1, n_cases, figsize=(5 * n_cases, 3.5), sharey=True)

    if n_cases == 1:
        axes = [axes]

    for j, idx in enumerate(indices):
        ax = axes[j]
        A = amp_true[idx]
        Ahat = amp_hat[idx]
        ax.plot(ts_np, A, lw=2, label="Simulation")
        ax.plot(ts_np, Ahat, "--", lw=2, label="Latent ODE")
        ax.set_xlabel(r"$t$")
        if j == 0:
            ax.set_ylabel(r"$A(t)$")
        ax.set_title(f"Case {idx}")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(frameon=False)

    fig.tight_layout()
    out = os.path.join(ml_dir, "tearing_ml_v2_latent_recon_examples.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out}")


def plot_latent_error_stats(
    ml_dir: str,
    ts: np.ndarray,
    amp_true: np.ndarray,
    amp_hat: np.ndarray,
    latent_model,
) -> None:
    """Dataset-level reconstruction error statistics for the latent ODE."""

    logA_true = safe_log(jnp.asarray(amp_true))
    logA_hat = safe_log(jnp.asarray(amp_hat))
    logA_true_n = (logA_true - latent_model.logA_mean) / latent_model.logA_std
    logA_hat_n = (logA_hat - latent_model.logA_mean) / latent_model.logA_std

    mse_per_case = np.array(
        jnp.mean((logA_hat_n - logA_true_n) ** 2, axis=1)
    )  # (N,)
    rel_err_amp = np.array(
        (amp_hat - amp_true) / (amp_true + 1e-16)
    )  # (N, T)

    # Time-averaged relative error magnitude per case
    rel_err_mean = np.mean(np.abs(rel_err_amp), axis=1)  # (N,)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.bar(np.arange(len(mse_per_case)), mse_per_case, alpha=0.8)
    ax.set_xlabel("Case index")
    ax.set_ylabel(r"MSE in normalized $\log A$")
    ax.set_title("Per-case reconstruction error (latent ODE)")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(rel_err_mean, bins=15, edgecolor="k", alpha=0.85)
    ax2.set_xlabel(r"Mean $|A_{\mathrm{LODE}}-A_{\mathrm{sim}}|/A_{\mathrm{sim}}$")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of time-averaged relative error")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(ml_dir, "tearing_ml_v2_latent_recon_error_stats.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out}")


def plot_latent_phase_portrait(
    ml_dir: str,
    ts: np.ndarray,
    amp_true: np.ndarray,
    eq_params: np.ndarray,
    latent_model,
    indices: List[int],
) -> None:
    """
    Latent phase portraits for representative cases, showing z_1 vs z_2 and
    z_i(t). We rely directly on latent_ode_forward_single to obtain both the
    reconstructed log-amplitude and the full latent trajectory.
    """
    ts_j = jnp.asarray(ts)

    fig = plt.figure(figsize=(11, 4))

    # z1 vs z2
    ax1 = fig.add_subplot(1, 2, 1)
    for idx in indices:
        amp_j = jnp.asarray(amp_true[idx])
        eq_j = jnp.asarray(eq_params[idx])
        _, z_traj_j = latent_ode_forward_single(latent_model, ts_j, amp_j, eq_j)
        z = np.array(z_traj_j)
        if z.ndim == 2 and z.shape[1] >= 2:
            ax1.plot(z[:, 0], z[:, 1], lw=1.8, label=f"Case {idx}")
    ax1.set_xlabel(r"$z_1$")
    ax1.set_ylabel(r"$z_2$")
    ax1.set_title("Latent trajectories (first two coordinates)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=False)

    # z_i(t)
    ax2 = fig.add_subplot(1, 2, 2)
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for j, idx in enumerate(indices):
        amp_j = jnp.asarray(amp_true[idx])
        eq_j = jnp.asarray(eq_params[idx])
        _, z_traj_j = latent_ode_forward_single(latent_model, ts_j, amp_j, eq_j)
        z = np.array(z_traj_j)
        if z.ndim != 2:
            continue
        for d in range(min(z.shape[1], 3)):
            label = None
            if j == 0:
                label = f"Case {idx}, $z_{d+1}$"
            ax2.plot(
                ts,
                z[:, d],
                lw=1.5,
                color=colors[d % len(colors)],
                alpha=0.8,
                label=label,
            )

    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$z_i(t)$")
    ax2.set_title("Latent coordinates vs time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", frameon=False)

    fig.tight_layout()
    out = os.path.join(ml_dir, "tearing_ml_v2_latent_phase.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out}")


def plot_prl_multi_panel(
    ml_dir: str,
    ts: np.ndarray,
    gamma_true: np.ndarray,
    gamma_pred: np.ndarray,
    Asat_true: np.ndarray,
    Asat_pred: np.ndarray,
    S_arr: np.ndarray,
    amp_true: np.ndarray,
    amp_hat: np.ndarray,
    eq_params: np.ndarray,
    latent_model,
    indices: List[int],
    gamma_th: np.ndarray,
    seed: int = 0,
) -> None:
    """
    Single PRL-style figure summarizing the whole story:

      (a)  γ surrogate parity (train vs val, R^2)
      (b)  A_sat surrogate parity (train vs val, R^2)
      (c)  γ vs S with Sweet–Parker reference ~ S^{-1/2}
      (d)  Surrogate γ relative-error histogram
      (e)  Latent-ODE reconstructions A(t) for a few cases
      (f)  Latent phase portrait z1 vs z2 for the same cases
    """
    N = gamma_true.shape[0]
    rng = np.random.default_rng(seed)
    idx_all = rng.permutation(N)
    n_train = max(2, int(0.75 * N))
    train_idx = idx_all[:n_train]
    val_idx = idx_all[n_train:]

    g_tr, g_v = gamma_true[train_idx], gamma_true[val_idx]
    gptr, gpv = gamma_pred[train_idx], gamma_pred[val_idx]
    A_tr, A_v = Asat_true[train_idx], Asat_true[val_idx]
    Aptr, Apv = Asat_pred[train_idx], Asat_pred[val_idx]

    r2_gamma = compute_r2(gamma_true, gamma_pred)
    r2_Asat = compute_r2(Asat_true, Asat_pred)

    # Time grid for later panels
    ts_np = np.array(ts)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.flatten()

    # --- (a) γ parity ---------------------------------------------------------
    ax = ax_a
    ax.scatter(g_tr, gptr, marker="o", label="Train", alpha=0.9)
    ax.scatter(g_v, gpv, marker="s", label="Val", alpha=0.9)
    gmin, gmax = np.min(gamma_true), np.max(gamma_true)
    pad = 0.05 * (gmax - gmin + 1e-8)
    ax.plot(
        [gmin - pad, gmax + pad],
        [gmin - pad, gmax + pad],
        "k--",
        lw=1,
        label="Ideal",
    )
    ax.set_xlabel(r"$\gamma_{\mathrm{fit}}$ (sim.)")
    ax.set_ylabel(r"$\gamma_{\mathrm{pred}}$ (sur.)")
    ax.set_title(r"(a) Growth rate surrogate, $R^2 \approx %.3f$" % r2_gamma)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper left")

    # --- (b) A_sat parity -----------------------------------------------------
    ax = ax_b
    ax.scatter(A_tr, Aptr, marker="o", label="Train", alpha=0.9)
    ax.scatter(A_v, Apv, marker="s", label="Val", alpha=0.9)
    Amin, Amax = np.min(Asat_true), np.max(Asat_true)
    padA = 0.05 * (Amax - Amin + 1e-8)
    ax.plot(
        [Amin - padA, Amax + padA],
        [Amin - padA, Amax + padA],
        "k--",
        lw=1,
        label="Ideal",
    )
    ax.set_xlabel(r"$A_{\mathrm{sat}}$ (sim.)")
    ax.set_ylabel(r"$A_{\mathrm{sat,pred}}$ (sur.)")
    ax.set_title(r"(b) Saturated amplitude surrogate, $R^2 \approx %.3f$" % r2_Asat)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper left")

    # --- (c) γ vs γ_FKR (theory) --------------------------------------------
    mask_th = np.isfinite(gamma_th) & (gamma_th > 0.0)
    g_th = gamma_th[mask_th]
    g_fit_plot = gamma_true[mask_th]
    g_pred_plot = gamma_pred[mask_th]

    ax = ax_c
    ax.scatter(g_th, g_fit_plot, s=28, marker="o",
               label=r"Sim. $\gamma_{\mathrm{fit}}$")
    ax.scatter(g_th, g_pred_plot, s=28, marker="s",
               label=r"Sur. $\gamma_{\mathrm{pred}}$")

    gmin = np.min(g_th)
    gmax = np.max(g_th)
    pad = 0.05 * (gmax - gmin + 1e-8)
    ax.plot(
        [gmin - pad, gmax + pad],
        [gmin - pad, gmax + pad],
        "k--",
        lw=1.1,
        label="Ideal",
    )

    ax.set_xlabel(r"$\gamma_{\mathrm{FKR}}$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("(c) Linear tearing physics across the scan")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper left")

    # --- (d) surrogate γ error histogram -------------------------------------
    rel_err_gamma = (gamma_pred - gamma_true) / (gamma_true + 1e-16)
    ax = ax_d
    ax.hist(rel_err_gamma, bins=12, edgecolor="k", alpha=0.85)
    ax.set_xlabel(
        r"Rel. error in $\gamma$ "
        r"$\left(\gamma_{\mathrm{pred}}-\gamma_{\mathrm{fit}}\right)"
        r"/\gamma_{\mathrm{fit}}$"
    )
    ax.set_ylabel("Count")
    ax.set_title("(d) Surrogate growth-rate errors")
    ax.grid(True, alpha=0.3)

    # --- (e) latent reconstructions A(t) --------------------------------------
    ax = ax_e
    for j, idx in enumerate(indices):
        A = amp_true[idx]
        Ahat = amp_hat[idx]
        if j == 0:
            ax.plot(ts_np, A, lw=2, label="Sim.")
            ax.plot(ts_np, Ahat, "--", lw=2, label="Latent ODE")
        else:
            ax.plot(ts_np, A, lw=1.5, alpha=0.8)
            ax.plot(ts_np, Ahat, "--", lw=1.5, alpha=0.8)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$A(t)$")
    ax.set_title("(e) Latent-ODE reconstructions")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower right")

    # --- (f) latent phase portrait z1 vs z2 ----------------------------------
    ts_j = jnp.asarray(ts)
    ax = ax_f
    for idx in indices:
        amp_j = jnp.asarray(amp_true[idx])
        eq_j = jnp.asarray(eq_params[idx])
        _, z_traj_j = latent_ode_forward_single(latent_model, ts_j, amp_j, eq_j)
        z = np.array(z_traj_j)
        if z.ndim == 2 and z.shape[1] >= 2:
            ax.plot(z[:, 0], z[:, 1], lw=1.7, label=f"Case {idx}")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_title("(f) Latent trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    out = os.path.join(ml_dir, "tearing_ml_v2_prl_figure.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out}")


# -----------------------------------------------------------------------------#
# Main driver
# -----------------------------------------------------------------------------#


def main():
    args = parse_args()
    ml_dir = args.ml_dir
    runs_subdir = args.runs_subdir
    n_examples = args.n_examples
    seed = args.seed

    print("=== v2 ML postprocessing for Harris-sheet tearing ===")
    print(f"ML directory   : {ml_dir}")
    print(f"Runs subfolder : {runs_subdir}")
    print("=====================================================\n")

    # 1) Rebuild dataset from saved MHD runs
    dataset = load_dataset_from_runs(ml_dir, runs_subdir, verbose=True)
    ts = dataset["ts"]
    amp = dataset["amp"]
    gamma_fit = dataset["gamma_fit"]
    A_sat = dataset["A_sat"]
    gamma_th = dataset["gamma_th"]  # currently unused, but kept for reference
    S_arr = dataset["S"]
    eq_params = dataset["eq_params"]

    N, T = amp.shape

    # 2) Load trained models
    surrogate_path = os.path.join(ml_dir, "surrogate_model.pkl")
    latent_path = os.path.join(ml_dir, "latent_ode_model.pkl")

    surrogate = load_surrogate_model(surrogate_path)
    latent_model = load_latent_ode_model(latent_path)

    print(f"\n[IO] Loaded surrogate from {surrogate_path}")
    print(f"[IO] Loaded latent ODE model from {latent_path}\n")

    # 3) Surrogate predictions for all cases
    eq_params_j = jnp.asarray(eq_params)
    gamma_pred_j, Asat_pred_j = surrogate_predict(surrogate, eq_params_j)
    gamma_pred = np.array(gamma_pred_j)
    Asat_pred = np.array(Asat_pred_j)

    print("[SURROGATE] Global diagnostics:")
    print(f"  γ_fit:   mean={gamma_fit.mean():.3e}, std={gamma_fit.std():.3e}")
    print(f"  γ_pred:  mean={gamma_pred.mean():.3e}, std={gamma_pred.std():.3e}")
    print(f"  A_sat:   mean={A_sat.mean():.3f},  std={A_sat.std():.3f}")
    print(f"  A_sat,p: mean={Asat_pred.mean():.3f},  std={Asat_pred.std():.3f}\n")

    # 4) Latent ODE reconstructions for all cases
    ts_j = jnp.asarray(ts)
    amp_hat_all = []
    for i in range(N):
        amp_i = jnp.asarray(amp[i])
        eq_i = jnp.asarray(eq_params[i])
        logA_hat_i, _ = latent_ode_forward_single(latent_model, ts_j, amp_i, eq_i)
        amp_hat_i = jnp.exp(logA_hat_i)
        amp_hat_all.append(np.array(amp_hat_i))
    amp_hat = np.stack(amp_hat_all, axis=0)

    # 5) Choose representative cases for visualization
    rng = np.random.default_rng(seed)
    order = np.argsort(gamma_fit)
    indices: List[int] = []
    if n_examples >= 1:
        indices.append(int(order[0]))              # slowest mode
    if n_examples >= 2:
        indices.append(int(order[len(order)//2]))  # intermediate
    if n_examples >= 3:
        indices.append(int(order[-1]))             # fastest / most nonlinear
    while len(indices) < n_examples:
        extra = int(rng.integers(0, N))
        if extra not in indices:
            indices.append(extra)

    print(f"[INFO] Representative example indices for A(t) plots: {indices}\n")

    # 6) Generate figures
    plot_surrogate_parity_and_errors(
        ml_dir, gamma_fit, gamma_pred, A_sat, Asat_pred, seed=seed
    )
    plot_gamma_vs_S(ml_dir, S_arr, gamma_fit, gamma_pred)
    plot_latent_recon_examples(ml_dir, ts, amp, amp_hat, indices)
    plot_latent_error_stats(ml_dir, ts, amp, amp_hat, latent_model)
    plot_latent_phase_portrait(ml_dir, ts, amp, eq_params, latent_model, indices)
    # Combined PRL-style multipanel summary figure
    plot_prl_multi_panel(ml_dir, ts,
        gamma_fit, gamma_pred, A_sat, Asat_pred, S_arr, amp,
        amp_hat, eq_params, latent_model, indices, gamma_th, seed=seed,)

    print("\n[DONE] v2 ML postprocessing complete.")
    print("Figures written to:")
    print(f"  {os.path.join(ml_dir, 'tearing_ml_v2_surrogate_parity.png')}")
    print(f"  {os.path.join(ml_dir, 'tearing_ml_v2_surrogate_error_hist.png')}")
    print(f"  {os.path.join(ml_dir, 'tearing_ml_v2_gamma_vs_S.png')}")
    print(f"  {os.path.join(ml_dir, 'tearing_ml_v2_latent_recon_examples.png')}")
    print(f"  {os.path.join(ml_dir, 'tearing_ml_v2_latent_recon_error_stats.png')}")
    print(f"  {os.path.join(ml_dir, 'tearing_ml_v2_latent_phase.png')}")
    print(f"  {os.path.join(ml_dir, 'tearing_ml_v2_prl_figure.png')}")

if __name__ == "__main__":
    main()
