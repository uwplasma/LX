#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_scan.py

"Loureiro-style" scan driver for Harris-sheet tearing.

This script does *both*:
  1) runs the incompressible pseudo-spectral MHD tearing simulation
     for a scan over (a, eta) by calling
        mhd_tearing_solve.solve_tearing_case,
     separately for each equilibrium branch:
        - "original"  : classic Harris-sheet / Sweet–Parker-like
        - "forcefree" : equilibrium-subtracted, FKR/plasmoid-like
  2) postprocesses each run to extract:
       - island width w(t),
       - linear growth rate γ_fit,
       - Rutherford slope (dw/dt)_R,
       - saturated width w_sat,
     and then builds Loureiro-style scan plots.

For each branch `equilibrium_mode` we create:
    outdir/<equilibrium_mode>/mhd_tearing_solution_*.npz
    outdir/<equilibrium_mode>/tearing_profile_*.png
    outdir/<equilibrium_mode>/tearing_nonlinear_*.png
    outdir/<equilibrium_mode>/scan_*.png
    outdir/<equilibrium_mode>/tearing_scan_summary_<equilibrium_mode>.npz

Typical usage
-------------

python mhd_tearing_scan.py \
    --outdir tearing_scan_plots

You can still override grid / box / viscosity / guide field / etc
from the command line, but the (a, eta), epsB and time windows are
chosen branch-by-branch inside the script for publication-ready
comparisons.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import List, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mhd_tearing_solve import solve_tearing_case, make_k_arrays

# -----------------------------------------------------------------------------#
# Matplotlib style (publication-ready)
# -----------------------------------------------------------------------------#

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.figsize": (5.5, 4.5),
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
})

# -----------------------------------------------------------------------------#
# Branch-specific scan configuration
# -----------------------------------------------------------------------------#
# You can tweak these lists if you want different coverage.
#
# "original"  branch: Sweet–Parker-like regime
# "forcefree" branch: FKR/plasmoid regime (equilibrium subtraction)
#
# Both branches use the same (a, eta) grid here so you can compare
# 1:1, but they differ in epsB and in the time window (t1, n_frames).

BRANCH_CONFIG = {
    "original": {
        # Sweet–Parker-like: moderate Lundquist numbers, larger seed
        "scan_a":  [0.5, 0.6, 0.785],          # gives Δ'a ~ O(1)
        "scan_eta": [1.25e-3, 6.25e-4, 3.125e-4],  # S ~ 10^3–10^4
        "epsB": 1e-2,
        "t1": 100.0,
        "n_frames": 200,
    },
    "forcefree": {
        # FKR/plasmoid-like: same (a, eta) but tiny seed for a long
        # clean linear stage and plasmoid-friendly evolution.
        "scan_a":  [0.5, 0.6, 0.785],
        "scan_eta": [1.25e-3, 6.25e-4, 3.125e-4],
        "epsB": 1e-5,
        "t1": 80.0,
        "n_frames": 220,
    },
}

# -----------------------------------------------------------------------------#
# Post-processing utilities (NumPy only)
# -----------------------------------------------------------------------------#

def compute_k_arrays_np(Nx, Ny, Nz, Lx, Ly, Lz):
    """
    Wrapper around make_k_arrays (from mhd_tearing_solve.py) but returning NumPy arrays.
    """
    kx_j, ky_j, kz_j, k2_j, NX_j, NY_j, NZ_j = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    return (np.array(kx_j),
            np.array(ky_j),
            np.array(kz_j),
            np.array(NX_j),
            np.array(NY_j),
            np.array(NZ_j))


def compute_Az_hat(B_hat, kx, ky):
    """
    Compute A_z(k) from B_hat(k):

        (B_x, B_y) = (-∂A_z/∂y, ∂A_z/∂x)
        => A_z_hat = i (k_x B_y_hat - k_y B_x_hat) / k_perp^2

    B_hat: (3,Nx,Ny,Nz) complex
    """
    Bx_hat, By_hat = B_hat[0], B_hat[1]
    k_perp2 = kx**2 + ky**2
    k_perp2_safe = np.where(k_perp2 == 0.0, 1.0, k_perp2)

    Az_hat = 1j * (kx * By_hat - ky * Bx_hat) / k_perp2_safe
    Az_hat = np.where(k_perp2 == 0.0, 0.0, Az_hat)
    return Az_hat


def compute_island_width_from_mode(Az_hat_mode, B0, a):
    """
    Island half-width proxy from tearing-mode amplitude:

        w ≈ 4 √(|Ã_1| / |B'_y(x_s)|),   B'_y(x_s=Lx/2) = B0/a  (Harris).

    Az_hat_mode: complex
    """
    A_amp = np.abs(Az_hat_mode)
    Bprime = B0 / a
    if Bprime <= 0.0:
        return np.nan
    return 4.0 * np.sqrt(A_amp / Bprime)


def energy_from_hat_np(v_hat, B_hat, Lx, Ly, Lz):
    """NumPy version for post-processing."""
    v = np.fft.ifftn(v_hat, axes=(1, 2, 3)).real
    B = np.fft.ifftn(B_hat, axes=(1, 2, 3)).real
    dv = (Lx * Ly * Lz) / (v[0].size)

    v2 = np.sum(v * v, axis=0)
    B2 = np.sum(B * B, axis=0)
    E_kin = 0.5 * np.sum(v2) * dv
    E_mag = 0.5 * np.sum(B2) * dv
    return float(E_kin), float(E_mag)


def _linear_regression_with_stats(x, y):
    """
    Simple y = a x + b regression with R^2 and standard error of slope.
    Returns (a, b, R2, a_err).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    N = x.size
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    resid = y - y_pred
    RSS = np.sum(resid**2)
    TSS = np.sum((y - np.mean(y))**2)
    R2 = 1.0 - RSS / TSS if TSS > 0 else np.nan
    if N > 2:
        sigma2 = RSS / (N - 2)
        x_var = np.sum((x - np.mean(x))**2)
        a_err = np.sqrt(sigma2 / x_var) if x_var > 0 else np.nan
    else:
        a_err = np.nan
    return a, b, R2, a_err


# -----------------------------------------------------------------------------#
# Robust automatic linear-window selector
# -----------------------------------------------------------------------------#

def select_linear_window(
    ts,
    w,
    w0=None,
    min_pts: int = 6,
    frac_sat: float = 0.13,
    nwin: int = 8,
    curv_tol: float = 0.02,
    R2_min: float = 0.97,
):
    """
    Data-driven selector for the exponentially growing window of ln(w).

    Strategy:
      * Avoid early transient: require w > 1.2 w0.
      * Avoid near-saturation: require w < w0 + frac_sat*(w_max - w0).
      * Require small curvature in ln(w): |d²/dt² ln(w)| < curv_tol.
      * Slide a window of size nwin over candidate points.
      * Among windows with R² >= R2_min, choose the one with largest R².
        If none passes R2_min, choose the window with maximum R².

    Returns
    -------
    mask_lin : boolean array
        True for points in the selected linear window.
    """
    ts = np.asarray(ts)
    w = np.asarray(w)

    if w0 is None:
        w0 = w[0]

    wmax = np.nanmax(w)
    upper = w0 + frac_sat * (wmax - w0)

    lnw = np.log(w)
    d1 = np.gradient(lnw, ts)
    d2 = np.gradient(d1, ts)

    mask = (
        (w > 1.2 * w0) &
        (w < upper) &
        np.isfinite(lnw) &
        (np.abs(d2) < curv_tol)
    )
    idx = np.where(mask)[0]

    if idx.size < max(min_pts, 3):
        # fallback: earliest quarter of the simulation
        mask_fb = ts < (ts[0] + 0.25 * (ts[-1] - ts[0]))
        return mask_fb

    nwin = min(nwin, idx.size)
    best_slice = None
    best_R2 = -np.inf

    # First pass: enforce R² >= R2_min if possible
    for s in range(0, idx.size - nwin + 1):
        win = idx[s:s+nwin]
        t_win = ts[win]
        y_win = lnw[win]
        a, b, R2, _ = _linear_regression_with_stats(t_win, y_win)
        if R2 >= R2_min and R2 > best_R2:
            best_R2 = R2
            best_slice = win

    # Second pass: if nothing reached R2_min, just pick max R²
    if best_slice is None:
        for s in range(0, idx.size - nwin + 1):
            win = idx[s:s+nwin]
            t_win = ts[win]
            y_win = lnw[win]
            a, b, R2, _ = _linear_regression_with_stats(t_win, y_win)
            if R2 > best_R2:
                best_R2 = R2
                best_slice = win

    mask_lin = np.zeros_like(w, dtype=bool)
    mask_lin[best_slice] = True
    return mask_lin


# -----------------------------------------------------------------------------#
# Scan analysis for a single run
# -----------------------------------------------------------------------------#

def analyze_single_run(
    fname: str,
    lin_tmin: float | None,
    lin_tmax: float | None,
    ruth_frac: Tuple[float, float],
) -> dict:
    """
    Load one NPZ file and extract w(t), γ_fit, (dw/dt)_R, w_sat, etc.
    Returns a dict with diagnostics and parameters.
    """
    print(f"\n[INFO] === Analyzing {fname} ===")
    data = np.load(fname, allow_pickle=True)

    ts = data["ts"]
    v_hat_frames = data["v_hat"]
    B_hat_frames = data["B_hat"]

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
    eps_B = float(data["eps_B"])
    gamma_FKR = float(data["gamma_FKR"])
    S = float(data["S"])
    Delta_prime_a = float(data["Delta_prime_a"])
    ix0 = int(data["ix0"])
    iy1 = int(data["iy1"])
    iz0 = int(data["iz0"])
    eq_mode = str(data["equilibrium_mode"])

    Delta_prime = Delta_prime_a / a
    etaDelta = eta * Delta_prime

    print(f"[RUN] Nx={Nx}, Ny={Ny}, Nz={Nz}, Lx={Lx:.3f}, Ly={Ly:.3f}, Lz={Lz:.3f}")
    print(f"[RUN] nu={nu:.3e}, eta={eta:.3e}, B0={B0:.3e}, a={a:.3e}, eps_B={eps_B:.3e}")
    print(f"[RUN] S={S:.3e}, Delta'*a={Delta_prime_a:.3e}, Delta'={Delta_prime:.3e}")
    print(f"[RUN] γ_FKR={gamma_FKR:.3e}, mode indices (ix0,iy1,iz0)=({ix0},{iy1},{iz0})")
    print(f"[RUN] equilibrium_mode = {eq_mode}")

    # k arrays (NumPy)
    kx, ky, kz, NX, NY, NZ = compute_k_arrays_np(Nx, Ny, Nz, Lx, Ly, Lz)
    ky_val = ky[ix0, iy1, iz0]
    print(f"[DEBUG] ky for tearing mode = {ky_val:.6f}")

    n_t = ts.size
    island_width = np.zeros(n_t)
    Az_amp = np.zeros(n_t)
    E_kin_arr = np.zeros(n_t)
    E_mag_arr = np.zeros(n_t)

    for it in range(n_t):
        B_hat = B_hat_frames[it]
        Az_hat = compute_Az_hat(B_hat, kx, ky)
        A_mode = Az_hat[ix0, iy1, iz0]
        Az_amp[it] = np.abs(A_mode)
        island_width[it] = compute_island_width_from_mode(A_mode, B0, a)

        v_hat = v_hat_frames[it]
        Ek, Em = energy_from_hat_np(v_hat, B_hat, Lx, Ly, Lz)
        E_kin_arr[it] = Ek
        E_mag_arr[it] = Em

    w0 = island_width[0]
    wmax = np.nanmax(island_width)
    print(f"[INFO] w0 = {w0:.3e}, w_max = {wmax:.3e}")

    # ----- Linear fit: automatic window selector ----- #
    mask_lin = select_linear_window(
        ts,
        island_width,
        w0=w0,
        min_pts=6,
        frac_sat=0.13,
        nwin=8,
        curv_tol=0.02,
        R2_min=0.97,
    )

    if np.count_nonzero(mask_lin) < 5:
        print("[WARN] Automatic selector failed, falling back to first 25% of time.")
        mask_lin = ts < (ts[0] + 0.25 * (ts[-1] - ts[0]))

    t_lin = ts[mask_lin]
    w_lin = island_width[mask_lin]
    lnw_lin = np.log(w_lin)

    a_lin, b_lin, gamma_R2, gamma_fit_err = _linear_regression_with_stats(t_lin, lnw_lin)
    gamma_fit = a_lin
    print(f"[INFO] Linear window: t = [{t_lin[0]:.3f}, {t_lin[-1]:.3f}], "
          f"{len(t_lin)} points")
    print(f"[RESULT] γ_fit = {gamma_fit:.3e},  γ_fit/γ_FKR = {gamma_fit/gamma_FKR:.3f}, "
          f"R²_lin = {gamma_R2:.3f}, σ_γ = {gamma_fit_err:.3e}")

    # ----- Rutherford slope: w(t) ~ w0_R + (dw/dt)_R t ----- #
    f_low, f_high = ruth_frac
    T = ts[-1] - ts[0]
    t_lin_end = t_lin[-1]
    t_start_nom = ts[0] + f_low * T
    t_end_nom = ts[0] + f_high * T
    # Enforce start > linear end + small margin
    t_start = max(t_start_nom, t_lin_end + 0.05 * T)
    t_end = max(t_end_nom, t_start + 0.05 * T)

    mask_ruth = (ts >= t_start) & (ts <= t_end)

    if np.count_nonzero(mask_ruth) < 5:
        print("[WARN] Too few points for Rutherford fit; "
              "using last half of time as fallback.")
        mask_ruth = ts >= (ts[0] + 0.5 * (ts[-1] - ts[0]))

    t_ruth = ts[mask_ruth]
    w_ruth = island_width[mask_ruth]

    while w_ruth.size >= 5 and np.any(np.diff(w_ruth) <= 0):
        t_ruth = t_ruth[1:]
        w_ruth = w_ruth[1:]

    if w_ruth.size < 5:
        print("[WARN] No clean monotonic Rutherford regime found.")
        dw_dt_R = np.nan
        b_R = np.nan
        dw_dt_R_R2 = np.nan
        dw_dt_R_err = np.nan
        t_Ruth_start = np.nan
        t_Ruth_end = np.nan
    else:
        t_Ruth_start = t_ruth[0]
        t_Ruth_end = t_ruth[-1]
        dw_dt_R, b_R, dw_dt_R_R2, dw_dt_R_err = _linear_regression_with_stats(
            t_ruth, w_ruth
        )
        print(f"[RESULT] (dw/dt)_R = {dw_dt_R:.3e}, R²_R = {dw_dt_R_R2:.3f}, "
              f"σ_{{dw/dt}} = {dw_dt_R_err:.3e}")
        if dw_dt_R_R2 < 0.8 or dw_dt_R <= 0:
            print("[WARN] Rutherford fit has low R² or non-positive slope; "
                  "marking as invalid for global scaling.")

    # ----- Saturated island width (last 20% of time) ----- #
    t_sat_min = ts[0] + 0.8 * (ts[-1] - ts[0])
    mask_sat = ts >= t_sat_min
    w_sat_samples = island_width[mask_sat]
    w_sat = float(np.mean(w_sat_samples))
    w_sat_std = float(np.std(w_sat_samples))
    print(f"[RESULT] w_sat = {w_sat:.3e} ± {w_sat_std:.3e}")

    # ----- Derived nonlinear diagnostics ----- #
    w_over_a = island_width / a
    lnw_over_a = np.log(w_over_a)
    gamma_inst = np.gradient(lnw_over_a, ts)
    psi_rec = 2.0 * Az_amp  # proxy for reconnected flux

    t_lin_start = t_lin[0]

    # ----- Per-run diagnostic plot: w/a and ln(w/a) with fit ----- #
    outdir = os.path.dirname(fname)
    w_lin_over_a = w_lin / a
    lnw_lin_over_a = np.log(w_lin_over_a)

    t_fit_line = np.linspace(t_lin[0], t_lin[-1], 200)
    lnw_fit_line = a_lin * t_fit_line + b_lin
    w_fit_line_over_a = np.exp(lnw_fit_line) / a

    fig_diag, axes = plt.subplots(2, 1, sharex=True, figsize=(5.5, 6.0))

    ax_top = axes[0]
    ax_top.plot(ts, w_over_a, "-", label=r"$w/a$")
    ax_top.plot(ts[mask_lin], w_over_a[mask_lin], "o", ms=4,
                label=r"linear window")
    ax_top.plot(t_fit_line, w_fit_line_over_a, "--", label=r"exp fit")
    ax_top.set_ylabel(r"$w/a$")
    ax_top.set_title(r"Island width evolution")
    ax_top.grid(True, ls=":")
    ax_top.legend(loc="best")

    ax_bottom = axes[1]
    ax_bottom.plot(ts, lnw_over_a, "-", label=r"$\ln(w/a)$")
    ax_bottom.plot(t_lin, lnw_lin_over_a, "o", ms=4, label=r"fit points")
    ax_bottom.plot(t_fit_line, lnw_fit_line, "--",
                   label=rf"fit: $\gamma={gamma_fit:.3e}$")
    ax_bottom.set_xlabel(r"$t$")
    ax_bottom.set_ylabel(r"$\ln(w/a)$")
    ax_bottom.grid(True, ls=":")
    ax_bottom.legend(loc="best")

    diag_name = os.path.join(
        outdir,
        "tearing_profile_" + os.path.basename(fname).replace(".npz", ".png"),
    )
    fig_diag.savefig(diag_name)
    plt.close(fig_diag)
    print(f"[SAVE] {diag_name}")

    # ----- Nonlinear diagnostics figure ----- #
    fig_nl, axes_nl = plt.subplots(3, 1, sharex=True, figsize=(5.5, 7.0))

    ax1 = axes_nl[0]
    ax1.plot(ts, gamma_inst, "-")
    ax1.axhline(gamma_fit, ls="--", lw=1.0, label=r"$\gamma_{\rm fit}$")
    ax1.set_ylabel(r"$\gamma_{\rm inst}(t)$")
    ax1.set_title(r"Instantaneous growth rate and energies")
    ax1.grid(True, ls=":")

    ax2 = axes_nl[1]
    ax2.semilogy(ts, E_kin_arr, "-", label=r"$E_{\rm kin}$")
    ax2.semilogy(ts, E_mag_arr, "-", label=r"$E_{\rm mag}$")
    ax2.set_ylabel(r"Energies")
    ax2.grid(True, ls=":")
    ax2.legend(loc="best")

    ax3 = axes_nl[2]
    ax3.plot(ts, psi_rec, "-")
    ax3.set_xlabel(r"$t$")
    ax3.set_ylabel(r"$\psi_{\rm rec} \propto |A_1|$")
    ax3.grid(True, ls=":")

    used_labels = set()

    def _add_marker(ax, t, label, style="--"):
        if not np.isfinite(t):
            return
        lab = label if label not in used_labels else None
        ax.axvline(t, ls=style, lw=0.8, color="k", alpha=0.7, label=lab)
        if lab is not None:
            used_labels.add(label)

    for ax in axes_nl:
        _add_marker(ax, t_lin_start, "linear start", "--")
        _add_marker(ax, t_lin_end, "linear end", "-.")
        _add_marker(ax, t_Ruth_start, "Rutherford start", "--")
        _add_marker(ax, t_Ruth_end, "Rutherford end", "-.")
        _add_marker(ax, t_sat_min, "saturation start", ":")

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, loc="best")

    nl_name = os.path.join(
        outdir,
        "tearing_nonlinear_" + os.path.basename(fname).replace(".npz", ".png"),
    )
    fig_nl.savefig(nl_name)
    plt.close(fig_nl)
    print(f"[SAVE] {nl_name}")

    return {
        "fname": os.path.basename(fname),
        "eta": eta,
        "B0": B0,
        "a": a,
        "S": S,
        "Delta_prime_a": Delta_prime_a,
        "Delta_prime": Delta_prime,
        "etaDelta": etaDelta,
        "gamma_FKR": gamma_FKR,
        "gamma_fit": gamma_fit,
        "gamma_fit_err": gamma_fit_err,
        "gamma_R2": gamma_R2,
        "dw_dt_R": dw_dt_R,
        "dw_dt_R_err": dw_dt_R_err,
        "dw_dt_R_R2": dw_dt_R_R2,
        "w_sat": w_sat,
        "w_sat_std": w_sat_std,
        "mask_lin": mask_lin,
        "equilibrium_mode": eq_mode,
    }


# -----------------------------------------------------------------------------#
# Summary + Loureiro-style plots for one branch
# -----------------------------------------------------------------------------#

def build_summary_and_plots(results, outdir: str, eq_mode: str):
    mode_str = "force-free" if eq_mode == "forcefree" else "original"

    fnames = np.array([r["fname"] for r in results], dtype=object)
    eta_arr = np.array([r["eta"] for r in results])
    a_arr = np.array([r["a"] for r in results])
    S_arr = np.array([r["S"] for r in results])
    Delta_prime_a_arr = np.array([r["Delta_prime_a"] for r in results])
    Delta_prime_arr = np.array([r["Delta_prime"] for r in results])
    etaDelta_arr = np.array([r["etaDelta"] for r in results])
    gamma_FKR_arr = np.array([r["gamma_FKR"] for r in results])
    gamma_fit_arr = np.array([r["gamma_fit"] for r in results])
    gamma_fit_err_arr = np.array([r["gamma_fit_err"] for r in results])
    gamma_R2_arr = np.array([r["gamma_R2"] for r in results])
    dw_dt_R_arr = np.array([r["dw_dt_R"] for r in results])
    dw_dt_R_err_arr = np.array([r["dw_dt_R_err"] for r in results])
    dw_dt_R_R2_arr = np.array([r["dw_dt_R_R2"] for r in results])
    w_sat_arr = np.array([r["w_sat"] for r in results])
    w_sat_std_arr = np.array([r["w_sat_std"] for r in results])

    w_sat_over_a_arr = w_sat_arr / a_arr
    w_sat_over_a_std_arr = w_sat_std_arr / a_arr

    summary_path = os.path.join(outdir, f"tearing_scan_summary_{eq_mode}.npz")
    np.savez(
        summary_path,
        fnames=fnames,
        eta=eta_arr,
        a=a_arr,
        S=S_arr,
        Delta_prime_a=Delta_prime_a_arr,
        Delta_prime=Delta_prime_arr,
        etaDelta=etaDelta_arr,
        gamma_FKR=gamma_FKR_arr,
        gamma_fit=gamma_fit_arr,
        gamma_fit_err=gamma_fit_err_arr,
        gamma_R2=gamma_R2_arr,
        dw_dt_R=dw_dt_R_arr,
        dw_dt_R_err=dw_dt_R_err_arr,
        dw_dt_R_R2=dw_dt_R_R2_arr,
        w_sat=w_sat_arr,
        w_sat_std=w_sat_std_arr,
        w_sat_over_a=w_sat_over_a_arr,
        w_sat_over_a_std=w_sat_over_a_std_arr,
        equilibrium_mode=eq_mode,
    )
    print(f"\n[SAVE] Summary saved to {summary_path}")

    # ------------------------------------------------------------------ #
    # Plot 1: γ_fit vs γ_FKR (with error bars)
    # ------------------------------------------------------------------ #
    fig1, ax1 = plt.subplots()
    ax1.errorbar(gamma_FKR_arr, gamma_fit_arr, yerr=gamma_fit_err_arr,
                 fmt="o", capsize=3, label=r"runs")

    if eq_mode == "forcefree":
        gmin = 0.5 * np.min(gamma_FKR_arr)
        gmax = 2.0 * np.max(gamma_FKR_arr)
        ref = np.linspace(gmin, gmax, 100)
        ax1.loglog(ref, ref, "--", color="0.4", lw=1.5,
                   label=r"$\gamma_{\rm fit}=\gamma_{\rm FKR}$")

    for i, name in enumerate(fnames):
        ax1.annotate(
            str(i),
            (gamma_FKR_arr[i], gamma_fit_arr[i]),
            textcoords="offset points",
            xytext=(4, 2),
            fontsize=8,
        )

    ax1.set_xlabel(r"$\gamma_{\rm FKR}$")
    ax1.set_ylabel(r"$\gamma_{\rm fit}$")
    ax1.set_title(rf"Linear growth: $\gamma_{{\rm fit}}$ vs FKR theory ({mode_str})")
    ax1.grid(True, which="both", ls=":")
    ax1.legend(loc="best")
    fig1.savefig(os.path.join(outdir, f"scan_gamma_fit_vs_FKR_{eq_mode}.png"))
    plt.close(fig1)
    print(f"[SAVE] scan_gamma_fit_vs_FKR_{eq_mode}.png")

    # ------------------------------------------------------------------ #
    # Plot 2: Rutherford scaling: (dw/dt)_R vs η Δ' (with error bars)
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots()
    ax2.errorbar(etaDelta_arr, dw_dt_R_arr, yerr=dw_dt_R_err_arr,
                 fmt="o", capsize=3, label=r"runs")

    R2_min_Ruth = 0.8
    mask_good_R = (
        np.isfinite(dw_dt_R_arr) &
        (dw_dt_R_arr > 0.0) &
        np.isfinite(etaDelta_arr) &
        (dw_dt_R_R2_arr >= R2_min_Ruth)
    )

    if np.count_nonzero(mask_good_R) >= 2:
        logx = np.log(etaDelta_arr[mask_good_R])
        logy = np.log(dw_dt_R_arr[mask_good_R])
        a_fit, b_fit = np.polyfit(logx, logy, 1)
        xfit = np.linspace(etaDelta_arr[mask_good_R].min() * 0.8,
                           etaDelta_arr[mask_good_R].max() * 1.2, 200)
        yfit = np.exp(b_fit) * xfit**a_fit
        ax2.loglog(xfit, yfit, "k--",
                   label=rf"fit (good runs): slope={a_fit:.2f}")
    else:
        print("[WARN] Too few good Rutherford points for a global scaling fit.")

    for i, name in enumerate(fnames):
        ax2.annotate(
            str(i),
            (etaDelta_arr[i], dw_dt_R_arr[i]),
            textcoords="offset points",
            xytext=(4, 2),
            fontsize=8,
        )

    ax2.set_xlabel(r"$\eta \Delta'$")
    ax2.set_ylabel(r"$(\mathrm{d}w/\mathrm{d}t)_R$")
    ax2.set_title(rf"Rutherford scaling ({mode_str})")
    ax2.grid(True, which="both", ls=":")
    ax2.legend(loc="best")
    fig2.savefig(os.path.join(outdir,
                              f"scan_Rutherford_dw_dt_vs_etaDelta_{eq_mode}.png"))
    plt.close(fig2)
    print(f"[SAVE] scan_Rutherford_dw_dt_vs_etaDelta_{eq_mode}.png")

    # ------------------------------------------------------------------ #
    # Plot 3: Saturated island width vs Δ' (normalized, with error bars)
    # ------------------------------------------------------------------ #
    fig3, ax3 = plt.subplots()
    ax3.errorbar(Delta_prime_arr, w_sat_over_a_arr, yerr=w_sat_over_a_std_arr,
                 fmt="o", capsize=3, label=r"runs")

    logx2 = np.log(Delta_prime_arr)
    logy2 = np.log(w_sat_over_a_arr)
    a_fit2, b_fit2 = np.polyfit(logx2, logy2, 1)
    xfit2 = np.linspace(Delta_prime_arr.min() * 0.8,
                        Delta_prime_arr.max() * 1.2, 200)
    yfit2 = np.exp(b_fit2) * xfit2**a_fit2
    ax3.loglog(xfit2, yfit2, "k--", label=rf"fit: slope={a_fit2:.2f}")

    for i, name in enumerate(fnames):
        ax3.annotate(
            str(i),
            (Delta_prime_arr[i], w_sat_over_a_arr[i]),
            textcoords="offset points",
            xytext=(4, 2),
            fontsize=8,
        )

    ax3.set_xlabel(r"$\Delta'$")
    ax3.set_ylabel(r"$w_{\rm sat}/a$")
    ax3.set_title(rf"Saturated island width vs $\Delta'$ ({mode_str})")
    ax3.grid(True, which="both", ls=":")
    ax3.legend(loc="best")
    fig3.savefig(os.path.join(outdir,
                              f"scan_wsat_over_a_vs_Deltaprime_{eq_mode}.png"))
    plt.close(fig3)
    print(f"[SAVE] scan_wsat_over_a_vs_Deltaprime_{eq_mode}.png")

    # ------------------------------------------------------------------ #
    # Plot 4: gamma_fit/gamma_FKR vs S and vs Delta'
    # ------------------------------------------------------------------ #
    ratio_arr = gamma_fit_arr / gamma_FKR_arr

    # (a) ratio vs S
    fig4a, ax4a = plt.subplots()
    ax4a.semilogx(S_arr, ratio_arr, "o")
    ax4a.set_xlabel(r"$S$")
    ax4a.set_ylabel(r"$\gamma_{\rm fit}/\gamma_{\rm FKR}$")
    ax4a.set_title(rf"Departure from FKR theory vs $S$ ({mode_str})")
    ax4a.grid(True, which="both", ls=":")
    fig4a.savefig(os.path.join(outdir,
                               f"scan_gamma_ratio_vs_S_{eq_mode}.png"))
    plt.close(fig4a)
    print(f"[SAVE] scan_gamma_ratio_vs_S_{eq_mode}.png")

    # (b) ratio vs Delta'
    fig4b, ax4b = plt.subplots()
    ax4b.semilogx(Delta_prime_arr, ratio_arr, "o")
    ax4b.set_xlabel(r"$\Delta'$")
    ax4b.set_ylabel(r"$\gamma_{\rm fit}/\gamma_{\rm FKR}$")
    ax4b.set_title(rf"Departure from FKR theory vs $\Delta'$ ({mode_str})")
    ax4b.grid(True, which="both", ls=":")
    fig4b.savefig(os.path.join(outdir,
                               f"scan_gamma_ratio_vs_Deltaprime_{eq_mode}.png"))
    plt.close(fig4b)
    print(f"[SAVE] scan_gamma_ratio_vs_Deltaprime_{eq_mode}.png")

    print("\n[DONE] Scan analysis complete for", eq_mode)
    print("      Each point index in the plots corresponds to:")
    for i, name in enumerate(fnames):
        print(f"        {i}: {name}")


# -----------------------------------------------------------------------------#
# CLI + driver
# -----------------------------------------------------------------------------#

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a multi-parameter tearing scan and build Loureiro-style plots "
                    "for both original and force-free equilibria."
    )

    # Grid and box (shared by both branches)
    p.add_argument("--Nx", type=int, default=152)
    p.add_argument("--Ny", type=int, default=152)
    p.add_argument("--Nz", type=int, default=1)
    p.add_argument("--Lx", type=float, default=2.0 * math.pi)
    p.add_argument("--Ly", type=float, default=2.0 * math.pi)
    p.add_argument("--Lz", type=float, default=2.0 * math.pi)

    # Physical parameters (shared)
    p.add_argument("--nu", type=float, default=5e-4)
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--Bg", type=float, default=0.0)
    # t0 and dt0 common; t1/n_frames are branch-specific in BRANCH_CONFIG
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--dt0", type=float, default=None)

    p.add_argument("--force-rerun", action="store_true",
                   help="Re-run simulations even if NPZ files already exist.")
    
    p.add_argument(
        "--keep-npz",
        action="store_true",
        help="Keep per-run mhd_tearing_solution_*.npz files. "
             "By default they are deleted after postprocessing to save disk."
    )

    # Fitting windows (kept for completeness; auto selector is used)
    p.add_argument(
        "--lin-tmin",
        type=float,
        default=None,
        help="(Unused) Minimum time for linear fit (auto window selector is used).",
    )
    p.add_argument(
        "--lin-tmax",
        type=float,
        default=None,
        help="(Unused) Maximum time for linear fit (auto window selector is used).",
    )
    p.add_argument(
        "--ruth-frac",
        type=float,
        nargs=2,
        default=(0.5, 0.95),
        metavar=("F_START", "F_END"),
        help=("Fractional window [F_START,F_END] of total time used as "
              "a nominal Rutherford interval (default: 0.5 0.95). "
              "The actual fit starts after the linear window."),
    )

    # Branch selection
    p.add_argument(
        "--equilibrium-modes",
        type=str,
        nargs="+",
        choices=["original", "forcefree"],
        default=["original", "forcefree"],
        help="Which equilibrium models to scan.",
    )

    # Output
    p.add_argument(
        "--outdir",
        type=str,
        default="tearing_scan_plots",
        help="Output directory for NPZ runs, summary, and plots.",
    )

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for eq_mode in args.equilibrium_modes:
        if eq_mode not in BRANCH_CONFIG:
            raise ValueError(f"No branch configuration for equilibrium_mode={eq_mode}")

        cfg = BRANCH_CONFIG[eq_mode]
        scan_a = cfg["scan_a"]
        scan_eta = cfg["scan_eta"]
        epsB = cfg["epsB"]
        t1 = cfg["t1"]
        n_frames = cfg["n_frames"]

        print("\n" + "=" * 72)
        print(f"[INFO] Starting scan for equilibrium_mode = '{eq_mode}'")
        print("=" * 72)
        print(f"[INFO] Branch parameters: epsB={epsB:g}, t1={t1:g}, "
              f"n_frames={n_frames}, scan_a={scan_a}, scan_eta={scan_eta}")

        eq_outdir = os.path.join(args.outdir, eq_mode)
        os.makedirs(eq_outdir, exist_ok=True)

        # Build list of (a, eta) combinations
        combos: List[Tuple[float, float]] = []
        for a in scan_a:
            for eta in scan_eta:
                combos.append((a, eta))

        print("[INFO] Scan combinations (a, eta):")
        for a, eta in combos:
            print(f"   a={a:.4g}, eta={eta:.4g}")

        results = []

        for idx, (a, eta) in enumerate(combos):
            tag = f"a{a:.3g}_eta{eta:.3g}_{eq_mode}"
            tag = tag.replace(".", "p").replace("-", "m")
            outfile = os.path.join(eq_outdir,
                                   f"mhd_tearing_solution_{tag}.npz")

            if os.path.exists(outfile) and not args.force_rerun:
                print(f"\n[INFO] Skipping solve for {tag}, file already exists.")
            else:
                print(f"\n[INFO] Running solve for {tag} ...")
                solve_tearing_case(
                    Nx=args.Nx,
                    Ny=args.Ny,
                    Nz=args.Nz,
                    Lx=args.Lx,
                    Ly=args.Ly,
                    Lz=args.Lz,
                    nu=args.nu,
                    eta=eta,
                    B0=args.B0,
                    a=a,
                    B_g=args.Bg,
                    eps_B=epsB,
                    t0=args.t0,
                    t1=t1,
                    n_frames=n_frames,
                    dt0=args.dt0,
                    outfile=outfile,
                    equilibrium_mode=eq_mode,
                )

            res = analyze_single_run(
                outfile,
                args.lin_tmin,
                args.lin_tmax,
                tuple(args.ruth_frac),
            )
            results.append(res)
            
            # Optionally delete the heavy NPZ file to save disk.
            if not args.keep_npz:
                try:
                    os.remove(outfile)
                    print(f"[CLEAN] Removed {outfile}")
                except OSError as e:
                    print(f"[WARN] Could not remove {outfile}: {e}")

        build_summary_and_plots(results, eq_outdir, eq_mode)


if __name__ == "__main__":
    main()
