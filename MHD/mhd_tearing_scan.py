#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_scan.py

"Loureiro-style" scan over many Harris-sheet tearing runs.

Each input file is a .npz produced by mhd_tearing_solve.py and contains
(ts, v_hat, B_hat, Nx,Ny,Nz, Lx,Ly,Lz, nu,eta, B0,a, ..., gamma_FKR,S,Delta_prime_a,ix0,iy1,iz0).

For each run, this script:

  * reconstructs A_z(k) and the tearing mode amplitude A_1(t),
  * computes island width proxy w(t),
  * extracts:
      - linear growth rate γ_fit from a user-defined or auto linear window,
      - Rutherford slope (dw/dt)_R from a late-time window,
      - saturated width w_sat (avg over last fraction of time),
  * collects η, a, Δ', S, γ_FKR,
  * and produces scan plots:

      1) γ_fit vs γ_FKR (log–log),
      2) (dw/dt)_R vs ηΔ',
      3) w_sat vs Δ'.

This is meant to reproduce "multi-run" style figures of nonlinear tearing
and Rutherford scaling in the spirit of classic reconnection papers.

Usage examples
--------------

  # Use glob pattern to pick up many runs
  python mhd_tearing_scan.py --pattern "runs/mhd_tearing_solution_*.npz"

  # Explicit list of files
  python mhd_tearing_scan.py run1.npz run2.npz run3.npz

  # Tweak linear and Rutherford fit ranges
  python mhd_tearing_scan.py --pattern "runs/*.npz" \
      --lin-tmin 0.0 --lin-tmax 20.0 \
      --ruth-frac 0.4 0.9

"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------#
# Matplotlib style
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
})


# -----------------------------------------------------------------------------#
# Helpers (mostly copied in spirit from mhd_tearing_island_evolution.py)
# -----------------------------------------------------------------------------#

def compute_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz):
    """Rebuild k-arrays consistent with the solver."""
    nx = np.fft.fftfreq(Nx) * Nx
    ny = np.fft.fftfreq(Ny) * Ny
    nz = np.fft.fftfreq(Nz) * Nz
    NX, NY, NZ = np.meshgrid(nx, ny, nz, indexing="ij")

    kx = 2.0 * np.pi * NX / Lx
    ky = 2.0 * np.pi * NY / Ly
    kz = 2.0 * np.pi * NZ / Lz
    return kx, ky, kz, NX, NY, NZ


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


def finite_difference(x, y):
    """Centered finite difference dy/dx."""
    dx = x[2:] - x[:-2]
    dy = y[2:] - y[:-2]
    x_mid = x[1:-1]
    dydx = dy / dx
    return x_mid, dydx


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

    Returns a dict with all diagnostics and parameters.
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

    Delta_prime = Delta_prime_a / a
    etaDelta = eta * Delta_prime

    print(f"[RUN] Nx={Nx}, Ny={Ny}, Nz={Nz}, Lx={Lx:.3f}, Ly={Ly:.3f}, Lz={Lz:.3f}")
    print(f"[RUN] nu={nu:.3e}, eta={eta:.3e}, B0={B0:.3e}, a={a:.3e}, eps_B={eps_B:.3e}")
    print(f"[RUN] S={S:.3e}, Delta'*a={Delta_prime_a:.3e}, Delta'={Delta_prime:.3e}")
    print(f"[RUN] γ_FKR={gamma_FKR:.3e}, mode indices (ix0,iy1,iz0)=({ix0},{iy1},{iz0})")

    # Build k arrays
    kx, ky, kz, NX, NY, NZ = compute_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    ky_val = ky[ix0, iy1, iz0]
    print(f"[DEBUG] ky for tearing mode = {ky_val:.6f}")

    n_t = ts.size
    island_width = np.zeros(n_t)
    Az_amp = np.zeros(n_t)

    # Compute A_z(k) and w(t)
    for it in range(n_t):
        B_hat = B_hat_frames[it]
        Az_hat = compute_Az_hat(B_hat, kx, ky)
        A_mode = Az_hat[ix0, iy1, iz0]

        Az_amp[it] = np.abs(A_mode)
        island_width[it] = compute_island_width_from_mode(A_mode, B0, a)

    w0 = island_width[0]
    print(f"[INFO] w0 = {w0:.3e}, w_max = {np.nanmax(island_width):.3e}")

    # ---- Linear fit: ln w vs t in a user-defined window ---- #
    if lin_tmin is None:
        lin_tmin = ts[0]
    if lin_tmax is None:
        lin_tmax = 1.05*ts[0] + 0.15 * (ts[-1] - ts[0])

    mask_lin = (ts >= lin_tmin) & (ts <= lin_tmax)
    # Require some growth: exclude points with w ~ w0
    mask_lin &= (island_width > 1.05 * w0)

    if np.count_nonzero(mask_lin) < 5:
        print("[WARN] Too few points in requested linear window; "
              "falling back to first 25% of time.")
        mask_lin = ts <= (ts[0] + 0.25 * (ts[-1] - ts[0]))

    t_lin = ts[mask_lin]
    w_lin = island_width[mask_lin]

    lnw_lin = np.log(w_lin)
    coeffs_lin = np.polyfit(t_lin, lnw_lin, 1)
    gamma_fit = coeffs_lin[0]
    lnw0_fit = coeffs_lin[1]
    print(f"[RESULT] γ_fit = {gamma_fit:.3e},  γ_fit/γ_FKR = {gamma_fit/gamma_FKR:.3f}")

    # ---- Rutherford slope: w(t) ~ w_R0 + (dw/dt)_R t in late-time window ---- #
    f_low, f_high = ruth_frac
    t_start = ts[0] + f_low * (ts[-1] - ts[0])
    t_end = ts[0] + f_high * (ts[-1] - ts[0])
    mask_ruth = (ts >= t_start) & (ts <= t_end)

    if np.count_nonzero(mask_ruth) < 5:
        print("[WARN] Too few points for Rutherford fit; "
              "using last half of time as fallback.")
        mask_ruth = ts >= (ts[0] + 0.5 * (ts[-1] - ts[0]))

    t_ruth = ts[mask_ruth]
    w_ruth = island_width[mask_ruth]
    coeffs_ruth = np.polyfit(t_ruth, w_ruth, 1)
    dw_dt_R = coeffs_ruth[0]
    print(f"[RESULT] (dw/dt)_R = {dw_dt_R:.3e}")

    # ---- Saturated island width: average over last 20% of time ---- #
    t_sat_min = ts[0] + 0.8 * (ts[-1] - ts[0])
    mask_sat = ts >= t_sat_min
    w_sat = float(np.mean(island_width[mask_sat]))
    print(f"[RESULT] w_sat = {w_sat:.3e}")

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
        "dw_dt_R": dw_dt_R,
        "w_sat": w_sat,
    }


# -----------------------------------------------------------------------------#
# Top-level CLI
# -----------------------------------------------------------------------------#

def parse_args():
    p = argparse.ArgumentParser(
        description="Scan over many tearing runs and build scaling plots."
    )
    p.add_argument(
        "inputs",
        nargs="*",
        help="List of .npz files to analyze (if --pattern not used).",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="runs/mhd_tearing_solution_*.npz",
        help="Glob pattern for input files, e.g. 'runs/mhd_tearing_*.npz'.",
    )
    p.add_argument(
        "--lin-tmin",
        type=float,
        default=None,
        help="Minimum time for linear fit (default: start time).",
    )
    p.add_argument(
        "--lin-tmax",
        type=float,
        default=None,
        help="Maximum time for linear fit (default: first 25%% of run).",
    )
    p.add_argument(
        "--ruth-frac",
        type=float,
        nargs=2,
        default=(0.4, 0.9),
        metavar=("F_START", "F_END"),
        help=("Fractional window [F_START,F_END] of total time used for "
              "Rutherford fit (default: 0.4 0.9)."),
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="tearing_scan_plots",
        help="Output directory for plots and summary NPZ.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Build file list
    file_list: List[str] = []
    if args.pattern is not None:
        file_list.extend(sorted(glob.glob(args.pattern)))
    file_list.extend(args.inputs)

    file_list = sorted(set(file_list))
    if not file_list:
        raise SystemExit("No input files found. Use --pattern or supply filenames.")

    print("[INFO] Files to analyze:")
    for f in file_list:
        print("   ", f)

    # Analyze each run
    results = []
    for f in file_list:
        res = analyze_single_run(f, args.lin_tmin, args.lin_tmax, tuple(args.ruth_frac))
        results.append(res)

    # Convert to arrays
    n = len(results)
    eta = np.array([r["eta"] for r in results])
    a = np.array([r["a"] for r in results])
    S = np.array([r["S"] for r in results])
    Delta_prime_a = np.array([r["Delta_prime_a"] for r in results])
    Delta_prime = np.array([r["Delta_prime"] for r in results])
    etaDelta = np.array([r["etaDelta"] for r in results])
    gamma_FKR = np.array([r["gamma_FKR"] for r in results])
    gamma_fit = np.array([r["gamma_fit"] for r in results])
    dw_dt_R = np.array([r["dw_dt_R"] for r in results])
    w_sat = np.array([r["w_sat"] for r in results])
    fnames = np.array([r["fname"] for r in results], dtype=object)

    # Save summary NPZ
    summary_path = os.path.join(args.outdir, "tearing_scan_summary.npz")
    np.savez(
        summary_path,
        fnames=fnames,
        eta=eta,
        a=a,
        S=S,
        Delta_prime_a=Delta_prime_a,
        Delta_prime=Delta_prime,
        etaDelta=etaDelta,
        gamma_FKR=gamma_FKR,
        gamma_fit=gamma_fit,
        dw_dt_R=dw_dt_R,
        w_sat=w_sat,
    )
    print(f"\n[SAVE] Summary saved to {summary_path}")

    # ------------------------------------------------------------------ #
    # Plot 1: γ_fit vs γ_FKR (Loureiro-style linear growth scaling)
    # ------------------------------------------------------------------ #
    fig1, ax1 = plt.subplots()
    ax1.loglog(gamma_FKR, gamma_fit, "o", label=r"runs")
    # Reference line y=x
    gmin = 0.5 * np.min(gamma_FKR)
    gmax = 2.0 * np.max(gamma_FKR)
    ref = np.linspace(gmin, gmax, 100)
    ax1.loglog(ref, ref, "k--", label=r"$\gamma_{\rm fit}=\gamma_{\rm FKR}$")

    for i, name in enumerate(fnames):
        ax1.annotate(
            str(i),
            (gamma_FKR[i], gamma_fit[i]),
            textcoords="offset points",
            xytext=(4, 2),
            fontsize=8,
        )

    ax1.set_xlabel(r"$\gamma_{\rm FKR}$")
    ax1.set_ylabel(r"$\gamma_{\rm fit}$")
    ax1.set_title(r"Linear growth: $\gamma_{\rm fit}$ vs FKR theory")
    ax1.grid(True, which="both", ls=":")
    ax1.legend(loc="best")
    fig1.savefig(os.path.join(args.outdir, "scan_gamma_fit_vs_FKR.png"))
    print(f"[SAVE] scan_gamma_fit_vs_FKR.png")

    # ------------------------------------------------------------------ #
    # Plot 2: Rutherford scaling: (dw/dt)_R vs η Δ'
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots()
    ax2.loglog(etaDelta, dw_dt_R, "o", label=r"runs")

    # Power-law fit in log-log
    logx = np.log(etaDelta)
    logy = np.log(dw_dt_R)
    a_fit, b_fit = np.polyfit(logx, logy, 1)  # log y = a_fit * log x + b_fit
    xfit = np.linspace(etaDelta.min() * 0.8, etaDelta.max() * 1.2, 100)
    yfit = np.exp(b_fit) * xfit**a_fit
    ax2.loglog(xfit, yfit, "k--", label=rf"fit: slope={a_fit:.2f}")

    for i, name in enumerate(fnames):
        ax2.annotate(
            str(i),
            (etaDelta[i], dw_dt_R[i]),
            textcoords="offset points",
            xytext=(4, 2),
            fontsize=8,
        )

    ax2.set_xlabel(r"$\eta \Delta'$")
    ax2.set_ylabel(r"$(\mathrm{d}w/\mathrm{d}t)_R$")
    ax2.set_title(r"Rutherford scaling")
    ax2.grid(True, which="both", ls=":")
    ax2.legend(loc="best")
    fig2.savefig(os.path.join(args.outdir, "scan_Rutherford_dw_dt_vs_etaDelta.png"))
    print(f"[SAVE] scan_Rutherford_dw_dt_vs_etaDelta.png")

    # ------------------------------------------------------------------ #
    # Plot 3: Saturated island width vs Δ'
    # ------------------------------------------------------------------ #
    fig3, ax3 = plt.subplots()
    ax3.loglog(Delta_prime, w_sat, "o", label=r"runs")

    # Optional power-law fit
    logx = np.log(Delta_prime)
    logy = np.log(w_sat)
    a_fit2, b_fit2 = np.polyfit(logx, logy, 1)
    xfit2 = np.linspace(Delta_prime.min() * 0.8, Delta_prime.max() * 1.2, 100)
    yfit2 = np.exp(b_fit2) * xfit2**a_fit2
    ax3.loglog(xfit2, yfit2, "k--", label=rf"fit: slope={a_fit2:.2f}")

    for i, name in enumerate(fnames):
        ax3.annotate(
            str(i),
            (Delta_prime[i], w_sat[i]),
            textcoords="offset points",
            xytext=(4, 2),
            fontsize=8,
        )

    ax3.set_xlabel(r"$\Delta'$")
    ax3.set_ylabel(r"$w_{\rm sat}$")
    ax3.set_title(r"Saturated island width vs $\Delta'$")
    ax3.grid(True, which="both", ls=":")
    ax3.legend(loc="best")
    fig3.savefig(os.path.join(args.outdir, "scan_wsat_vs_Deltaprime.png"))
    print(f"[SAVE] scan_wsat_vs_Deltaprime.png")

    print("\n[DONE] Scan analysis complete.")
    print("      Each point index in the plots corresponds to:")
    for i, name in enumerate(fnames):
        print(f"        {i}: {name}")


if __name__ == "__main__":
    main()
