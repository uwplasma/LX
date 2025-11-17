#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_island_evolution.py

Nonlinear tearing and island evolution analysis for the Harris-sheet MHD run.

Usage:
  python mhd_tearing_island_evolution.py \
      --input mhd_tearing_solution.npz \
      --make-movie

This script (single-run diagnostics):

  1) Island width and growth rates
     - island half-width w(t) from the (kx=0, ky=1, kz=0) mode of A_z,
     - linear phase fit (γ_fit) vs FKR γ_theory,
     - Rutherford-like regime fit (dw/dt)_R vs η Δ'.

  2) Field structure
     - snapshots of A_z(x,y,z=0) at representative times,
     - optional movie A_z(x,y,z=0,t) for talks.

  3) Reconnection rate diagnostics (NEW)
     - reconnected flux Δψ(t) = A_O(t) - A_X(t),
     - reconnection rate dΔψ/dt vs η J_z at the X-point.

  4) Energy budget (NEW)
     - E_mag(t), E_kin(t), E_tot(t),
     - Ohmic power P_η(t) = η ∫ J^2 dV vs -dE_mag/dt.

  5) Profile evolution (NEW)
     - B_y(x,y=0,z=0) and J_z(x,y=0,z=0) at
       early-linear, Rutherford, and saturated times,
       including initial analytic Harris profile for comparison.

These diagnostics are in the spirit of classical nonlinear tearing
and reconnection studies (Rutherford, Loureiro et al., Murphy et al.).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------------------------------#
# Matplotlib / plotting style (publication-ready-ish)
# -----------------------------------------------------------------------------#

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#

def select_linear_window(ts, w, w0=None, min_pts=6, frac_sat=0.3, nwin=8):
    """
    Automatically pick a 'best' linear window for ln w(t):

    - Avoids very early transient (w ~ w0) and near-saturation.
    - Restricts to  w > 1.2 w0  and  w < w0 + frac_sat*(w_max - w0).
    - Slides a fixed-size window of length nwin over candidates and
      chooses the one that minimizes the mean-squared residual of a
      linear fit to ln w.

    Parameters
    ----------
    ts : array, shape (N,)
        Time grid.
    w : array, shape (N,)
        Island half-width.
    w0 : float or None
        Reference initial width; if None, uses w[0].
    min_pts : int
        Minimum # of points required for a linear window.
    frac_sat : float
        Fraction of the distance from w0 to w_max to allow; e.g. 0.3
        means we only use w up to ~30% towards saturation.
    nwin : int
        Sliding window length for the search.

    Returns
    -------
    mask_lin : bool array, shape (N,)
        True where points are in the chosen linear window.
    """
    ts = np.asarray(ts)
    w = np.asarray(w)

    if w0 is None:
        w0 = w[0]

    # Basic sanity
    if ts.size < min_pts + 2:
        # not enough points, fall back to earliest quarter
        return ts < (ts[0] + 0.25 * (ts[-1] - ts[0]))

    w_max = np.nanmax(w)

    # Candidates: bigger than initial, but not too close to saturation
    upper = w0 + frac_sat * (w_max - w0)
    mask_cand = (w > 1.2 * w0) & (w < upper) & np.isfinite(w)
    idx_cand = np.where(mask_cand)[0]

    if idx_cand.size < min_pts:
        # fall back: early chunk
        return ts < (ts[0] + 0.25 * (ts[-1] - ts[0]))

    nwin = min(nwin, idx_cand.size)
    lnw = np.log(w)

    best_err = np.inf
    best_idx_slice = None

    # Slide an nwin-sized window over candidate indices
    for start in range(0, idx_cand.size - nwin + 1):
        win_idx = idx_cand[start:start + nwin]
        t_win = ts[win_idx]
        y_win = lnw[win_idx]

        # linear fit ln w = a t + b
        coeffs = np.polyfit(t_win, y_win, 1)
        fit = np.polyval(coeffs, t_win)
        err = np.mean((y_win - fit)**2)

        if err < best_err:
            best_err = err
            best_idx_slice = win_idx

    mask_lin = np.zeros_like(w, dtype=bool)
    if best_idx_slice is not None and best_idx_slice.size >= min_pts:
        mask_lin[best_idx_slice] = True
    else:
        # conservative fallback
        mask_lin = ts < (ts[0] + 0.25 * (ts[-1] - ts[0]))

    return mask_lin

def compute_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz):
    """Rebuild kx, ky, kz consistent with mhd_tearing_solve.py."""
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
    (k_perp^2 = kx^2 + ky^2).

    Parameters
    ----------
    B_hat : (3, Nx, Ny, Nz) complex
    kx, ky : (Nx, Ny, Nz) real

    Returns
    -------
    Az_hat : (Nx, Ny, Nz) complex
    """
    Bx_hat, By_hat = B_hat[0], B_hat[1]
    k_perp2 = kx**2 + ky**2
    k_perp2_safe = np.where(k_perp2 == 0.0, 1.0, k_perp2)

    Az_hat = 1j * (kx * By_hat - ky * Bx_hat) / k_perp2_safe
    Az_hat = np.where(k_perp2 == 0.0, 0.0, Az_hat)
    return Az_hat


def compute_island_width_from_mode(Az_hat_mode, B0, a):
    """
    Proxy island half-width from Ã_1 amplitude:

      w ≈ 4 * sqrt( |Ã_1| / |B'_y(x_s)| ),

    where for the Harris sheet B_y(x) = B0 tanh((x - Lx/2)/a),
    we have B'_y(x_s = Lx/2) = B0 / a.

    Parameters
    ----------
    Az_hat_mode : complex
        Ã_1(t) = A_z_hat(kx=0, ky=1, kz=0) at given time.
    B0, a : floats

    Returns
    -------
    w : float
        Proxy island half-width.
    """
    A_amp = np.abs(Az_hat_mode)
    Bprime = B0 / a
    if Bprime <= 0.0:
        return np.nan
    return 4.0 * np.sqrt(A_amp / Bprime)


def finite_difference(x, y):
    """
    Compute dy/dx via centered finite differences (1D).

    Returns
    -------
    x_mid : (N-2,)
    dydx  : (N-2,)
    """
    dx = x[2:] - x[:-2]
    dy = y[2:] - y[:-2]
    x_mid = x[1:-1]
    dydx = dy / dx
    return x_mid, dydx


def energy_from_hat_np(v_hat, B_hat, Lx, Ly, Lz):
    """
    Compute kinetic and magnetic energy from Fourier coefficients.
    """
    v = np.fft.ifftn(v_hat, axes=(1, 2, 3)).real
    B = np.fft.ifftn(B_hat, axes=(1, 2, 3)).real
    dv = (Lx * Ly * Lz) / (v.shape[1] * v.shape[2] * v.shape[3])

    v2 = np.sum(v * v, axis=0)
    B2 = np.sum(B * B, axis=0)
    E_kin = 0.5 * np.sum(v2) * dv
    E_mag = 0.5 * np.sum(B2) * dv
    return E_kin, E_mag


def compute_J_from_hat(B_hat, kx, ky, kz):
    """
    J = ∇×B from Fourier coefficients of B.
    """
    Bx_hat, By_hat, Bz_hat = B_hat[0], B_hat[1], B_hat[2]

    Jx_hat = 1j * (ky * Bz_hat - kz * By_hat)
    Jy_hat = 1j * (kz * Bx_hat - kx * Bz_hat)
    Jz_hat = 1j * (kx * By_hat - ky * Bx_hat)

    J_hat = np.stack([Jx_hat, Jy_hat, Jz_hat], axis=0)
    J = np.fft.ifftn(J_hat, axes=(1, 2, 3)).real
    return J


# -----------------------------------------------------------------------------#
# Main analysis
# -----------------------------------------------------------------------------#

def parse_args():
    p = argparse.ArgumentParser(
        description="Nonlinear tearing island evolution analysis."
    )
    p.add_argument("--input", nargs="?", default="mhd_tearing_solution.npz",
                        help="Input .npz file produced by mhd_tearing_solve.py")
    p.add_argument("--lin-tmax", type=float, default=None,
                   help="Maximum time for linear fit (default: auto).")
    p.add_argument("--ruth-tmin", type=float, default=None,
                   help="Minimum time for Rutherford fit (default: auto).")
    p.add_argument("--ruth-tmax", type=float, default=None,
                   help="Maximum time for Rutherford fit (default: final time).")
    p.add_argument("--movie-nframes", type=int, default=150,
                   help="Number of frames for the A_z movie (subsampled).")
    p.add_argument("--make-movie", action="store_true",
                   help="If set, build a movie A_z(x,y,z=0,t) -> mp4.")
    p.add_argument("--movie-fname", type=str, default="island_Az_movie.mp4",
                   help="Filename for the movie.")
    p.add_argument("--outdir", type=str, default="tearing_plots",
                   help="Directory to save plots and movie.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[INFO] Loading solution from {args.input} ...")
    data = np.load(args.input, allow_pickle=True)

    ts = data["ts"]            # (n_frames,)
    v_hat_frames = data["v_hat"]   # (n_frames, 3, Nx, Ny, Nz)
    B_hat_frames = data["B_hat"]   # (n_frames, 3, Nx, Ny, Nz)

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

    print("========== Loaded run parameters ==========")
    print(f"Nx,Ny,Nz = {Nx},{Ny},{Nz}")
    print(f"Lx,Ly,Lz = {Lx},{Ly},{Lz}")
    print(f"nu={nu:.3e}, eta={eta:.3e}, S={S:.3e}")
    print(f"B0={B0:.3e}, a={a:.3e}, eps_B={eps_B:.3e}")
    print(f"Delta' * a = {Delta_prime_a:.3e}")
    print(f"FKR γ_theory ≈ {gamma_FKR:.3e}")
    print(f"Mode indices: ix0={ix0}, iy1={iy1}, iz0={iz0}")
    print("===========================================")

    # Rebuild k-arrays
    kx, ky, kz, NX, NY, NZ = compute_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    ky_val = ky[ix0, iy1, iz0]
    print(f"[DEBUG] ky for tearing mode = {ky_val:.6f}")

    n_t = ts.size
    island_width = np.zeros(n_t)
    Az_mode_amp = np.zeros(n_t)

    print("[INFO] Computing A_z(k) and island width w(t) from tearing mode ...")
    for it in range(n_t):
        B_hat = B_hat_frames[it]  # (3,Nx,Ny,Nz)
        Az_hat = compute_Az_hat(B_hat, kx, ky)
        A_mode = Az_hat[ix0, iy1, iz0]   # (kx=0, ky=1, kz=0)

        Az_mode_amp[it] = np.abs(A_mode)
        island_width[it] = compute_island_width_from_mode(A_mode, B0, a)

        if it % max(1, n_t // 10) == 0:
            print(f"[DEBUG] t={ts[it]:7.3f}, |A_mode|={Az_mode_amp[it]:.3e}, "
                  f"w={island_width[it]:.3e}")

    print("[INFO] Island width stats:")
    print(f"    w_min={np.nanmin(island_width):.3e}")
    print(f"    w_max={np.nanmax(island_width):.3e}")

    # --------------------------------------------------------------------- #
    # Linear phase fit: w(t) ~ w0 exp(γ t)
    # --------------------------------------------------------------------- #
    w0 = island_width[0]

    if args.lin_tmax is None:
        # Automatic, data-driven choice of the linear window
        mask_lin = select_linear_window(
            ts,
            island_width,
            w0=w0,
            min_pts=6,
            frac_sat=0.3,
            nwin=8,
        )
    else:
        # User overrides upper bound, but still avoid very early transient
        mask_lin = (ts <= args.lin_tmax) & (island_width > 1.2 * w0)

    if np.count_nonzero(mask_lin) < 5:
        print("[WARN] Not enough points for a robust linear-phase fit, "
              "falling back to an early-time window.")
        mask_lin = ts < (ts[0] + 0.25 * (ts[-1] - ts[0]))

    t_lin = ts[mask_lin]
    w_lin = island_width[mask_lin]
    lnw_lin = np.log(w_lin)

    print(f"[INFO] Linear fit window: t ∈ [{t_lin[0]:.3f}, {t_lin[-1]:.3f}] "
          f"({t_lin.size} points)")

    coeffs_lin = np.polyfit(t_lin, lnw_lin, 1)
    gamma_fit = coeffs_lin[0]
    lnw0_fit = coeffs_lin[1]
    w_fit_lin = np.exp(lnw0_fit + gamma_fit * ts)
    if np.count_nonzero(mask_lin) < 5:
        print("[WARN] Not enough points for a robust linear-phase fit, "
              "trying a simple early-time window.")
        mask_lin = ts < (ts[0] + 0.25 * (ts[-1] - ts[0]))

    t_lin = ts[mask_lin]
    w_lin = island_width[mask_lin]
    lnw_lin = np.log(w_lin)

    coeffs_lin = np.polyfit(t_lin, lnw_lin, 1)
    gamma_fit = coeffs_lin[0]
    lnw0_fit = coeffs_lin[1]
    w_fit_lin = np.exp(lnw0_fit + gamma_fit * ts)

    print("---------- Linear phase fit ----------")
    print(f"[RESULT] γ_fit (from w)     = {gamma_fit:.3e}")
    print(f"[RESULT] γ_FKR (theory)     = {gamma_FKR:.3e}")
    print(f"[RESULT] γ_fit / γ_FKR      = {gamma_fit / gamma_FKR:.3f}")
    print("--------------------------------------")

    # --------------------------------------------------------------------- #
    # Rutherford phase fit: w(t) ~ w_R0 + (d w/dt)_R t
    # --------------------------------------------------------------------- #
    t_start_ruth = args.ruth_tmin if args.ruth_tmin is not None else ts[0] + 0.4 * (ts[-1] - ts[0])
    t_end_ruth = args.ruth_tmax if args.ruth_tmax is not None else ts[-1]

    mask_ruth = (ts >= t_start_ruth) & (ts <= t_end_ruth)
    if np.count_nonzero(mask_ruth) < 5:
        print("[WARN] Not enough points for Rutherford fit with given window, "
              "expanding automatically.")
        mask_ruth = ts >= (ts[0] + 0.3 * (ts[-1] - ts[0]))

    t_ruth = ts[mask_ruth]
    w_ruth = island_width[mask_ruth]

    coeffs_ruth = np.polyfit(t_ruth, w_ruth, 1)
    dw_dt_fit = coeffs_ruth[0]
    w_r0 = coeffs_ruth[1]
    w_fit_ruth = dw_dt_fit * ts + w_r0

    print("---------- Rutherford phase fit ----------")
    print(f"[RESULT] (dw/dt)_R,fit         = {dw_dt_fit:.3e}")
    Delta_prime = Delta_prime_a / a
    C_R = 1.0
    dw_dt_Ruth = C_R * eta * Delta_prime
    print(f"[RESULT] (dw/dt)_R,Ruth (C_R={C_R:.2f}) = {dw_dt_Ruth:.3e}")
    if dw_dt_Ruth != 0.0:
        print(f"[RESULT] (dw/dt)_fit / (dw/dt)_Ruth = {dw_dt_fit / dw_dt_Ruth:.3f}")
    print("------------------------------------------")

    # Instantaneous growth rates γ_inst = d(ln w)/dt
    t_gamma, gamma_inst = finite_difference(ts, np.log(island_width))

    # --------------------------------------------------------------------- #
    # Figure 1: island width and growth rate diagnostics
    # --------------------------------------------------------------------- #
    print("[INFO] Building island-width diagnostic plots ...")
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    # (a) w(t)
    ax = axs[0, 0]
    ax.plot(ts, island_width, label=r"$w(t)$")
    ax.plot(ts, w_fit_lin, "--", label=rf"Linear fit, $\gamma={gamma_fit:.2e}$")
    ax.plot(ts, w_fit_ruth, ":", label=r"Rutherford fit")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$w$")
    ax.set_title(r"Island half-width $w(t)$")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="best")

    # (b) log w vs t
    ax = axs[0, 1]
    ax.plot(ts, np.log(island_width), label=r"$\ln w$")
    ax.plot(t_lin, lnw_lin, "o", ms=3, label="Linear fit window")
    ax.plot(ts, np.log(w_fit_lin), "--", label=rf"Fit: $\gamma={gamma_fit:.2e}$")
    ax.axhline(np.log(w0), color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\ln w$")
    ax.set_title(r"Linear phase")
    ax.grid(True, ls=":")
    ax.legend(loc="best")

    # (c) dw/dt vs t
    t_dw, dw_dt_inst = finite_difference(ts, island_width)
    ax = axs[1, 0]
    ax.plot(t_dw, dw_dt_inst, label=r"$\mathrm{d}w/\mathrm{d}t$ (inst.)")
    ax.axhline(dw_dt_fit, color="C1", ls="--",
               label=rf"Rutherford fit: $(\mathrm{{d}}w/\mathrm{{d}}t)_R={dw_dt_fit:.2e}$")
    ax.axhline(dw_dt_Ruth, color="C2", ls=":",
               label=rf"Rutherford theory: $\propto \eta \Delta'$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathrm{d}w/\mathrm{d}t$")
    ax.set_title(r"Rutherford regime")
    ax.grid(True, ls=":")
    ax.legend(loc="best")

    # (d) γ_inst vs t
    ax = axs[1, 1]
    ax.plot(t_gamma, gamma_inst, label=r"$\gamma_{\mathrm{inst}}$")
    ax.axhline(gamma_FKR, color="C2", ls="--",
               label=rf"FKR $\gamma={gamma_FKR:.2e}$")
    ax.axhline(gamma_fit, color="C1", ls=":",
               label=rf"Fit $\gamma={gamma_fit:.2e}$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\gamma_{\mathrm{inst}}$")
    ax.set_title(r"Instantaneous growth rate")
    ax.grid(True, ls=":")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "island_width_diagnostics.png"))
    print(f"[SAVE] island_width_diagnostics.png")

    # --------------------------------------------------------------------- #
    # Precompute Az(x,y,z=0) helper (used by several diagnostics)
    # --------------------------------------------------------------------- #
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    def Az_xy_at_frame(it):
        B_hat = B_hat_frames[it]
        Az_hat = compute_Az_hat(B_hat, kx, ky)
        Az = np.fft.ifftn(Az_hat, axes=(0, 1, 2)).real
        return Az[:, :, 0]  # z=0 plane

    # --------------------------------------------------------------------- #
    # NEW: reconnected flux and reconnection rate
    # --------------------------------------------------------------------- #
    print("[INFO] Computing reconnected flux Δψ(t) and reconnection rate ...")
    ix_sheet = Nx // 2  # x index closest to Lx/2

    Az_O = np.zeros(n_t)
    Az_X = np.zeros(n_t)
    Delta_psi = np.zeros(n_t)
    iy_X_arr = np.zeros(n_t, dtype=int)

    for it in range(n_t):
        Az_xy = Az_xy_at_frame(it)
        Az_line = Az_xy[ix_sheet, :]  # along the sheet center

        iy_O = np.argmax(Az_line)
        iy_X = np.argmin(Az_line)

        Az_O[it] = Az_xy[ix_sheet, iy_O]
        Az_X[it] = Az_xy[ix_sheet, iy_X]
        Delta_psi[it] = Az_O[it] - Az_X[it]
        iy_X_arr[it] = iy_X

        if it % max(1, n_t // 10) == 0:
            print(f"[DEBUG] t={ts[it]:7.3f}, A_O={Az_O[it]:.3e}, "
                  f"A_X={Az_X[it]:.3e}, Δψ={Delta_psi[it]:.3e}")

    t_psi, dpsi_dt = finite_difference(ts, Delta_psi)

    # Compute J and Ez at X-point for each frame
    print("[INFO] Computing J_z and E_z at X-point ...")
    dv = (Lx * Ly * Lz) / (Nx * Ny * Nz)
    Jz_X = np.zeros(n_t)
    Ez_X = np.zeros(n_t)
    P_ohm = np.zeros(n_t)   # volume-integrated η J^2

    for it in range(n_t):
        B_hat = B_hat_frames[it]
        J = compute_J_from_hat(B_hat, kx, ky, kz)  # (3,Nx,Ny,Nz)
        Jx, Jy, Jz = J[0], J[1], J[2]

        iy_X = iy_X_arr[it]
        Jz_X[it] = Jz[ix_sheet, iy_X, 0]
        Ez_X[it] = eta * Jz_X[it]

        P_ohm[it] = eta * np.sum(Jx**2 + Jy**2 + Jz**2) * dv

        if it % max(1, n_t // 10) == 0:
            print(f"[DEBUG] t={ts[it]:7.3f}, Jz_X={Jz_X[it]:.3e}, "
                  f"E_z={Ez_X[it]:.3e}, P_ohm={P_ohm[it]:.3e}")

    # Figure 2: Δψ and reconnection rate
    figR, axsR = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True)

    ax = axsR[0]
    ax.plot(ts, Delta_psi, label=r"$\Delta\psi(t)$")
    ax.set_ylabel(r"$\Delta\psi$")
    ax.set_title(r"Reconnected flux")
    ax.grid(True, ls=":")
    ax.legend(loc="best")

    ax = axsR[1]
    ax.plot(t_psi, dpsi_dt, label=r"$\mathrm{d}\Delta\psi/\mathrm{d}t$")
    ax.plot(ts, Ez_X, "--", label=r"$E_z^{\rm X}(t)=\eta J_z^{\rm X}$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"Rate")
    ax.set_title(r"Reconnection rate at X-point")
    ax.grid(True, ls=":")
    ax.legend(loc="best")

    figR.tight_layout()
    figR.savefig(os.path.join(args.outdir, "reconnected_flux_reconnection_rate.png"))
    print(f"[SAVE] reconnected_flux_reconnection_rate.png")

    # --------------------------------------------------------------------- #
    # NEW: Energy budget
    # --------------------------------------------------------------------- #
    print("[INFO] Computing energy budget (E_mag, E_kin, E_tot, P_ohm) ...")
    E_kin = np.zeros(n_t)
    E_mag = np.zeros(n_t)
    for it in range(n_t):
        Ek, Em = energy_from_hat_np(v_hat_frames[it], B_hat_frames[it], Lx, Ly, Lz)
        E_kin[it] = Ek
        E_mag[it] = Em
        if it % max(1, n_t // 10) == 0:
            print(f"[DEBUG] t={ts[it]:7.3f}, E_kin={Ek:.3e}, E_mag={Em:.3e}")

    E_tot = E_kin + E_mag
    t_dEm, dEm_dt = finite_difference(ts, E_mag)

    figE, axsE = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True)

    ax = axsE[0]
    ax.plot(ts, E_mag, label=r"$E_{\rm mag}$")
    ax.plot(ts, E_kin, label=r"$E_{\rm kin}$")
    ax.plot(ts, E_tot, "--", label=r"$E_{\rm tot}$")
    ax.set_ylabel(r"Energy")
    ax.set_title(r"Global energy evolution")
    ax.grid(True, ls=":")
    ax.legend(loc="best")

    ax = axsE[1]
    ax.plot(t_dEm, -dEm_dt, label=r"$-\,\mathrm{d}E_{\rm mag}/\mathrm{d}t$")
    ax.plot(ts, P_ohm, "--", label=r"$P_\eta(t)=\eta\!\int J^2 dV$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"Power")
    ax.set_title(r"Magnetic energy loss vs Ohmic heating")
    ax.grid(True, ls=":")
    ax.legend(loc="best")

    figE.tight_layout()
    figE.savefig(os.path.join(args.outdir, "energy_budget.png"))
    print(f"[SAVE] energy_budget.png")

    # --------------------------------------------------------------------- #
    # Figure: snapshots of A_z(x,y,z=0,t) (as before)
    # --------------------------------------------------------------------- #
    print("[INFO] Building A_z snapshot figure ...")
    t_early = ts[int(0.2 * (n_t - 1))]
    t_mid = ts[int(0.6 * (n_t - 1))]
    t_late = ts[-1]

    idx_early = np.argmin(np.abs(ts - t_early))
    idx_mid = np.argmin(np.abs(ts - t_mid))
    idx_late = np.argmin(np.abs(ts - t_late))

    Az_early = Az_xy_at_frame(idx_early)
    Az_mid = Az_xy_at_frame(idx_mid)
    Az_late = Az_xy_at_frame(idx_late)

    vmin = min(Az_early.min(), Az_mid.min(), Az_late.min())
    vmax = max(Az_early.max(), Az_mid.max(), Az_late.max())

    fig2, axs2 = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True, sharey=True)

    # Symmetric contour levels around zero
    A_abs = max(abs(vmin), abs(vmax))
    levels = np.linspace(-A_abs, A_abs, 17)

    # X- and O-point flux levels for the three snapshot times
    A_X_early, A_O_early = Az_X[idx_early], Az_O[idx_early]
    A_X_mid,   A_O_mid   = Az_X[idx_mid],   Az_O[idx_mid]
    A_X_late,  A_O_late  = Az_X[idx_late],  Az_O[idx_late]

    for ax, Az_xy, label, t_val, A_X_val, A_O_val in zip(
        axs2,
        [Az_early, Az_mid, Az_late],
        [r"Early (linear)", r"Rutherford", r"Saturated"],
        [ts[idx_early], ts[idx_mid], ts[idx_late]],
        [A_X_early, A_X_mid, A_X_late],
        [A_O_early, A_O_mid, A_O_late],
    ):
        # background color map
        im = ax.pcolormesh(X, Y, Az_xy, shading="auto", vmin=vmin, vmax=vmax)
        # many thin contours (field lines)
        cs = ax.contour(X, Y, Az_xy, levels=levels, colors="k",
                        linewidths=0.4, alpha=0.7)
        # highlight X-point separatrix (solid white)
        ax.contour(X, Y, Az_xy, levels=[A_X_val], colors="w",
                   linewidths=1.5)
        # highlight O-point contour (dashed white)
        ax.contour(X, Y, Az_xy, levels=[A_O_val], colors="w",
                   linewidths=1.0, linestyles="--")

        ax.set_title(fr"{label}, $t={t_val:.1f}$")
        ax.set_xlabel(r"$x$")
        ax.set_aspect("equal")

    axs2[0].set_ylabel(r"$y$")

    cbar = fig2.colorbar(im, ax=axs2.ravel().tolist(), shrink=0.8)
    cbar.set_label(r"$A_z(x,y,z=0)$")

    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, "Az_snapshots.png"))
    print(f"[SAVE] Az_snapshots.png")

    # --------------------------------------------------------------------- #
    # NEW: profiles of B_y(x) and J_z(x) at three stages
    # --------------------------------------------------------------------- #
    print("[INFO] Building B_y and J_z profile figure ...")
    def B_and_J_xy_at_frame(it):
        B_hat = B_hat_frames[it]
        B = np.fft.ifftn(B_hat, axes=(1, 2, 3)).real  # (3,Nx,Ny,Nz)
        J = compute_J_from_hat(B_hat, kx, ky, kz)
        By_xy = B[1, :, :, 0]
        Jz_xy = J[2, :, :, 0]
        return By_xy, Jz_xy

    By_early_xy, Jz_early_xy = B_and_J_xy_at_frame(idx_early)
    By_mid_xy, Jz_mid_xy = B_and_J_xy_at_frame(idx_mid)
    By_late_xy, Jz_late_xy = B_and_J_xy_at_frame(idx_late)

    iy_line = 0  # y=0 line (O-line for cos ky y perturbation)
    By_early = By_early_xy[:, iy_line]
    By_mid = By_mid_xy[:, iy_line]
    By_late = By_late_xy[:, iy_line]

    Jz_early = Jz_early_xy[:, iy_line]
    Jz_mid = Jz_mid_xy[:, iy_line]
    Jz_late = Jz_late_xy[:, iy_line]

    # Analytic initial Harris equilibrium
    sx = (x - 0.5 * Lx) / a
    By_eq = B0 * np.tanh(sx)

    figP, axsP = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True)

    ax = axsP[0]
    ax.plot(x, By_eq, "k--", label=r"Harris eq.")
    ax.plot(x, By_early, label=fr"Early, $t={ts[idx_early]:.1f}$")
    ax.plot(x, By_mid, label=fr"Rutherford, $t={ts[idx_mid]:.1f}$")
    ax.plot(x, By_late, label=fr"Saturated, $t={ts[idx_late]:.1f}$")
    ax.set_ylabel(r"$B_y(x,y=0,z=0)$")
    ax.set_title(r"Current-sheet flattening")
    ax.grid(True, ls=":")
    ax.legend(loc="best")

    ax = axsP[1]
    ax.plot(x, Jz_early, label=fr"Early, $t={ts[idx_early]:.1f}$")
    ax.plot(x, Jz_mid, label=fr"Rutherford, $t={ts[idx_mid]:.1f}$")
    ax.plot(x, Jz_late, label=fr"Saturated, $t={ts[idx_late]:.1f}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$J_z(x,y=0,z=0)$")
    ax.set_title(r"Current density evolution")
    ax.grid(True, ls=":")
    ax.legend(loc="best")

    figP.tight_layout()
    figP.savefig(os.path.join(args.outdir, "profiles_By_Jz.png"))
    print(f"[SAVE] profiles_By_Jz.png")

    # --------------------------------------------------------------------- #
    # Movie: A_z(x,y,z=0,t)
    # --------------------------------------------------------------------- #
    if args.make_movie:
        print("[INFO] Building A_z movie ...")
        n_movie = min(args.movie_nframes, n_t)
        idx_movie = np.linspace(0, n_t - 1, n_movie, dtype=int)

        Az_movie = []
        for i, it in enumerate(idx_movie):
            Az_xy = Az_xy_at_frame(it)
            Az_movie.append(Az_xy)
            if i % max(1, n_movie // 10) == 0:
                print(f"[DEBUG] Movie frame {i+1}/{n_movie} at t={ts[it]:.3f}")

        Az_movie = np.array(Az_movie)
        vmin_m = Az_movie.min()
        vmax_m = Az_movie.max()

        figm, axm = plt.subplots(figsize=(5, 4))
        im = axm.pcolormesh(X, Y, Az_movie[0], shading="auto",
                            vmin=vmin_m, vmax=vmax_m)
        cbar = figm.colorbar(im)
        cbar.set_label(r"$A_z(x,y,z=0)$")
        axm.set_xlabel(r"$x$")
        axm.set_ylabel(r"$y$")
        axm.set_title(fr"$A_z(x,y,z=0,t)$, $t={ts[idx_movie[0]]:.2f}$")
        axm.set_aspect("equal")

        def update(frame_idx):
            it = idx_movie[frame_idx]
            im.set_array(Az_movie[frame_idx].ravel())
            axm.set_title(fr"$A_z(x,y,z=0,t)$, $t={ts[it]:.2f}$")
            return im,

        anim = FuncAnimation(figm, update, frames=n_movie, blit=False)

        movie_path = os.path.join(args.outdir, args.movie_fname)
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=15, bitrate=1800)
            anim.save(movie_path, writer=writer)
            print(f"[SAVE] Movie saved to {movie_path}")
        except Exception as e:
            print("[ERROR] Could not write movie with FFMpegWriter.")
            print("       Make sure ffmpeg is installed and on your PATH.")
            print(f"       Error: {e}")

    print("[DONE] Nonlinear tearing analysis finished.")


if __name__ == "__main__":
    main()
