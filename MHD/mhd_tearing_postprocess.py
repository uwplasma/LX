#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_postprocess.py

Post-process saved MHD tearing-mode solutions produced by
mhd_tearing_solve.py and generate publication-ready diagnostics:

  - Energies and dissipation (E_kin, E_mag, E_tot, E_cons)
  - Tearing-mode amplitudes (RMS Bx and |Bx(kx=0,ky=1,kz=0)|)
  - Robust linear-phase fit for gamma (vs FKR gamma)
  - Energy invariant error
  - Reconnected flux and reconnection rate (X-point based if available)
  - Shell spectra E_B(k_perp), E_v(k_perp) at selected times
  - PDFs of Jz at selected times, with kurtosis
  - Snapshots of Bx, Jz, and A_z field lines
  - Movies of Bx, Jz, and Jz+flux contours
  - Current-sheet thickness vs time
  - Plasmoid/island count vs time
  - Jz kurtosis vs time
  - Dimensionless reconnection rate
  - Amplitude–flux & reconnection phase-space plots
  - Plasmoid/intermittency correlation plots

The script accepts a glob pattern and processes all matching .npz files,
placing outputs into separate folders:

    figures_<stem>/

where <stem> is the filename without extension, e.g.:

    mhd_tearing_solution_original.npz -> figures_mhd_tearing_solution_original
"""

from __future__ import annotations

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

plt.rcParams.update({
    "font.size": 13,
    "text.usetex": False,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 220,
    "axes.linewidth": 1.1,
    "lines.linewidth": 1.8,
})

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from mhd_tearing_solve import (
    make_k_arrays, make_dealias_mask, project_div_free,
    energy_from_hat, dissipation_rates,
    curl_from_hat, compute_Az_from_hat, grad_from_hat,
    count_local_extrema_1d
)

# -----------------------------------------------------------------------------#
# Linear regression helper (used for spectra + old gamma fallback)
# -----------------------------------------------------------------------------#

def _linear_regression_with_stats(t, y):
    """
    Simple least-squares linear regression y ~ a t + b with R^2.

    Returns
    -------
    a, b, R2, y_fit
    """
    t = np.asarray(t)
    y = np.asarray(y)
    A = np.vstack([t, np.ones_like(t)]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeffs
    y_fit = a * t + b
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, R2, y_fit

# -----------------------------------------------------------------------------#
# Robust automatic linear-window selector (fallback when gamma not saved)
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

    lnw = np.log(w + 1e-30)  # small offset for safety
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
        win = idx[s:s + nwin]
        t_win = ts[win]
        y_win = lnw[win]
        a, b, R2, _ = _linear_regression_with_stats(t_win, y_win)
        if R2 >= R2_min and R2 > best_R2:
            best_R2 = R2
            best_slice = win

    # Second pass: if nothing reached R2_min, just pick max R²
    if best_slice is None:
        for s in range(0, idx.size - nwin + 1):
            win = idx[s:s + nwin]
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
# Fallback tearing amplitude (for old .npz without tearing_amp_series)
# -----------------------------------------------------------------------------#

def tearing_amplitude_np(B_hat, Lx, Ly, Lz, band_width_frac=0.25):
    """
    RMS of Bx in a band around the current sheet (|x-Lx/2| < band_width_frac*Lx/2).

    Pure NumPy version for post-processing / backward compatibility.
    """
    B = np.fft.ifftn(B_hat, axes=(1, 2, 3)).real
    Bx = B[0]

    Nx = Bx.shape[0]
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    xc = 0.5 * Lx
    band_half = band_width_frac * 0.5 * Lx

    mask = (np.abs(x - xc)[:, None, None] < band_half)
    Bx_band = np.where(mask, Bx, 0.0)

    num = np.sum(Bx_band**2)
    den = np.sum(mask.astype(float)) + 1e-16
    rms = np.sqrt(num / den)
    return rms

# -----------------------------------------------------------------------------#
# Shell spectrum helper
# -----------------------------------------------------------------------------#

def compute_shell_spectrum(kx, ky, kz, B_hat, v_hat, nbins=32):
    """
    Compute isotropic shell spectra E_B(k_perp), E_v(k_perp) by binning in
    k_perp = sqrt(kx^2 + ky^2).

    B_hat, v_hat: arrays (3, Nx, Ny, Nz)
    """
    kx = np.asarray(kx)
    ky = np.asarray(ky)
    k_perp = np.sqrt(kx**2 + ky**2)

    E_B_3d = 0.5 * np.sum(np.abs(B_hat) ** 2, axis=0)
    E_v_3d = 0.5 * np.sum(np.abs(v_hat) ** 2, axis=0)

    k_flat = k_perp.ravel()
    EB_flat = E_B_3d.ravel()
    Ev_flat = E_v_3d.ravel()

    kmax = k_flat.max()
    if kmax <= 0:
        kmax = 1.0

    bins = np.linspace(0.0, kmax, nbins + 1)
    EB_spec, edges = np.histogram(k_flat, bins=bins, weights=EB_flat)
    Ev_spec, _ = np.histogram(k_flat, bins=bins, weights=Ev_flat)
    k_centers = 0.5 * (edges[:-1] + edges[1:])

    return k_centers, EB_spec, Ev_spec

# -----------------------------------------------------------------------------#
# Generic movie helper
# -----------------------------------------------------------------------------#

def make_movie(
    field_slices,
    filename,
    ts,
    Lx,
    Ly,
    title,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    add_flux_contours=False,
    flux_slices=None,
    n_flux_levels=15,
):
    """
    Build a 2D movie of a mid-plane slice.

    field_slices: (Nt, Nx, Ny) array
    """
    field_slices = np.asarray(field_slices)
    n_frames, Nx, Ny = field_slices.shape

    if vmin is None or vmax is None:
        f_absmax = float(np.max(np.abs(field_slices)))
        vmin = -f_absmax
        vmax = +f_absmax

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(
        field_slices[0].T,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(title)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}, t={ts[0]:.3f}")

    cs = None
    if add_flux_contours:
        assert flux_slices is not None
        cs = ax.contour(
            flux_slices[0].T,
            levels=n_flux_levels,
            colors="k",
            linewidths=0.7,
            origin="lower",
            extent=[0, Lx, 0, Ly],
        )

    def update(i):
        nonlocal cs
        im.set_data(field_slices[i].T)
        ax.set_title(f"{title}, t={ts[i]:.3f}")

        if add_flux_contours:
            if cs is not None:
                for coll in cs.collections:
                    coll.remove()
            cs = ax.contour(
                flux_slices[i].T,
                levels=n_flux_levels,
                colors="k",
                linewidths=0.7,
                origin="lower",
                extent=[0, Lx, 0, Ly],
            )
        return (im,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=100,
        blit=False,
    )
    writer = animation.FFMpegWriter(fps=10, bitrate=2000)
    ani.save(filename, writer=writer)
    plt.close(fig)
    print(f"[MOVIE] Saved {filename}")

# -----------------------------------------------------------------------------#
# Argument parsing
# -----------------------------------------------------------------------------#

def parse_args():
    p = argparse.ArgumentParser(
        description="Post-process MHD tearing solution(s) and make diagnostics."
    )
    p.add_argument(
        "pattern",
        nargs="?",
        default="mhd_tearing*.npz",
        help="Input .npz filename or glob pattern "
             "(default: mhd_tearing*.npz)",
    )
    p.add_argument(
        "--no-make-movies",
        dest="make_movies",
        action="store_false",
        help="Do not build mp4 movies.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Extra prefix for all output figure/movie filenames.",
    )
    return p.parse_args()

# -----------------------------------------------------------------------------#
# Core per-file post-processing
# -----------------------------------------------------------------------------#

def process_one_file(infile: str, extra_prefix: str, make_movies: bool):
    data = np.load(infile, allow_pickle=True)

    ts = np.array(data["ts"])
    v_hat_frames = np.array(data["v_hat"])
    B_hat_frames = np.array(data["B_hat"])

    Nx = int(data["Nx"]); Ny = int(data["Ny"]); Nz = int(data["Nz"])
    Lx = float(data["Lx"]); Ly = float(data["Ly"]); Lz = float(data["Lz"])
    nu = float(data["nu"]); eta = float(data["eta"])
    B0 = float(data["B0"]); a = float(data["a"])
    B_g = float(data["B_g"]); eps_B = float(data["eps_B"])
    gamma_FKR = float(data.get("gamma_FKR", np.nan))
    # S may be stored under "S", "S_sheet", or "S_a"
    if "S" in data:
        S = float(data["S"])
    elif "S_sheet" in data:
        S = float(data["S_sheet"])
    elif "S_a" in data:
        S = float(data["S_a"])
    else:
        S = np.nan
    Delta_prime_a = float(data.get("Delta_prime_a", np.nan))

    # Sweet–Parker metrics (if present)
    vA = float(data["vA"]) if "vA" in data else B0  # default ρ = 1
    S_sheet = float(data.get("S_sheet", S))
    delta_SP = float(data["delta_SP"]) if "delta_SP" in data else np.nan
    v_in_SP = float(data["v_in_SP"]) if "v_in_SP" in data else np.nan
    E_SP = float(data["E_SP"]) if "E_SP" in data else np.nan

    # New diagnostic time series if present
    if "mode_amp_series" in data:
        mode_amp_arr = np.array(data["mode_amp_series"])
    else:
        mode_amp_arr = None

    if "tearing_amp_series" in data:
        tearing_amp_arr = np.array(data["tearing_amp_series"])
    else:
        tearing_amp_arr = None

    gamma_fit_saved = data.get("gamma_fit", None)
    lnA_fit_saved = data.get("lnA_fit", None)
    mask_lin_saved = data.get("mask_lin", None)

    if gamma_fit_saved is not None:
        gamma_fit = float(np.array(gamma_fit_saved))
    else:
        gamma_fit = np.nan

    Az_xpt_series = np.array(data["Az_xpt_series"]) if "Az_xpt_series" in data else None
    E_rec_series_saved = np.array(data["E_rec_series"]) if "E_rec_series" in data else None
    n_plasmoids_final_saved = int(data["n_plasmoids_final"]) if "n_plasmoids_final" in data else None

    ix0 = int(data["ix0"]); iy1 = int(data["iy1"]); iz0 = int(data["iz0"])

    eq_mode = data.get("equilibrium_mode", "original")
    if isinstance(eq_mode, np.ndarray):
        eq_mode = eq_mode.item()
    eq_mode = str(eq_mode)

    # Alfven speed (rho0 = 1)
    rho0 = 1.0
    V_A = vA  # already B0 / sqrt(rho0) in new files; fallback B0 above

    # Figures directory: figures_<stem>/
    stem = os.path.splitext(os.path.basename(infile))[0]
    fig_dir = f"figures_{stem}"
    os.makedirs(fig_dir, exist_ok=True)
    prefix = os.path.join(fig_dir, extra_prefix)

    print("=== Post-processing MHD tearing solution ===")
    print(f"infile          = {infile}")
    print(f"equilibrium_mode= {eq_mode}")
    print(f"Nx,Ny,Nz        = {Nx},{Ny},{Nz}")
    print(f"Lx,Ly,Lz        = {Lx},{Ly},{Lz}")
    print(f"nu={nu}, eta={eta}, B0={B0}, a={a}, B_g={B_g}, eps_B={eps_B}")
    print(f"S={S:.3e}, Delta' a={Delta_prime_a:.3e}, gamma_FKR={gamma_FKR:.3e}")
    print(f"Sweet–Parker: S_sheet={S_sheet:.3e}, delta_SP={delta_SP:.3e}, "
          f"v_in_SP={v_in_SP:.3e}, E_SP={E_SP:.3e}")
    print(f"Assuming rho0={rho0}, V_A={V_A:.3f}")
    print("Output figures/movies in:", fig_dir)
    print("============================================")

    # Spectral operators
    kx, ky, kz, k2, NX_arr, NY_arr, NZ_arr = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    mask_dealias = make_dealias_mask(Nx, Ny, Nz, NX_arr, NY_arr, NZ_arr)
    mid_z = Nz // 2
    iy_center = Ny // 2
    ix_center = Nx // 2

    # Diagnostics arrays
    E_kin_list = []
    E_mag_list = []
    E_tot_list = []
    eps_visc_list = []
    eps_ohm_list = []
    E_cons_list = []
    tearing_amp_list = []
    mode_amp_list = []
    v_rms_list = []
    v_max_list = []
    psi_rec_list = []   # reconnected flux Δψ on mid-plane

    # New diagnostics
    sheet_thickness_list = []  # delta(t)
    Jpeak_list = []            # peak |Jz| in central region
    island_count_list = []     # number of extrema of Az line
    Jkurt_list = []            # kurtosis of normalized Jz mid-plane

    E_cons_running = 0.0
    E_cons0 = None

    x_coords = np.linspace(0.0, Lx, Nx, endpoint=False)
    xc = 0.5 * Lx
    central_mask = np.abs(x_coords - xc) < 0.25 * Lx  # central half of the box

    for i in range(len(ts)):
        v_hat_i = jnp.array(v_hat_frames[i]) * mask_dealias
        B_hat_i = jnp.array(B_hat_frames[i]) * mask_dealias
        v_hat_i = project_div_free(v_hat_i, kx, ky, kz, k2)
        B_hat_i = project_div_free(B_hat_i, kx, ky, kz, k2)

        # Energies & dissipation
        E_kin_i, E_mag_i = energy_from_hat(v_hat_i, B_hat_i, Lx, Ly, Lz)
        eps_visc_i, eps_ohm_i = dissipation_rates(
            v_hat_i, B_hat_i, k2, nu, eta, Lx, Ly, Lz
        )
        E_tot_i = E_kin_i + E_mag_i

        E_kin_list.append(float(E_kin_i))
        E_mag_list.append(float(E_mag_i))
        E_tot_list.append(float(E_tot_i))
        eps_visc_list.append(float(eps_visc_i))
        eps_ohm_list.append(float(eps_ohm_i))

        # Tearing amplitude (RMS Bx near sheet) – use saved series if present
        if tearing_amp_arr is not None:
            A_rms = tearing_amp_arr[i]
        else:
            A_rms = tearing_amplitude_np(
                np.array(B_hat_frames[i]), Lx, Ly, Lz, band_width_frac=0.25
            )
        tearing_amp_list.append(float(A_rms))

        # Single Fourier mode amplitude – use saved series if present
        if mode_amp_arr is not None:
            mode_amp = mode_amp_arr[i]
        else:
            Bx_hat_i = np.array(B_hat_i[0])
            mode_amp = np.abs(Bx_hat_i[ix0, iy1, iz0])
        mode_amp_list.append(float(mode_amp))

        # Velocity diagnostics
        v_i = np.fft.ifftn(np.array(v_hat_i), axes=(1, 2, 3)).real
        v_mag = np.sqrt(np.sum(v_i**2, axis=0))
        v_rms_list.append(float(np.sqrt(np.mean(v_mag**2))))
        v_max_list.append(float(np.max(v_mag)))

        # Flux function Az on mid-plane
        Az_i = compute_Az_from_hat(B_hat_i, kx, ky)  # (Nx,Ny,Nz)
        Az_mid = np.array(Az_i[:, :, mid_z])
        psi_rec = float(Az_mid.max() - Az_mid.min())
        psi_rec_list.append(psi_rec)

        # Current Jz on mid-plane
        J_i = curl_from_hat(B_hat_i, kx, ky, kz)  # real-space J
        Jz_mid = np.array(J_i[2, :, :, mid_z])

        # Sheet thickness (FWHM of |Jz| near x=Lx/2, y=Ly/2)
        J_line = np.abs(Jz_mid[:, iy_center])
        J_center = J_line[central_mask]
        x_center = x_coords[central_mask]
        if J_center.size > 3 and np.max(J_center) > 0.0:
            J_peak = float(np.max(J_center))
            mask_fwhm = J_center >= 0.5 * J_peak
            if np.any(mask_fwhm):
                x_min = float(np.min(x_center[mask_fwhm]))
                x_max = float(np.max(x_center[mask_fwhm]))
                delta = 0.5 * (x_max - x_min)
            else:
                J_peak = 0.0
                delta = np.nan
        else:
            J_peak = 0.0
            delta = np.nan

        sheet_thickness_list.append(delta)
        Jpeak_list.append(J_peak)

        # Island / plasmoid count: count extrema of Az(x, y=L_y/2)
        # Az_line = Az_mid[:, iy_center]
        # dAz = np.diff(Az_line)
        # sign = np.sign(dAz)
        # n_islands = int(np.sum(sign[:-1] * sign[1:] < 0))
        n_plasmoids_final = count_local_extrema_1d(Az_mid[-1, :])
        n_islands = n_plasmoids_final // 2
        
        island_count_list.append(n_islands)

        # Jz kurtosis on mid-plane (Zhdankin-style intermittency)
        J_flat = Jz_mid.ravel()
        J_rms = np.sqrt(np.mean(J_flat**2)) + 1e-16
        J_norm = J_flat / J_rms
        kurt = float(np.mean(J_norm**4))
        Jkurt_list.append(kurt)

        # Energy invariant
        if i > 0:
            dt = ts[i] - ts[i - 1]
            eps_prev = eps_visc_list[i - 1] + eps_ohm_list[i - 1]
            eps_curr = eps_visc_list[i] + eps_ohm_list[i]
            E_cons_running += 0.5 * (eps_prev + eps_curr) * dt

        E_cons_val = float(E_tot_i + E_cons_running)
        if E_cons0 is None:
            E_cons0 = E_cons_val
        E_cons_list.append(E_cons_val)

        print(
            f"[POST] frame {i}/{len(ts) - 1}, t={ts[i]:.4f}, "
            f"E_kin={E_kin_i:.3e}, E_mag={E_mag_i:.3e}, E_tot={E_tot_i:.3e}, "
            f"eps_visc={eps_visc_i:.3e}, eps_ohm={eps_ohm_i:.3e}, "
            f"E_cons={E_cons_val:.3e}, A_tearing={A_rms:.3e}, "
            f"delta/a={delta/a if a>0 else np.nan:.3e}, islands={n_islands}"
        )

    ts_np = ts
    E_kin_arr = np.array(E_kin_list)
    E_mag_arr = np.array(E_mag_list)
    E_tot_arr = np.array(E_tot_list)
    eps_visc_arr = np.array(eps_visc_list)
    eps_ohm_arr = np.array(eps_ohm_list)
    E_cons_arr = np.array(E_cons_list)
    tearing_amp_arr = np.array(tearing_amp_list) if tearing_amp_arr is None else tearing_amp_arr
    mode_amp_arr = np.array(mode_amp_list) if mode_amp_arr is None else mode_amp_arr
    v_rms_arr = np.array(v_rms_list)
    v_max_arr = np.array(v_max_list)
    psi_rec_arr = np.array(psi_rec_list)
    rel_E_cons_err = (E_cons_arr - E_cons_arr[0]) / E_cons_arr[0]

    sheet_thickness_arr = np.array(sheet_thickness_list)
    Jpeak_arr = np.array(Jpeak_list)
    island_count_arr = np.array(island_count_list)
    Jkurt_arr = np.array(Jkurt_list)

    # ---------------------------------------------------------------------#
    # 1) Energy and dissipation plots (log scale)
    # ---------------------------------------------------------------------#
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axs

    epsE = 1e-12 * E_tot_arr.max()
    epsD = 1e-12 * (eps_visc_arr + eps_ohm_arr).max()

    ax1.semilogy(ts_np, E_kin_arr + epsE, label=r"$E_{\rm kin}$")
    ax1.semilogy(ts_np, E_mag_arr + epsE, label=r"$E_{\rm mag}$")
    ax1.semilogy(ts_np, E_tot_arr + epsE, "--", label=r"$E_{\rm tot}$")
    ax1.semilogy(ts_np, E_cons_arr + epsE, "-.", label=r"$E_{\rm cons}$")
    ax1.set_xlabel("t")
    ax1.set_ylabel("Energy")
    ax1.set_title("MHD energies and invariant (log scale)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.semilogy(ts_np, eps_visc_arr + epsD, label=r"$\epsilon_{\rm visc}$")
    ax2.semilogy(ts_np, eps_ohm_arr + epsD, label=r"$\epsilon_{\rm ohm}$")
    ax2.semilogy(ts_np, eps_visc_arr + eps_ohm_arr + epsD, "--",
                 label=r"$\epsilon_{\rm tot}$")
    ax2.set_xlabel("t")
    ax2.set_ylabel("Dissipation rate")
    ax2.set_title("Dissipation rates vs time (log scale)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(prefix + "mhd_energy_invariants.png", bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Diagnostics saved to "
          f"{prefix}mhd_energy_invariants.png")

    # ---------------------------------------------------------------------#
    # 2) Tearing mode & velocity scales (normalized by V_A)
    # ---------------------------------------------------------------------#
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(ts_np, mode_amp_arr)
    axs[0].set_xlabel("t")
    axs[0].set_ylabel(r"$|B_x(k_x=0,k_y=1,k_z=0)|$")
    axs[0].set_title("Tearing-mode Fourier amplitude")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(ts_np, v_rms_arr / V_A, label=r"$v_{\rm rms}/V_A$")
    axs[1].plot(ts_np, v_max_arr / V_A, "--", label=r"$v_{\max}/V_A$")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel(r"Velocity / $V_A$")
    axs[1].set_title("Reconnection outflow speeds")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    fig.tight_layout()
    fig.savefig(prefix + "tearing_mode_velocity_scales.png",
                bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved "
          f"{prefix}tearing_mode_velocity_scales.png")

    # ---------------------------------------------------------------------#
    # 3) Tearing diagnostics with robust linear fit (using saved gamma if any)
    # ---------------------------------------------------------------------#
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    A_rms0 = tearing_amp_arr[0] if tearing_amp_arr[0] != 0 else 1.0
    A_mode0 = mode_amp_arr[0] if mode_amp_arr[0] != 0 else 1.0

    axs[0].plot(ts_np, tearing_amp_arr / A_rms0,
                label=r"${\rm RMS}\,B_x / ({\rm RMS}\,B_x)_0$")
    axs[0].plot(ts_np, mode_amp_arr / A_mode0, '--',
                label=r"$|B_x(k_x=0,k_y=1,k_z=0)| / |B_x|_0$")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("Normalized amplitude")
    axs[0].set_title("Tearing amplitude (normalized)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    log_mode = np.log(mode_amp_arr + 1e-30)

    # Linear-phase window & fit:
    if mask_lin_saved is not None:
        mask_lin = np.array(mask_lin_saved, dtype=bool)
    else:
        mask_lin = select_linear_window(ts_np, mode_amp_arr, w0=mode_amp_arr[0])
    idx_lin = np.where(mask_lin)[0]
    if idx_lin.size < 3:
        idx_lin = np.arange(2, max(5, len(ts_np) // 3))

    i0, i1 = int(idx_lin[0]), int(idx_lin[-1])
    t_fit = ts_np[idx_lin]
    logA_fit = log_mode[idx_lin]

    if np.isnan(gamma_fit):
        # Fallback: re-fit if not saved
        gamma_fit_local, b_fit, R2_fit, _ = _linear_regression_with_stats(t_fit, logA_fit)
        logA_line = b_fit + gamma_fit_local * ts_np
        gamma_fit_plot = gamma_fit_local
        R2_fit_plot = R2_fit
    else:
        gamma_fit_plot = gamma_fit
        if lnA_fit_saved is not None:
            logA_line = np.array(lnA_fit_saved)
        else:
            # reconstruct line
            a_tmp, b_tmp, R2_tmp, _ = _linear_regression_with_stats(t_fit, logA_fit)
            logA_line = b_tmp + a_tmp * ts_np
        # estimate R² for display
        _, _, R2_fit_plot, _ = _linear_regression_with_stats(t_fit, logA_fit)

    print(f"[FIT] Measured tearing gamma ≈ {gamma_fit_plot:.3e}, "
          f"R^2 ≈ {R2_fit_plot:.4f}")
    if not np.isnan(gamma_FKR):
        ratio = gamma_fit_plot / gamma_FKR
        print(f"[COMP] gamma_fit/gamma_FKR ≈ {ratio:.3f}")
    else:
        ratio = np.nan

    axs[1].plot(ts_np, log_mode, label=r"$\ln|B_x(k_x=0,k_y=1)|$")
    axs[1].axvspan(ts_np[i0], ts_np[i1], color="grey", alpha=0.2,
                   label="linear fit window")
    axs[1].plot(ts_np, logA_line, "k--",
                label=rf"fit: $\gamma \approx {gamma_fit_plot:.3e}$")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel(r"$\ln |B_x(k_x=0,k_y=1)|$")
    axs[1].set_title("Mode growth (linear phase shaded)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best")

    if not np.isnan(gamma_FKR):
        txt = (
            rf"$\gamma_\mathrm{{fit}} \approx {gamma_fit_plot:.3e}$" + "\n"
            rf"$\gamma_\mathrm{{FKR}} \approx {gamma_FKR:.3e}$" + "\n"
            rf"$\gamma_\mathrm{{fit}}/\gamma_\mathrm{{FKR}} \approx {ratio:.2f}$"
        )
        axs[1].text(
            0.05, 0.05, txt,
            transform=axs[1].transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    axs[2].plot(ts_np, rel_E_cons_err)
    axs[2].set_xlabel("t")
    axs[2].set_ylabel(
        r"$(E_{\rm cons}-E_{\rm cons}(0))/E_{\rm cons}(0)$"
    )
    axs[2].set_title("Energy-invariant relative error")
    axs[2].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(prefix + "tearing_mode_diagnostics.png",
                bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved "
          f"{prefix}tearing_mode_diagnostics.png")

    # ---------------------------------------------------------------------#
    # 4) Reconnected flux and reconnection rate
    #     Prefer E_rec_series from solver (X-point based), fallback to dΔψ/dt
    # ---------------------------------------------------------------------#
    if E_rec_series_saved is not None:
        recon_rate_raw = E_rec_series_saved
    else:
        recon_rate_raw = np.gradient(psi_rec_arr, ts_np)

    # Simple moving-average smoothing for reconnection rate
    window = 5 if len(ts_np) >= 5 else 3
    kernel = np.ones(window) / window
    recon_rate_smooth = np.convolve(recon_rate_raw, kernel, mode="same")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))

    axs[0].plot(ts_np, psi_rec_arr)
    axs[0].axvspan(ts_np[i0], ts_np[i1], color="grey", alpha=0.15,
                   label="linear phase")
    axs[0].axvline(ts_np[np.argmax(tearing_amp_arr)],
                   color="k", linestyle=":", linewidth=1.5,
                   label="nonlinear peak")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel(r"$\Delta\psi = \max(A_z) - \min(A_z)$")
    axs[0].set_title("Reconnected flux on mid-plane")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best", fontsize=9)

    axs[1].plot(ts_np, recon_rate_raw, alpha=0.4,
                label=r"$E_{\rm rec}$ (raw)")
    axs[1].plot(ts_np, recon_rate_smooth, "C1",
                label=r"$E_{\rm rec}$ (smoothed)")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel(r"$E_{\rm rec}$")
    axs[1].set_title("Reconnection rate (X-point proxy)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(prefix + "reconnected_flux_and_rate.png",
                bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved "
          f"{prefix}reconnected_flux_and_rate.png")

    # ---------------------------------------------------------------------#
    # 4b) Sweet–Parker / plasmoid diagnostics:
    #     sheet thickness, dimensionless reconnection rate, island count
    # ---------------------------------------------------------------------#
    E_rec_dimless = recon_rate_smooth / (B0 * V_A + 1e-16)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(ts_np, sheet_thickness_arr / a, label=r"$\delta/a$")
    if not np.isnan(delta_SP):
        axs[0].axhline(delta_SP / a, color="k", linestyle="--",
                       label=r"Sweet–Parker $\delta_{\rm SP}/a$")
    axs[0].axvspan(ts_np[i0], ts_np[i1], color="grey", alpha=0.15)
    axs[0].set_xlabel("t")
    axs[0].set_ylabel(r"$\delta / a$")
    axs[0].set_title("Current-sheet half-thickness")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=9)

    axs[1].plot(ts_np, E_rec_dimless)
    axs[1].axhline(0.0, color="k", linewidth=1.0)
    axs[1].set_xlabel("t")
    axs[1].set_ylabel(r"$E_{\rm rec} / (B_0 V_A)$")
    axs[1].set_title("Dimensionless reconnection rate")
    axs[1].grid(True, alpha=0.3)

    axs[2].step(ts_np, island_count_arr, where="mid")
    axs[2].set_xlabel("t")
    axs[2].set_ylabel("Island count")
    axs[2].set_title("Magnetic islands on mid-plane")
    axs[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(prefix + "sheet_and_plasmoids.png",
                bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved "
          f"{prefix}sheet_and_plasmoids.png")

    # ---------------------------------------------------------------------#
    # 5) Shell spectra at selected times (Boldyrev-style)
    # ---------------------------------------------------------------------#
    i_lin_mid = int(0.5 * (i0 + i1))
    i_peak = int(np.argmax(tearing_amp_arr))
    i_final = len(ts_np) - 1
    spec_indices = sorted(set([i_lin_mid, i_peak, i_final]))
    spec_labels = []
    for idx in spec_indices:
        if idx == i_lin_mid:
            spec_labels.append("linear")
        elif idx == i_peak:
            spec_labels.append("nonlinear peak")
        else:
            spec_labels.append("late")

    fig, ax = plt.subplots(figsize=(6.5, 5))

    k_shell_last = None
    EB_spec_last = None

    for idx, lab in zip(spec_indices, spec_labels):
        B_hat_i = np.array(B_hat_frames[idx])
        v_hat_i = np.array(v_hat_frames[idx])
        k_shell, EB_spec, Ev_spec = compute_shell_spectrum(
            kx, ky, kz, B_hat_i, v_hat_i, nbins=32
        )
        k_shell_last = k_shell
        EB_spec_last = EB_spec

        mask_pos = (k_shell > 0) & (EB_spec > 0) & (Ev_spec > 0)
        k_plot = k_shell[mask_pos]
        EB_plot = EB_spec[mask_pos]
        Ev_plot = Ev_spec[mask_pos]

        ax.loglog(k_plot, EB_plot,
                  label=fr"$E_B$, {lab}, $t={ts_np[idx]:.1f}$")
        ax.loglog(k_plot, Ev_plot, '--',
                  label=fr"$E_v$, {lab}, $t={ts_np[idx]:.1f}$")

        # Inertial-range fit for magnetic spectrum at nonlinear peak
        if idx == i_peak and k_plot.size > 8:
            kmin_fit = k_plot[int(0.2 * len(k_plot))]
            kmax_fit = k_plot[int(0.7 * len(k_plot))]
            mask_fit = (k_plot >= kmin_fit) & (k_plot <= kmax_fit)
            if np.count_nonzero(mask_fit) >= 5:
                logk = np.log(k_plot[mask_fit])
                logE = np.log(EB_plot[mask_fit])
                a_s, b_s, R2_s, _ = _linear_regression_with_stats(logk, logE)
                slope = a_s
                k0 = np.exp(np.mean(logk))
                E0 = np.exp(np.mean(logE))
                k_line = np.array([kmin_fit, kmax_fit])
                E_line = E0 * (k_line / k0) ** slope
                ax.loglog(k_line, E_line, 'k-',
                          linewidth=2.0,
                          label=rf"peak fit: $E_B \propto k^{{{slope:.2f}}}$")

    if (k_shell_last is not None) and np.any(k_shell_last > 0):
        k_shell = k_shell_last
        EB_spec = EB_spec_last
        k_ref = np.median(k_shell[k_shell > 0])
        E_ref = np.max(EB_spec) if np.any(EB_spec > 0) else 1.0
        k_line_ref = np.array([k_ref / 4, k_ref * 4])
        line_32 = E_ref * (k_line_ref / k_ref) ** (-3.0 / 2.0)
        line_53 = E_ref * (k_line_ref / k_ref) ** (-5.0 / 3.0)
        ax.loglog(k_line_ref, line_32, 'k:', label=r"$k^{-3/2}$ ref")
        ax.loglog(k_line_ref, line_53, 'k-.', label=r"$k^{-5/3}$ ref")

    ax.set_xlabel(r"$k_\perp$")
    ax.set_ylabel(r"$E(k_\perp)$ (arb. units)")
    ax.set_title("Shell spectra $E_B$, $E_v$ vs $k_\perp$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(prefix + "spectra_shell_kperp.png",
                bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved "
          f"{prefix}spectra_shell_kperp.png")

    # ---------------------------------------------------------------------#
    # 6) PDFs of Jz at selected times (Zhdankin-style intermittency)
    # ---------------------------------------------------------------------#
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for idx, lab in zip(spec_indices, spec_labels):
        B_hat_i = np.array(B_hat_frames[idx])
        Bx_hat_i, By_hat_i, Bz_hat_i = B_hat_i[0], B_hat_i[1], B_hat_i[2]
        dBy_dx, dBy_dy, dBy_dz = grad_from_hat(
            jnp.array(By_hat_i), kx, ky, kz
        )
        dBx_dx, dBx_dy, dBx_dz = grad_from_hat(
            jnp.array(Bx_hat_i), kx, ky, kz
        )
        Jz_i = (dBy_dx - dBx_dy).astype(np.float64)[:, :, mid_z]

        J_flat = Jz_i.ravel()
        J_rms = np.sqrt(np.mean(J_flat**2))
        if J_rms == 0:
            continue
        J_norm = J_flat / J_rms

        Jmax = max(4.0, 3.0 * np.std(J_norm))
        bins = np.linspace(-Jmax, Jmax, 80)
        hist, edges = np.histogram(J_norm, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        kurt = float(np.mean(J_norm**4))

        ax.semilogy(centers, hist,
                    label=fr"{lab}, $t={ts_np[idx]:.1f}$, "
                          fr"$\kappa \approx {kurt:.1f}$")

    xg = np.linspace(-4, 4, 200)
    pdf_gauss = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * xg**2)
    ax.semilogy(xg, pdf_gauss, 'k--', label="Gaussian N(0,1)")

    ax.set_xlabel(r"$J_z / J_{\rm rms}$")
    ax.set_ylabel("PDF")
    ax.set_title(r"PDFs of $J_z$ on mid-plane")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(prefix + "Jz_PDFs.png",
                bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved "
          f"{prefix}Jz_PDFs.png")

    # ---------------------------------------------------------------------#
    # 6b) Jz kurtosis vs time
    # ---------------------------------------------------------------------#
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.semilogy(ts_np, Jkurt_arr, label=r"$\kappa(J_z)$")
    ax.axhline(3.0, color="k", linestyle="--", label="Gaussian (3)")
    ax.set_xlabel("t")
    ax.set_ylabel("kurtosis")
    ax.set_title(r"$J_z$ kurtosis on mid-plane vs time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(prefix + "Jz_kurtosis_vs_time.png", bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved "
          f"{prefix}Jz_kurtosis_vs_time.png")

    # ---------------------------------------------------------------------#
    # 7) Snapshots: Bx, Jz, Az field lines (with consistent color limits)
    # ---------------------------------------------------------------------#
    idxs = [0, len(ts_np) // 2, len(ts_np) - 1]
    labels = [f"t = {ts_np[i]:.2f}" for i in idxs]

    Bx_list = []
    Jz_list = []
    Az_list = []

    for i in idxs:
        B_hat_i = np.array(B_hat_frames[i])
        B_i = np.fft.ifftn(B_hat_i, axes=(1, 2, 3)).real
        Bx = B_i[0, :, :, mid_z]

        J_i = curl_from_hat(jnp.array(B_hat_frames[i]),
                            kx, ky, kz)
        J_i = np.array(J_i)
        Jz = J_i[2, :, :, mid_z]

        Az = compute_Az_from_hat(jnp.array(B_hat_frames[i]),
                                 kx, ky)
        Az = np.array(Az[:, :, mid_z])

        Bx_list.append(Bx)
        Jz_list.append(Jz)
        Az_list.append(Az)

    Bx_absmax = max(np.max(np.abs(Bx)) for Bx in Bx_list) + 1e-15
    Jz_absmax = max(np.max(np.abs(Jz)) for Jz in Jz_list) + 1e-15

    fig, axs = plt.subplots(len(idxs), 3,
                            figsize=(11, 3.8 * len(idxs)))
    if len(idxs) == 1:
        axs = np.array([axs])

    for row, (Bx, Jz, Az, lab) in enumerate(zip(Bx_list, Jz_list, Az_list, labels)):
        im0 = axs[row, 0].imshow(
            Bx.T,
            origin="lower",
            extent=[0, Lx, 0, Ly],
            aspect="equal",
            cmap="RdBu_r",
            vmin=-Bx_absmax,
            vmax=+Bx_absmax,
        )
        axs[row, 0].set_title(r"$B_x(x,y,z=0)$, " + lab)
        axs[row, 0].set_ylabel("y")
        fig.colorbar(im0, ax=axs[row, 0])

        im1 = axs[row, 1].imshow(
            Jz.T,
            origin="lower",
            extent=[0, Lx, 0, Ly],
            aspect="equal",
            cmap="RdBu_r",
            vmin=-Jz_absmax,
            vmax=+Jz_absmax,
        )
        axs[row, 1].set_title(r"$J_z(x,y,z=0)$")
        fig.colorbar(im1, ax=axs[row, 1])

        cs = axs[row, 2].contour(
            Az.T,
            levels=25,
            extent=[0, Lx, 0, Ly],
        )
        axs[row, 2].set_title(r"$A_z(x,y,z=0)$ (field lines)")
        axs[row, 2].set_xlim(0, Lx)
        axs[row, 2].set_ylim(0, Ly)
        axs[row, 2].set_aspect("equal")

        if row == len(idxs) - 1:
            for c in axs[row, :]:
                c.set_xlabel("x")

    fig.tight_layout()
    fig.savefig(prefix + "tearing_snapshots.png",
                bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved "
          f"{prefix}tearing_snapshots.png")

    # ---------------------------------------------------------------------#
    # 8) NEW (1/2): Amplitude–flux & reconnection phase-space (literature style)
    # ---------------------------------------------------------------------#
    # Following modern tearing/reconnection papers, show:
    #   (a) Δψ vs tearing amplitude (normalized)
    #   (b) E_rec/(B0 V_A) vs tearing amplitude (normalized)
    E_rec_dimless_full = recon_rate_raw / (B0 * V_A + 1e-16)
    A_norm = tearing_amp_arr / A_rms0
    psi_norm = psi_rec_arr / (B0 * a + 1e-16)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))

    axs[0].loglog(
        A_norm, psi_norm, "o-",
        ms=3,
        label="trajectory"
    )
    axs[0].set_xlabel(r"${\rm RMS}\,B_x / ({\rm RMS}\,B_x)_0$")
    axs[0].set_ylabel(r"$\Delta\psi / (B_0 a)$")
    axs[0].set_title("Amplitude–flux relation")
    axs[0].grid(True, which="both", alpha=0.3)

    axs[1].plot(
        A_norm, E_rec_dimless_full, "o-",
        ms=3,
        label=r"$E_{\rm rec}/(B_0 V_A)$"
    )
    axs[1].set_xlabel(r"${\rm RMS}\,B_x / ({\rm RMS}\,B_x)_0$")
    axs[1].set_ylabel(r"$E_{\rm rec} / (B_0 V_A)$")
    axs[1].set_title("Reconnection vs tearing amplitude")
    axs[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(prefix + "amplitude_flux_phase_space.png",
                bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved "
          f"{prefix}amplitude_flux_phase_space.png")

    # ---------------------------------------------------------------------#
    # 9) NEW (2/2): Plasmoids vs intermittency / reconnection
    # ---------------------------------------------------------------------#
    # Inspired by plasmoid-dominated reconnection literature:
    #   (a) E_rec_dimless vs island_count
    #   (b) Jz kurtosis vs island_count
    t_norm = (ts_np - ts_np[0]) / (ts_np[-1] - ts_np[0] + 1e-16)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))

    sc0 = axs[0].scatter(
        island_count_arr, E_rec_dimless_full,
        c=t_norm, cmap="viridis", s=20, edgecolor="none"
    )
    axs[0].set_xlabel("Island count")
    axs[0].set_ylabel(r"$E_{\rm rec} / (B_0 V_A)$")
    axs[0].set_title("Reconnection vs plasmoid count")
    axs[0].grid(True, alpha=0.3)
    cbar0 = fig.colorbar(sc0, ax=axs[0])
    cbar0.set_label(r"normalized time $t/t_{\rm end}$")

    sc1 = axs[1].scatter(
        island_count_arr, Jkurt_arr,
        c=t_norm, cmap="viridis", s=20, edgecolor="none"
    )
    axs[1].axhline(3.0, color="k", linestyle="--", linewidth=1.0)
    axs[1].set_xlabel("Island count")
    axs[1].set_ylabel(r"$\kappa(J_z)$")
    axs[1].set_title("Intermittency vs plasmoid count")
    axs[1].grid(True, which="both", alpha=0.3)
    cbar1 = fig.colorbar(sc1, ax=axs[1])
    cbar1.set_label(r"normalized time $t/t_{\rm end}$")

    fig.tight_layout()
    fig.savefig(prefix + "plasmoids_vs_intermittency.png",
                bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved "
          f"{prefix}plasmoids_vs_intermittency.png")

    # ---------------------------------------------------------------------#
    # 10) Movies
    # ---------------------------------------------------------------------#
    if make_movies:
        print("[MOVIE] Building tearing-mode movies ...")
        Bx_slices = []
        Jz_slices = []
        Az_slices = []

        for i in range(len(ts_np)):
            B_hat_i = B_hat_frames[i]
            B_i = np.fft.ifftn(B_hat_i, axes=(1, 2, 3)).real

            Bx_hat_i, By_hat_i, Bz_hat_i = B_hat_i[0], B_hat_i[1], B_hat_i[2]
            dBy_dx, dBy_dy, dBy_dz = grad_from_hat(
                jnp.array(By_hat_i), kx, ky, kz
            )
            dBx_dx, dBx_dy, dBx_dz = grad_from_hat(
                jnp.array(Bx_hat_i), kx, ky, kz
            )
            Jz_i = (dBy_dx - dBx_dy).astype(np.float64)

            Az_i = compute_Az_from_hat(jnp.array(B_hat_i), kx, ky)

            Bx_slices.append(B_i[0, :, :, mid_z])
            Jz_slices.append(Jz_i[:, :, mid_z])
            Az_slices.append(np.array(Az_i[:, :, mid_z]))

        Bx_slices = np.array(Bx_slices)
        Jz_slices = np.array(Jz_slices)
        Az_slices = np.array(Az_slices)

        Bx_absmax_movie = float(np.max(np.abs(Bx_slices))) + 1e-15
        Jz_absmax_movie = float(np.max(np.abs(Jz_slices))) + 1e-15

        make_movie(
            Bx_slices,
            prefix + "mhd_tearing_Bx_xy.mp4",
            ts_np,
            Lx,
            Ly,
            title=r"$B_x(x,y,z=0)$",
            cmap="RdBu_r",
            vmin=-Bx_absmax_movie,
            vmax=Bx_absmax_movie,
        )

        make_movie(
            Jz_slices,
            prefix + "mhd_tearing_Jz_xy.mp4",
            ts_np,
            Lx,
            Ly,
            title=r"$J_z(x,y,z=0)$",
            cmap="RdBu_r",
            vmin=-Jz_absmax_movie,
            vmax=Jz_absmax_movie,
        )

        make_movie(
            Jz_slices,
            prefix + "mhd_tearing_flux_contours.mp4",
            ts_np,
            Lx,
            Ly,
            title=r"$J_z$ with flux contours",
            cmap="RdBu_r",
            vmin=-Jz_absmax_movie,
            vmax=Jz_absmax_movie,
            add_flux_contours=True,
            flux_slices=Az_slices,
            n_flux_levels=15,
        )

        print("[DONE] Movie generation complete.")

# -----------------------------------------------------------------------------#
# Main: loop over all matching files
# -----------------------------------------------------------------------------#

def main():
    args = parse_args()
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"[WARN] No files matched pattern {args.pattern!r}")
        return

    print(f"[INFO] Found {len(files)} file(s):")
    for f in files:
        print("  ", f)

    for f in files:
        print("\n==============================================")
        print(f"[INFO] Processing file: {f}")
        print("==============================================")
        process_one_file(f, args.prefix, args.make_movies)

if __name__ == "__main__":
    main()
