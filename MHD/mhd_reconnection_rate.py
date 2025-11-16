#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_reconnection_rate.py

Publication-ready reconnection diagnostics for the Harris-sheet tearing run.

Reads solutions produced by mhd_tearing_solve.py (.npz) and computes:

  - Reconnection electric field at the X-point:
        E_rec = E_z(x_X, y_X, z_X)
              = - (v × B)_z + η J_z
  - Split into inductive and Ohmic parts:
        E_ind = - (v × B)_z
        E_ohm = η J_z
  - Tearing amplitude (RMS B_x near the current sheet)

Outputs (per run):
  1) E_rec(t), E_ind(t), E_ohm(t) and RMS B_x(t)
  2) Relative magnitudes |E_ind|/(|E_ind|+|E_ohm|),
                          |E_ohm|/(|E_ind|+|E_ohm|)
  3) ln|E_rec| vs t with linear-phase window determined from RMS B_x,
     including:
        - fitted γ_B from RMS B_x
        - FKR γ_FKR estimate for comparison
  4) A movie showing J_z(x,y,z=0) with contours of E_z and the X-point.

If multiple runs are provided, also produces:
  - Peak |E_rec| vs Lundquist number S
  - Peak |E_rec| vs reversing field amplitude B0
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from mhd_tearing_solve import (
    make_k_arrays, project_div_free, curl_from_hat, tearing_amplitude
)

plt.rcParams.update({
    "font.size": 12,
    "text.usetex": False,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
})

# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#

def find_xpoint_indices(Nx, Ny, Nz, Lx, Ly, Lz):
    """
    Approximate X-point location for Harris sheet:
      x = Lx/2, y = 0, z = 0
    Returns indices (ix0, iy0, iz0).
    """
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    z = np.linspace(0.0, Lz, Nz, endpoint=False)

    ix0 = int(np.argmin(np.abs(x - 0.5 * Lx)))
    iy0 = int(np.argmin(np.abs(y - 0.0)))
    iz0 = int(np.argmin(np.abs(z - 0.0)))
    return ix0, iy0, iz0


def patch_average(field, ix0, iy0, iz0, half_width=1):
    """
    Average 'field' over a small cubic patch around (ix0,iy0,iz0) with size
    (2*half_width+1)^3 using periodic indexing.
    """
    Nx, Ny, Nz = field.shape
    xs = [(ix0 + dx) % Nx for dx in range(-half_width, half_width + 1)]
    ys = [(iy0 + dy) % Ny for dy in range(-half_width, half_width + 1)]
    zs = [(iz0 + dz) % Nz for dz in range(-half_width, half_width + 1)]

    vals = []
    for i in xs:
        for j in ys:
            for k in zs:
                vals.append(field[i, j, k])
    return float(np.mean(vals))


def compute_reconnection_time_series(ts,
                                     v_hat_frames,
                                     B_hat_frames,
                                     Nx, Ny, Nz,
                                     Lx, Ly, Lz,
                                     eta):
    """
    For a given run, compute the reconnection electric field at the X-point
    as a function of time.

    Returns:
        E_rec_t      (ntimes,)
        E_ind_t      (ntimes,)
        E_ohm_t      (ntimes,)
        A_tearing_t  (ntimes,)  # band-averaged tearing amplitude
    """
    kx, ky, kz, k2, NX_arr, NY_arr, NZ_arr = make_k_arrays(
        Nx, Ny, Nz, Lx, Ly, Lz
    )

    ix0, iy0, iz0 = find_xpoint_indices(Nx, Ny, Nz, Lx, Ly, Lz)

    E_rec_list = []
    E_ind_list = []
    E_ohm_list = []
    A_tearing_list = []

    for ti, v_hat_i_np, B_hat_i_np in zip(ts, v_hat_frames, B_hat_frames):
        # Convert to jax for operators that rely on jnp
        v_hat_i = jnp.array(v_hat_i_np)
        B_hat_i = jnp.array(B_hat_i_np)

        # Ensure divergence-free (safety)
        v_hat_i = project_div_free(v_hat_i, kx, ky, kz, k2)
        B_hat_i = project_div_free(B_hat_i, kx, ky, kz, k2)

        # Real-space fields
        v_i = np.fft.ifftn(np.array(v_hat_i), axes=(1, 2, 3)).real
        B_i = np.fft.ifftn(np.array(B_hat_i), axes=(1, 2, 3)).real

        vx, vy, vz = v_i[0], v_i[1], v_i[2]
        Bx, By, Bz = B_i[0], B_i[1], B_i[2]

        # Current density J = ∇×B and J_z
        J_i = curl_from_hat(B_hat_i, kx, ky, kz)
        J_i_np = np.array(J_i)
        Jz = J_i_np[2]

        # (v × B)_z = v_x B_y - v_y B_x
        v_cross_B_z = vx * By - vy * Bx

        # Ohmic + inductive contribution at each point
        E_ind_field = -v_cross_B_z
        E_ohm_field = eta * Jz
        Ez_field    = E_ind_field + E_ohm_field

        # Patch-average at the X-point
        E_ind_rec = patch_average(E_ind_field, ix0, iy0, iz0, half_width=1)
        E_ohm_rec = patch_average(E_ohm_field, ix0, iy0, iz0, half_width=1)
        Ez_rec    = patch_average(Ez_field,    ix0, iy0, iz0, half_width=1)

        E_ind_list.append(E_ind_rec)
        E_ohm_list.append(E_ohm_rec)
        E_rec_list.append(Ez_rec)

        # Tearing amplitude (for linear-phase and plotting)
        A_rms = tearing_amplitude(B_hat_i, Lx, Ly, Lz, band_width_frac=0.25)
        A_tearing_list.append(A_rms)

    return (np.array(E_rec_list),
            np.array(E_ind_list),
            np.array(E_ohm_list),
            np.array(A_tearing_list))


def compute_movie_slices(ts,
                         v_hat_frames,
                         B_hat_frames,
                         Nx, Ny, Nz,
                         Lx, Ly, Lz,
                         eta):
    """
    Precompute mid-plane slices of J_z and E_z for movie.

    Returns:
        Jz_slices  (ntimes, Nx, Ny)
        Ez_slices  (ntimes, Nx, Ny)
    """
    kx, ky, kz, k2, NX_arr, NY_arr, NZ_arr = make_k_arrays(
        Nx, Ny, Nz, Lx, Ly, Lz
    )

    mid_z = Nz // 2
    Jz_list = []
    Ez_list = []

    for v_hat_i_np, B_hat_i_np in zip(v_hat_frames, B_hat_frames):
        v_hat_i = jnp.array(v_hat_i_np)
        B_hat_i = jnp.array(B_hat_i_np)

        v_hat_i = project_div_free(v_hat_i, kx, ky, kz, k2)
        B_hat_i = project_div_free(B_hat_i, kx, ky, kz, k2)

        v_i = np.fft.ifftn(np.array(v_hat_i), axes=(1, 2, 3)).real
        B_i = np.fft.ifftn(np.array(B_hat_i), axes=(1, 2, 3)).real

        vx, vy, vz = v_i[0], v_i[1], v_i[2]
        Bx, By, Bz = B_i[0], B_i[1], B_i[2]

        # J_z from curl(B)
        J_i = curl_from_hat(B_hat_i, kx, ky, kz)
        J_i_np = np.array(J_i)
        Jz = J_i_np[2]

        # E_z
        v_cross_B_z = vx * By - vy * Bx
        E_ind = -v_cross_B_z
        E_ohm = eta * Jz
        Ez = E_ind + E_ohm

        Jz_list.append(Jz[:, :, mid_z])
        Ez_list.append(Ez[:, :, mid_z])

    return np.array(Jz_list), np.array(Ez_list)


def fit_gamma_from_rmsBx(ts, A_tearing):
    """
    Fit linear growth rate gamma_B from RMS Bx(t) in the usual way:
      - determine linear phase via heuristic on amplitude
      - fit log A(t) in that window.

    Returns:
        gamma_B, (i0, i1) indices for the fit window.
    """
    A = np.array(A_tearing)
    A0 = A[1] if A.shape[0] > 1 else A[0]
    Amax = A.max()

    # heuristic: between 5× initial amplitude and 30% of saturation
    f_min = 5.0
    f_max = 0.30
    mask = (A > f_min * A0) & (A < f_max * Amax)

    idx_lin = np.where(mask)[0]
    if idx_lin.size < 3:
        # fallback: early third of the time series (excluding first two points)
        idx_lin = np.arange(2, max(5, len(ts) // 3))

    i0, i1 = int(idx_lin[0]), int(idx_lin[-1])

    t_fit = ts[i0:i1 + 1]
    logA_fit = np.log(A[i0:i1 + 1] + 1e-32)

    coeffs = np.polyfit(t_fit, logA_fit, 1)
    gamma_B = coeffs[0]
    return gamma_B, (i0, i1)


# -----------------------------------------------------------------------------#
# Movie helper (no deprecated collections API)
# -----------------------------------------------------------------------------#

from matplotlib import animation


def make_reconnection_movie(Jz_slices,
                            Ez_slices,
                            ts,
                            Lx, Ly,
                            ix0, iy0,
                            E_rec_t,
                            filename):
    """
    Movie of J_z(x,y,z=0) with contours of E_z and a marker at the X-point.
    Title shows time and E_rec(t).
    """
    Jz_slices = np.asarray(Jz_slices)
    Ez_slices = np.asarray(Ez_slices)
    n_frames, Nx, Ny = Jz_slices.shape

    vmin = float(Jz_slices.min())
    vmax = float(Jz_slices.max())

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)

    im = ax.imshow(
        Jz_slices[0].T,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r",
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$J_z(x,y,z=0)$")

    # initial contours of E_z
    cs = ax.contour(
        Ez_slices[0].T,
        levels=15,
        colors="k",
        linewidths=0.7,
        origin="lower",
        extent=[0, Lx, 0, Ly],
    )

    # X-point marker
    x_vals = np.linspace(0.0, Lx, Nx, endpoint=False)
    y_vals = np.linspace(0.0, Ly, Ny, endpoint=False)
    xX = x_vals[ix0]
    yX = y_vals[iy0]
    marker, = ax.plot([xX], [yX], "ko", markersize=4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        rf"$J_z$ and $E_z$ contours, t={ts[0]:.3f}, "
        rf"$E_{{\rm rec}}\approx{E_rec_t[0]:.2e}$"
    )

    def update(i):
        nonlocal cs
        im.set_data(Jz_slices[i].T)

        # remove previous contour set (new Matplotlib API)
        if cs is not None:
            cs.remove()
        cs = ax.contour(
            Ez_slices[i].T,
            levels=15,
            colors="k",
            linewidths=0.7,
            origin="lower",
            extent=[0, Lx, 0, Ly],
        )

        ax.set_title(
            rf"$J_z$ and $E_z$ contours, t={ts[i]:.3f}, "
            rf"$E_{{\rm rec}}\approx{E_rec_t[i]:.2e}$"
        )
        return (im, cs, marker)

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
# CLI and main
# -----------------------------------------------------------------------------#

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute reconnection E_z at X-point from MHD runs."
    )
    p.add_argument("runs", nargs="?", default=["mhd_tearing_solution.npz"],
                        help="Input .npz file produced by mhd_tearing_solve.py")
    p.add_argument("--prefix", type=str, default="",
                   help="Prefix for output plots/movies.")
    
    return p.parse_args()


def main():
    args = parse_args()
    run_files = args.runs
    prefix = args.prefix

    # Arrays for multi-run scan
    S_list = []
    B0_list = []
    Emax_list = []

    for run_idx, fname in enumerate(run_files):
        print(f"=== Processing run {run_idx+1}/{len(run_files)}: {fname} ===")
        data = np.load(fname, allow_pickle=True)

        ts = np.array(data["ts"])
        v_hat_frames = np.array(data["v_hat"])
        B_hat_frames = np.array(data["B_hat"])

        Nx = int(data["Nx"]); Ny = int(data["Ny"]); Nz = int(data["Nz"])
        Lx = float(data["Lx"]); Ly = float(data["Ly"]); Lz = float(data["Lz"])
        nu = float(data["nu"]); eta = float(data["eta"])
        B0 = float(data["B0"])
        a  = float(data["a"])
        S  = float(data["S"]) if "S" in data else a * B0 / eta

        # FKR-like theoretical tearing rate
        ky_val = 2.0 * np.pi / Ly   # m_y = 1
        ka = ky_val * a
        Delta_prime_a = 2.0 * (1.0 / ka - ka)
        vA = B0  # ρ = 1
        C_fkr = 0.55
        if Delta_prime_a > 0.0:
            gamma_FKR = (
                C_fkr * vA / a *
                (Delta_prime_a ** (4.0 / 5.0)) *
                (S ** (-3.0 / 5.0))
            )
        else:
            gamma_FKR = float("nan")

        print(f"Nx,Ny,Nz = {Nx},{Ny},{Nz}")
        print(f"Lx,Ly,Lz = {Lx},{Ly},{Lz}")
        print(f"nu={nu:.3e}, eta={eta:.3e}, B0={B0:.2f}, a={a:.3e}, S={S:.3e}")
        print(f"gamma_FKR = {gamma_FKR:.3e}")

        # --- Reconnection time series -------------------------------------- #
        E_rec_t, E_ind_t, E_ohm_t, A_tearing_t = compute_reconnection_time_series(
            ts, v_hat_frames, B_hat_frames,
            Nx, Ny, Nz, Lx, Ly, Lz, eta
        )

        # --- Movie slices --------------------------------------------------- #
        Jz_slices, Ez_slices = compute_movie_slices(
            ts, v_hat_frames, B_hat_frames,
            Nx, Ny, Nz, Lx, Ly, Lz, eta
        )
        ix0, iy0, iz0 = find_xpoint_indices(Nx, Ny, Nz, Lx, Ly, Lz)

        # --- 1) E_rec, components, RMS Bx ---------------------------------- #
        fig, ax1 = plt.subplots(figsize=(6.5, 4), dpi=200)

        ax1.plot(ts, E_rec_t, "k-", label=r"$E_{\rm rec}$")
        ax1.plot(ts, E_ind_t, "b--", label=r"$E_{\rm ind} = -(\mathbf{v}\times\mathbf{B})_z$")
        ax1.plot(ts, E_ohm_t, "r:", label=r"$E_{\rm ohm} = \eta J_z$")
        ax1.set_xlabel("t")
        ax1.set_ylabel(r"$E_z$ at X-point")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(ts, A_tearing_t, "g-.", label=r"${\rm RMS}\,B_x$")
        ax2.set_ylabel(r"${\rm RMS}\,B_x$")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        ax1.set_title(
            r"Reconnection electric field, "
            + fr"$S = {S:.2e}$, $B_0 = {B0:.2f}$, $\eta = {eta:.1e}$"
        )

        fig.tight_layout()
        out_components = (prefix
                          + f"reconnection_E_components_vs_time_S{S:.2e}_B0{B0:.2f}.png")
        fig.savefig(out_components, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved {out_components}")

        # --- 2) Relative magnitudes of inductive vs Ohmic ------------------ #
        abs_ind = np.abs(E_ind_t)
        abs_ohm = np.abs(E_ohm_t)
        denom = abs_ind + abs_ohm + 1e-30

        frac_ind = abs_ind / denom
        frac_ohm = abs_ohm / denom

        fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200)
        ax.plot(ts, frac_ind, label=r"$|E_{\rm ind}|/(|E_{\rm ind}|+|E_{\rm ohm}|)$")
        ax.plot(ts, frac_ohm, label=r"$|E_{\rm ohm}|/(|E_{\rm ind}|+|E_{\rm ohm}|)$")
        ax.set_xlabel("t")
        ax.set_ylabel("Relative magnitude")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title("Relative inductive vs Ohmic contributions")
        ax.legend(loc="best")

        fig.tight_layout()
        out_frac = (prefix
                    + f"reconnection_fraction_S{S:.2e}_B0{B0:.2f}.png")
        fig.savefig(out_frac, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved {out_frac}")

        # --- 3) ln|E_rec| vs t with γ_B and γ_FKR -------------------------- #
        # Fit γ from RMS Bx
        gamma_B, (i0, i1) = fit_gamma_from_rmsBx(ts, A_tearing_t)
        print(f"[FIT] γ_B from RMS Bx ≈ {gamma_B:.3e}, "
              f"fit window t ∈ [{ts[i0]:.3f}, {ts[i1]:.3f}]")

        logE = np.log(np.abs(E_rec_t) + 1e-32)

        fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200)
        ax.plot(ts, logE, label=r"$\ln|E_{\rm rec}(t)|$")
        ax.axvspan(ts[i0], ts[i1], color="grey", alpha=0.2,
                   label="fit window (from RMS $B_x$)")

        # Build lines with slopes γ_B and γ_FKR, anchored at t[i0]
        t0_line = ts[i0]
        y0_line = logE[i0]

        logE_fitB = y0_line + gamma_B * (ts - t0_line)
        ax.plot(ts, logE_fitB, "k--",
                label=rf"$\gamma_B \approx {gamma_B:.3e}$ (from RMS $B_x$)")

        if not np.isnan(gamma_FKR):
            logE_FKR = y0_line + gamma_FKR * (ts - t0_line)
            ax.plot(ts, logE_FKR, "r:",
                    label=rf"$\gamma_{{\rm FKR}} \approx {gamma_FKR:.3e}$")

        ax.set_xlabel("t")
        ax.set_ylabel(r"$\ln|E_{\rm rec}|$")
        ax.set_title("Reconnection rate growth in linear phase")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        fig.tight_layout()
        out_logE = (prefix
                    + f"reconnection_logE_S{S:.2e}_B0{B0:.2f}.png")
        fig.savefig(out_logE, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved {out_logE}")
        
        # --- 3b) ln(RMS Bx) vs t with same γ_B and γ_FKR -------------------- #
        logA = np.log(A_tearing_t + 1e-32)

        fig2, axB = plt.subplots(figsize=(6.5, 4), dpi=200)
        axB.plot(ts, logA, label=r"$\ln({\rm RMS}\,B_x)$")

        # same fit window as used for γ_B
        axB.axvspan(ts[i0], ts[i1], color="grey", alpha=0.2,
                    label="fit window")

        # line with slope γ_B anchored at (t[i0], logA[i0])
        yA0 = logA[i0]
        logA_fitB = yA0 + gamma_B * (ts - ts[i0])
        axB.plot(ts, logA_fitB, "k--",
                 label=rf"$\gamma_B \approx {gamma_B:.3e}$ (fit)")

        # FKR prediction drawn with same anchor
        if not np.isnan(gamma_FKR):
            logA_FKR = yA0 + gamma_FKR * (ts - ts[i0])
            axB.plot(ts, logA_FKR, "r:",
                     label=rf"$\gamma_{{\rm FKR}} \approx {gamma_FKR:.3e}$")

        axB.set_xlabel("t")
        axB.set_ylabel(r"$\ln({\rm RMS}\,B_x)$")
        axB.set_title(r"Tearing-mode growth from ${\rm RMS}\,B_x$")
        axB.grid(True, alpha=0.3)
        axB.legend(loc="best")

        fig2.tight_layout()
        out_logA = (prefix
                    + f"tearing_logRMSBx_S{S:.2e}_B0{B0:.2f}.png")
        fig2.savefig(out_logA, bbox_inches="tight")
        plt.close(fig2)
        print(f"[PLOT] Saved {out_logA}")


        # --- 4) Movie of J_z with E_z contours ----------------------------- #
        movie_name = (prefix
                      + f"reconnection_movie_S{S:.2e}_B0{B0:.2f}.mp4")
        make_reconnection_movie(
            Jz_slices, Ez_slices, ts, Lx, Ly,
            ix0, iy0, E_rec_t, movie_name
        )

        # --- Multi-run scan accumulation ----------------------------------- #
        Emax = float(np.max(np.abs(E_rec_t)))
        S_list.append(S)
        B0_list.append(B0)
        Emax_list.append(Emax)
        print(f"[SCAN] Peak |E_rec| for this run = {Emax:.3e}")

    # ---------------------------------------------------------------------- #
    # Parameter scans across runs
    # ---------------------------------------------------------------------- #
    if len(run_files) > 1:
        S_arr = np.array(S_list)
        B0_arr = np.array(B0_list)
        Emax_arr = np.array(Emax_list)

        # Emax vs S
        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        ax.loglog(S_arr, Emax_arr, "o-")
        ax.set_xlabel(r"Lundquist number $S$")
        ax.set_ylabel(r"Peak $|E_{\rm rec}|$")
        ax.set_title("Reconnection rate vs Lundquist number")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(prefix + "reconnection_Emax_vs_S.png",
                    bbox_inches="tight")
        plt.close(fig)
        print("[PLOT] Saved " + prefix + "reconnection_Emax_vs_S.png")

        # Emax vs B0
        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        ax.plot(B0_arr, Emax_arr, "o-")
        ax.set_xlabel(r"Reversing field amplitude $B_0$")
        ax.set_ylabel(r"Peak $|E_{\rm rec}|$")
        ax.set_title("Reconnection rate vs reversing field")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(prefix + "reconnection_Emax_vs_B0.png",
                    bbox_inches="tight")
        plt.close(fig)
        print("[PLOT] Saved " + prefix + "reconnection_Emax_vs_B0.png")


if __name__ == "__main__":
    main()
