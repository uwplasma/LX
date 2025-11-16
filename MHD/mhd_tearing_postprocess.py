#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_postprocess.py

Post-process a saved MHD tearing-mode solution produced by
mhd_tearing_solve.py and generate publication-ready diagnostics:

  - Energies and dissipation (E_kin, E_mag, E_tot, E_cons)
  - Tearing-mode amplitudes (RMS Bx and |Bx(kx=0,ky=1,kz=0)|)
  - Automatic linear-phase fit for gamma
  - Energy invariant error
  - Snapshots of Bx, Jz, and A_z field lines
  - Movies of Bx, Jz, and Jz with flux contours
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

plt.rcParams.update({
    "font.size": 12,
    "text.usetex": False,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
})

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from mhd_tearing_solve import (
    make_k_arrays, make_dealias_mask, project_div_free,
    energy_from_hat, dissipation_rates, tearing_amplitude,
    curl_from_hat, compute_Az_from_hat, grad_from_hat
)

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
    field_slices = np.asarray(field_slices)
    n_frames, Nx, Ny = field_slices.shape

    if vmin is None:
        vmin = float(field_slices.min())
    if vmax is None:
        vmax = float(field_slices.max())

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)

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
                cs.remove()
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
# Main post-processing
# -----------------------------------------------------------------------------#

def parse_args():
    p = argparse.ArgumentParser(
        description="Post-process MHD tearing solution and make diagnostics."
    )
    p.add_argument("infile", nargs="?", default="mhd_tearing_solution.npz",
                        help="Input .npz file produced by mhd_tearing_solve.py")
    p.add_argument("--no-make-movies", dest="make_movies", action="store_false",
                   help="Do not build mp4 movies.")
    p.add_argument("--prefix", type=str, default="",
                   help="Prefix for all output figures/movies.")
    return p.parse_args()

def main():
    args = parse_args()
    data = np.load(args.infile, allow_pickle=True)

    ts = np.array(data["ts"])
    v_hat_frames = np.array(data["v_hat"])
    B_hat_frames = np.array(data["B_hat"])

    Nx = int(data["Nx"]); Ny = int(data["Ny"]); Nz = int(data["Nz"])
    Lx = float(data["Lx"]); Ly = float(data["Ly"]); Lz = float(data["Lz"])
    nu = float(data["nu"]); eta = float(data["eta"])
    B0 = float(data["B0"]); a = float(data["a"])
    B_g = float(data["B_g"]); eps_B = float(data["eps_B"])
    gamma_FKR = float(data["gamma_FKR"])
    S = float(data["S"])
    Delta_prime_a = float(data["Delta_prime_a"])
    ix0 = int(data["ix0"]); iy1 = int(data["iy1"]); iz0 = int(data["iz0"])

    print("=== Post-processing MHD tearing solution ===")
    print(f"infile = {args.infile}")
    print(f"Nx,Ny,Nz = {Nx},{Ny},{Nz}")
    print(f"Lx,Ly,Lz = {Lx},{Ly},{Lz}")
    print(f"nu={nu}, eta={eta}, B0={B0}, a={a}, B_g={B_g}")
    print(f"S={S:.3e}, Delta' a={Delta_prime_a:.3e}, gamma_FKR={gamma_FKR:.3e}")
    print("============================================")

    # Spectral operators
    kx, ky, kz, k2, NX_arr, NY_arr, NZ_arr = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    mask_dealias = make_dealias_mask(Nx, Ny, Nz, NX_arr, NY_arr, NZ_arr)

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

    E_cons_running = 0.0
    E_cons0 = None

    for i in range(len(ts)):
        v_hat_i = jnp.array(v_hat_frames[i]) * mask_dealias
        B_hat_i = jnp.array(B_hat_frames[i]) * mask_dealias
        v_hat_i = project_div_free(v_hat_i, kx, ky, kz, k2)
        B_hat_i = project_div_free(B_hat_i, kx, ky, kz, k2)

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

        # tearing amplitude (RMS Bx in band)
        A_rms = tearing_amplitude(B_hat_i, Lx, Ly, Lz, band_width_frac=0.25)
        tearing_amp_list.append(A_rms)

        # single Fourier mode amplitude
        Bx_hat_i = np.array(B_hat_i[0])
        mode_amp_list.append(float(np.abs(Bx_hat_i[ix0, iy1, iz0])))

        # velocity diagnostics
        v_i = np.fft.ifftn(np.array(v_hat_i), axes=(1, 2, 3)).real
        v_mag = np.sqrt(np.sum(v_i**2, axis=0))
        v_rms_list.append(float(np.sqrt(np.mean(v_mag**2))))
        v_max_list.append(float(np.max(v_mag)))

        # energy invariant
        if i > 0:
            dt = ts[i] - ts[i-1]
            eps_prev = eps_visc_list[i-1] + eps_ohm_list[i-1]
            eps_curr = eps_visc_list[i]   + eps_ohm_list[i]
            E_cons_running += 0.5 * (eps_prev + eps_curr) * dt

        E_cons_val = float(E_tot_i + E_cons_running)
        if E_cons0 is None:
            E_cons0 = E_cons_val
        E_cons_list.append(E_cons_val)

        print(
            f"[POST] frame {i}/{len(ts)-1}, t={ts[i]:.4f}, "
            f"E_kin={E_kin_i:.3e}, E_mag={E_mag_i:.3e}, E_tot={E_tot_i:.3e}, "
            f"eps_visc={eps_visc_i:.3e}, eps_ohm={eps_ohm_i:.3e}, "
            f"E_cons={E_cons_val:.3e}, A_tearing={tearing_amp_list[-1]:.3e}"
        )

    ts_np = ts
    E_kin_arr = np.array(E_kin_list)
    E_mag_arr = np.array(E_mag_list)
    E_tot_arr = np.array(E_tot_list)
    eps_visc_arr = np.array(eps_visc_list)
    eps_ohm_arr = np.array(eps_ohm_list)
    E_cons_arr = np.array(E_cons_list)
    tearing_amp_arr = np.array(tearing_amp_list)
    mode_amp_arr = np.array(mode_amp_list)
    v_rms_arr = np.array(v_rms_list)
    v_max_arr = np.array(v_max_list)
    rel_E_cons_err = (E_cons_arr - E_cons_arr[0]) / E_cons_arr[0]

    prefix = args.prefix

    # --------------------------- Energy plot -------------------------------- #
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=200)
    ax1, ax2 = axs

    ax1.plot(ts_np, E_kin_arr, label=r"$E_{\rm kin}$")
    ax1.plot(ts_np, E_mag_arr, label=r"$E_{\rm mag}$")
    ax1.plot(ts_np, E_tot_arr, "--", label=r"$E_{\rm tot}$")
    ax1.plot(ts_np, E_cons_arr, "-.", label=r"$E_{\rm cons}$")
    ax1.set_xlabel("t")
    ax1.set_ylabel("Energy")
    ax1.set_title("MHD energies and invariant")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(ts_np, eps_visc_arr, label=r"$\epsilon_{\rm visc}$")
    ax2.plot(ts_np, eps_ohm_arr, label=r"$\epsilon_{\rm ohm}$")
    ax2.plot(ts_np, eps_visc_arr + eps_ohm_arr, "--",
             label=r"$\epsilon_{\rm tot}$")
    ax2.set_xlabel("t")
    ax2.set_ylabel("Dissipation rate")
    ax2.set_title("Dissipation rates vs time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(prefix + "mhd_energy_invariants.png", bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Diagnostics saved to "
          f"{prefix}mhd_energy_invariants.png")

    # ------------------ Tearing mode & velocity scales plot ----------------- #
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=200)

    axs[0].plot(ts_np, mode_amp_arr)
    axs[0].set_xlabel("t")
    axs[0].set_ylabel(r"$|B_x(k_x=0,k_y=1,k_z=0)|$")
    axs[0].set_title("Tearing-mode Fourier amplitude")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(ts_np, v_rms_arr, label=r"$v_{\rm rms}$")
    axs[1].plot(ts_np, v_max_arr, "--", label=r"$v_{\max}$")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel(r"Velocity")
    axs[1].set_title("Reconnection outflow speeds")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    fig.tight_layout()
    fig.savefig(prefix + "tearing_mode_velocity_scales.png",
                bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved "
          f"{prefix}tearing_mode_velocity_scales.png")

    # ---------------------- Tearing-mode diagnostics plot ------------------- #
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), dpi=200)

    # Panel 1
    axs[0].plot(ts_np, tearing_amp_arr, label=r"${\rm RMS}\,B_x$")
    axs[0].plot(ts_np, mode_amp_arr, '--',
                label=r"$|B_x(k_x=0,k_y=1,k_z=0)|$")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Tearing amplitude")
    axs[0].legend()

    # Linear-phase fit
    log_mode = np.log(mode_amp_arr + 1e-16)
    A = mode_amp_arr
    A0 = A[1] if A.shape[0] > 1 else A[0]
    Amax = A.max()

    f_min = 5.0
    f_max = 0.30
    mask = (A > f_min * A0) & (A < f_max * Amax)
    idx_lin = np.where(mask)[0]
    if idx_lin.size < 3:
        idx_lin = np.arange(2, max(5, len(ts_np)//3))

    i0, i1 = int(idx_lin[0]), int(idx_lin[-1])
    t_fit = ts_np[i0:i1+1]
    logA_fit = log_mode[i0:i1+1]
    coeffs = np.polyfit(t_fit, logA_fit, 1)
    gamma_fit = coeffs[0]
    logA_line = coeffs[1] + coeffs[0] * ts_np

    print(f"[FIT] Measured tearing gamma ≈ {gamma_fit:.3e}")
    if not np.isnan(gamma_FKR):
        ratio = gamma_fit / gamma_FKR
        print(f"[COMP] gamma_fit/gamma_FKR ≈ {ratio:.3f}")
    else:
        ratio = np.nan

    # Panel 2
    axs[1].plot(ts_np, log_mode, label=r"$\ln|B_x(k_x=0,k_y=1)|$")
    axs[1].axvspan(ts_np[i0], ts_np[i1], color="grey", alpha=0.2,
                   label="fit window")
    axs[1].plot(ts_np, logA_line, "k--",
                label=rf"fit: $\gamma \approx {gamma_fit:.3e}$")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel(r"$\ln |B_x(k_x=0,k_y=1)|$")
    axs[1].set_title("Mode growth (linear phase shaded)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best")

    if not np.isnan(gamma_FKR):
        txt = (
            rf"$\gamma_\mathrm{{fit}} \approx {gamma_fit:.3e}$" + "\n" +
            rf"$\gamma_\mathrm{{FKR}} \approx {gamma_FKR:.3e}$" + "\n" +
            rf"$\gamma_\mathrm{{fit}}/\gamma_\mathrm{{FKR}} \approx {ratio:.2f}$"
        )
        axs[1].text(
            0.05, 0.05, txt,
            transform=axs[1].transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Panel 3
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

    # ---------------------------- Snapshots plot ---------------------------- #
    idxs = [0, len(ts_np)//2, len(ts_np)-1]
    labels = [f"t = {ts_np[i]:.2f}" for i in idxs]
    mid_z = Nz // 2

    fig, axs = plt.subplots(len(idxs), 3,
                            figsize=(11, 3.6*len(idxs)), dpi=200)
    if len(idxs) == 1:
        axs = np.array([axs])

    for row, (i, lab) in enumerate(zip(idxs, labels)):
        v_hat_i = np.array(v_hat_frames[i])
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

        im0 = axs[row, 0].imshow(Bx.T, origin="lower",
                                 extent=[0, Lx, 0, Ly],
                                 aspect="equal")
        axs[row, 0].set_title(r"$B_x(x,y,z=0)$, " + lab)
        axs[row, 0].set_ylabel("y")
        fig.colorbar(im0, ax=axs[row, 0])

        im1 = axs[row, 1].imshow(Jz.T, origin="lower",
                                 extent=[0, Lx, 0, Ly],
                                 aspect="equal")
        axs[row, 1].set_title(r"$J_z(x,y,z=0)$")
        fig.colorbar(im1, ax=axs[row, 1])

        cs = axs[row, 2].contour(Az.T, levels=25,
                                 extent=[0, Lx, 0, Ly])
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

    # ---------------------------- Movies ------------------------------------ #
    if args.make_movies:
        print("[MOVIE] Building tearing-mode movies ...")
        mid_z = Nz // 2
        Bx_slices = []
        Jz_slices = []
        Az_slices = []

        for i in range(len(ts_np)):
            B_hat_i = B_hat_frames[i]
            v_hat_i = v_hat_frames[i]

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

        make_movie(
            Bx_slices,
            prefix + "mhd_tearing_Bx_xy.mp4",
            ts_np,
            Lx,
            Ly,
            title=r"$B_x(x,y,z=0)$",
        )

        make_movie(
            Jz_slices,
            prefix + "mhd_tearing_Jz_xy.mp4",
            ts_np,
            Lx,
            Ly,
            title=r"$J_z(x,y,z=0)$",
        )

        make_movie(
            Jz_slices,
            prefix + "mhd_tearing_flux_contours.mp4",
            ts_np,
            Lx,
           Ly,
            title=r"$J_z$ with flux contours",
            add_flux_contours=True,
            flux_slices=Az_slices,
            n_flux_levels=15,
        )

        print("[DONE] Movie generation complete.")

if __name__ == "__main__":
    main()
