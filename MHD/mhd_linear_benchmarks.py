#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_linear_benchmarks.py

Linear tearing-mode benchmark suite for the Harris-sheet MHD solver.

This script automatically:
  1) Performs a RESOLUTION CONVERGENCE study for a reference case, comparing
     the measured tearing growth rate γ_mode (from the single Fourier harmonic
     Bx(kx=0, ky=1, kz=0)) with the FKR theoretical prediction γ_FKR.

  2) Performs a LUNDQUIST-NUMBER SCAN (varying S via η) and compares
     γ_mode(S) with γ_FKR(S) ~ S^{-3/5}.

  3) Performs a GUIDE-FIELD SCAN (varying B_g) and shows the impact of the
     guide field on the measured growth rate γ_mode / γ_FKR.

  4) Performs a VISCOSITY SCAN (varying Pm = ν/η) and shows γ_mode / γ_FKR
     vs Pm.

Defaults are **lightweight** to run in a couple of minutes:
  - quasi-2D (Nz_ref = 1),
  - reference resolution N_ref = 32,
  - t1 = 60, n_frames = 40,
  - small parameter grids (few S, B_g, Pm).

Usage (default: run all suites):

  python mhd_tearing_linear_benchmarks.py

Optional flags:

  --outdir OUTDIR          (default: linear_benchmarks)
  --no_convergence         Skip resolution convergence tests
  --no_Sscan               Skip Lundquist-number scan
  --no_guide_scan          Skip guide-field scan
  --no_visc_scan           Skip viscosity scan
  --t1 T1                  Final time for all runs (default: 60.0)
  --n_frames NFR           Number of saved frames (default: 40)
"""

from __future__ import annotations

import os
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax as dfx

# Import building blocks from the main solver
import mhd_tearing_solve as mhd

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": False,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "figure.dpi": 200,
    }
)

# -----------------------------------------------------------------------------#
# Helper: run one tearing-mode simulation
# -----------------------------------------------------------------------------#

def run_tearing_case(
    Nx: int,
    Ny: int,
    Nz: int,
    Lx: float,
    Ly: float,
    Lz: float,
    nu: float,
    eta: float,
    B0: float,
    a: float | None,
    B_g: float,
    eps_B: float,
    t0: float,
    t1: float,
    n_frames: int,
    dt0: float | None = None,
):
    """
    Run a single tearing-mode simulation with given parameters.

    Returns a dict with:
      ts, v_hat_frames, B_hat_frames,
      gamma_FKR, S, Delta_prime_a,
      mode indices (ix0, iy1, iz0) for kx=0, ky=1, kz=0,
      and metadata (Nx,Ny,Nz,Lx,Ly,Lz,nu,eta,B0,a,B_g,eps_B).
    """
    if a is None:
        a = Lx / 16.0

    print("--------------------------------------------------------")
    print(f"[RUN] Nx={Nx}, Ny={Ny}, Nz={Nz}")
    print(f"[RUN] Lx={Lx:.3f}, Ly={Ly:.3f}, Lz={Lz:.3f}")
    print(f"[RUN] nu={nu:.3e}, eta={eta:.3e}")
    print(f"[RUN] B0={B0:.3f}, a={a:.3f}, B_g={B_g:.3f}, eps_B={eps_B:.3e}")
    print(f"[RUN] t0={t0:.3f}, t1={t1:.3f}, n_frames={n_frames}")

    # Spectral stuff
    kx, ky, kz, k2, NX, NY, NZ = mhd.make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    mask_dealias = mhd.make_dealias_mask(Nx, Ny, Nz, NX, NY, NZ)

    # Indices for the tearing mode (kx=0, ky=1, kz=0), same convention as solver
    NX_np = np.array(NX)
    NY_np = np.array(NY)
    NZ_np = np.array(NZ)
    ix0 = int(np.where(NX_np[:, 0, 0] == 0)[0][0])
    iy1 = int(np.where(NY_np[0, :, 0] == 1)[0][0])
    iz0 = int(np.where(NZ_np[0, 0, :] == 0)[0][0])

    # FKR reference
    gamma_FKR, S, Delta_p = mhd.fkr_gamma(B0, a, Ly, eta)
    print(f"[THEORY] γ_FKR ≈ {gamma_FKR:.3e}, S={S:.3e}, Δ'a={Delta_p:.3e}")

    # Initial conditions
    v0_real, B0_real = mhd.init_equilibrium(
        Nx, Ny, Nz, Lx, Ly, Lz, B0=B0, a=a, B_g=B_g, eps_B=eps_B
    )
    v0_hat = jnp.fft.fftn(v0_real, axes=(1, 2, 3))
    B0_hat = jnp.fft.fftn(B0_real, axes=(1, 2, 3))

    v0_hat = v0_hat * mask_dealias
    B0_hat = B0_hat * mask_dealias
    v0_hat = mhd.project_div_free(v0_hat, kx, ky, kz, k2)
    B0_hat = mhd.project_div_free(B0_hat, kx, ky, kz, k2)

    E_kin0, E_mag0 = mhd.energy_from_hat(v0_hat, B0_hat, Lx, Ly, Lz)
    print(
        f"[INIT] E_kin0={float(E_kin0):.6e}, "
        f"E_mag0={float(E_mag0):.6e}, "
        f"E_tot0={float(E_kin0 + E_mag0):.6e}"
    )

    # Time step
    if dt0 is None:
        dt_max = mhd.estimate_max_dt(v0_hat, B0_hat, Lx, Ly, Lz, nu, eta)
        # slightly larger dt is okay for linear runs; keep a safety factor
        dt0 = min(2e-3, 0.5 * float(dt_max))
        print(f"[DT] Estimated dt_max={dt_max:.3e}, using dt0={dt0:.3e}")
    else:
        print(f"[DT] Using user-specified dt0={dt0:.3e}")

    rhs = mhd.make_mhd_rhs(nu, eta, kx, ky, kz, k2, mask_dealias)
    term = dfx.ODETerm(rhs)

    solver = dfx.Dopri8()
    stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-7)
    ts_save = jnp.linspace(t0, t1, n_frames)
    saveat = dfx.SaveAt(ts=ts_save)

    print("[RUN] Calling diffrax.diffeqsolve ...")
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=(v0_hat, B0_hat),
        args=None,
        saveat=saveat,
        max_steps=int((t1 - t0) / dt0) + 10_000,
        stepsize_controller=stepsize_controller,
        progress_meter=dfx.TqdmProgressMeter(),
    )
    print("[RUN] Solve finished.")
    print("[RUN] Stats:", sol.stats)

    ts = np.array(sol.ts)
    v_hat_frames, B_hat_frames = sol.ys
    v_hat_frames = np.array(v_hat_frames)
    B_hat_frames = np.array(B_hat_frames)

    return dict(
        ts=ts,
        v_hat=v_hat_frames,
        B_hat=B_hat_frames,
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
        t0=t0,
        t1=t1,
        n_frames=n_frames,
        dt0=dt0,
        gamma_FKR=gamma_FKR,
        S=S,
        Delta_p=Delta_p,
        ix0=ix0,
        iy1=iy1,
        iz0=iz0,
    )


# -----------------------------------------------------------------------------#
# Helper: measure tearing growth rate from single Fourier mode Bx(kx=0, ky=1)
# -----------------------------------------------------------------------------#

def fit_gamma_from_mode(
    ts,
    B_hat,
    ix0,
    iy1,
    iz0,
    label="Bx",
    t_min_frac: float = 0.1,   # start at 05% of run time
    t_max_frac: float = 0.3,   # end at 18% of run time
):
    """
    Extract Bx(kx=0,ky=1,kz=0), then fit γ from ln|Bx| over a *time-based*
    window [t_min_frac*(t1-t0), t_max_frac*(t1-t0)].

    Returns:
      gamma, (i0, i1), Bx_mode, logA
    """
    ts = np.asarray(ts)
    B_hat = np.asarray(B_hat)

    # B_hat has shape (n_frames, 3, Nx, Ny, Nz)
    Bx_mode = B_hat[:, 0, ix0, iy1, iz0]

    amp = np.abs(Bx_mode)
    logA = np.log(np.maximum(amp, 1e-30))

    if ts.size < 4:
        raise RuntimeError("Not enough time points to fit growth rate.")

    # --- fixed early/mid-time window in *time*, not amplitude ---
    t0 = ts[0]
    t1 = ts[-1]
    t_min = t0 + t_min_frac * (t1 - t0)
    t_max = t0 + t_max_frac * (t1 - t0)

    mask = (ts >= t_min) & (ts <= t_max)
    idx = np.where(mask)[0]

    if idx.size < 3:
        # Fallback: use middle third of points
        i0 = max(1, ts.size // 6)
        i1 = min(ts.size - 2, 5 * ts.size // 6)
        idx = np.arange(i0, i1 + 1)

    i0, i1 = int(idx[0]), int(idx[-1])
    t_fit = ts[i0 : i1 + 1]
    logA_fit = logA[i0 : i1 + 1]

    coeffs = np.polyfit(t_fit, logA_fit, 1)
    gamma = coeffs[0]

    print(
        f"[FIT] γ_{label} from |{label}(kx=0,ky=1)| ≈ {gamma:.3e}, "
        f"fit window t ∈ [{ts[i0]:.3f}, {ts[i1]:.3f}]"
    )

    return gamma, (i0, i1), Bx_mode, logA

# -----------------------------------------------------------------------------#
# 1) Resolution convergence study (fast defaults)
# -----------------------------------------------------------------------------#

def run_resolution_convergence(outdir, base_params, t1, n_frames):
    """
    Study convergence of γ_mode toward γ_FKR as resolution increases.

    Defaults: quasi-2D (Nz=1), N_list=[32,48] for speed.
    """
    print("\n=== 1) Resolution convergence study ===")

    N_list = [32, 48]  # lightweight default; extend later if desired
    gamma_num = []
    gamma_FKR_list = []
    rel_err = []

    for N in N_list:
        data = run_tearing_case(
            Nx=N,
            Ny=N,
            Nz=base_params["Nz_ref"],
            Lx=base_params["Lx"],
            Ly=base_params["Ly"],
            Lz=base_params["Lz"],
            nu=base_params["nu"],
            eta=base_params["eta"],
            B0=base_params["B0"],
            a=base_params["a"],
            B_g=base_params["B_g"],
            eps_B=base_params["eps_B"],
            t0=base_params["t0"],
            t1=t1,
            n_frames=n_frames,
            dt0=None,
        )
        # --- NEW: extract tearing mode Bx(kx=0, ky=1, kz=0) over time ---
        gamma_mode, (i0, i1), Bx_mode, logA = fit_gamma_from_mode(
            data["ts"],
            data["B_hat"],
            data["ix0"],
            data["iy1"],
            data["iz0"],
            label="Bx",
        )
        gamma_num.append(gamma_mode)
        gamma_FKR_list.append(data["gamma_FKR"])
        rel_err.append((gamma_mode - data["gamma_FKR"]) / data["gamma_FKR"])

        # diagnostic plot of ln|Bx_mode|
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot(data["ts"], logA, label=r"$\ln|B_x(k_x=0,k_y=1)|$")
        # highlight fit window
        ax.axvspan(data["ts"][i0], data["ts"][i1], color="gray", alpha=0.2,
                   label="fit window")
        # FKR line through A_mode[i0]
        t_line = np.array([data["ts"][i0], data["ts"][i1]])
        logA0 = logA[i0]
        ax.plot(
            t_line,
            logA0 + data["gamma_FKR"] * (t_line - t_line[0]),
            "k--",
            label=r"FKR slope $\gamma_{\rm FKR}$",
        )

        ax.set_xlabel("t")
        ax.set_ylabel(r"$\ln|B_x(k_x=0,k_y=1)|$")
        ax.set_title(fr"Resolution N={N}, Nz={base_params['Nz_ref']}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(
            os.path.join(outdir, f"conv_logBxmode_N{N}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)

    N_arr = np.array(N_list, dtype=float)
    gamma_num = np.array(gamma_num)
    gamma_FKR_arr = np.array(gamma_FKR_list)
    rel_err = np.array(rel_err)

    # Plot γ_mode vs N with γ_FKR
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(N_arr, gamma_num, "o-", label=r"$\gamma_{\rm mode}$ (numerical)")
    ax.hlines(
        gamma_FKR_arr[0],
        N_arr[0],
        N_arr[-1],
        colors="k",
        linestyles="--",
        label=r"$\gamma_{\rm FKR}$",
    )
    ax.set_xlabel(r"Resolution $N_x=N_y$")
    ax.set_ylabel(r"Growth rate $\gamma$")
    ax.set_title("Resolution convergence of tearing growth rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, "convergence_gamma_vs_N.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # Plot relative error vs N (log-log)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(N_arr, np.abs(rel_err), "o-")
    ax.set_xlabel(r"Resolution $N$")
    ax.set_ylabel(r"$|\gamma_{\rm mode} - \gamma_{\rm FKR}| / \gamma_{\rm FKR}$")
    ax.set_title("Convergence of growth rate toward FKR theory")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, "convergence_relerr_vs_N.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    print("[DONE] Resolution convergence plots saved.")


# -----------------------------------------------------------------------------#
# 2) Lundquist-number scan γ(S) vs FKR
# -----------------------------------------------------------------------------#

def run_S_scan(outdir, base_params, t1, n_frames):
    """
    Scan over Lundquist number S by varying η and compare γ_mode(S) with
    γ_FKR(S).

    Lightweight default: 3 S values over a modest range.
    """
    print("\n=== 2) Lundquist-number scan ===")

    # Choose S_list and compute η = a B0 / S.
    # Modest range to keep runs short but show trend:
    S_list = np.array([2.0e2, 4.0e2, 8.0e2])
    gamma_mode_list = []
    gamma_FKR_list = []

    for S_target in S_list:
        eta = base_params["a"] * base_params["B0"] / S_target
        print(f"\n[SCAN S] Target S={S_target:.3e}, eta={eta:.3e}")
        data = run_tearing_case(
            Nx=base_params["N_ref"],
            Ny=base_params["N_ref"],
            Nz=base_params["Nz_ref"],
            Lx=base_params["Lx"],
            Ly=base_params["Ly"],
            Lz=base_params["Lz"],
            nu=base_params["nu"],
            eta=eta,
            B0=base_params["B0"],
            a=base_params["a"],
            B_g=base_params["B_g"],
            eps_B=base_params["eps_B"],
            t0=base_params["t0"],
            t1=t1,
            n_frames=n_frames,
            dt0=None,
        )
        gamma_mode, _, _, _ = fit_gamma_from_mode(
            data["ts"],
            data["B_hat"],
            data["ix0"],
            data["iy1"],
            data["iz0"],
            label="Bx",
        )
        gamma_mode_list.append(gamma_mode)
        gamma_FKR_list.append(data["gamma_FKR"])

    S_arr = np.array(S_list)
    gamma_mode_arr = np.array(gamma_mode_list)
    gamma_FKR_arr = np.array(gamma_FKR_list)

    # γ vs S on log-log
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(S_arr, np.abs(gamma_mode_arr), "o-", label=r"$|\gamma_{\rm mode}|$ (num.)")
    ax.loglog(S_arr, np.abs(gamma_FKR_arr), "k--", label=r"$\gamma_{\rm FKR}$")
    ax.set_xlabel(r"Lundquist number $S$")
    ax.set_ylabel(r"Growth rate $|\gamma|$")
    ax.set_title(r"Tearing growth vs $S$ (FKR scaling $\sim S^{-3/5}$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, "S_scan_gamma_vs_S.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # γ_mode / γ_FKR vs S
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogx(S_arr, gamma_mode_arr / gamma_FKR_arr, "o-")
    ax.set_xlabel(r"Lundquist number $S$")
    ax.set_ylabel(r"$\gamma_{\rm mode} / \gamma_{\rm FKR}$")
    ax.set_title("Agreement with FKR scaling across S")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, "S_scan_gamma_over_FKR_vs_S.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    print("[DONE] Lundquist-number scan plots saved.")


# -----------------------------------------------------------------------------#
# 3) Guide-field scan
# -----------------------------------------------------------------------------#

def run_guide_scan(outdir, base_params, t1, n_frames):
    """
    Scan over guide field B_g and show γ_mode / γ_FKR vs B_g.

    Lightweight default: 3 guide-field values.
    """
    print("\n=== 3) Guide-field scan ===")

    Bg_list = [0.0, 0.2, 1.0]
    gamma_mode_list = []
    gamma_FKR_list = []

    for Bg in Bg_list:
        print(f"\n[SCAN Bg] B_g={Bg:.3f}")
        data = run_tearing_case(
            Nx=base_params["N_ref"],
            Ny=base_params["N_ref"],
            Nz=base_params["Nz_ref"],
            Lx=base_params["Lx"],
            Ly=base_params["Ly"],
            Lz=base_params["Lz"],
            nu=base_params["nu"],
            eta=base_params["eta"],
            B0=base_params["B0"],
            a=base_params["a"],
            B_g=Bg,
            eps_B=base_params["eps_B"],
            t0=base_params["t0"],
            t1=t1,
            n_frames=n_frames,
            dt0=None,
        )
        gamma_mode, _, _, _ = fit_gamma_from_mode(
            data["ts"],
            data["B_hat"],
            data["ix0"],
            data["iy1"],
            data["iz0"],
            label="Bx",
        )
        gamma_mode_list.append(gamma_mode)
        gamma_FKR_list.append(data["gamma_FKR"])

    Bg_arr = np.array(Bg_list, dtype=float)
    gamma_mode_arr = np.array(gamma_mode_list)
    gamma_FKR_arr = np.array(gamma_FKR_list)

    # γ vs B_g
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(Bg_arr, gamma_mode_arr, "o-", label=r"$\gamma_{\rm mode}$ (num.)")
    ax.hlines(
        gamma_FKR_arr[0],
        Bg_arr[0],
        Bg_arr[-1],
        colors="k",
        linestyles="--",
        label=r"$\gamma_{\rm FKR}$ (no guide field)",
    )
    ax.set_xlabel(r"Guide field $B_g$")
    ax.set_ylabel(r"Growth rate $\gamma$")
    ax.set_title("Effect of guide field on tearing growth")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, "Bg_scan_gamma_vs_Bg.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # γ_mode / γ_FKR vs B_g
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(Bg_arr, gamma_mode_arr / gamma_FKR_arr, "o-")
    ax.set_xlabel(r"Guide field $B_g$")
    ax.set_ylabel(r"$\gamma_{\rm mode} / \gamma_{\rm FKR}$")
    ax.set_title("Guide-field modification of growth rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, "Bg_scan_gamma_over_FKR_vs_Bg.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    print("[DONE] Guide-field scan plots saved.")


# -----------------------------------------------------------------------------#
# 4) Viscosity (Pm) scan
# -----------------------------------------------------------------------------#

def run_viscosity_scan(outdir, base_params, t1, n_frames):
    """
    Scan over magnetic Prandtl number Pm = ν/η and show
    γ_mode / γ_FKR vs Pm.

    Lightweight default: 3 Pm values.
    """
    print("\n=== 4) Viscosity / Pm scan ===")

    # Keep η fixed, vary ν
    Pm_list = [0.3, 1.0, 3.0]
    gamma_mode_list = []
    gamma_FKR_list = []

    for Pm in Pm_list:
        nu = Pm * base_params["eta"]
        print(f"\n[SCAN Pm] Pm={Pm:.2f}, nu={nu:.3e}")
        data = run_tearing_case(
            Nx=base_params["N_ref"],
            Ny=base_params["N_ref"],
            Nz=base_params["Nz_ref"],
            Lx=base_params["Lx"],
            Ly=base_params["Ly"],
            Lz=base_params["Lz"],
            nu=nu,
            eta=base_params["eta"],
            B0=base_params["B0"],
            a=base_params["a"],
            B_g=base_params["B_g"],
            eps_B=base_params["eps_B"],
            t0=base_params["t0"],
            t1=t1,
            n_frames=n_frames,
            dt0=None,
        )
        gamma_mode, _, _, _ = fit_gamma_from_mode(
            data["ts"],
            data["B_hat"],
            data["ix0"],
            data["iy1"],
            data["iz0"],
            label="Bx",
        )
        gamma_mode_list.append(gamma_mode)
        gamma_FKR_list.append(data["gamma_FKR"])

    Pm_arr = np.array(Pm_list, dtype=float)
    gamma_mode_arr = np.array(gamma_mode_list)
    gamma_FKR_arr = np.array(gamma_FKR_list)

    # γ vs Pm
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogx(Pm_arr, gamma_mode_arr, "o-", label=r"$\gamma_{\rm mode}$ (num.)")
    ax.hlines(
        gamma_FKR_arr[0],
        Pm_arr[0],
        Pm_arr[-1],
        colors="k",
        linestyles="--",
        label=r"$\gamma_{\rm FKR}$",
    )
    ax.set_xlabel(r"Magnetic Prandtl number ${\rm Pm}=\nu/\eta$")
    ax.set_ylabel(r"Growth rate $\gamma$")
    ax.set_title("Effect of viscosity (Pm) on tearing growth")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, "Pm_scan_gamma_vs_Pm.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # γ_mode / γ_FKR vs Pm
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogx(Pm_arr, gamma_mode_arr / gamma_FKR_arr, "o-")
    ax.set_xlabel(r"Magnetic Prandtl number ${\rm Pm}=\nu/\eta$")
    ax.set_ylabel(r"$\gamma_{\rm mode} / \gamma_{\rm FKR}$")
    ax.set_title("Viscosity modification of growth rate")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, "Pm_scan_gamma_over_FKR_vs_Pm.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    print("[DONE] Viscosity scan plots saved.")


# -----------------------------------------------------------------------------#
# CLI and main
# -----------------------------------------------------------------------------#

def parse_args():
    p = argparse.ArgumentParser(
        description="Linear tearing-mode benchmark suite for mhd_tearing_solve."
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="linear_benchmarks",
        help="Directory to store benchmark plots.",
    )
    p.add_argument(
        "--no_convergence",
        action="store_true",
        help="Skip resolution convergence study.",
    )
    p.add_argument(
        "--no_Sscan",
        action="store_true",
        help="Skip Lundquist-number scan.",
    )
    p.add_argument(
        "--no_guide_scan",
        action="store_true",
        help="Skip guide-field scan.",
    )
    p.add_argument(
        "--no_visc_scan",
        action="store_true",
        help="Skip viscosity (Pm) scan.",
    )
    p.add_argument(
        "--t1",
        type=float,
        default=60.0,
        help="Final time for all benchmark runs (default: 60).",
    )
    p.add_argument(
        "--n_frames",
        type=int,
        default=40,
        help="Number of saved frames per run (default: 40).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Base physical parameters for benchmarks (edit as desired).
    # Defaults are quasi-2D (Nz_ref=1) and moderate resolution.
    base_params = dict(
        N_ref=128,          # reference Nx=Ny for scans
        Nz_ref=1,          # Nz_ref=1: quasi-2D to keep runs very cheap
        Lx=2.0 * math.pi,
        Ly=2.0 * math.pi,
        Lz=2.0 * math.pi,
        nu=1e-3,
        eta=5e-3,
        B0=1.0,
        a=(2.0 * math.pi) / 16.0,  # consistent with default in solver
        B_g=0.2,
        eps_B=0.01,
        t0=0.0,
    )
    
    args.t1 = 20

    print("========================================================")
    print(" Linear tearing-mode benchmark suite")
    print(" Output directory:", outdir)
    print(f" Defaults: Nz={base_params['Nz_ref']}, N_ref={base_params['N_ref']}, "
          f"t1={args.t1}, n_frames={args.n_frames}")
    print("========================================================")

    if not args.no_convergence:
        run_resolution_convergence(
            outdir, base_params, t1=args.t1, n_frames=args.n_frames
        )

    if not args.no_Sscan:
        run_S_scan(outdir, base_params, t1=args.t1, n_frames=args.n_frames)

    if not args.no_guide_scan:
        run_guide_scan(
            outdir, base_params, t1=args.t1, n_frames=args.n_frames
        )

    if not args.no_visc_scan:
        run_viscosity_scan(
            outdir, base_params, t1=args.t1, n_frames=args.n_frames
        )

    print("\n[ALL DONE] Linear benchmark suite complete.")


if __name__ == "__main__":
    main()
