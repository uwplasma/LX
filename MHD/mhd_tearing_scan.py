#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_scan.py

"Loureiro-style" scan driver for Harris-sheet tearing.

This script does *both*:
  1) runs the incompressible pseudo-spectral MHD tearing simulation
     (JAX + diffrax) for a scan over (a, eta), and
  2) postprocesses each run to extract:
       - island width w(t),
       - linear growth rate γ_fit,
       - Rutherford slope (dw/dt)_R,
       - saturated width w_sat,
     and then builds scan plots:

       (i)   γ_fit vs γ_FKR,
       (ii)  (dw/dt)_R vs ηΔ',
       (iii) w_sat vs Δ',
       (iv)  γ_fit/γ_FKR vs S and vs Δ'.

All runs are saved as mhd_tearing_solution_*.npz in --outdir, and a
summary file tearing_scan_summary.npz plus PNGs are written there.

Example usage
-------------

python mhd_tearing_scan.py \
    --scan-a 0.25 0.35 0.45 \
    --scan-eta 5e-4 1e-3 2e-3 \
    --Nx 48 --Ny 48 --Nz 48 \
    --Lx 6.283185307179586 --Ly 6.283185307179586 --Lz 6.283185307179586 \
    --t0 0.0 --t1 100.0 --n-frames 80 \
    --outdir tearing_scan

"""

from __future__ import annotations

import argparse
import math
import os
from typing import List, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax as dfx


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
# Solver utilities (copied/adapted from mhd_tearing_solve.py)
# -----------------------------------------------------------------------------#

def estimate_max_dt(v_hat, B_hat, Lx, Ly, Lz, nu, eta,
                    CFL_adv=0.4, CFL_diff=0.2):
    """
    Estimate a safe maximum timestep from CFL + diffusion constraints.
    v_hat, B_hat: (3, Nx, Ny, Nz), complex
    """
    v = jnp.fft.ifftn(v_hat, axes=(1, 2, 3)).real
    B = jnp.fft.ifftn(B_hat, axes=(1, 2, 3)).real

    v_mag = jnp.sqrt(jnp.sum(v * v, axis=0))
    B_mag = jnp.sqrt(jnp.sum(B * B, axis=0))

    v_char = jnp.max(v_mag + B_mag)

    Nx = v.shape[1]
    Ny = v.shape[2]
    Nz = v.shape[3]
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    hmin = jnp.min(jnp.array([dx, dy, dz]))

    dt_adv = jnp.where(v_char > 0.0, CFL_adv * hmin / v_char, 1e9)

    nu_eff = jnp.maximum(nu, eta)
    dt_diff = CFL_diff * hmin * hmin / jnp.maximum(nu_eff, 1e-16)

    dt_max = jnp.minimum(dt_adv, dt_diff)
    return float(dt_max)


def make_grid(Nx, Ny, Nz, Lx, Ly, Lz):
    x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0.0, Ly, Ny, endpoint=False)
    z = jnp.linspace(0.0, Lz, Nz, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z


def make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz):
    nx = jnp.fft.fftfreq(Nx) * Nx
    ny = jnp.fft.fftfreq(Ny) * Ny
    nz = jnp.fft.fftfreq(Nz) * Nz
    NX, NY, NZ = jnp.meshgrid(nx, ny, nz, indexing="ij")

    kx = 2.0 * jnp.pi * NX / Lx
    ky = 2.0 * jnp.pi * NY / Ly
    kz = 2.0 * jnp.pi * NZ / Lz

    k2 = kx**2 + ky**2 + kz**2
    k2 = jnp.where(k2 == 0.0, 1.0, k2)  # avoid divide-by-zero at k=0
    return kx, ky, kz, k2, NX, NY, NZ


def make_dealias_mask(Nx, Ny, Nz, NX, NY, NZ):
    kx_cut = Nx // 3
    ky_cut = Ny // 3
    kz_cut = Nz // 3
    mask = (
        (jnp.abs(NX) <= kx_cut) &
        (jnp.abs(NY) <= ky_cut) &
        (jnp.abs(NZ) <= kz_cut)
    )
    return mask.astype(jnp.complex128)


def project_div_free(v_hat, kx, ky, kz, k2):
    """
    Project a vector field in Fourier space onto divergence-free subspace:
      v_hat -> (I - k k^T / k^2) v_hat
    v_hat shape: (3, Nx, Ny, Nz)
    """
    vx_hat, vy_hat, vz_hat = v_hat[0], v_hat[1], v_hat[2]
    k_dot_v = kx * vx_hat + ky * vy_hat + kz * vz_hat
    factor = k_dot_v / k2

    vx_hat_proj = vx_hat - factor * kx
    vy_hat_proj = vy_hat - factor * ky
    vz_hat_proj = vz_hat - factor * kz
    return jnp.stack([vx_hat_proj, vy_hat_proj, vz_hat_proj], axis=0)


def grad_vec_from_hat(F_hat, kx, ky, kz):
    """
    Gradient of a vector field from Fourier coefficients.

    F_hat: (3, Nx, Ny, Nz) complex, components (F_x, F_y, F_z)
    Returns grad_F[i,j,...] = ∂F_j/∂x_i
    """
    df_dx_hat = 1j * kx * F_hat
    df_dy_hat = 1j * ky * F_hat
    df_dz_hat = 1j * kz * F_hat

    df_dx = jnp.fft.ifftn(df_dx_hat, axes=(1, 2, 3)).real
    df_dy = jnp.fft.ifftn(df_dy_hat, axes=(1, 2, 3)).real
    df_dz = jnp.fft.ifftn(df_dz_hat, axes=(1, 2, 3)).real

    grad_F = jnp.stack([
        jnp.stack([df_dx[0], df_dx[1], df_dx[2]], axis=0),
        jnp.stack([df_dy[0], df_dy[1], df_dy[2]], axis=0),
        jnp.stack([df_dz[0], df_dz[1], df_dz[2]], axis=0),
    ], axis=0)
    return grad_F  # (3,3,Nx,Ny,Nz)


def directional_derivative_vec(A, grad_B):
    """
    Compute (A · ∇) B in real space.

    A:       (3, Nx, Ny, Nz)
    grad_B:  (3, 3, Nx, Ny, Nz) with grad_B[i,j,...] = ∂B_j/∂x_i
    Returns adv_j = Σ_i A_i ∂B_j/∂x_i
    """
    return jnp.einsum("i...,ij...->j...", A, grad_B)


def init_equilibrium(Nx, Ny, Nz, Lx, Ly, Lz, B0=1.0, a=None,
                     B_g=0.2, eps_B=0.01, m_y=1, m_z=0):
    """
    Harris-sheet-like slab tearing equilibrium in a periodic box.

      B_y(x) = B0 * tanh((x - Lx/2)/a)
      B_z    = B_g
      B_x    = 0

    Perturbation via δA_z = eps_B cos(k_y y) cos(k_z z):
      => δB_x = -eps_B * k_y sin(k_y y) cos(k_z z)
    """
    if a is None:
        a = Lx / 16.0

    X, Y, Z = make_grid(Nx, Ny, Nz, Lx, Ly, Lz)

    sx = (X - 0.5 * Lx) / a
    By0 = B0 * jnp.tanh(sx)
    Bx0 = jnp.zeros_like(By0)
    Bz0 = B_g * jnp.ones_like(By0)

    k_y = 2.0 * jnp.pi * m_y / Ly
    k_z = 2.0 * jnp.pi * m_z / Lz

    phase_y = k_y * Y
    phase_z = k_z * Z

    delta_Bx = -eps_B * k_y * jnp.sin(phase_y) * jnp.cos(phase_z)
    delta_By = jnp.zeros_like(delta_Bx)
    delta_Bz = jnp.zeros_like(delta_Bx)

    Bx = Bx0 + delta_Bx
    By = By0 + delta_By
    Bz = Bz0 + delta_Bz

    B0_real = jnp.stack([Bx, By, Bz], axis=0)
    v0_real = jnp.zeros_like(B0_real)

    return v0_real, B0_real


def energy_from_hat(v_hat, B_hat, Lx, Ly, Lz):
    v = jnp.fft.ifftn(v_hat, axes=(1, 2, 3)).real
    B = jnp.fft.ifftn(B_hat, axes=(1, 2, 3)).real
    dv = (Lx * Ly * Lz) / (v[0].size)

    v2 = jnp.sum(v * v, axis=0)
    B2 = jnp.sum(B * B, axis=0)
    E_kin = 0.5 * jnp.sum(v2) * dv
    E_mag = 0.5 * jnp.sum(B2) * dv
    return E_kin, E_mag


def make_mhd_rhs(nu, eta, kx, ky, kz, k2, mask_dealias):

    def rhs(t, y_hat, args_unused):
        v_hat, B_hat = y_hat

        v_hat = v_hat * mask_dealias
        B_hat = B_hat * mask_dealias

        v_hat_p = project_div_free(v_hat, kx, ky, kz, k2)
        B_hat_p = project_div_free(B_hat, kx, ky, kz, k2)

        v = jnp.fft.ifftn(v_hat_p, axes=(1, 2, 3)).real
        B = jnp.fft.ifftn(B_hat_p, axes=(1, 2, 3)).real

        grad_v = grad_vec_from_hat(v_hat_p, kx, ky, kz)
        grad_B = grad_vec_from_hat(B_hat_p, kx, ky, kz)

        adv_v  = directional_derivative_vec(v, grad_v)
        strB_v = directional_derivative_vec(B, grad_B)

        adv_B  = directional_derivative_vec(v, grad_B)
        strv_B = directional_derivative_vec(B, grad_v)

        Nv = -adv_v + strB_v
        NB = -adv_B + strv_B

        Nv_hat = jnp.fft.fftn(Nv, axes=(1, 2, 3)) * mask_dealias
        NB_hat = jnp.fft.fftn(NB, axes=(1, 2, 3)) * mask_dealias

        Nv_hat = project_div_free(Nv_hat, kx, ky, kz, k2)

        lap_factor = -k2
        dv_hat_dt = Nv_hat + nu * lap_factor * v_hat_p
        dB_hat_dt = NB_hat + eta * lap_factor * B_hat_p

        return (dv_hat_dt, dB_hat_dt)

    return jax.jit(rhs)


def fkr_gamma(B0, a, Ly, eta):
    ky_val = 2.0 * math.pi / Ly   # m_y = 1
    ka = ky_val * a
    Delta_prime_a = 2.0 * (1.0/ka - ka)
    vA = B0  # ρ = 1
    S = a * vA / eta
    C_fkr = 0.55
    if Delta_prime_a > 0.0:
        gamma_theory = C_fkr * vA / a * (Delta_prime_a**(4.0/5.0)) * (S**(-3.0/5.0))
    else:
        gamma_theory = float("nan")
    return gamma_theory, S, Delta_prime_a


# -----------------------------------------------------------------------------#
# Run a single tearing simulation and save NPZ
# -----------------------------------------------------------------------------#

def solve_tearing_case(
    Nx: int,
    Ny: int,
    Nz: int,
    Lx: float,
    Ly: float,
    Lz: float,
    nu: float,
    eta: float,
    B0: float,
    a: float,
    B_g: float,
    eps_B: float,
    t0: float,
    t1: float,
    n_frames: int,
    dt0: float | None,
    outfile: str,
) -> str:
    """
    Full MHD tearing solve for a single (a, eta) and save to `outfile`.
    Returns the outfile path.
    """
    print("\n=== Incompressible pseudo-spectral MHD Parameters ===")
    print(f"Nx,Ny,Nz = {Nx},{Ny},{Nz}")
    print(f"Lx,Ly,Lz = {Lx},{Ly},{Lz}")
    print(f"nu={nu}, eta={eta}")
    print(f"B0={B0}, a={a}, B_g={B_g}, eps_B={eps_B}")
    print(f"t0={t0}, t1={t1}, n_frames={n_frames}")
    print("=====================================================")

    kx, ky, kz, k2, NX, NY, NZ = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    mask_dealias = make_dealias_mask(Nx, Ny, Nz, NX, NY, NZ)

    # indices for tearing mode (kx=0, ky=1, kz=0)
    NX_np = np.array(NX)
    NY_np = np.array(NY)
    NZ_np = np.array(NZ)
    ix0 = int(np.where(NX_np[:, 0, 0] == 0)[0][0])
    iy1 = int(np.where(NY_np[0, :, 0] == 1)[0][0])
    iz0 = int(np.where(NZ_np[0, 0, :] == 0)[0][0])

    gamma_theory, S, Delta_prime_a = fkr_gamma(B0, a, Ly, eta)
    print(f"[THEORY] FKR-like tearing estimate: gamma ≈ {gamma_theory:.3e}")
    print(f"[THEORY] S = {S:.3e}, Delta' a = {Delta_prime_a:.3e}")

    v0_real, B0_real = init_equilibrium(
        Nx, Ny, Nz, Lx, Ly, Lz,
        B0=B0, a=a, B_g=B_g, eps_B=eps_B
    )
    v0_hat = jnp.fft.fftn(v0_real, axes=(1, 2, 3))
    B0_hat = jnp.fft.fftn(B0_real, axes=(1, 2, 3))

    v0_hat = v0_hat * mask_dealias
    B0_hat = B0_hat * mask_dealias
    v0_hat = project_div_free(v0_hat, kx, ky, kz, k2)
    B0_hat = project_div_free(B0_hat, kx, ky, kz, k2)

    E_kin0, E_mag0 = energy_from_hat(v0_hat, B0_hat, Lx, Ly, Lz)
    print(f"[INIT] E_kin0={float(E_kin0):.6e}, "
          f"E_mag0={float(E_mag0):.6e}, "
          f"E_tot0={float(E_kin0+E_mag0):.6e}")

    # Timestep
    if dt0 is None:
        dt_max = estimate_max_dt(v0_hat, B0_hat, Lx, Ly, Lz, nu, eta)
        print(f"[DT] Estimated dt_max from CFL/diffusion = {dt_max:.3e}")
        dt0 = min(1e-3, 0.5 * dt_max)
    print(f"[DT] Using dt0 = {dt0:.3e}")

    rhs = make_mhd_rhs(nu, eta, kx, ky, kz, k2, mask_dealias)
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

    v_hat_end = jnp.array(v_hat_frames[-1])
    B_hat_end = jnp.array(B_hat_frames[-1])
    E_kin_end, E_mag_end = energy_from_hat(v_hat_end, B_hat_end, Lx, Ly, Lz)
    print(f"[FINAL] E_kin={float(E_kin_end):.6e}, "
          f"E_mag={float(E_mag_end):.6e}, "
          f"E_tot={float(E_kin_end+E_mag_end):.6e}")

    out = {
        "ts": ts,
        "v_hat": v_hat_frames,
        "B_hat": B_hat_frames,
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
        "Lx": Lx, "Ly": Ly, "Lz": Lz,
        "nu": nu, "eta": eta,
        "B0": B0, "a": a, "B_g": B_g, "eps_B": eps_B,
        "t0": t0, "t1": t1,
        "n_frames": n_frames,
        "dt0": dt0,
        "gamma_FKR": gamma_theory,
        "S": S,
        "Delta_prime_a": Delta_prime_a,
        "ix0": ix0, "iy1": iy1, "iz0": iz0,
    }
    np.savez(outfile, **out)
    print(f"[SAVE] Solution saved to {outfile}")
    return outfile


# -----------------------------------------------------------------------------#
# Post-processing utilities (scan analysis)
# -----------------------------------------------------------------------------#

def compute_k_arrays_np(Nx, Ny, Nz, Lx, Ly, Lz):
    """Wrapper around make_k_arrays but returning NumPy arrays."""
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


# -----------------------------------------------------------------------------#
# Robust automatic linear-window selector
# -----------------------------------------------------------------------------#

def select_linear_window(ts, w, w0=None, min_pts=6, frac_sat=0.3, nwin=8):
    """
    Data-driven selector for the exponentially growing window of ln(w).

    - Avoids early transient: requires w > 1.2 w0.
    - Avoids near-saturation: requires w < w0 + frac_sat*(w_max-w0).
    - Slides a window of size nwin over candidate points.
    - Picks window with smallest mean-squared error in ln(w) fit.

    Returns mask_lin (boolean array).
    """
    ts = np.asarray(ts)
    w = np.asarray(w)

    if w0 is None:
        w0 = w[0]

    wmax = np.nanmax(w)
    upper = w0 + frac_sat*(wmax - w0)

    # Candidate indices
    mask = (w > 1.2*w0) & (w < upper) & np.isfinite(w)
    idx = np.where(mask)[0]

    if idx.size < min_pts:
        # fallback: earliest quarter of the simulation
        return ts < (ts[0] + 0.25*(ts[-1]-ts[0]))

    nwin = min(nwin, idx.size)
    lnw = np.log(w)
    best_err = np.inf
    best_slice = None

    for s in range(0, idx.size - nwin + 1):
        win = idx[s:s+nwin]
        t_win = ts[win]
        y_win = lnw[win]
        a, b = np.polyfit(t_win, y_win, 1)
        fit = a*t_win + b
        err = np.mean((y_win - fit)**2)
        if err < best_err:
            best_err = err
            best_slice = win

    mask_lin = np.zeros_like(w, dtype=bool)
    mask_lin[best_slice] = True
    return mask_lin


# -----------------------------------------------------------------------------#
# Scan analysis for a single run
# -----------------------------------------------------------------------------#

def _linear_regression_with_stats(x, y):
    """
    Simple y = a x + b regression with R^2 and standard error of slope.
    Returns (a, b, R2, a_err).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    N = x.size
    a, b = np.polyfit(x, y, 1)
    y_pred = a*x + b
    resid = y - y_pred
    RSS = np.sum(resid**2)
    TSS = np.sum((y - np.mean(y))**2)
    R2 = 1.0 - RSS/TSS if TSS > 0 else np.nan
    if N > 2:
        sigma2 = RSS/(N - 2)
        x_var = np.sum((x - np.mean(x))**2)
        a_err = np.sqrt(sigma2/x_var) if x_var > 0 else np.nan
    else:
        a_err = np.nan
    return a, b, R2, a_err


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

    Delta_prime = Delta_prime_a / a
    etaDelta = eta * Delta_prime

    print(f"[RUN] Nx={Nx}, Ny={Ny}, Nz={Nz}, Lx={Lx:.3f}, Ly={Ly:.3f}, Lz={Lz:.3f}")
    print(f"[RUN] nu={nu:.3e}, eta={eta:.3e}, B0={B0:.3e}, a={a:.3e}, eps_B={eps_B:.3e}")
    print(f"[RUN] S={S:.3e}, Delta'*a={Delta_prime_a:.3e}, Delta'={Delta_prime:.3e}")
    print(f"[RUN] γ_FKR={gamma_FKR:.3e}, mode indices (ix0,iy1,iz0)=({ix0},{iy1},{iz0})")

    # k arrays (NumPy)
    kx, ky, kz, NX, NY, NZ = compute_k_arrays_np(Nx, Ny, Nz, Lx, Ly, Lz)
    ky_val = ky[ix0, iy1, iz0]
    print(f"[DEBUG] ky for tearing mode = {ky_val:.6f}")

    n_t = ts.size
    island_width = np.zeros(n_t)
    Az_amp = np.zeros(n_t)

    for it in range(n_t):
        B_hat = B_hat_frames[it]
        Az_hat = compute_Az_hat(B_hat, kx, ky)
        A_mode = Az_hat[ix0, iy1, iz0]
        Az_amp[it] = np.abs(A_mode)
        island_width[it] = compute_island_width_from_mode(A_mode, B0, a)

    w0 = island_width[0]
    wmax = np.nanmax(island_width)
    print(f"[INFO] w0 = {w0:.3e}, w_max = {wmax:.3e}")

    # ----- Linear fit: use automatic window selector ----- #
    mask_lin = select_linear_window(
        ts,
        island_width,
        w0=w0,
        min_pts=6,
        frac_sat=0.30,
        nwin=8,
    )

    if np.count_nonzero(mask_lin) < 5:
        print("[WARN] Automatic selector failed, falling back to first 25% of time.")
        mask_lin = ts < (ts[0] + 0.25*(ts[-1]-ts[0]))

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
    t_start = ts[0] + f_low * (ts[-1] - ts[0])
    t_end = ts[0] + f_high * (ts[-1] - ts[0])
    mask_ruth = (ts >= t_start) & (ts <= t_end)

    if np.count_nonzero(mask_ruth) < 5:
        print("[WARN] Too few points for Rutherford fit; "
              "using last half of time as fallback.")
        mask_ruth = ts >= (ts[0] + 0.5 * (ts[-1] - ts[0]))

    t_ruth = ts[mask_ruth]
    w_ruth = island_width[mask_ruth]
    dw_dt_R, b_R, dw_dt_R_R2, dw_dt_R_err = _linear_regression_with_stats(t_ruth, w_ruth)
    print(f"[RESULT] (dw/dt)_R = {dw_dt_R:.3e}, R²_R = {dw_dt_R_R2:.3f}, "
          f"σ_{'{'}dw/dt{'}'} = {dw_dt_R_err:.3e}")

    # ----- Saturated island width (last 20% of time) ----- #
    t_sat_min = ts[0] + 0.8 * (ts[-1] - ts[0])
    mask_sat = ts >= t_sat_min
    w_sat_samples = island_width[mask_sat]
    w_sat = float(np.mean(w_sat_samples))
    w_sat_std = float(np.std(w_sat_samples))
    print(f"[RESULT] w_sat = {w_sat:.3e} ± {w_sat_std:.3e}")

    # ----- Per-run diagnostic plot: w/a and ln(w/a) with fit ----- #
    outdir = os.path.dirname(fname)
    w_over_a = island_width / a
    w_lin_over_a = w_lin / a
    lnw_lin_over_a = np.log(w_lin_over_a)
    # Fit line in normalized units for plotting
    t_fit_line = np.linspace(t_lin[0], t_lin[-1], 200)
    lnw_fit_line = a_lin * t_fit_line + b_lin
    w_fit_line_over_a = np.exp(lnw_fit_line) / a

    fig_diag, axes = plt.subplots(2, 1, sharex=True, figsize=(5.5, 6.0))

    # Top: w/a vs t
    ax_top = axes[0]
    ax_top.plot(ts, w_over_a, "-", label=r"$w/a$")
    ax_top.plot(ts[mask_lin], w_over_a[mask_lin], "o", ms=4,
                label=r"linear window")
    ax_top.plot(t_fit_line, w_fit_line_over_a, "--", label=r"exp fit")
    ax_top.set_ylabel(r"$w/a$")
    ax_top.set_title(r"Island width evolution")
    ax_top.grid(True, ls=":")
    ax_top.legend(loc="best")

    # Bottom: ln(w/a) vs t
    ax_bottom = axes[1]
    ax_bottom.plot(ts, np.log(w_over_a), "-", label=r"$\ln(w/a)$")
    ax_bottom.plot(t_lin, lnw_lin_over_a, "o", ms=4, label=r"fit points")
    ax_bottom.plot(t_fit_line, lnw_fit_line, "--", label=rf"fit: $\gamma={gamma_fit:.3e}$")
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
    }


# -----------------------------------------------------------------------------#
# CLI + driver
# -----------------------------------------------------------------------------#

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a multi-parameter tearing scan and build Loureiro-style plots."
    )

    # Scan parameters
    p.add_argument(
        "--scan-a",
        type=float,
        nargs="+",
        default=[0.25, 0.35, 0.45],
        help="List of current-sheet half-widths a to scan "
             "(defaults chosen to give positive Δ' and span typical tearing values).",
    )
    p.add_argument(
        "--scan-eta",
        type=float,
        nargs="+",
        default=[5e-4, 1e-3, 2e-3],
        help="List of resistivities η to scan "
             "(defaults give S ~ 10^2–10^3 for B0=1, suitable for Rutherford scaling tests).",
    )

    # Grid and box
    p.add_argument("--Nx", type=int, default=48)
    p.add_argument("--Ny", type=int, default=48)
    p.add_argument("--Nz", type=int, default=48)
    p.add_argument("--Lx", type=float, default=2.0 * math.pi)
    p.add_argument("--Ly", type=float, default=2.0 * math.pi)
    p.add_argument("--Lz", type=float, default=2.0 * math.pi)

    # Physical parameters
    p.add_argument("--nu", type=float, default=1e-3)
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--Bg", type=float, default=0.2)
    p.add_argument("--epsB", type=float, default=0.01)

    # Time integration
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=100.0)
    p.add_argument("--n-frames", type=int, default=80)
    p.add_argument("--dt0", type=float, default=None)
    p.add_argument("--force-rerun", action="store_true",
                   help="Re-run simulations even if NPZ files already exist.")

    # Fitting windows (kept for CLI completeness, not used in auto selector)
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
        default=(0.4, 0.9),
        metavar=("F_START", "F_END"),
        help=("Fractional window [F_START,F_END] of total time used for "
              "Rutherford fit (default: 0.4 0.9)."),
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

    # Build list of (a, eta) combinations
    combos: List[Tuple[float, float]] = []
    for a in args.scan_a:
        for eta in args.scan_eta:
            combos.append((a, eta))

    print("[INFO] Scan combinations (a, eta):")
    for a, eta in combos:
        print(f"   a={a:.4g}, eta={eta:.4g}")

    results = []

    for idx, (a, eta) in enumerate(combos):
        tag = f"a{a:.3g}_eta{eta:.3g}"
        tag = tag.replace(".", "p").replace("-", "m")
        outfile = os.path.join(args.outdir, f"mhd_tearing_solution_{tag}.npz")

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
                eps_B=args.epsB,
                t0=args.t0,
                t1=args.t1,
                n_frames=args.n_frames,
                dt0=args.dt0,
                outfile=outfile,
            )

        # Postprocess
        res = analyze_single_run(
            outfile,
            args.lin_tmin,
            args.lin_tmax,
            tuple(args.ruth_frac),
        )
        results.append(res)

    # Convert to arrays
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

    # Normalized saturated width
    w_sat_over_a_arr = w_sat_arr / a_arr
    w_sat_over_a_std_arr = w_sat_std_arr / a_arr

    # Save summary
    summary_path = os.path.join(args.outdir, "tearing_scan_summary.npz")
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
    )
    print(f"\n[SAVE] Summary saved to {summary_path}")

    # ------------------------------------------------------------------ #
    # Plot 1: γ_fit vs γ_FKR (with error bars)
    # ------------------------------------------------------------------ #
    fig1, ax1 = plt.subplots()
    ax1.errorbar(gamma_FKR_arr, gamma_fit_arr, yerr=gamma_fit_err_arr,
                 fmt="o", capsize=3, label=r"runs")

    gmin = 0.5 * np.min(gamma_FKR_arr)
    gmax = 2.0 * np.max(gamma_FKR_arr)
    ref = np.linspace(gmin, gmax, 100)
    ax1.loglog(ref, ref, "k--", label=r"$\gamma_{\rm fit}=\gamma_{\rm FKR}$")

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
    ax1.set_title(r"Linear growth: $\gamma_{\rm fit}$ vs FKR theory")
    ax1.grid(True, which="both", ls=":")
    ax1.legend(loc="best")
    fig1.savefig(os.path.join(args.outdir, "scan_gamma_fit_vs_FKR.png"))
    plt.close(fig1)
    print("[SAVE] scan_gamma_fit_vs_FKR.png")

    # ------------------------------------------------------------------ #
    # Plot 2: Rutherford scaling: (dw/dt)_R vs η Δ' (with error bars)
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots()
    ax2.errorbar(etaDelta_arr, dw_dt_R_arr, yerr=dw_dt_R_err_arr,
                 fmt="o", capsize=3, label=r"runs")

    logx = np.log(etaDelta_arr)
    logy = np.log(dw_dt_R_arr)
    a_fit, b_fit = np.polyfit(logx, logy, 1)
    xfit = np.linspace(etaDelta_arr.min() * 0.8, etaDelta_arr.max() * 1.2, 200)
    yfit = np.exp(b_fit) * xfit**a_fit
    ax2.loglog(xfit, yfit, "k--", label=rf"fit: slope={a_fit:.2f}")

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
    ax2.set_title(r"Rutherford scaling")
    ax2.grid(True, which="both", ls=":")
    ax2.legend(loc="best")
    fig2.savefig(os.path.join(args.outdir, "scan_Rutherford_dw_dt_vs_etaDelta.png"))
    plt.close(fig2)
    print("[SAVE] scan_Rutherford_dw_dt_vs_etaDelta.png")

    # ------------------------------------------------------------------ #
    # Plot 3: Saturated island width vs Δ' (normalized, with error bars)
    # ------------------------------------------------------------------ #
    fig3, ax3 = plt.subplots()
    ax3.errorbar(Delta_prime_arr, w_sat_over_a_arr, yerr=w_sat_over_a_std_arr,
                 fmt="o", capsize=3, label=r"runs")

    logx2 = np.log(Delta_prime_arr)
    logy2 = np.log(w_sat_over_a_arr)
    a_fit2, b_fit2 = np.polyfit(logx2, logy2, 1)
    xfit2 = np.linspace(Delta_prime_arr.min() * 0.8, Delta_prime_arr.max() * 1.2, 200)
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
    ax3.set_title(r"Saturated island width vs $\Delta'$ (normalized)")
    ax3.grid(True, which="both", ls=":")
    ax3.legend(loc="best")
    fig3.savefig(os.path.join(args.outdir, "scan_wsat_over_a_vs_Deltaprime.png"))
    plt.close(fig3)
    print("[SAVE] scan_wsat_over_a_vs_Deltaprime.png")

    # ------------------------------------------------------------------ #
    # Plot 4: gamma_fit/gamma_FKR vs S and vs Delta'
    # ------------------------------------------------------------------ #
    ratio_arr = gamma_fit_arr / gamma_FKR_arr

    # (a) ratio vs S
    fig4a, ax4a = plt.subplots()
    ax4a.semilogx(S_arr, ratio_arr, "o")
    ax4a.set_xlabel(r"$S$")
    ax4a.set_ylabel(r"$\gamma_{\rm fit}/\gamma_{\rm FKR}$")
    ax4a.set_title(r"Departure from FKR theory vs Lundquist number")
    ax4a.grid(True, which="both", ls=":")
    fig4a.savefig(os.path.join(args.outdir, "scan_gamma_ratio_vs_S.png"))
    plt.close(fig4a)
    print("[SAVE] scan_gamma_ratio_vs_S.png")

    # (b) ratio vs Delta'
    fig4b, ax4b = plt.subplots()
    ax4b.semilogx(Delta_prime_arr, ratio_arr, "o")
    ax4b.set_xlabel(r"$\Delta'$")
    ax4b.set_ylabel(r"$\gamma_{\rm fit}/\gamma_{\rm FKR}$")
    ax4b.set_title(r"Departure from FKR theory vs $\Delta'$")
    ax4b.grid(True, which="both", ls=":")
    fig4b.savefig(os.path.join(args.outdir, "scan_gamma_ratio_vs_Deltaprime.png"))
    plt.close(fig4b)
    print("[SAVE] scan_gamma_ratio_vs_Deltaprime.png")

    print("\n[DONE] Scan analysis complete.")
    print("      Each point index in the plots corresponds to:")
    for i, name in enumerate(fnames):
        print(f"        {i}: {name}")


if __name__ == "__main__":
    main()
