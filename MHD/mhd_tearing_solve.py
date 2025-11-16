#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_solve.py

Incompressible pseudo-spectral MHD in a periodic box
for a Harris-sheet tearing mode test case.

This script:
  - builds the grid and spectral operators,
  - initializes a Harris-sheet equilibrium + perturbation,
  - runs the MHD equations with diffrax,
  - saves the full solution (Fourier coefficients) and metadata to a .npz file.

You can later post-process with:
  - mhd_tearing_postprocess.py
  - mhd_reconnection_rate.py
"""

from __future__ import annotations

import math
import argparse
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import diffrax as dfx
import numpy as np

# -----------------------------------------------------------------------------#
# Grid & spectral tools
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

# -----------------------------------------------------------------------------#
# Projection operator
# -----------------------------------------------------------------------------#

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

# -----------------------------------------------------------------------------#
# Gradient & directional derivatives
# -----------------------------------------------------------------------------#

def grad_from_hat(f_hat, kx, ky, kz):
    """
    Gradient of a scalar field from its Fourier coefficients.
    """
    df_dx_hat = 1j * kx * f_hat
    df_dy_hat = 1j * ky * f_hat
    df_dz_hat = 1j * kz * f_hat
    df_dx = jnp.fft.ifftn(df_dx_hat, axes=(0, 1, 2)).real
    df_dy = jnp.fft.ifftn(df_dy_hat, axes=(0, 1, 2)).real
    df_dz = jnp.fft.ifftn(df_dz_hat, axes=(0, 1, 2)).real
    return df_dx, df_dy, df_dz

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

# -----------------------------------------------------------------------------#
# Initial equilibrium & perturbation: Harris sheet
# -----------------------------------------------------------------------------#

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

# -----------------------------------------------------------------------------#
# Curl & flux function helpers
# -----------------------------------------------------------------------------#

def curl_from_hat(B_hat, kx, ky, kz):
    """
    Compute J = ∇×B from Fourier coefficients of B.
    """
    Bx_hat, By_hat, Bz_hat = B_hat[0], B_hat[1], B_hat[2]

    Jx_hat = 1j * (ky * Bz_hat - kz * By_hat)
    Jy_hat = 1j * (kz * Bx_hat - kx * Bz_hat)
    Jz_hat = 1j * (kx * By_hat - ky * Bx_hat)

    J_hat = jnp.stack([Jx_hat, Jy_hat, Jz_hat], axis=0)
    J = jnp.fft.ifftn(J_hat, axes=(1, 2, 3)).real
    return J

def compute_Az_from_hat(B_hat, kx, ky):
    """
    Compute A_z such that (B_x, B_y) = (-∂A_z/∂y, ∂A_z/∂x).
    Only uses kx, ky (perpendicular).
    """
    Bx_hat, By_hat = B_hat[0], B_hat[1]
    k_perp2 = kx**2 + ky**2
    k_perp2 = jnp.where(k_perp2 == 0.0, 1.0, k_perp2)

    Az_hat = 1j * (kx * By_hat - ky * Bx_hat) / k_perp2
    Az_hat = jnp.where(k_perp2 == 0.0, 0.0, Az_hat)

    Az = jnp.fft.ifftn(Az_hat, axes=(0, 1, 2)).real
    return Az

# -----------------------------------------------------------------------------#
# Energies & dissipation
# -----------------------------------------------------------------------------#

def energy_from_hat(v_hat, B_hat, Lx, Ly, Lz):
    v = jnp.fft.ifftn(v_hat, axes=(1, 2, 3)).real
    B = jnp.fft.ifftn(B_hat, axes=(1, 2, 3)).real
    dv = (Lx * Ly * Lz) / (v[0].size)

    v2 = jnp.sum(v * v, axis=0)
    B2 = jnp.sum(B * B, axis=0)
    E_kin = 0.5 * jnp.sum(v2) * dv
    E_mag = 0.5 * jnp.sum(B2) * dv
    return E_kin, E_mag

def dissipation_rates(v_hat, B_hat, k2, nu, eta, Lx, Ly, Lz):
    Nx = v_hat.shape[1]
    Ny = v_hat.shape[2]
    Nz = v_hat.shape[3]
    Npoints = Nx * Ny * Nz

    volume = Lx * Ly * Lz
    factor = volume / (Npoints**2)  # Parseval factor

    v_power = jnp.sum(jnp.abs(v_hat)**2, axis=0)  # sum over components
    B_power = jnp.sum(jnp.abs(B_hat)**2, axis=0)

    eps_visc = nu * factor * jnp.sum(k2 * v_power)
    eps_ohm  = eta * factor * jnp.sum(k2 * B_power)
    return eps_visc, eps_ohm

# -----------------------------------------------------------------------------#
# Tearing amplitude diagnostic
# -----------------------------------------------------------------------------#

def tearing_amplitude(B_hat, Lx, Ly, Lz, band_width_frac=0.25):
    """
    RMS of Bx in a band around the current sheet (|x-Lx/2| < band_width_frac*Lx/2).
    """
    B = jnp.fft.ifftn(B_hat, axes=(1, 2, 3)).real
    Bx = B[0]

    Nx = Bx.shape[0]
    x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    xc = 0.5 * Lx
    band_half = band_width_frac * 0.5 * Lx

    mask = (jnp.abs(x - xc)[:, None, None] < band_half)
    Bx_band = jnp.where(mask, Bx, 0.0)

    num = jnp.sum(Bx_band**2)
    den = jnp.sum(mask.astype(jnp.float64)) + 1e-16
    rms = jnp.sqrt(num / den)
    return float(rms)

# -----------------------------------------------------------------------------#
# RHS builder
# -----------------------------------------------------------------------------#

def make_mhd_rhs(nu, eta, kx, ky, kz, k2, mask_dealias):

    def rhs(t, y_hat, args_unused):
        v_hat, B_hat = y_hat

        v_hat = v_hat * mask_dealias
        B_hat = B_hat * mask_dealias

        v_hat = project_div_free(v_hat, kx, ky, kz, k2)
        B_hat = project_div_free(B_hat, kx, ky, kz, k2)

        v = jnp.fft.ifftn(v_hat, axes=(1, 2, 3)).real
        B = jnp.fft.ifftn(B_hat, axes=(1, 2, 3)).real

        grad_v = grad_vec_from_hat(v_hat, kx, ky, kz)
        grad_B = grad_vec_from_hat(B_hat, kx, ky, kz)

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
        dv_hat_dt = Nv_hat + nu * lap_factor * v_hat
        dB_hat_dt = NB_hat + eta * lap_factor * B_hat

        return (dv_hat_dt, dB_hat_dt)

    return jax.jit(rhs)

# -----------------------------------------------------------------------------#
# FKR-like theoretical estimate
# -----------------------------------------------------------------------------#

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
# Main driver
# -----------------------------------------------------------------------------#

def parse_args():
    p = argparse.ArgumentParser(
        description="Harris-sheet tearing mode MHD solver (pseudo-spectral)."
    )
    p.add_argument("--Nx", type=int, default=48)
    p.add_argument("--Ny", type=int, default=48)
    p.add_argument("--Nz", type=int, default=48)
    p.add_argument("--Lx", type=float, default=2.0 * math.pi)
    p.add_argument("--Ly", type=float, default=2.0 * math.pi)
    p.add_argument("--Lz", type=float, default=2.0 * math.pi)
    p.add_argument("--nu", type=float, default=1e-3)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--a", type=float, default=None,
                   help="current sheet half-width (default Lx/16)")
    p.add_argument("--Bg", type=float, default=0.2, help="guide field B_g")
    p.add_argument("--epsB", type=float, default=0.01,
                   help="perturbation amplitude in A_z")
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=100.0)
    p.add_argument("--n_frames", type=int, default=80)
    p.add_argument("--dt0", type=float, default=None,
                   help="initial dt; if None, estimated via CFL")
    p.add_argument("--outfile", type=str,
                   default="mhd_tearing_solution.npz",
                   help="output .npz file with solution and metadata")
    return p.parse_args()

def main():
    args = parse_args()

    Nx, Ny, Nz = args.Nx, args.Ny, args.Nz
    Lx, Ly, Lz = args.Lx, args.Ly, args.Lz
    nu, eta = args.nu, args.eta
    B0, a, B_g, eps_B = args.B0, args.a, args.Bg, args.epsB
    t0, t1, n_frames = args.t0, args.t1, args.n_frames

    if a is None:
        a = Lx / 16.0

    print("=== Incompressible pseudo-spectral MHD Parameters ===")
    print(f"Nx,Ny,Nz = {Nx},{Ny},{Nz}")
    print(f"Lx,Ly,Lz = {Lx},{Ly},{Lz}")
    print(f"nu={nu}, eta={eta}")
    print(f"B0={B0}, a={a}, B_g={B_g}, eps_B={eps_B}")
    print(f"t0={t0}, t1={t1}, n_frames={n_frames}")
    print("=====================================================")

    kx, ky, kz, k2, NX, NY, NZ = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    mask_dealias = make_dealias_mask(Nx, Ny, Nz, NX, NY, NZ)

    # Indices for the tearing mode (kx=0, ky=1, kz=0)
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

    # Time step
    if args.dt0 is None:
        dt_max = estimate_max_dt(v0_hat, B0_hat, Lx, Ly, Lz, nu, eta)
        print(f"[DT] Estimated dt_max from CFL/diffusion = {dt_max:.3e}")
        dt0 = min(1e-3, 0.5 * dt_max)
    else:
        dt0 = args.dt0

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

    # Minimal check: print final energies
    v_hat_end = jnp.array(v_hat_frames[-1])
    B_hat_end = jnp.array(B_hat_frames[-1])
    E_kin_end, E_mag_end = energy_from_hat(v_hat_end, B_hat_end, Lx, Ly, Lz)
    print(f"[FINAL] E_kin={float(E_kin_end):.6e}, "
          f"E_mag={float(E_mag_end):.6e}, "
          f"E_tot={float(E_kin_end+E_mag_end):.6e}")

    # Save everything needed for post-processing
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
    np.savez(args.outfile, **out)
    print(f"[SAVE] Solution saved to {args.outfile}")

if __name__ == "__main__":
    main()
