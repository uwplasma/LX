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

There are two modes for the equilibria:
  - original : classic tearing Harris sheet
  - forcefree: equilibrium-subtracted, plasmoid-like regime

Post-processing / ML codes can use the saved .npz, which now also contains
simple but useful diagnostic quantities such as:
  - Sweet–Parker estimates (v_A, S, delta_SP, v_in_SP, E_SP),
  - tearing-mode amplitude time traces,
  - an automatic estimate of the linear growth rate γ_fit,
  - reconnection-rate proxy from A_z at the X-point,
  - simple plasmoid / island count from extrema of A_z on a midplane.

The heavy lifting (MHD RHS + diagnostics) is JAX-based so that pieces of this
solver can be JIT-compiled and differentiated if desired.
"""

from __future__ import annotations

import os
import math
import argparse
from typing import Dict, Any

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import diffrax as dfx
import numpy as np
from jax import lax

Array = jnp.ndarray

# -----------------------------------------------------------------------------#
# Grid & spectral tools
# -----------------------------------------------------------------------------#

def estimate_max_dt(v_hat: Array,
                    B_hat: Array,
                    Lx: float,
                    Ly: float,
                    Lz: float,
                    nu: float,
                    eta: float,
                    CFL_adv: float = 0.4,
                    CFL_diff: float = 0.2) -> float:
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
    # Return as Python float (used only outside JIT)
    return float(dt_max)


def make_grid(Nx: int, Ny: int, Nz: int,
              Lx: float, Ly: float, Lz: float):
    x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0.0, Ly, Ny, endpoint=False)
    z = jnp.linspace(0.0, Lz, Nz, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z


def make_k_arrays(Nx: int, Ny: int, Nz: int,
                  Lx: float, Ly: float, Lz: float):
    """
    Build integer mode numbers and physical wavevectors.

    Note: for fftfreq(N)*N, the index of k=0 is 0, k=1 is 1, etc.
    """
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


def make_dealias_mask(Nx: int, Ny: int, Nz: int,
                      NX: Array, NY: Array, NZ: Array) -> Array:
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

def project_div_free(v_hat: Array,
                     kx: Array, ky: Array, kz: Array, k2: Array) -> Array:
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

def grad_from_hat(f_hat: Array,
                  kx: Array, ky: Array, kz: Array):
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


def grad_vec_from_hat(F_hat: Array,
                      kx: Array, ky: Array, kz: Array) -> Array:
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


def directional_derivative_vec(A: Array, grad_B: Array) -> Array:
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

def init_equilibrium(Nx: int, Ny: int, Nz: int,
                     Lx: float, Ly: float, Lz: float,
                     B0: float = 1.0, a: float | None = None,
                     B_g: float = 0.2, eps_B: float = 0.01,
                     m_y: int = 1, m_z: int = 0):
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

def curl_from_hat(B_hat: Array,
                  kx: Array, ky: Array, kz: Array) -> Array:
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


def compute_Az_from_hat(B_hat: Array,
                        kx: Array, ky: Array) -> Array:
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

def energy_from_hat(v_hat: Array,
                    B_hat: Array,
                    Lx: float, Ly: float, Lz: float):
    """
    Total kinetic and magnetic energies from Fourier-space fields.
    """
    v = jnp.fft.ifftn(v_hat, axes=(1, 2, 3)).real
    B = jnp.fft.ifftn(B_hat, axes=(1, 2, 3)).real
    dv = (Lx * Ly * Lz) / (v[0].size)

    v2 = jnp.sum(v * v, axis=0)
    B2 = jnp.sum(B * B, axis=0)
    E_kin = 0.5 * jnp.sum(v2) * dv
    E_mag = 0.5 * jnp.sum(B2) * dv
    return E_kin, E_mag


def dissipation_rates(v_hat: Array,
                      B_hat: Array,
                      k2: Array,
                      nu: float, eta: float,
                      Lx: float, Ly: float, Lz: float):
    """
    Viscous and Ohmic dissipation rates from Fourier-space fields.
    """
    v_power = jnp.sum(jnp.abs(v_hat)**2, axis=0)  # sum over components
    B_power = jnp.sum(jnp.abs(B_hat)**2, axis=0)

    volume = Lx * Ly * Lz
    Npoints = v_hat.shape[1] * v_hat.shape[2] * v_hat.shape[3]
    factor = volume / (Npoints**2)  # Parseval factor

    eps_visc = nu * factor * jnp.sum(k2 * v_power)
    eps_ohm  = eta * factor * jnp.sum(k2 * B_power)
    return eps_visc, eps_ohm

# -----------------------------------------------------------------------------#
# Tearing amplitude + diagnostics (JAX versions)
# -----------------------------------------------------------------------------#

def tearing_amplitude(B_hat: Array,
                          Lx: float, Ly: float, Lz: float,
                          band_width_frac: float = 0.25) -> Array:
    """
    RMS of Bx in a band around the current sheet (|x-Lx/2| < band_width_frac*Lx/2).
    Pure JAX version returning a JAX scalar.
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
    den = jnp.sum(mask.astype(Bx.dtype)) + 1e-16
    rms = jnp.sqrt(num / den)
    return rms


def tearing_mode_amplitude_from_hat(B_hat_frames: Array,
                                    ix0: int, iy1: int, iz0: int) -> Array:
    """
    Amplitude of the primary tearing mode from Fourier-space Bx:

      |B_x(kx=0, ky=1, kz=0)|(t)

    B_hat_frames: (T, 3, Nx, Ny, Nz)
    """
    Bx_mode = B_hat_frames[:, 0, ix0, iy1, iz0]
    return jnp.abs(Bx_mode)

# -----------------------------------------------------------------------------#
# Sweet–Parker and basic equilibrium scalings
# -----------------------------------------------------------------------------#

def sweet_parker_metrics(B0: float, a: float, eta: float) -> Dict[str, Array]:
    """
    Simple Sweet–Parker-like scalings based on current-sheet half-width a.

    We take:
      v_A = B0   (ρ = 1)
      S_a = a v_A / η
      δ_SP ~ a / sqrt(S_a)
      v_in_SP ~ v_A / sqrt(S_a)
      E_SP ~ v_in_SP * B0

    These are order-of-magnitude estimates meant for ML/optimization targets.
    """
    vA = B0
    S_a = a * vA / eta
    S_a = jnp.asarray(S_a)
    vA = jnp.asarray(vA)
    delta_SP = a / jnp.sqrt(S_a + 1e-30)
    v_in_SP = vA / jnp.sqrt(S_a + 1e-30)
    E_SP = v_in_SP * B0
    return dict(
        vA=vA,
        S_a=S_a,
        delta_SP=delta_SP,
        v_in_SP=v_in_SP,
        E_SP=E_SP,
    )

# -----------------------------------------------------------------------------#
# Growth-rate estimator (JAX, robust-ish)
# -----------------------------------------------------------------------------#

def estimate_growth_rate(
    ts: jnp.ndarray,
    mode_amp: jnp.ndarray,
    w0: jnp.ndarray,
    *,
    lower_factor: float = 5.0,
    upper_frac_of_max: float = 0.3,
    min_points: int = 8,
):
    """
    JAX-friendly growth-rate estimate.

    - Picks an amplitude window:
          lower_factor * w0 <= |B_x| <= upper_frac_of_max * max(|B_x|)
      to isolate the exponential phase.

    - Performs masked linear regression of ln|B_x| vs t using weighted sums
      (no boolean indexing), so it works under jit/vmap.

    Returns:
        gamma_fit, lnA_fit_full(ts), mask_lin
    """

    ts = jnp.asarray(ts)
    A = jnp.asarray(mode_amp)
    eps = 1e-30

    # Build amplitude-based mask
    wmax = jnp.max(A)
    lower = lower_factor * w0
    upper = upper_frac_of_max * wmax
    mask = (A >= lower) & (A <= upper)         # bool[T]
    w = mask.astype(ts.dtype)                  # 0/1 weights, same shape as ts

    def _fit_weighted(args):
        ts_local, A_local, w_local = args

        lnA = jnp.log(jnp.maximum(A_local, eps))

        S_w  = jnp.sum(w_local) + 1e-30
        S_t  = jnp.sum(w_local * ts_local)
        S_y  = jnp.sum(w_local * lnA)
        S_tt = jnp.sum(w_local * ts_local * ts_local)
        S_ty = jnp.sum(w_local * ts_local * lnA)

        denom = (S_tt - S_t * S_t / S_w) + 1e-30
        gamma = (S_ty - S_t * S_y / S_w) / denom
        c = (S_y - gamma * S_t) / S_w

        lnA_fit_full = gamma * ts_local + c
        return gamma, lnA_fit_full, mask

    def _fit_all(args):
        ts_local, A_local, _ = args
        lnA = jnp.log(jnp.maximum(A_local, eps))
        w_local = jnp.ones_like(ts_local, dtype=ts_local.dtype)

        S_w  = jnp.sum(w_local)
        S_t  = jnp.sum(w_local * ts_local)
        S_y  = jnp.sum(w_local * lnA)
        S_tt = jnp.sum(w_local * ts_local * ts_local)
        S_ty = jnp.sum(w_local * ts_local * lnA)

        denom = (S_tt - S_t * S_t / S_w) + 1e-30
        gamma = (S_ty - S_t * S_y / S_w) / denom
        c = (S_y - gamma * S_t) / S_w

        lnA_fit_full = gamma * ts_local + c
        mask_all = jnp.ones_like(ts_local, dtype=bool)
        return gamma, lnA_fit_full, mask_all

    # How many masked points?
    n_points = jnp.sum(w).astype(jnp.int32)

    gamma, lnA_fit_full, mask_lin = lax.cond(
        n_points >= min_points,
        _fit_weighted,
        _fit_all,
        operand=(ts, A, w),
    )

    return gamma, lnA_fit_full, mask_lin

# -----------------------------------------------------------------------------#
# Reconnection and island / plasmoid diagnostics
# -----------------------------------------------------------------------------#

def reconnection_rate_from_Az(ts: Array, Az_xpt: Array) -> Array:
    """
    Reconnection-rate proxy from the time derivative of the flux function A_z
    at the X-point:

      E_rec(t) ~ - d/dt A_z(x_X, y_X, z_X)

    Here x_X, y_X, z_X are fixed grid points near the X-point.
    """
    dAz_dt = jnp.gradient(Az_xpt, ts)
    E_rec = -dAz_dt
    return E_rec


def count_local_extrema_1d(f: Array) -> Array:
    """
    Count local extrema (minima + maxima) in a 1D array f(j).

    This is a crude plasmoid / island-count proxy.
    """
    df = jnp.diff(f)
    sign = jnp.sign(df)
    # A sign change in df indicates a local extremum
    sign_prod = sign[:-1] * sign[1:]
    extrema_mask = sign_prod < 0.0
    return jnp.sum(extrema_mask.astype(jnp.int32))

def plasmoid_complexity_metric(A_mid: Array) -> Array:
    """
    Smooth proxy for plasmoid / island complexity on a 1D cut of A_z.

    Instead of a discrete island count (which is non-differentiable), we use
    the mean-squared curvature of A(z or y):

        C = ⟨ (∂²A/∂s²)^2 ⟩_s

    where s is the coordinate along the cut (e.g., y on the midplane).

    This quantity is:
      * JAX- and autodiff-friendly,
      * larger when the profile contains more fine-scale structure / plasmoids.
    """
    A_mid = jnp.asarray(A_mid)
    s = jnp.arange(A_mid.shape[0], dtype=A_mid.dtype)

    # First and second derivatives via finite differences
    dA_ds = jnp.gradient(A_mid, s)
    d2A_ds2 = jnp.gradient(dA_ds, s)

    return jnp.mean(d2A_ds2**2)


# -----------------------------------------------------------------------------#
# RHS builders
# -----------------------------------------------------------------------------#

def make_mhd_rhs_original(nu: float, eta: float,
                          kx: Array, ky: Array, kz: Array, k2: Array,
                          mask_dealias: Array):
    """Original incompressible MHD RHS (no equilibrium subtraction)."""
    def rhs(t, y_hat, args_unused):
        v_hat, B_hat = y_hat

        v_hat_ = v_hat * mask_dealias
        B_hat_ = B_hat * mask_dealias

        # Project to div-free
        v_hat_ = project_div_free(v_hat_, kx, ky, kz, k2)
        B_hat_ = project_div_free(B_hat_, kx, ky, kz, k2)

        # Real-space fields
        v = jnp.fft.ifftn(v_hat_, axes=(1, 2, 3)).real
        B = jnp.fft.ifftn(B_hat_, axes=(1, 2, 3)).real

        grad_v = grad_vec_from_hat(v_hat_, kx, ky, kz)
        grad_B = grad_vec_from_hat(B_hat_, kx, ky, kz)

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
        dv_hat_dt = Nv_hat + nu * lap_factor * v_hat_
        dB_hat_dt = NB_hat + eta * lap_factor * B_hat_

        return (dv_hat_dt, dB_hat_dt)

    return jax.jit(rhs)


def make_mhd_rhs_forcefree(nu: float, eta: float,
                           kx: Array, ky: Array, kz: Array, k2: Array,
                           mask_dealias: Array,
                           B0_hat_eq: Array, Nv0_hat_eq: Array):
    """Equilibrium-subtracted RHS so (v=0, B=B0) is a solution."""
    lap_factor = -k2

    def rhs(t, y_hat, args_unused):
        v_hat, B_hat = y_hat

        v_hat_ = v_hat * mask_dealias
        B_hat_ = B_hat * mask_dealias

        # Project to div-free
        v_hat_ = project_div_free(v_hat_, kx, ky, kz, k2)
        B_hat_ = project_div_free(B_hat_, kx, ky, kz, k2)

        # Real-space fields
        v = jnp.fft.ifftn(v_hat_, axes=(1, 2, 3)).real
        B = jnp.fft.ifftn(B_hat_, axes=(1, 2, 3)).real

        grad_v = grad_vec_from_hat(v_hat_, kx, ky, kz)
        grad_B = grad_vec_from_hat(B_hat_, kx, ky, kz)

        adv_v  = directional_derivative_vec(v, grad_v)
        strB_v = directional_derivative_vec(B, grad_B)

        adv_B  = directional_derivative_vec(v, grad_B)
        strv_B = directional_derivative_vec(B, grad_v)

        Nv = -adv_v + strB_v
        NB = -adv_B + strv_B

        Nv_hat = jnp.fft.fftn(Nv, axes=(1, 2, 3)) * mask_dealias
        NB_hat = jnp.fft.fftn(NB, axes=(1, 2, 3)) * mask_dealias

        Nv_hat = project_div_free(Nv_hat, kx, ky, kz, k2)

        # subtract equilibrium force
        Nv_hat = Nv_hat - Nv0_hat_eq

        # evolve δB = B - B0
        dB_hat_dt = NB_hat + eta * lap_factor * (B_hat_ - B0_hat_eq)
        dv_hat_dt = Nv_hat + nu * lap_factor * v_hat_

        return (dv_hat_dt, dB_hat_dt)

    return jax.jit(rhs)

# -----------------------------------------------------------------------------#
# FKR-like theoretical estimate
# -----------------------------------------------------------------------------#

def fkr_gamma(B0: float, a: float, Ly: float, eta: float):
    """
    Simple FKR-like tearing estimate for the k_y = 1 mode.

    Returns
    -------
    gamma_fkr : float
    S_a       : float (Lundquist number based on sheet half-width a)
    Delta_p_a : float (Δ' a)
    """
    ky_val = 2.0 * math.pi / Ly   # m_y = 1
    ka = ky_val * a
    Delta_prime_a = 2.0 * (1.0/ka - ka)
    vA = B0  # ρ = 1
    S_a = a * vA / eta
    C_fkr = 0.55
    if Delta_prime_a > 0.0:
        gamma_theory = C_fkr * vA / a * (Delta_prime_a**(4.0/5.0)) * (S_a**(-3.0/5.0))
    else:
        gamma_theory = float("nan")
    return gamma_theory, S_a, Delta_prime_a

# -----------------------------------------------------------------------------#
# NumPy post-processing helpers for diagnostics loading
# -----------------------------------------------------------------------------#

def compute_k_arrays_np(Nx: int, Ny: int, Nz: int,
                        Lx: float, Ly: float, Lz: float):
    """NumPy version of make_k_arrays (k-space) for postprocessing."""
    nx = np.fft.fftfreq(Nx) * Nx
    ny = np.fft.fftfreq(Ny) * Ny
    nz = np.fft.fftfreq(Nz) * Nz
    NX, NY, NZ = np.meshgrid(nx, ny, nz, indexing="ij")

    kx = 2.0 * np.pi * NX / Lx
    ky = 2.0 * np.pi * NY / Ly
    kz = 2.0 * np.pi * NZ / Lz
    return kx, ky, kz, NX, NY, NZ


def energy_from_hat_np(v_hat, B_hat, Lx: float, Ly: float, Lz: float):
    """NumPy version of energy_from_hat, used in load_tearing_diagnostics."""
    v = np.fft.ifftn(v_hat, axes=(1, 2, 3)).real
    B = np.fft.ifftn(B_hat, axes=(1, 2, 3)).real
    dv = (Lx * Ly * Lz) / v[0].size

    v2 = np.sum(v * v, axis=0)
    B2 = np.sum(B * B, axis=0)
    E_kin = 0.5 * np.sum(v2) * dv
    E_mag = 0.5 * np.sum(B2) * dv
    return E_kin, E_mag


def compute_Az_hat_np(B_hat, kx, ky):
    """
    Fourier-space A_z such that (B_x, B_y) = (-∂A_z/∂y, ∂A_z/∂x).
    NumPy analogue of compute_Az_from_hat, but keeps Az_hat in k-space.
    """
    Bx_hat, By_hat = B_hat[0], B_hat[1]
    k_perp2 = kx**2 + ky**2
    k_perp2_safe = np.where(k_perp2 == 0.0, 1.0, k_perp2)

    Az_hat = 1j * (kx * By_hat - ky * Bx_hat) / k_perp2_safe
    Az_hat = np.where(k_perp2 == 0.0, 0.0, Az_hat)
    return Az_hat


def compute_island_width_from_mode(A_mode, B0: float, a: float) -> float:
    """
    Simple slab-tearing estimate converting A_mode -> island half-width w.

    For a Harris sheet By ~ B0 tanh(x/a),
      dBy/dx|_{x=0} ~ B0/a.

    In a standard slab-tearing scaling we can take roughly:
        w ~ 4 * sqrt(|A_mode| / |dBy/dx|_0).

    You can tweak this formula if your previous code used a slightly
    different normalization, but this keeps everything in one place.
    """
    dBydx0 = B0 / a
    return 4.0 * np.sqrt(np.abs(A_mode) / (np.abs(dBydx0) + 1e-30))

# -----------------------------------------------------------------------------#
# Diagnostic loading from .npz file
# -----------------------------------------------------------------------------#
def load_tearing_diagnostics(fname: str) -> Dict[str, Any]:
    """
    Unified loader for tearing diagnostics from a solution .npz.

    It provides a backwards/forwards-compatible interface that both
    mhd_tearing_scan.py and mhd_tearing_ml_v2.py can use.

    Returns a dict with at least:
      - ts
      - Nx,Ny,Nz, Lx,Ly,Lz
      - nu, eta, B0, a, eps_B
      - S (Lundquist # based on sheet width), gamma_FKR, Delta_prime_a
      - ix0,iy1,iz0, equilibrium_mode
      - island_width(t): preferred tearing-width proxy (w(t))
      - Az_amp(t): |A_z(kx=0,ky=1,kz=0)|
      - E_kin(t), E_mag(t)
      - gamma_fit (from solver, if present; NaN otherwise)
    """
    data = np.load(fname, allow_pickle=True)

    # Basic parameters
    ts = np.array(data["ts"])
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

    gamma_FKR = float(np.array(data["gamma_FKR"])) if "gamma_FKR" in data.files else np.nan
    Delta_prime_a = float(np.array(data["Delta_prime_a"])) if "Delta_prime_a" in data.files else np.nan

    ix0 = int(data["ix0"]) if "ix0" in data.files else 0
    iy1 = int(data["iy1"]) if "iy1" in data.files else 1
    iz0 = int(data["iz0"]) if "iz0" in data.files else 0

    # Equilibrium mode: "original" or "forcefree"
    eq_mode = data.get("equilibrium_mode", "original")
    if isinstance(eq_mode, np.ndarray):
        eq_mode = eq_mode.item()
    eq_mode = str(eq_mode)

    # --- Lundquist number S: accommodate old/new NPZs -----------------
    if "S" in data.files:
        S = float(np.array(data["S"]))
    elif "S_sheet" in data.files:
        S = float(np.array(data["S_sheet"]))
    elif "S_a" in data.files:
        S = float(np.array(data["S_a"]))
    else:
        # Fallback: a vA / eta with vA=B0, for Harris sheet (ρ=1)
        S = float(a * B0 / eta)

    # --- k arrays for any spectral fallback work ----------------------
    kx, ky, kz, NX, NY, NZ = compute_k_arrays_np(Nx, Ny, Nz, Lx, Ly, Lz)

    # ------------------------------------------------------------------
    # Island width w(t):
    #   1) If solver saved "island_width", use it.
    #   2) Else if solver saved "mode_amp_series", use that as w(t) proxy.
    #   3) Else reconstruct from B_hat using A_z tearing mode.
    # ------------------------------------------------------------------
    if "island_width" in data.files:
        island_width = np.array(data["island_width"])
    elif "mode_amp_series" in data.files:
        # Directly use |B_x(kx=0,ky=1,kz=0)| as tearing amplitude proxy
        island_width = np.array(data["mode_amp_series"])
    else:
        print("[load_tearing_diagnostics] island_width/mode_amp_series not found; "
              "reconstructing from B_hat and A_z mode.")
        B_hat_frames = data["B_hat"]
        n_t = ts.size
        island_width = np.zeros(n_t)
        for it in range(n_t):
            B_hat = B_hat_frames[it]
            Az_hat = compute_Az_hat_np(B_hat, kx, ky)
            A_mode = Az_hat[ix0, iy1, iz0]
            island_width[it] = compute_island_width_from_mode(A_mode, B0, a)

    # ------------------------------------------------------------------
    # A_z mode amplitude: |A_z(kx=0,ky=1,kz=0)|(t)
    # ------------------------------------------------------------------
    if "Az_mode_amp" in data.files:
        Az_amp = np.array(data["Az_mode_amp"])
    else:
        B_hat_frames = data["B_hat"]
        n_t = ts.size
        Az_amp = np.zeros(n_t)
        for it in range(n_t):
            B_hat = B_hat_frames[it]
            Az_hat = compute_Az_hat_np(B_hat, kx, ky)
            Az_amp[it] = np.abs(Az_hat[ix0, iy1, iz0])

    # ------------------------------------------------------------------
    # Energies: prefer full time traces if present; otherwise reconstruct.
    # ------------------------------------------------------------------
    if "E_kin" in data.files and "E_mag" in data.files:
        E_kin_arr = np.array(data["E_kin"])
        E_mag_arr = np.array(data["E_mag"])
    else:
        print("[load_tearing_diagnostics] E_kin/E_mag not saved; "
              "recomputing from v_hat, B_hat.")
        v_hat_frames = data["v_hat"]
        B_hat_frames = data["B_hat"]
        n_t = ts.size
        E_kin_arr = np.zeros(n_t)
        E_mag_arr = np.zeros(n_t)
        for it in range(n_t):
            Ek, Em = energy_from_hat_np(v_hat_frames[it], B_hat_frames[it], Lx, Ly, Lz)
            E_kin_arr[it] = Ek
            E_mag_arr[it] = Em

    # Growth rate from solver, if present
    if "gamma_fit" in data.files:
        gamma_fit = float(np.array(data["gamma_fit"]))
    else:
        gamma_fit = np.nan

    return dict(
        ts=ts,
        Nx=Nx, Ny=Ny, Nz=Nz,
        Lx=Lx, Ly=Ly, Lz=Lz,
        nu=nu, eta=eta,
        B0=B0, a=a, eps_B=eps_B,
        S=S,
        gamma_FKR=gamma_FKR,
        Delta_prime_a=Delta_prime_a,
        ix0=ix0, iy1=iy1, iz0=iz0,
        equilibrium_mode=eq_mode,
        island_width=island_width,
        Az_amp=Az_amp,
        E_kin=E_kin_arr,
        E_mag=E_mag_arr,
        gamma_fit=gamma_fit,
        B_hat=data["B_hat"] if "B_hat" in data.files else None,
    )

# -----------------------------------------------------------------------------#
# Main driver (JAX core + I/O wrapper)
# -----------------------------------------------------------------------------#

def parse_args():
    p = argparse.ArgumentParser(
        description="Harris-sheet tearing mode MHD solver (pseudo-spectral)."
    )
    p.add_argument("--Nx", type=int, default=52)
    p.add_argument("--Ny", type=int, default=52)
    p.add_argument("--Nz", type=int, default=52)
    p.add_argument("--Lx", type=float, default=2.0 * math.pi)
    p.add_argument("--Ly", type=float, default=2.0 * math.pi)
    p.add_argument("--Lz", type=float, default=2.0 * math.pi)
    p.add_argument("--nu", type=float, default=1e-3)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--a", type=float, default=None,
                   help="current sheet half-width (default Lx/16)")
    p.add_argument("--Bg", type=float, default=0.2, help="guide field B_g")
    p.add_argument("--epsB", type=float, default=1e-2,
                   help="perturbation amplitude in A_z")
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=50.0)
    p.add_argument("--n_frames", type=int, default=140)
    p.add_argument("--dt0", type=float, default=None,
                   help="initial dt; if None, estimated via CFL")
    p.add_argument("--outfile", type=str,
                   default="mhd_tearing_solution.npz",
                   help="output .npz file with solution and metadata")
    p.add_argument(
        "--equilibrium-mode",
        type=str,
        choices=["original", "forcefree"],
        default="original",
        help="Choose equilibrium model: 'original' or 'forcefree' (subtract equilibrium force)."
    )
    return p.parse_args()

# --- Core JAX simulation + diagnostics (no disk I/O) ------------------------#

def _run_tearing_simulation_and_diagnostics(
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
    dt0: float,
    equilibrium_mode: str = "original",
) -> Dict[str, Any]:
    """
    Core JAX-based simulation + diagnostics.

    Returns a dictionary of JAX arrays (ts, v_hat_frames, B_hat_frames)
    and diagnostic quantities (tearing amplitudes, growth rate, etc.).
    """
    # Spectral stuff
    kx, ky, kz, k2, NX, NY, NZ = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    mask_dealias = make_dealias_mask(Nx, Ny, Nz, NX, NY, NZ)

    # Indices for the tearing mode (kx=0, ky=1, kz=0) in fftfreq convention
    ix0 = 0          # kx = 0
    iy1 = 1          # ky = 1
    iz0 = 0          # kz = 0

    # Theory
    gamma_fkr, S_a, Delta_prime_a = fkr_gamma(B0, a, Ly, eta)

    # Equilibrium + perturbation
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

    if equilibrium_mode == "forcefree":
        # precompute equilibrium "force" P[(B0·∇)B0]
        grad_B0 = grad_vec_from_hat(B0_hat, kx, ky, kz)
        strB_v0 = directional_derivative_vec(B0_real, grad_B0)
        Nv0_hat = jnp.fft.fftn(strB_v0, axes=(1, 2, 3)) * mask_dealias
        Nv0_hat = project_div_free(Nv0_hat, kx, ky, kz, k2)

        B0_hat_const = B0_hat
        Nv0_hat_const = Nv0_hat
        rhs = make_mhd_rhs_forcefree(
            nu, eta, kx, ky, kz, k2, mask_dealias,
            B0_hat_const, Nv0_hat_const
        )
    else:
        rhs = make_mhd_rhs_original(
            nu, eta, kx, ky, kz, k2, mask_dealias
        )

    E_kin0, E_mag0 = energy_from_hat(v0_hat, B0_hat, Lx, Ly, Lz)

    # Time stepping with diffrax
    term = dfx.ODETerm(rhs)
    solver = dfx.Dopri8()
    stepsize_controller = dfx.PIDController(rtol=1e-7, atol=1e-7)
    ts_save = jnp.linspace(t0, t1, n_frames)
    saveat = dfx.SaveAt(ts=ts_save)

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
        progress_meter=dfx.TqdmProgressMeter()
    )

    ts = sol.ts  # (T,)
    v_hat_frames, B_hat_frames = sol.ys  # each (T, 3, Nx, Ny, Nz)
    
    # Energies over time (time series)
    def energy_single(v_hat_t, B_hat_t):
        Ek, Em = energy_from_hat(v_hat_t, B_hat_t, Lx, Ly, Lz)
        return Ek, Em

    E_kin_series, E_mag_series = jax.vmap(energy_single)(v_hat_frames, B_hat_frames)
    # Final energies (can also be taken from the series)
    E_kin0 = E_kin_series[0]
    E_mag0 = E_mag_series[0]
    E_kin_end = E_kin_series[-1]
    E_mag_end = E_mag_series[-1]

    # Final energies
    v_hat_end = v_hat_frames[-1]
    B_hat_end = B_hat_frames[-1]
    E_kin_end, E_mag_end = energy_from_hat(v_hat_end, B_hat_end, Lx, Ly, Lz)

    # Sweet–Parker metrics (sheet-based)
    sp = sweet_parker_metrics(B0=B0, a=a, eta=eta)

    # Tearing amplitudes
    tearing_amp_series = jax.vmap(
        lambda Bh: tearing_amplitude(Bh, Lx, Ly, Lz)
    )(B_hat_frames)

    # Mode amplitude for kx=0, ky=1, kz=0
    mode_amp_series = tearing_mode_amplitude_from_hat(
        B_hat_frames, ix0=ix0, iy1=iy1, iz0=iz0
    )

    # Automatic growth-rate estimate γ_fit
    gamma_fit, lnA_fit, mask_lin = estimate_growth_rate(ts, mode_amp_series, w0=mode_amp_series[0])

    # Flux function A_z on all frames (for reconnection / islands)
    def Az_from_B(Bh):
        return compute_Az_from_hat(Bh, kx, ky)

    Az_frames = jax.vmap(Az_from_B)(B_hat_frames)  # (T, Nx, Ny, Nz)

    # X-point choice: mid-plane in x, y=0, z=0
    ix_mid = Nx // 2
    iy0 = 0
    iz0_real = 0
    Az_xpt_series = Az_frames[:, ix_mid, iy0, iz0_real]
    E_rec_series = reconnection_rate_from_Az(ts, Az_xpt_series)

    # Island / plasmoid count from A_z on midplane at final time
    Az_final_mid = Az_frames[-1, ix_mid, :, iz0_real]
    n_plasmoids_final = count_local_extrema_1d(Az_final_mid)
    complexity_final = plasmoid_complexity_metric(Az_final_mid)

    return dict(
        ts=ts,
        v_hat=v_hat_frames,
        B_hat=B_hat_frames,
        Nx=Nx, Ny=Ny, Nz=Nz,
        Lx=Lx, Ly=Ly, Lz=Lz,
        nu=nu, eta=eta,
        B0=B0, a=a, B_g=B_g, eps_B=eps_B,
        t0=t0, t1=t1,
        n_frames=n_frames,
        dt0=dt0,
        # Energies: time series + endpoints
        E_kin=E_kin_series,
        E_mag=E_mag_series,
        E_kin0=E_kin0,
        E_mag0=E_mag0,
        E_kin_end=E_kin_end,
        E_mag_end=E_mag_end,
        # FKR-like theory + SP metrics
        gamma_FKR=gamma_fkr,
        S_a=S_a,
        Delta_prime_a=Delta_prime_a,
        vA=sp["vA"],
        S_sheet=sp["S_a"],
        delta_SP=sp["delta_SP"],
        v_in_SP=sp["v_in_SP"],
        E_SP=sp["E_SP"],
        S=sp["S_a"],
        # Tearing / reconnection diagnostics
        mode_amp_series=mode_amp_series,
        tearing_amp_series=tearing_amp_series,
        gamma_fit=gamma_fit,
        lnA_fit=lnA_fit,
        mask_lin=mask_lin,
        Az_xpt_series=Az_xpt_series,
        E_rec_series=E_rec_series,
        n_plasmoids_final=n_plasmoids_final,
        Az_final_mid=Az_final_mid,
        complexity_final=complexity_final,
        ix0=ix0, iy1=iy1, iz0=iz0,
        equilibrium_mode=equilibrium_mode,
    )

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
    a: float | None,
    B_g: float,
    eps_B: float,
    t0: float,
    t1: float,
    n_frames: int,
    dt0: float | None,
    outfile: str,
    equilibrium_mode: str = "original",
) -> str:
    """
    Single Harris-sheet tearing run, used both by CLI (this file) and by
    the scan script.

    This wrapper:
      * handles default a and dt0,
      * calls the JAX core for the simulation + diagnostics,
      * converts to NumPy and saves an .npz to `outfile`,
      * returns the outfile path.

    The JAX core `_run_tearing_simulation_and_diagnostics` can be imported and
    used directly for JIT / autodiff-based optimization workflows.
    """
    if a is None:
        a = Lx / 16.0

    print("=== Incompressible pseudo-spectral MHD Parameters ===")
    print(f"Nx,Ny,Nz = {Nx},{Ny},{Nz}")
    print(f"Lx,Ly,Lz = {Lx},{Ly},{Lz}")
    print(f"nu={nu}, eta={eta}")
    print(f"B0={B0}, a={a}, B_g={B_g}, eps_B={eps_B}")
    print(f"t0={t0}, t1={t1}, n_frames={n_frames}")
    print(f"equilibrium_mode = {equilibrium_mode}")
    print("=====================================================")

    # If user did not override the default name, append equilibrium_mode
    default_out = "mhd_tearing_solution.npz"
    if outfile == default_out:
        stem, ext = os.path.splitext(default_out)
        outfile = f"{stem}_{equilibrium_mode}{ext}"
        print(f"[OUT] Using automatic outfile name: {outfile}")

    # Time step (host-side estimate is fine; not part of JAX/AD path)
    # Use an initial guess based on the equilibrium.
    # Build a tiny equilibrium in JAX to estimate dt if needed.
    if dt0 is None:
        # Build spectral operators and equilibrium for dt estimate
        kx, ky, kz, k2, NX, NY, NZ = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
        mask_dealias = make_dealias_mask(Nx, Ny, Nz, NX, NY, NZ)
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

        dt_max = estimate_max_dt(v0_hat, B0_hat, Lx, Ly, Lz, nu, eta)
        print(f"[DT] Estimated dt_max from CFL/diffusion = {dt_max:.3e}")
        dt0 = min(5e-4, 0.5 * dt_max)
    print(f"[DT] Using dt0 = {dt0:.3e}")

    # Run core JAX simulation + diagnostics
    res = _run_tearing_simulation_and_diagnostics(
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
        equilibrium_mode=equilibrium_mode,
    )

    # Convert JAX arrays to NumPy for saving
    out_np = {}
    for key, val in res.items():
        if isinstance(val, jnp.ndarray):
            out_np[key] = np.array(val)
        else:
            # Scalars / ints / floats
            out_np[key] = np.array(val)

    np.savez(outfile, **out_np)
    print(f"[SAVE] Solution + diagnostics saved to {outfile}")
    return outfile


def main():
    args = parse_args()

    solve_tearing_case(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        nu=args.nu,
        eta=args.eta,
        B0=args.B0,
        a=args.a,
        B_g=args.Bg,
        eps_B=args.epsB,
        t0=args.t0,
        t1=args.t1,
        n_frames=args.n_frames,
        dt0=args.dt0,
        outfile=args.outfile,
        equilibrium_mode=args.equilibrium_mode,
    )


if __name__ == "__main__":
    main()
