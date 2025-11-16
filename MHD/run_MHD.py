#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy pseudo-spectral MHD in cylindrical coordinates (R, theta, z)
================================================================

- Incompressible-ish resistive MHD with constant density rho=1.
- Geometry: cylindrical shell R ∈ [R_in, R_out],
             theta ∈ [0, 2π) (periodic),
             z ∈ [0, Lz) (periodic).

- Fields in cylindrical components:
    v = (v_R, v_theta, v_z)
    B = (B_R, B_theta, B_z)

- Derivatives:
    * Spectral (FFT) in theta, z.
    * 2nd-order finite differences in R.
    * Divergence, curl, and vector Laplacian use cylindrical formulas.

- Equations (dimensionless, mu0 = rho = 1):
    dv/dt = -(v·∇)v + (∇×B)×B + nu ∇²v
    dB/dt = ∇×(v×B) + eta ∇²B

  (No explicit pressure term; flow is "weakly compressible".)

- Diagnostics:
    * E_kin(t) = 0.5 ∫ |v|^2 dV
    * E_mag(t) = 0.5 ∫ |B|^2 dV
    * E_tot(t) = E_kin + E_mag
      with cylindrical volume element dV = R dR dtheta dz.

    * Divergence of B measured with cylindrical formula:
        divB = (1/R) ∂(R B_R)/∂R + (1/R) ∂B_theta/∂theta + ∂B_z/∂z

The script produces mhd_diagnostics_cylindrical.png with:
  - Left: E_kin, E_mag, E_tot vs time
  - Right: ||divB||_2 and ||divB||_∞ vs time

This is a **toy code**, not a production-quality tokamak/stellarator solver,
but it puts the previous box MHD into (R, theta, z) geometry with proper
cylindrical operators and pseudo-spectral treatment in the periodic directions.
"""

from __future__ import annotations

import math
import functools
from dataclasses import dataclass

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap

import diffrax as dfx
import matplotlib.pyplot as plt


# ---------------------------- Utility functions ---------------------------- #

def make_kvec(N: int, L: float) -> jnp.ndarray:
    """
    Spectral wavenumbers for periodic domain of length L with N points.
    """
    return 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=L / N)


def spectral_deriv_theta(f, k_theta):
    """∂f/∂theta using FFT along axis=1."""
    f_hat = jnp.fft.fft(f, axis=1)
    d_hat = f_hat * (1j * k_theta)[None, :, None]
    df = jnp.fft.ifft(d_hat, axis=1).real
    return df


def spectral_deriv_z(f, k_z):
    """∂f/∂z using FFT along axis=2."""
    f_hat = jnp.fft.fft(f, axis=2)
    d_hat = f_hat * (1j * k_z)[None, None, :]
    df = jnp.fft.ifft(d_hat, axis=2).real
    return df


def spectral_d2_theta(f, k_theta):
    """∂²f/∂theta² via FFT."""
    f_hat = jnp.fft.fft(f, axis=1)
    d2_hat = f_hat * (-(k_theta**2))[None, :, None]
    d2f = jnp.fft.ifft(d2_hat, axis=1).real
    return d2f


def spectral_d2_z(f, k_z):
    """∂²f/∂z² via FFT."""
    f_hat = jnp.fft.fft(f, axis=2)
    d2_hat = f_hat * (-(k_z**2))[None, None, :]
    d2f = jnp.fft.ifft(d2_hat, axis=2).real
    return d2f


def deriv_R(f, dR):
    """
    ∂f/∂R with 2nd-order finite differences in R (axis=0).
    One-sided 2nd-order at boundaries.
    """
    # central differences for interior
    df = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dR)
    # one-sided at boundaries
    df = df.at[0].set((-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * dR))
    df = df.at[-1].set((3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * dR))
    return df


def scalar_laplacian_cyl(f, R, dR, k_theta, k_z):
    """
    Scalar Laplacian in cylindrical coordinates for a scalar f(R, theta, z):

      ∇² f = (1/R) ∂/∂R (R ∂f/∂R)
             + (1/R²) ∂²f/∂theta²
             + ∂²f/∂z²
    """
    R3 = R[:, None, None]

    df_dR = deriv_R(f, dR)
    term_R = deriv_R(R3 * df_dR, dR) / R3
    term_theta = spectral_d2_theta(f, k_theta)[...] / (R3**2)
    term_z = spectral_d2_z(f, k_z)
    return term_R + term_theta + term_z


def vector_laplacian_cyl(v, R, dR, k_theta, k_z):
    """
    Vector Laplacian in cylindrical coordinates using the decomposition
    from e.g. Wikipedia:

      (∇²A)_R     = ∇²A_R     - A_R/R² - (2/R²) ∂A_θ/∂θ
      (∇²A)_θ     = ∇²A_θ     - A_θ/R² + (2/R²) ∂A_R/∂θ
      (∇²A)_z     = ∇²A_z

    where ∇² is the scalar Laplacian above.
    """
    R3 = R[:, None, None]

    A_R, A_th, A_z = v[..., 0], v[..., 1], v[..., 2]

    lap_AR = scalar_laplacian_cyl(A_R, R, dR, k_theta, k_z)
    lap_Ath = scalar_laplacian_cyl(A_th, R, dR, k_theta, k_z)
    lap_Az = scalar_laplacian_cyl(A_z, R, dR, k_theta, k_z)

    dAth_dtheta = spectral_deriv_theta(A_th, k_theta)
    dAR_dtheta = spectral_deriv_theta(A_R, k_theta)

    lap_R = lap_AR - A_R / (R3**2) - 2.0 * dAth_dtheta / (R3**2)
    lap_th = lap_Ath - A_th / (R3**2) + 2.0 * dAR_dtheta / (R3**2)
    lap_z = lap_Az

    return jnp.stack([lap_R, lap_th, lap_z], axis=-1)


def divergence_cyl(B, R, dR, k_theta, k_z):
    """
    Divergence in cylindrical coordinates:

      ∇·B = 1/R ∂(R B_R)/∂R + 1/R ∂B_θ/∂θ + ∂B_z/∂z
    """
    R3 = R[:, None, None]
    BR, Bth, Bz = B[..., 0], B[..., 1], B[..., 2]

    term_R = deriv_R(R3 * BR, dR) / R3
    term_th = spectral_deriv_theta(Bth, k_theta) / R3
    term_z = spectral_deriv_z(Bz, k_z)
    return term_R + term_th + term_z


def curl_cyl(A, R, dR, k_theta, k_z):
    """
    Curl in cylindrical coordinates:

      (∇×A)_R     = (1/R) ∂A_z/∂θ - ∂A_θ/∂z
      (∇×A)_θ     = ∂A_R/∂z - ∂A_z/∂R
      (∇×A)_z     = (1/R) ∂(R A_θ)/∂R - (1/R) ∂A_R/∂θ
    """
    R3 = R[:, None, None]
    A_R, A_th, A_z = A[..., 0], A[..., 1], A[..., 2]

    dAz_dtheta = spectral_deriv_theta(A_z, k_theta)
    dAth_dz = spectral_deriv_z(A_th, k_z)

    dAR_dz = spectral_deriv_z(A_R, k_z)
    dAz_dR = deriv_R(A_z, dR)

    dR_Ath_dR = deriv_R(R3 * A_th, dR)
    dAR_dtheta = spectral_deriv_theta(A_R, k_theta)

    curl_R = dAz_dtheta / R3 - dAth_dz
    curl_th = dAR_dz - dAz_dR
    curl_z = dR_Ath_dR / R3 - dAR_dtheta / R3

    return jnp.stack([curl_R, curl_th, curl_z], axis=-1)


def directional_derivative_cyl(A, B, R, dR, k_theta, k_z):
    """
    Cylindrical directional derivative (A·∇)B using the formula for
    the "Directional derivative" in cylindrical coordinates:

      (A·∇B)_R   = A_R ∂B_R/∂R + (A_θ/R) ∂B_R/∂θ + A_z ∂B_R/∂z - (A_θ B_θ)/R
      (A·∇B)_θ   = A_R ∂B_θ/∂R + (A_θ/R) ∂B_θ/∂θ + A_z ∂B_θ/∂z + (A_θ B_R)/R
      (A·∇B)_z   = A_R ∂B_z/∂R + (A_θ/R) ∂B_z/∂θ + A_z ∂B_z/∂z

    (See "Del in cylindrical and spherical coordinates", Wikipedia.)
    """
    R3 = R[:, None, None]

    A_R, A_th, A_z = A[..., 0], A[..., 1], A[..., 2]
    B_R, B_th, B_z = B[..., 0], B[..., 1], B[..., 2]

    dBR_dR = deriv_R(B_R, dR)
    dBR_dtheta = spectral_deriv_theta(B_R, k_theta)
    dBR_dz = spectral_deriv_z(B_R, k_z)

    dBth_dR = deriv_R(B_th, dR)
    dBth_dtheta = spectral_deriv_theta(B_th, k_theta)
    dBth_dz = spectral_deriv_z(B_th, k_z)

    dBz_dR = deriv_R(B_z, dR)
    dBz_dtheta = spectral_deriv_theta(B_z, k_theta)
    dBz_dz = spectral_deriv_z(B_z, k_z)

    adv_R = (
        A_R * dBR_dR
        + (A_th / R3) * dBR_dtheta
        + A_z * dBR_dz
        - (A_th * B_th) / R3
    )

    adv_th = (
        A_R * dBth_dR
        + (A_th / R3) * dBth_dtheta
        + A_z * dBth_dz
        + (A_th * B_R) / R3
    )

    adv_z = (
        A_R * dBz_dR
        + (A_th / R3) * dBz_dtheta
        + A_z * dBz_dz
    )

    return jnp.stack([adv_R, adv_th, adv_z], axis=-1)


def cross(a, b):
    """Vector cross product for last-dimension-3 arrays."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return jnp.stack([cx, cy, cz], axis=-1)


# --------------------------- Problem definition --------------------------- #

@dataclass
class CylindricalMHDParams:
    nu: float
    eta: float
    R: jnp.ndarray
    dR: float
    k_theta: jnp.ndarray
    k_z: jnp.ndarray


def mhd_rhs(t, state, args: CylindricalMHDParams):
    """RHS for cylindrical MHD ODE system."""
    v, B = state
    nu, eta = args.nu, args.eta
    R, dR, k_theta, k_z = args.R, args.dR, args.k_theta, args.k_z

    # Nonlinear advection term (v·∇)v in cylindrical
    adv_v = directional_derivative_cyl(v, v, R, dR, k_theta, k_z)

    # Magnetic terms
    J = curl_cyl(B, R, dR, k_theta, k_z)      # current J = ∇×B
    lorentz = cross(J, B)                     # (∇×B)×B

    # Diffusion
    lap_v = vector_laplacian_cyl(v, R, dR, k_theta, k_z)
    lap_B = vector_laplacian_cyl(B, R, dR, k_theta, k_z)

    dvdt = -adv_v + lorentz + nu * lap_v

    # Induction equation: dB/dt = ∇×(v×B) + eta ∇² B
    v_cross_B = cross(v, B)
    curl_v_cross_B = curl_cyl(v_cross_B, R, dR, k_theta, k_z)
    dBdt = curl_v_cross_B + eta * lap_B

    return (dvdt, dBdt)


mhd_rhs_jit = jit(mhd_rhs)


# ------------------------------ Main driver ------------------------------ #

def main():
    # Parameters (adjust to taste)
    Nr, Nth, Nz = 32, 32, 32
    R_in, R_out = 0.5, 1.5
    Lz = 2.0

    nu = 1e-3
    eta = 1e-3
    rho = 1.0

    t0, t1 = 0.0, 1.2
    dt0 = 1e-3
    n_frames = 40

    # Diagnostics printout
    print("========== Cylindrical pseudo-spectral MHD Parameters ==========")
    print(f"Nr,Ntheta,Nz = {Nr},{Nth},{Nz}")
    print(f"R_in,R_out,Lz = {R_in},{R_out},{Lz}")
    print(f"nu={nu}, eta={eta}, rho={rho}")
    print(f"t0={t0}, t1={t1}, dt0={dt0}")
    print("solver=tsit5, stepsize_controller=pid, rtol=1e-7, atol=1e-7, max_steps=20000")
    print("n_frames=", n_frames)
    print("================================================================")

    # Grid
    R = jnp.linspace(R_in, R_out, Nr)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, Nth, endpoint=False)
    z = jnp.linspace(0.0, Lz, Nz, endpoint=False)

    dR = float(R[1] - R[0])
    dtheta = float(theta[1] - theta[0])
    dz = float(z[1] - z[0])

    # Wavenumbers for spectral derivatives
    k_theta = make_kvec(Nth, 2.0 * math.pi)
    k_z = make_kvec(Nz, Lz)

    # 3D grids
    R3, TH3, Z3 = jnp.meshgrid(R, theta, z, indexing="ij")

    # Initial B: toroidal-ish field + small helical perturbation
    B0 = 0.5
    R0 = 1.0
    m = 1
    kz_mode = 2.0 * math.pi / Lz

    B_R0 = 0.02 * jnp.sin(m * TH3) * jnp.sin(kz_mode * Z3)
    B_th0 = B0 * R0 / R3
    B_z0 = 0.02 * jnp.cos(m * TH3) * jnp.cos(kz_mode * Z3)

    B0_field = jnp.stack([B_R0, B_th0, B_z0], axis=-1)

    # Initial v: small shear flow
    v0_amp = 0.2
    v_R0 = jnp.zeros_like(R3)
    v_th0 = v0_amp * jnp.sin(jnp.pi * (R3 - R_in) / (R_out - R_in))**2 * jnp.sin(2.0 * jnp.pi * Z3 / Lz)
    v_z0 = v0_amp * jnp.cos(m * TH3) * jnp.sin(2.0 * jnp.pi * Z3 / Lz)

    v0_field = jnp.stack([v_R0, v_th0, v_z0], axis=-1)

    # Diagnostic: initial energies and divB
    def energy(v, B):
        # Volume element: dV = R dR dtheta dz
        R3_local = R[:, None, None]
        dv = R3_local * dR * dtheta * dz
        e_kin = 0.5 * rho * jnp.sum(jnp.sum(v**2, axis=-1) * dv)
        e_mag = 0.5 * jnp.sum(jnp.sum(B**2, axis=-1) * dv)
        return e_kin, e_mag

    E_kin0, E_mag0 = energy(v0_field, B0_field)

    divB0 = divergence_cyl(B0_field, R, dR, k_theta, k_z)
    divB_L2_0 = jnp.sqrt(jnp.mean(divB0**2))
    divB_Linf_0 = jnp.max(jnp.abs(divB0))

    print(f"[INIT] E_kin0={E_kin0:.6e}, E_mag0={E_mag0:.6e}")
    print(f"[INIT] ||divB||_2={divB_L2_0:.3e}, ||divB||_∞={divB_Linf_0:.3e}")

    # Build args
    params = CylindricalMHDParams(
        nu=nu,
        eta=eta,
        R=R,
        dR=dR,
        k_theta=k_theta,
        k_z=k_z,
    )

    # ODE setup
    term = dfx.ODETerm(mhd_rhs_jit)
    solver = dfx.Tsit5()
    stepsize_controller = dfx.PIDController(rtol=1e-7, atol=1e-7)

    ts_save = jnp.linspace(t0, t1, n_frames)
    saveat = dfx.SaveAt(ts=ts_save)

    print("[RUN] Calling diffrax.diffeqsolve ...")
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=(v0_field, B0_field),
        args=params,
        saveat=saveat,
        max_steps=20000,
        stepsize_controller=stepsize_controller,
    )

    print("[RUN] Solve finished. stats:", sol.stats())

    # -------------------------- Diagnostics in time ------------------------- #

    ts = sol.ts
    v_frames, B_frames = zip(*sol.ys)

    E_kin_list = []
    E_mag_list = []
    E_tot_list = []
    divB_L2_list = []
    divB_Linf_list = []

    print("[POST] Computing diagnostic curves...")
    for i, (t, v_f, B_f) in enumerate(zip(ts, v_frames, B_frames)):
        E_kin, E_mag = energy(v_f, B_f)
        E_tot = E_kin + E_mag

        divB = divergence_cyl(B_f, R, dR, k_theta, k_z)
        divB_L2 = jnp.sqrt(jnp.mean(divB**2))
        divB_Linf = jnp.max(jnp.abs(divB))

        E_kin_list.append(float(E_kin))
        E_mag_list.append(float(E_mag))
        E_tot_list.append(float(E_tot))
        divB_L2_list.append(float(divB_L2))
        divB_Linf_list.append(float(divB_Linf))

        print(
            f"[POST] frame {i}/{len(ts)-1}, t={float(t):.4f}, "
            f"E_kin={E_kin:.3e}, E_mag={E_mag:.3e}, "
            f"||divB||_2={divB_L2:.3e}, ||divB||_∞={divB_Linf:.3e}"
        )

    E_kin_arr = jnp.array(E_kin_list)
    E_mag_arr = jnp.array(E_mag_list)
    E_tot_arr = jnp.array(E_tot_list)
    divB_L2_arr = jnp.array(divB_L2_list)
    divB_Linf_arr = jnp.array(divB_Linf_list)

    # ----------------------------- Plot results ----------------------------- #

    print("[PLOT] Saving diagnostics figure to mhd_diagnostics_cylindrical.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)

    # Energies
    ax = axes[0]
    ax.plot(ts, E_kin_arr, label=r"$E_{\rm kin}$")
    ax.plot(ts, E_mag_arr, label=r"$E_{\rm mag}$")
    ax.plot(ts, E_tot_arr, label=r"$E_{\rm tot}$", linestyle="--")
    ax.set_xlabel("t")
    ax.set_ylabel("Energy")
    ax.set_title("Cylindrical MHD Energies vs time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Divergence
    ax2 = axes[1]
    ax2.semilogy(ts, divB_L2_arr, label=r"$\|\nabla\cdot B\|_2$")
    ax2.semilogy(ts, divB_Linf_arr, label=r"$\|\nabla\cdot B\|_\infty$")
    ax2.set_xlabel("t")
    ax2.set_ylabel("Divergence norm")
    ax2.set_title("Divergence of B (cylindrical)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig("mhd_diagnostics_cylindrical.png", bbox_inches="tight")
    plt.close(fig)

    print("[DONE] Diagnostics figure written.")


if __name__ == "__main__":
    main()
