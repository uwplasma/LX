#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incompressible pseudo-spectral MHD in a periodic box
====================================================

Harris-sheet tearing mode test:
  B_y(x) = B0 * tanh((x-Lx/2)/a),  B_z = B_g, B_x = 0
  + small sinusoidal perturbation δB_x ~ sin(k_y y)

Diagnostics:
  - Energies and dissipation (E_kin, E_mag, E_tot, E_cons, eps_visc, eps_ohm)
  - Tearing-mode amplitude (RMS B_x near sheet) and growth rate
  - Current density J_z and flux function A_z (field lines)
  - Velocity magnitude and arrows
  - Several publication-ready plots
  - Movies of B_x, J_z, A_z-contours and velocity in the mid-plane z=0
"""

from __future__ import annotations

import math
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import diffrax as dfx
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

# -----------------------------------------------------------------------------#
# Grid & spectral tools
# -----------------------------------------------------------------------------#

def estimate_max_dt(v_hat, B_hat, Lx, Ly, Lz, nu, eta,
                    CFL_adv=0.4, CFL_diff=0.2):
    """
    Estimate a safe maximum timestep from CFL + diffusion constraints.

    v_hat, B_hat: (3, Nx, Ny, Nz), complex
    """
    # Real-space fields
    v = jnp.fft.ifftn(v_hat, axes=(1, 2, 3)).real
    B = jnp.fft.ifftn(B_hat, axes=(1, 2, 3)).real

    # Magnitudes
    v_mag = jnp.sqrt(jnp.sum(v * v, axis=0))
    B_mag = jnp.sqrt(jnp.sum(B * B, axis=0))

    # Characteristic max speed (advection + Alfvén, ρ=1)
    v_char = jnp.max(v_mag + B_mag)

    # Grid spacing
    Nx = v.shape[1]
    Ny = v.shape[2]
    Nz = v.shape[3]
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    hmin = jnp.min(jnp.array([dx, dy, dz]))

    # Advective CFL
    dt_adv = jnp.where(v_char > 0.0,
                       CFL_adv * hmin / v_char,
                       1e9)

    # Diffusive constraint
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
    k2 = jnp.where(k2 == 0.0, 1.0, k2)  # avoid divide-by-zero at k=0 mode
    return kx, ky, kz, k2, NX, NY, NZ

def make_dealias_mask(Nx, Ny, Nz, NX, NY, NZ):
    # 2/3 rule: keep modes with |n| <= N/3 in each direction
    kx_cut = Nx // 3
    ky_cut = Ny // 3
    kz_cut = Nz // 3
    mask = (
        (jnp.abs(NX) <= kx_cut) &
        (jnp.abs(NY) <= ky_cut) &
        (jnp.abs(NZ) <= kz_cut)
    )
    return mask.astype(jnp.complex128)  # multiplies Fourier fields

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
    df_dx = jnp.fft.ifftn(df_dx_hat, axes=(0,1,2)).real
    df_dy = jnp.fft.ifftn(df_dy_hat, axes=(0,1,2)).real
    df_dz = jnp.fft.ifftn(df_dz_hat, axes=(0,1,2)).real
    return df_dx, df_dy, df_dz

def grad_vec_from_hat(F_hat, kx, ky, kz):
    """
    Gradient of a vector field from Fourier coefficients.

    F_hat: (3, Nx, Ny, Nz) complex, components (F_x, F_y, F_z)

    Returns:
        grad_F: array of shape (3, 3, Nx, Ny, Nz)
                grad_F[i, j, ...] = ∂F_j / ∂x_i
                i = 0,1,2 -> x,y,z derivatives
                j = 0,1,2 -> components x,y,z
    """
    # derivative in Fourier space
    df_dx_hat = 1j * kx * F_hat
    df_dy_hat = 1j * ky * F_hat
    df_dz_hat = 1j * kz * F_hat

    # back to real space (batched over all 3 components)
    df_dx = jnp.fft.ifftn(df_dx_hat, axes=(1, 2, 3)).real  # (3,Nx,Ny,Nz)
    df_dy = jnp.fft.ifftn(df_dy_hat, axes=(1, 2, 3)).real
    df_dz = jnp.fft.ifftn(df_dz_hat, axes=(1, 2, 3)).real

    # grad_F[i,j,...] = ∂F_j / ∂x_i
    grad_F = jnp.stack([
        jnp.stack([df_dx[0], df_dx[1], df_dx[2]], axis=0),  # d/dx
        jnp.stack([df_dy[0], df_dy[1], df_dy[2]], axis=0),  # d/dy
        jnp.stack([df_dz[0], df_dz[1], df_dz[2]], axis=0),  # d/dz
    ], axis=0)
    return grad_F  # (3,3,Nx,Ny,Nz)

def directional_derivative_vec(A, grad_B):
    """
    Compute (A · ∇) B in real space.

    A:       (3, Nx, Ny, Nz)
    grad_B:  (3, 3, Nx, Ny, Nz) with grad_B[i,j,...] = ∂B_j/∂x_i

    Returns:
        adv: (3, Nx, Ny, Nz) with adv_j = Σ_i A_i ∂B_j/∂x_i
    """
    return jnp.einsum("i...,ij...->j...", A, grad_B)

# -----------------------------------------------------------------------------#
# Initial equilibrium & perturbation: Harris sheet
# -----------------------------------------------------------------------------#

def init_equilibrium(Nx, Ny, Nz, Lx, Ly, Lz):
    """
    Harris-sheet-like slab tearing equilibrium in a periodic box.

    Equilibrium:
      B_y(x) = B0 * tanh((x - Lx/2)/a)
      B_z    = B_g (guide field)
      B_x    = 0

    Perturbation:
      δA_z = eps_B * cos(k_y y)
      => δB_x = -eps_B * k_y * sin(k_y y)

    Initial velocity v = 0.
    """
    X, Y, Z = make_grid(Nx, Ny, Nz, Lx, Ly, Lz)

    # Equilibrium parameters
    B0 = 1.0        # reversing field amplitude
    a  = Lx / 16.0  # current sheet half-width
    B_g = 0.2       # guide field

    sx = (X - 0.5 * Lx) / a
    By0 = B0 * jnp.tanh(sx)
    Bx0 = jnp.zeros_like(By0)
    Bz0 = B_g * jnp.ones_like(By0)

    # Tearing perturbation (m_y = 1, 2D)
    m_y = 1
    m_z = 0
    k_y = 2.0 * jnp.pi * m_y / Ly
    k_z = 2.0 * jnp.pi * m_z / Lz

    eps_B = 0.01
    phase_y = k_y * Y
    phase_z = k_z * Z  # = 0 here

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
    J = jnp.fft.ifftn(J_hat, axes=(1,2,3)).real
    return J

def compute_Az_from_hat(B_hat, kx, ky):
    """
    Compute A_z such that (B_x, B_y) = (-∂A_z/∂y, ∂A_z/∂x).
    Only uses kx, ky (perpendicular).
    B_hat shape: (3, Nx, Ny, Nz)
    """
    Bx_hat, By_hat = B_hat[0], B_hat[1]          # (Nx, Ny, Nz)
    k_perp2 = kx**2 + ky**2                      # (Nx, Ny, Nz)
    k_perp2 = jnp.where(k_perp2 == 0.0, 1.0, k_perp2)

    # A_z(k) = i (k_x B_y - k_y B_x) / |k_perp|^2  up to gauge at k_perp=0
    Az_hat = 1j * (kx * By_hat - ky * Bx_hat) / k_perp2
    Az_hat = jnp.where(k_perp2 == 0.0, 0.0, Az_hat)  # gauge: A_z(k_perp=0)=0

    Az = jnp.fft.ifftn(Az_hat, axes=(0, 1, 2)).real
    return Az

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

        v = jnp.fft.ifftn(v_hat, axes=(1,2,3)).real
        B = jnp.fft.ifftn(B_hat, axes=(1,2,3)).real

        # batched gradients for v and B
        grad_v = grad_vec_from_hat(v_hat, kx, ky, kz)
        grad_B = grad_vec_from_hat(B_hat, kx, ky, kz)

        # (v·∇)v, (B·∇)B, etc.
        adv_v  = directional_derivative_vec(v, grad_v)
        strB_v = directional_derivative_vec(B, grad_B)

        adv_B  = directional_derivative_vec(v, grad_B)
        strv_B = directional_derivative_vec(B, grad_v)

        Nv = -adv_v + strB_v
        NB = -adv_B + strv_B

        Nv_hat = jnp.fft.fftn(Nv, axes=(1,2,3)) * mask_dealias
        NB_hat = jnp.fft.fftn(NB, axes=(1,2,3)) * mask_dealias

        Nv_hat = project_div_free(Nv_hat, kx, ky, kz, k2)

        lap_factor = -k2
        dv_hat_dt = Nv_hat + nu * lap_factor * v_hat
        dB_hat_dt = NB_hat + eta * lap_factor * B_hat

        return (dv_hat_dt, dB_hat_dt)

    return jax.jit(rhs)

# -----------------------------------------------------------------------------#
# Energies & dissipation
# -----------------------------------------------------------------------------#

def energy_from_hat(v_hat, B_hat, Lx, Ly, Lz):
    v = jnp.fft.ifftn(v_hat, axes=(1,2,3)).real
    B = jnp.fft.ifftn(B_hat, axes=(1,2,3)).real
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

    # NOTE: no factor 2 here – that was the bug
    eps_visc = nu * factor * jnp.sum(k2 * v_power)
    eps_ohm  = eta * factor * jnp.sum(k2 * B_power)
    return eps_visc, eps_ohm

# -----------------------------------------------------------------------------#
# Tearing amplitude diagnostic
# -----------------------------------------------------------------------------#

def tearing_amplitude(B_hat, Lx, Ly, Lz, band_width_frac=0.25):
    """
    RMS of Bx in a band around the current sheet (|x-Lx/2| < band_width_frac*Lx/2).
    This correlates well with island amplitude.
    """
    B = jnp.fft.ifftn(B_hat, axes=(1,2,3)).real
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
# Plotting / movie helpers
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
    Generic 2D movie helper.

    field_slices:   array (n_frames, Nx, Ny) to show as colored image.
    filename:       output .mp4 filename.
    ts:             1D array of times, length n_frames.
    Lx, Ly:         box size in x,y (for extent).
    title:          base title string.
    add_flux_contours: if True, overlay contours of flux_slices[i].
    flux_slices:    array (n_frames, Nx, Ny) with A_z or flux function.
    """
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
            # Remove old contours as a single Artist
            if cs is not None:
                cs.remove()
            # Draw new contours
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
# Main driver
# -----------------------------------------------------------------------------#

def main():
    Nx = Ny = Nz = 48
    Lx = Ly = Lz = 2.0 * math.pi

    nu = 1e-3
    eta = 1e-3

    t0, t1 = 0.0, 100.0
    n_frames = 80

    MAKE_MOVIES = True  # toggle if running on a slow machine

    print("=== Incompressible pseudo-spectral MHD Parameters ===")
    print(f"Nx,Ny,Nz = {Nx},{Ny},{Nz}")
    print(f"Lx,Ly,Lz = {Lx},{Ly},{Lz}")
    print(f"nu={nu}, eta={eta}")
    print(f"t0={t0}, t1={t1}")
    print(f"n_frames = {n_frames}")
    print("=====================================================")

    kx, ky, kz, k2, NX, NY, NZ = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    mask_dealias = make_dealias_mask(Nx, Ny, Nz, NX, NY, NZ)

    # --- Indices for the tearing mode (kx=0, ky=1, kz=0) in Fourier space --- #
    NX_np = np.array(NX)
    NY_np = np.array(NY)
    NZ_np = np.array(NZ)

    ix0 = int(np.where(NX_np[:, 0, 0] == 0)[0][0])
    iy1 = int(np.where(NY_np[0, :, 0] == 1)[0][0])   # m_y = 1
    iz0 = int(np.where(NZ_np[0, 0, :] == 0)[0][0])

    # --- Simple resistive tearing (FKR-like) theoretical estimate --- #
    B0 = 1.0
    a = Lx / 16.0
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
    print(f"[THEORY] FKR-like tearing estimate: gamma ≈ {gamma_theory:.3e}")

    v0_real, B0_real = init_equilibrium(Nx, Ny, Nz, Lx, Ly, Lz)

    v0_hat = jnp.fft.fftn(v0_real, axes=(1,2,3))
    B0_hat = jnp.fft.fftn(B0_real, axes=(1,2,3))

    v0_hat = v0_hat * mask_dealias
    B0_hat = B0_hat * mask_dealias
    v0_hat = project_div_free(v0_hat, kx, ky, kz, k2)
    B0_hat = project_div_free(B0_hat, kx, ky, kz, k2)

    E_kin0, E_mag0 = energy_from_hat(v0_hat, B0_hat, Lx, Ly, Lz)
    print(f"[INIT] E_kin0={float(E_kin0):.6e}, E_mag0={float(E_mag0):.6e}, "
          f"E_tot0={float(E_kin0+E_mag0):.6e}")
    
    # --- CFL / diffusion-based dt_max estimate ---
    dt_max = estimate_max_dt(v0_hat, B0_hat, Lx, Ly, Lz, nu, eta)
    print(f"[DT] Estimated dt_max from CFL/diffusion = {dt_max:.3e}")

    # Choose initial dt smaller than dt_max
    dt0 = min(1e-3, 0.5 * dt_max)
    print(f"[DT] Using dt0 = {dt0:.3e}")

    rhs = make_mhd_rhs(nu, eta, kx, ky, kz, k2, mask_dealias)
    term = dfx.ODETerm(rhs)

    solver = dfx.Dopri8()
    stepsize_controller = dfx.PIDController(
        rtol=1e-5,
        atol=1e-7,
    )
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
    print("[RUN] Solve finished. stats:", sol.stats)

    ts = np.array(sol.ts)
    v_hat_frames, B_hat_frames = sol.ys

    # ------------------------ Energies & diagnostics ------------------------ #

    E_kin_list = []
    E_mag_list = []
    E_tot_list = []
    eps_visc_list = []
    eps_ohm_list = []
    E_cons_list = []
    tearing_amp_list = []
    mode_amp_list = []   # |Bx_hat(kx=0, ky=1, kz=0)|
    v_rms_list = []
    v_max_list = []

    E_cons_running = 0.0
    E_cons0 = None

    for i in range(len(ts)):
        v_hat_i = v_hat_frames[i] * mask_dealias
        B_hat_i = B_hat_frames[i] * mask_dealias
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

        # tearing amplitude (RMS in physical band)
        A_rms = tearing_amplitude(B_hat_i, Lx, Ly, Lz, band_width_frac=0.25)
        tearing_amp_list.append(A_rms)

        # tearing Fourier mode amplitude |Bx(kx=0, ky=1, kz=0)|
        Bx_hat_i = np.array(B_hat_i[0])
        mode_amp_list.append(float(np.abs(Bx_hat_i[ix0, iy1, iz0])))
        
        # velocity diagnostics
        v_i = np.fft.ifftn(np.array(v_hat_i), axes=(1, 2, 3)).real
        v_mag = np.sqrt(np.sum(v_i**2, axis=0))
        v_rms_list.append(float(np.sqrt(np.mean(v_mag**2))))
        v_max_list.append(float(np.max(v_mag)))

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
    rel_E_cons_err = (E_cons_arr - E_cons_arr[0]) / E_cons_arr[0]
    mode_amp_arr = np.array(mode_amp_list)
    v_rms_arr = np.array(v_rms_list)
    v_max_arr = np.array(v_max_list)

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
    ax2.plot(ts_np, eps_visc_arr + eps_ohm_arr, "--", label=r"$\epsilon_{\rm tot}$")
    ax2.set_xlabel("t")
    ax2.set_ylabel("Dissipation rate")
    ax2.set_title("Dissipation rates vs time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig("mhd_energy_invariants.png", bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Diagnostics saved to mhd_energy_invariants.png")

    # ------------------ Tearing mode & velocity scales plot ----------------- #

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=200)

    # Left: tearing mode amplitude (Fourier) vs time
    axs[0].plot(ts_np, mode_amp_arr)
    axs[0].set_xlabel("t")
    axs[0].set_ylabel(r"$|B_x(k_x=0,k_y=1,k_z=0)|$")
    axs[0].set_title("Tearing-mode Fourier amplitude")
    axs[0].grid(True, alpha=0.3)

    # Right: velocity scales vs time
    axs[1].plot(ts_np, v_rms_arr, label=r"$v_{\rm rms}$")
    axs[1].plot(ts_np, v_max_arr, "--", label=r"$v_{\max}$")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel(r"Velocity")
    axs[1].set_title("Reconnection outflow speeds")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    fig.tight_layout()
    fig.savefig("tearing_mode_velocity_scales.png", bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved tearing_mode_velocity_scales.png")

    # ---------------------- Tearing-mode diagnostics plot ------------------- #

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), dpi=200)

    # Panel 1: RMS Bx and single Fourier mode
    axs[0].plot(ts_np, tearing_amp_arr, label=r"${\rm RMS}\,B_x$")
    axs[0].plot(ts_np, mode_amp_arr, '--', label=r"$|B_x(k_x=0,k_y=1,k_z=0)|$")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Tearing amplitude")
    axs[0].legend()

    # ----------- Automatic linear-phase window detection + fit ------------- #
    log_mode = np.log(mode_amp_arr + 1e-16)

    A = mode_amp_arr
    # use the first nonzero time-point as A0
    A0 = A[1] if A.shape[0] > 1 else A[0]
    Amax = A.max()

    # heuristic: between 5× initial amplitude and 30% of saturation
    f_min = 5.0
    f_max = 0.30
    mask = (A > f_min * A0) & (A < f_max * Amax)

    idx_lin = np.where(mask)[0]
    if idx_lin.size < 3:
        # fallback: early third of the time series (excluding first two points)
        idx_lin = np.arange(2, max(5, len(ts_np)//3))

    i0, i1 = int(idx_lin[0]), int(idx_lin[-1])

    t_fit = ts_np[i0:i1+1]
    logA_fit = log_mode[i0:i1+1]

    # linear fit: log A ≈ logA0 + γ t
    coeffs = np.polyfit(t_fit, logA_fit, 1)
    gamma_fit = coeffs[0]
    logA_line = coeffs[1] + coeffs[0] * ts_np

    print(f"[FIT] Measured tearing gamma ≈ {gamma_fit:.3e}")
    if not np.isnan(gamma_theory):
        ratio = gamma_fit / gamma_theory
        print(f"[COMP] gamma_fit/gamma_theory ≈ {ratio:.3f}")

    # Panel 2: log mode amplitude, shaded fit region, and fitted line
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

    # --- Add text box comparing fit vs theory ---
    if not np.isnan(gamma_theory):
        ratio = gamma_fit / gamma_theory
        txt = (
            rf"$\gamma_\mathrm{{fit}} \approx {gamma_fit:.3e}$" + "\n" +
            rf"$\gamma_\mathrm{{FKR}} \approx {gamma_theory:.3e}$" + "\n" +
            rf"$\gamma_\mathrm{{fit}}/\gamma_\mathrm{{FKR}} \approx {ratio:.2f}$"
        )
        axs[1].text(
            0.05, 0.05, txt,
            transform=axs[1].transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Panel 3: invariant error
    axs[2].plot(ts_np, rel_E_cons_err)
    axs[2].set_xlabel("t")
    axs[2].set_ylabel(r"$(E_{\rm cons}-E_{\rm cons}(0))/E_{\rm cons}(0)$")
    axs[2].set_title("Energy-invariant relative error")
    axs[2].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("tearing_mode_diagnostics.png", bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved tearing_mode_diagnostics.png")

    # ---------------------------- Snapshots plot ---------------------------- #

    # pick three characteristic times
    idxs = [0, len(ts_np)//2, len(ts_np)-1]
    labels = [f"t = {ts_np[i]:.2f}" for i in idxs]
    mid_z = Nz // 2

    fig, axs = plt.subplots(len(idxs), 3, figsize=(11, 3.6*len(idxs)), dpi=200)
    if len(idxs) == 1:
        axs = np.array([axs])

    for row, (i, lab) in enumerate(zip(idxs, labels)):
        v_hat_i = np.array(v_hat_frames[i])
        B_hat_i = np.array(B_hat_frames[i])

        B_i = np.fft.ifftn(B_hat_i, axes=(1,2,3)).real
        Bx = B_i[0, :, :, mid_z]

        J_i = curl_from_hat(jnp.array(B_hat_frames[i]), kx, ky, kz)
        J_i = np.array(J_i)
        Jz = J_i[2, :, :, mid_z]

        Az = compute_Az_from_hat(jnp.array(B_hat_frames[i]), kx, ky)
        Az = np.array(Az[ :, :, mid_z])

        im0 = axs[row, 0].imshow(Bx.T, origin="lower",
                                 extent=[0, Lx, 0, Ly], aspect="equal")
        axs[row, 0].set_title(r"$B_x(x,y,z=0)$, " + lab)
        axs[row, 0].set_ylabel("y")
        fig.colorbar(im0, ax=axs[row, 0])

        im1 = axs[row, 1].imshow(Jz.T, origin="lower",
                                 extent=[0, Lx, 0, Ly], aspect="equal")
        axs[row, 1].set_title(r"$J_z(x,y,z=0)$")
        fig.colorbar(im1, ax=axs[row, 1])

        cs = axs[row, 2].contour(Az.T, levels=25,
                                 extent=[0, Lx, 0, Ly])
        axs[row, 2].set_title(r"$A_z(x,y,z=0)$ (field lines)")
        axs[row, 2].set_xlim(0, Lx)
        axs[row, 2].set_ylim(0, Ly)
        axs[row, 2].set_aspect("equal")

        if row == len(idxs)-1:
            for c in axs[row, :]:
                c.set_xlabel("x")

    fig.tight_layout()
    fig.savefig("tearing_snapshots.png", bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved tearing_snapshots.png")

    # ---------------------------- Movies ------------------------------------ #

    if MAKE_MOVIES:
        print("[MOVIE] Building tearing-mode movies ...")
        mid_z = Nz // 2
        Bx_slices = []
        Jz_slices = []
        Az_slices = []

        for i in range(len(ts_np)):
            B_hat_i = B_hat_frames[i]              # (3, Nx, Ny, Nz)
            v_hat_i = v_hat_frames[i]

            # Real-space B and v
            B_i = np.fft.ifftn(np.array(B_hat_i), axes=(1, 2, 3)).real
            v_i = np.fft.ifftn(np.array(v_hat_i), axes=(1, 2, 3)).real

            # Current density J = ∇×B  (only Jz for tearing visualization)
            Bx_hat_i, By_hat_i, Bz_hat_i = B_hat_i[0], B_hat_i[1], B_hat_i[2]
            dBy_dx, dBy_dy, dBy_dz = grad_from_hat(By_hat_i, kx, ky, kz)
            dBx_dx, dBx_dy, dBx_dz = grad_from_hat(Bx_hat_i, kx, ky, kz)
            Jz_i = dBy_dx - dBx_dy

            # A_z from Bx, By
            Az_i = compute_Az_from_hat(jnp.array(B_hat_i), kx, ky)

            Bx_slices.append(B_i[0, :, :, mid_z])
            Jz_slices.append(Jz_i[:, :, mid_z])
            Az_slices.append(Az_i[:, :, mid_z])

        Bx_slices = np.array(Bx_slices)
        Jz_slices = np.array(Jz_slices)
        Az_slices = np.array(Az_slices)

        # Movies
        make_movie(
            Bx_slices,
            "mhd_tearing_Bx_xy.mp4",
            ts_np,
            Lx,
            Ly,
            title=r"$B_x(x,y,z=0)$",
        )

        make_movie(
            Jz_slices,
            "mhd_tearing_Jz_xy.mp4",
            ts_np,
            Lx,
            Ly,
            title=r"$J_z(x,y,z=0)$",
        )

        make_movie(
            Jz_slices,
            "mhd_tearing_flux_contours.mp4",
            ts_np,
            Lx,
            Ly,
            title=r"$J_z$ with flux contours",
            add_flux_contours=True,
            flux_slices=Az_slices,
            n_flux_levels=15,
        )

    print("[DONE] All diagnostics complete.")

if __name__ == "__main__":
    main()
