#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incompressible pseudo-spectral MHD in a periodic box
====================================================

- 3D periodic box (x,y,z) ∈ [0,Lx]×[0,Ly]×[0,Lz].
- Incompressible MHD:
    ∂v/∂t = -(v·∇)v + (B·∇)B - ∇p + ν ∇²v
    ∂B/∂t = (B·∇)v - (v·∇)B + η ∇²B
  with ∇·v = 0, ∇·B ≈ 0 (spectral projection).

- Numerical method:
    * Fourier pseudo-spectral in all directions
    * 2/3 de-aliasing rule
    * Projection in k-space to enforce incompressibility
    * JAX + Diffrax time stepping

- Diagnostics:
    * E_kin(t), E_mag(t), E_tot(t)
    * Dissipation rates eps_visc(t), eps_ohm(t)
    * "Conserved" total:
          E_cons(t) = E_tot(t) + ∫ (eps_visc+eps_ohm) dt
      which should be ~constant if everything is consistent.

This is a good baseline to:
    - test MHD stability of given equilibria,
    - see growth/decay of perturbations,
    - later plug in local Miller geometry / flux-tube approximations.
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
    "text.usetex": False,   # True if you have LaTeX installed
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
})

# ---------------------------- Grid & spectral tools ---------------------------- #

def make_grid(Nx, Ny, Nz, Lx, Ly, Lz):
    x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0.0, Ly, Ny, endpoint=False)
    z = jnp.linspace(0.0, Lz, Nz, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z

def make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz):
    # Integer mode indices
    nx = jnp.fft.fftfreq(Nx) * Nx
    ny = jnp.fft.fftfreq(Ny) * Ny
    nz = jnp.fft.fftfreq(Nz) * Nz
    NX, NY, NZ = jnp.meshgrid(nx, ny, nz, indexing="ij")

    # Physical wavenumbers (2π/L factor)
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
    return mask.astype(jnp.float64)  # multiplies Fourier fields

# ----------------------------- Projection operator ----------------------------- #

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

# -------------------------- Gradient & directional derivatives ----------------- #

def grad_from_hat(f_hat, kx, ky, kz):
    """
    Gradient of scalar field whose Fourier transform is f_hat.
    Returns real-space arrays (df/dx, df/dy, df/dz).
    """
    df_dx_hat = 1j * kx * f_hat
    df_dy_hat = 1j * ky * f_hat
    df_dz_hat = 1j * kz * f_hat
    df_dx = jnp.fft.ifftn(df_dx_hat, axes=(0,1,2)).real
    df_dy = jnp.fft.ifftn(df_dy_hat, axes=(0,1,2)).real
    df_dz = jnp.fft.ifftn(df_dz_hat, axes=(0,1,2)).real
    return df_dx, df_dy, df_dz

def directional_derivative(v, grad_fx, grad_fy, grad_fz):
    """
    v · ∇f for scalar f.
    v shape: (3, Nx, Ny, Nz)
    grad_f* shapes: (Nx, Ny, Nz)
    """
    vx, vy, vz = v[0], v[1], v[2]
    return vx * grad_fx + vy * grad_fy + vz * grad_fz

def directional_derivative_vec(A, grad_Bx, grad_By, grad_Bz):
    """
    (A·∇)B where A = (Ax,Ay,Az), B has components Bx,By,Bz.
    grad_Bx etc: tuples (dBx/dx, dBx/dy, dBx/dz), etc.
    Returns vector field with shape (3, Nx, Ny, Nz).
    """
    Ax, Ay, Az = A[0], A[1], A[2]

    dBx_dx, dBx_dy, dBx_dz = grad_Bx
    dBy_dx, dBy_dy, dBy_dz = grad_By
    dBz_dx, dBz_dy, dBz_dz = grad_Bz

    adv_x = Ax * dBx_dx + Ay * dBx_dy + Az * dBx_dz
    adv_y = Ax * dBy_dx + Ay * dBy_dy + Az * dBy_dz
    adv_z = Ax * dBz_dx + Ay * dBz_dy + Az * dBz_dz

    return jnp.stack([adv_x, adv_y, adv_z], axis=0)

# --------------------------- Initial equilibrium & perturbation ---------------- #

def init_equilibrium(Nx, Ny, Nz, Lx, Ly, Lz):
    """
    Harris-sheet-like slab tearing equilibrium in a periodic box.

    Equilibrium:
      B_y(x) = B0 * tanh((x - Lx/2)/a)
      B_z    = B_g (guide field)
      B_x    = 0

    Perturbation:
      δA_z = eps_B * cos(k_y y) * cos(k_z z)
      => δB_x = ∂δA_z/∂y = -eps_B * k_y * sin(k_y y) * cos(k_z z)
         δB_y = -∂δA_z/∂x = 0
         δB_z = 0

    Initial velocity v = 0.
    """
    X, Y, Z = make_grid(Nx, Ny, Nz, Lx, Ly, Lz)

    # Equilibrium parameters
    B0 = 1.0       # reversing field amplitude
    a  = Lx / 16.0 # current sheet half-width
    B_g = 0.2      # guide field

    # Harris-sheet B_y(x)
    sx = (X - 0.5 * Lx) / a
    By0 = B0 * jnp.tanh(sx)
    Bx0 = jnp.zeros_like(By0)
    Bz0 = B_g * jnp.ones_like(By0)

    # Tearing perturbation
    m_y = 1
    m_z = 0  # classical 2D tearing, no variation in z
    k_y = 2.0 * jnp.pi * m_y / Ly
    k_z = 2.0 * jnp.pi * m_z / Lz  # = 0

    eps_B = 0.01  # perturbation amplitude
    phase_y = k_y * Y
    phase_z = k_z * Z  # zero if m_z = 0

    # δA_z and δB
    # δA_z = eps_B * cos(k_y y) * cos(k_z z)
    # δB_x = ∂δA_z/∂y = -eps_B * k_y * sin(k_y y) * cos(k_z z)
    delta_Bx = -eps_B * k_y * jnp.sin(phase_y) * jnp.cos(phase_z)
    delta_By = jnp.zeros_like(delta_Bx)
    delta_Bz = jnp.zeros_like(delta_Bx)

    Bx = Bx0 + delta_Bx
    By = By0 + delta_By
    Bz = Bz0 + delta_Bz

    B0_real = jnp.stack([Bx, By, Bz], axis=0)

    # Initial velocity: at rest (small numerical noise + Lorentz force
    # will drive flows as tearing develops)
    v0_real = jnp.zeros_like(B0_real)

    return v0_real, B0_real

# --------------------------------- RHS builder -------------------------------- #

def make_mhd_rhs(nu, eta, kx, ky, kz, k2, mask_dealias):

    def rhs(t, y_hat, args_unused):
        """
        y_hat: (v_hat, B_hat)
          v_hat, B_hat have shape (3, Nx, Ny, Nz), complex
        """
        v_hat, B_hat = y_hat

        # Dealias: always keep fields within 2/3 rule
        v_hat = v_hat * mask_dealias
        B_hat = B_hat * mask_dealias

        # Enforce incompressibility on v (and also clean B a bit)
        v_hat = project_div_free(v_hat, kx, ky, kz, k2)
        B_hat = project_div_free(B_hat, kx, ky, kz, k2)

        # Real-space fields
        v = jnp.fft.ifftn(v_hat, axes=(1,2,3)).real
        B = jnp.fft.ifftn(B_hat, axes=(1,2,3)).real

        # Gradients of v and B components (via spectral derivatives)
        vx_hat, vy_hat, vz_hat = v_hat[0], v_hat[1], v_hat[2]
        Bx_hat, By_hat, Bz_hat = B_hat[0], B_hat[1], B_hat[2]

        dvx_dx, dvx_dy, dvx_dz = grad_from_hat(vx_hat, kx, ky, kz)
        dvy_dx, dvy_dy, dvy_dz = grad_from_hat(vy_hat, kx, ky, kz)
        dvz_dx, dvz_dy, dvz_dz = grad_from_hat(vz_hat, kx, ky, kz)

        dBx_dx, dBx_dy, dBx_dz = grad_from_hat(Bx_hat, kx, ky, kz)
        dBy_dx, dBy_dy, dBy_dz = grad_from_hat(By_hat, kx, ky, kz)
        dBz_dx, dBz_dy, dBz_dz = grad_from_hat(Bz_hat, kx, ky, kz)

        grad_vx = (dvx_dx, dvx_dy, dvx_dz)
        grad_vy = (dvy_dx, dvy_dy, dvy_dz)
        grad_vz = (dvz_dx, dvz_dy, dvz_dz)

        grad_Bx = (dBx_dx, dBx_dy, dBx_dz)
        grad_By = (dBy_dx, dBy_dy, dBy_dz)
        grad_Bz = (dBz_dx, dBz_dy, dBz_dz)

        # (v·∇)v and (B·∇)B
        adv_v = directional_derivative_vec(
            v, grad_vx, grad_vy, grad_vz
        )
        strB_v = directional_derivative_vec(
            B, grad_Bx, grad_By, grad_Bz
        )

        # (v·∇)B and (B·∇)v for induction equation
        adv_B = directional_derivative_vec(
            v, grad_Bx, grad_By, grad_Bz
        )
        strv_B = directional_derivative_vec(
            B, grad_vx, grad_vy, grad_vz
        )

        # Nonlinear terms (real space)
        Nv = -adv_v + strB_v          # RHS for v (before pressure projection)
        NB = -adv_B + strv_B          # RHS for B

        # Transform nonlinear terms to Fourier
        Nv_hat = jnp.fft.fftn(Nv, axes=(1,2,3))
        NB_hat = jnp.fft.fftn(NB, axes=(1,2,3))

        # Dealias nonlinear terms
        Nv_hat = Nv_hat * mask_dealias
        NB_hat = NB_hat * mask_dealias

        # Project velocity RHS to remove pressure (∇·v = 0)
        Nv_hat = project_div_free(Nv_hat, kx, ky, kz, k2)

        # Diffusion in Fourier space
        lap_factor = -k2
        dv_hat_dt = Nv_hat + nu * lap_factor * v_hat
        dB_hat_dt = NB_hat + eta * lap_factor * B_hat

        # Optional: clean B RHS a bit too
        # dB_hat_dt = project_div_free(dB_hat_dt, kx, ky, kz, k2)

        return (dv_hat_dt, dB_hat_dt)

    return jax.jit(rhs)

# -------------------------------- Energy diagnostics -------------------------- #

def energy_from_hat(v_hat, B_hat, Lx, Ly, Lz):
    """
    Compute kinetic and magnetic energy from Fourier fields using Parseval.
    Depending on FFT conventions, overall constant might differ by (NxNyNz),
    but relative conservation is what we care about.

    Here we just use real-space form for clarity.
    """
    v = jnp.fft.ifftn(v_hat, axes=(1,2,3)).real
    B = jnp.fft.ifftn(B_hat, axes=(1,2,3)).real
    dv = (Lx * Ly * Lz) / (v[0].size)  # volume / number of grid points

    v2 = jnp.sum(v * v, axis=0)
    B2 = jnp.sum(B * B, axis=0)
    E_kin = 0.5 * jnp.sum(v2) * dv
    E_mag = 0.5 * jnp.sum(B2) * dv
    return E_kin, E_mag

def dissipation_rates(v_hat, B_hat, k2, nu, eta, Lx, Ly, Lz):
    """
    Compute viscous and ohmic dissipation rates using Fourier representation:
      eps_visc = 2 ν ∫ |∇v|^2 dV
      eps_ohm  = 2 η ∫ |∇B|^2 dV

    With numpy/jax FFT conventions (ifft includes 1/N):
      ∑_x |f(x)|^2 = (1/Npoints) ∑_k |F(k)|^2

    Energy in energy_from_hat is:
      E = 0.5 * (Volume / Npoints) * ∑_x |f(x)|^2
        = 0.5 * (Volume / Npoints^2) * ∑_k |F(k)|^2

    So ∫ |∇v|^2 dV = (Volume / Npoints^2) * ∑_k k^2 |v_hat(k)|^2, etc.
    """
    Nx = v_hat.shape[1]
    Ny = v_hat.shape[2]
    Nz = v_hat.shape[3]
    Npoints = Nx * Ny * Nz

    volume = Lx * Ly * Lz
    factor = volume / (Npoints**2)

    v_power = jnp.sum(jnp.abs(v_hat)**2, axis=0)  # sum over components
    B_power = jnp.sum(jnp.abs(B_hat)**2, axis=0)

    eps_visc = 2.0 * nu * factor * jnp.sum(k2 * v_power)
    eps_ohm  = 2.0 * eta * factor * jnp.sum(k2 * B_power)
    return eps_visc, eps_ohm


# --------------------------------------- Main --------------------------------- #

def main():
    # Grid & physical parameters
    Nx = Ny = Nz = 32
    Lx = Ly = Lz = 2.0 * math.pi

    nu = 1e-3
    eta = 1e-3

    t0, t1 = 0.0, 5.0       # evolve long enough to see instability / decay
    n_frames = 80
    dt0 = 1e-3

    print("=== Incompressible pseudo-spectral MHD Parameters ===")
    print(f"Nx,Ny,Nz = {Nx},{Ny},{Nz}")
    print(f"Lx,Ly,Lz = {Lx},{Ly},{Lz}")
    print(f"nu={nu}, eta={eta}")
    print(f"t0={t0}, t1={t1}, dt0={dt0}")
    print(f"n_frames = {n_frames}")
    print("=====================================================")

    # Spectral arrays
    kx, ky, kz, k2, NX, NY, NZ = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    mask_dealias = make_dealias_mask(Nx, Ny, Nz, NX, NY, NZ)

    # Initial equilibrium & perturbation in real space
    v0_real, B0_real = init_equilibrium(Nx, Ny, Nz, Lx, Ly, Lz)

    # Transform to Fourier
    v0_hat = jnp.fft.fftn(v0_real, axes=(1,2,3))
    B0_hat = jnp.fft.fftn(B0_real, axes=(1,2,3))

    # Dealias & project divergence-free
    v0_hat = v0_hat * mask_dealias
    B0_hat = B0_hat * mask_dealias
    v0_hat = project_div_free(v0_hat, kx, ky, kz, k2)
    B0_hat = project_div_free(B0_hat, kx, ky, kz, k2)

    # Initial energy
    E_kin0, E_mag0 = energy_from_hat(v0_hat, B0_hat, Lx, Ly, Lz)
    print(f"[INIT] E_kin0={float(E_kin0):.6e}, E_mag0={float(E_mag0):.6e}, "
          f"E_tot0={float(E_kin0+E_mag0):.6e}")

    # Build RHS and ODE term
    rhs = make_mhd_rhs(nu, eta, kx, ky, kz, k2, mask_dealias)
    term = dfx.ODETerm(rhs)

    solver = dfx.Tsit5()
    stepsize_controller = dfx.ConstantStepSize()
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
    v_hat_frames, B_hat_frames = sol.ys  # shapes (n_frames, 3, Nx,Ny,Nz)

    # ---------------------- Energy & dissipation diagnostics ------------------ #

    E_kin_list = []
    E_mag_list = []
    E_tot_list = []
    eps_visc_list = []
    eps_ohm_list = []
    E_cons_list = []

    E_cons_running = 0.0

    for i in range(len(ts)):
        v_hat_i = v_hat_frames[i]
        B_hat_i = B_hat_frames[i]

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

        if i > 0:
            dt = ts[i] - ts[i-1]
            eps_prev = eps_visc_list[i-1] + eps_ohm_list[i-1]
            eps_curr = eps_visc_list[i]   + eps_ohm_list[i]
            E_cons_running += 0.5 * (eps_prev + eps_curr) * dt

        E_cons_list.append(float(E_tot_i + E_cons_running))

        print(
            f"[POST] frame {i}/{len(ts)-1}, t={ts[i]:.4f}, "
            f"E_kin={E_kin_i:.3e}, E_mag={E_mag_i:.3e}, E_tot={E_tot_i:.3e}, "
            f"eps_visc={eps_visc_i:.3e}, eps_ohm={eps_ohm_i:.3e}, "
            f"E_cons={E_tot_i + E_cons_running:.3e}"
        )

    # ------------------------------ Plot diagnostics -------------------------- #

    ts_np = ts
    E_kin_arr = np.array(E_kin_list)
    E_mag_arr = np.array(E_mag_list)
    E_tot_arr = np.array(E_tot_list)
    eps_visc_arr = np.array(eps_visc_list)
    eps_ohm_arr = np.array(eps_ohm_list)
    E_cons_arr = np.array(E_cons_list)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    ax1, ax2 = axs

    # Energies
    ax1.plot(ts_np, E_kin_arr, label=r"$E_{\rm kin}$")
    ax1.plot(ts_np, E_mag_arr, label=r"$E_{\rm mag}$")
    ax1.plot(ts_np, E_tot_arr, "--", label=r"$E_{\rm tot}$")
    ax1.plot(ts_np, E_cons_arr, "-.", label=r"$E_{\rm cons}$")
    ax1.set_xlabel("t")
    ax1.set_ylabel("Energy")
    ax1.set_title("MHD energies and invariant")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Dissipation rates
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
    
    # --------------------------- Tearing-mode movie --------------------------- #

    # Reconstruct B in real space on each saved frame
    # and build a 2D movie of Bx(x,y) at mid-plane z = 0.
    from matplotlib import animation

    print("[MOVIE] Building tearing-mode Bx(x,y,z=0) movie ...")

    mid_z = Nz // 2
    Bx_slices = []

    for i in range(len(ts_np)):
        B_hat_i = B_hat_frames[i]              # (3, Nx, Ny, Nz)
        B_i = np.fft.ifftn(np.array(B_hat_i), axes=(1, 2, 3)).real
        Bx_slices.append(B_i[0, :, :, mid_z])  # Bx at z = const

    Bx_slices = np.array(Bx_slices)           # (n_frames, Nx, Ny)
    Bx_min = float(Bx_slices.min())
    Bx_max = float(Bx_slices.max())

    fig2, ax2 = plt.subplots(figsize=(5, 4), dpi=150)
    im = ax2.imshow(
        Bx_slices[0].T,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        vmin=Bx_min,
        vmax=Bx_max,
        aspect="equal",
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(r"$B_x(x,y,z=0)$")
    cbar = fig2.colorbar(im, ax=ax2)
    cbar.set_label(r"$B_x$")

    def update_frame(i):
        im.set_data(Bx_slices[i].T)
        ax2.set_title(r"$B_x(x,y,z=0)$" + f", t={ts_np[i]:.3f}")
        return (im,)

    ani = animation.FuncAnimation(
        fig2, update_frame, frames=len(ts_np), interval=100, blit=True
    )
    writer = animation.FFMpegWriter(fps=10, bitrate=2000)
    ani.save("mhd_tearing_Bx_xy.mp4", writer=writer)
    plt.close(fig2)
    print("[MOVIE] Saved tearing movie to mhd_tearing_Bx_xy.mp4")


if __name__ == "__main__":
    main()
