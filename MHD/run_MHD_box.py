#!/usr/bin/env python3
"""
3D incompressible MHD in a periodic box (JAX + Equinox + Diffrax)
=================================================================

This script implements a *standard* pseudo-spectral incompressible MHD solver:

  ∂v/∂t = P[ - (v·∇)v + (∇×B)×B ] + ν ∇²v
  ∂B/∂t = P[ ∇×(v×B) ] + η ∇²B

where P is the divergence-free projection in Fourier space.  All spatial
derivatives (divergence, curl, Laplacian) are computed spectrally, and the
nonlinearity is evaluated pseudo-spectrally (products in real space).

This matches the approach used in many MHD turbulence codes:
- periodic cubic domain,
- Fourier projection to enforce ∇·v = 0 and ∇·B = 0,
- diffusion via -k² factors in Fourier space,
- divergence diagnostics computed with the same spectral operators.

Initial conditions are divergence-free ABC fields for both v and B, so
∇·B and ∇·v are ~machine zero at t=0 and remain small up to numerical error.

Outputs:
  * Diagnostics vs time: kinetic and magnetic energy, ‖∇·B‖₂ and ‖∇·B‖∞
  * 2D |B|(x,y,z=const) movie
  * 3D scatter movie of |B| on a coarse subset of points
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx
import diffrax as dfx

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

jax.config.update("jax_enable_x64", True)

script_dir = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# Fourier utilities
# ---------------------------------------------------------------------

def make_k_vectors(Nx, Ny, Nz, Lx, Ly, Lz):
    """
    Build physical wave numbers kx,ky,kz and k^2 on the 3D FFT grid.
    """
    kx_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(Nx, d=Lx / Nx)
    ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(Ny, d=Ly / Ny)
    kz_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(Nz, d=Lz / Nz)

    kx = kx_1d[:, None, None]
    ky = ky_1d[None, :, None]
    kz = kz_1d[None, None, :]

    k2 = kx * kx + ky * ky + kz * kz
    return kx, ky, kz, k2


def make_dealias_mask(Nx, Ny, Nz):
    """
    2/3-rule dealiasing mask based on integer mode indices.

    Modes with |n_i| > N_i/3 are zeroed.
    """
    nx = jnp.fft.fftfreq(Nx) * Nx
    ny = jnp.fft.fftfreq(Ny) * Ny
    nz = jnp.fft.fftfreq(Nz) * Nz

    nx3 = nx[:, None, None]
    ny3 = ny[None, :, None]
    nz3 = nz[None, None, :]

    mask = (
        (jnp.abs(nx3) <= Nx / 3)
        & (jnp.abs(ny3) <= Ny / 3)
        & (jnp.abs(nz3) <= Nz / 3)
    )
    # Broadcast over vector components
    return mask[..., None]  # (Nx,Ny,Nz,1)


def fft_vector(v):
    """
    FFT a real-space vector field v[...,3] -> v_hat[...,3] (complex).
    """
    vx_hat = jnp.fft.fftn(v[..., 0], axes=(0, 1, 2))
    vy_hat = jnp.fft.fftn(v[..., 1], axes=(0, 1, 2))
    vz_hat = jnp.fft.fftn(v[..., 2], axes=(0, 1, 2))
    return jnp.stack([vx_hat, vy_hat, vz_hat], axis=-1)


def ifft_vector(v_hat):
    """
    Inverse FFT a vector field v_hat[...,3] -> real v[...,3].
    """
    vx = jnp.real(jnp.fft.ifftn(v_hat[..., 0], axes=(0, 1, 2)))
    vy = jnp.real(jnp.fft.ifftn(v_hat[..., 1], axes=(0, 1, 2)))
    vz = jnp.real(jnp.fft.ifftn(v_hat[..., 2], axes=(0, 1, 2)))
    return jnp.stack([vx, vy, vz], axis=-1)


def project_div_free_hat(F_hat, kx, ky, kz, k2):
    """
    Project a vector field in Fourier space onto its divergence-free part.

    F_hat: (...,3) complex, kx,ky,kz,k2: (Nx,Ny,Nz) real.
    """
    k2_safe = jnp.where(k2 > 0.0, k2, 1.0)
    Fx_hat = F_hat[..., 0]
    Fy_hat = F_hat[..., 1]
    Fz_hat = F_hat[..., 2]

    k_dot_F = kx * Fx_hat + ky * Fy_hat + kz * Fz_hat
    factor = k_dot_F / k2_safe

    Fx_proj = Fx_hat - kx * factor
    Fy_proj = Fy_hat - ky * factor
    Fz_proj = Fz_hat - kz * factor
    return jnp.stack([Fx_proj, Fy_proj, Fz_proj], axis=-1)


def spectral_curl_hat(B_hat, kx, ky, kz):
    """
    curl B in Fourier space: ∇×B = i k×B_hat
    """
    Bx_hat = B_hat[..., 0]
    By_hat = B_hat[..., 1]
    Bz_hat = B_hat[..., 2]

    cx_hat = ky * Bz_hat - kz * By_hat
    cy_hat = kz * Bx_hat - kx * Bz_hat
    cz_hat = kx * By_hat - ky * Bx_hat

    curl_x = 1j * cx_hat
    curl_y = 1j * cy_hat
    curl_z = 1j * cz_hat
    return jnp.stack([curl_x, curl_y, curl_z], axis=-1)


def spectral_laplacian_hat(v_hat, k2):
    """
    Laplacian in Fourier: ∇²v = -k² v_hat
    """
    return -k2[..., None] * v_hat


def spectral_div_B(B_hat, kx, ky, kz):
    """
    ∇·B in real space computed via Fourier: i k·B_hat.
    """
    Bx_hat = B_hat[..., 0]
    By_hat = B_hat[..., 1]
    Bz_hat = B_hat[..., 2]

    div_hat = 1j * (kx * Bx_hat + ky * By_hat + kz * Bz_hat)
    div = jnp.real(jnp.fft.ifftn(div_hat, axes=(0, 1, 2)))
    return div


# ---------------------------------------------------------------------
# Grid and initial conditions
# ---------------------------------------------------------------------

def make_grid(Nx, Ny, Nz, Lx, Ly, Lz):
    xs = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    ys = jnp.linspace(0.0, Ly, Ny, endpoint=False)
    zs = jnp.linspace(0.0, Lz, Nz, endpoint=False)
    X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
    return X, Y, Z


def abc_field(X, Y, Z, A=1.0, B=1.0, C=1.0, k=1.0):
    """
    Classical ABC flow, which is exactly divergence-free:

        Bx = A sin(k z) + C cos(k y)
        By = B sin(k x) + A cos(k z)
        Bz = C sin(k y) + B cos(k x)
    """
    Bx = A * jnp.sin(k * Z) + C * jnp.cos(k * Y)
    By = B * jnp.sin(k * X) + A * jnp.cos(k * Z)
    Bz = C * jnp.sin(k * Y) + B * jnp.cos(k * X)
    return jnp.stack([Bx, By, Bz], axis=-1)


def initial_v_abc(X, Y, Z, amplitude=0.1):
    v = amplitude * abc_field(X, Y, Z, A=1.0, B=1.0, C=1.0, k=1.0)
    return v


def initial_B_abc(X, Y, Z, B0=0.3):
    B = B0 * abc_field(X, Y, Z, A=1.0, B=1.0, C=1.0, k=1.0)
    return B


# ---------------------------------------------------------------------
# Minimal MFS grad φ evaluator (optional vacuum field)
# ---------------------------------------------------------------------

def green_grad_x(xn, y):
    r = xn - y
    r2 = jnp.sum(r * r, axis=-1)
    r3 = jnp.maximum(1e-30, r2 * jnp.sqrt(r2))
    return -r / (4.0 * jnp.pi * r3[..., None])


def make_mfs_grad_phi(center, scale, Yn, alpha):
    center = jnp.asarray(center)
    scale = jnp.asarray(scale)
    Yn = jnp.asarray(Yn)
    alpha = jnp.asarray(alpha)

    def grad_phi_point(x):
        xn = (x - center) * scale
        grads = jax.vmap(lambda y: green_grad_x(xn, y))(Yn)
        return scale * jnp.sum(grads * alpha[:, None], axis=0)

    grad_phi_point_vmap = jax.vmap(grad_phi_point, in_axes=0)

    def grad_phi_field(X, Y, Z):
        coords = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        B_flat = grad_phi_point_vmap(coords)
        return B_flat.reshape(X.shape + (3,))

    return grad_phi_field


# ---------------------------------------------------------------------
# State, model, RHS
# ---------------------------------------------------------------------

class MHDState(eqx.Module):
    v: jnp.ndarray  # (Nx,Ny,Nz,3), real
    B: jnp.ndarray  # (Nx,Ny,Nz,3), real


class MHDModel(eqx.Module):
    Nx: int
    Ny: int
    Nz: int
    Lx: float
    Ly: float
    Lz: float
    nu: float
    eta: float
    rho: float
    kx: jnp.ndarray
    ky: jnp.ndarray
    kz: jnp.ndarray
    k2: jnp.ndarray
    dealias: jnp.ndarray  # (Nx,Ny,Nz,1) bool mask for nonlinear terms

    # ---- main RHS ----

    def rhs(self, state: MHDState, t=None) -> MHDState:
        v = state.v
        B = state.B

        # Enforce divergence-free v and B (projection in Fourier space)
        v_hat = fft_vector(v)
        B_hat = fft_vector(B)
        v_hat = project_div_free_hat(v_hat, self.kx, self.ky, self.kz, self.k2)
        B_hat = project_div_free_hat(B_hat, self.kx, self.ky, self.kz, self.k2)
        v = ifft_vector(v_hat)
        B = ifft_vector(B_hat)

        # Current J = ∇×B (spectral)
        J_hat = spectral_curl_hat(B_hat, self.kx, self.ky, self.kz)
        J = ifft_vector(J_hat)

        # Gradients of v: ∂_i v_j via spectral derivative
        vx_hat = v_hat[..., 0]
        vy_hat = v_hat[..., 1]
        vz_hat = v_hat[..., 2]

        dvx_dx = ifft_vector(jnp.stack([
            1j * self.kx * vx_hat,
            1j * self.ky * vx_hat,
            1j * self.kz * vx_hat
        ], axis=-1))[..., 0]  # we only need the scalar result
        dvy_dx = ifft_vector(jnp.stack([
            1j * self.kx * vy_hat,
            1j * self.ky * vy_hat,
            1j * self.kz * vy_hat
        ], axis=-1))[..., 0]
        dvz_dx = ifft_vector(jnp.stack([
            1j * self.kx * vz_hat,
            1j * self.ky * vz_hat,
            1j * self.kz * vz_hat
        ], axis=-1))[..., 0]

        # Build full grad_v (...,3,3) from scalar derivatives
        # grad_v[...,i,j] = ∂_i v_j
        # We already packed them as vectors; easier: recompute via einsum.
        # Better and clearer: use direct spectral gradient.

        def grad_component(vj_hat):
            gx = ifft_vector(jnp.stack([
                1j * self.kx * vj_hat,
                1j * self.ky * vj_hat,
                1j * self.kz * vj_hat
            ], axis=-1))
            return gx  # (...,3) -> ∂_i vj

        gvx = grad_component(vx_hat)
        gvy = grad_component(vy_hat)
        gvz = grad_component(vz_hat)
        # Stack to (...,3,3) with axis order (..., i, j)
        grad_v = jnp.stack([gvx, gvy, gvz], axis=-1)  # (...,3,3)

        # Nonlinear term: (v·∇)v
        v_dot_grad_v = jnp.einsum("...i,...ij->...j", v, grad_v)

        # Lorentz force J×B
        J_cross_B = jnp.stack([
            J[..., 1] * B[..., 2] - J[..., 2] * B[..., 1],
            J[..., 2] * B[..., 0] - J[..., 0] * B[..., 2],
            J[..., 0] * B[..., 1] - J[..., 1] * B[..., 0],
        ], axis=-1)

        # Velocity RHS: - (v·∇)v + J×B + ν ∇²v
        NL_v = -v_dot_grad_v + J_cross_B / self.rho
        NL_v_hat = fft_vector(NL_v) * self.dealias  # dealiased nonlinearity
        visc_v_hat = spectral_laplacian_hat(v_hat, self.k2)
        dvdt_hat = NL_v_hat + self.nu * visc_v_hat
        dvdt_hat = project_div_free_hat(dvdt_hat, self.kx, self.ky, self.kz, self.k2)
        dvdt = ifft_vector(dvdt_hat)

        # Induction: ∂B/∂t = ∇×(v×B) + η ∇²B
        v_cross_B = jnp.stack([
            v[..., 1] * B[..., 2] - v[..., 2] * B[..., 1],
            v[..., 2] * B[..., 0] - v[..., 0] * B[..., 2],
            v[..., 0] * B[..., 1] - v[..., 1] * B[..., 0],
        ], axis=-1)

        v_cross_B_hat = fft_vector(v_cross_B) * self.dealias
        curl_v_cross_B_hat = spectral_curl_hat(v_cross_B_hat, self.kx, self.ky, self.kz)
        diff_B_hat = spectral_laplacian_hat(B_hat, self.k2)
        dBdt_hat = curl_v_cross_B_hat + self.eta * diff_B_hat
        dBdt_hat = project_div_free_hat(dBdt_hat, self.kx, self.ky, self.kz, self.k2)
        dBdt = ifft_vector(dBdt_hat)

        return MHDState(v=dvdt, B=dBdt)


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def compute_energies(state: MHDState, Lx, Ly, Lz, rho=1.0):
    v = state.v
    B = state.B
    Nx, Ny, Nz, _ = v.shape
    dv = (Lx / Nx) * (Ly / Ny) * (Lz / Nz)
    v2 = jnp.sum(v * v, axis=-1)
    B2 = jnp.sum(B * B, axis=-1)
    E_kin = 0.5 * rho * dv * jnp.sum(v2)
    E_mag = 0.5 * dv * jnp.sum(B2)
    return E_kin, E_mag


def compute_divB_norm(state: MHDState, Lx, Ly, Lz, kx, ky, kz):
    B = state.B
    Nx, Ny, Nz, _ = B.shape
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    dv = dx * dy * dz

    B_hat = fft_vector(B)
    divB = spectral_div_B(B_hat, kx, ky, kz)
    L2 = jnp.sqrt(dv * jnp.sum(divB * divB))
    Linf = jnp.max(jnp.abs(divB))
    return L2, Linf


diagnostics_jit = eqx.filter_jit(
    lambda state, Lx, Ly, Lz, rho, kx, ky, kz: (
        compute_energies(state, Lx, Ly, Lz, rho),
        compute_divB_norm(state, Lx, Ly, Lz, kx, ky, kz),
    )
)


# ---------------------------------------------------------------------
# CFL-like timestep estimate
# ---------------------------------------------------------------------

def estimate_cfl_dt(state: MHDState, Lx, Ly, Lz, nu, eta, rho):
    v = state.v
    B = state.B
    Nx, Ny, Nz, _ = v.shape
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    dmin = min(dx, dy, dz)

    vmag = jnp.sqrt(jnp.sum(v * v, axis=-1))
    Bmag = jnp.sqrt(jnp.sum(B * B, axis=-1))
    u_max = float(jnp.max(vmag))
    cA_max = float(jnp.max(Bmag) / jnp.sqrt(rho + 1e-30))
    speed = u_max + cA_max + 1e-30

    nu_max = max(float(nu), float(eta), 1e-30)

    C_adv = 0.4
    C_diff = 0.2

    dt_adv = C_adv * dmin / speed
    dt_diff = C_diff * (dmin ** 2) / nu_max
    return float(dt_adv), float(dt_diff)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3D incompressible MHD (pseudo-spectral, periodic box).")

    # Grid & domain
    parser.add_argument("--Nx", type=int, default=42)
    parser.add_argument("--Ny", type=int, default=42)
    parser.add_argument("--Nz", type=int, default=42)
    parser.add_argument("--Lx", type=float, default=2.0)
    parser.add_argument("--Ly", type=float, default=2.0)
    parser.add_argument("--Lz", type=float, default=2.0)

    # Physical params
    parser.add_argument("--nu", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=1.0)

    # Time
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t1", type=float, default=10.0)
    parser.add_argument("--dt0", type=float, default=1e-2)

    # Diffrax solver config
    parser.add_argument("--solver", choices=["dopri5", "tsit5", "heun", "euler", "dopri8"], default="dopri8")
    parser.add_argument("--stepsize-controller", choices=["pid", "constant"], default="pid")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--progress-meter", choices=["none", "text", "tqdm"], default="tqdm")

    # Snapshots / visualization
    parser.add_argument("--n-frames", type=int, default=40, help="Number of output frames.")
    parser.add_argument("--z-index", type=int, default=None)
    parser.add_argument("--vis-stride", type=int, default=4, help="Stride for 3D scatter subsampling.")

    # Initial B choice
    parser.add_argument("--initial-B-mode", choices=["abc", "mfs"], default="abc")
    parser.add_argument("--B0", type=float, default=0.3)

    # MFS (optional)
    parser.add_argument("--mfs-npz", type=str, default=None, help="MFS solution checkpoint .npz")
    parser.add_argument("--mfs-subdir", type=str, default="outputs")

    # Velocity seed (only used for optional random tweaks; ABC is deterministic)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--v0-amplitude", type=float, default=0.1)

    # Output files
    parser.add_argument("--movie-file", type=str, default="mhd_B_slice_spectral.mp4")
    parser.add_argument("--movie-3d-file", type=str, default="mhd_B_3d_spectral.mp4")
    parser.add_argument("--diag-file", type=str, default="mhd_diagnostics_spectral.png")

    args = parser.parse_args()

    print("========== Spectral MHD (periodic box) Parameters ==========")
    print(f"Nx,Ny,Nz = {args.Nx},{args.Ny},{args.Nz}")
    print(f"Lx,Ly,Lz = {args.Lx},{args.Ly},{args.Lz}")
    print(f"nu={args.nu}, eta={args.eta}, rho={args.rho}")
    print(f"t0={args.t0}, t1={args.t1}, dt0={args.dt0}")
    print(f"solver={args.solver}, stepsize_controller={args.stepsize_controller}, "
          f"rtol={args.rtol}, atol={args.atol}, max_steps={args.max_steps}")
    print(f"n_frames={args.n_frames}")
    print(f"initial_B_mode={args.initial_B_mode}, B0={args.B0}")
    print("============================================================")

    Nx, Ny, Nz = args.Nx, args.Ny, args.Nz
    Lx, Ly, Lz = float(args.Lx), float(args.Ly), float(args.Lz)

    print("[SETUP] Building grid and k-vectors...")
    X, Y, Z = make_grid(Nx, Ny, Nz, Lx, Ly, Lz)
    kx, ky, kz, k2 = make_k_vectors(Nx, Ny, Nz, Lx, Ly, Lz)
    dealias = make_dealias_mask(Nx, Ny, Nz)

    k2_nonzero = k2[k2 > 0.0]
    if k2_nonzero.size > 0:
        print(f"[SETUP] k2 stats: min>0 = {float(jnp.min(k2_nonzero)):.3e}")
    else:
        print("[SETUP] k2 has no nonzero modes? (Nx=Ny=Nz=1 case)")

    # Initial fields
    print(f"[SETUP] Initial B mode: {args.initial_B_mode}")
    if args.initial_B_mode == "abc":
        B0 = initial_B_abc(X, Y, Z, B0=args.B0)
    elif args.initial_B_mode == "mfs":
        if args.mfs_npz is None:
            raise ValueError("initial-B-mode=mfs requires --mfs-npz")
        npz_path = (script_dir / ".." / args.mfs_subdir / args.mfs_npz).resolve()
        print(f"[MFS] Loading NPZ from {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        center = data["center"]
        scale = data["scale"]
        Yn = data["Yn"]
        alpha = data["alpha"]
        grad_phi_field = make_mfs_grad_phi(center, scale, Yn, alpha)
        B0 = grad_phi_field(X, Y, Z)
        Bmag_mean = float(jnp.sqrt(jnp.mean(jnp.sum(B0 * B0, axis=-1))))
        if Bmag_mean > 0:
            rescale = args.B0 / Bmag_mean
            print(f"[MFS] Rescaling B by factor {rescale:.3e} to set mean |B|~B0.")
            B0 = rescale * B0
    else:
        raise ValueError("Unknown initial_B_mode")

    print("[SETUP] Building initial ABC velocity...")
    v0 = initial_v_abc(X, Y, Z, amplitude=args.v0_amplitude)

    state0 = MHDState(v=v0, B=B0)

    # Initial diagnostics
    (E0, Em0), (L2_0, Linf_0) = diagnostics_jit(
        state0, Lx, Ly, Lz, args.rho, kx, ky, kz
    )
    print(f"[INIT] E_kin0={float(E0):.6e}, E_mag0={float(Em0):.6e}")
    print(f"[INIT] ||divB||_2={float(L2_0):.6e}, ||divB||_∞={float(Linf_0):.6e}")

    # CFL-based dt advice
    dt_adv, dt_diff = estimate_cfl_dt(state0, Lx, Ly, Lz, args.nu, args.eta, args.rho)
    dt_rec = min(dt_adv, dt_diff)
    print(f"[CFL] Estimated advective dt <= {dt_adv:.3e}")
    print(f"[CFL] Estimated diffusive dt <= {dt_diff:.3e}")
    print(f"[CFL] Recommended dt0 ~ {dt_rec:.3e}")
    ratio = args.dt0 / dt_rec if dt_rec > 0 else np.inf
    if ratio < 0.1:
        print("[CFL] Your dt0 is much smaller than recommended (very conservative; may be slow).")
    elif ratio > 5.0:
        print("[CFL] Your dt0 is much larger than recommended (risk of instability / rejected steps).")
    else:
        print("[CFL] Your dt0 is in a reasonable range.")

    # Build model
    model = MHDModel(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        nu=args.nu,
        eta=args.eta,
        rho=args.rho,
        kx=kx,
        ky=ky,
        kz=kz,
        k2=k2,
        dealias=dealias,
    )

    # Diffrax setup
    def dynamics(t, y, args_model):
        return args_model.rhs(y, t)

    term = dfx.ODETerm(dynamics)

    if args.solver == "dopri5":
        solver = dfx.Dopri5()
    elif args.solver == "tsit5":
        solver = dfx.Tsit5()
    elif args.solver == "heun":
        solver = dfx.Heun()
    elif args.solver == "euler":
        solver = dfx.Euler()
    elif args.solver == "dopri8":
        solver = dfx.Dopri8()
    else:
        raise ValueError("Unknown solver")

    if args.stepsize_controller == "pid":
        stepsize_controller = dfx.PIDController(rtol=args.rtol, atol=args.atol, dtmax=0.6 * dt_rec)
    else:
        stepsize_controller = dfx.ConstantStepSize()

    if args.progress_meter == "none":
        progress_meter = None
    elif args.progress_meter == "text":
        progress_meter = dfx.TextProgressMeter()
    else:
        progress_meter = dfx.TqdmProgressMeter()

    t0, t1 = args.t0, args.t1
    ts = jnp.linspace(t0, t1, args.n_frames)

    print("[RUN] Calling diffrax.diffeqsolve with progress meter...")
    sol = dfx.diffeqsolve(
        term,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=args.dt0,
        y0=state0,
        args=model,
        saveat=dfx.SaveAt(ts=ts),
        stepsize_controller=stepsize_controller,
        max_steps=args.max_steps,
        progress_meter=progress_meter,
    )
    print("[RUN] Solve finished. stats:", sol.stats)

    # Extract frames
    frames_t = np.asarray(sol.ts)
    frames_B = np.asarray(sol.ys.B)  # (n_frames, Nx,Ny,Nz,3)
    frames_v = np.asarray(sol.ys.v)

    n_frames_used = frames_B.shape[0]
    print(f"[RUN] Stored {n_frames_used} frames.")

    # Diagnostics over frames
    print("[POST] Computing diagnostic curves...")
    E_kin_list = []
    E_mag_list = []
    divB_L2_list = []
    divB_Linf_list = []
    for i in range(n_frames_used):
        s = MHDState(v=jnp.asarray(frames_v[i]), B=jnp.asarray(frames_B[i]))
        (E_kin, E_mag), (L2, Linf) = diagnostics_jit(
            s, Lx, Ly, Lz, args.rho, kx, ky, kz
        )
        E_kin_list.append(float(E_kin))
        E_mag_list.append(float(E_mag))
        divB_L2_list.append(float(L2))
        divB_Linf_list.append(float(Linf))
        print(f"[POST] frame {i}/{n_frames_used-1}, t={frames_t[i]:.4f}, "
              f"E_kin={E_kin_list[-1]:.3e}, E_mag={E_mag_list[-1]:.3e}, "
              f"||divB||_2={divB_L2_list[-1]:.3e}, ||divB||_∞={divB_Linf_list[-1]:.3e}")

    # Plot diagnostics
    print(f"[PLOT] Saving diagnostics figure to {args.diag_file}")
    fig_diag, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(frames_t, E_kin_list, label="E_kin")
    ax1.plot(frames_t, E_mag_list, label="E_mag")
    ax1.set_xlabel("t"); ax1.set_ylabel("Energy"); ax1.set_title("MHD Energies vs time")
    ax1.legend(); ax1.grid(True)

    ax2.semilogy(frames_t, divB_L2_list, label="||∇·B||_2")
    ax2.semilogy(frames_t, divB_Linf_list, label="||∇·B||_∞")
    ax2.set_xlabel("t"); ax2.set_ylabel("Divergence norm"); ax2.set_title("Divergence of B")
    ax2.legend(); ax2.grid(True)

    fig_diag.tight_layout()
    fig_diag.savefig(args.diag_file, dpi=200)

    # 2D |B| slice movie
    z_index = args.z_index if args.z_index is not None else Nz // 2
    print(f"[PLOT] Building |B| movie for z-index={z_index}; saving to {args.movie_file}")

    B_slices = []
    for i in range(n_frames_used):
        B = frames_B[i]
        B_slice = B[:, :, z_index, :]
        Bmag = np.sqrt(np.sum(B_slice**2, axis=-1))
        B_slices.append(Bmag)
    B_slices = np.stack(B_slices, axis=0)

    vmin = B_slices.min()
    vmax = B_slices.max()

    fig2, ax2d = plt.subplots(figsize=(5, 4))
    im = ax2d.imshow(
        B_slices[0].T,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y")
    ax2d.set_title(f"|B|(x,y,z={z_index})")
    cbar = fig2.colorbar(im, ax=ax2d); cbar.set_label("|B|")

    def update2d(frame):
        im.set_data(B_slices[frame].T)
        ax2d.set_title(f"|B|(x,y,z={z_index}), t={frames_t[frame]:.3f}")
        return (im,)

    anim2d = FuncAnimation(fig2, update2d, frames=n_frames_used, interval=100, blit=True)
    anim2d.save(args.movie_file, fps=15, dpi=150)
    print("[PLOT] 2D movie saved.")

    # 3D scatter movie (optional)
    if args.movie_3d_file is not None and len(args.movie_3d_file) > 0:
        print(f"[PLOT] Building 3D |B| scatter movie (stride={args.vis_stride}); saving to {args.movie_3d_file}")
        stride = max(1, int(args.vis_stride))
        X_np = np.asarray(X)
        Y_np = np.asarray(Y)
        Z_np = np.asarray(Z)
        xs = X_np[::stride, ::stride, ::stride].ravel()
        ys = Y_np[::stride, ::stride, ::stride].ravel()
        zs = Z_np[::stride, ::stride, ::stride].ravel()

        Bmag_all = np.sqrt(np.sum(frames_B**2, axis=-1))
        Bmag_coarse = Bmag_all[:, ::stride, ::stride, ::stride].reshape(n_frames_used, -1)
        vmin3 = float(Bmag_coarse.min())
        vmax3 = float(Bmag_coarse.max())

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig3 = plt.figure(figsize=(6, 5))
        ax3d = fig3.add_subplot(111, projection="3d")
        sc = ax3d.scatter(xs, ys, zs, c=Bmag_coarse[0], s=5, vmin=vmin3, vmax=vmax3)
        ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
        ax3d.set_title(f"|B| 3D, t={frames_t[0]:.3f}")
        fig3.colorbar(sc, ax=ax3d, label="|B|")

        def update3d(frame):
            sc.set_array(Bmag_coarse[frame])
            ax3d.set_title(f"|B| 3D, t={frames_t[frame]:.3f}")
            return (sc,)

        anim3d = FuncAnimation(fig3, update3d, frames=n_frames_used, interval=100, blit=True)
        anim3d.save(args.movie_3d_file, fps=10, dpi=150)
        print("[PLOT] 3D movie saved.")

    print("[DONE]")


if __name__ == "__main__":
    main()
