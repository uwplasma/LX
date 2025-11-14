###!!!!!!
###### THIS WOULD BE A GREAT NEXT STEP FOR THE CODE  ######
##### MAYBE IMPLEMENT LATER #############################
# if kind=torus, this means that it is a donut with a mostly cylindrical geometry. That means that it would be more natural to use a cylindrical grid to save space. if kind=mirror, let's leave it cartesian. Help me implement a cylindrical coordinate system for our points so that we save as much space as possible. Let's keep all the gains we made using FCI up to now, so that it is a publication ready code with the flux coordinate independent approach. It should be a fully working, metric-correct, symmetric 3D anisotropic diffusion operator in (R, φ, Z)
###!!!!!!
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern implementation of a flux–coordinate independent (FCI) field aligned flux function
solver.  The goal is to construct a scalar potential ψ that is approximately constant
along magnetic field lines while diffusing across those lines.  The governing
equation is the anisotropic diffusion problem

    div( D(x) ∇ψ(x) ) = 0  on Ω    with  ψ=1 on Γ_bnd  and  ψ=0 on Γ_axis,

where D(x) = eps * P_perp + P_par encodes a large conductivity parallel to
the magnetic field (defined by grad φ) and a small conductivity perpendicular
to the field.  The boundary Γ_bnd is a thin ribbon near the physical surface
(the outer boundary of the domain) and Γ_axis is a thin tube around the
magnetic axis.  This formulation follows the flux–coordinate independent
approach pioneered by Hariri and Ottaviani and later developed in numerous
plasma simulation codes.

This version:

  * Uses JAX to evaluate φ and ∇φ.
  * Uses Diffrax to find the magnetic axis.
  * Discretises ∇·(D∇ψ) with a **full-tensor 27-point stencil** on a Cartesian grid:
      - face fluxes F_x, F_y, F_z include all tensor components D_ij,
        not just n·D·n, so cross-derivative terms are represented.
  * Imposes ψ=1 on a boundary band and ψ=0 on an axis band via a standard
    linear "lifting" so the matrix-free operator remains a pure PDE operator.
  * Includes diagnostics on residuals and a field-alignment metric q = |t·∇ψ|/|∇ψ|.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap

from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import LinearOperator, cg
import diffrax as dfx
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
from pathlib import Path

# -------------------------- Paths and utils ------------------------- #
script_dir = Path(__file__).resolve().parent

def resolve_npz_file_location(npz_file, subdir="outputs"):
    try:
        npz_name = os.path.basename(str(npz_file))
        candidate = (script_dir / ".." / subdir / npz_name).resolve()
        if candidate.exists():
            npz_file = str(candidate)
            print(f"Resolved checkpoint path -> {npz_file}")
        else:
            print(f"[WARN] Expected checkpoint not found at {candidate}; using provided path: {npz_file}")
    except Exception as e:
        print(f"[WARN] Failed to resolve ../{subdir} path: {e}; using provided path: {npz_file}")
    return npz_file

# ---------------------------- Debug utils ---------------------------- #
def pct(a, p): return float(np.percentile(np.asarray(a), p))
def pinfo(msg): print(f"[INFO] {msg}")
def pstat(msg, v):
    v = np.asarray(v)
    print(f"[STAT] {msg}: min={v.min():.3e} med={np.median(v):.3e} max={v.max():.3e} L2={np.linalg.norm(v):.3e}")

# ----------------------------- Geometry utilities ---------------------------- #
def axis_band_mask(P_axis, Xq, rad):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P_axis)
    d, _ = nbrs.kneighbors(Xq)
    return (d[:, 0] < rad)

# ----------------------------- Plotting utilities ---------------------------- #
def build_psi_RZphi_volume(psi3, xs, ys, zs, P, inside3,
                           nR=128, nphi=64, nZ=128):
    Rb = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
    Rs = np.linspace(Rb.min(), Rb.max(), nR)
    Zs = np.linspace(P[:, 2].min(), P[:, 2].max(), nZ)
    phis = np.linspace(0.0, 2.0*np.pi, nphi, endpoint=False)

    interp_psi = RegularGridInterpolator(
        (xs, ys, zs), psi3,
        bounds_error=False, fill_value=np.nan
    )
    interp_inside = RegularGridInterpolator(
        (xs, ys, zs), inside3.astype(float),
        bounds_error=False, fill_value=0.0
    )

    psi_RZphi = np.zeros((nR, nphi, nZ))
    mask_RZphi = np.zeros((nR, nphi, nZ), dtype=bool)

    for j, phi in enumerate(phis):
        R_grid, Z_grid = np.meshgrid(Rs, Zs, indexing="ij")
        X = R_grid * np.cos(phi)
        Y = R_grid * np.sin(phi)
        pts = np.stack([X.ravel(), Y.ravel(), Z_grid.ravel()], axis=-1)

        vals = interp_psi(pts).reshape(nR, nZ)
        inside_vals = interp_inside(pts).reshape(nR, nZ) > 0.5

        psi_RZphi[:, j, :] = np.where(inside_vals, vals, np.nan)
        mask_RZphi[:, j, :] = inside_vals

    return psi_RZphi, Rs, phis, Zs, mask_RZphi

def plot_psi_maps_RZ_panels(psi_RZphi, Rs, phis, Zs, jj_list,
                            Rb=None, Zb=None, phi_b=None,
                            title="ψ(R,Z)"):
    fig, axa = plt.subplots(2, 2, figsize=(7, 7), constrained_layout=True)
    axa = axa.ravel()
    Rmin, Rmax = float(np.nanmin(Rs)), float(np.nanmax(Rs))
    Zmin, Zmax = float(np.nanmin(Zs)), float(np.nanmax(Zs))
    extent = [Rmin, Rmax, Zmin, Zmax]

    for kk, jj in enumerate(jj_list):
        psi_slice = psi_RZphi[:, jj, :].T  # shape (nZ, nR)
        im = axa[kk].imshow(
            psi_slice,
            origin='lower',
            aspect='equal',
            extent=extent
        )
        axa[kk].contour(
            Rs, Zs, psi_RZphi[:, jj, :].T,
            levels=10, colors='white', linewidths=0.5, alpha=1.0
        )
        axa[kk].set_title(f"{title} @ φ≈{phis[jj]:+.2f}")
        axa[kk].set_xlabel("R"); axa[kk].set_ylabel("Z")
        plt.colorbar(im, ax=axa[kk], shrink=0.85)

        if Rb is not None and phi_b is not None and Zb is not None:
            dphi = np.abs(np.angle(np.exp(1j*(phi_b - phis[jj]))))
            mask = dphi < (np.pi / len(phis))
            axa[kk].scatter(Rb[mask], Zb[mask], s=5, c='k', alpha=0.7)

    return fig

def plot_3d_axis_boundary_interp(P, axis_pts, psi3, xs, ys, zs):
    interp_psi = RegularGridInterpolator((xs, ys, zs), psi3,
                                         bounds_error=False, fill_value=np.nan)
    psi_on_P = interp_psi(P)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        P[:, 0], P[:, 1], P[:, 2],
        c=psi_on_P, s=5, alpha=0.8
    )
    ax.plot(
        axis_pts[:, 0], axis_pts[:, 1], axis_pts[:, 2],
        "k-", linewidth=2.0, label="Magnetic axis"
    )

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Boundary vs magnetic axis (coloured by ψ)")
    fig.colorbar(sc, ax=ax, shrink=0.7, label=r"$\psi$")
    ax.legend(loc="best")
    return fig

###############################################################################
# JAX helpers for Green's function and gradient
###############################################################################

@jit
def green_G(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    r = jnp.linalg.norm(x - y, axis=-1)
    return 1.0 / (4.0 * jnp.pi * jnp.maximum(1e-30, r))

@jit
def grad_green_x(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    r = x - y
    r2 = jnp.sum(r * r, axis=-1)
    r3 = jnp.maximum(1e-30, r2 * jnp.sqrt(r2))
    return -r / (4.0 * jnp.pi * r3[..., None])

@jit
def grad_azimuth_about_axis(Xn: jnp.ndarray, a_hat: jnp.ndarray) -> jnp.ndarray:
    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par = jnp.sum(Xn * a[None, :], axis=1, keepdims=True) * a[None, :]
    r_perp = Xn - r_par
    r2 = jnp.maximum(1e-30, jnp.sum(r_perp * r_perp, axis=1, keepdims=True))
    return jnp.cross(a[None, :], r_perp) / r2

def make_mv_grads(a_vec: jnp.ndarray, a_hat: jnp.ndarray,
                  sc_center: jnp.ndarray, sc_scale: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    a_vec = jnp.asarray(a_vec)
    a_hat = jnp.asarray(a_hat)
    sc_center = jnp.asarray(sc_center)
    sc_scale = float(sc_scale)

    @jit
    def grad_t(Xn: jnp.ndarray) -> jnp.ndarray:
        return grad_azimuth_about_axis(Xn, a_hat)

    @jit
    def grad_p(Xn: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(Xn)

    def grad_mv_world(X: jnp.ndarray) -> jnp.ndarray:
        Xn = (X - sc_center) * sc_scale
        return sc_scale * (a_vec[0] * grad_t(Xn) + a_vec[1] * grad_p(Xn))

    return grad_mv_world

###############################################################################
# Multipole expansion evaluators
###############################################################################

@dataclass
class Evaluators:
    center: jnp.ndarray
    scale: float
    Yn: jnp.ndarray
    alpha: jnp.ndarray
    a: jnp.ndarray
    a_hat: jnp.ndarray

    def build(self) -> Tuple[Callable[[jnp.ndarray], jnp.ndarray],
                             Callable[[jnp.ndarray], jnp.ndarray]]:
        sc_c = jnp.asarray(self.center)
        sc_s = float(self.scale)
        Yn_c = jnp.asarray(self.Yn)
        alpha_c = jnp.asarray(self.alpha)
        a_c = jnp.asarray(self.a)
        a_hatc = jnp.asarray(self.a_hat)

        S_single = jit(lambda xn: jnp.dot(vmap(lambda y: green_G(xn, y))(Yn_c), alpha_c))
        dS_single = jit(lambda xn: jnp.sum(
            vmap(lambda y: grad_green_x(xn, y))(Yn_c) * alpha_c[:, None],
            axis=0
        ))

        @jit
        def phi_fn(X: jnp.ndarray) -> jnp.ndarray:
            Xn = (X - sc_c) * sc_s
            return vmap(S_single)(Xn)

        grad_mv = make_mv_grads(a_c, a_hatc, sc_c, sc_s)

        @jit
        def grad_phi_fn(X: jnp.ndarray) -> jnp.ndarray:
            Xn = (X - sc_c) * sc_s
            return grad_mv(X) + sc_s * vmap(dS_single)(Xn)

        return phi_fn, grad_phi_fn

###############################################################################
# Geometry: inside mask and bands
###############################################################################

def inside_mask_from_surface(P_surf: np.ndarray, N_surf: np.ndarray,
                             Xq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P_surf)
    d, idx = nbrs.kneighbors(Xq)
    p = P_surf[idx[:, 0], :]
    n = N_surf[idx[:, 0], :]
    signed_dist = np.sum((Xq - p) * n, axis=1)
    inside = signed_dist < 0.0
    return inside, idx[:, 0], signed_dist

###############################################################################
# Axis finding via Poincaré map with Diffrax
###############################################################################

def collapse_to_axis(grad_phi: Callable[[jnp.ndarray], jnp.ndarray], R0: float, Z0: float,
                     nfp: int, nsteps: int = 1000, max_newton: int = 12,
                     tol: float = 1e-8) -> Tuple[float, float]:
    @jit
    def B_cyl(R: float, phi: float, Z: float) -> Tuple[jnp.ndarray, float, float]:
        x = R * jnp.cos(phi)
        y = R * jnp.sin(phi)
        X = jnp.stack([x, y, Z])
        B = grad_phi(X[None, :])[0]
        eR = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0])
        ephi = jnp.array([-jnp.sin(phi), jnp.cos(phi), 0.0])
        BR = jnp.dot(B, eR)
        Bphi = jnp.dot(B, ephi)
        BZ = B[2]
        Bphi = jnp.where(jnp.abs(Bphi) < 1e-12, jnp.sign(Bphi) * 1e-12, Bphi)
        return BR, Bphi, BZ

    @jit
    def fieldline_rhs(phi: float, RZ: jnp.ndarray, args: Any) -> jnp.ndarray:
        R, Z = RZ
        BR, Bphi, BZ = B_cyl(R, phi, Z)
        dR_dphi = R * BR / Bphi
        dZ_dphi = R * BZ / Bphi
        return jnp.stack([dR_dphi, dZ_dphi])

    term = dfx.ODETerm(fieldline_rhs)
    solver = dfx.Dopri5()
    t0 = 0.0
    t1 = 2.0 * jnp.pi / nfp
    dt0 = float(t1) / float(nsteps)
    saveat_t1 = dfx.SaveAt(t1=True)
    stepsize_controller = dfx.PIDController(rtol=1e-4, atol=1e-5)

    @jit
    def integrate_one_turn(RZ0: jnp.ndarray) -> jnp.ndarray:
        sol = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=RZ0,
                              saveat=saveat_t1, max_steps=200_000,
                              stepsize_controller=stepsize_controller)
        return sol.ys[-1]

    @jit
    def poincare_residual(RZ: jnp.ndarray) -> jnp.ndarray:
        RZ1 = integrate_one_turn(RZ)
        return RZ1 - RZ

    poincare_jac = jit(jax.jacobian(poincare_residual))

    RZ = jnp.asarray([R0, Z0], dtype=jnp.float64)
    for _ in range(max_newton):
        F = poincare_residual(RZ)
        if float(jnp.linalg.norm(F)) < tol:
            break
        J = poincare_jac(RZ)
        delta = jnp.linalg.solve(J, -F)
        RZ = RZ + delta
    return float(RZ[0]), float(RZ[1])

###############################################################################
# Diffusion tensor
###############################################################################

def diffusion_tensor(gradphi: np.ndarray, eps: float,
                     delta: float = 5e-3) -> np.ndarray:
    Npts = gradphi.shape[0]
    I = np.eye(3)[None, :, :]
    n = np.linalg.norm(gradphi, axis=-1, keepdims=True)
    n0 = 5e-3 * np.nanmax(n) + 1e-30
    s = np.clip(n / n0, 0.0, 1.0)
    t = np.divide(gradphi, np.maximum(n, 1e-30), out=np.zeros_like(gradphi))
    ax = np.abs(t[:, 0]) > 0.70710678
    anchor = np.zeros_like(t)
    anchor[~ax, 0] = 1.0
    anchor[ax, 1] = 1.0
    b1 = np.cross(t, anchor)
    nb1 = np.linalg.norm(b1, axis=-1, keepdims=True)
    ok = nb1[:, 0] > 1e-15
    b1[ok] /= nb1[ok]
    b1[~ok] = 0.0
    b2 = np.cross(t, b1)
    R = np.stack([t, b1, b2], axis=-1)  # shape (N,3,3)
    Lam = np.zeros((1, 3, 3), dtype=float)
    Lam[..., 0, 0] = 1.0
    Lam[..., 1, 1] = eps
    Lam[..., 2, 2] = eps
    Daniso = R @ Lam @ np.swapaxes(R, -1, -2)
    Diso = np.eye(3)[None, :, :]
    D = (s[..., None] * Daniso) + ((1.0 - s[..., None]) * Diso) + delta * I
    return D

###############################################################################
# Full-tensor Cartesian FV operator (27-point stencil via face fluxes)
###############################################################################

def make_linear_operator(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    inside: np.ndarray,
    Dfield: np.ndarray,
):
    """
    Build a matrix–free LinearOperator A_pde for the anisotropic diffusion equation

        A_pde[u] = -div( D ∇u )

    on a uniform Cartesian grid, using a 27-point stencil implied by
    face fluxes:

        F_x = (Dxx ∂xψ + Dxy ∂yψ + Dxz ∂zψ) on x-faces,
        F_y, F_z analogous.

    We:
      * restrict fluxes to faces where both neighbour cells are inside;
      * compute cross derivatives with centred differences, only where the
        necessary neighbours exist inside (otherwise cross-terms are dropped);
      * apply the operator only on "deep interior" nodes (interior_core).
    """
    inside3 = inside.reshape(nx, ny, nz)
    D3 = Dfield.reshape(nx, ny, nz, 3, 3)

    # Deep interior: not on bounding box, and inside
    interior_core = np.ones((nx, ny, nz), dtype=bool)
    interior_core[0, :, :] = False
    interior_core[-1, :, :] = False
    interior_core[:, 0, :] = False
    interior_core[:, -1, :] = False
    interior_core[:, :, 0] = False
    interior_core[:, :, -1] = False
    interior_core &= inside3

    deep_inside_mask = interior_core.ravel(order="C")

    # Precompute face diffusion tensors and masks

    # x-faces: between i and i+1, with i=0..nx-2, j=1..ny-2, k=1..nz-2
    D_x = 0.5 * (D3[1:, 1:-1, 1:-1, :, :] + D3[:-1, 1:-1, 1:-1, :, :])
    mask_x = inside3[1:, 1:-1, 1:-1] & inside3[:-1, 1:-1, 1:-1]

    # y-faces: between j and j+1, i=1..nx-2, j=0..ny-2, k=1..nz-2
    D_y = 0.5 * (D3[1:-1, 1:, 1:-1, :, :] + D3[1:-1, :-1, 1:-1, :, :])
    mask_y = inside3[1:-1, 1:, 1:-1] & inside3[1:-1, :-1, 1:-1]

    # z-faces: between k and k+1, i=1..nx-2, j=1..ny-2, k=0..nz-2
    D_z = 0.5 * (D3[1:-1, 1:-1, 1:, :, :] + D3[1:-1, 1:-1, :-1, :, :])
    mask_z = inside3[1:-1, 1:-1, 1:] & inside3[1:-1, 1:-1, :-1]

    def matvec(u: np.ndarray) -> np.ndarray:
        u3 = u.reshape(nx, ny, nz)
        out3 = np.zeros_like(u3)

        # ---------------- x-faces ----------------
        # central differences at x-faces
        dpsi_dx_xp = (u3[1:, 1:-1, 1:-1] - u3[:-1, 1:-1, 1:-1]) / dx

        dpsi_dy_xp = (
            (u3[1:, 2:, 1:-1] - u3[1:, :-2, 1:-1]) +
            (u3[:-1, 2:, 1:-1] - u3[:-1, :-2, 1:-1])
        ) * (0.25 / dy)

        dpsi_dz_xp = (
            (u3[1:, 1:-1, 2:] - u3[1:, 1:-1, :-2]) +
            (u3[:-1, 1:-1, 2:] - u3[:-1, 1:-1, :-2])
        ) * (0.25 / dz)

        # drop cross-terms where the needed neighbours are not all inside
        # (simple, conservative choice)
        valid_cross_x = (
            mask_x &
            inside3[1:, 2:, 1:-1] & inside3[1:, :-2, 1:-1] &
            inside3[:-1, 2:, 1:-1] & inside3[:-1, :-2, 1:-1] &
            inside3[1:, 1:-1, 2:] & inside3[1:, 1:-1, :-2] &
            inside3[:-1, 1:-1, 2:] & inside3[:-1, 1:-1, :-2]
        )

        dpsi_dy_xp = np.where(valid_cross_x, dpsi_dy_xp, 0.0)
        dpsi_dz_xp = np.where(valid_cross_x, dpsi_dz_xp, 0.0)

        qx_xp = (
            D_x[..., 0, 0] * dpsi_dx_xp +
            D_x[..., 0, 1] * dpsi_dy_xp +
            D_x[..., 0, 2] * dpsi_dz_xp
        )
        qx_xp *= mask_x

        out3[:-1, 1:-1, 1:-1] -= qx_xp / dx
        out3[1:, 1:-1, 1:-1]  += qx_xp / dx

        # ---------------- y-faces ----------------
        dpsi_dy_yp = (u3[1:-1, 1:, 1:-1] - u3[1:-1, :-1, 1:-1]) / dy

        dpsi_dx_yp = (
            (u3[2:, 1:, 1:-1] - u3[:-2, 1:, 1:-1]) +
            (u3[2:, :-1, 1:-1] - u3[:-2, :-1, 1:-1])
        ) * (0.25 / dx)

        dpsi_dz_yp = (
            (u3[1:-1, 1:, 2:] - u3[1:-1, 1:, :-2]) +
            (u3[1:-1, :-1, 2:] - u3[1:-1, :-1, :-2])
        ) * (0.25 / dz)

        valid_cross_y = (
            mask_y &
            inside3[2:, 1:, 1:-1] & inside3[:-2, 1:, 1:-1] &
            inside3[2:, :-1, 1:-1] & inside3[:-2, :-1, 1:-1] &
            inside3[1:-1, 1:, 2:] & inside3[1:-1, 1:, :-2] &
            inside3[1:-1, :-1, 2:] & inside3[1:-1, :-1, :-2]
        )

        dpsi_dx_yp = np.where(valid_cross_y, dpsi_dx_yp, 0.0)
        dpsi_dz_yp = np.where(valid_cross_y, dpsi_dz_yp, 0.0)

        qy_yp = (
            D_y[..., 1, 0] * dpsi_dx_yp +
            D_y[..., 1, 1] * dpsi_dy_yp +
            D_y[..., 1, 2] * dpsi_dz_yp
        )
        qy_yp *= mask_y

        out3[1:-1, :-1, 1:-1] -= qy_yp / dy
        out3[1:-1, 1:, 1:-1]  += qy_yp / dy

        # ---------------- z-faces ----------------
        dpsi_dz_zp = (u3[1:-1, 1:-1, 1:] - u3[1:-1, 1:-1, :-1]) / dz

        dpsi_dx_zp = (
            (u3[2:, 1:-1, 1:] - u3[:-2, 1:-1, 1:]) +
            (u3[2:, 1:-1, :-1] - u3[:-2, 1:-1, :-1])
        ) * (0.25 / dx)

        dpsi_dy_zp = (
            (u3[1:-1, 2:, 1:] - u3[1:-1, :-2, 1:]) +
            (u3[1:-1, 2:, :-1] - u3[1:-1, :-2, :-1])
        ) * (0.25 / dy)

        valid_cross_z = (
            mask_z &
            inside3[2:, 1:-1, 1:] & inside3[:-2, 1:-1, 1:] &
            inside3[2:, 1:-1, :-1] & inside3[:-2, 1:-1, :-1] &
            inside3[1:-1, 2:, 1:] & inside3[1:-1, :-2, 1:] &
            inside3[1:-1, 2:, :-1] & inside3[1:-1, :-2, :-1]
        )

        dpsi_dx_zp = np.where(valid_cross_z, dpsi_dx_zp, 0.0)
        dpsi_dy_zp = np.where(valid_cross_z, dpsi_dy_zp, 0.0)

        qz_zp = (
            D_z[..., 2, 0] * dpsi_dx_zp +
            D_z[..., 2, 1] * dpsi_dy_zp +
            D_z[..., 2, 2] * dpsi_dz_zp
        )
        qz_zp *= mask_z

        out3[1:-1, 1:-1, :-1] -= qz_zp / dz
        out3[1:-1, 1:-1, 1:]  += qz_zp / dz

        # Only act on true PDE interior; zero elsewhere
        out3[~interior_core] = 0.0
        out3[~inside3] = 0.0

        return out3.ravel(order="C")

    N = nx * ny * nz
    A = LinearOperator((N, N), matvec=matvec, rmatvec=matvec, dtype=float)
    return A, deep_inside_mask

###############################################################################
# Main solver routine
###############################################################################

def solve_fci(npz_file: str, grid_N: int = 64, eps: float = 1e-3, band_h: float = 1.5,
              axis_band_radius: float = 0.0, cg_tol: float = 1e-8, cg_maxit: int = 2000,
              verbose: bool = True, plot: bool = False, nfp: int = 2,
              delta: float = 5e-3, no_amg: bool = False,
              axis_seed_count: int = 0, save_figures: bool = True) -> Dict[str, Any]:

    data = np.load(npz_file, allow_pickle=True)
    center = data["center"]; scale = float(data["scale"])
    Yn = data["Yn"]; alpha = data["alpha"]
    a = data["a"]; a_hat = data["a_hat"]
    P = data["P"]; Nsurf = data["N"]
    kind = str(data["kind"])
    kind_str = kind.strip().lower()
    kind_is_torus = (kind_str == "torus")
    if verbose:
        pinfo(f"Loaded checkpoint with {P.shape[0]} boundary points and {Yn.shape[0]} multipole sources (kind={kind_str}).")

    evals = Evaluators(center=jnp.asarray(center), scale=scale,
                       Yn=jnp.asarray(Yn), alpha=jnp.asarray(alpha),
                       a=jnp.asarray(a), a_hat=jnp.asarray(a_hat))
    phi_fn, grad_phi = evals.build()

    mins = P.min(axis=0); maxs = P.max(axis=0); span = maxs - mins
    mins = mins - 0.01 * span; maxs = maxs + 0.01 * span
    nx = ny = nz = int(grid_N)
    xs = np.linspace(mins[0], maxs[0], nx)
    ys = np.linspace(mins[1], maxs[1], ny)
    zs = np.linspace(mins[2], maxs[2], nz)
    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)
    Xq = np.column_stack([XX.ravel(order="C"),
                          YY.ravel(order="C"),
                          ZZ.ravel(order="C")])
    if verbose:
        pinfo(f"Grid size: {nx}x{ny}x{nz} = {Xq.shape[0]} nodes.  Spacing dx≈{dx:.3g}, dy≈{dy:.3g}, dz≈{dz:.3g}")
    voxel = min(dx, dy, dz)

    c = np.mean(P, axis=0)
    s = np.sum((P - c) * Nsurf, axis=1)
    avg = float(np.mean(s))
    if avg < 0:
        if verbose:
            pinfo("Normals appear inward on average; flipping.")
        Nsurf = -Nsurf
    else:
        if verbose:
            pinfo("Normals appear outward on average.")

    inside_mask, nn_idx, signed_dist = inside_mask_from_surface(P, Nsurf, Xq)
    if verbose:
        pstat("Inside mask", inside_mask.astype(float))

    if not np.any(inside_mask):
        raise RuntimeError("Inside mask is empty; check surface normals or grid bounds.")

    x0 = Xq[inside_mask][0]

    # Direction sign heuristic (still degenerate but harmless; keeps +1)
    def decide_sign(x_init: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.asarray(x_init, dtype=jnp.float64)
        def advance(xstart: jnp.ndarray, sgn: float, K: int = 8, step: float = 1e-2) -> jnp.ndarray:
            xs = xstart
            g = grad_phi(xs[None, :])[0]
            nrm = jnp.linalg.norm(g) + 1e-30
            g_unit = g / nrm
            for _ in range(K):
                xs = xs + sgn * step * g_unit
                g = grad_phi(xs[None, :])[0]
                nrm = jnp.linalg.norm(g) + 1e-30
                g_unit = g / nrm
            return xs
        xp = advance(x, +1.0)
        xm = advance(x, -1.0)
        return xp, xm

    xp, xm = decide_sign(x0)
    d_plus = float(np.linalg.norm(xp - x0))
    d_minus = float(np.linalg.norm(xm - x0))
    if abs(d_plus - d_minus) <= 5e-5 * max(d_plus, d_minus):
        dir_sign = +1.0
    else:
        dir_sign = +1.0 if d_plus > d_minus else -1.0
    if verbose:
        pinfo(f"Toroidal direction sign chosen as {dir_sign:+.0f} (d+= {d_plus:.3e}, d-= {d_minus:.3e})")

    a_flipped = np.array([float(dir_sign) * float(a[0]), float(a[1])], dtype=float)
    evals = Evaluators(center=jnp.asarray(center), scale=scale,
                       Yn=jnp.asarray(Yn), alpha=jnp.asarray(alpha),
                       a=jnp.asarray(a_flipped), a_hat=jnp.asarray(a_hat))
    phi_fn, grad_phi = evals.build()

    phi_tol = 0.2
    phiq = np.arctan2(Xq[:, 1], Xq[:, 0])
    Z_span = Xq[:, 2].max() - Xq[:, 2].min()
    Z_tol = 0.25 * Z_span
    mask_slice = inside_mask & (np.abs(phiq) < phi_tol) & (np.abs(Xq[:, 2]) < Z_tol)
    if not np.any(mask_slice):
        mask_slice = inside_mask
    R_slice = np.sqrt(Xq[mask_slice][:, 0]**2 + Xq[mask_slice][:, 1]**2)
    R_inner = float(R_slice.min()); R_outer = float(R_slice.max())
    R0_guess = 0.5 * (R_inner + R_outer); Z0_guess = 0.0
    if verbose:
        pinfo(f"Initial axis guess R≈{R0_guess:.3e}, Z≈{Z0_guess:.3e}")
    start_axis_time = time.time()
    R_axis, Z_axis = collapse_to_axis(grad_phi, R0_guess, Z0_guess, nfp=nfp)
    if verbose:
        pinfo(f"Solved axis in {(time.time() - start_axis_time):.2f} s: R={R_axis:.3e}, Z={Z_axis:.3e}")

    n_axis_pts = 512
    phis_axis = jnp.linspace(0.0, 2.0 * jnp.pi, n_axis_pts, endpoint=False)

    @jit
    def B_cyl_axis(R: float, phi: float, Z: float) -> Tuple[jnp.ndarray, float, float]:
        x = R * jnp.cos(phi); y = R * jnp.sin(phi)
        X = jnp.stack([x, y, Z])
        B = grad_phi(X[None, :])[0]
        eR = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0])
        ephi = jnp.array([-jnp.sin(phi), jnp.cos(phi), 0.0])
        BR = jnp.dot(B, eR)
        Bphi = jnp.dot(B, ephi)
        BZ = B[2]
        Bphi = jnp.where(jnp.abs(Bphi) < 1e-12, jnp.sign(Bphi) * 1e-12, Bphi)
        return BR, Bphi, BZ

    @jit
    def fieldline_rhs_axis(phi: float, RZ: jnp.ndarray, args: Any) -> jnp.ndarray:
        R, Z = RZ
        BR, Bphi, BZ = B_cyl_axis(R, phi, Z)
        return jnp.stack([R * BR / Bphi, R * BZ / Bphi])

    term_axis = dfx.ODETerm(fieldline_rhs_axis)
    solver_axis = dfx.Dopri5()
    dt0_axis = float(2.0 * jnp.pi) / 4096.0
    saveat_axis = dfx.SaveAt(ts=phis_axis)
    sol_axis = dfx.diffeqsolve(term_axis, solver_axis, t0=0.0, t1=2.0 * jnp.pi,
                               dt0=dt0_axis, y0=jnp.asarray([R_axis, Z_axis], dtype=jnp.float64),
                               saveat=saveat_axis, max_steps=65536)
    R_path = np.asarray(sol_axis.ys[:, 0]); Z_path = np.asarray(sol_axis.ys[:, 1])
    axis_pts = np.stack([R_path * np.cos(np.asarray(phis_axis)),
                         R_path * np.sin(np.asarray(phis_axis)),
                         Z_path], axis=1)
    if verbose:
        pinfo(f"Axis orbit integrated; sample point: R={axis_pts[0,0]:.3e}, Z={axis_pts[0,2]:.3e}")

    inside_axis, _, signed_axis = inside_mask_from_surface(P, Nsurf, axis_pts)
    if verbose:
        pinfo(
            f"Axis vs surface signed distance: "
            f"min={signed_axis.min():.3e}, max={signed_axis.max():.3e}"
        )
        pinfo(f"Axis points classified inside: {inside_axis.sum()} / {inside_axis.size}")

    bbox_diag = float(np.linalg.norm(maxs - mins))
    if axis_band_radius == 0.0:
        axis_band_radius_eff = 2.5 * voxel
    elif axis_band_radius < 1.0:
        axis_band_radius_eff = max(2.5 * voxel, axis_band_radius * bbox_diag)
    else:
        axis_band_radius_eff = max(2.5 * voxel, float(axis_band_radius))

    h_band_vox = max(1.5 * voxel, float(band_h) * voxel)

    axis_band = axis_band_mask(axis_pts, Xq, axis_band_radius_eff) & inside_mask

    if not np.any(axis_band):
        if verbose:
            pinfo("Axis band empty; rebuilding adaptively based on distance to axis.")

        nbrs_axis = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(axis_pts)
        d_axis, _ = nbrs_axis.kneighbors(Xq)
        d_axis = d_axis[:, 0]

        inside_idx = np.where(inside_mask)[0]
        if inside_idx.size == 0:
            raise RuntimeError("Inside mask empty when building axis band.")

        frac = 0.02
        n_axis_nodes = max(10, int(frac * inside_idx.size))
        order = np.argsort(d_axis[inside_idx])
        chosen = inside_idx[order[:n_axis_nodes]]

        axis_band = np.zeros_like(inside_mask, dtype=bool)
        axis_band[chosen] = True
        axis_band_radius_eff = float(d_axis[chosen].max())

        if verbose:
            pinfo(
                f"Axis band rebuilt adaptively with {n_axis_nodes} nodes; "
                f"effective radius ≈ {axis_band_radius_eff:.3e}"
            )

    band = (inside_mask & (np.abs(signed_dist) <= h_band_vox))

    overlap = band & axis_band
    axis_band[overlap] = True
    band[overlap] = False

    Ntot = Xq.shape[0]
    fixed = np.zeros(Ntot, dtype=bool)
    val = np.zeros(Ntot, dtype=float)
    fixed[band] = True; val[band] = 1.0
    fixed[axis_band] = True; val[axis_band] = 0.0

    if verbose:
        print(f"[INFO] #inside nodes        : {inside_mask.sum()} / {Ntot}")
        print(f"[INFO] #boundary band nodes : {band.sum()} / {Ntot}")
        print(f"[INFO] #axis band nodes     : {axis_band.sum()} / {Ntot}")
        print(f"[INFO] boundary band width  ≈ {h_band_vox:.3e}")
        print(f"[INFO] axis band radius     ≈ {axis_band_radius_eff:.3e}")

    if verbose:
        pinfo("Evaluating ∇φ on grid ...")
    G = np.zeros_like(Xq)
    chunk = 20000
    for start in range(0, Xq.shape[0], chunk):
        Xc = jnp.asarray(Xq[start:start + chunk])
        Gi = np.asarray(grad_phi(Xc))
        bad = ~np.isfinite(Gi).all(axis=1)
        if np.any(bad):
            Gi[bad] = 0.0
        G[start:start + chunk] = Gi

    D = diffusion_tensor(G, eps=eps, delta=delta)

    A_pde, deep_inside = make_linear_operator(
        nx, ny, nz, dx, dy, dz,
        inside_mask, D
    )

    free = deep_inside & (~fixed)
    if not np.any(free):
        raise RuntimeError("No free deep-interior nodes; bands / geometry too tight.")

    # Build lifting for Dirichlet bands: ψ = ψ_free + ψ_fixed
    psi_fixed_full = np.zeros(Ntot, dtype=float)
    psi_fixed_full[fixed] = val[fixed]

    F0_full = A_pde @ psi_fixed_full  # A_pde applied to known fixed field
    b_free = -F0_full[free]

    Nfree = int(free.sum())

    def matvec_free(u_free: np.ndarray) -> np.ndarray:
        u_full = np.zeros(Ntot, dtype=float)
        u_full[free] = u_free
        Au_full = A_pde @ u_full
        return Au_full[free]

    A_eff = LinearOperator(
        (Nfree, Nfree),
        matvec=matvec_free,
        rmatvec=matvec_free,
        dtype=float
    )

    if verbose:
        pinfo("Solving linear system (CG) ...")
    # start with zeros
    psi_free, info = cg(A_eff, b_free, rtol=cg_tol, maxiter=cg_maxit)
    if info != 0 and verbose:
        pinfo(f"[WARN] CG returned info={info} (0 means full convergence).")

    psi = np.array(psi_fixed_full)
    psi[free] = psi_free

    # Small safety clamp to [0,1] inside the domain
    psi_inside = psi[inside_mask]
    psi_inside = np.clip(psi_inside, 0.0, 1.0)
    psi[inside_mask] = psi_inside

    r_full = A_pde @ psi

    if verbose:
        pstat("ψ (all nodes)", psi)
        pstat("ψ on boundary band (should be ~1)", psi[band])
        pstat("ψ on axis band (should be ~0)", psi[axis_band])
        free_inside = inside_mask & (~fixed)
        if np.any(free_inside):
            pstat("ψ on free interior", psi[free_inside])
        pstat("Full residual", r_full)
        # We don't have a nonzero RHS on free nodes anymore; measure absolute residual
        print(f"||r||_2 over free nodes = {np.linalg.norm(r_full[free]):.3e}")

    psi3 = psi.reshape(nx, ny, nz)

    dpsidx = (psi3[2:, 1:-1, 1:-1] - psi3[:-2, 1:-1, 1:-1]) / (2 * dx)
    dpsidy = (psi3[1:-1, 2:, 1:-1] - psi3[1:-1, :-2, 1:-1]) / (2 * dy)
    dpsidz = (psi3[1:-1, 1:-1, 2:] - psi3[1:-1, 1:-1, :-2]) / (2 * dz)

    G3 = G.reshape(nx, ny, nz, 3)
    B_core = G3[1:-1, 1:-1, 1:-1, :]
    gnorm_core = np.linalg.norm(B_core, axis=-1)
    core_mask = (gnorm_core > 1e-10)

    t_hat_core = np.zeros_like(B_core)
    t_hat_core[core_mask] = (
        B_core[core_mask].T / gnorm_core[core_mask]
    ).T

    par_grad_full = (
        t_hat_core[..., 0] * dpsidx +
        t_hat_core[..., 1] * dpsidy +
        t_hat_core[..., 2] * dpsidz
    )
    grad_mag_full = np.sqrt(dpsidx**2 + dpsidy**2 + dpsidz**2)

    par_grad = par_grad_full[core_mask]
    grad_mag = grad_mag_full[core_mask]

    q_metric = np.abs(par_grad) / np.maximum(grad_mag, 1e-14)

    if verbose and q_metric.size > 0:
        p = np.percentile(q_metric, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        print(
            "[ALIGN] q = |t·∇ψ|/|∇ψ| stats: "
            f"min={p[0]:.3e} p1={p[1]:.3e} p5={p[2]:.3e} "
            f"p50={p[4]:.3e} p95={p[6]:.3e} p99={p[7]:.3e} max={p[8]:.3e}"
        )

    if plot:
        if q_metric.size > 0:
            fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

            ax1[0].hist(q_metric, bins=80)
            ax1[0].set_yscale("log")
            ax1[0].set_xlabel(r"$q = |t\cdot\nabla\psi|/|\nabla\psi|$")
            ax1[0].set_ylabel("count")
            ax1[0].set_title("Alignment error histogram")

            res_free = r_full[free]
            ax1[1].hist(np.abs(res_free), bins=80)
            ax1[1].set_yscale("log")
            ax1[1].set_xlabel(r"$|r|$")
            ax1[1].set_ylabel("count")
            ax1[1].set_title("PDE residual |r| on free nodes")

            fig1.suptitle("FCI ψ diagnostics")
            if save_figures:
                fig1.savefig("fci_psi_diagnostics.png")

        try:
            psi3 = psi.reshape(nx, ny, nz)
            inside3 = inside_mask.reshape(nx, ny, nz)
            psi_RZphi, Rs_cyl, phis_cyl, Zs_cyl, mask_RZphi = build_psi_RZphi_volume(
                psi3, xs, ys, zs, P, inside3, nR=128, nphi=64, nZ=128
            )

            Rb = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
            phi_b = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2*np.pi)
            Zb = P[:, 2]

            jj_list = np.linspace(0, len(phis_cyl) - 1, 4, dtype=int)
            figRZ = plot_psi_maps_RZ_panels(
                np.sqrt(psi_RZphi), Rs_cyl, phis_cyl, Zs_cyl, jj_list,
                Rb=Rb, Zb=Zb, phi_b=phi_b, title="ψ(R,Z)"
            )
            figRZ.suptitle("ψ(R,Z) at selected toroidal angles")
            if save_figures:
                figRZ.savefig("fci_psi_RZ_panels.png")

        except Exception as e:
            pinfo(f"[WARN] Failed to build RZφ panels: {e}")

        try:
            psi3 = psi.reshape(nx, ny, nz)
            fig3d = plot_3d_axis_boundary_interp(P, axis_pts, psi3, xs, ys, zs)
            if save_figures:
                fig3d.savefig("fci_psi_3d_axis_boundary.png")
        except Exception as e:
            pinfo(f"[WARN] Failed to plot 3D axis/boundary: {e}")

        R = np.sqrt(Xq[:,0]**2 + Xq[:,1]**2)
        R_axis_line = np.sqrt(axis_pts[:,0]**2 + axis_pts[:,1]**2).mean()
        dist_to_axis = np.abs(R - R_axis_line)    # crude measure

        try:
            mask_core = inside_mask & (~fixed)
            plt.figure()
            plt.scatter(dist_to_axis[mask_core], psi[mask_core], s=2, alpha=0.3)
            plt.xlabel("distance to axis (rough)")
            plt.ylabel("ψ")
            plt.title("ψ vs distance to axis (inside free region)")
            if save_figures:
                plt.savefig("fci_psi_vs_distance_to_axis.png")
        except Exception as e:
            pinfo(f"[WARN] Failed to plot ψ vs distance to axis: {e}")

        plt.show()

    result = {
        'psi': psi,
        'grid': {
            'xs': xs, 'ys': ys, 'zs': zs,
            'mins': mins, 'maxs': maxs,
        },
        'inside': inside_mask,
        'bands': {
            'boundary': band,
            'axis': axis_band,
        },
        'quality': {
            'q_metric': q_metric,
            'parallel_dot_grad': par_grad,
            'residual': r_full,
        },
        'axis': {
            'R': R_axis,
            'Z': Z_axis,
            'points': axis_pts,
        },
    }
    return result

###############################################################################
# Command line interface
###############################################################################

if __name__ == "__main__":

    default_solution = "wout_precise_QA_solution.npz"
    # default_solution = "wout_precise_QH_solution.npz"
    # default_solution = "wout_SLAM_4_coils_solution.npz"
    # default_solution = "wout_SLAM_6_coils_solution.npz"

    nfp_default = 2
    if 'QH' in default_solution:
        nfp_default = 4

    parser = argparse.ArgumentParser(description="Solve field–aligned flux function ψ via FCI diffusion.")
    parser.add_argument("npz", nargs="?", default=resolve_npz_file_location(default_solution),
                        help="MFS solution checkpoint (*.npz) containing center, scale, Yn, alpha, a, a_hat, P, N")
    parser.add_argument("--N", type=int, default=56, help="Grid resolution per axis")
    parser.add_argument("--eps", type=float, default=2e-2, help="Perpendicular diffusion weight")
    parser.add_argument("--delta", type=float, default=5e-2, help="Isotropic diffusion floor")
    parser.add_argument("--band-h", type=float, default=1.0, help="Boundary band thickness multiplier")
    parser.add_argument("--axis-band-radius", type=float, default=0.0, help="Axis band radius; 0=auto, <1=fraction of bbox, ≥1=absolute")
    parser.add_argument("--cg-tol", type=float, default=1e-8, help="CG tolerance (default: 1e-8)")
    parser.add_argument("--cg-maxit", type=int, default=2000, help="CG maximum iterations (default: 2000)")
    parser.add_argument("--nfp", type=int, default=nfp_default, help="Number of field periods (default: 2)")
    parser.add_argument("--no-amg", action="store_true", help="(Ignored in this version; AMG removed.)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--save-figures", action="store_true", default=True, help="Save diagnostic figures to disk.")
    args = parser.parse_args()

    res = solve_fci(
        args.npz, grid_N=args.N, eps=args.eps, band_h=args.band_h,
        axis_band_radius=args.axis_band_radius, cg_tol=args.cg_tol,
        cg_maxit=args.cg_maxit, verbose=True, plot=(not args.no_plot),
        nfp=args.nfp, delta=args.delta, no_amg=args.no_amg,
        save_figures=args.save_figures
    )

    psi_all = res['psi']
    inside_mask = res['inside']
    fixed_nodes = (np.abs(psi_all - 1.0) < 1e-10) | (np.abs(psi_all) < 1e-10)
    free_inside = inside_mask & (~fixed_nodes)
    psi_in = psi_all[free_inside]
    if psi_in.size > 0:
        pstat("Solution ψ (inside free region)", psi_in)
