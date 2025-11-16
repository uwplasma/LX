#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flux surface solver inspired by flux–coordinate independent (FCI) approaches (Hariri & Ottaviani),
using a strongly anisotropic diffusion tensor aligned with ∇φ to construct a flux-like scalar ψ.
The goal is to construct a scalar potential ψ that is approximately constant
along magnetic field lines while diffusing across those lines. 
Conceptually, we want ∇·[(P_par + ε P_perp) ∇ψ], where P_par = b b and
P_perp = I - b b. In practice we implement this in two ways:

  • use_fci=True: parallel part from a field-line mapped ∂_s^2 (FCI),
    perpendicular part from a small isotropic Laplacian (approx. ε ∇²).

  • use_fci=False: full tensor anisotropic operator -∇·(D ∇ψ) with
    D ≈ P_par + ε P_perp + δ I.

The boundary Γ_bnd is a thin ribbon near the physical surface
(the outer boundary of the domain) and Γ_axis is a thin tube around the
magnetic axis.  This formulation follows the flux–coordinate independent
approach pioneered by Hariri and Ottaviani and later developed in numerous
plasma simulation codes.

This version:

  * Uses JAX to evaluate φ and ∇φ.
  * Uses Diffrax to find the magnetic axis.
  * Imposes ψ=1 on a boundary band and ψ=0 on an axis band via a standard
    linear "lifting" so the matrix-free operator remains a pure PDE operator.
  * Includes diagnostics on residuals and a field-alignment error metric q = |t·∇ψ|/|∇ψ|.
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
import jax.lax as lax
from functools import partial

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

# ----------------------------- Plotting utilities ---------------------------- #
def build_psi_RZphi_volume(psi3, xs, ys, zs, P, inside3,
                           nR=128, nphi=64, nZ=128):
    Rb = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
    Rs = np.linspace(Rb.min(), Rb.max(), nR)
    Zs = np.linspace(P[:, 2].min(), P[:, 2].max(), nZ)
    phis = np.linspace(0.0, 2.0*np.pi, nphi, endpoint=True)

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
                            R_axis=None, Z_axis=None, phi_axis=None,
                            title="ψ(R,Z)"):
    fig, axa = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)
    axa = axa.ravel()
    Rmin, Rmax = float(np.nanmin(Rs)), float(np.nanmax(Rs))
    Zmin, Zmax = float(np.nanmin(Zs)), float(np.nanmax(Zs))
    extent = [Rmin, Rmax, Zmin, Zmax]

    for kk, jj in enumerate(jj_list):
        # psi_slice = psi_RZphi[:, jj, :].T  # shape (nZ, nR)
        raw_slice = psi_RZphi[:, jj, :].T  # (nZ, nR)
        psi_slice = np.ma.masked_invalid(raw_slice)
        
        im = axa[kk].imshow(psi_slice, origin='lower', aspect='equal', extent=extent)
        
        # make outside-domain transparent/white
        im.cmap.set_bad("white", alpha=0.0)
        
        axa[kk].contour(Rs, Zs, psi_slice, levels=10, colors='white', linewidths=0.5, alpha=1.0)
        axa[kk].set_title(f"{title}: φ≈{phis[jj]:+.2f}")
        axa[kk].set_xlabel("R"); axa[kk].set_ylabel("Z")
        im.set_clim(0, 1)
        plt.colorbar(im, ax=axa[kk], shrink=0.85)

        # boundary overlay (markers above contours/images)
        if Rb is not None and phi_b is not None and Zb is not None:
            dphi_b = np.abs(np.angle(np.exp(1j * (phi_b - phis[jj]))))
            # mask_b = dphi_b < (np.pi / len(phis))
            mask_b = dphi_b < 0.05#2.0 * 2.0 * np.pi / len(phis)
            axa[kk].scatter(
                Rb[mask_b], Zb[mask_b],
                s=10, c='k', alpha=1.0,
                zorder=5, label="boundary" if kk == 0 else None,
            )

        # magnetic axis overlay (markers above everything)
        if R_axis is not None and phi_axis is not None and Z_axis is not None:
            dphi_a = np.abs(np.angle(np.exp(1j * (phi_axis - phis[jj]))))
            mask_a = dphi_a < 0.2 * 2.0 * np.pi / len(phis)
            axa[kk].scatter(
                R_axis[mask_a], Z_axis[mask_a],
                s=30, c='white', marker='.',
                zorder=6, label="axis" if kk == 0 else None,
            )
    # one legend
    handles, labels = axa[0].get_legend_handles_labels()
    if handles: fig.legend(handles, labels, loc="center", framealpha=0.7, facecolor='lightgray')

    return fig

def fix_matplotlib_3d(ax):
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1]-x_limits[0]); x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1]-y_limits[0]); y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1]-z_limits[0]); z_mid = np.mean(z_limits)
    R = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid-R, x_mid+R]); ax.set_ylim3d([y_mid-R, y_mid+R]); ax.set_zlim3d([z_mid-R, z_mid+R])

def plot_3d_axis_boundary_interp(P, axis_pts, psi3, xs, ys, zs):
    interp_psi = RegularGridInterpolator((xs, ys, zs), psi3,
                                         bounds_error=False, fill_value=np.nan)

    psi_on_P = interp_psi(P)
    psi_axis = interp_psi(axis_pts)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Boundary coloured by ψ ~ 1
    sc_bnd = ax.scatter(
        P[:, 0], P[:, 1], P[:, 2],# c=psi_on_P,
        s=0.3, alpha=0.7, edgecolor="b", label="boundary",)

    # Axis coloured by ψ ~ 0
    sc_axis = ax.scatter(
        axis_pts[:, 0], axis_pts[:, 1], axis_pts[:, 2],# c=psi_axis,
        s=0.3, alpha=0.9, edgecolor="k", label="magnetic axis",)

    # sc_bnd.set_clim(0, 1)
    # sc_axis.set_clim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.set_title("Boundary vs magnetic axis (coloured by ψ)")

    # fig.colorbar(sc_bnd, ax=ax, shrink=0.7, label=r"$\psi$")
    ax.legend(loc="best")
    fix_matplotlib_3d(ax)
    plt.tight_layout()

    return fig

###############################################################################
# JAX helpers for Green's function and gradient
###############################################################################

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

    def build(self):
        sc_c = jnp.asarray(self.center)
        sc_s = float(self.scale)
        Yn_c = jnp.asarray(self.Yn)
        alpha_c = jnp.asarray(self.alpha)
        a_c = jnp.asarray(self.a)
        a_hatc = jnp.asarray(self.a_hat)

        dS_single = jit(lambda xn: jnp.sum(
            vmap(lambda y: grad_green_x(xn, y))(Yn_c) * alpha_c[:, None],
            axis=0
        ))

        grad_mv = make_mv_grads(a_c, a_hatc, sc_c, sc_s)

        @jit
        def grad_phi_fn(X: jnp.ndarray) -> jnp.ndarray:
            Xn = (X - sc_c) * sc_s
            return grad_mv(X) + sc_s * vmap(dS_single)(Xn)

        return grad_phi_fn

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
                     nfp: int, nsteps: int = 500, max_newton: int = 12,
                     tol: float = 1e-6) -> Tuple[float, float]:
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
        Bphi = jnp.where(jnp.abs(Bphi) < 1e-12, 1e-12, Bphi)
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
    stepsize_controller = dfx.PIDController(rtol=1e-4, atol=1e-4)

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

def make_fieldline_phi_rhs_jax(grad_phi_fn):
    """
    RHS for tracing field lines using toroidal angle φ.
    Works for scalar or batched R, Z, φ (leading dimensions broadcast).
    """

    def B_cyl(R, phi, Z):
        # R, phi, Z can be scalars or arrays with same shape
        x = R * jnp.cos(phi)
        y = R * jnp.sin(phi)
        X = jnp.stack([x, y, Z], axis=-1)  # (..., 3)
        B = grad_phi_fn(X)                 # (..., 3)

        eR   = jnp.stack([jnp.cos(phi),  jnp.sin(phi), 0.0 * phi], axis=-1)
        ephi = jnp.stack([-jnp.sin(phi), jnp.cos(phi), 0.0 * phi], axis=-1)

        BR   = jnp.sum(B * eR,   axis=-1)
        Bphi = jnp.sum(B * ephi, axis=-1)
        BZ   = B[..., 2]

        Bnorm      = jnp.linalg.norm(B, axis=-1)
        Bphi_floor = 1e-7 * Bnorm + 1e-14
        Bphi_safe  = jnp.where(jnp.abs(Bphi) < Bphi_floor,
                               jnp.sign(Bphi) * Bphi_floor,
                               Bphi)
        return BR, Bphi_safe, BZ

    def rhs(phi, RZ, args):
        R = RZ[..., 0]
        Z = RZ[..., 1]
        BR, Bphi, BZ = B_cyl(R, phi, Z)
        dR_dphi = R * BR / Bphi
        dZ_dphi = R * BZ / Bphi
        return jnp.stack([dR_dphi, dZ_dphi], axis=-1)

    return rhs

@partial(jax.jit, static_argnames=("grad_phi_fn", "nsteps"))
def trace_to_delta_phi_batched(
    grad_phi_fn,
    R0: jnp.ndarray,
    Z0: jnp.ndarray,
    phi0: jnp.ndarray,
    dphi_target: jnp.ndarray,
    nsteps: int = 16,
):
    """
    Batched version of trace_to_delta_phi.

    Inputs:
      R0, Z0, phi0, dphi_target: arrays with the same leading shape (...,).
    Returns:
      R1, Z1, phi1, L: arrays with the same shape as R0.
    """

    rhs = make_fieldline_phi_rhs_jax(grad_phi_fn)

    R  = jnp.asarray(R0,        dtype=jnp.float64)
    Z  = jnp.asarray(Z0,        dtype=jnp.float64)
    ph = jnp.asarray(phi0,      dtype=jnp.float64)
    L0 = jnp.zeros_like(R,      dtype=jnp.float64)

    dphi = jnp.asarray(dphi_target, dtype=jnp.float64) / nsteps

    def body_fun(i, carry):
        R, Z, phi, L = carry

        RZ = jnp.stack([R, Z], axis=-1)        # (..., 2)
        k1 = rhs(phi, RZ, None)               # (..., 2)
        k1R, k1Z = k1[..., 0], k1[..., 1]

        R_pred   = R   + dphi * k1R
        Z_pred   = Z   + dphi * k1Z
        phi_pred = phi + dphi

        RZ_pred = jnp.stack([R_pred, Z_pred], axis=-1)
        k2 = rhs(phi_pred, RZ_pred, None)
        k2R, k2Z = k2[..., 0], k2[..., 1]

        R_new   = R   + 0.5 * dphi * (k1R + k2R)
        Z_new   = Z   + 0.5 * dphi * (k1Z + k2Z)
        phi_new = phi + dphi

        x  = R   * jnp.cos(phi)
        y  = R   * jnp.sin(phi)
        z  = Z
        x2 = R_new   * jnp.cos(phi_new)
        y2 = R_new   * jnp.sin(phi_new)
        z2 = Z_new
        dL = jnp.sqrt((x2 - x)**2 + (y2 - y)**2 + (z2 - z)**2)

        return (R_new, Z_new, phi_new, L + dL)

    R1, Z1, phi1, L1 = lax.fori_loop(0, nsteps, body_fun, (R, Z, ph, L0))
    return R1, Z1, phi1, L1

def trace_many_to_delta_phi(grad_phi_fn, seeds_RZphi, dphi_target, nsteps=16):
    """
    seeds_RZphi: (N, 3) array of [R0, Z0, phi0]
    dphi_target: scalar or (N,) array
    """
    R0  = jnp.asarray(seeds_RZphi[:, 0])
    Z0  = jnp.asarray(seeds_RZphi[:, 1])
    ph0 = jnp.asarray(seeds_RZphi[:, 2])
    dphi = jnp.asarray(dphi_target) * jnp.ones_like(R0)

    R1, Z1, ph1, L = trace_to_delta_phi_batched(grad_phi_fn, R0, Z0, ph0, dphi, nsteps=nsteps)
    return np.stack([np.array(R1), np.array(Z1), np.array(ph1), np.array(L)], axis=-1)

###############################################################################
# Diffusion tensor
###############################################################################

def diffusion_tensor_jax(gradphi_j: jnp.ndarray, eps: float, delta: float=0.0, b_floor: float=1e-12) -> jnp.ndarray:
    I = jnp.eye(3)[None, :, :]
    g = gradphi_j
    n = jnp.linalg.norm(g, axis=-1, keepdims=True)

    good = n[..., 0] > b_floor
    b = jnp.where(good[..., None], g / n, 0.0)

    P_par  = jnp.einsum("ni,nj->nij", b, b)
    P_perp = I - P_par
    D = P_par + eps * P_perp
    if delta != 0.0:
        D = D + delta * I
    D = jnp.where(good[..., None, None], D, (1.0 + delta) * I)
    return D

###############################################################################
# Full-tensor Cartesian FV operator (27-point stencil via face fluxes)
###############################################################################

@dataclass
class FCIConnectivity:
    idx_plus: np.ndarray   # shape (N, 8) int64
    w_plus:   np.ndarray   # shape (N, 8) float64
    L_plus:   np.ndarray   # shape (N,)
    idx_minus: np.ndarray
    w_minus:   np.ndarray
    L_minus:   np.ndarray
    valid:     np.ndarray  # shape (N,) bool: true where both ends are inside
    
def build_fci_connectivity_chunked(
    xs,
    ys,
    zs,
    inside_mask: np.ndarray,
    grad_phi_fn,
    nfp: int,
    dphi_per_step: float = None,
    nsteps: int = 32,
    verbose: bool = True,
    chunk_size: int | None = None,
    fci_planes_per_field_period: int = 16
) -> FCIConnectivity:
    """
    Chunked / batched version of build_fci_connectivity.

    - Loops only over interior nodes (inside_mask == True)
    - Processes them in chunks with a batched JAX integrator
    - Still uses numpy + a small Python loop inside each chunk for trilinear weights
    """
    nx, ny, nz = len(xs), len(ys), len(zs)
    N = nx * ny * nz
    assert inside_mask.shape[0] == N

    # choose Δφ if not provided
    if dphi_per_step is None:
        N_par = fci_planes_per_field_period  # planes per field period
        dphi_per_step = 2.0 * np.pi / (nfp * N_par)

    # allocate outputs
    idx_plus  = np.zeros((N, 8), dtype=np.int64)
    idx_minus = np.zeros((N, 8), dtype=np.int64)
    w_plus    = np.zeros((N, 8), dtype=float)
    w_minus   = np.zeros((N, 8), dtype=float)
    L_plus    = np.zeros(N, dtype=float)
    L_minus   = np.zeros(N, dtype=float)
    valid     = np.zeros(N, dtype=bool)

    # flattened grid
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)
    Xflat = np.column_stack([XX.ravel(order="C"),
                             YY.ravel(order="C"),
                             ZZ.ravel(order="C")])

    inside_indices = np.where(inside_mask)[0]
    n_inside_total = int(inside_indices.size)
    if n_inside_total == 0:
        raise RuntimeError("No interior nodes in build_fci_connectivity_chunked.")

    start_time = time.time()
    last_print = start_time
    n_inside_processed = 0
    
    # --- automatic chunk_size selection ---
    if chunk_size is None:
        n_devices = max(1, len(jax.devices()))
        # aim for ~ 8 chunks per device, clamp to [128, 4096]
        target_chunks_per_device = 8
        est_chunk = n_inside_total // (n_devices * target_chunks_per_device + 1)
        chunk_size = int(np.clip(est_chunk, 128, 4096))
        if verbose:
            pinfo(f"[FCI] Auto chunk_size={chunk_size} "
                  f"(N_inside={n_inside_total}, n_devices={n_devices})")

    # helper: batched grad_phi to check finiteness
    @jax.jit
    def _grad_phi_batched(X_batch):
        return grad_phi_fn(X_batch)

    for chunk_start in range(0, n_inside_total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_inside_total)
        idx_chunk = inside_indices[chunk_start:chunk_end]
        X_chunk = Xflat[idx_chunk]   # (Nc, 3)
        Nc = X_chunk.shape[0]

        x = X_chunk[:, 0]
        y = X_chunk[:, 1]
        z = X_chunk[:, 2]

        R = np.sqrt(x * x + y * y)
        phi0 = np.arctan2(y, x)
        Z = z

        # Near-axis nodes: skip (we don’t trust cylindrical step there)
        mask_R_ok = R > 1e-5

        # Check grad_phi at start points in batch
        X_chunk_j = jnp.asarray(X_chunk, dtype=jnp.float64)
        B_chunk = np.asarray(_grad_phi_batched(X_chunk_j))   # (Nc, 3)
        mask_finite = np.isfinite(B_chunk).all(axis=1)

        # Nodes in this chunk that we’ll try to trace
        mask_ok = mask_R_ok & mask_finite
        if not np.any(mask_ok):
            # nothing usable in this chunk
            n_inside_processed += Nc
            if verbose:
                now = time.time()
                dt = now - last_print
                total_dt = now - start_time
                print(
                    f"[FCI] inside nodes {n_inside_processed}/{n_inside_total} "
                    f"({100.0*n_inside_processed/max(1,n_inside_total):.1f}%) "
                    f"valid={int(valid.sum())} elapsed={total_dt:.1f}s (+{dt:.1f}s) [all skipped in chunk]"
                )
                last_print = now
            continue

        idx_chunk_ok = idx_chunk[mask_ok]
        R0  = R[mask_ok]
        Z0  = Z[mask_ok]
        ph0 = phi0[mask_ok]

        # seeds_RZphi: (Nc_ok, 3)
        seeds_RZphi = np.stack([R0, Z0, ph0], axis=1)

        # forward & backward batched traces (returns numpy arrays)
        fwd = trace_many_to_delta_phi(grad_phi_fn, seeds_RZphi,
                                      +dphi_per_step, nsteps=nsteps)
        bwd = trace_many_to_delta_phi(grad_phi_fn, seeds_RZphi,
                                      -dphi_per_step, nsteps=nsteps)
        # fwd / bwd shape: (Nc_ok, 4) with columns [R1, Z1, phi1, L]

        Rf, Zf, phif, Lf = fwd.T
        Rb, Zb, phib, Lb = bwd.T

        # Map cylindrical to Cartesian
        xf = Rf * np.cos(phif)
        yf = Rf * np.sin(phif)
        zf = Zf

        xb = Rb * np.cos(phib)
        yb = Rb * np.sin(phib)
        zb = Zb

        # For each successful node in this chunk, compute trilinear weights
        for local_idx, p in enumerate(idx_chunk_ok):
            # If arc length came back NaN or inf, skip
            if not np.isfinite(Lf[local_idx]) or not np.isfinite(Lb[local_idx]):
                continue

            idxp, wp = trilinear_weights(xs, ys, zs,
                                         (xf[local_idx], yf[local_idx], zf[local_idx]))
            idxm, wm = trilinear_weights(xs, ys, zs,
                                         (xb[local_idx], yb[local_idx], zb[local_idx]))

            # require that at least some contributing nodes are inside
            if not (inside_mask[idxp].any() and inside_mask[idxm].any()):
                continue

            idx_plus[p, :]  = idxp
            w_plus[p, :]    = wp
            L_plus[p]       = max(float(Lf[local_idx]), 1e-8)

            idx_minus[p, :] = idxm
            w_minus[p, :]   = wm
            L_minus[p]      = max(float(Lb[local_idx]), 1e-8)

            valid[p] = True

        n_inside_processed += Nc
        if verbose:
            now = time.time()
            dt  = now - last_print
            total_dt = now - start_time
            print(
                f"[FCI] inside nodes {n_inside_processed}/{n_inside_total} "
                f"({100.0*n_inside_processed/max(1,n_inside_total):.1f}%) "
                f"valid={int(valid.sum())} elapsed={total_dt:.1f}s (+{dt:.1f}s)"
            )
            last_print = now

    if verbose:
        print(
            f"[FCI] Connectivity build finished: "
            f"valid nodes={valid.sum()} / {inside_mask.sum()}, "
            f"total time={time.time() - start_time:.1f}s"
        )

    return FCIConnectivity(
        idx_plus=idx_plus, w_plus=w_plus, L_plus=L_plus,
        idx_minus=idx_minus, w_minus=w_minus, L_minus=L_minus,
        valid=valid,
    )

def build_fci_connectivity_cylindrical(
    Rs,
    phis,
    Zs,
    inside_mask: np.ndarray,
    grad_phi_fn,
    nfp: int,
    dphi_per_step: float = None,
    nsteps: int = 32,
    verbose: bool = True,
    chunk_size: int | None = None,
    fci_planes_per_field_period: int = 16,
) -> FCIConnectivity:
    """
    FCI connectivity builder on a cylindrical grid (R, φ, Z).

    Rs    : 1D array, size nR
    phis  : 1D array, size nphi (assumed 0..2π-periodic)
    Zs    : 1D array, size nZ
    inside_mask : length N=nR*nphi*nZ, True for interior nodes

    Produces the same FCIConnectivity structure, but interpolation is done
    in cylindrical coordinates (R,φ,Z). Field-line tracing is done in
    physical space using grad_phi_fn, as before.
    """
    nR, nphi, nZ = len(Rs), len(phis), len(Zs)
    N = nR * nphi * nZ
    assert inside_mask.shape[0] == N

    # choose Δφ for mapping between FCI planes if not provided
    if dphi_per_step is None:
        N_par = fci_planes_per_field_period  # planes per field period
        dphi_per_step = 2.0 * np.pi / (nfp * N_par)

    # allocate outputs
    idx_plus  = np.zeros((N, 8), dtype=np.int64)
    idx_minus = np.zeros((N, 8), dtype=np.int64)
    w_plus    = np.zeros((N, 8), dtype=float)
    w_minus   = np.zeros((N, 8), dtype=float)
    L_plus    = np.zeros(N, dtype=float)
    L_minus   = np.zeros(N, dtype=float)
    valid     = np.zeros(N, dtype=bool)

    # Build (R,φ,Z) arrays and flatten them
    RR, PHI, ZZ = np.meshgrid(Rs, phis, Zs, indexing="ij")  # (nR,nphi,nZ)
    R_flat   = RR.ravel(order="C")
    phi_flat = PHI.ravel(order="C")
    Z_flat   = ZZ.ravel(order="C")

    inside_indices = np.where(inside_mask)[0]
    n_inside_total = int(inside_indices.size)
    if n_inside_total == 0:
        raise RuntimeError("No interior nodes in build_fci_connectivity_cylindrical.")

    start_time = time.time()
    last_print = start_time
    n_inside_processed = 0

    # automatic chunk_size selection
    if chunk_size is None:
        n_devices = max(1, len(jax.devices()))
        target_chunks_per_device = 8
        est_chunk = n_inside_total // (n_devices * target_chunks_per_device + 1)
        chunk_size = int(np.clip(est_chunk, 128, 4096))
        if verbose:
            pinfo(f"[FCI-cyl] Auto chunk_size={chunk_size} "
                  f"(N_inside={n_inside_total}, n_devices={n_devices})")

    @jax.jit
    def _grad_phi_batched(X_batch):
        return grad_phi_fn(X_batch)

    # total φ-period for wrapping
    phi_min, phi_max = phis[0], phis[-1]
    phi_period = phi_max - phi_min + (phis[1] - phis[0])  # ~2π, robust to endpoint

    for chunk_start in range(0, n_inside_total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_inside_total)
        idx_chunk = inside_indices[chunk_start:chunk_end]
        Nc = idx_chunk.size

        R0  = R_flat[idx_chunk]
        phi0 = phi_flat[idx_chunk]
        Z0  = Z_flat[idx_chunk]

        # Skip near-axis nodes (R very small) to avoid cylindrical singularity
        mask_R_ok = R0 > 1e-5

        # Physical coordinates for grad_phi sanity check
        x0 = R0 * np.cos(phi0)
        y0 = R0 * np.sin(phi0)
        z0 = Z0
        X_chunk = np.column_stack([x0, y0, z0])

        X_chunk_j = jnp.asarray(X_chunk, dtype=jnp.float64)
        B_chunk = np.asarray(_grad_phi_batched(X_chunk_j))   # (Nc, 3)
        mask_finite = np.isfinite(B_chunk).all(axis=1)

        mask_ok = mask_R_ok & mask_finite
        if not np.any(mask_ok):
            n_inside_processed += Nc
            if verbose:
                now = time.time()
                dt  = now - last_print
                total_dt = now - start_time
                print(
                    f"[FCI-cyl] inside nodes {n_inside_processed}/{n_inside_total} "
                    f"({100.0*n_inside_processed/max(1,n_inside_total):.1f}%) "
                    f"valid={int(valid.sum())} elapsed={total_dt:.1f}s (+{dt:.1f}s) [all skipped in chunk]"
                )
                last_print = now
            continue

        idx_chunk_ok = idx_chunk[mask_ok]
        R0_ok  = R0[mask_ok]
        Z0_ok  = Z0[mask_ok]
        phi0_ok = phi0[mask_ok]

        seeds_RZphi = np.stack([R0_ok, Z0_ok, phi0_ok], axis=1)

        # forward & backward batched traces (returns numpy arrays)
        fwd = trace_many_to_delta_phi(grad_phi_fn, seeds_RZphi,
                                      +dphi_per_step, nsteps=nsteps)
        bwd = trace_many_to_delta_phi(grad_phi_fn, seeds_RZphi,
                                      -dphi_per_step, nsteps=nsteps)
        Rf, Zf, phif, Lf = fwd.T
        Rb, Zb, phib, Lb = bwd.T

        # wrap φ into the grid interval (assume uniform 0..2π)
        def wrap_phi(phi_arr):
            phi_wrapped = (phi_arr - phi_min) % phi_period + phi_min
            return phi_wrapped

        phif = wrap_phi(phif)
        phib = wrap_phi(phib)

        # For each successful node in this chunk, compute trilinear weights
        for local_idx, p in enumerate(idx_chunk_ok):
            if (not np.isfinite(Lf[local_idx])) or (not np.isfinite(Lb[local_idx])):
                continue

            # interpolation in cylindrical coordinates (R,φ,Z)
            idxp, wp = trilinear_weights(Rs, phis, Zs,
                                         (Rf[local_idx], phif[local_idx], Zf[local_idx]))
            idxm, wm = trilinear_weights(Rs, phis, Zs,
                                         (Rb[local_idx], phib[local_idx], Zb[local_idx]))

            if not (inside_mask[idxp].any() and inside_mask[idxm].any()):
                continue

            idx_plus[p, :]  = idxp
            w_plus[p, :]    = wp
            L_plus[p]       = max(float(Lf[local_idx]), 1e-8)

            idx_minus[p, :] = idxm
            w_minus[p, :]   = wm
            L_minus[p]      = max(float(Lb[local_idx]), 1e-8)

            valid[p] = True

        n_inside_processed += Nc
        if verbose:
            now = time.time()
            dt  = now - last_print
            total_dt = now - start_time
            print(
                f"[FCI-cyl] inside nodes {n_inside_processed}/{n_inside_total} "
                f"({100.0*n_inside_processed/max(1,n_inside_total):.1f}%) "
                f"valid={int(valid.sum())} elapsed={total_dt:.1f}s (+{dt:.1f}s)"
            )
            last_print = now

    if verbose:
        print(
            f"[FCI-cyl] Connectivity build finished: "
            f"valid nodes={valid.sum()} / {inside_mask.sum()}, "
            f"total time={time.time() - start_time:.1f}s"
        )

    return FCIConnectivity(
        idx_plus=idx_plus, w_plus=w_plus, L_plus=L_plus,
        idx_minus=idx_minus, w_minus=w_minus, L_minus=L_minus,
        valid=valid,
    )

def _find_cell_indices(coord, grid):
    """
    Given a coordinate array coord (scalar) and a 1D grid array grid (monotone),
    return (i, w) such that coord is between grid[i] and grid[i+1], and
      value(coord) ≈ (1-w)*val[i] + w*val[i+1].
    Clamp coord into [grid[0], grid[-1]] and index into interior cells.
    """
    # --- clamp coord into [grid[0], grid[-1]] ---
    coord_clamped = np.clip(coord, grid[0], grid[-1])

    # fractional index
    t = (coord_clamped - grid[0]) / (grid[-1] - grid[0])
    idx_float = t * (len(grid) - 1)
    i = int(np.floor(idx_float))
    # clamp i to [0, len(grid)-2]
    i = max(0, min(i, len(grid) - 2))

    x0 = grid[i]
    x1 = grid[i + 1]
    if x1 == x0:
        w = 0.0
    else:
        w = (coord_clamped - x0) / (x1 - x0)
    return i, w


def trilinear_weights(xs, ys, zs, point):
    """
    Given a point (x,y,z) and 1D grids xs, ys, zs, return:
      indices: 8 indices into the flattened (nx,ny,nz) array
      weights: 8 interpolation weights that sum to 1.
    Flattening is in C order on (nx,ny,nz).
    """
    x, y, z = point
    nx, ny, nz = len(xs), len(ys), len(zs)

    ix, wx = _find_cell_indices(x, xs)
    iy, wy = _find_cell_indices(y, ys)
    iz, wz = _find_cell_indices(z, zs)

    # corners
    idx = []
    w = []

    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                i = ix + dx
                j = iy + dy
                k = iz + dz
                # clamp to valid range
                i = max(0, min(i, nx-1))
                j = max(0, min(j, ny-1))
                k = max(0, min(k, nz-1))
                # weight contributions
                wxc = wx if dx == 1 else (1.0 - wx)
                wyc = wy if dy == 1 else (1.0 - wy)
                wzc = wz if dz == 1 else (1.0 - wz)
                ww = wxc * wyc * wzc
                flat_idx = i * (ny * nz) + j * nz + k
                idx.append(flat_idx)
                w.append(ww)

    # normalize small numerical error
    w = np.array(w, dtype=float)
    s = w.sum()
    if s != 0.0:
        w /= s
    return np.array(idx, dtype=np.int64), w

###############################################################################
# Field-aligned FCI operator (parallel via FCI, perpendicular via 7-point Laplacian)
################################################################################
def make_fci_operator_jax(
    nx: int,
    ny: int,
    nz: int,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    inside: np.ndarray,
    fci: FCIConnectivity,
    core_mask: np.ndarray,
    kappa_par: float,
    kappa_perp: float,
):
    idx_plus  = jnp.asarray(fci.idx_plus)
    idx_minus = jnp.asarray(fci.idx_minus)
    w_plus    = jnp.asarray(fci.w_plus)
    w_minus   = jnp.asarray(fci.w_minus)
    L_plus    = jnp.asarray(fci.L_plus)
    L_minus   = jnp.asarray(fci.L_minus)

    inside_flat = jnp.asarray(inside)
    core_flat_j = jnp.asarray(core_mask)
    valid_par = jnp.asarray(fci.valid) & inside_flat & core_flat_j

    # Debug coverage (move to numpy to avoid jax inside print)
    n_valid = int(np.array(jnp.sum(valid_par)))
    n_inside = int(np.array(jnp.sum(inside_flat)))
    print(f"[DEBUG] FCI coverage: valid={n_valid} / inside={n_inside} "
          f"({100.0 * n_valid / max(1, n_inside):.1f} %)")

    inside3 = inside_flat.reshape(nx, ny, nz)
    
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]

    @jax.jit
    def A_pde_jax(u_flat: jnp.ndarray) -> jnp.ndarray:
        u = u_flat
        u_p = jnp.sum(w_plus * u[idx_plus], axis=1)
        u_m = jnp.sum(w_minus * u[idx_minus], axis=1)
        u0  = u

        Lp_safe = jnp.where(L_plus > 0.0, L_plus, 1e-8)
        Lm_safe = jnp.where(L_minus > 0.0, L_minus, 1e-8)
        Ltot    = Lp_safe + Lm_safe

        dpar = 2.0 * ((u_p - u0) / Lp_safe + (u_m - u0) / Lm_safe) / Ltot
        dpar = jnp.where(valid_par, dpar, 0.0)

        out = -kappa_par * dpar

        # 7-point Laplacian (Cartesian, perpendicular)
        u3 = u_flat.reshape((nx, ny, nz))
        lap3 = jnp.zeros_like(u3)
        lap3 = lap3.at[1:-1, 1:-1, 1:-1].set(
            (u3[2:, 1:-1, 1:-1] - 2*u3[1:-1, 1:-1, 1:-1] + u3[:-2, 1:-1, 1:-1]) / dx**2 +
            (u3[1:-1, 2:, 1:-1] - 2*u3[1:-1, 1:-1, 1:-1] + u3[1:-1, :-2, 1:-1]) / dy**2 +
            (u3[1:-1, 1:-1, 2:] - 2*u3[1:-1, 1:-1, 1:-1] + u3[1:-1, 1:-1, :-2]) / dz**2
        )
        lap3 = jnp.where(inside3, lap3, 0.0)
        out -= kappa_perp * lap3.ravel(order="C")
        
        return out

    deep_inside_mask = np.asarray(inside3.ravel(order="C"))
    return A_pde_jax, deep_inside_mask

def make_fci_operator_cylindrical(
    nR: int,
    nphi: int,
    nZ: int,
    Rs: np.ndarray,
    phis: np.ndarray,
    Zs: np.ndarray,
    inside: np.ndarray,
    fci: FCIConnectivity,
    core_mask: np.ndarray,
    kappa_par: float,
    kappa_perp: float,
):
    """
    FCI operator on a cylindrical grid (R, φ, Z):

        -kappa_par ∂_s^2 ψ  -  kappa_perp ∇^2ψ = 0

    where ∂_s^2 is the field-aligned second derivative represented via FCI
    connectivity, and ∇^2 is the *physical* isotropic Laplacian in
    cylindrical coordinates:

        ∇²ψ = (1/R)∂_R(R ∂_R ψ) + (1/R²)∂²_φ ψ + ∂²_Z ψ.

    The FCI part only acts in the "core" region; the perpendicular Laplacian
    acts in the whole inside region, masked by `inside`.
    """
    idx_plus  = jnp.asarray(fci.idx_plus)
    idx_minus = jnp.asarray(fci.idx_minus)
    w_plus    = jnp.asarray(fci.w_plus)
    w_minus   = jnp.asarray(fci.w_minus)
    L_plus    = jnp.asarray(fci.L_plus)
    L_minus   = jnp.asarray(fci.L_minus)

    inside_flat = jnp.asarray(inside)
    core_flat_j = jnp.asarray(core_mask)

    valid_par = jnp.asarray(fci.valid) & inside_flat & core_flat_j

    inside3 = inside_flat.reshape(nR, nphi, nZ)

    # 1D grids and spacings
    Rs_j   = jnp.asarray(Rs)
    phis_j = jnp.asarray(phis)
    Zs_j   = jnp.asarray(Zs)

    dR   = float(Rs[1]   - Rs[0])   if nR   > 1 else 1.0
    dphi = float(phis[1] - phis[0]) if nphi > 1 else 1.0
    dZ   = float(Zs[1]   - Zs[0])   if nZ   > 1 else 1.0

    # R array to use inside interior region (broadcast)
    R3 = jnp.broadcast_to(Rs_j[:, None, None], (nR, nphi, nZ))

    @jax.jit
    def A_pde_jax(u_flat: jnp.ndarray) -> jnp.ndarray:
        u = u_flat

        # ---------------- Field-aligned second derivative --------------
        u_p = jnp.sum(w_plus * u[idx_plus], axis=1)
        u_m = jnp.sum(w_minus * u[idx_minus], axis=1)
        u0  = u

        Lp_safe = jnp.where(L_plus  > 0.0, L_plus,  1e-8)
        Lm_safe = jnp.where(L_minus > 0.0, L_minus, 1e-8)
        Ltot    = Lp_safe + Lm_safe

        dpar = 2.0 * ((u_p - u0) / Lp_safe + (u_m - u0) / Lm_safe) / Ltot
        dpar = jnp.where(valid_par, dpar, 0.0)

        out = -kappa_par * dpar

        # ---------------- Cylindrical Laplacian ∇²ψ (PERIODIC in φ) -------------------
        u3 = u_flat.reshape((nR, nphi, nZ))
        lap3 = jnp.zeros_like(u3)

        # R-term: interior in R,Z, all φ
        uR_plus  = u3[2:,   :, 1:-1]
        uR_0     = u3[1:-1, :, 1:-1]
        uR_minus = u3[:-2,  :, 1:-1]
        R_mid    = R3[1:-1, :, 1:-1]

        d2u_dR2 = (uR_plus - 2.0 * uR_0 + uR_minus) / (dR * dR)
        du_dR   = (uR_plus - uR_minus) / (2.0 * dR)
        lap_R   = d2u_dR2 + du_dR / jnp.maximum(R_mid, 1e-8)

        # φ-term: periodic on full φ range
        u_phi = u3[1:-1, :, 1:-1]                      # (nR-2, nphi, nZ-2)
        u_phi_plus  = jnp.roll(u_phi, -1, axis=1)
        u_phi_minus = jnp.roll(u_phi, +1, axis=1)
        d2u_dphi2   = (u_phi_plus - 2.0 * u_phi + u_phi_minus) / (dphi * dphi)

        R_mid_phi = R3[1:-1, :, 1:-1]
        lap_phi   = d2u_dphi2 / jnp.maximum(R_mid_phi * R_mid_phi, 1e-12)

        # Z-term: interior in R,Z, all φ
        uZ_plus  = u3[1:-1, :, 2:]
        uZ_0     = u3[1:-1, :, 1:-1]
        uZ_minus = u3[1:-1, :, :-2]
        d2u_dZ2  = (uZ_plus - 2.0 * uZ_0 + uZ_minus) / (dZ * dZ)

        total_lap = lap_R + lap_phi + d2u_dZ2

        # write back on i=1..nR-2, all j, k=1..nZ-2
        lap3 = lap3.at[1:-1, :, 1:-1].set(total_lap)

        # zero outside inside-mask
        lap3 = jnp.where(inside3, lap3, 0.0)

        out -= kappa_perp * lap3.ravel(order="C")
        return out

    deep_inside_mask = np.asarray(inside3.ravel(order="C"))
    return A_pde_jax, deep_inside_mask

def make_linear_operator_jax(
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
    JAX version of A_pde[u] = -div( D ∇u )

    Returns:
      A_pde_jax(u_flat)  -> u_flat_out   (both jnp.ndarray)
      deep_inside_mask   -> numpy bool array (as before)
    """
    inside_j = jnp.asarray(inside.reshape(nx, ny, nz))
    D3 = jnp.asarray(Dfield.reshape(nx, ny, nz, 3, 3))

    domain_mask = inside_j

    Dx = dx; Dy = dy; Dz = dz

    # precompute face tensors & masks in JAX
    D_x = 0.5 * (D3[1:, 1:-1, 1:-1, :, :] + D3[:-1, 1:-1, 1:-1, :, :])
    mask_x = domain_mask[1:, 1:-1, 1:-1] & domain_mask[:-1, 1:-1, 1:-1]

    D_y = 0.5 * (D3[1:-1, 1:, 1:-1, :, :] + D3[1:-1, :-1, 1:-1, :, :])
    mask_y = domain_mask[1:-1, 1:, 1:-1] & domain_mask[1:-1, :-1, 1:-1]

    D_z = 0.5 * (D3[1:-1, 1:-1, 1:, :, :] + D3[1:-1, 1:-1, :-1, :, :])
    mask_z = domain_mask[1:-1, 1:-1, 1:] & domain_mask[1:-1, 1:-1, :-1]

    @jit
    def A_pde_jax(u_flat: jnp.ndarray) -> jnp.ndarray:
        u3 = u_flat.reshape((nx, ny, nz))
        out3 = jnp.zeros_like(u3)

        # ---------------- x-faces ----------------
        dpsi_dx_xp = (u3[1:, 1:-1, 1:-1] - u3[:-1, 1:-1, 1:-1]) / Dx

        dpsi_dy_xp = (
            (u3[1:, 2:, 1:-1] - u3[1:, :-2, 1:-1]) +
            (u3[:-1, 2:, 1:-1] - u3[:-1, :-2, 1:-1])
        ) * (0.25 / Dy)

        dpsi_dz_xp = (
            (u3[1:, 1:-1, 2:] - u3[1:, 1:-1, :-2]) +
            (u3[:-1, 1:-1, 2:] - u3[:-1, 1:-1, :-2])
        ) * (0.25 / Dz)

        valid_dy_x = (
            domain_mask[1:, 2:, 1:-1] & domain_mask[1:, :-2, 1:-1] &
            domain_mask[:-1, 2:, 1:-1] & domain_mask[:-1, :-2, 1:-1]
        )
        valid_dz_x = (
            domain_mask[1:, 1:-1, 2:] & domain_mask[1:, 1:-1, :-2] &
            domain_mask[:-1, 1:-1, 2:] & domain_mask[:-1, 1:-1, :-2]
        )

        dpsi_dy_xp = jnp.where(valid_dy_x & mask_x, dpsi_dy_xp, 0.0)
        dpsi_dz_xp = jnp.where(valid_dz_x & mask_x, dpsi_dz_xp, 0.0)

        qx_xp = (
            D_x[..., 0, 0] * dpsi_dx_xp +
            D_x[..., 0, 1] * dpsi_dy_xp +
            D_x[..., 0, 2] * dpsi_dz_xp
        )
        qx_xp = jnp.where(mask_x, qx_xp, 0.0)

        out3 = out3.at[:-1, 1:-1, 1:-1].add(-qx_xp / Dx)
        out3 = out3.at[1:, 1:-1, 1:-1].add(+qx_xp / Dx)

        # ---------------- y-faces ----------------
        dpsi_dy_yp = (u3[1:-1, 1:, 1:-1] - u3[1:-1, :-1, 1:-1]) / Dy

        dpsi_dx_yp = (
            (u3[2:, 1:, 1:-1] - u3[:-2, 1:, 1:-1]) +
            (u3[2:, :-1, 1:-1] - u3[:-2, :-1, 1:-1])
        ) * (0.25 / Dx)

        dpsi_dz_yp = (
            (u3[1:-1, 1:, 2:] - u3[1:-1, 1:, :-2]) +
            (u3[1:-1, :-1, 2:] - u3[1:-1, :-1, :-2])
        ) * (0.25 / Dz)

        valid_dx_y = (
            domain_mask[2:, 1:, 1:-1] & domain_mask[:-2, 1:, 1:-1] &
            domain_mask[2:, :-1, 1:-1] & domain_mask[:-2, :-1, 1:-1]
        )
        valid_dz_y = (
            domain_mask[1:-1, 1:, 2:] & domain_mask[1:-1, 1:, :-2] &
            domain_mask[1:-1, :-1, 2:] & domain_mask[1:-1, :-1, :-2]
        )

        dpsi_dx_yp = jnp.where(valid_dx_y & mask_y, dpsi_dx_yp, 0.0)
        dpsi_dz_yp = jnp.where(valid_dz_y & mask_y, dpsi_dz_yp, 0.0)

        qy_yp = (
            D_y[..., 1, 0] * dpsi_dx_yp +
            D_y[..., 1, 1] * dpsi_dy_yp +
            D_y[..., 1, 2] * dpsi_dz_yp
        )
        qy_yp = jnp.where(mask_y, qy_yp, 0.0)

        out3 = out3.at[1:-1, :-1, 1:-1].add(-qy_yp / Dy)
        out3 = out3.at[1:-1, 1:, 1:-1].add(+qy_yp / Dy)

        # ---------------- z-faces ----------------
        dpsi_dz_zp = (u3[1:-1, 1:-1, 1:] - u3[1:-1, 1:-1, :-1]) / Dz

        dpsi_dx_zp = (
            (u3[2:, 1:-1, 1:] - u3[:-2, 1:-1, 1:]) +
            (u3[2:, 1:-1, :-1] - u3[:-2, 1:-1, :-1])
        ) * (0.25 / Dx)

        dpsi_dy_zp = (
            (u3[1:-1, 2:, 1:] - u3[1:-1, :-2, 1:]) +
            (u3[1:-1, 2:, :-1] - u3[1:-1, :-2, :-1])
        ) * (0.25 / Dy)

        valid_dx_z = (
            domain_mask[2:, 1:-1, 1:] & domain_mask[:-2, 1:-1, 1:] &
            domain_mask[2:, 1:-1, :-1] & domain_mask[:-2, 1:-1, :-1]
        )
        valid_dy_z = (
            domain_mask[1:-1, 2:, 1:] & domain_mask[1:-1, :-2, 1:] &
            domain_mask[1:-1, 2:, :-1] & domain_mask[1:-1, :-2, :-1]
        )

        dpsi_dx_zp = jnp.where(valid_dx_z & mask_z, dpsi_dx_zp, 0.0)
        dpsi_dy_zp = jnp.where(valid_dy_z & mask_z, dpsi_dy_zp, 0.0)

        qz_zp = (
            D_z[..., 2, 0] * dpsi_dx_zp +
            D_z[..., 2, 1] * dpsi_dy_zp +
            D_z[..., 2, 2] * dpsi_dz_zp
        )
        qz_zp = jnp.where(mask_z, qz_zp, 0.0)

        out3 = out3.at[1:-1, 1:-1, :-1].add(-qz_zp / Dz)
        out3 = out3.at[1:-1, 1:-1, 1:].add(+qz_zp / Dz)

        # zero outside domain
        out3 = jnp.where(domain_mask, out3, 0.0)

        return out3.ravel(order="C")

    deep_inside_mask = np.asarray(domain_mask.ravel(order="C"))
    return A_pde_jax, deep_inside_mask

def cg_jax(matvec, b, x0=None, tol=1e-8, maxiter=1000):
    b = jnp.asarray(b)
    if x0 is None:
        x = jnp.zeros_like(b)
    else:
        x = jnp.asarray(x0)

    r = b - matvec(x)
    p = r
    rs_old = jnp.vdot(r, r)
    rs0 = rs_old  # store initial residual norm^2

    def body_fun(k, state):
        x, r, p, rs_old = state
        Ap = matvec(p)
        alpha = rs_old / jnp.vdot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = jnp.vdot(r_new, r_new)
        beta = rs_new / rs_old
        p_new = r_new + beta * p
        return (x_new, r_new, p_new, rs_new)

    def cond_fun(val):
        k, state = val
        _, r, _, rs_old = state
        # relative residual sqrt(rs_old/rs0)
        rel = jnp.sqrt(rs_old / rs0)
        return jnp.logical_and(
            k < maxiter,
            rel > tol
        )

    def loop_fun(val):
        k, state = val
        new_state = body_fun(k, state)
        return (k + 1, new_state)

    init_state = (x, r, p, rs_old)
    k0 = jnp.array(0, dtype=jnp.int32)
    k_final, (x_final, r_final, _, rs_final) = jax.lax.while_loop(
        cond_fun, loop_fun, (k0, init_state)
    )
    return x_final, jnp.sqrt(rs_final / rs0), k_final


###############################################################################
# Main solver routine
###############################################################################

def solve_fci(npz_file: str, grid_N: int = 64, N_phi: int = 128, eps: float = 1e-3, band_h: float = 1.5,
              cg_tol: float = 1e-8, cg_maxit: int = 2000,
              verbose: bool = True, plot: bool = False, nfp: int = 2,
              delta: float = 5e-3, save_figures: bool = True,
              use_fci: bool = True, fci_nsteps: int = 16,
              fci_planes_per_field_period = 16, psi_power_for_plot=2) -> Dict[str, Any]:
    
    time_start_solve_fci = time.time()
    
    data = np.load(npz_file, allow_pickle=True)
    center = data["center"]; scale = float(data["scale"])
    Yn = data["Yn"]; alpha = data["alpha"]
    a = data["a"]; a_hat = data["a_hat"]
    P = data["P"]; Nsurf = data["N"]
    kind = str(data["kind"])
    kind_str = kind.strip().lower()
    if verbose:
        pinfo(f"Loaded checkpoint with {P.shape[0]} boundary points and {Yn.shape[0]} multipole sources (kind={kind_str}).")

    evals = Evaluators(center=jnp.asarray(center), scale=scale,
                       Yn=jnp.asarray(Yn), alpha=jnp.asarray(alpha),
                       a=jnp.asarray(a), a_hat=jnp.asarray(a_hat))
    grad_phi = evals.build()

    # ------------------------------------------------------------------
    # Build grid: cylindrical (R, φ, Z) when kind=torus; Cartesian when mirror
    # ------------------------------------------------------------------
    kind_str = kind.strip().lower()

    if kind_str == "torus":
        # --- Cylindrical grid (R, φ, Z) ---
        # Compute cylindrical coordinates of boundary points
        Rb = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
        Zb = P[:, 2]

        # Slight padding around boundary
        R_min, R_max = Rb.min(), Rb.max()
        Z_min, Z_max = Zb.min(), Zb.max()
        R_span = R_max - R_min
        Z_span = Z_max - Z_min
        R_min -= 0.02 * R_span
        R_max += 0.02 * R_span
        Z_min -= 0.02 * Z_span
        Z_max += 0.02 * Z_span

        # Use grid_N for radial and vertical; use grid_phi for toroidal
        nR = int(grid_N)
        nphi = int(N_phi)
        nZ = int(grid_N)

        Rs = np.linspace(R_min, R_max, nR)
        phis = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
        Zs = np.linspace(Z_min, Z_max, nZ)

        dR = Rs[1] - Rs[0] if nR > 1 else 1.0
        dphi = phis[1] - phis[0] if nphi > 1 else 1.0
        dZ = Zs[1] - Zs[0] if nZ > 1 else 1.0

        # Build physical coordinates Xq from cylindrical grid
        RR, PHI, ZZ_cyl = np.meshgrid(Rs, phis, Zs, indexing="ij")  # (nR,nphi,nZ)
        XX = RR * np.cos(PHI)
        YY = RR * np.sin(PHI)
        Xq = np.column_stack([
            XX.ravel(order="C"),
            YY.ravel(order="C"),
            ZZ_cyl.ravel(order="C"),
        ])

        # For later reshaping
        nx, ny, nz = nR, nphi, nZ

        if verbose:
            pinfo(f"[GRID] kind=torus, cylindrical grid: nR={nR}, nphi={nphi}, nZ={nZ}, Ntot={Xq.shape[0]}")
            pinfo(f"[GRID] R∈[{R_min:.3g},{R_max:.3g}], Z∈[{Z_min:.3g},{Z_max:.3g}], φ∈[0,2π)")
            pinfo(f"[GRID] dR≈{dR:.3g}, dφ≈{dphi:.3g}, dZ≈{dZ:.3g}")

        # Characteristic "voxel size" in physical space
        # The toroidal direction has physical length ~ R*dphi; use mid-radius as estimate
        R_mid = 0.5 * (R_min + R_max)
        voxel = min(dR, R_mid * dphi, dZ)

        grid_is_cyl = True
        grid_axes = {"Rs": Rs, "phis": phis, "Zs": Zs, "dR": dR, "dphi": dphi, "dZ": dZ}

    else:
        # --- Cartesian grid (mirror or generic) ---
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
            pinfo(f"[GRID] kind={kind_str}, Cartesian grid: {nx}x{ny}x{nz} = {Xq.shape[0]} nodes.")
            pinfo(f"[GRID] Spacing dx≈{dx:.3g}, dy≈{dy:.3g}, dz≈{dz:.3g}")
            pinfo(f"[GRID] Bounds x=[{xs[0]:.3g}, {xs[-1]:.3g}], y=[{ys[0]:.3g}, {ys[-1]:.3g}], z=[{zs[0]:.3g}, {zs[-1]:.3g}]")
        voxel = min(dx, dy, dz)

        grid_is_cyl = False
        grid_axes = {"xs": xs, "ys": ys, "zs": zs, "dx": dx, "dy": dy, "dz": dz}

    # bounding box valid in both cases
    mins = Xq.min(axis=0)
    maxs = Xq.max(axis=0)

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
        Bphi = jnp.where(jnp.abs(Bphi) < 1e-12, 1e-12, Bphi)
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

    # bbox_diag = float(np.linalg.norm(maxs - mins))
    # h_band_vox = max(1.5 * voxel, float(band_h) * voxel)
    h_band_vox = float(band_h) * voxel

    # --- build axis band from nearest neighbours to axis ---
    inside_idx = np.where(inside_mask)[0]
    if inside_idx.size == 0:
        raise RuntimeError("Inside mask empty when building axis band.")

    # Work only with interior nodes for the KD-tree
    X_inside = Xq[inside_idx]

    # How many grid nodes per axis point to pin?
    n_per_axis_pt = 1  # 1 is usually enough; you can try 2–3 if you want a fatter tube

    nbrs_axis = NearestNeighbors(n_neighbors=n_per_axis_pt, algorithm="kd_tree").fit(X_inside)
    d_axis, idx_near = nbrs_axis.kneighbors(axis_pts)

    chosen = inside_idx[idx_near.ravel()]
    chosen = np.unique(chosen)

    axis_band = np.zeros_like(inside_mask, dtype=bool)
    axis_band[chosen] = True
    axis_band_radius_eff = float(d_axis.max())

    if verbose:
        pinfo(f"Axis band built from nearest neighbours: "
              f"{axis_band.sum()} nodes, effective radius ≈ {axis_band_radius_eff:.3e}")

    if not np.any(axis_band):
        if verbose:
            pinfo("Axis band empty; rebuilding adaptively based on distance to axis.")

        nbrs_axis = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(axis_pts)
        d_axis, _ = nbrs_axis.kneighbors(Xq)
        d_axis = d_axis[:, 0]  # true Euclidean distance to nearest axis point

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
        pinfo("Evaluating ∇φ on grid (inside nodes only) ...")

    time0_grad = time.time()
    # Only evaluate at interior nodes
    X_inside = Xq[inside_mask]
    X_inside_j = jnp.asarray(X_inside, dtype=jnp.float64)
    G_inside = np.asarray(grad_phi(X_inside_j))     # (N_inside, 3)
    # Scatter back into a full-sized array (zero outside)
    G = np.zeros_like(Xq)
    G[inside_mask] = G_inside
    # Clean up any non-finite values inside
    bad_inside = ~np.isfinite(G_inside).all(axis=1)
    if np.any(bad_inside):
        G_inside[bad_inside] = 0.0
        G[inside_mask] = G_inside
    if verbose:
        pinfo(f"Direct JAX evaluation for ∇φ on inside nodes took {(time.time() - time0_grad):.2f} s.")
        gnorm_inside = np.linalg.norm(G_inside, axis=1)
        pstat("|∇φ| (inside)", gnorm_inside)
        n_bad = np.count_nonzero(~np.isfinite(G_inside).all(axis=1))
        print(f"[INFO] grad_phi non-finite points (inside only): {n_bad} / {G_inside.shape[0]}")


    if use_fci:
        if verbose:
            pinfo("Building FCI connectivity (field-line mapping) ...")

        if grid_is_cyl and kind_str == "torus":
            Rs = grid_axes["Rs"]
            phis = grid_axes["phis"]
            Zs = grid_axes["Zs"]

            fci = build_fci_connectivity_cylindrical(
                Rs, phis, Zs,
                inside_mask,
                grad_phi_fn=grad_phi,
                nfp=nfp,
                dphi_per_step=None,
                nsteps=fci_nsteps,
                verbose=verbose,
                chunk_size=None,
                fci_planes_per_field_period=fci_planes_per_field_period,
            )
        else:
            xs = grid_axes["xs"]
            ys = grid_axes["ys"]
            zs = grid_axes["zs"]

            fci = build_fci_connectivity_chunked(
                xs, ys, zs,
                inside_mask,
                grad_phi_fn=grad_phi,
                nfp=nfp,
                dphi_per_step=None,
                nsteps=fci_nsteps,
                verbose=verbose,
                chunk_size=None,
                fci_planes_per_field_period=fci_planes_per_field_period,
            )

        if verbose:
            pinfo(f"FCI connectivity: valid nodes = {fci.valid.sum()} / {inside_mask.sum()}")

        if verbose and fci.valid.sum() == 0:
            pinfo("[WARN] FCI connectivity has zero valid nodes; parallel operator is effectively disabled.")
        
        kappa_par = 1.0
        kappa_perp = eps

        inside3 = inside_mask.reshape(nx, ny, nz)
        band3 = band.reshape(nx, ny, nz)
        axis3 = axis_band.reshape(nx, ny, nz)
        # core = at least 1 cell away from domain boundary AND not in bands
        core3 = np.zeros_like(inside3, dtype=bool)
        # interior in R and Z, but *all* φ are core (since φ is periodic)
        core3[1:-1, :, 1:-1] = True
        core3 &= inside3
        core3 &= ~band3
        core3 &= ~axis3
        core_flat = core3.ravel(order="C")
        if verbose and core_flat.sum() < 10:
            pinfo(f"[WARN] FCI core region has only {core_flat.sum()} nodes; parallel operator is almost inactive.")

        if grid_is_cyl and kind_str == "torus":
            Rs   = grid_axes["Rs"]
            phis = grid_axes["phis"]
            Zs   = grid_axes["Zs"]
            A_pde_jax, deep_inside = make_fci_operator_cylindrical(
                nx, ny, nz,
                Rs, phis, Zs,
                inside_mask,
                fci,
                core_mask=core_flat,
                kappa_par=kappa_par,
                kappa_perp=kappa_perp,
            )
        else:
            xs = grid_axes["xs"]
            ys = grid_axes["ys"]
            zs = grid_axes["zs"]
            A_pde_jax, deep_inside = make_fci_operator_jax(
                nx, ny, nz,
                xs, ys, zs,
                inside_mask,
                fci,
                core_mask=core_flat,
                kappa_par=kappa_par,
                kappa_perp=kappa_perp,
            )

    else:
        if grid_is_cyl and kind_str == "torus":
            raise NotImplementedError(
                "use_fci=False not implemented for cylindrical torus grids; "
                "run with FCI (default) for kind='torus'."
            )
        if verbose:
            pinfo("Building Cartesian anisotropic tensor operator (no FCI, JAX) ...")
        dx = grid_axes["dx"]
        dy = grid_axes["dy"]
        dz = grid_axes["dz"]
        D = diffusion_tensor_jax(G, eps=eps, delta=delta)  # now jax
        A_pde_jax, deep_inside = make_linear_operator_jax(
            nx, ny, nz, dx, dy, dz,
            inside_mask, D
        )

    # Wrap common JAX matvec in a SciPy LinearOperator for CG
    def matvec_np(u_np: np.ndarray) -> np.ndarray:
        u_j = jnp.asarray(u_np)
        out_j = A_pde_jax(u_j)
        return np.asarray(out_j)

    N = nx * ny * nz
    A_pde = LinearOperator((N, N), matvec=matvec_np, rmatvec=matvec_np, dtype=float)

    free = deep_inside & (~fixed)
    if not np.any(free):
        raise RuntimeError("No free deep-interior nodes; bands / geometry too tight.")

    # Build lifting for Dirichlet bands: ψ = ψ_free + ψ_fixed
    psi_fixed_full = np.zeros(Ntot, dtype=float)
    psi_fixed_full[fixed] = val[fixed]

    # F0_full = A_pde @ psi_fixed_full  # A_pde applied to known fixed field
    F0_full = np.asarray(A_pde_jax(jnp.asarray(psi_fixed_full)))
    b_free = -F0_full[free]

    # ## Using scipy CG with LinearOperator wrapper around JAX matvec
    # Nfree = int(free.sum())
    # def matvec_free(u_free: np.ndarray) -> np.ndarray:
    #     u_full = np.zeros(Ntot, dtype=float)
    #     u_full[free] = u_free
    #     Au_full = A_pde @ u_full
    #     return Au_full[free]
    # A_eff = LinearOperator(
    #     (Nfree, Nfree),
    #     matvec=matvec_free,
    #     rmatvec=matvec_free,
    #     dtype=float
    # )
    # if verbose:
    #     pinfo("Solving linear system (CG) ...")
    # # start with zeros
    # psi_free, info = cg(A_eff, b_free, rtol=cg_tol, maxiter=cg_maxit)
    # if verbose:
    #     pinfo(f"CG solve completed with {info} info.")
    # if info != 0 and verbose:
    #     pinfo(f"[WARN] CG returned info={info} (0 means full convergence).")
    ## Using JAX CG directly
    def matvec_free_jax(u_free_j):
        # u_free_j: jnp array
        u_full = jnp.zeros(Ntot)
        u_full = u_full.at[free].set(u_free_j)
        Au_full = A_pde_jax(u_full)
        return Au_full[free]
    if verbose:
        pinfo("Solving linear system (JAX CG) ...")
    b_free_j = jnp.asarray(b_free)
    psi_free_j, res_norm, k_final = cg_jax(matvec_free_jax, b_free_j, tol=cg_tol, maxiter=cg_maxit)
    psi_free = np.asarray(psi_free_j)
    if verbose:
        pinfo(f"JAX CG: k={int(k_final)} iters, final relative residual {res_norm:.3e}")

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
        r_free = r_full[free]
        pstat("Residual on free nodes", r_free)
        # We don't have a nonzero RHS on free nodes anymore; measure absolute residual
        pstat(f"||r||_2 over free nodes", np.linalg.norm(r_full[free]))

    psi3 = psi.reshape(nx, ny, nz)

    # ------------------------------------------------------------------
    # Compute ∇ψ on the core for alignment metric q = |t·∇ψ| / |∇ψ|
    # Use (x,y,z) finite differences on Cartesian grid, and cylindrical
    # (R,φ,Z) derivatives with tensor transform on cylindrical grid.
    # ------------------------------------------------------------------
    G3 = G.reshape(nx, ny, nz, 3)

    if grid_is_cyl and kind_str == "torus":
        # Cylindrical grid: psi3[iR, jphi, kZ] with axes Rs, phis, Zs
        Rs   = grid_axes["Rs"]
        phis = grid_axes["phis"]
        Zs   = grid_axes["Zs"]
        dR   = grid_axes["dR"]
        dphi = grid_axes["dphi"]
        dZ   = grid_axes["dZ"]

        nR, nphi, nZ = psi3.shape

        # Build R, φ 3D arrays for metric transform
        RR, PHI, ZZ = np.meshgrid(Rs, phis, Zs, indexing="ij")  # (nR,nphi,nZ)

        # Central differences in cylindrical coordinates on interior cells
        # R: i = 1..nR-2, φ: j = 1..nphi-2, Z: k = 1..nZ-2
        dpsi_dR = (psi3[2:, 1:-1, 1:-1] - psi3[:-2, 1:-1, 1:-1]) / (2.0 * dR)
        # φ: periodic, central differences, then restrict to j=1..nphi-2
        psi_phi = psi3[1:-1, :, 1:-1]  # (nR-2, nphi, nZ-2)
        psi_phi_plus  = np.roll(psi_phi, -1, axis=1)
        psi_phi_minus = np.roll(psi_phi, +1, axis=1)
        dpsi_dphi_full = (psi_phi_plus - psi_phi_minus) / (2.0 * dphi)
        dpsi_dphi = dpsi_dphi_full[:, 1:-1, :]  # (nR-2, nphi-2, nZ-2)

        dpsi_dZ = (psi3[1:-1, 1:-1, 2:] - psi3[1:-1, 1:-1, :-2]) / (2.0 * dZ)

        # Metric transform from (R,φ,Z) to (x,y,z)
        R_mid   = RR[1:-1, 1:-1, 1:-1]
        PHI_mid = PHI[1:-1, 1:-1, 1:-1]

        cosphi = np.cos(PHI_mid)
        sinphi = np.sin(PHI_mid)
        R_safe = np.maximum(R_mid, 1e-8)

        dpsidx = cosphi * dpsi_dR - (sinphi / R_safe) * dpsi_dphi
        dpsidy = sinphi * dpsi_dR + (cosphi / R_safe) * dpsi_dphi
        dpsidz = dpsi_dZ

        # B on interior (1:-1,1:-1,1:-1)
        # B_core = G3[1:-1, 1:-1, 1:-1, :]

    else:
        # Cartesian finite differences as before
        dx = grid_axes["dx"]
        dy = grid_axes["dy"]
        dz = grid_axes["dz"]

        dpsidx = (psi3[2:, 1:-1, 1:-1] - psi3[:-2, 1:-1, 1:-1]) / (2 * dx)
        dpsidy = (psi3[1:-1, 2:, 1:-1] - psi3[1:-1, :-2, 1:-1]) / (2 * dy)
        dpsidz = (psi3[1:-1, 1:-1, 2:] - psi3[1:-1, 1:-1, :-2]) / (2 * dz)

        # B_core = G3[1:-1, 1:-1, 1:-1, :]

    # gnorm_core = np.linalg.norm(B_core, axis=-1)
    # core_mask = (gnorm_core > 1e-10)

    # t_hat_core = np.zeros_like(B_core)
    # t_hat_core[core_mask] = (
    #     B_core[core_mask].T / gnorm_core[core_mask]
    # ).T

    # par_grad_full = (
    #     t_hat_core[..., 0] * dpsidx +
    #     t_hat_core[..., 1] * dpsidy +
    #     t_hat_core[..., 2] * dpsidz
    # )
    # grad_mag_full = np.sqrt(dpsidx**2 + dpsidy**2 + dpsidz**2)

    # par_grad = par_grad_full[core_mask]
    # grad_mag = grad_mag_full[core_mask]

    # q_metric = np.abs(par_grad) / np.maximum(grad_mag, 1e-14)
    
    # --- Build metric mask on the interior (1:-1,1:-1,1:-1) ---
    inside3 = inside_mask.reshape(nx, ny, nz)
    band3   = band.reshape(nx, ny, nz)
    axis3   = axis_band.reshape(nx, ny, nz)
    
    # FCI-valid mask in 3D
    fci_valid3 = np.zeros_like(inside3, dtype=bool)
    if use_fci:
        fci_valid3 = fci.valid.reshape(nx, ny, nz)
    
    # This matches the "core3" you used for the operator,
    # but restricted to the central stencil region and FCI-valid points:
    core_metric = (
        inside3[1:-1, 1:-1, 1:-1]
        & (~band3[1:-1, 1:-1, 1:-1])
        & (~axis3[1:-1, 1:-1, 1:-1])
        & (fci_valid3[1:-1, 1:-1, 1:-1])
    )

    # Now compute |B| and |∇ψ| on that interior region
    B_core = G3[1:-1, 1:-1, 1:-1, :]  # already used above
    gnorm_core = np.linalg.norm(B_core, axis=-1)
    grad_mag_full = np.sqrt(dpsidx**2 + dpsidy**2 + dpsidz**2)

    # Only keep points where |B| and |∇ψ| are "well defined"
    grad_floor = 1e-3 * np.nanmax(grad_mag_full)  # e.g. 0.1% of max
    metric_mask = (
        core_metric
        & (gnorm_core > 1e-10)
        & (grad_mag_full > grad_floor)
    )

    # Build t_hat only on metric_mask
    t_hat_core = np.zeros_like(B_core)
    t_hat_core[metric_mask] = (
        B_core[metric_mask].T / gnorm_core[metric_mask]
    ).T

    par_grad_full = (
        t_hat_core[..., 0] * dpsidx +
        t_hat_core[..., 1] * dpsidy +
        t_hat_core[..., 2] * dpsidz
    )

    par_grad = par_grad_full[metric_mask]
    grad_mag = grad_mag_full[metric_mask]

    q_metric = np.abs(par_grad) / np.maximum(grad_mag, 1e-14)

    if verbose and q_metric.size > 0:
        p = np.percentile(q_metric, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        print(
            "[ALIGN] q = |t·∇ψ|/|∇ψ| stats: "
            f"min={p[0]:.3e} p1={p[1]:.3e} p5={p[2]:.3e} "
            f"p50={p[4]:.3e} p95={p[6]:.3e} p99={p[7]:.3e} max={p[8]:.3e}"
        )
    
    inside3 = inside_mask.reshape(nx, ny, nz)

    print(f"Solved FCI ψ in {(time.time() - time_start_solve_fci):.2f} s.")

    bad_core = (inside_mask & (~band) & (~axis_band) & (np.abs(psi) < 1e-10))
    print(f"[DEBUG] interior nodes with ψ≈0 (excluding bands): {bad_core.sum()}")
    
    if q_metric.size > 0:
        n = q_metric.size
        for thresh in [0.05, 0.1, 0.5, 0.9]:
            frac = np.count_nonzero(q_metric < thresh) / n
            print(f"[ALIGN] frac(q < {thresh}) = {frac:.3f}")
        frac_bad = np.count_nonzero(q_metric > 0.8) / n
        print(f"[ALIGN] frac(q > 0.8) = {frac_bad:.3f}")

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
            if grid_is_cyl and kind_str == "torus":
                # ψ is already on (R,φ,Z) grid
                Rs_cyl   = grid_axes["Rs"]
                phis_cyl = grid_axes["phis"]
                Zs_cyl   = grid_axes["Zs"]
                psi_RZphi = psi3  # shape (nR, nphi, nZ)

                Rb    = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
                phi_b = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2*np.pi)
                Zb    = P[:, 2]

                R_axis   = np.sqrt(axis_pts[:, 0]**2 + axis_pts[:, 1]**2)
                phi_axis = np.mod(np.arctan2(axis_pts[:, 1], axis_pts[:, 0]), 2*np.pi)
                Z_axis   = axis_pts[:, 2]

                jj_list = np.linspace(0, int((len(phis_cyl) - 1)/nfp), 4, dtype=int, endpoint=False)
                power = psi_power_for_plot
                # Mask out points outside the torus by setting them to NaN
                psi_RZphi_plot = np.where(inside3, psi3, np.nan)

                figRZ = plot_psi_maps_RZ_panels(
                    np.power(psi_RZphi_plot, power), Rs_cyl, phis_cyl, Zs_cyl, jj_list,
                    Rb=Rb, Zb=Zb, phi_b=phi_b,
                    R_axis=R_axis, Z_axis=Z_axis, phi_axis=phi_axis,
                    title=rf"$\psi(R,Z)^{power}$"
                )
            else:
                # old Cartesian path
                xs = grid_axes["xs"]
                ys = grid_axes["ys"]
                zs = grid_axes["zs"]
                time0 = time.time()
                psi_RZphi, Rs_cyl, phis_cyl, Zs_cyl, mask_RZphi = build_psi_RZphi_volume(
                    psi3, xs, ys, zs, P, inside3, nR=128, nphi=256, nZ=128)
                pinfo(f"Built ψ(R,Z,φ) volume in {(time.time() - time0):.2f} s.")

                Rb    = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
                phi_b = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2*np.pi)
                Zb    = P[:, 2]

                R_axis   = np.sqrt(axis_pts[:, 0]**2 + axis_pts[:, 1]**2)
                phi_axis = np.mod(np.arctan2(axis_pts[:, 1], axis_pts[:, 0]), 2*np.pi)
                Z_axis   = axis_pts[:, 2]

                jj_list = np.linspace(0, int((len(phis_cyl) - 1)/nfp), 4, dtype=int, endpoint=False)
                power = psi_power_for_plot
                figRZ = plot_psi_maps_RZ_panels(
                    jnp.pow(psi_RZphi, power), Rs_cyl, phis_cyl, Zs_cyl, jj_list,
                    Rb=Rb, Zb=Zb, phi_b=phi_b,
                    R_axis=R_axis, Z_axis=Z_axis, phi_axis=phi_axis,
                    title=rf"$\psi(R,Z)^{power}$"
                )

            if save_figures:
                figRZ.savefig("fci_psi_RZ_panels.png")

        except Exception as e:
            pinfo(f"[WARN] Failed to build RZφ panels: {e}")

        try:
            band3 = band.reshape(nx, ny, nz)
            P_bnd_grid = np.column_stack(np.nonzero(band3))  # (i,j,k) indices

            if grid_is_cyl and kind_str == "torus":
                pinfo("[WARN] 3D ψ-coloured boundary plot not implemented in cylindrical mode; skipping.")
            else:
                xs = grid_axes["xs"]
                ys = grid_axes["ys"]
                zs = grid_axes["zs"]
                X_bnd = np.column_stack([xs[P_bnd_grid[:,0]],
                                         ys[P_bnd_grid[:,1]],
                                         zs[P_bnd_grid[:,2]]])
                fig3d = plot_3d_axis_boundary_interp(X_bnd, axis_pts, psi3, xs, ys, zs)
                if save_figures:
                    fig3d.savefig("fci_psi_3d_axis_boundary.png")
        except Exception as e:
            pinfo(f"[WARN] Failed to plot 3D axis/boundary: {e}")

        try:
            nbrs_axis = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(axis_pts)
            d_axis, _ = nbrs_axis.kneighbors(Xq)
            dist_to_axis = d_axis[:, 0]   # true Euclidean distance to magnetic axis
            # For diagnostics we want to see the axis band as well,
            # so we exclude only the boundary band (ψ=1), not the axis band (ψ=0).
            mask_diag = inside_mask & (~band)
            plt.figure()
            plt.scatter(dist_to_axis[mask_diag], psi[mask_diag], s=2, alpha=0.3)
            plt.xlabel("distance to axis")
            plt.ylabel("ψ")
            plt.title("ψ vs distance to axis (inside domain, excluding boundary band)")
            if save_figures:
                plt.savefig("fci_psi_vs_distance_to_axis.png")
        except Exception as e:
            pinfo(f"[WARN] Failed to plot ψ vs distance to axis: {e}")

        plt.show()

    if grid_is_cyl and kind_str == "torus":
        grid_info = {
            'grid_type': 'cylindrical',
            'Rs': grid_axes['Rs'],
            'phis': grid_axes['phis'],
            'Zs': grid_axes['Zs'],
            'mins': mins,
            'maxs': maxs,
        }
    else:
        grid_info = {
            'grid_type': 'cartesian',
            'xs': grid_axes['xs'],
            'ys': grid_axes['ys'],
            'zs': grid_axes['zs'],
            'mins': mins,
            'maxs': maxs,
        }

    result = {
        'psi': psi,
        'grid': grid_info,
        'inside': inside_mask,
        'bands': {'boundary': band,'axis': axis_band,},
        'quality': {'q_metric': q_metric,'parallel_dot_grad': par_grad,'residual': r_full,},
        'axis': {'R': R_axis,'Z': Z_axis,'points': axis_pts,},
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
    # default_solution = "knot_tube_solution.npz"

    nfp_default = 2
    if 'QH' in default_solution:
        nfp_default = 4

    parser = argparse.ArgumentParser(description="Solve field–aligned flux function ψ via FCI diffusion.")
    parser.add_argument("npz", nargs="?", default=resolve_npz_file_location(default_solution),
                        help="MFS solution checkpoint (*.npz) containing center, scale, Yn, alpha, a, a_hat, P, N")
    parser.add_argument("--N", type=int, default=64, help="Grid resolution per axis")
    parser.add_argument("--N_phi", type=int, default=128, help="Grid resolution in φ (only for cylindrical grids)")
    parser.add_argument("--eps", type=float, default=1e-5, help="Perpendicular diffusion weight")
    parser.add_argument("--delta", type=float, default=0, help="Isotropic diffusion floor")
    parser.add_argument("--band-h", type=float, default=2.0, help="Boundary band thickness multiplier")
    parser.add_argument("--cg-tol", type=float, default=1e-12, help="CG tolerance (default: 1e-8)")
    parser.add_argument("--cg-maxit", type=int, default=4000, help="CG maximum iterations (default: 2000)")
    parser.add_argument("--nfp", type=int, default=nfp_default, help="Number of field periods (default: 2)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--no-save-figures", action="store_true", help="Do NOT save diagnostic figures to disk.")
    parser.add_argument("--no-fci", action="store_true", help="Disable FCI and use tensor Laplacian only.")
    parser.add_argument("--fci-nsteps", type=int, default=26, help="Number of RK2 steps per FCI Δφ trace.")
    parser.add_argument("--fci-planes-per-field-period", type=int, default=4, help="Number of FCI planes per field period")
    parser.add_argument("--psi-power-for-plot", type=int, default=1, help="Power of psi when plotting RZ panels")
    args = parser.parse_args()
    
    res = solve_fci(
        args.npz, grid_N=args.N, N_phi=args.N_phi, eps=args.eps, band_h=args.band_h,
        cg_tol=args.cg_tol, cg_maxit=args.cg_maxit, verbose=True, plot=(not args.no_plot),
        nfp=args.nfp, delta=args.delta, save_figures=not args.no_save_figures,
        use_fci=not args.no_fci, fci_nsteps=args.fci_nsteps,
        fci_planes_per_field_period=args.fci_planes_per_field_period,
        psi_power_for_plot=args.psi_power_for_plot
    )

    psi_all = res['psi']
    inside_mask = res['inside']
    fixed_nodes = (np.abs(psi_all - 1.0) < 1e-10) | (np.abs(psi_all) < 1e-10)
    free_inside = inside_mask & (~fixed_nodes)
    psi_in = psi_all[free_inside]
    if psi_in.size > 0:
        pstat("Solution ψ (inside free region)", psi_in)

    # ------------------------------------------------------------------
    # Save ψ solution + metadata for post-processing / analysis
    # ------------------------------------------------------------------
    grid = res["grid"]
    bands = res["bands"]
    axis_info = res["axis"]
    quality = res["quality"]

    base = Path(args.npz).with_suffix("").name
    if grid["grid_type"] == "cylindrical":
        out_name = f"{base}_psi_fci_cyl_N{args.N}_Nphi{args.N_phi}.npz"
        np.savez(
            out_name,
            psi=res["psi"],
            inside=res["inside"],
            boundary_band=bands["boundary"],
            axis_band=bands["axis"],
            grid_type="cylindrical",
            Rs=grid["Rs"],
            phis=grid["phis"],
            Zs=grid["Zs"],
            mins=grid["mins"],
            maxs=grid["maxs"],
            mfs_npz=os.path.basename(args.npz),
            R_axis=axis_info["R"],
            Z_axis=axis_info["Z"],
            axis_points=axis_info["points"],
            q_metric=quality["q_metric"],
            parallel_dot_grad=quality["parallel_dot_grad"],
            residual=quality["residual"],
        )
    else:
        out_name = f"{base}_psi_fci_cart_N{args.N}.npz"
        np.savez(
            out_name,
            psi=res["psi"],
            inside=res["inside"],
            boundary_band=bands["boundary"],
            axis_band=bands["axis"],
            grid_type="cartesian",
            xs=grid["xs"],
            ys=grid["ys"],
            zs=grid["zs"],
            mins=grid["mins"],
            maxs=grid["maxs"],
            mfs_npz=os.path.basename(args.npz),
            R_axis=axis_info["R"],
            Z_axis=axis_info["Z"],
            axis_points=axis_info["points"],
            q_metric=quality["q_metric"],
            parallel_dot_grad=quality["parallel_dot_grad"],
            residual=quality["residual"],
        )

    pinfo(f"[SAVE] Saved ψ-solution snapshot to: {out_name}")
