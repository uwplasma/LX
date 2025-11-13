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

Compared to previous scripts, this version makes several improvements:

  * **Simplified geometry** – the solver works on a single uniform Cartesian
    grid, independent of whether the underlying surface is toroidal or
    mirror‑symmetric.  All cylindrical–specific routines have been removed.

  * **JAX acceleration** – evaluation of the magnetic potential φ and its
    gradient is vectorised and JIT compiled with JAX.  This yields orders of
    magnitude speed‑ups when evaluating thousands of points.

  * **Diffrax integration** – magnetic axis detection uses the Poincaré map
    integration implemented with Diffrax, a modern ODE solver library for
    JAX.  This ensures robust and differentiable integration of field
    lines.

  * **Matrix–free solve** – the diffusion equation is discretised using
    finite differences on a regular grid.  We assemble a sparse linear
    operator implicitly and solve using conjugate gradients with an
    algebraic multigrid (AMG) preconditioner.  This avoids forming a huge
    dense matrix and allows the solver to scale to large grids.

  * **Rich diagnostics** – the code prints informative statistics at each
    stage of the computation.  Residual norms, axis coordinates, inside
    masks and anisotropy metrics are reported.  Developers can use these
    outputs to verify correctness and monitor convergence.

The overall workflow is:

1. **Load MFS checkpoint** from an ``npz`` file which provides the
   multipole coefficients for φ, the boundary point cloud ``P`` and
   associated normals ``N``.  See ``Evaluators`` below.

2. **Define a regular Cartesian grid** surrounding the surface.  Compute
   the inside mask of grid points using a nearest–neighbor signed distance
   test with the surface normals.

3. **Determine the toroidal sign** by comparing short field line
   integrations in both directions and flip the multivalued gradient
   accordingly.  Then find the magnetic axis in cylindrical coordinates
   using a two–by–two Newton iteration over a Poincaré map implemented with
   Diffrax.  A full field line orbit is also integrated for visualisation
   and to build the axis band mask.

4. **Construct boundary and axis bands** – thin ribbons of grid cells near
   the physical boundary and the magnetic axis on which Dirichlet
   conditions are imposed.  The user can control the thickness of these
   bands through command line parameters.

5. **Evaluate ∇φ on the grid** with JAX and build the anisotropic
   diffusion tensor D.  Then assemble a matrix–free linear operator for
   the anisotropic diffusion equation with Dirichlet values stamped on
   band nodes.  Use an AMG preconditioner to accelerate conjugate
   gradients to solve for ψ.

6. **Compute diagnostics** – the residual of the solution, percentiles of ψ
   inside the domain, and a quality metric |t·∇ψ|/|∇ψ| which should be
   small away from the bands.  Plotting functions are included but can be
   disabled via command line options.

This script is intentionally verbose.  Print statements highlight the
progress through each stage and display intermediate statistics.  It is
designed both as a working solver and a teaching example of the FCI
methodology.

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
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg, minres
from pyamg import smoothed_aggregation_solver
import diffrax as dfx
import matplotlib.pyplot as plt
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
    v=np.asarray(v); print(f"[STAT] {msg}: min={v.min():.3e} med={np.median(v):.3e} max={v.max():.3e} L2={np.linalg.norm(v):.3e}")

# ----------------------------- Geometry utilities ---------------------------- #
def axis_band_mask(P_axis, Xq, rad):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P_axis)
    d, _ = nbrs.kneighbors(Xq)
    return (d[:,0] < rad)

# ----------------------------- Plotting utilities ---------------------------- #
def plot_psi_maps_RZ_panels(psi3, Rs, phis, Zs, jj_list, title="ψ(R,Z)"):
    fig, axa = plt.subplots(2, 2, figsize=(7, 7), constrained_layout=True)
    axa = axa.ravel()
    Rmin, Rmax = float(np.min(Rs)), float(np.max(Rs))
    Zmin, Zmax = float(np.min(Zs)), float(np.max(Zs))
    extent = [Rmin, Rmax, Zmin, Zmax]
    for kk, jj in enumerate(jj_list):
        im = axa[kk].imshow(psi3[:, jj, :].T, origin='lower', aspect='equal', extent=extent)
        axa[kk].contour(psi3[:, jj, :].T, levels=10, colors='white', linewidths=0.5, alpha=1.0, extent=extent)
        axa[kk].set_title(f"{title} @ φ≈{phis[jj]:+.2f}")
        axa[kk].set_xlabel("R"); axa[kk].set_ylabel("Z")
        plt.colorbar(im, ax=axa[kk], shrink=0.85)
    return fig

###############################################################################
# JAX helpers for Green's function and gradient
###############################################################################

@jit
def green_G(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Green's function 1/(4π r) used to evaluate φ.

    Parameters
    ----------
    x, y : array of shape (..., 3)
        Evaluation and source points in normalised coordinates.

    Returns
    -------
    out : array of shape (...,)
        Value of the Green's function at x due to a unit charge at y.
    """
    r = jnp.linalg.norm(x - y, axis=-1)
    return 1.0 / (4.0 * jnp.pi * jnp.maximum(1e-30, r))


@jit
def grad_green_x(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Gradient in x of the Green's function."""
    r = x - y
    r2 = jnp.sum(r * r, axis=-1)
    r3 = jnp.maximum(1e-30, r2 * jnp.sqrt(r2))
    return -r / (4.0 * jnp.pi * r3[..., None])


@jit
def grad_azimuth_about_axis(Xn: jnp.ndarray, a_hat: jnp.ndarray) -> jnp.ndarray:
    """Return the gradient of the multivalued toroidal angle term t̂ about an axis.

    Given an axis direction a_hat, compute ∂θ/∂x in Cartesian coordinates for each
    point Xn in normalised coordinates.  This is used to add a toroidal
    component to the multivalued gradient of φ.  See Hariri & Ottaviani.
    """
    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    # Decompose Xn into parallel and perpendicular parts
    r_par = jnp.sum(Xn * a[None, :], axis=1, keepdims=True) * a[None, :]
    r_perp = Xn - r_par
    r2 = jnp.maximum(1e-30, jnp.sum(r_perp * r_perp, axis=1, keepdims=True))
    return jnp.cross(a[None, :], r_perp) / r2


def make_mv_grads(a_vec: jnp.ndarray, a_hat: jnp.ndarray, sc_center: jnp.ndarray, sc_scale: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Build a closure returning the multivalued gradient ∇_MV φ for physical points.

    The potential φ includes a linear combination of two multivalued terms: a_vec[0]
    times the toroidal angle θ about a_hat and a_vec[1] times the poloidal angle.
    In practice we only use the toroidal part; the poloidal term is left as a
    zero function.  The gradient is scaled appropriately by sc_scale.
    """
    a_vec = jnp.asarray(a_vec)
    a_hat = jnp.asarray(a_hat)
    sc_center = jnp.asarray(sc_center)
    sc_scale = float(sc_scale)

    @jit
    def grad_t(Xn: jnp.ndarray) -> jnp.ndarray:
        return grad_azimuth_about_axis(Xn, a_hat)

    @jit
    def grad_p(Xn: jnp.ndarray) -> jnp.ndarray:
        # Poloidal component could be implemented for general geometries;
        # here we return zero to avoid numerical artefacts inside the domain.
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
    """
    Wrapper for the magnetic scalar potential φ and its gradient ∇φ.

    The potential is represented as a multipole expansion with sources at
    positions ``Yn`` with strengths ``alpha``.  A multivalued component
    controlled by ``a`` and ``a_hat`` is added to enforce the desired
    toroidal symmetry of the solution.  Scaling by ``scale`` and shifting
    by ``center`` maps physical coordinates into the unit sphere used
    internally by the multipole solver.
    """
    center: jnp.ndarray
    scale: float
    Yn: jnp.ndarray
    alpha: jnp.ndarray
    a: jnp.ndarray
    a_hat: jnp.ndarray

    def build(self) -> Tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
        sc_c = jnp.asarray(self.center)
        sc_s = float(self.scale)
        Yn_c = jnp.asarray(self.Yn)
        alpha_c = jnp.asarray(self.alpha)
        a_c = jnp.asarray(self.a)
        a_hatc = jnp.asarray(self.a_hat)

        # Build vmap versions of the Green's function and its gradient
        S_single = jit(lambda xn: jnp.dot(vmap(lambda y: green_G(xn, y))(Yn_c), alpha_c))
        dS_single = jit(lambda xn: jnp.sum(vmap(lambda y: grad_green_x(xn, y))(Yn_c) * alpha_c[:, None], axis=0))

        @jit
        def phi_fn(X: jnp.ndarray) -> jnp.ndarray:
            Xn = (X - sc_c) * sc_s
            return vmap(S_single)(Xn)

        # Multivalued gradient closure
        grad_mv = make_mv_grads(a_c, a_hatc, sc_c, sc_s)

        @jit
        def grad_phi_fn(X: jnp.ndarray) -> jnp.ndarray:
            Xn = (X - sc_c) * sc_s
            return grad_mv(X) + sc_s * vmap(dS_single)(Xn)

        return phi_fn, grad_phi_fn


###############################################################################
# Geometry: inside mask and bands
###############################################################################

def inside_mask_from_surface(P_surf: np.ndarray, N_surf: np.ndarray, Xq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify grid points as inside or outside a closed surface using nearest
    neighbours and oriented normals.

    For each query point x, find its nearest surface point p and normal n.
    Compute the signed distance s = (x-p)·n.  Points with s < 0 are inside.

    Parameters
    ----------
    P_surf, N_surf : array of shape (M, 3)
        Boundary point cloud and corresponding outward normals.
    Xq : array of shape (N, 3)
        Coordinates of query points.

    Returns
    -------
    inside : array of shape (N,), bool
        True for points inside the surface.
    idx : array of shape (N,), int
        Index of the nearest surface point for each query point.
    signed_dist : array of shape (N,), float
        Signed distance to the surface (negative inside).
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P_surf)
    d, idx = nbrs.kneighbors(Xq)
    p = P_surf[idx[:, 0], :]
    n = N_surf[idx[:, 0], :]
    signed_dist = np.sum((Xq - p) * n, axis=1)  # >0 outside
    inside = signed_dist < 0.0
    return inside, idx[:, 0], signed_dist


###############################################################################
# Axis finding via Poincaré map with Diffrax
###############################################################################

def collapse_to_axis(grad_phi: Callable[[jnp.ndarray], jnp.ndarray], R0: float, Z0: float,
                     nfp: int, nsteps: int = 1000, max_newton: int = 12, tol: float = 1e-8) -> Tuple[float, float]:
    """
    Solve for the magnetic axis in cylindrical coordinates using a Poincaré map.

    This routine follows the approach described by Anderson and others for
    computing closed field lines in toroidal plasmas.  We treat φ as
    the independent variable and integrate the field line equations

        dR/dφ = R * B_R / B_φ,
        dZ/dφ = R * B_Z / B_φ,

    where B = ∇φ.  The Poincaré map P(R,Z) advances (R,Z) by one toroidal
    period 2π/nfp and returns the field line starting at φ=0 to the
    original plane.  A fixed point of P corresponds to the magnetic axis.

    Parameters
    ----------
    grad_phi : callable
        Function returning ∇φ at arbitrary Cartesian coordinates.
    R0, Z0 : float
        Initial guess for the axis in cylindrical coordinates at φ=0.
    nfp : int
        Number of field periods.
    nsteps : int
        Number of steps in the Diffrax integration per toroidal turn.
    max_newton : int
        Maximum number of Newton iterations.
    tol : float
        Convergence tolerance on the residual of the Poincaré map.

    Returns
    -------
    R_axis, Z_axis : float
        Cylindrical coordinates of the axis at φ=0.
    """
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

    # Diffrax integrator for one toroidal period
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
                              saveat=saveat_t1, max_steps=nsteps * 16,
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
# Diffusion tensor and stencil
###############################################################################

def diffusion_tensor(gradphi: np.ndarray, eps: float, delta: float = 5e-3) -> np.ndarray:
    """
    Construct the anisotropic diffusion tensor D for each grid point.

    D = t̂ t̂^T + eps * (I - t̂ t̂^T) + delta * I

    where t̂ = ∇φ/|∇φ| is the field direction, eps controls the cross–field
    diffusion and delta adds a small isotropic floor to ensure the operator
    is strictly elliptic.  A smooth taper is applied based on |∇φ| to
    transition from anisotropic inside to isotropic near the boundaries.

    Parameters
    ----------
    gradphi : array of shape (N, 3)
        Gradient of φ at each grid point.
    eps : float
        Ratio of perpendicular to parallel diffusivities (ε ≪ 1).
    delta : float
        Isotropic floor added to all components of D.

    Returns
    -------
    D : array of shape (N, 3, 3)
        Diffusion tensor for each grid point.
    """
    Npts = gradphi.shape[0]
    I = np.eye(3)[None, :, :]
    n = np.linalg.norm(gradphi, axis=-1, keepdims=True)
    n0 = 5e-3 * np.nanmax(n) + 1e-30
    s = np.clip(n / n0, 0.0, 1.0)
    t = np.divide(gradphi, np.maximum(n, 1e-30), out=np.zeros_like(gradphi))
    # Build local orthonormal frame (t, b1, b2)
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


def _face_T(D3: np.ndarray, dx: float, dy: float, dz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute harmonic mean transmissibilities on cell faces for anisotropic diffusion.

    D3 has shape (nx, ny, nz, 3, 3).  For axis aligned faces the flux across a
    face depends on the component n·D·n where n is the face normal.  For a
    uniform Cartesian grid with spacing dx, dy, dz in x,y,z directions the
    harmonic mean of neighbouring values gives the appropriate face
    transmissibility.
    """
    Dxx = D3[..., 0, 0]; Dyy = D3[..., 1, 1]; Dzz = D3[..., 2, 2]
    kx = 2.0 * Dxx[1:, :, :] * Dxx[:-1, :, :] / np.maximum(Dxx[1:, :, :] + Dxx[:-1, :, :], 1e-30) / (dx * dx)
    ky = 2.0 * Dyy[:, 1:, :] * Dyy[:, :-1, :] / np.maximum(Dyy[:, 1:, :] + Dyy[:, :-1, :], 1e-30) / (dy * dy)
    kz = 2.0 * Dzz[:, :, 1:] * Dzz[:, :, :-1] / np.maximum(Dzz[:, :, 1:] + Dzz[:, :, :-1], 1e-30) / (dz * dz)
    return kx, ky, kz


def make_linear_operator(nx: int, ny: int, nz: int, dx: float, dy: float, dz: float,
                         inside: np.ndarray, Dfield: np.ndarray, fixed_mask: np.ndarray,
                         fixed_val: np.ndarray) -> LinearOperator:
    """
    Build a matrix–free LinearOperator for the anisotropic diffusion equation.

    The operator applies

        A[u] = -div( D ∇u )  on free nodes

    and stamps Dirichlet rows on nodes where fixed_mask is True, returning
    u - fixed_val.  All vectors passed to matvec and rmatvec are full 1D
    arrays of length N = nx*ny*nz.  The returned vector has the same length.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    dx, dy, dz : float
        Grid spacings.
    inside : array of shape (N,), bool
        Mask of grid points inside the surface.
    Dfield : array of shape (N,3,3)
        Diffusion tensor at each grid point.
    fixed_mask : array of shape (N,), bool
        True on nodes with Dirichlet boundary conditions.
    fixed_val : array of shape (N,), float
        Fixed values for Dirichlet nodes.

    Returns
    -------
    A_full : LinearOperator
        Matrix–free operator such that A_full @ u_full applies the anisotropic
        diffusion operator and stamps Dirichlet conditions.
    """
    inside3 = inside.reshape(nx, ny, nz)
    D3 = Dfield.reshape(nx, ny, nz, 3, 3)
    fixed_mask = fixed_mask.astype(bool)
    free_mask = ~fixed_mask
    rows_fixed = np.where(fixed_mask)[0]

    def matvec(u: np.ndarray) -> np.ndarray:
        # Reshape for convenience
        u3 = u.reshape(nx, ny, nz)
        Tx, Ty, Tz = _face_T(D3, dx, dy, dz)
        # Differences across faces (zero outside domain)
        dux = np.zeros_like(Tx); duy = np.zeros_like(Ty); duz = np.zeros_like(Tz)
        mx = inside3[1:, :, :] & inside3[:-1, :, :]
        my = inside3[:, 1:, :] & inside3[:, :-1, :]
        mz = inside3[:, :, 1:] & inside3[:, :, :-1]
        dux[mx] = (u3[1:, :, :] - u3[:-1, :, :])[mx]
        duy[my] = (u3[:, 1:, :] - u3[:, :-1, :])[my]
        duz[mz] = (u3[:, :, 1:] - u3[:, :, :-1])[mz]
        # Fluxes across faces
        Fx = np.zeros_like(Tx); Fx[mx] = Tx[mx] * dux[mx]
        Fy = np.zeros_like(Ty); Fy[my] = Ty[my] * duy[my]
        Fz = np.zeros_like(Tz); Fz[mz] = Tz[mz] * duz[mz]
        # Divergence on cell centres
        divF = np.zeros((nx, ny, nz))
        # x faces
        divF[1:-1, :, :] += (Fx[1:, :, :] - Fx[:-1, :, :])
        # y faces
        divF[:, 1:-1, :] += (Fy[:, 1:, :] - Fy[:, :-1, :])
        # z faces
        divF[:, :, 1:-1] += (Fz[:, :, 1:] - Fz[:, :, :-1])
        # Zero outside interior
        divF[~inside3] = 0.0
        out = (-divF).ravel(order="C")
        # Stamp Dirichlet rows
        out[rows_fixed] = u[rows_fixed] - fixed_val[rows_fixed]
        return out
    N = nx * ny * nz
    return LinearOperator((N, N), matvec=matvec, rmatvec=matvec, dtype=float)


###############################################################################
# Main solver routine
###############################################################################

def solve_fci(npz_file: str, grid_N: int = 64, eps: float = 1e-3, band_h: float = 1.5,
              axis_band_radius: float = 0.0, cg_tol: float = 1e-8, cg_maxit: int = 2000,
              verbose: bool = True, plot: bool = False, nfp: int = 2,
              delta: float = 5e-3, no_amg: bool = False,
              axis_seed_count: int = 0) -> Dict[str, Any]:
    """
    Solve the field–aligned flux function ψ via anisotropic diffusion on a uniform grid.

    Parameters
    ----------
    npz_file : str
        Path to the multipole solution checkpoint containing ``center``, ``scale``,
        ``Yn``, ``alpha``, ``a``, ``a_hat``, ``P``, ``N``.  See Evaluators.
    grid_N : int
        Number of grid points per axis for the Cartesian grid.  Total nodes N = grid_N³.
    eps : float
        Ratio of perpendicular to parallel diffusivities.
    band_h : float
        Boundary band thickness multiplier.  The band width is band_h * voxel size.
    axis_band_radius : float
        Radius of the axis band.  If 0, choose automatically; if <1, interpret as
        fraction of the bounding box diagonal; if ≥1, use absolute units.
    cg_tol : float
        Relative tolerance for the conjugate–gradient solver.
    cg_maxit : int
        Maximum number of CG iterations.
    verbose : bool
        If True, print diagnostic information.
    plot : bool
        If True, produce matplotlib plots of the solution and residuals (requires
        matplotlib).  Disabled in headless environments by default.
    nfp : int
        Number of field periods (for computing the Poincaré map).
    delta : float
        Isotropic floor added to the diffusion tensor.
    no_amg : bool
        If True, do not attempt to build an AMG preconditioner; use Jacobi instead.
    axis_seed_count : int
        Unused parameter retained for compatibility.  Axis seeds are now
        determined automatically.

    Returns
    -------
    result : dict
        Dictionary containing the solution ψ, the grid coordinates, inside mask,
        and diagnostic information.
    """
    # Load checkpoint data
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

    # Build evaluators for φ and ∇φ
    evals = Evaluators(center=jnp.asarray(center), scale=scale,
                       Yn=jnp.asarray(Yn), alpha=jnp.asarray(alpha),
                       a=jnp.asarray(a), a_hat=jnp.asarray(a_hat))
    phi_fn, grad_phi = evals.build()

    # Build uniform Cartesian grid surrounding the boundary with margin
    mins = P.min(axis=0); maxs = P.max(axis=0); span = maxs - mins
    mins = mins - 0.05 * span; maxs = maxs + 0.05 * span
    nx = ny = nz = int(grid_N)
    xs = np.linspace(mins[0], maxs[0], nx)
    ys = np.linspace(mins[1], maxs[1], ny)
    zs = np.linspace(mins[2], maxs[2], nz)
    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)
    Xq = np.column_stack([XX.ravel(order="C"), YY.ravel(order="C"), ZZ.ravel(order="C")])
    if verbose:
        pinfo(f"Grid size: {nx}x{ny}x{nz} = {Xq.shape[0]} nodes.  Spacing dx≈{dx:.3g}, dy≈{dy:.3g}, dz≈{dz:.3g}")
    voxel = min(dx, dy, dz)

    # Flip normals if necessary so that outward normals have positive mean (P-c)·N
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

    # Compute inside mask via signed distance to surface
    inside_mask, nn_idx, signed_dist = inside_mask_from_surface(P, Nsurf, Xq)
    if verbose:
        pstat("Inside mask", inside_mask.astype(float))

    # Decide toroidal direction via local gradient integration
    if not np.any(inside_mask):
        raise RuntimeError("Inside mask is empty; check surface normals or grid bounds.")
    x0 = Xq[inside_mask][0]
    # Integrate a few steps along ±∇φ to see which direction increases φ
    def decide_sign(x_init: jnp.ndarray) -> Tuple[str, jnp.ndarray]:
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

    # Rebuild evaluators with flipped sign if needed
    a_flipped = np.array([float(dir_sign) * float(a[0]), float(a[1])], dtype=float)
    evals = Evaluators(center=jnp.asarray(center), scale=scale,
                       Yn=jnp.asarray(Yn), alpha=jnp.asarray(alpha),
                       a=jnp.asarray(a_flipped), a_hat=jnp.asarray(a_hat))
    phi_fn, grad_phi = evals.build()

    # Find magnetic axis via Poincaré map
    # Use interior slice near φ=0 and Z≈0 to estimate initial guess
    phi_tol = 0.2
    phiq = np.arctan2(Xq[:, 1], Xq[:, 0])
    Z_span = Xq[:, 2].max() - Xq[:, 2].min()
    Z_tol = 0.25 * Z_span
    mask_slice = inside_mask & (np.abs(phiq) < phi_tol) & (np.abs(Xq[:, 2]) < Z_tol)
    if not np.any(mask_slice):
        mask_slice = inside_mask
    R_slice = np.sqrt(Xq[mask_slice][:, 0] ** 2 + Xq[mask_slice][:, 1] ** 2)
    R_inner = float(R_slice.min()); R_outer = float(R_slice.max())
    R0_guess = 0.5 * (R_inner + R_outer); Z0_guess = 0.0
    if verbose:
        pinfo(f"Initial axis guess R≈{R0_guess:.3e}, Z≈{Z0_guess:.3e}")
    start_axis_time = time.time()
    R_axis, Z_axis = collapse_to_axis(grad_phi, R0_guess, Z0_guess, nfp=nfp)
    if verbose:
        pinfo(f"Solved axis in {(time.time() - start_axis_time):.2f} s: R={R_axis:.3e}, Z={Z_axis:.3e}")
    # Integrate full axis orbit for diagnostics
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

    # Debug: are axis points inside or outside the surface?
    inside_axis, _, signed_axis = inside_mask_from_surface(P, Nsurf, axis_pts)
    if verbose:
        pinfo(
            f"Axis vs surface signed distance: "
            f"min={signed_axis.min():.3e}, max={signed_axis.max():.3e}"
        )
        pinfo(f"Axis points classified inside: {inside_axis.sum()} / {inside_axis.size}")

    # Build boundary band and axis band
    bbox_diag = float(np.linalg.norm(maxs - mins))
    if axis_band_radius == 0.0:
        # Start with a few voxels; will adjust if needed
        axis_band_radius_eff = 2.5 * voxel
    elif axis_band_radius < 1.0:
        axis_band_radius_eff = max(2.5 * voxel, axis_band_radius * bbox_diag)
    else:
        axis_band_radius_eff = max(2.5 * voxel, float(axis_band_radius))

    h_band_vox = max(1.5 * voxel, float(band_h) * voxel)

    # ---------------- Axis band (primary guess) ----------------
    axis_band = axis_band_mask(axis_pts, Xq, axis_band_radius_eff) & inside_mask

    # If the band is empty, adaptively pick the innermost inside nodes
    if not np.any(axis_band):
        if verbose:
            pinfo("Axis band empty; rebuilding adaptively based on distance to axis.")

        # Distances to axis curve for all grid points
        nbrs_axis = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(axis_pts)
        d_axis, _ = nbrs_axis.kneighbors(Xq)
        d_axis = d_axis[:, 0]

        inside_idx = np.where(inside_mask)[0]
        if inside_idx.size == 0:
            raise RuntimeError("Inside mask empty when building axis band.")

        # Choose a small fraction of innermost inside nodes as axis band
        frac = 0.02  # 2% of inside nodes
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

    # ---------------- Boundary band ----------------
    band = (inside_mask & (np.abs(signed_dist) <= h_band_vox))

    # Ensure axis band dominates boundary band where they overlap
    overlap = band & axis_band
    axis_band[overlap] = True
    band[overlap] = False

    # Prepare Dirichlet masks and values
    Ntot = Xq.shape[0]
    fixed = np.zeros(Ntot, dtype=bool)
    val = np.zeros(Ntot, dtype=float)
    fixed[band] = True; val[band] = 1.0
    fixed[axis_band] = True; val[axis_band] = 0.0
    free = inside_mask & (~fixed)

    if verbose:
        print(f"[INFO] #inside nodes        : {inside_mask.sum()} / {Ntot}")
        print(f"[INFO] #boundary band nodes : {band.sum()} / {Ntot}")
        print(f"[INFO] #axis band nodes     : {axis_band.sum()} / {Ntot}")
        print(f"[INFO] boundary band width  ≈ {h_band_vox:.3e}")
        print(f"[INFO] axis band radius     ≈ {axis_band_radius_eff:.3e}")

    if not np.any(free):
        raise RuntimeError("No free interior nodes; bands cover the entire domain.")
    # Evaluate ∇φ on the grid (chunked to save memory)
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
    # Build diffusion tensor
    D = diffusion_tensor(G, eps=eps, delta=delta)
    # Build matrix–free operator
    A_full = make_linear_operator(nx, ny, nz, dx, dy, dz, inside_mask, D, fixed, val)
    # Compute RHS for fixed values
    def compute_rhs_full() -> np.ndarray:
        u_full = np.array(val, copy=True)
        r_full = A_full @ u_full
        return -r_full
    b_full = compute_rhs_full()
    b_free = b_full[free]
    # Construct restricted operator on free nodes
    def matvec_free(u_free: np.ndarray) -> np.ndarray:
        u_full = np.zeros_like(val)
        u_full[free] = u_free
        Au_full = A_full @ u_full
        return Au_full[free]
    Nfree = int(free.sum())
    A_free = LinearOperator((Nfree, Nfree), matvec=matvec_free, rmatvec=matvec_free, dtype=float)
    # Volume scaling for SPD
    cell_vol = dx * dy * dz
    sqrtV = np.sqrt(cell_vol)
    def A_sym_matvec(y: np.ndarray) -> np.ndarray:
        u_free = y / sqrtV
        r_free = A_free @ u_free
        return sqrtV * r_free
    A_sym = LinearOperator((Nfree, Nfree), matvec=A_sym_matvec, rmatvec=A_sym_matvec, dtype=float)
    b_sym = sqrtV * b_free
    # Build AMG preconditioner (optional)
    if not no_amg:
        try:
            if verbose:
                pinfo("Assembling CSR matrix for AMG preconditioner ...")

            def build_csr_free():
                """Assemble explicit CSR matrix A_ff on free nodes for AMG."""
                inside3 = inside_mask.reshape(nx, ny, nz)
                fixed1d = fixed.ravel(order="C")
                D3 = D.reshape(nx, ny, nz, 3, 3)
                Tx, Ty, Tz = _face_T(D3, dx, dy, dz)
                idx3 = np.arange(nx * ny * nz).reshape(nx, ny, nz)

                rows, cols, vals = [], [], []
                diag = np.zeros(nx * ny * nz, dtype=float)

                for i in range(nx):
                    for j in range(ny):
                        for k in range(nz):
                            if not inside3[i, j, k]:
                                continue
                            p = idx3[i, j, k]
                            if fixed1d[p]:
                                continue

                            # x-
                            if i > 0 and inside3[i - 1, j, k]:
                                q = idx3[i - 1, j, k]
                                K = Tx[i - 1, j, k]
                                if fixed1d[q]:
                                    # Dirichlet contribution to RHS (handled in b_free)
                                    pass
                                else:
                                    rows.append(p); cols.append(q); vals.append(-K)
                                diag[p] += K
                            # x+
                            if i + 1 < nx and inside3[i + 1, j, k]:
                                q = idx3[i + 1, j, k]
                                K = Tx[i, j, k]
                                if fixed1d[q]:
                                    pass
                                else:
                                    rows.append(p); cols.append(q); vals.append(-K)
                                diag[p] += K
                            # y-
                            if j > 0 and inside3[i, j - 1, k]:
                                q = idx3[i, j - 1, k]
                                K = Ty[i, j - 1, k]
                                if fixed1d[q]:
                                    pass
                                else:
                                    rows.append(p); cols.append(q); vals.append(-K)
                                diag[p] += K
                            # y+
                            if j + 1 < ny and inside3[i, j + 1, k]:
                                q = idx3[i, j + 1, k]
                                K = Ty[i, j, k]
                                if fixed1d[q]:
                                    pass
                                else:
                                    rows.append(p); cols.append(q); vals.append(-K)
                                diag[p] += K
                            # z-
                            if k > 0 and inside3[i, j, k - 1]:
                                q = idx3[i, j, k - 1]
                                K = Tz[i, j, k - 1]
                                if fixed1d[q]:
                                    pass
                                else:
                                    rows.append(p); cols.append(q); vals.append(-K)
                                diag[p] += K
                            # z+
                            if k + 1 < nz and inside3[i, j, k + 1]:
                                q = idx3[i, j, k + 1]
                                K = Tz[i, j, k]
                                if fixed1d[q]:
                                    pass
                                else:
                                    rows.append(p); cols.append(q); vals.append(-K)
                                diag[p] += K

                # Add diagonal entries for free nodes
                p_idx = np.where(inside_mask & (~fixed))[0]
                rows += list(p_idx)
                cols += list(p_idx)
                vals += list(diag[p_idx])

                A_full_csr = coo_matrix((vals, (rows, cols)),
                                        shape=(nx * ny * nz, nx * ny * nz)).tocsr()
                A_ff = A_full_csr[free][:, free]
                return A_ff

            A_csr_free = build_csr_free()

            # Basic smoothed aggregation AMG with default options
            ml = smoothed_aggregation_solver(A_csr_free, symmetry='symmetric')
            M_amg = ml.aspreconditioner(cycle='V')

            def M_apply(x):
                return M_amg @ x

            M = LinearOperator((Nfree, Nfree), matvec=M_apply,
                               rmatvec=M_apply, dtype=float)

        except Exception as e:
            if verbose:
                pinfo(f"AMG preconditioner failed: {e}; falling back to Jacobi.")
            no_amg = True

    if no_amg:
        # Jacobi-like fallback preconditioner
        diag_approx = np.maximum(1e-8, np.ones(Nfree))

        def M_apply(x):
            return x / diag_approx

        M = LinearOperator((Nfree, Nfree), matvec=M_apply,
                           rmatvec=M_apply, dtype=float)

    # Initial guess for CG: apply preconditioner to RHS
    x0 = M @ b_sym
    if verbose:
        pinfo("Solving linear system (CG/Minres) ...")
    psi_free_scaled, info = cg(A_sym, b_sym, rtol=cg_tol, maxiter=cg_maxit, M=M, x0=x0)
    if (not np.all(np.isfinite(psi_free_scaled))) or info != 0:
        if verbose:
            pinfo(f"CG returned info={info}; switching to MINRES.")
        psi_free_scaled, info = minres(A_sym, b_sym, rtol=cg_tol, maxiter=cg_maxit, M=M, x0=x0)
    # Unscale to obtain ψ on free nodes
    psi_free = psi_free_scaled / sqrtV
    psi = np.array(val)
    psi[free] = psi_free
    # Compute full residual
    r_full = A_full @ psi
    if verbose:
        # Basic ψ statistics everywhere
        pstat("ψ (all nodes)", psi)

        # ψ on boundary and axis bands
        pstat("ψ on boundary band (should be ~1)", psi[band])
        pstat("ψ on axis band (should be ~0)", psi[axis_band])

        # ψ on free interior nodes (inside, not in bands)
        free_inside = inside_mask & (~fixed)
        if np.any(free_inside):
            pstat("ψ on free interior", psi[free_inside])

        pstat("Full residual", r_full)
        print(f"Reduced residual ||r||/||b|| = {np.linalg.norm(r_full[free]) / (np.linalg.norm(b_full[free]) + 1e-30):.3e}")
    # Quality metric: q = |t·∇ψ| / |∇ψ|, with t = ∇φ / |∇φ|
    psi3 = psi.reshape(nx, ny, nz)

    # Central differences for ∇ψ on the interior region
    dpsidx = (psi3[2:, 1:-1, 1:-1] - psi3[:-2, 1:-1, 1:-1]) / (2 * dx)
    dpsidy = (psi3[1:-1, 2:, 1:-1] - psi3[1:-1, :-2, 1:-1]) / (2 * dy)
    dpsidz = (psi3[1:-1, 1:-1, 2:] - psi3[1:-1, 1:-1, :-2]) / (2 * dz)

    # Core region for ∇φ
    G3 = G.reshape(nx, ny, nz, 3)
    B_core = G3[1:-1, 1:-1, 1:-1, :]          # shape (nx-2, ny-2, nz-2, 3)
    gnorm_core = np.linalg.norm(B_core, axis=-1)
    core_mask = (gnorm_core > 1e-10)

    # Normalised field direction t̂
    t_hat_core = np.zeros_like(B_core)
    t_hat_core[core_mask] = (
        B_core[core_mask].T / gnorm_core[core_mask]
    ).T

    # Parallel and total gradient of ψ
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

    # Optional diagnostic plots
    if plot:
        # Histogram of alignment error q_metric
        if q_metric.size > 0:
            fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

            ax1[0].hist(q_metric, bins=80)
            ax1[0].set_yscale("log")
            ax1[0].set_xlabel(r"$q = |t\cdot\nabla\psi|/|\nabla\psi|$")
            ax1[0].set_ylabel("count")
            ax1[0].set_title("Alignment error histogram")

            # Residual distribution on free interior nodes
            res_free = r_full[free]
            ax1[1].hist(np.abs(res_free), bins=80)
            ax1[1].set_yscale("log")
            ax1[1].set_xlabel(r"$|r|$")
            ax1[1].set_ylabel("count")
            ax1[1].set_title("PDE residual |r| on free nodes")

            fig1.suptitle("FCI ψ diagnostics")

        # A simple ψ slice (x–z plane at mid y) for a quick visual check
        psi3 = psi.reshape(nx, ny, nz)
        j_mid = ny // 2
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im = ax2.imshow(
            psi3[:, j_mid, :].T,
            origin="lower",
            extent=[xs[0], xs[-1], zs[0], zs[-1]],
            aspect="equal",
        )
        ax2.set_xlabel("x")
        ax2.set_ylabel("z")
        ax2.set_title(r"$\psi(x,z)$ slice at mid $y$")
        plt.colorbar(im, ax=ax2, shrink=0.8)

        # figP = plot_psi_maps_RZ_panels(psi3_c, Rs, phis, Zs, jj_list, title="ψ(R,Z)")
        # if save_figures:
        #     figP.savefig("solve_flux_maps_RZ_panels.png", dpi=150)

        plt.show()

    # Prepare result dictionary
    result = {
        'psi': psi,
        'grid': {'xs': xs,'ys': ys,'zs': zs,'mins': mins,'maxs': maxs,},
        'inside': inside_mask,
        'bands': {'boundary': band,'axis': axis_band,},
        'quality': {
            # q = |t·∇ψ|/|∇ψ| sampled on core region
            'q_metric': q_metric,
            # parallel component t·∇ψ at sampled points
            'parallel_dot_grad': par_grad,
            # full residual A_full @ psi on all nodes
            'residual': r_full,},
        'axis': {'R': R_axis,'Z': Z_axis,'points': axis_pts,},}
    return result


###############################################################################
# Command line interface
###############################################################################

if __name__ == "__main__":
    default_solution = "wout_precise_QA_solution.npz"
    # default_solution = "wout_precise_QH_solution.npz"
    # default_solution = "wout_SLAM_4_coils_solution.npz"
    # default_solution = "wout_SLAM_6_coils_solution.npz"

    parser = argparse.ArgumentParser(description="Solve field–aligned flux function ψ via FCI diffusion.")
    parser.add_argument("npz", nargs="?", default=resolve_npz_file_location(default_solution),
                help="MFS solution checkpoint (*.npz) containing center, scale, Yn, alpha, a, a_hat, P, N")
    parser.add_argument("--N", type=int, default=128, help="Grid resolution per axis (default: 128)")
    parser.add_argument("--eps", type=float, default=1e-3, help="Perpendicular diffusion weight (default: 1e-3)")
    parser.add_argument("--delta", type=float, default=5e-3, help="Isotropic diffusion floor (default: 5e-3)")
    parser.add_argument("--band-h", type=float, default=1.0, help="Boundary band thickness multiplier (default: 1.5)")
    parser.add_argument("--axis-band-radius", type=float, default=0.0, help="Axis band radius; 0=auto, <1=fraction of bbox, ≥1=absolute")
    parser.add_argument("--cg-tol", type=float, default=1e-8, help="CG tolerance (default: 1e-8)")
    parser.add_argument("--cg-maxit", type=int, default=2000, help="CG maximum iterations (default: 2000)")
    parser.add_argument("--nfp", type=int, default=2, help="Number of field periods (default: 2)")
    parser.add_argument("--no-amg", action="store_true", help="Disable AMG preconditioner and use Jacobi")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting (not implemented in this version)")
    args = parser.parse_args()
    res = solve_fci(args.npz, grid_N=args.N, eps=args.eps, band_h=args.band_h,
                    axis_band_radius=args.axis_band_radius, cg_tol=args.cg_tol,
                    cg_maxit=args.cg_maxit, verbose=True, plot=(not args.no_plot),
                    nfp=args.nfp, delta=args.delta, no_amg=args.no_amg)
    # Print solution statistics on inside nodes (excluding boundary and axis bands)
    psi_all = res['psi']
    inside_mask = res['inside']
    # Exclude axis band and boundary band nodes for this summary by checking where Dirichlet conditions were applied
    # Since the solve stamps ψ=1 on band and ψ=0 on axis_band, we identify fixed nodes by comparing ψ to 0 or 1.
    fixed_nodes = (np.abs(psi_all - 1.0) < 1e-10) | (np.abs(psi_all) < 1e-10)
    free_inside = inside_mask & (~fixed_nodes)
    psi_in = psi_all[free_inside]
    if psi_in.size > 0:
        pstat("Solution ψ (inside free region)", psi_in)