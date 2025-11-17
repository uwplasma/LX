#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Rotational transform and shear diagnostics from MFS vacuum field and ψ(𝐱).

This script:
  1) Loads ψ(𝐱) from a psi_fci snapshot (.npz).
  2) Loads the MFS solution checkpoint (center, scale, Yn, alpha, a, a_hat, P, N).
  3) Rebuilds ∇φ(𝐱) (so 𝐁 = ∇φ) using the same Evaluators pattern as in the MFS code.
  4) Constructs geometric angles (φ, θ) from the magnetic axis.
  5) Selects several ψ-levels (flux surfaces) and seeds field lines on each surface.
  6) For each seed, traces a field line for several toroidal turns, computing
       Δθ and Δφ in geometric coordinates, and estimates

         ι ≈ Δθ / Δφ

     averaged over seeds on that surface.
  7) Builds a radial profile ι(ψ) and computes magnetic shear

         ι'(ψ) = dι/dψ

     by finite differences.
  8) Optionally loads a VMEC wout file to compare ι(ψ) and shear with a
     standard equilibrium code.
  9) Generates publication-ready plots:
       - ι vs normalized flux ψ̂
       - Shear ι' vs ψ̂
       - If VMEC present: overlay of ι and ι' and relative error Δι/ι_VMEC
 10) Saves all numerical data to an .npz file.

Notes
-----
- The ψ(𝐱) snapshot is assumed to be the same format as produced by your
  `solve_flux_psi_fci_cyl.py` (cylindrical or Cartesian grid, with inside mask,
  axis points, mins/maxs, etc.), as used in `qs_diagnostics.py`.

- The rotational transform computed here uses *geometric* poloidal and toroidal
  angles:
      φ = atan2(y, x)
      θ = atan2(z - Z_axis(φ), R - R_axis(φ))
  where (R_axis(φ), Z_axis(φ)) is the magnetic axis in cylindrical coordinates.
  This yields a well-defined "geometric transform" suitable for diagnostics
  and comparison with VMEC.

- The VMEC comparison reads `iotaf` and `s` from the wout NetCDF file,
  and plots ι(s) vs normalized flux ŝ, interpolated onto the ψ̂ grid for
  error plots.

References (for context)
------------------------
- A. H. Boozer, "Guiding center drift equations", Phys. Fluids 23, 904 (1980).
- A. H. Boozer, "Plasma equilibrium with rational magnetic surfaces",
  Phys. Fluids 24, 1999 (1981).
- J. Nuhrenberg and R. Zille, "Stable stellarators with medium β and aspect
  ratio", Phys. Lett. A 114, 129 (1986).
- S. P. Hirshman and J. C. Whitson, "Steepest-descent moment method for
  three-dimensional magnetohydrodynamic equilibria", Phys. Fluids 26, 3553 (1983).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

import os
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Optional VMEC dependency
try:
    from netCDF4 import Dataset  # type: ignore
    _HAS_NETCDF4 = True
except Exception:
    _HAS_NETCDF4 = False

# ============================================================
# Paths / utilities
# ============================================================

script_dir = Path(__file__).resolve().parent

def resolve_npz_file_location(npz_file: str, subdir: str = "outputs") -> str:
    """Try to resolve an npz file into ../subdir if it lives there."""
    try:
        npz_name = os.path.basename(str(npz_file))
        candidate = (script_dir / ".." / subdir / npz_name).resolve()
        if candidate.exists():
            npz_file = str(candidate)
            print(f"[INFO] Resolved checkpoint path -> {npz_file}")
        else:
            print(f"[WARN] Expected checkpoint not found at {candidate}; using provided path: {npz_file}")
    except Exception as e:
        print(f"[WARN] Failed to resolve ../{subdir} path: {e}; using provided path: {npz_file}")
    return npz_file

# ============================================================
# Green's function gradients and MFS Evaluators (copied/trimmed)
# ============================================================

@jit
def grad_green_x(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    r = x - y
    r2 = jnp.sum(r * r, axis=-1)
    r3 = jnp.maximum(1e-30, r2 * jnp.sqrt(r2))
    return -r / (4.0 * jnp.pi * r3[..., None])

@jit
def grad_azimuth_about_axis(Xn: jnp.ndarray, a_hat: jnp.ndarray) -> jnp.ndarray:
    """
    Gradient of the multivalued azimuthal potential about an axis direction a_hat.
    """
    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par = jnp.sum(Xn * a[None, :], axis=1, keepdims=True) * a[None, :]
    r_perp = Xn - r_par
    r2 = jnp.maximum(1e-30, jnp.sum(r_perp * r_perp, axis=1, keepdims=True))
    return jnp.cross(a[None, :], r_perp) / r2

def make_mv_grads(a_vec: jnp.ndarray,
                  a_hat: jnp.ndarray,
                  sc_center: jnp.ndarray,
                  sc_scale: float):
    """
    Build gradient of multivalued piece (e.g. toroidal potential) in world coordinates.
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
        return jnp.zeros_like(Xn)

    def grad_mv_world(X: jnp.ndarray) -> jnp.ndarray:
        Xn = (X - sc_center) * sc_scale
        return sc_scale * (a_vec[0] * grad_t(Xn) + a_vec[1] * grad_p(Xn))

    return grad_mv_world

@dataclass
class Evaluators:
    center: jnp.ndarray
    scale: float
    Yn: jnp.ndarray
    alpha: jnp.ndarray
    a: jnp.ndarray
    a_hat: jnp.ndarray

    def build_grad_phi(self):
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

# ============================================================
# Loading ψ snapshot and MFS solution
# ============================================================

def load_psi_snapshot(psi_npz: str) -> Dict[str, Any]:
    data = np.load(psi_npz, allow_pickle=True)
    grid_type = str(data["grid_type"])
    psi_flat = np.asarray(data["psi"])
    inside = np.asarray(data["inside"], dtype=bool)
    boundary_band = np.asarray(data["boundary_band"], dtype=bool)
    axis_band = np.asarray(data["axis_band"], dtype=bool)
    mins = np.asarray(data["mins"])
    maxs = np.asarray(data["maxs"])

    axis_points = np.asarray(data["axis_points"])
    R_axis = np.asarray(data["R_axis"])
    Z_axis = np.asarray(data["Z_axis"])

    if grid_type == "cylindrical":
        Rs = np.asarray(data["Rs"])
        phis = np.asarray(data["phis"])
        Zs = np.asarray(data["Zs"])
        nR, nphi, nZ = len(Rs), len(phis), len(Zs)
        psi3 = psi_flat.reshape((nR, nphi, nZ))
        grid = {"type": "cylindrical", "Rs": Rs, "phis": phis, "Zs": Zs,
                "shape": (nR, nphi, nZ)}
    else:
        xs = np.asarray(data["xs"])
        ys = np.asarray(data["ys"])
        zs = np.asarray(data["zs"])
        nx, ny, nz = len(xs), len(ys), len(zs)
        psi3 = psi_flat.reshape((nx, ny, nz))
        grid = {"type": "cartesian", "xs": xs, "ys": ys, "zs": zs,
                "shape": (nx, ny, nz)}

    return {
        "psi3": psi3,
        "grid": grid,
        "inside": inside,
        "boundary_band": boundary_band,
        "axis_band": axis_band,
        "mins": mins,
        "maxs": maxs,
        "axis_points": axis_points,
        "R_axis": R_axis,
        "Z_axis": Z_axis,
    }

def load_mfs_grad_phi(mfs_npz: str):
    data = np.load(mfs_npz, allow_pickle=True)
    center = data["center"]
    scale = float(data["scale"])
    Yn = data["Yn"]
    alpha = data["alpha"]
    a = data["a"]
    a_hat = data["a_hat"]
    P = data["P"]
    N = data["N"]

    evals = Evaluators(center=jnp.asarray(center),
                       scale=scale,
                       Yn=jnp.asarray(Yn),
                       alpha=jnp.asarray(alpha),
                       a=jnp.asarray(a),
                       a_hat=jnp.asarray(a_hat))
    grad_phi = evals.build_grad_phi()
    return grad_phi, P, N

# ============================================================
# Axis geometry and angles
# ============================================================

def build_axis_interp(axis_points: np.ndarray) -> Tuple[interp1d, interp1d]:
    """
    axis_points[k] = (x,y,z) along the magnetic axis, roughly uniform in φ.
    Build R_axis(φ), Z_axis(φ) with φ ∈ [0, 2π).
    """
    x = axis_points[:, 0]
    y = axis_points[:, 1]
    z = axis_points[:, 2]

    phi_axis = np.mod(np.arctan2(y, x), 2*np.pi)
    R_axis = np.sqrt(x*x + y*y)

    order = np.argsort(phi_axis)
    phi_sorted = phi_axis[order]
    R_sorted = R_axis[order]
    Z_sorted = z[order]

    # enforce periodicity
    phi_ext = np.concatenate([phi_sorted, phi_sorted[0:1] + 2*np.pi])
    R_ext = np.concatenate([R_sorted, R_sorted[0:1]])
    Z_ext = np.concatenate([Z_sorted, Z_sorted[0:1]])

    R_interp = interp1d(phi_ext, R_ext, kind="cubic", assume_sorted=True)
    Z_interp = interp1d(phi_ext, Z_ext, kind="cubic", assume_sorted=True)
    return R_interp, Z_interp

def geometric_angles_from_axis(X: np.ndarray,
                               R_axis_interp,
                               Z_axis_interp) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given points X (N,3), compute:
       φ = atan2(y,x)
       θ = poloidal angle in (R,Z) around axis position at each φ
       R = sqrt(x^2 + y^2)
    """
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    phi = np.mod(np.arctan2(y, x), 2*np.pi)
    R = np.sqrt(x*x + y*y)

    Rax = R_axis_interp(phi)
    Zax = Z_axis_interp(phi)

    dR = R - Rax
    dZ = z - Zax
    theta = np.mod(np.arctan2(dZ, dR), 2*np.pi)

    return phi, theta, R

# ============================================================
# ∇ψ on grid (for possible extensions; not strictly needed for ι)
# ============================================================

def compute_grad_psi_cartesian(psi3: np.ndarray,
                               xs: np.ndarray,
                               ys: np.ndarray,
                               zs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = psi3.shape
    gx = np.zeros_like(psi3)
    gy = np.zeros_like(psi3)
    gz = np.zeros_like(psi3)

    # d/dx
    for i in range(nx):
        if i == 0:
            dx = xs[1] - xs[0]
            gx[i, :, :] = (psi3[1, :, :] - psi3[0, :, :]) / dx
        elif i == nx - 1:
            dx = xs[-1] - xs[-2]
            gx[i, :, :] = (psi3[-1, :, :] - psi3[-2, :, :]) / dx
        else:
            dx = xs[i+1] - xs[i-1]
            gx[i, :, :] = (psi3[i+1, :, :] - psi3[i-1, :, :]) / dx

    # d/dy
    for j in range(ny):
        if j == 0:
            dy = ys[1] - ys[0]
            gy[:, j, :] = (psi3[:, 1, :] - psi3[:, 0, :]) / dy
        elif j == ny - 1:
            dy = ys[-1] - ys[-2]
            gy[:, j, :] = (psi3[:, -1, :] - psi3[:, -2, :]) / dy
        else:
            dy = ys[j+1] - ys[j-1]
            gy[:, j, :] = (psi3[:, j+1, :] - psi3[:, j-1, :]) / dy

    # d/dz
    for k in range(nz):
        if k == 0:
            dz = zs[1] - zs[0]
            gz[:, :, k] = (psi3[:, :, 1] - psi3[:, :, 0]) / dz
        elif k == nz - 1:
            dz = zs[-1] - zs[-2]
            gz[:, :, k] = (psi3[:, :, -1] - psi3[:, :, -2]) / dz
        else:
            dz = zs[k+1] - zs[k-1]
            gz[:, :, k] = (psi3[:, :, k+1] - psi3[:, :, k-1]) / dz

    return gx, gy, gz

def compute_grad_psi_cylindrical(psi3: np.ndarray,
                                 Rs: np.ndarray,
                                 phis: np.ndarray,
                                 Zs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ∇ψ on cylindrical grid (R,φ,Z), returned as Cartesian (ψ_x, ψ_y, ψ_z).
    Included here mainly for possible future extensions / consistency checks.
    """
    nR, nphi, nZ = psi3.shape
    dpsi_dR = np.zeros_like(psi3)
    dpsi_dphi = np.zeros_like(psi3)
    dpsi_dZ = np.zeros_like(psi3)

    # ∂ψ/∂R
    for i in range(nR):
        if i == 0:
            dR = Rs[1] - Rs[0]
            dpsi_dR[i, :, :] = (psi3[1, :, :] - psi3[0, :, :]) / dR
        elif i == nR - 1:
            dR = Rs[-1] - Rs[-2]
            dpsi_dR[i, :, :] = (psi3[-1, :, :] - psi3[-2, :, :]) / dR
        else:
            dR = Rs[i+1] - Rs[i-1]
            dpsi_dR[i, :, :] = (psi3[i+1, :, :] - psi3[i-1, :, :]) / dR

    # ∂ψ/∂φ (periodic)
    for j in range(nphi):
        jm = (j - 1) % nphi
        jp = (j + 1) % nphi
        dphi = phis[jp] - phis[jm]
        dpsi_dphi[:, j, :] = (psi3[:, jp, :] - psi3[:, jm, :]) / dphi

    # ∂ψ/∂Z
    for k in range(nZ):
        if k == 0:
            dZ = Zs[1] - Zs[0]
            dpsi_dZ[:, :, k] = (psi3[:, :, 1] - psi3[:, :, 0]) / dZ
        elif k == nZ - 1:
            dZ = Zs[-1] - Zs[-2]
            dpsi_dZ[:, :, k] = (psi3[:, :, -1] - psi3[:, :, -2]) / dZ
        else:
            dZ = Zs[k+1] - Zs[k-1]
            dpsi_dZ[:, :, k] = (psi3[:, :, k+1] - psi3[:, :, k-1]) / dZ

    psi_x = np.zeros_like(psi3)
    psi_y = np.zeros_like(psi3)
    psi_z = dpsi_dZ

    for i, R in enumerate(Rs):
        if R == 0.0:
            psi_x[i, :, :] = 0.0
            psi_y[i, :, :] = 0.0
        else:
            for j, phi in enumerate(phis):
                cosp = np.cos(phi)
                sinp = np.sin(phi)
                dRpsi = dpsi_dR[i, j, :]
                dphipsi = dpsi_dphi[i, j, :] / R
                psi_x[i, j, :] = dRpsi * cosp - dphipsi * sinp
                psi_y[i, j, :] = dRpsi * sinp + dphipsi * cosp

    return psi_x, psi_y, psi_z

# ============================================================
# Sampling points on ψ levels
# ============================================================

def sample_points_on_psi_levels(
    psi3: np.ndarray,
    grid: Dict[str, Any],
    inside: np.ndarray,
    psi_levels: np.ndarray,
    band_frac: float = 0.01,
    max_points_per_level: int = 8000
) -> List[Dict[str, Any]]:
    """
    For each ψ_level, build a thin band:
        |ψ - ψ_level| < band_width
    where band_width = band_frac * (ψ_max - ψ_min).

    Returns a list of dicts with keys:
      "psi_level" : float
      "X"         : (Ni,3) Cartesian points
      "psi"       : (Ni,) local ψ values at those nodes
    """
    psi_flat = psi3.ravel(order="C")
    mask_inside = inside.astype(bool)
    psi_inside = psi_flat[mask_inside]

    psi_min = float(np.min(psi_inside))
    psi_max = float(np.max(psi_inside))
    band_width = band_frac * (psi_max - psi_min)

    # Build Xall consistent with psi_flat ordering
    if grid["type"] == "cylindrical":
        Rs = grid["Rs"]
        phis = grid["phis"]
        Zs = grid["Zs"]
        nR, nphi, nZ = grid["shape"]
        RR, PHI, ZZ = np.meshgrid(Rs, phis, Zs, indexing="ij")
        XX = RR * np.cos(PHI)
        YY = RR * np.sin(PHI)
        Xall = np.column_stack([XX.ravel(order="C"),
                                YY.ravel(order="C"),
                                ZZ.ravel(order="C")])
    else:
        xs = grid["xs"]
        ys = grid["ys"]
        zs = grid["zs"]
        nx, ny, nz = grid["shape"]
        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
        XX = XX.transpose(1, 0, 2)
        YY = YY.transpose(1, 0, 2)
        ZZ = ZZ.transpose(1, 0, 2)
        Xall = np.column_stack([XX.ravel(order="C"),
                                YY.ravel(order="C"),
                                ZZ.ravel(order="C")])

    level_data: List[Dict[str, Any]] = []

    for psi0 in psi_levels:
        mask_band = (
            mask_inside &
            (np.abs(psi_flat - psi0) < band_width)
        )
        idx = np.where(mask_band)[0]
        if idx.size == 0:
            continue

        if idx.size > max_points_per_level:
            idx = np.random.choice(idx, size=max_points_per_level,
                                   replace=False)

        X_band = Xall[idx, :]
        psi_band = psi_flat[idx]
        level_data.append({
            "psi_level": float(psi0),
            "X": X_band,
            "psi": psi_band,
        })

    return level_data

# ============================================================
# Field-line tracing and rotational transform
# ============================================================

def _grad_phi_np(grad_phi, x_np: np.ndarray) -> np.ndarray:
    """Small helper: call JAX grad_phi from NumPy."""
    x_j = jnp.asarray(x_np[None, :], dtype=jnp.float64)  # (1,3)
    B_j = grad_phi(x_j)[0]
    return np.asarray(B_j, dtype=float)

def _unwrap_angle_delta(prev: float, new: float) -> float:
    """Return unwrapped delta angle in (-π, π]."""
    delta = new - prev
    while delta > np.pi:
        delta -= 2*np.pi
    while delta <= -np.pi:
        delta += 2*np.pi
    return delta

def trace_fieldline(
    X0: np.ndarray,
    grad_phi,
    R_axis_interp,
    Z_axis_interp,
    mins: np.ndarray,
    maxs: np.ndarray,
    dl: float = 0.01,
    n_steps_max: int = 40000,
    n_tor_target: int = 3
) -> Tuple[bool, float, float]:
    """
    Trace a field line starting at X0 by integrating dX/dλ = B/|B|.

    Returns
    -------
    success : bool
        True if we reached the requested number of toroidal turns before
        leaving the domain or hitting max steps.
    dtheta_total : float
        Total change in geometric poloidal angle θ (unwrapped).
    dphi_total : float
        Total change in geometric toroidal angle φ (unwrapped).
    """
    X = np.asarray(X0, dtype=float)
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)

    # initial angles
    phi_prev, theta_prev, _ = geometric_angles_from_axis(
        X[None, :], R_axis_interp, Z_axis_interp
    )
    phi_prev = float(phi_prev[0])
    theta_prev = float(theta_prev[0])

    dphi_tot = 0.0
    dtheta_tot = 0.0
    toroidal_turns_target = 2.0 * np.pi * n_tor_target

    for _ in range(n_steps_max):
        B = _grad_phi_np(grad_phi, X)
        Bnorm = np.linalg.norm(B)
        if not np.isfinite(Bnorm) or Bnorm < 1e-16:
            return False, dtheta_tot, dphi_tot

        # normalized direction
        X = X + dl * (B / Bnorm)

        # safety: inside bounding box?
        if np.any(X < (mins - 0.1)) or np.any(X > (maxs + 0.1)):
            return False, dtheta_tot, dphi_tot

        phi_new, theta_new, _ = geometric_angles_from_axis(
            X[None, :], R_axis_interp, Z_axis_interp
        )
        phi_new = float(phi_new[0])
        theta_new = float(theta_new[0])

        dphi = _unwrap_angle_delta(phi_prev, phi_new)
        dtheta = _unwrap_angle_delta(theta_prev, theta_new)

        dphi_tot += dphi
        dtheta_tot += dtheta

        phi_prev = phi_new
        theta_prev = theta_new

        if abs(dphi_tot) >= toroidal_turns_target:
            return True, dtheta_tot, dphi_tot

    # reached max steps
    return False, dtheta_tot, dphi_tot

def choose_seeds_on_surface(
    X: np.ndarray,
    R_axis_interp,
    Z_axis_interp,
    n_seeds: int
) -> np.ndarray:
    """
    Choose n_seeds points on a surface X (N,3) roughly uniformly in θ.
    """
    phi, theta, _ = geometric_angles_from_axis(X, R_axis_interp, Z_axis_interp)
    N = X.shape[0]
    if N <= n_seeds:
        return X.copy()

    # desired θ values
    target_theta = np.linspace(0.0, 2.0*np.pi, n_seeds, endpoint=False)
    seeds = []
    used_idx = set()
    for t in target_theta:
        # angular distance with periodicity
        dtheta = np.angle(np.exp(1j*(theta - t)))
        idx = np.argmin(np.abs(dtheta))
        # avoid duplicates by small random jitter if necessary
        if idx in used_idx:
            # pick a slightly different index
            idx = (idx + 1) % N
        used_idx.add(idx)
        seeds.append(X[idx, :])
    return np.array(seeds)

def estimate_iota_on_surface(
    surf_data: Dict[str, Any],
    grad_phi,
    R_axis_interp,
    Z_axis_interp,
    mins: np.ndarray,
    maxs: np.ndarray,
    n_seeds: int = 8,
    dl: float = 0.01,
    n_steps_max: int = 40000,
    n_tor_target: int = 3
) -> Tuple[float, float, int]:
    """
    Estimate ι on a given surface by averaging Δθ/Δφ over several seeds.

    Returns
    -------
    iota_mean : float
    iota_std : float
    n_success : int
    """
    X = surf_data["X"]  # (N,3)
    seeds = choose_seeds_on_surface(X, R_axis_interp, Z_axis_interp, n_seeds)
    iotas = []

    for iseed, X0 in enumerate(seeds):
        success, dtheta, dphi = trace_fieldline(
            X0, grad_phi,
            R_axis_interp, Z_axis_interp,
            mins, maxs,
            dl=dl,
            n_steps_max=n_steps_max,
            n_tor_target=n_tor_target
        )
        if not success:
            print(f"[WARN] Field line seed {iseed} failed to reach "
                  f"{n_tor_target} toroidal turns (dφ={dphi:.2f}).")
            continue
        if abs(dphi) < 1e-6:
            print(f"[WARN] Tiny Δφ for seed {iseed}; skipping.")
            continue

        iota_val = dtheta / dphi
        if np.isfinite(iota_val):
            iotas.append(iota_val)

    if len(iotas) == 0:
        return np.nan, np.nan, 0

    iotas = np.asarray(iotas)
    return float(np.mean(iotas)), float(np.std(iotas)), len(iotas)

# ============================================================
# VMEC loader
# ============================================================

def load_vmec_iota(wout_file: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Load ι(s) from a VMEC wout NetCDF file, if netCDF4 is available.
    """
    if not _HAS_NETCDF4:
        print("[WARN] netCDF4 not available; VMEC comparison disabled.")
        return None

    if not os.path.exists(wout_file):
        print(f"[WARN] VMEC file not found: {wout_file}")
        return None

    print(f"[INFO] Loading VMEC equilibrium from: {wout_file}")
    with Dataset(wout_file, mode="r") as ds:
        if "iotaf" not in ds.variables:
            print("[WARN] 'iotaf' not found in VMEC file; skipping VMEC diagnostics.")
            return None

        iotaf = np.array(ds.variables["iotaf"][:])  # typically size ns
        # try to get radial coordinate 's' (normalized toroidal flux)
        if "s" in ds.variables:
            s = np.array(ds.variables["s"][:])
        elif "phi" in ds.variables:
            phi = np.array(ds.variables["phi"][:])
            s = phi / np.max(phi)
        else:
            s = np.linspace(0.0, 1.0, len(iotaf))

    return {"s": s, "iotaf": iotaf}

# ============================================================
# Main diagnostics
# ============================================================

def compute_iota_profiles(
    psi_npz: str,
    mfs_npz: str,
    n_surfaces: int = 8,
    band_frac: float = 0.01,
    max_points_per_level: int = 8000,
    n_seeds_per_surface: int = 8,
    dl: float = 0.01,
    n_tor_target: int = 3,
) -> Dict[str, Any]:
    """
    High-level routine: assemble everything and compute ι(ψ) and shear.
    """
    # 1) Load ψ snapshot and MFS grad_phi
    psi_data = load_psi_snapshot(psi_npz)
    psi3 = psi_data["psi3"]
    grid = psi_data["grid"]
    inside = psi_data["inside"]
    mins = psi_data["mins"]
    maxs = psi_data["maxs"]
    axis_points = psi_data["axis_points"]

    grad_phi, P_surf, N_surf = load_mfs_grad_phi(mfs_npz)

    # Axis interpolants
    R_axis_interp, Z_axis_interp = build_axis_interp(axis_points)

    # ψ range
    psi_flat = psi3.ravel(order="C")
    mask_inside = inside.astype(bool)
    psi_inside = psi_flat[mask_inside]
    psi_min = float(np.min(psi_inside))
    psi_max = float(np.max(psi_inside))

    eps_psi = 0.05 * (psi_max - psi_min)
    psi_levels = np.linspace(psi_min + eps_psi,
                             psi_max - eps_psi,
                             n_surfaces)

    print("[INFO] ψ range used for ι diagnostics "
          f"[{psi_min:.3e}, {psi_max:.3e}] (excluding 5% at ends).")

    # 2) Sample surfaces
    level_data = sample_points_on_psi_levels(
        psi3, grid, inside,
        psi_levels=psi_levels,
        band_frac=band_frac,
        max_points_per_level=max_points_per_level,
    )

    if len(level_data) == 0:
        raise RuntimeError("No ψ-levels with valid samples; "
                           "check band_frac and ψ-range.")

    # 3) Estimate ι on each surface
    psi_used = []
    iota_vals = []
    iota_errs = []
    n_success_list = []

    for isurf, surf in enumerate(level_data):
        psi0 = surf["psi_level"]
        X = surf["X"]
        print(f"[INFO] Surface {isurf}: ψ≈{psi0:.3e}, N_points={X.shape[0]}")

        iota_mean, iota_std, n_success = estimate_iota_on_surface(
            surf, grad_phi,
            R_axis_interp, Z_axis_interp,
            mins, maxs,
            n_seeds=n_seeds_per_surface,
            dl=dl,
            n_steps_max=40000,
            n_tor_target=n_tor_target,
        )
        if n_success == 0 or not np.isfinite(iota_mean):
            print(f"[WARN] Unable to compute ι on ψ≈{psi0:.3e}; skipping.")
            continue

        psi_used.append(psi0)
        iota_vals.append(iota_mean)
        iota_errs.append(iota_std)
        n_success_list.append(n_success)

        print(f"[IOTA] ψ≈{psi0:.3e}: ι = {iota_mean:.6f} ± {iota_std:.3e} "
              f"(N_success={n_success})")

    if len(psi_used) < 3:
        raise RuntimeError("Fewer than 3 valid surfaces for ι; "
                           "cannot build a meaningful profile.")

    psi_used = np.array(psi_used)
    iota_vals = np.array(iota_vals)
    iota_errs = np.array(iota_errs)
    n_success_list = np.array(n_success_list, dtype=int)

    # Sort by ψ
    order = np.argsort(psi_used)
    psi_used = psi_used[order]
    iota_vals = iota_vals[order]
    iota_errs = iota_errs[order]
    n_success_list = n_success_list[order]

    # 4) Magnetic shear: ι'(ψ) via finite differences
    shear = np.zeros_like(iota_vals)
    # central differences
    for i in range(len(psi_used)):
        if i == 0:
            shear[i] = (iota_vals[i+1] - iota_vals[i]) / (psi_used[i+1] - psi_used[i])
        elif i == len(psi_used) - 1:
            shear[i] = (iota_vals[i] - iota_vals[i-1]) / (psi_used[i] - psi_used[i-1])
        else:
            shear[i] = (iota_vals[i+1] - iota_vals[i-1]) / (psi_used[i+1] - psi_used[i-1])

    # Normalized ψ (0 at ψ_min, 1 at ψ_max)
    psi_norm = (psi_used - psi_min) / (psi_max - psi_min)

    return {
        "psi_used": psi_used,
        "psi_norm": psi_norm,
        "iota": iota_vals,
        "iota_std": iota_errs,
        "shear": shear,
        "psi_min": psi_min,
        "psi_max": psi_max,
        "n_success": n_success_list,
    }

# ============================================================
# Plotting
# ============================================================

def publication_matplotlib_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": False,
        "figure.figsize": (6.0, 4.5),
        "savefig.dpi": 300,
    })

def make_iota_plots(
    base: str,
    diag: Dict[str, Any],
    vmec: Optional[Dict[str, np.ndarray]] = None
):
    """
    Generate publication-ready plots for ι and shear, optionally with VMEC.
    """
    psi_norm = diag["psi_norm"]
    iota = diag["iota"]
    iota_std = diag["iota_std"]
    shear = diag["shear"]

    # 1) ι vs ψ̂
    fig1, ax1 = plt.subplots()
    ax1.errorbar(psi_norm, iota, yerr=iota_std,
                 fmt="o", capsize=3, label="MFS+ψ (this work)")
    if vmec is not None:
        s = vmec["s"]
        iotaf = vmec["iotaf"]
        s_norm = (s - s.min()) / (s.max() - s.min())
        ax1.plot(s_norm, iotaf, "-", label="VMEC")
    ax1.set_xlabel(r"Normalized flux $\hat{\psi}$")
    ax1.set_ylabel(r"$\iota(\hat{\psi})$")
    ax1.set_title("Rotational transform profile")
    ax1.legend(loc="best")
    fig1.tight_layout()
    out1 = base + "_iota_profile.png"
    fig1.savefig(out1, bbox_inches="tight")
    print(f"[PLOT] Saved iota profile to {out1}")

    # 2) Shear vs ψ̂
    fig2, ax2 = plt.subplots()
    ax2.plot(psi_norm, shear, "o-", label="MFS+ψ shear")
    if vmec is not None:
        s = vmec["s"]
        iotaf = vmec["iotaf"]
        s_norm = (s - s.min()) / (s.max() - s.min())
        shear_vmec = np.zeros_like(iotaf)
        for i in range(len(iotaf)):
            if i == 0:
                shear_vmec[i] = (iotaf[i+1] - iotaf[i]) / (s[i+1] - s[i])
            elif i == len(iotaf) - 1:
                shear_vmec[i] = (iotaf[i] - iotaf[i-1]) / (s[i] - s[i-1])
            else:
                shear_vmec[i] = (iotaf[i+1] - iotaf[i-1]) / (s[i+1] - s[i-1])
        ax2.plot(s_norm, shear_vmec, "-", label="VMEC shear")
    ax2.set_xlabel(r"Normalized flux $\hat{\psi}$")
    ax2.set_ylabel(r"$\iota'(\hat{\psi})$")
    ax2.set_title("Magnetic shear profile")
    ax2.legend(loc="best")
    fig2.tight_layout()
    out2 = base + "_shear_profile.png"
    fig2.savefig(out2, bbox_inches="tight")
    print(f"[PLOT] Saved shear profile to {out2}")

    # 3) Relative error Δι/ι_VMEC if VMEC present
    if vmec is not None:
        s = vmec["s"]
        iotaf = vmec["iotaf"]
        s_norm = (s - s.min()) / (s.max() - s.min())

        interp_iotaf = interp1d(s_norm, iotaf, kind="cubic",
                                bounds_error=False,
                                fill_value="extrapolate")
        iotaf_on_psi = interp_iotaf(psi_norm)

        mask = (np.abs(iotaf_on_psi) > 1e-8)
        rel_err = np.zeros_like(iota)
        rel_err[mask] = (iota[mask] - iotaf_on_psi[mask]) / iotaf_on_psi[mask]

        fig3, ax3 = plt.subplots()
        ax3.plot(psi_norm[mask], rel_err[mask], "o-")
        ax3.axhline(0.0, color="k", lw=0.8)
        ax3.set_xlabel(r"Normalized flux $\hat{\psi}$")
        ax3.set_ylabel(r"$\Delta \iota / \iota_{\mathrm{VMEC}}$")
        ax3.set_title("Relative error in rotational transform")
        fig3.tight_layout()
        out3 = base + "_iota_rel_error.png"
        fig3.savefig(out3, bbox_inches="tight")
        print(f"[PLOT] Saved relative iota error plot to {out3}")

# ============================================================
# CLI
# ============================================================

def main():
    default_solution = "wout_precise_QA_solution.npz"
    default_psi_npz = default_solution.replace(".npz", "_psi_fci_cyl_N64_Nphi128.npz")

    parser = argparse.ArgumentParser(
        description="Rotational transform and shear diagnostics from MFS+ψ field."
    )
    parser.add_argument(
        "mfs_npz",
        nargs="?",
        default=resolve_npz_file_location(default_solution),
        help="MFS solution checkpoint (.npz) to rebuild grad_phi",
    )
    parser.add_argument(
        "--psi-npz",
        default=default_psi_npz,
        help="psi_fci snapshot (.npz) produced by FCI solver",
    )
    parser.add_argument(
        "--vmec-wout",
        default=None,
        help="Optional VMEC wout NetCDF file for comparison",
    )
    parser.add_argument(
        "--n-surfaces",
        type=int,
        default=8,
        help="Number of ψ surfaces to sample",
    )
    parser.add_argument(
        "--band-frac",
        type=float,
        default=0.01,
        help="Relative ψ-band half-width around each level",
    )
    parser.add_argument(
        "--max-points-per-level",
        type=int,
        default=8000,
        help="Max points per surface used for seeding",
    )
    parser.add_argument(
        "--n-seeds-per-surface",
        type=int,
        default=8,
        help="Number of field-line seeds per surface",
    )
    parser.add_argument(
        "--dl",
        type=float,
        default=0.01,
        help="Field-line step size in configuration space",
    )
    parser.add_argument(
        "--n-tor-target",
        type=int,
        default=3,
        help="Number of toroidal turns per field line",
    )
    args = parser.parse_args()

    publication_matplotlib_style()

    psi_npz = args.psi_npz
    mfs_npz = args.mfs_npz

    print("==============================================")
    print("[INFO] iota_diagnostics.py")
    print(f"[INFO] MFS solution: {mfs_npz}")
    print(f"[INFO] ψ snapshot : {psi_npz}")
    if args.vmec_wout is not None:
        print(f"[INFO] VMEC wout  : {args.vmec_wout}")
    print("==============================================")

    diag = compute_iota_profiles(
        psi_npz=psi_npz,
        mfs_npz=mfs_npz,
        n_surfaces=args.n_surfaces,
        band_frac=args.band_frac,
        max_points_per_level=args.max_points_per_level,
        n_seeds_per_surface=args.n_seeds_per_surface,
        dl=args.dl,
        n_tor_target=args.n_tor_target,
    )

    vmec_data = None
    if args.vmec_wout is not None:
        vmec_data = load_vmec_iota(args.vmec_wout)

    # Base for output filenames
    base = psi_npz.replace(".npz", "_iota")

    make_iota_plots(base, diag, vmec=vmec_data)

    # Save numerical data
    out_npz = base + "_profiles.npz"
    np.savez(
        out_npz,
        psi_used=diag["psi_used"],
        psi_norm=diag["psi_norm"],
        iota=diag["iota"],
        iota_std=diag["iota_std"],
        shear=diag["shear"],
        psi_min=diag["psi_min"],
        psi_max=diag["psi_max"],
        n_success=diag["n_success"],
        vmec_s=None if vmec_data is None else vmec_data["s"],
        vmec_iotaf=None if vmec_data is None else vmec_data["iotaf"],
    )
    print(f"[SAVE] Saved iota/shear profiles to {out_npz}")
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
