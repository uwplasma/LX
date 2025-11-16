#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Quasisymmetry triple-product metric from flux-function ψ and MFS magnetic field.

This script:
  1) Loads ψ(𝐱) from a psi_fci snapshot (.npz).
  2) Loads the MFS solution checkpoint (center, scale, Yn, alpha, a, a_hat, P, N).
  3) Rebuilds ∇φ(𝐱) (so B = ∇φ) using the same Evaluators pattern.
  4) Precomputes ∇ψ on the ψ grid (finite differences in cylindrical or Cartesian).
  5) For several ψ-levels (flux surfaces), samples points in a thin ψ-band.
  6) At each surface point, evaluates the Landreman/DESC triple-product quantity

         f_T = (∇ψ × ∇B) · ∇( B · ∇B ),

     where B = |𝐁| and ∇B and ∇(B·∇B) are obtained via JAX autodiff.
  7) Defines a normalized QS error per surface

         \hat f_T(ψ) = <R>^2 <|f_T|> / <B>^4,

     approximating <·> as simple averages over sampled points.
  8) Aggregates to a global QS metric and makes publication-ready plots:

       - |B|(θ, φ) on a mid-radius surface
       - |B|(θ, φ) on an outer surface
       - \hat f_T vs normalized ψ.

References:
  - Landreman, "Introduction to quasisymmetry", Sec. 8, condition (d).
  - DESC documentation, basic_optimization tutorial (quasisymmetry triple-product objective).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap

from scipy.interpolate import interp1d

import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

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

# ---------------------------------------------------------------------------
# Reuse Evaluators from your MFS code (slightly trimmed)
# ---------------------------------------------------------------------------

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
                  sc_center: jnp.ndarray, sc_scale: float):
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
            # X: (..., 3)
            Xn = (X - sc_c) * sc_s
            return grad_mv(X) + sc_s * vmap(dS_single)(Xn)

        return grad_phi_fn

# ---------------------------------------------------------------------------
# Helpers to load psi and axis, and generate coordinates
# ---------------------------------------------------------------------------

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

def load_mfs_grad_phi(mfs_npz: str) -> Tuple[Any, np.ndarray, np.ndarray]:
    data = np.load(mfs_npz, allow_pickle=True)
    center = data["center"]
    scale = float(data["scale"])
    Yn = data["Yn"]
    alpha = data["alpha"]
    a = data["a"]
    a_hat = data["a_hat"]
    P = data["P"]
    N = data["N"]

    evals = Evaluators(center=jnp.asarray(center), scale=scale,
                       Yn=jnp.asarray(Yn), alpha=jnp.asarray(alpha),
                       a=jnp.asarray(a), a_hat=jnp.asarray(a_hat))
    grad_phi = evals.build_grad_phi()
    return grad_phi, P, N

# ---------------------------------------------------------------------------
# Axis-based geometric angles (φ, θ)
# ---------------------------------------------------------------------------

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

    # sort by φ and build periodic interpolants
    order = np.argsort(phi_axis)
    phi_sorted = phi_axis[order]
    R_sorted = R_axis[order]
    Z_sorted = z[order]

    # enforce periodicity by adding 2π endpoint copies
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
    Returns (phi, theta, R): all shape (N,).
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
    theta = np.arctan2(dZ, dR)  # geometric poloidal angle in local (R,Z) plane

    return phi, theta, R

# ---------------------------------------------------------------------------
# ∇ψ on the ψ grid (finite differences)
# ---------------------------------------------------------------------------

def compute_grad_psi_cartesian(psi3: np.ndarray,
                               xs: np.ndarray,
                               ys: np.ndarray,
                               zs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite-difference gradient of ψ on a Cartesian grid."""
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
    Finite-difference gradient of ψ on a cylindrical grid (R,φ,Z),
    returned as Cartesian components (ψ_x, ψ_y, ψ_z) on the same grid.
    """
    nR, nphi, nZ = psi3.shape
    dpsi_dR = np.zeros_like(psi3)
    dpsi_dphi = np.zeros_like(psi3)
    dpsi_dZ = np.zeros_like(psi3)

    # ∂ψ/∂R (one-sided at boundaries, central inside)
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

    # Convert to Cartesian: ∇ψ = (∂ψ/∂R) e_R + (1/R) ∂ψ/∂φ e_φ + (∂ψ/∂Z) e_Z
    psi_x = np.zeros_like(psi3)
    psi_y = np.zeros_like(psi3)
    psi_z = np.zeros_like(psi3)

    for i, R in enumerate(Rs):
        if R == 0.0:
            # On-axis: cylindrical basis is singular; set to 0 (we avoid axis surfaces anyway)
            psi_x[i, :, :] = 0.0
            psi_y[i, :, :] = 0.0
        else:
            for j, phi in enumerate(phis):
                cosp = np.cos(phi)
                sinp = np.sin(phi)
                dRpsi = dpsi_dR[i, j, :]
                dphipsi = dpsi_dphi[i, j, :] / R
                # e_R = (cosφ, sinφ, 0), e_φ = (-sinφ, cosφ, 0)
                psi_x[i, j, :] = dRpsi * cosp - dphipsi * sinp
                psi_y[i, j, :] = dRpsi * sinp + dphipsi * cosp

    psi_z = dpsi_dZ
    return psi_x, psi_y, psi_z

# ---------------------------------------------------------------------------
# Sampling points on flux surfaces (ψ bands)
# ---------------------------------------------------------------------------

def sample_points_on_psi_levels(psi3: np.ndarray,
                                grid: Dict[str, Any],
                                inside: np.ndarray,
                                psi_levels: np.ndarray,
                                band_frac: float = 0.01,
                                max_points_per_level: int = 8000,
                                gradpsi_flat: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
                                ) -> List[Dict[str, Any]]:
    """
    For each ψ_level, build a thin band:
        |ψ - ψ_level| < band_width
    where band_width = band_frac * (ψ_max - ψ_min).

    Returns:
      level_data: list of dicts with keys:
        "psi_level": float
        "X": (Ni,3) points in Cartesian coords
        "psi": (Ni,) local ψ
        "grad_psi": (Ni,3) gradient vectors (if gradpsi_flat is provided)
    """
    psi_flat = psi3.ravel(order="C")
    mask_inside = inside.astype(bool)
    psi_inside = psi_flat[mask_inside]

    psi_min = float(np.min(psi_inside))
    psi_max = float(np.max(psi_inside))
    band_width = band_frac * (psi_max - psi_min)

    level_data: List[Dict[str, Any]] = []

    # build Xall mapping consistent with psi_flat ordering
    if grid["type"] == "cylindrical":
        Rs = grid["Rs"]
        phis = grid["phis"]
        Zs = grid["Zs"]
        nR, nphi, nZ = grid["shape"]

        RR, PHI, ZZ = np.meshgrid(Rs, phis, Zs, indexing="ij")  # (nR,nphi,nZ)
        XX = RR * np.cos(PHI)
        YY = RR * np.sin(PHI)

        Xall = np.column_stack([
            XX.ravel(order="C"),
            YY.ravel(order="C"),
            ZZ.ravel(order="C"),
        ])

    else:
        xs = grid["xs"]
        ys = grid["ys"]
        zs = grid["zs"]
        nx, ny, nz = grid["shape"]

        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
        XX = XX.transpose(1, 0, 2)
        YY = YY.transpose(1, 0, 2)
        ZZ = ZZ.transpose(1, 0, 2)
        Xall = np.column_stack([
            XX.ravel(order="C"),
            YY.ravel(order="C"),
            ZZ.ravel(order="C"),
        ])

    if gradpsi_flat is not None:
        gx_flat, gy_flat, gz_flat = gradpsi_flat
    else:
        gx_flat = gy_flat = gz_flat = None

    for psi0 in psi_levels:
        mask_band = (
            mask_inside &
            (np.abs(psi_flat - psi0) < band_width)
        )
        idx = np.where(mask_band)[0]
        if idx.size == 0:
            continue

        if idx.size > max_points_per_level:
            # randomly subsample to control cost
            idx = np.random.choice(idx, size=max_points_per_level, replace=False)

        X_band = Xall[idx, :]
        psi_band = psi_flat[idx]
        surf_dict: Dict[str, Any] = {
            "psi_level": float(psi0),
            "X": X_band,
            "psi": psi_band,
        }

        if gx_flat is not None:
            g_band = np.column_stack([
                gx_flat[idx],
                gy_flat[idx],
                gz_flat[idx],
            ])
            surf_dict["grad_psi"] = g_band

        level_data.append(surf_dict)

    return level_data

# ---------------------------------------------------------------------------
# Triple-product f_T via JAX autodiff
# ---------------------------------------------------------------------------

def build_triple_product_fn(grad_phi):
    """
    Build a JAX-jitted function that, for a batch of points X (N,3) and
    precomputed ∇ψ(X) (N,3), returns:
      f_T(X) = (∇ψ × ∇B) · ∇(B · ∇B),
      Bmag(X), and R(X).
    """

    def fT_surface(X_pts: np.ndarray,
                   gradpsi_pts: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        X = jnp.asarray(X_pts, dtype=jnp.float64)          # (N,3)
        Gpsi = jnp.asarray(gradpsi_pts, dtype=jnp.float64) # (N,3)

        def Bmag_fn(x):
            B = grad_phi(x[jnp.newaxis, :])[0]
            return jnp.linalg.norm(B)

        grad_Bmag_fn = jax.grad(Bmag_fn)

        def BdotgradB_fn(x):
            B = grad_phi(x[jnp.newaxis, :])[0]
            gB = grad_Bmag_fn(x)
            return jnp.dot(B, gB)

        grad_BdotgradB_fn = jax.grad(BdotgradB_fn)

        # Evaluate gradients in batch
        grad_Bmag = vmap(grad_Bmag_fn)(X)           # (N,3)
        grad_BdotgradB = vmap(grad_BdotgradB_fn)(X) # (N,3)

        # Triple product
        cross = jnp.cross(Gpsi, grad_Bmag)
        fT = jnp.sum(cross * grad_BdotgradB, axis=1)   # (N,)

        # Also return |B| and R at these points
        B_vec = grad_phi(X)                           # (N,3)
        Bmag = jnp.linalg.norm(B_vec, axis=1)         # (N,)
        R = jnp.sqrt(X[:, 0]**2 + X[:, 1]**2)         # (N,)

        return fT, Bmag, R

    return jit(fT_surface)

# ---------------------------------------------------------------------------
# Main: put it all together
# ---------------------------------------------------------------------------

def main(psi_npz: str,
         mfs_npz: str,
         n_surfaces: int = 8,
         band_frac: float = 0.01,
         max_points_per_level: int = 8000):

    # Matplotlib style (tweak as needed)
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.figsize": (6.0, 4.5),
    })

    # 1) Load ψ snapshot and MFS grad_phi
    psi_data = load_psi_snapshot(psi_npz)
    psi3 = psi_data["psi3"]
    grid = psi_data["grid"]
    inside = psi_data["inside"]
    axis_points = psi_data["axis_points"]

    grad_phi, P_surf, N_surf = load_mfs_grad_phi(mfs_npz)

    # 2) Precompute ∇ψ on the grid (Cartesian components)
    if grid["type"] == "cylindrical":
        Rs = grid["Rs"]
        phis = grid["phis"]
        Zs = grid["Zs"]
        psi_x, psi_y, psi_z = compute_grad_psi_cylindrical(psi3, Rs, phis, Zs)
    else:
        xs = grid["xs"]
        ys = grid["ys"]
        zs = grid["zs"]
        psi_x, psi_y, psi_z = compute_grad_psi_cartesian(psi3, xs, ys, zs)

    gx_flat = psi_x.ravel(order="C")
    gy_flat = psi_y.ravel(order="C")
    gz_flat = psi_z.ravel(order="C")

    # 3) Build axis interpolants (for θ,φ coordinates and plotting)
    R_axis_interp, Z_axis_interp = build_axis_interp(axis_points)

    # 4) Choose ψ-levels for surfaces (exclude bands near ψ≈min and ψ≈max)
    psi_flat = psi3.ravel(order="C")
    mask_inside = inside.astype(bool)
    psi_inside = psi_flat[mask_inside]
    psi_min = float(np.min(psi_inside))
    psi_max = float(np.max(psi_inside))

    eps_psi = 0.05 * (psi_max - psi_min)
    psi_levels = np.linspace(psi_min + eps_psi, psi_max - eps_psi, n_surfaces)

    # 5) Sample points in thin ψ-bands, with ∇ψ evaluated at those nodes
    level_data = sample_points_on_psi_levels(
        psi3, grid, inside,
        psi_levels=psi_levels,
        band_frac=band_frac,
        max_points_per_level=max_points_per_level,
        gradpsi_flat=(gx_flat, gy_flat, gz_flat),
    )

    if len(level_data) == 0:
        print("[ERROR] No ψ-levels with valid points; check band_frac and ψ-range.")
        return

    # 6) Build triple-product evaluator
    triple_prod_fn = build_triple_product_fn(grad_phi)

    qs_values = []
    surfaces_for_B_plots: List[Dict[str, Any]] = []

    n_levels_total = len(level_data)
    mid_idx = n_levels_total // 2
    outer_idx = n_levels_total - 1

    for isurf, surf in enumerate(level_data):
        psi0 = surf["psi_level"]
        X = surf["X"]               # (N,3)
        gradpsi = surf["grad_psi"]  # (N,3)

        # Evaluate triple product and |B|
        fT_j, Bmag_j, R_j = triple_prod_fn(X, gradpsi)
        fT = np.asarray(fT_j)
        Bmag = np.asarray(Bmag_j)
        R_vals = np.asarray(R_j)

        # Filter out any pathological points
        mask_good = (
            np.isfinite(fT) &
            np.isfinite(Bmag) &
            (Bmag > 1e-14)
        )
        if mask_good.sum() < 200:
            print(f"[WARN] Not enough good points on ψ≈{psi0:.3f}, skipping.")
            continue

        fT = fT[mask_good]
        Bmag = Bmag[mask_good]
        R_vals = R_vals[mask_good]
        X_good = X[mask_good, :]

        # Geometric angles (φ, θ) using axis
        phi, theta, _ = geometric_angles_from_axis(X_good,
                                                   R_axis_interp,
                                                   Z_axis_interp)

        # Normalized triple-product QS error per surface:
        #   \hat f_T(ψ) = <R>^2 <|f_T|> / <B>^4
        mean_abs_fT = float(np.mean(np.abs(fT)))
        mean_R = float(np.mean(R_vals))
        mean_B = float(np.mean(Bmag))
        if mean_B <= 0.0:
            qs_surf = 0.0
        else:
            qs_surf = (mean_R**2 * mean_abs_fT) / (mean_B**4)

        qs_values.append((psi0, qs_surf))
        print(f"[QS] ψ≈{psi0:.3f}:  hat(f_T) = {qs_surf:.3e}")

        # Store data for |B|(θ, φ) plots on mid and outer surfaces
        if isurf in (mid_idx, outer_idx):
            surfaces_for_B_plots.append({
                "psi_level": psi0,
                "phi": phi,
                "theta": theta,
                "Bmag": Bmag,
            })

    if len(qs_values) == 0:
        print("[ERROR] No surfaces produced a QS metric; aborting.")
        return

    qs_values = np.array(qs_values)  # (nsurf, 2)
    psi_levels_used = qs_values[:, 0]
    qs_errors = qs_values[:, 1]

    # 7) Global QS metric (simple average over surfaces)
    qs_global = float(np.mean(qs_errors))
    print("==============================================")
    print(f"[QS] Global triple-product metric (mean hat f_T) = {qs_global:.5e}")
    print("==============================================")

    # 8) Publication-style plots
    base = psi_npz.replace(".npz", "_QS_triple")

    # (a) |B|(θ, φ) on chosen surfaces
    if len(surfaces_for_B_plots) > 0:
        nfig = len(surfaces_for_B_plots)
        fig, axes = plt.subplots(1, nfig, figsize=(6.0 * nfig, 4.5),
                                 constrained_layout=True)
        if nfig == 1:
            axes = [axes]

        for ax, surf_plot in zip(axes, surfaces_for_B_plots):
            phi = np.mod(surf_plot["phi"], 2*np.pi)
            theta = np.mod(surf_plot["theta"], 2*np.pi)
            Bmag = surf_plot["Bmag"]
            psi0 = surf_plot["psi_level"]

            triang = Triangulation(phi, theta)
            tcf = ax.tricontourf(triang, Bmag, levels=40)
            cbar = fig.colorbar(tcf, ax=ax)
            cbar.set_label(r"$|B|$")

            ax.set_xlabel(r"$\varphi$")
            ax.set_ylabel(r"$\theta$")
            ax.set_title(rf"$|B|(\theta,\varphi)$ on $\psi \approx {psi0:.3f}$")

        fig.suptitle(r"Magnetic field strength on selected flux surfaces",
                     y=1.02)
        out_B = base + "_Bmaps.png"
        fig.savefig(out_B, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Saved |B|(θ,φ) maps to {out_B}")

    # (b) hat(f_T) vs normalized ψ
    psi_norm = (psi_levels_used - psi_min) / (psi_max - psi_min)
    fig2, ax2 = plt.subplots()
    ax2.plot(psi_norm, qs_errors, "o-", lw=1.5)
    ax2.set_xlabel(r"Normalized flux $(\psi - \psi_\mathrm{min}) / (\psi_\mathrm{max} - \psi_\mathrm{min})$")
    ax2.set_ylabel(r"$\hat f_T(\psi)$")
    ax2.set_title("Quasisymmetry triple-product diagnostic")
    ax2.grid(True, alpha=0.3)
    out_prof = base + "_profile.png"
    fig2.savefig(out_prof, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved triple-product profile to {out_prof}")

    # 9) Save numerical data
    out_name = psi_npz.replace(".npz", "_QS_triple_product.npz")
    np.savez(
        out_name,
        psi_levels=psi_levels_used,
        qs_errors=qs_errors,
        qs_global=qs_global,
        psi_min=psi_min,
        psi_max=psi_max,
    )
    print(f"[SAVE] Saved QS triple-product data to {out_name}")

if __name__ == "__main__":
    default_solution = "wout_precise_QA_solution.npz"
    # default_solution = "wout_precise_QH_solution.npz"
    # default_solution = "wout_SLAM_4_coils_solution.npz"
    # default_solution = "wout_SLAM_6_coils_solution.npz"
    # default_solution = "knot_tube_solution.npz"

    default_psi_npz = default_solution.replace(".npz", "_psi_fci_cyl_N64_Nphi128.npz")

    parser = argparse.ArgumentParser(
        description="Quasisymmetry triple-product metric from ψ and MFS field."
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
        help="psi_fci snapshot (.npz)",
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
        help="Maximum number of points per ψ level used in QS diagnostic",
    )
    args = parser.parse_args()

    main(
        psi_npz=args.psi_npz,
        mfs_npz=args.mfs_npz,
        n_surfaces=args.n_surfaces,
        band_frac=args.band_frac,
        max_points_per_level=args.max_points_per_level,
    )
