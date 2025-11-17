#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quasisymmetry metric from flux-function ψ and MFS magnetic field.

This script:
  1) Loads ψ(𝐱) from a psi_fci snapshot (.npz).
  2) Loads the MFS solution checkpoint (center, scale, Yn, alpha, a, a_hat, P, N).
  3) Rebuilds grad_phi(𝐱) (so B = ∇φ) using the same Evaluators pattern.
  4) For several ψ-levels (flux surfaces), samples points in a thin ψ-band.
  5) For each point, computes geometric angles (φ, θ) using the magnetic axis:
        φ = atan2(y, x)
        θ = poloidal angle in (R,Z) around the axis position at that φ.
  6) Fits a Fourier series B(θ, φ) on each surface via least-squares.
  7) Computes a QA-style QS metric: fraction of Fourier power in n≠0 modes.
  8) Aggregates to a global metric and prints/saves it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap

from scipy.interpolate import interp1d
from scipy.interpolate import griddata        ### NEW

import matplotlib.pyplot as plt               ### NEW

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
# Sampling points on flux surfaces (ψ bands)
# ---------------------------------------------------------------------------

def sample_points_on_psi_levels(psi3: np.ndarray,
                                grid: Dict[str, Any],
                                inside: np.ndarray,
                                psi_levels: np.ndarray,
                                band_frac: float = 0.01,
                                max_points_per_level: int = 8000):
    """
    For each ψ_level, build a thin band:
        |ψ - ψ_level| < band_width
    where band_width = band_frac * (ψ_max - ψ_min).

    Returns:
      level_data: list of dicts with keys:
        "psi_level": float
        "X": (Ni,3) points in Cartesian coords
        "psi": (Ni,) local ψ
    """
    psi_flat = psi3.ravel(order="C")
    mask_inside = inside.astype(bool)
    psi_inside = psi_flat[mask_inside]

    psi_min = float(np.min(psi_inside))
    psi_max = float(np.max(psi_inside))
    band_width = band_frac * (psi_max - psi_min)

    level_data = []

    if grid["type"] == "cylindrical":
        Rs = grid["Rs"]
        phis = grid["phis"]
        Zs = grid["Zs"]
        nR, nphi, nZ = grid["shape"]

        RR, PHI, ZZ = np.meshgrid(Rs, phis, Zs, indexing="ij")  # (nR,nphi,nZ)
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
        level_data.append({
            "psi_level": float(psi0),
            "X": X_band,
            "psi": psi_band,
        })

    return level_data

# ---------------------------------------------------------------------------
# Fourier fit of B(θ, φ) on a flux surface
# ---------------------------------------------------------------------------

def build_fourier_design(theta: np.ndarray,
                         phi: np.ndarray,
                         m_max: int,
                         n_max: int):
    """
    Build design matrix A for Fourier expansion:

      B ≈ Σ_{m=-m_max..m_max} Σ_{n=-n_max..n_max}
           C_{mn} cos(m θ + n φ) + S_{mn} sin(m θ + n φ)

    We index coefficients in a 2D (m_idx, n_idx) grid, but store them flattened.

    Returns:
      A: (N_points, 2 * n_modes)
      mode_list: list of (m,n) pairs corresponding to columns.
    """
    theta = theta[:, None]  # (N,1)
    phi = phi[:, None]      # (N,1)

    modes = []
    cols_cos = []
    cols_sin = []

    for m in range(-m_max, m_max + 1):
        for n in range(-n_max, n_max + 1):
            arg = m*theta + n*phi
            cols_cos.append(np.cos(arg))
            cols_sin.append(np.sin(arg))
            modes.append((m, n))

    # concatenate along columns: [cos(m1n1) ... cos(mK nK) | sin(m1 n1) ...]
    A_cos = np.concatenate(cols_cos, axis=1)   # (N, K)
    A_sin = np.concatenate(cols_sin, axis=1)   # (N, K)
    A = np.concatenate([A_cos, A_sin], axis=1) # (N, 2K)

    return A, modes

def fit_fourier_B(theta: np.ndarray,
                  phi: np.ndarray,
                  Bvals: np.ndarray,
                  m_max: int,
                  n_max: int):
    """
    Least-squares fit of Fourier series for B(θ, φ) on one flux surface.
    Returns:
      C: (K,) cos coefficients
      S: (K,) sin coefficients
      modes: list of (m,n) with length K
    """
    A, modes = build_fourier_design(theta, phi, m_max, n_max)
    # regularized least squares to avoid pathological ill-conditioning
    lam = 1e-10
    ATA = A.T @ A + lam * np.eye(A.shape[1])
    ATb = A.T @ Bvals
    coeff = np.linalg.solve(ATA, ATb)  # (2K,)

    K = len(modes)
    C = coeff[:K]
    S = coeff[K:]
    return C, S, modes

# ---------------------------------------------------------------------------
# QS metrics (QA and QH-style)
# ---------------------------------------------------------------------------

def qs_metric_QA(C: np.ndarray, S: np.ndarray, modes) -> float:
    """
    QA-style metric: only modes with n=0 are "QS-allowed".
    Return fraction of Fourier power in n≠0 modes.
    """
    power_all = 0.0
    power_qs = 0.0
    for k, (m, n) in enumerate(modes):
        amp2 = C[k]**2 + S[k]**2
        power_all += amp2
        if n == 0:
            power_qs += amp2
    if power_all == 0.0:
        return 0.0
    return float((power_all - power_qs) / power_all)

def qs_metric_QH(C: np.ndarray, S: np.ndarray, modes,
                 helicity_M: int, helicity_N: int) -> float:
    """
    QH-style metric: "QS-allowed" modes satisfy N*m + M*n = 0 in Boozer coords.
    For a simple choice, set M = 1, N = Nfp (field periods).

    This is approximate in geometric coordinates.
    """
    power_all = 0.0
    power_qs = 0.0
    for k, (m, n) in enumerate(modes):
        amp2 = C[k]**2 + S[k]**2
        power_all += amp2
        if (helicity_N * m + helicity_M * n) == 0:
            power_qs += amp2
    if power_all == 0.0:
        return 0.0
    return float((power_all - power_qs) / power_all)

# ---------------------------------------------------------------------------
# Helper for B(θ,φ) grid and plots                ### NEW
# ---------------------------------------------------------------------------

def make_Btheta_phi_grid(theta, phi, Bmag,
                         n_theta=64, n_phi=128):
    """
    Interpolate scattered (theta,phi,B) onto a regular grid for plotting.
    All angles mapped into [0, 2π).
    """
    theta_mod = np.mod(theta, 2.0 * np.pi)
    phi_mod = np.mod(phi, 2.0 * np.pi)

    # Regular grid
    theta_grid = np.linspace(0.0, 2.0*np.pi, n_theta, endpoint=False)
    phi_grid = np.linspace(0.0, 2.0*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta_grid, phi_grid, indexing="ij")  # (n_theta,n_phi)

    # Interpolate using griddata
    pts = np.column_stack([theta_mod, phi_mod])
    grid_pts = np.column_stack([TH.ravel(), PH.ravel()])

    B_grid = griddata(pts, Bmag, grid_pts, method="linear")
    B_grid = B_grid.reshape(TH.shape)

    # Fill NaNs with nearest-neighbor interpolation
    if np.any(~np.isfinite(B_grid)):
        B_nn = griddata(pts, Bmag, grid_pts, method="nearest").reshape(TH.shape)
        mask_bad = ~np.isfinite(B_grid)
        B_grid[mask_bad] = B_nn[mask_bad]

    return theta_grid, phi_grid, B_grid

# ---------------------------------------------------------------------------
# Main: put it all together
# ---------------------------------------------------------------------------

def main(psi_npz: str,
         mfs_npz: str,
         n_surfaces: int = 8,
         band_frac: float = 0.01,
         m_max: int = 4,
         n_max: int = 4,
         qs_type: str = "QA",
         nfp: int = 2):

    # 1) Load ψ snapshot and MFS grad_phi
    psi_data = load_psi_snapshot(psi_npz)
    psi3 = psi_data["psi3"]
    grid = psi_data["grid"]
    inside = psi_data["inside"]
    boundary_band = psi_data["boundary_band"]
    axis_points = psi_data["axis_points"]

    grad_phi, P_surf, N_surf = load_mfs_grad_phi(mfs_npz)

    # 2) Build axis interpolants
    R_axis_interp, Z_axis_interp = build_axis_interp(axis_points)

    # 3) Choose ψ-levels for surfaces (exclude bands near ψ≈0 and ψ≈1)
    psi_flat = psi3.ravel(order="C")
    mask_inside = inside.astype(bool)
    psi_inside = psi_flat[mask_inside]
    psi_min = float(np.min(psi_inside))
    psi_max = float(np.max(psi_inside))

    # Avoid extreme axis/boundary bands
    eps_psi = 0.05 * (psi_max - psi_min)
    psi_levels = np.linspace(psi_min + eps_psi, psi_max - eps_psi, n_surfaces)

    # 4) Sample points in thin ψ-bands
    level_data = sample_points_on_psi_levels(
        psi3, grid, inside,
        psi_levels=psi_levels,
        band_frac=band_frac,
        max_points_per_level=8000,
    )

    if len(level_data) == 0:
        print("[ERROR] No ψ-levels with valid points; check band_frac and ψ-range.")
        return

    # 5) Evaluate B = ∇φ and build angles, then Fourier-fit per surface
    qs_values = []
    plot_data = []     ### NEW: store for plotting

    for surf in level_data:
        psi0 = surf["psi_level"]
        X = surf["X"]   # (N,3)

        # Evaluate B via JAX grad_phi in batch
        X_j = jnp.asarray(X, dtype=jnp.float64)
        B = np.asarray(grad_phi(X_j))  # (N,3)

        Bmag = np.linalg.norm(B, axis=1)
        # Filter out weird points
        mask = np.isfinite(Bmag) & (Bmag > 1e-12)
        if mask.sum() < 200:
            print(f"[WARN] Not enough good points on ψ≈{psi0:.3f}, skipping.")
            continue

        X_good = X[mask, :]
        Bmag_good = Bmag[mask]

        # Geometric angles
        phi, theta, R = geometric_angles_from_axis(X_good, R_axis_interp, Z_axis_interp)

        # Fourier fit
        C, S, modes = fit_fourier_B(theta, phi, Bmag_good, m_max=m_max, n_max=n_max)

        # QS metric
        if qs_type.upper() == "QA":
            q_surf = qs_metric_QA(C, S, modes)
        elif qs_type.upper() == "QH":
            # For QH with field-period Nfp, typical choice is helicity (M=1, N=nfp)
            q_surf = qs_metric_QH(C, S, modes, helicity_M=1, helicity_N=nfp)
        else:
            raise ValueError(f"Unknown qs_type={qs_type}; use 'QA' or 'QH'.")

        qs_values.append((psi0, q_surf))
        print(f"[QS] ψ≈{psi0:.3f}: QS error = {q_surf:.3e}")

        # Store for plotting
        plot_data.append({
            "psi_level": psi0,
            "theta": theta,
            "phi": phi,
            "Bmag": Bmag_good,
        })

    if len(qs_values) == 0:
        print("[ERROR] No surfaces produced a QS metric; aborting.")
        return

    qs_values = np.array(qs_values)  # (nsurf, 2)
    psi_levels_used = qs_values[:, 0]
    qs_errors = qs_values[:, 1]

    # 6) Aggregate to a single global QS metric (simple average for now)
    qs_global = float(np.mean(qs_errors))
    print("=========================================")
    print(f"[QS] Global {qs_type} metric (mean over surfaces) = {qs_global:.5e}")
    print("=========================================")

    # Save for post-processing
    out_name = psi_npz.replace(".npz", f"_QS_{qs_type.upper()}.npz")
    np.savez(
        out_name,
        psi_levels=psi_levels_used,
        qs_errors=qs_errors,
        qs_global=qs_global,
        qs_type=qs_type,
        m_max=m_max,
        n_max=n_max,
    )
    print(f"[SAVE] Saved QS metric data to {out_name}")

    # 7) Plot |B|(θ,φ) on inner-radius and outer surfaces     ### NEW
    if len(plot_data) >= 2:
        # Sort by psi_level (inner to outer)
        plot_data_sorted = sorted(plot_data, key=lambda d: d["psi_level"])
        # inner-radius: middle index
        mid_idx = len(plot_data_sorted) // 4
        # boundary-ish: outermost surface
        outer_idx = len(plot_data_sorted) - 1

        surfaces_to_plot = [
            ("inner-radius", plot_data_sorted[mid_idx]),
            ("outer", plot_data_sorted[outer_idx]),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        if len(axes.shape) == 0:  # just in case matplotlib returns single Axes
            axes = np.array([axes])

        for ax, (label, sdata) in zip(axes, surfaces_to_plot):
            theta = sdata["theta"]
            phi = sdata["phi"]
            Bmag = sdata["Bmag"]
            psi0 = sdata["psi_level"]

            theta_grid, phi_grid, B_grid = make_Btheta_phi_grid(theta, phi, Bmag)

            cf = ax.contourf(phi_grid, theta_grid, B_grid, levels=40)
            fig.colorbar(cf, ax=ax)

            ax.set_xlabel(r"$\phi$ (geom)")
            ax.set_ylabel(r"$\theta$ (geom)")
            ax.set_title(rf"$|B|(\theta,\phi)$, {label}, $\psi \approx {psi0:.3f}$")

        # Save figure
        png_name = psi_npz.replace(".npz", f"_Btheta_phi_contours.png")
        fig.suptitle("|B|(θ,φ) contours on selected flux surfaces")
        fig.savefig(png_name, dpi=200)
        plt.close(fig)
        print(f"[PLOT] Saved |B|(θ,φ) contour plot to {png_name}")
    else:
        print("[PLOT] Not enough surfaces for B(theta,phi) plotting; skipping.")

if __name__ == "__main__":
    default_solution = "wout_precise_QA_solution.npz"
    # default_solution = "wout_precise_QH_solution.npz"
    # default_solution = "wout_SLAM_4_coils_solution.npz"
    # default_solution = "wout_SLAM_6_coils_solution.npz"
    # default_solution = "knot_tube_solution.npz"
    
    default_psi_npz = default_solution.replace(".npz", "_psi_fci_cyl_N64_Nphi128.npz")

    nfp_default = 2
    if 'QH' in default_solution:
        nfp_default = 4

    parser = argparse.ArgumentParser(description="Quasisymmetry metric from ψ and MFS field.")
    parser.add_argument("mfs_npz", nargs="?", default=resolve_npz_file_location(default_solution),
                        help="MFS solution checkpoint (.npz) to rebuild grad_phi")
    parser.add_argument("--psi-npz", default=default_psi_npz, help="psi_fci snapshot (.npz)")
    parser.add_argument("--n-surfaces", type=int, default=8, help="Number of ψ surfaces to sample")
    parser.add_argument("--band-frac", type=float, default=0.01,
                        help="Relative ψ-band half-width around each level")
    parser.add_argument("--m-max", type=int, default=4, help="Maximum poloidal mode index |m|")
    parser.add_argument("--n-max", type=int, default=4, help="Maximum toroidal mode index |n|")
    parser.add_argument("--qs-type", choices=["QA", "QH"], default="QA",
                        help="QS flavor: QA or QH")
    parser.add_argument("--nfp", type=int, default=2, help="Number of field periods (used for QH)")
    args = parser.parse_args()

    main(args.psi_npz,
         mfs_npz=args.mfs_npz,
         n_surfaces=args.n_surfaces,
         band_frac=args.band_frac,
         m_max=args.m_max,
         n_max=args.n_max,
         qs_type=args.qs_type,
         nfp=args.nfp)