#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze FCI flux function ψ along field lines and compare with Poincaré plots.

Workflow:
  1) Solve for ψ using solve_flux_psi_fci_cyl.py, which produces a file like
       wout_precise_QA_solution_psi_fci_cyl_N64_Nphi128.npz
  2) Run this analysis script:

       python analyze_psi_along_fieldlines.py \
           wout_precise_QA_solution.npz \
           wout_precise_QA_solution_psi_fci_cyl_N64_Nphi128.npz \
           --tfinal 800 --n-save 2000 --nseed 12 --poincare-nphi 4 --save-figures

This script:
  * Rebuilds ∇φ from the MFS solution checkpoint.
  * Traces field lines from a set of seeds.
  * Interpolates ψ along those lines and plots ψ(s).
  * Builds Poincaré R–Z plots overlaying ψ(R,Z) contours on geometric flux surfaces.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from fractions import Fraction
import pyvista as pv

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax as dfx
from jax import jit, vmap, lax

from scipy.interpolate import RegularGridInterpolator

# -------------------------- Paths and utils ------------------------- #
script_dir = Path(__file__).resolve().parent

def resolve_npz_file_location(npz_file, subdir="outputs"):
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

# ----------------------------- Styling ----------------------------- #
def apply_paper_style():
    fig_w = 5.5
    mpl.rcParams.update({
        "figure.figsize": (fig_w, fig_w),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "text.usetex": False,
    })
    return fig_w

def phi_label_pi(phi: float, wrap=True, max_den=24) -> str:
    if wrap:
        phi = (phi + np.pi) % (2*np.pi) - np.pi
    r = Fraction(phi / np.pi).limit_denominator(max_den)
    p, q = r.numerator, r.denominator

    def _mul_pi(pp, qq):
        if pp == 0:
            return "0"
        sign = "-" if pp < 0 else ""
        pp = abs(pp)
        if qq == 1:
            coeff = "" if pp == 1 else f"{pp}"
            return f"{sign}{coeff}\\pi"
        else:
            coeff = "" if pp == 1 else f"{pp}"
            return f"{sign}{coeff}\\pi/{qq}"

    return rf"$\phi={_mul_pi(p,q)}$"

# ------------------------ MFS evaluators ------------------------- #

@jit
def _green_G(x, Y):  # x:(3,), Y:(M,3)
    r = jnp.linalg.norm(x[None, :] - Y, axis=1)
    return 1.0 / (4.0 * jnp.pi * jnp.maximum(1e-30, r))

@jit
def _grad_green_x(x, Y):  # -> (M,3)
    r = x[None, :] - Y
    r2 = jnp.sum(r * r, axis=1)
    r3 = jnp.maximum(1e-30, r2 * jnp.sqrt(r2))
    return -r / (4.0 * jnp.pi * r3)[:, None]

def _unit(v, eps=1e-30):
    n = jnp.linalg.norm(v, axis=1, keepdims=True)
    return v / jnp.maximum(eps, n)

def _nearest_normal_jax(Xn, Pn, Nn):
    # brute force nearest neighbor in JAX
    X2 = jnp.sum(Xn * Xn, axis=1, keepdims=True)
    P2 = jnp.sum(Pn * Pn, axis=1, keepdims=True)
    dist2 = X2 + P2.T - 2.0 * (Xn @ Pn.T)
    idx = jnp.argmin(dist2, axis=1)
    return Nn[idx, :]

def _grad_azimuth_about_axis(Xn, a_hat):
    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par = jnp.sum(Xn * a[None, :], axis=1, keepdims=True) * a[None, :]
    r_perp = Xn - r_par
    r2 = jnp.maximum(1e-30, jnp.sum(r_perp * r_perp, axis=1, keepdims=True))
    cr = jnp.cross(a[None, :], r_perp)
    return cr / r2

def _make_mv_grads(a_hat, P, N, center, scale):
    Pn = (P - center[None, :]) * scale
    Nn = N
    a_hat = jnp.asarray(a_hat)

    def grad_t(Xn):   # ∇φ_a, accepts (N,3)
        Xn = Xn.reshape((-1, 3))
        return _grad_azimuth_about_axis(Xn, a_hat)

    def grad_p(Xn):
        Xn = Xn.reshape((-1, 3))
        n = _nearest_normal_jax(Xn, Pn, Nn)
        a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
        rpar = jnp.sum(Xn * a[None, :], axis=1, keepdims=True) * a[None, :]
        rper = Xn - rpar
        phi_hat = _unit(jnp.cross(a[None, :], rper))
        phi_tan = _unit(phi_hat - jnp.sum(phi_hat * n, axis=1, keepdims=True) * n)
        theta_hat = _unit(jnp.cross(n, phi_tan))
        return theta_hat

    return grad_t, grad_p

def load_mfs_grad_phi(mfs_npz_path: str):
    d = np.load(mfs_npz_path, allow_pickle=False)
    center = jnp.asarray(d["center"], dtype=jnp.float64)
    scale  = jnp.asarray(d["scale"].item() if d["scale"].shape == () else float(d["scale"]), dtype=jnp.float64)
    Yn     = jnp.asarray(d["Yn"], dtype=jnp.float64)
    alpha  = jnp.asarray(d["alpha"], dtype=jnp.float64)
    a      = jnp.asarray(d["a"], dtype=jnp.float64)
    a_hat  = jnp.asarray(d["a_hat"], dtype=jnp.float64)
    P      = jnp.asarray(d["P"], dtype=jnp.float64)
    N      = jnp.asarray(d["N"], dtype=jnp.float64)
    kind   = str(d["kind"])

    print(f"[INFO] Loaded MFS checkpoint '{mfs_npz_path}' with {Yn.shape[0]} sources, {P.shape[0]} boundary points (kind={kind}).")

    grad_t, grad_p = _make_mv_grads(a_hat, P, N, center, scale)

    @jit
    def grad_phi_point(x: jnp.ndarray) -> jnp.ndarray:
        xn = (x - center) * scale
        dG = _grad_green_x(xn, Yn)
        grad_s = scale * jnp.sum(dG * alpha[:, None], axis=0)
        xn_b = xn[None, :]
        gt = grad_t(xn_b)[0]
        gp = grad_p(xn_b)[0]
        grad_mv = scale * (a[0] * gt + a[1] * gp)
        return grad_s + grad_mv

    def seeds_from_boundary(nseed: int = 16, eps: float = 1e-3) -> np.ndarray:
        Pb = np.asarray(P); Nb = np.asarray(N)
        Pi = Pb - eps * Nb
        if Pi.shape[0] > nseed:
            stride = max(1, Pi.shape[0] // nseed)
            Pi = Pi[::stride][:nseed]
        return Pi.astype(np.float64)

    return dict(
        grad_phi_point=grad_phi_point,
        seeds_from_boundary=seeds_from_boundary,
        P=np.asarray(P),
        N=np.asarray(N),
        center=np.asarray(center),
        a_hat=np.asarray(a_hat),
        kind=kind,
    )

# --------------------- Simple seeding along axis chord --------------------- #

def _orthonormal_complement(a_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a_hat, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-30)
    t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = t - np.dot(t, a) * a
    e1 /= (np.linalg.norm(e1) + 1e-30)
    e2 = np.cross(a, e1)
    e2 /= (np.linalg.norm(e2) + 1e-30)
    return e1, e2

def seeds_along_axis_from_boundary(
    P: np.ndarray,
    N: np.ndarray,
    center: np.ndarray,
    a_hat: np.ndarray,
    kind: str,
    nseed: int = 16,
    strip_tol_frac: float = 0.03,
    plane_tol_frac: float = 0.10,
    inward_frac: float = 0.02,
) -> np.ndarray:
    P = np.asarray(P); N = np.asarray(N)
    c = np.asarray(center); a = np.asarray(a_hat)
    e1, e2 = _orthonormal_complement(a)

    X = P - c[None, :]
    u1 = X @ e1
    u2 = X @ e2
    s  = X @ (a / (np.linalg.norm(a) + 1e-30))

    u2_span = np.percentile(np.abs(u2), 99.0) + 1e-12
    s_span  = np.percentile(np.abs(s),  99.0) + 1e-12

    u2_tol = strip_tol_frac * u2_span
    if kind.lower().strip() == "torus":
        s_tol = plane_tol_frac * s_span
        mask = (np.abs(u2) <= u2_tol) & (np.abs(s) <= s_tol)
    else:
        mask = (np.abs(u2) <= u2_tol)

    if not np.any(mask):
        print("[WARN] Seed strip empty; using whole cloud for seeding.")
        mask = np.ones_like(u1, dtype=bool)

    u1_sel = u1[mask]
    idx = np.where(mask)[0]
    iL = idx[np.argmin(u1_sel)]
    iR = idx[np.argmax(u1_sel)]
    pL, nL = P[iL], N[iL]
    pR, nR = P[iR], N[iR]

    # small tweak to avoid exactly symmetric seeds
    pL = (pL + pR) / 2.01
    pR = pR * 0.99

    if u1_sel.size >= 8:
        u1_sorted = np.sort(u1_sel)
        du = np.median(np.diff(u1_sorted))
        h_med = max(1e-6, float(du))
    else:
        bb = np.max(P, axis=0) - np.min(P, axis=0)
        h_med = max(1e-6, 0.01 * float(np.linalg.norm(bb)))

    eps = inward_frac * h_med
    tau = np.linspace(0.0, 1.0, max(2, nseed))
    chord = (1.0 - tau)[:, None] * pL[None, :] + tau[:, None] * pR[None, :]

    def _nearest(i):
        d2 = np.sum((P - chord[i])**2, axis=1)
        j = int(np.argmin(d2))
        return N[j]

    normals = np.stack([_nearest(i) for i in range(chord.shape[0])], axis=0)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-30)

    seeds = chord - eps * normals
    print(f"[SEEDS] Created {seeds.shape[0]} seeds along chord.")
    return seeds.astype(np.float64)

# ------------------------- RHS & integrators ------------------------- #

def make_rhs(grad_phi_point: Callable[[jnp.ndarray], jnp.ndarray],
             normalize: bool = False,
             clip_grad: Optional[float] = None):
    @jax.jit
    def f(t, y, args):
        g = grad_phi_point(y)
        if normalize:
            n = jnp.linalg.norm(g) + 1e-12
            g = g / n
        if (clip_grad is not None) and (clip_grad > 0.0):
            n = jnp.linalg.norm(g) + 1e-12
            g = jnp.where(n > clip_grad, g * (clip_grad / n), g)
        return g
    return f

def integrate_streamlines(
    seeds: np.ndarray,
    f,
    t_final: float = 10.0,
    n_save: int = 2001,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray]:
    seeds_j = jnp.asarray(seeds, dtype=jnp.float64)
    ts = jnp.linspace(0.0, float(t_final), int(n_save), dtype=jnp.float64)

    solver = dfx.Tsit5()
    term = dfx.ODETerm(f)
    stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)
    saveat = dfx.SaveAt(ts=ts)

    def _solve_one(y0):
        sol = dfx.diffeqsolve(
            term, solver,
            t0=0.0, t1=float(t_final), dt0=float(t_final) / 1024.0,
            y0=y0,
            saveat=saveat,
            max_steps=200_000,
            stepsize_controller=stepsize_controller,
        )
        return sol.ys

    solve_vmap = jax.jit(vmap(_solve_one))

    print(f"[INTEGRATION] Integrating {seeds.shape[0]} field lines up to t_final={t_final}, n_save={n_save}...")
    ys = solve_vmap(seeds_j)      # (S, T, 3)
    print(f"[INTEGRATION] Done. Field-line array shape: {ys.shape}")
    return np.asarray(ts), np.asarray(ys)

# ------------------------- Poincaré machinery ------------------------- #

def _angle_wrap_jnp(a):
    return (a + jnp.pi) % (2*jnp.pi) - jnp.pi

def _wrap_diff_jnp(a_minus_b):
    return _angle_wrap_jnp(a_minus_b)

@jax.jit
def poincare_RZ_points_jax_dense(Y_all: jnp.ndarray, phi0: float):
    # Y_all: (S, T, 3)
    valid = ~jnp.any(jnp.isnan(Y_all), axis=-1)            # (S,T)
    X = Y_all[..., 0]; Y = Y_all[..., 1]; Z = Y_all[..., 2]
    phi = jnp.arctan2(Y, X)
    dphi = _wrap_diff_jnp(phi - phi0)
    s = jnp.sign(dphi)
    s = jnp.where(s == 0.0, 1.0, s)

    valid_seg = valid[..., :-1] & valid[..., 1:]
    changed   = (s[..., :-1] * s[..., 1:] < 0.0) & valid_seg

    p0 = Y_all[:, :-1, :]
    p1 = Y_all[:,  1:, :]
    d0 = dphi[:, :-1]
    d1 = dphi[:,  1:]
    t = jnp.clip(d0 / (d0 - d1), 0.0, 1.0)
    p = p0 + t[..., None] * (p1 - p0)

    R  = jnp.linalg.norm(p[..., :2], axis=-1)  # (S, T-1)
    Zc = p[..., 2]                             # (S, T-1)

    S, Tm1 = R.shape
    R_flat    = R.reshape(-1)
    Z_flat    = Zc.reshape(-1)
    mask_flat = changed.reshape(-1)

    seed_idx  = jnp.tile(jnp.arange(S)[:, None], (1, Tm1))
    seed_flat = seed_idx.reshape(-1)

    return R_flat, Z_flat, mask_flat, seed_flat

def poincare_multi_phi_jax(Y_all: jnp.ndarray, phis: jnp.ndarray):
    R_flat, Z_flat, M_flat, seed_flat = jax.vmap(
        poincare_RZ_points_jax_dense, in_axes=(None, 0))(Y_all, phis)
    return R_flat, Z_flat, M_flat, seed_flat

# ---------------------- ψ interpolation on the grid ---------------------- #

def build_psi_interpolant(psi_npz_path: str) -> Dict[str, Any]:
    data = np.load(psi_npz_path, allow_pickle=True)
    grid_type = str(data["grid_type"])
    psi_flat  = np.asarray(data["psi"])
    inside    = np.asarray(data["inside"], dtype=bool)

    print(f"[INFO] Loaded ψ-solution '{psi_npz_path}'")
    print(f"[DEBUG] grid_type = {grid_type}")
    print(f"[DEBUG] ψ.shape (flat) = {psi_flat.shape}, inside.sum = {inside.sum()}")

    info: Dict[str, Any] = {
        "grid_type": grid_type,
        "inside": inside,
        "psi_flat": psi_flat,
        "mins": np.asarray(data["mins"]),
        "maxs": np.asarray(data["maxs"]),
        "psi_file": psi_npz_path,
    }

    if grid_type == "cylindrical":
        Rs   = np.asarray(data["Rs"])
        phis = np.asarray(data["phis"])
        Zs   = np.asarray(data["Zs"])
        nR, nphi, nZ = len(Rs), len(phis), len(Zs)
        psi3 = psi_flat.reshape(nR, nphi, nZ)
        info.update(dict(Rs=Rs, phis=phis, Zs=Zs, psi3=psi3))

        print(f"[GRID] Cylindrical: nR={nR}, nphi={nphi}, nZ={nZ}")
        print(f"[GRID] R∈[{Rs.min():.3g},{Rs.max():.3g}], Z∈[{Zs.min():.3g},{Zs.max():.3g}], φ∈[{phis.min():.3g},{phis.max():.3g}]")
        print(f"[STATS] ψ: min={np.nanmin(psi3):.3e}, max={np.nanmax(psi3):.3e}")

        interp = RegularGridInterpolator(
            (Rs, phis, Zs),
            psi3,
            bounds_error=False,
            fill_value=np.nan,
        )

        def psi_eval(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X)
            R = np.sqrt(X[..., 0]**2 + X[..., 1]**2)
            phi = np.arctan2(X[..., 1], X[..., 0])
            # wrap φ into [phis[0], phis[-1]+Δφ)
            dphi = phis[1] - phis[0]
            phi_wrapped = (phi - phis[0]) % (phis[-1] - phis[0] + dphi) + phis[0]
            Z = X[..., 2]
            pts = np.stack([R, phi_wrapped, Z], axis=-1)
            return interp(pts)
    else:
        xs = np.asarray(data["xs"])
        ys = np.asarray(data["ys"])
        zs = np.asarray(data["zs"])
        nx, ny, nz = len(xs), len(ys), len(zs)
        psi3 = psi_flat.reshape(nx, ny, nz)
        info.update(dict(xs=xs, ys=ys, zs=zs, psi3=psi3))

        print(f"[GRID] Cartesian: nx={nx}, ny={ny}, nz={nz}")
        print(f"[GRID] x∈[{xs.min():.3g},{xs.max():.3g}], y∈[{ys.min():.3g},{ys.max():.3g}], z∈[{zs.min():.3g},{zs.max():.3g}]")
        print(f"[STATS] ψ: min={np.nanmin(psi3):.3e}, max={np.nanmax(psi3):.3e}")

        interp = RegularGridInterpolator(
            (xs, ys, zs),
            psi3,
            bounds_error=False,
            fill_value=np.nan,
        )

        def psi_eval(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X)
            return interp(X)

    # Optional: magnetic axis information if present in the NPZ
    for key in ["R_axis", "Z_axis", "axis_points"]:
        if key in data.files:
            info[key] = np.asarray(data[key])

    info["psi_eval"] = psi_eval
    return info

def _set_axes_equal_3d(ax):
    """Set 3D axes to equal scale (so spheres / tori aren't distorted)."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    x_mid = 0.5 * (x_limits[0] + x_limits[1])
    y_mid = 0.5 * (y_limits[0] + y_limits[1])
    z_mid = 0.5 * (z_limits[0] + z_limits[1])

    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])


def plot_3d_flux_surfaces_and_fieldlines(
    psi_info: Dict[str, Any],
    grad_info: Dict[str, Any],
    Y: np.ndarray,
    psi_seed: np.ndarray,
    n_surfaces: int = 1,
    max_pts_per_surface: int = 6000,
    phi_center: float = np.pi,
    delta_phi_boundary: float = 0.8,
    delta_phi_surface_max: float = 0.6,
    delta_phi_fieldline: Optional[float] = None,
):
    """
    3D visualization using PyVista:

      * boundary (P from MFS checkpoint)
      * one or several ψ≈const surfaces (isosurfaces from the cylindrical ψ grid)
      * field lines (Y: (S,T,3))

    Cutaway:
      - We "open" the geometry by removing points whose cylindrical angle φ lies
        within a window of width delta_phi around phi_center.
      - The boundary uses delta_phi_boundary (largest opening).
      - Each ψ-surface uses a smaller opening (delta_phi_surface_max decreasing
        towards the core).
      - Field lines are given the *same* opening as the ψ-surface whose ψ-level
        is closest to their seed ψ (optionally scaled by delta_phi_fieldline).
      - If axis_points are present in psi_info, the magnetic axis is shown.
    """
    import pyvista as pv

    grid_type = psi_info["grid_type"]
    if grid_type != "cylindrical":
        print("[3D] ψ-grid is not cylindrical; skipping 3D isosurface plot.")
        return

    # ------------------------------------------------------------------ #
    # Angle utilities
    # ------------------------------------------------------------------ #

    def angular_distance(phi: np.ndarray, center: float) -> np.ndarray:
        """Signed minimal angular distance from 'center' on the circle."""
        # Result in (-π, π]
        return np.angle(np.exp(1j * (phi - center)))

    def angular_mask(phi: np.ndarray, center: float, delta_phi: float) -> np.ndarray:
        """
        Keep points whose angle φ is OUTSIDE a window of width delta_phi
        centered at 'center'.

        If delta_phi <= 0, no cut is applied (all True).
        """
        if delta_phi is None or delta_phi <= 0.0:
            return np.ones_like(phi, dtype=bool)
        d = np.abs(angular_distance(phi, center))
        return d > 0.5 * delta_phi

    # ------------------------------------------------------------------ #
    # Basic ψ info
    # ------------------------------------------------------------------ #

    psi3 = psi_info["psi3"]                     # shape (nR, nphi, nZ)
    inside3 = psi_info["inside"].reshape(psi3.shape)
    Rs = psi_info["Rs"]
    phis_grid = psi_info["phis"]
    Zs = psi_info["Zs"]

    psi_inside = psi3[inside3 & np.isfinite(psi3)]
    if psi_inside.size == 0:
        print("[3D] No valid interior ψ values; skipping 3D view.")
        return

    psi_min = float(np.nanmin(psi_inside))
    psi_max = float(np.nanmax(psi_inside))

    margin = 0.05 * (psi_max - psi_min)
    psi_lo = psi_min + margin
    psi_hi = psi_max - margin

    psi_seed_valid = psi_seed[np.isfinite(psi_seed)]
    if psi_seed_valid.size > 0:
        psi_seed_clip = np.clip(psi_seed_valid, psi_lo, psi_hi)
    else:
        psi_seed_clip = np.array([])

    # ------------------------------------------------------------------ #
    # Choose ψ levels for isosurfaces
    # ------------------------------------------------------------------ #

    if n_surfaces <= 0:
        print("[3D] n_surfaces <= 0; nothing to show.")
        return

    if n_surfaces == 1:
        # Single surface: choose one close to the boundary
        psi_levels = np.array([psi_hi], dtype=float)
    else:
        if psi_seed_clip.size >= 2:
            psi_levels = np.linspace(psi_seed_clip.min(), psi_seed_clip.max(), n_surfaces)
        else:
            psi_levels = np.linspace(psi_lo, psi_hi, n_surfaces)

    print("[3D] ψ levels for isosurfaces:", psi_levels)

    # ------------------------------------------------------------------ #
    # Δφ per surface: from large (outer) to smaller (inner)
    # ------------------------------------------------------------------ #

    if n_surfaces == 1:
        delta_phi_surfaces = np.array([delta_phi_surface_max], dtype=float)
    else:
        # Inner surface: smallest opening; outer surface: largest opening
        delta_phi_surfaces = np.linspace(
            0.3 * delta_phi_surface_max,  # inner
            delta_phi_surface_max,        # outer
            n_surfaces,
        )

    # ------------------------------------------------------------------ #
    # Build structured grid in (x,y,z) for ψ
    # ------------------------------------------------------------------ #

    R3, PHI3, Z3 = np.meshgrid(Rs, phis_grid, Zs, indexing="ij")  # (nR, nphi, nZ)
    X3 = R3 * np.cos(PHI3)
    Y3c = R3 * np.sin(PHI3)
    Z3c = Z3

    grid = pv.StructuredGrid(
        X3.astype(np.float64),
        Y3c.astype(np.float64),
        Z3c.astype(np.float64),
    )

    psi_scalar = np.where(inside3 & np.isfinite(psi3), psi3, np.nan)
    grid["psi"] = psi_scalar.ravel(order="F")

    # ------------------------------------------------------------------ #
    # Boundary from MFS (with a big opening)
    # ------------------------------------------------------------------ #

    P = grad_info["P"]  # (Nb, 3)
    boundary = pv.PolyData(P.astype(np.float64))

    phi_b = np.arctan2(P[:, 1], P[:, 0])
    mask_b = angular_mask(phi_b, phi_center, delta_phi_boundary)
    if np.any(mask_b):
        boundary_cut = boundary.extract_points(np.where(mask_b)[0])
    else:
        print("[3D] Angular mask removed all boundary points; showing full boundary.")
        boundary_cut = boundary

    # ------------------------------------------------------------------ #
    # PyVista plotter (larger window)
    # ------------------------------------------------------------------ #

    pl = pv.Plotter(window_size=(1400, 900))

    # Boundary as semi-transparent blue-ish points
    pl.add_mesh(
        boundary_cut,
        color="#51a0d8",
        opacity=0.95,
        point_size=6.0,
        render_points_as_spheres=True,
    )

    # Color palette for surfaces
    surface_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # grey
        "#bcbd22",  # yellow-green
        "#17becf",  # teal
    ]

    # ------------------------------------------------------------------ #
    # ψ≈const surfaces with angular cuts
    # ------------------------------------------------------------------ #

    for i, lev in enumerate(psi_levels):
        try:
            surf = grid.contour(isosurfaces=[float(lev)], scalars="psi")
        except Exception as e:
            print(f"[3D] contour failed at level {lev}: {e}")
            continue

        if surf.n_points == 0:
            print(f"[3D] No points in isosurface at ψ={lev:.3e}")
            continue

        pts = np.asarray(surf.points)
        phi_s = np.arctan2(pts[:, 1], pts[:, 0])
        delta_phi_here = float(delta_phi_surfaces[i])

        mask_s = angular_mask(phi_s, phi_center, delta_phi_here)
        if np.any(mask_s):
            surf = surf.extract_points(np.where(mask_s)[0])
        else:
            print(f"[3D] Angular mask removed all points for ψ={lev:.3e}; skipping.")
            continue

        if surf.n_points > max_pts_per_surface:
            surf = surf.extract_points(
                np.random.choice(surf.n_points, size=max_pts_per_surface, replace=False)
            )

        color = surface_colors[i % len(surface_colors)]
        pl.add_mesh(
            surf,
            color=color,
            opacity=0.7,
            show_scalar_bar=False,
        )

    # ------------------------------------------------------------------ #
    # Magnetic axis (if present) – full, not cut
    # ------------------------------------------------------------------ #

    axis_pts = psi_info.get("axis_points", None)
    if axis_pts is not None:
        axis_pts = np.asarray(axis_pts)
        if axis_pts.ndim == 2 and axis_pts.shape[1] == 3:
            axis_line = pv.lines_from_points(axis_pts.astype(np.float64))
            pl.add_mesh(axis_line, color="red", line_width=4.0)
        else:
            print("[3D] axis_points present but not of shape (N,3); skipping axis plot.")

    # ------------------------------------------------------------------ #
    # Field lines (black) with per-surface angular cuts
    # ------------------------------------------------------------------ #

    S, T, _ = Y.shape

    # Map each field line to nearest ψ-level, then use that surface's Δφ
    fieldline_delta_phi = np.zeros(S, dtype=float)
    fieldline_has_surface = np.zeros(S, dtype=bool)

    if psi_levels.size > 0:
        psi_min_lvl = float(psi_levels.min())
        psi_max_lvl = float(psi_levels.max())
        # small tolerance in ψ-space, relative to span
        tol_psi = 1e-3 * max(1.0, psi_max_lvl - psi_min_lvl)

        for i in range(S):
            psi_i = psi_seed[i] if i < len(psi_seed) else np.nan
            if not np.isfinite(psi_i):
                # NaN or outside ψ grid: no associated surface
                continue

            # Only keep lines whose seed ψ lies in the range spanned by
            # the plotted surfaces. Lines near the boundary (ψ above psi_max_lvl)
            # or outside the core (ψ below psi_min_lvl) are dropped.
            if (psi_i < psi_min_lvl - tol_psi) or (psi_i > psi_max_lvl + tol_psi):
                # This is typically a boundary-hugging field line → skip it
                continue

            j = int(np.argmin(np.abs(psi_i - psi_levels)))
            j = max(0, min(j, n_surfaces - 1))
            base = float(delta_phi_surfaces[j])

            # If delta_phi_fieldline is provided, treat it as a multiplier
            if delta_phi_fieldline is not None:
                # Normalise by outermost opening so delta_phi_fieldline≈1 keeps base
                scale = float(delta_phi_fieldline) / float(delta_phi_surface_max)
                fieldline_delta_phi[i] = base * scale
            else:
                fieldline_delta_phi[i] = base

            fieldline_has_surface[i] = True

        n_valid = int(fieldline_has_surface.sum())
        print(f"[3D] Field lines with ψ in surface range: {n_valid}/{S}")

    for i in range(S):
        # Skip lines that do not have a matching ψ-surface
        if not fieldline_has_surface[i]:
            continue

        Yi = Y[i, :, :]
        if not np.any(np.isfinite(Yi)):
            continue

        phi_line = np.arctan2(Yi[:, 1], Yi[:, 0])
        delta_phi_line = float(fieldline_delta_phi[i])

        # Safety: if somehow Δφ ended up non-positive, just skip
        if delta_phi_line <= 0.0:
            continue

        mask_line = angular_mask(phi_line, phi_center, delta_phi_line)

        if not np.any(mask_line):
            continue

        idx = np.where(mask_line)[0]

        # Split into contiguous segments so we don't connect across the opening
        splits = np.where(np.diff(idx) > 1)[0]
        start = 0
        segments = []
        for s in splits:
            segments.append(idx[start : s + 1])
            start = s + 1
        segments.append(idx[start:])

        for seg in segments:
            if seg.size < 2:
                continue
            pts_seg = Yi[seg, :]
            line = pv.lines_from_points(pts_seg.astype(np.float64))
            pl.add_mesh(line, color="black", line_width=0.7, opacity=0.4)

    return pl

# ---------------------------- Main analysis ---------------------------- #

def analyze(
    mfs_npz: str,
    psi_npz: str,
    nseed: int = 16,
    tfinal: float = 800.0,
    n_save: int = 2000,
    normalize: bool = False,
    clip_grad: Optional[float] = None,
    poincare_nphi: int = 4,
    nfp: int = 2,
    seeds_str: Optional[str] = None,
    save_figures: bool = True,
):

    apply_paper_style()

    mfs_npz = resolve_npz_file_location(mfs_npz, subdir="outputs")
    psi_npz = resolve_npz_file_location(psi_npz, subdir="outputs")

    psi_info = build_psi_interpolant(psi_npz)
    grad_info = load_mfs_grad_phi(mfs_npz)

    grad_phi_point = grad_info["grad_phi_point"]
    P = grad_info["P"]
    N = grad_info["N"]
    center = grad_info["center"]
    a_hat = grad_info["a_hat"]
    kind = grad_info["kind"]

    # ---------- Seeds ----------
    seeds: np.ndarray
    if seeds_str is not None:
        user_seeds: List[Tuple[float, float, float]] = []
        for item in seeds_str.split(","):
            xyz = tuple(float(v) for v in item.split(":"))
            if len(xyz) == 3:
                user_seeds.append(xyz)
        seeds = np.asarray(user_seeds, dtype=np.float64)
        print(f"[SEEDS] Using user-provided seeds ({seeds.shape[0]} points):\n{seeds}")
    else:
        # More "axis-aligned" seeds by default
        seeds = seeds_along_axis_from_boundary(
            P=P, N=N, center=center, a_hat=a_hat, kind=kind,
            nseed=nseed, strip_tol_frac=0.03, plane_tol_frac=0.10, inward_frac=0.02,
        )

    # ---------- Integrate field lines ----------
    f = make_rhs(grad_phi_point, normalize=normalize, clip_grad=clip_grad)
    ts, Y = integrate_streamlines(seeds, f, t_final=tfinal, n_save=n_save)
    S, T, _ = Y.shape

    # ---------- Compute s and ψ(s) ----------
    psi_eval = psi_info["psi_eval"]
    print("[DEBUG] Interpolating ψ along field lines...")
    X_flat = Y.reshape(-1, 3)
    psi_flat = psi_eval(X_flat)
    psi_lines = psi_flat.reshape(S, T)
    print(f"[DEBUG] ψ_lines shape: {psi_lines.shape}")

    print(f"[STATS] ψ along lines: global min={np.nanmin(psi_lines):.3e}, max={np.nanmax(psi_lines):.3e}")

    # ψ values at the initial seed positions (t = 0 along each line)
    psi_seed = psi_lines[:, 0]
    psi_seed_valid = psi_seed[np.isfinite(psi_seed)]
    print("[SEEDS] ψ at seed positions:", psi_seed_valid)

    # arclength s along each line
    s_lines = np.zeros_like(psi_lines)
    for i in range(S):
        Xi = Y[i]  # (T,3)
        dX = Xi[1:, :] - Xi[:-1, :]
        ds = np.linalg.norm(dX, axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        s_lines[i, :] = s

    print(f"[STATS] ψ along lines: global min={np.nanmin(psi_lines):.3e}, max={np.nanmax(psi_lines):.3e}")

    # ---------- Diagnostics: how constant is ψ along each field line? ----------
    psi_min_global = float(np.nanmin(psi_lines))
    psi_max_global = float(np.nanmax(psi_lines))

    print(f"[STATS] global ψ range along lines: [{psi_min_global:.3e}, {psi_max_global:.3e}]")

    line_stats = []
    for i in range(S):
        psi_i = psi_lines[i, :]
        psi_mean = float(np.nanmean(psi_i))
        dpsi = psi_i - psi_mean
        denom = max(abs(psi_mean), 1e-12)  # normalize by line-averaged ψ

        dpsi_rel = dpsi / denom
        max_abs_rel = float(np.nanmax(np.abs(dpsi_rel)))
        std_rel = float(np.nanstd(dpsi_rel))

        line_stats.append((max_abs_rel, std_rel))
        print(
            f"[LINE {i:02d}] mean={psi_mean:.5f}, "
            f"max|Δψ|/|⟨ψ⟩|={max_abs_rel:.3e}, "
            f"std(Δψ/⟨ψ⟩)={std_rel:.3e}"
        )

    # ---------- Plot ψ(s) ----------
    base_label = Path(mfs_npz).with_suffix("").name
    fig1, ax1 = plt.subplots()
    for ii in range(S):
        i = S-ii-1  # plot in reverse order so lower seeds are on top
        psi_i = psi_lines[i, :]
        psi_mean = np.nanmean(psi_i)
        dpsi = psi_i - psi_mean
        denom = max(abs(psi_mean), 1e-12)
        dpsi_rel = dpsi / denom

        ax1.plot(
            s_lines[i, :],
            dpsi_rel,
            lw=0.5,
            label=f"seed {i+1}" if i < 8 else None,
            linestyle="-.",
        )

    ax1.set_xlabel(r"Field-line arclength $s$ [arb. units]")
    ax1.set_ylabel(r"$\Delta\psi(s)/\langle\psi\rangle_{\rm line}$")
    ax1.set_title(r"Field-line constancy of $\psi$ (FCI)")
    if S <= 8:
        ax1.legend(loc="best", frameon=True, framealpha=0.8)
    fig1.tight_layout()
    if save_figures:
        out1 = f"{base_label}_psi_vs_s.png"
        fig1.savefig(out1)
        print(f"[SAVE] Saved ψ(s) figure to {out1}")

    # ---------- Poincaré + ψ(R,Z) contours ----------
    # Use forward integration data Y
    Yj = jnp.asarray(Y, dtype=jnp.float64)
    phis = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, poincare_nphi, endpoint=False)
    print(f"[DEBUG] Computing Poincaré intersections on {poincare_nphi} planes, nfp={nfp}...")
    R_flat, Z_flat, M_flat, seed_flat = poincare_multi_phi_jax(Yj, phis)

    psi3 = psi_info["psi3"]
    grid_type = psi_info["grid_type"]

    # Use the inside mask to get only physical-domain values
    inside3 = psi_info["inside"].reshape(psi3.shape)
    psi_inside = psi3[inside3 & np.isfinite(psi3)]

    psi_min = float(np.nanmin(psi_inside))
    psi_max = float(np.nanmax(psi_inside))

    # Keep contour levels away from the very inner/outer bands
    level_margin = 0.02 * (psi_max - psi_min)
    psi_lo = psi_min + level_margin
    psi_hi = psi_max - level_margin

    # Use ψ at the seed positions as contour levels, but clipped to [psi_lo, psi_hi]
    psi_seed_for_levels = np.clip(psi_seed_valid, psi_lo, psi_hi)
    levs = np.unique(np.sort(psi_seed_for_levels))

    # Fallback in case something weird happens
    if levs.size < 2:
        levs = np.linspace(psi_lo, psi_hi, max(4, nseed))

    print("[DEBUG] ψ contour levels for overlay (from seeds):", levs)

    fig2, axs2 = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=True)
    axs2 = axs2.ravel()

    for k, phi0 in enumerate(np.asarray(phis)):
        if k >= len(axs2):
            break
        ax = axs2[k]

        if grid_type == "cylindrical":
            Rs   = psi_info["Rs"]
            Zs   = psi_info["Zs"]
            phis_grid = psi_info["phis"]
            inside3 = psi_info["inside"].reshape(psi3.shape)

            jphi = int(np.argmin(np.abs(phis_grid - phi0)))
            psi_slice = psi3[:, jphi, :].T          # (nZ, nR)
            inside_slice = inside3[:, jphi, :].T    # (nZ, nR)

            # Erode mask by one cell to stay away from the sharp boundary edge
            mask_interior = inside_slice.copy()
            # vertical neighbors
            mask_interior[0, :]  &= inside_slice[1, :]
            mask_interior[-1, :] &= inside_slice[-2, :]
            # horizontal neighbors
            mask_interior[:, 0]  &= inside_slice[:, 1]
            mask_interior[:, -1] &= inside_slice[:, -2]

            # Use a masked array so contours don't try to cross masked regions
            psi_plot = np.ma.masked_where(~mask_interior, psi_slice)

            im = ax.imshow(
                psi_plot,
                origin="lower",
                aspect="equal",
                extent=[Rs.min(), Rs.max(), Zs.min(), Zs.max()],
            )

            # Build explicit (R,Z) grid for contour — shape (nZ, nR)
            RR, ZZ = np.meshgrid(Rs, Zs, indexing="xy")
            cs = ax.contour(
                RR, ZZ, psi_plot,
                levels=levs,
                colors="black",
                linewidths=1.5,
                alpha=0.9,
            )
            ax.clabel(cs, fmt="%.3f", fontsize=7)
            im.set_clim(psi_min, psi_max)
            im.cmap.set_bad("black", alpha=0.7)
            plt.colorbar(im, ax=ax, shrink=0.85, label=r"$\psi$")

        else:
            xs = psi_info["xs"]
            zs = psi_info["zs"]
            # crude cylindrical R for background: take a mid y-plane
            ys = psi_info["ys"]
            jmid = len(ys) // 2
            psi_slice = psi3[:, jmid, :].T
            Rg = np.sqrt(xs**2 + ys[jmid]**2)
            Rg2, Zg = np.meshgrid(Rg, zs, indexing="ij")
            im = ax.imshow(
                psi_slice,
                origin="lower",
                aspect="equal",
                extent=[Rg.min(), Rg.max(), zs.min(), zs.max()],
            )
            cs = ax.contour(
                Rg, zs, psi_slice,
                levels=levs,
                colors="black",
                linewidths=1.4,
                alpha=0.9,
            )
            ax.clabel(cs, fmt="%.2f", fontsize=7)
            im.set_clim(psi_inside.min(), psi_inside.max())
            im.cmap.set_bad("white", alpha=0.0)
            plt.colorbar(im, ax=ax, shrink=0.85, label=r"$\psi$")

        # overlay Poincaré points at this φ
        Rf = np.asarray(R_flat[k])
        Zf = np.asarray(Z_flat[k])
        Mf = np.asarray(M_flat[k])
        Sf = np.asarray(seed_flat[k])

        Rk = Rf[Mf]
        Zk = Zf[Mf]
        Sk = Sf[Mf]

        if Rk.size > 0:
            for sidx in range(S):
                mask_s = (Sk == sidx)
                if not np.any(mask_s):
                    continue
                ax.scatter(
                    Rk[mask_s], Zk[mask_s],
                    s=0.4,
                    alpha=1.0,
                    rasterized=True,
                    label=f"seed {sidx}" if (k == 0 and sidx < 4) else None,
                )

        ax.set_xlabel(r"$R$")
        ax.set_ylabel(r"$Z$")
        ax.set_title(phi_label_pi(phi0, wrap=True))

    handles, labels = axs2[0].get_legend_handles_labels()
    if handles:
        fig2.legend(handles, labels, loc="lower center", ncol=4, frameon=True, framealpha=0.8)

    fig2.suptitle(r"Poincaré R–Z vs. $\psi(R,Z)$ contours", y=1.02)
    if save_figures:
        out2 = f"{base_label}_psi_poincare_overlay.png"
        fig2.savefig(out2)
        print(f"[SAVE] Saved Poincaré + ψ-overlay figure to {out2}")
        
    # ---------- 3D visualization: boundary, ψ-surfaces, and field lines ----------
    try:
        pl = plot_3d_flux_surfaces_and_fieldlines(
            psi_info=psi_info,
            grad_info=grad_info,
            Y=Y,
            psi_seed=psi_seed,
            n_surfaces=3,                # e.g., 3 nested ψ surfaces
            phi_center=1.0,            # cut around φ = π, π+π
            delta_phi_boundary=1.2,      # wider opening for boundary
            delta_phi_surface_max=1.2,   # inner surfaces shrink to 0
            delta_phi_fieldline=0.9,
        )

        # Make the object fill more of the window
        pl.camera_position = "iso"   # nice 3D view; or "xy", "xz", "yz"
        pl.camera.zoom(1.8)          # increase to fill more, decrease if too tight
        # Render interactively but keep the window open
        pl.show(auto_close=False)
        pl.camera.zoom(1.1)          # tiny extra zoom now that the title is gone
        # Take screenshot of the current view (tight framing, little white border)
        pl.screenshot(f"{base_label}_3d_visualization.png")
        pl.close()

        print("[3D] Finished PyVista 3D visualization.")
    except TypeError as e:
        if "theme" in str(e) and "DataSetMapper" in str(e):
            print("[3D] PyVista/VTK version mismatch: VTK is too old for this PyVista. "
                "Please upgrade both: pip install --upgrade 'vtk>=9.2' 'pyvista>=0.43'")
        else:
            print(f"[3D] Failed to build 3D visualization (TypeError): {e}")
    except Exception as e:
        print(f"[3D] Failed to build 3D visualization: {e}")

    plt.show()


# ------------------------------- CLI ------------------------------- #

if __name__ == "__main__":
    
    default_solution = "wout_precise_QA_solution.npz"
    # default_solution = "wout_precise_QH_solution.npz"
    # default_solution = "wout_SLAM_4_coils_solution.npz"
    # default_solution = "wout_SLAM_6_coils_solution.npz"
    # default_solution = "knot_tube_solution.npz"
    
    default_psi_npz = default_solution.replace(".npz", "_psi_fci_cyl_N64_Nphi128.npz")

    nfp_default = 2; tfinal_default = 2500.0; seeds_default = None; n_save_default = 12000
    if 'QH' in default_solution:
        nfp_default = 4
        
    if 'SLAM' in default_solution:
        tfinal_default = 15000; n_save_default = 3000
        seeds_default = "2.55:0:0,2.65:0:0,2.75:0:0,2.8:0:0,2.85:0:0,2.9:0:0,2.95:0:0,3.0:0:0"
    
    parser = argparse.ArgumentParser(
        description="Probe FCI ψ along field lines and compare with Poincaré plots."
    )
    parser.add_argument("mfs_npz", nargs="?", default=resolve_npz_file_location(default_solution),
                        help="MFS solution checkpoint (*.npz) containing center, scale, Yn, alpha, a, a_hat, P, N")
    parser.add_argument("psi_npz", nargs="?", default=default_psi_npz,
                        help="ψ-solution NPZ created by solve_flux_psi_fci_cyl.py.")
    parser.add_argument("--nseed", type=int, default=6,
                        help="Number of field-line seeds (used if --seeds is not given).")
    parser.add_argument("--tfinal", type=float, default=tfinal_default,
                        help="Final integration time for field lines.")
    parser.add_argument("--n-save", type=int, default=n_save_default,
                        help="Number of output samples per field line.")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize ∇φ when tracing (unit-speed field lines).")
    parser.add_argument("--clip-grad", type=float, default=None,
                        help="Optionally clip |∇φ| during tracing.")
    parser.add_argument("--poincare-nphi", type=int, default=4,
                        help="Number of Poincaré planes per field period.")
    parser.add_argument("--nfp", type=int, default=nfp_default,
                        help="Number of field periods.")
    parser.add_argument("--seeds", type=str, default=seeds_default,
                        help="Comma-separated list of seed points as x:y:z,x:y:z,...")
    parser.add_argument("--no-save-figures", action="store_true",
                        help="Do not save figures, just show them.")
    args = parser.parse_args()

    analyze(
        mfs_npz=args.mfs_npz,
        psi_npz=args.psi_npz,
        nseed=args.nseed,
        tfinal=args.tfinal,
        n_save=args.n_save,
        normalize=args.normalize,
        clip_grad=args.clip_grad,
        poincare_nphi=args.poincare_nphi,
        nfp=args.nfp,
        seeds_str=args.seeds,
        save_figures=not args.no_save_figures,
    )
