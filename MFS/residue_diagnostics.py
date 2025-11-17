#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Greene's residue diagnostics from MFS magnetic field and ψ snapshot.

This script:
  1) Loads ψ(𝐱) from a psi_fci snapshot (.npz) to obtain the magnetic axis.
  2) Loads the MFS solution checkpoint (center, scale, Yn, alpha, a, a_hat, P, N).
  3) Rebuilds ∇φ(𝐱) so that 𝐁 = ∇φ.
  4) Defines a Poincaré map at φ = 0 (toroidal angle) using a simple RK4
     field-line integrator in Cartesian coordinates.
  5) For a set of rational resonances p/q and initial guesses (R,Z) on the
     φ = 0 plane, finds periodic field lines such that P^q(u*) = u* with
     u = (R,Z), and computes the monodromy matrix via finite differences.
  6) Evaluates Greene's residue

         R = (2 - Tr M) / 4

     for each periodic orbit.
  7) Generates publication-ready plots:
       - Poincaré maps near each resonance, with the periodic orbit marked.
       - Bar chart of Greene residues vs resonance label p/q.
       - Greene residue vs (approximate) minor radius r at the section.

Usage (example):
  python residue_diagnostics.py \
      wout_precise_QA_solution.npz \
      --psi-npz wout_precise_QA_solution_psi_fci_cyl_N64_Nphi128.npz \
      --nfp 2 \
      --resonances 1/4,2/5 \
      --guesses "3.3:0.0;3.5:0.0"

If --guesses is omitted, crude guesses are constructed from axis + boundary;
for robust convergence, providing guesses is recommended.

Notes:
  - This script is intended as a diagnostic/visualization tool, not a
    highly-optimized JAX pipeline. It uses NumPy + SciPy for the residue
    computations while reusing JAX-based ∇φ evaluators.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import os
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap

from scipy.interpolate import interp1d
from scipy.optimize import root

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# -------------------------- Paths and utils ------------------------- #

script_dir = Path(__file__).resolve().parent

def resolve_npz_file_location(npz_file, subdir="outputs"):
    """Try ../subdir/<basename>.npz if the provided path doesn't exist."""
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
# ψ snapshot loader (for axis and approximate "minor radius")
# ---------------------------------------------------------------------------

def load_psi_snapshot(psi_npz: str) -> Dict[str, Any]:
    data = np.load(psi_npz, allow_pickle=True)
    grid_type = str(data["grid_type"])
    psi_flat = np.asarray(data["psi"])
    inside = np.asarray(data["inside"], dtype=bool)
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
        "mins": mins,
        "maxs": maxs,
        "axis_points": axis_points,
        "R_axis": R_axis,
        "Z_axis": Z_axis,
    }

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

def minor_radius_from_axis(R: float, Z: float, phi: float,
                           R_axis_interp, Z_axis_interp) -> float:
    """Approximate minor radius r at (R,phi,Z) as distance from axis in (R,Z)."""
    Rax = float(R_axis_interp(phi))
    Zax = float(Z_axis_interp(phi))
    dR = R - Rax
    dZ = Z - Zax
    return float(np.sqrt(dR*dR + dZ*dZ))

# ---------------------------------------------------------------------------
# B-field evaluator and field-line integrator
# ---------------------------------------------------------------------------

def make_B_eval(grad_phi):
    """
    Wrap JAX grad_phi into a NumPy-callable B_eval(x) with x.shape = (3,).
    """
    def B_eval(x: np.ndarray) -> np.ndarray:
        xj = jnp.asarray(x[None, :], dtype=jnp.float64)  # (1,3)
        Bj = grad_phi(xj)[0]
        return np.asarray(Bj, dtype=float)
    return B_eval

def rk4_step(x: np.ndarray, ds: float, B_eval) -> np.ndarray:
    """
    Single RK4 step for field line:
      dx/ds = B/|B|
    """
    def v(xloc):
        B = B_eval(xloc)
        normB = np.linalg.norm(B)
        if normB < 1e-14:
            return np.zeros_like(B)
        return B / normB

    k1 = v(x)
    k2 = v(x + 0.5*ds*k1)
    k3 = v(x + 0.5*ds*k2)
    k4 = v(x + ds*k3)
    return x + (ds/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def unwrap_delta_phi(phi_new: float, phi_prev: float) -> float:
    """
    Return a delta φ in (-π,π] so we can accumulate a continuous toroidal angle.
    """
    d = phi_new - phi_prev
    while d <= -np.pi:
        d += 2*np.pi
    while d > np.pi:
        d -= 2*np.pi
    return d

def integrate_to_delta_phi(x0: np.ndarray,
                           B_eval,
                           delta_phi_target: float,
                           ds: float = 0.05,
                           max_steps: int = 200000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate a field line starting at x0 until the total toroidal angle φ
    has advanced by delta_phi_target (unwrapped). Return (x_end, track),
    where track is an array of points along the path for plotting.
    """
    x = np.array(x0, dtype=float)
    track = [x.copy()]

    phi = np.arctan2(x[1], x[0])
    phi_cum = 0.0
    phi_prev = phi

    for _ in range(max_steps):
        x_new = rk4_step(x, ds, B_eval)
        phi_new = np.arctan2(x_new[1], x_new[0])
        dphi = unwrap_delta_phi(phi_new, phi_prev)
        phi_cum += dphi
        phi_prev = phi_new
        x = x_new
        track.append(x.copy())

        if phi_cum >= delta_phi_target:
            break
    else:
        print("[WARN] integrate_to_delta_phi: max_steps reached before hitting target Δφ")

    return x, np.array(track)

# ---------------------------------------------------------------------------
# Poincaré map and periodic orbit finder
# ---------------------------------------------------------------------------

def poincare_map(u: np.ndarray,
                 B_eval,
                 nfp: int,
                 toroidal_periods: int = 1,
                 phi_plane: float = 0.0,
                 ds: float = 0.05) -> np.ndarray:
    """
    Poincaré map on the plane φ = φ_plane (mod 2π/nfp).

    Input:
      u = (R,Z) on φ = φ_plane.
    Output:
      u' = (R',Z') after toroidal_periods field periods (2π/nfp each),
           mapped back onto the same section.
    """
    R, Z = float(u[0]), float(u[1])
    # Start at φ = phi_plane
    x0 = np.array([R*np.cos(phi_plane), R*np.sin(phi_plane), Z], dtype=float)
    delta_phi_target = toroidal_periods * 2.0*np.pi / float(nfp)

    x_end, _ = integrate_to_delta_phi(x0, B_eval, delta_phi_target, ds=ds)
    # At the end, we are at φ = φ_plane + toroidal_periods*2π/nfp (unwrapped)
    # but in actual coordinates R' = sqrt(x^2 + y^2), Z' = z
    R_new = np.sqrt(x_end[0]**2 + x_end[1]**2)
    Z_new = x_end[2]
    return np.array([R_new, Z_new], dtype=float)

def iterate_poincare(u: np.ndarray,
                     B_eval,
                     nfp: int,
                     q: int,
                     phi_plane: float = 0.0,
                     ds: float = 0.05) -> np.ndarray:
    """
    Apply the Poincaré map q times: P^q(u).
    """
    v = np.array(u, dtype=float)
    for _ in range(q):
        v = poincare_map(v, B_eval, nfp=nfp,
                         toroidal_periods=1,
                         phi_plane=phi_plane,
                         ds=ds)
    return v

def find_periodic_orbit(p: int,
                        q: int,
                        u_guess: np.ndarray,
                        B_eval,
                        nfp: int,
                        phi_plane: float = 0.0,
                        ds: float = 0.05,
                        tol: float = 1e-10,
                        maxiter: int = 30) -> Tuple[np.ndarray, bool]:
    """
    Find u* such that P^q(u*) = u* using a 2D root solver.

    p is not used explicitly by this map (we parameterize by q,
    corresponding to the number of toroidal periods), but is kept
    for labeling purposes.
    """
    def F(u):
        u = np.array(u, dtype=float)
        return iterate_poincare(u, B_eval, nfp=nfp,
                                q=q, phi_plane=phi_plane, ds=ds) - u

    res = root(F, np.asarray(u_guess, dtype=float),
               tol=tol, options={"maxiter": maxiter})
    if not res.success:
        print(f"[WARN] find_periodic_orbit: root solver failed for (p,q)=({p},{q}): {res.message}")
        return np.asarray(u_guess, dtype=float), False
    return np.asarray(res.x, dtype=float), True

def monodromy_fd(u_star: np.ndarray,
                 B_eval,
                 nfp: int,
                 q: int,
                 phi_plane: float = 0.0,
                 ds: float = 0.05,
                 delta: float = 1e-4) -> np.ndarray:
    """
    Finite-difference approximation to the monodromy matrix
      M = DP^q(u_star),
    using central differences.
    """
    u0 = np.asarray(u_star, dtype=float)
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])

    def G(u):
        return iterate_poincare(u, B_eval, nfp=nfp,
                                q=q, phi_plane=phi_plane, ds=ds)

    G_u0 = G(u0)

    # dG/du1
    G_p = G(u0 + delta*e1)
    G_m = G(u0 - delta*e1)
    dG_du1 = (G_p - G_m) / (2.0 * delta)

    # dG/du2
    G_p = G(u0 + delta*e2)
    G_m = G(u0 - delta*e2)
    dG_du2 = (G_p - G_m) / (2.0 * delta)

    M = np.column_stack([dG_du1, dG_du2])
    return M

def greene_residue_from_monodromy(M: np.ndarray) -> float:
    """
    Greene's residue:
        R = (2 - Tr M) / 4
    """
    trace = np.trace(M)
    return float((2.0 - trace) / 4.0)

# ---------------------------------------------------------------------------
# Poincaré cloud for visualization
# ---------------------------------------------------------------------------

def poincare_cloud_around_orbit(u_star: np.ndarray,
                                B_eval,
                                nfp: int,
                                n_lines: int = 20,
                                n_iter: int = 80,
                                radial_spread: float = 0.02,
                                phi_plane: float = 0.0,
                                ds: float = 0.05) -> np.ndarray:
    """
    Generate a cloud of Poincaré points near a periodic orbit, by launching
    several nearby field lines and recording intersections with φ = φ_plane.
    Returns array of shape (N, 2) with columns (R,Z).
    """
    R0, Z0 = float(u_star[0]), float(u_star[1])
    pts = []

    # Launch lines in a small radial/poloidal neighborhood
    for k in range(n_lines):
        angle = 2*np.pi * k / max(1, n_lines)
        dR = radial_spread * np.cos(angle)
        dZ = radial_spread * np.sin(angle)
        u_init = np.array([R0 + dR, Z0 + dZ], dtype=float)
        u = u_init.copy()
        for _ in range(n_iter):
            u = poincare_map(u, B_eval, nfp=nfp,
                             toroidal_periods=1,
                             phi_plane=phi_plane,
                             ds=ds)
            pts.append(u.copy())

    return np.asarray(pts, dtype=float)

# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------

def parse_resonance_list(res_str: str) -> List[Tuple[int, int]]:
    """
    Parse string like "1/4,2/5,3/7" into list [(1,4),(2,5),(3,7)].
    """
    items = [s.strip() for s in res_str.split(",") if s.strip()]
    res = []
    for it in items:
        p_str, q_str = it.split("/")
        res.append((int(p_str), int(q_str)))
    return res

def parse_guess_list(guess_str: str) -> List[Tuple[float, float]]:
    """
    Parse string like "3.3:0.0;3.5:0.0" into [(3.3,0.0),(3.5,0.0)].
    """
    items = [s.strip() for s in guess_str.split(";") if s.strip()]
    out = []
    for it in items:
        R_str, Z_str = it.split(":")
        out.append((float(R_str), float(Z_str)))
    return out

# ---------------------------------------------------------------------------
# Main diagnostics: residues + plots
# ---------------------------------------------------------------------------

def main(mfs_npz: str,
         psi_npz: str,
         nfp: int,
         resonances: List[Tuple[int, int]],
         guesses: List[Tuple[float, float]] | None = None,
         phi_plane: float = 0.0,
         ds: float = 0.05):

    # Matplotlib style
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.figsize": (6.0, 4.5),
        "savefig.dpi": 300,
    })

    # 1) Load ψ snapshot and magnetic axis
    psi_data = load_psi_snapshot(psi_npz)
    axis_points = psi_data["axis_points"]
    R_axis_interp, Z_axis_interp = build_axis_interp(axis_points)

    # 2) Load MFS grad_phi and wrap into B_eval
    grad_phi, P_surf, N_surf = load_mfs_grad_phi(mfs_npz)
    B_eval = make_B_eval(grad_phi)

    # 3) Build initial guesses if none provided
    if guesses is None or len(guesses) == 0:
        print("[INFO] No guesses provided; constructing crude radial guesses.")
        # Use axis point at φ=phi_plane and one boundary point at minimum poloidal distance
        Rax0 = float(R_axis_interp(phi_plane))
        Zax0 = float(Z_axis_interp(phi_plane))

        # Project boundary points onto that section, pick those near φ=phi_plane
        xP, yP, zP = P_surf[:, 0], P_surf[:, 1], P_surf[:, 2]
        phiP = np.mod(np.arctan2(yP, xP), 2*np.pi)
        dphi = np.abs(np.unwrap(phiP - phi_plane))
        mask = dphi < (np.pi / nfp)  # thin wedge around φ=phi_plane
        if not np.any(mask):
            raise RuntimeError("Could not find boundary points near the chosen φ plane.")
        Rb = np.sqrt(xP[mask]**2 + yP[mask]**2)
        Zb = zP[mask]

        # Take median boundary radius at that φ
        Rb_med = float(np.median(Rb))
        Zb_med = float(np.median(Zb))

        # Minor radius from axis to boundary
        r_b = minor_radius_from_axis(Rb_med, Zb_med, phi_plane, R_axis_interp, Z_axis_interp)

        guesses = []
        for k in range(len(resonances)):
            # spread resonances across 0.2..0.8 of that radius
            frac = 0.2 + 0.6 * (k / max(1, len(resonances) - 1))
            r_k = frac * r_b
            # place along horizontal line through axis
            guesses.append((Rax0 + r_k, Zax0))

        print(f"[INFO] Constructed {len(guesses)} crude guesses for resonances.")

    if len(guesses) != len(resonances):
        raise ValueError("Number of guesses must match number of resonances (p/q).")

    # 4) For each resonance, find periodic orbit and compute residue
    residues = []
    orbit_positions = []   # (R,Z) of u_star
    orbit_radii = []       # minor radius r at φ-plane
    poincare_clouds = []   # Poincaré points for plotting

    for (p, q), (Rg, Zg) in zip(resonances, guesses):
        print("----------------------------------------------------")
        print(f"[INFO] Processing resonance (p,q)=({p},{q}) with guess (R,Z)=({Rg:.3f},{Zg:.3f})")

        u_guess = np.array([Rg, Zg], dtype=float)
        u_star, success = find_periodic_orbit(p, q, u_guess, B_eval,
                                              nfp=nfp,
                                              phi_plane=phi_plane,
                                              ds=ds)
        if not success:
            R_val = np.nan
            print(f"[INFO] Failed to converge periodic orbit for (p,q)=({p},{q}).")
            residues.append(R_val)
            orbit_positions.append((u_guess[0], u_guess[1]))
            r_k = minor_radius_from_axis(u_guess[0], u_guess[1], phi_plane,
                                         R_axis_interp, Z_axis_interp)
            orbit_radii.append(r_k)
            poincare_clouds.append(np.empty((0, 2)))
            continue

        print(f"[INFO] Found periodic orbit at (R,Z)=({u_star[0]:.6f},{u_star[1]:.6f})")

        M = monodromy_fd(u_star, B_eval, nfp=nfp,
                         q=q,
                         phi_plane=phi_plane,
                         ds=ds,
                         delta=1e-4)
        R_val = greene_residue_from_monodromy(M)
        print(f"[INFO] Greene residue for (p,q)=({p},{q}) is R={R_val:.6e}")

        residues.append(R_val)
        orbit_positions.append((u_star[0], u_star[1]))

        r_k = minor_radius_from_axis(u_star[0], u_star[1], phi_plane,
                                     R_axis_interp, Z_axis_interp)
        orbit_radii.append(r_k)

        cloud = poincare_cloud_around_orbit(u_star, B_eval, nfp=nfp,
                                            n_lines=24, n_iter=80,
                                            radial_spread=0.02 * max(1.0, r_k),
                                            phi_plane=phi_plane,
                                            ds=ds)
        poincare_clouds.append(cloud)

    residues = np.asarray(residues, dtype=float)
    orbit_radii = np.asarray(orbit_radii, dtype=float)
    orbit_positions = np.asarray(orbit_positions, dtype=float)

    # 5) Publication-style plots
    base = mfs_npz.replace(".npz", "_residue")

    # (a) Poincaré maps for each resonance
    n_res = len(resonances)
    ncols = min(3, n_res)
    nrows = int(np.ceil(n_res / ncols))
    figP, axesP = plt.subplots(nrows, ncols,
                               figsize=(4.5*ncols, 4.0*nrows),
                               constrained_layout=True)
    if nrows == 1 and ncols == 1:
        axesP = np.array([[axesP]])
    elif nrows == 1:
        axesP = axesP[None, :]
    elif ncols == 1:
        axesP = axesP[:, None]

    for idx, ((p, q), cloud, u_star, R_val) in enumerate(
        zip(resonances, poincare_clouds, orbit_positions, residues)
    ):
        row = idx // ncols
        col = idx % ncols
        ax = axesP[row, col]

        if cloud.size > 0:
            ax.scatter(cloud[:, 0], cloud[:, 1],
                       s=4, alpha=0.4, edgecolors="none", label="Nearby lines")
        ax.plot(u_star[0], u_star[1],
                marker="o", markersize=7, color="black",
                label="Periodic orbit")

        # Axis point at this φ_plane for context
        Rax0 = float(R_axis_interp(phi_plane))
        Zax0 = float(Z_axis_interp(phi_plane))
        ax.plot(Rax0, Zax0, marker="x", color="red", label="Axis")

        ax.set_aspect("equal", "box")
        ax.set_xlabel(r"$R$")
        ax.set_ylabel(r"$Z$")
        if np.isfinite(R_val):
            ax.set_title(rf"$p/q = {p}/{q}$, $R = {R_val:.3f}$")
        else:
            ax.set_title(rf"$p/q = {p}/{q}$ (no convergence)")

        if idx == 0:
            ax.legend(frameon=False, loc="best")

    # Hide unused subplots
    for k in range(n_res, nrows*ncols):
        row = k // ncols
        col = k % ncols
        axesP[row, col].axis("off")

    figP.suptitle("Poincaré maps near selected periodic field lines", y=1.02)
    outP = base + "_poincare.png"
    figP.savefig(outP, bbox_inches="tight")
    print(f"[PLOT] Saved Poincaré maps to {outP}")

    # (b) Bar chart of Greene residues vs resonance
    labels = [f"{p}/{q}" for (p, q) in resonances]
    x = np.arange(len(resonances))
    figB, axB = plt.subplots()
    bar_colors = []
    for R_val in residues:
        if not np.isfinite(R_val):
            bar_colors.append("gray")
        elif 0.0 < R_val < 1.0:
            bar_colors.append("C0")  # elliptic
        else:
            bar_colors.append("C3")  # hyperbolic/unstable
    axB.bar(x, residues, color=bar_colors)
    axB.axhline(0.0, color="k", linewidth=0.8)
    axB.axhline(1.0, color="k", linewidth=0.8, linestyle="--")
    axB.set_xticks(x)
    axB.set_xticklabels(labels)
    axB.set_ylabel(r"Greene residue $R$")
    axB.set_xlabel(r"Resonance $p/q$")
    axB.set_title("Greene residues of selected periodic field lines")
    figB.tight_layout()
    outB = base + "_bars.png"
    figB.savefig(outB, bbox_inches="tight")
    print(f"[PLOT] Saved residue bar chart to {outB}")

    # (c) Residue vs minor radius
    figR, axR = plt.subplots()
    # Sort by radius
    order = np.argsort(orbit_radii)
    axR.plot(orbit_radii[order], residues[order], "o-", lw=1.5)
    axR.axhline(0.0, color="k", linewidth=0.8)
    axR.axhline(1.0, color="k", linewidth=0.8, linestyle="--")
    axR.set_xlabel(r"Minor radius $r$ at $\varphi = 0$")
    axR.set_ylabel(r"Greene residue $R$")
    axR.set_title("Greene residue vs minor radius")
    axR.grid(True, alpha=0.3)
    figR.tight_layout()
    outR = base + "_r_profile.png"
    figR.savefig(outR, bbox_inches="tight")
    print(f"[PLOT] Saved residue vs radius profile to {outR}")

    # (d) Save numerical data for paper / postprocessing
    out_data = base + "_data.npz"
    p_list = np.array([p for (p, q) in resonances], dtype=int)
    q_list = np.array([q for (p, q) in resonances], dtype=int)
    np.savez(
        out_data,
        p_list=p_list,
        q_list=q_list,
        residues=residues,
        orbit_positions=orbit_positions,
        orbit_radii=orbit_radii,
    )
    print(f"[SAVE] Saved residue diagnostic data to {out_data}")

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print('##################################')
    print('#### FILE NOT BENCHMARKED YET ####')
    print('#### FILE NOT BENCHMARKED YET ####')
    print('##################################')
    default_solution = "wout_precise_QA_solution.npz"
    default_psi_npz = default_solution.replace(".npz", "_psi_fci_cyl_N64_Nphi128.npz")
    default_solution = resolve_npz_file_location(default_solution)

    parser = argparse.ArgumentParser(
        description="Greene's residue diagnostics from MFS field and ψ snapshot."
    )
    parser.add_argument(
        "mfs_npz",
        nargs="?",
        default=default_solution,
        help="MFS solution checkpoint (.npz) to rebuild grad_phi",
    )
    parser.add_argument(
        "--psi-npz",
        default=default_psi_npz,
        help="psi_fci snapshot (.npz) (used to obtain magnetic axis)",
    )
    parser.add_argument(
        "--nfp",
        type=int,
        default=2,
        help="Number of field periods of the device",
    )
    parser.add_argument(
        "--resonances",
        type=str,
        default="1/4,2/5",
        help="Comma-separated list of resonances p/q, e.g. '1/4,2/5'",
    )
    parser.add_argument(
        "--guesses",
        type=str,
        default="",
        help="Optional guesses for (R,Z) on φ=0 plane, "
             "semicolon-separated 'R:Z;R:Z;...', e.g. '3.3:0.0;3.5:0.0'",
    )
    parser.add_argument(
        "--phi-plane",
        type=float,
        default=0.0,
        help="Toroidal angle φ (radians) of the Poincaré section (default 0)",
    )
    parser.add_argument(
        "--ds",
        type=float,
        default=0.05,
        help="Field-line integration step size (arc-length units)",
    )

    args = parser.parse_args()

    res_pairs = parse_resonance_list(args.resonances)
    if args.guesses.strip():
        guess_pairs = parse_guess_list(args.guesses)
    else:
        guess_pairs = None

    main(
        mfs_npz=args.mfs_npz,
        psi_npz=args.psi_npz,
        nfp=args.nfp,
        resonances=res_pairs,
        guesses=guess_pairs,
        phi_plane=args.phi_plane,
        ds=args.ds,
    )
