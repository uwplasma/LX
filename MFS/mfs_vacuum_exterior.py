#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exterior vacuum-field solver via MFS, consistent with an interior MFS solution
=============================================================================

This script constructs a *vacuum magnetic field outside* a given closed surface
Γ, using a previously computed *interior* Laplace solution φ_in from your
MFS-based Neumann solver (main.py).

Physics and formulation
-----------------------

We assume a smooth closed surface Γ bounding a domain Ω_in (interior) and
its complement Ω_out (exterior). The interior script has already produced a
harmonic potential

    φ_in(x) = φ_mv(x) + ψ_in(x),

with B_in = ∇φ_in a vacuum magnetic field in Ω_in, and Neumann BC

    n·∇φ_in = g_n(x)  on Γ

satisfying flux-compatibility and any prescribed topological fluxes via a
multivalued "harmonic" part φ_mv.

In the *absence of sheet currents* on Γ, the vacuum field is continuous
across the surface. In particular the *normal component* of B must be
continuous:

    n·∇φ_out = n·∇φ_in       on Γ

where φ_out is the exterior vacuum potential in Ω_out.

We construct an exterior MFS ansatz

    φ_out(x) = φ_mv(x) + ψ_out(x),  ψ_out(x) = Σ_j α_ext,j G(x, y_j^int),

where G is the 3D Laplace fundamental solution

    G(x,y) = 1/(4π |x-y|),

and the source points y_j^int lie on an *interior* fictitious surface obtained
by offsetting Γ inwards along the (outward) unit normal. This is the natural
"mirror" of the interior construction, where the sources live outside Γ.

We then solve a Tikhonov-regularized weighted least-squares problem for
α_ext such that the Neumann data of φ_out matches that of φ_in on Γ in a
weighted sense.

Inputs
------

This script expects as input the portable MFS checkpoint produced by your
interior solver (main.py), which must contain:

  center (3,)      -- geometry center used for normalization
  scale  (scalar)  -- geometry scale factor (median radius → 1)
  Yn     (M,3)     -- normalized interior source locations (for φ_in)
  alpha  (M,)      -- interior MFS coefficients (for φ_in)
  a      (2,)      -- multivalued coefficients [a_t, a_p]
  a_hat  (3,)      -- unit axis vector used for multivalued bases
  P      (N,3)     -- boundary points on Γ (world coordinates)
  N      (N,3)     -- outward unit normals on Γ (world coordinates)
  kind   ()        -- "torus" or "mirror" (string) for metadata

Output
------

A new checkpoint file (by default <input_stem>_exterior_solution.npz) with
the same layout:

  center, scale, Yn (now interior sources for φ_out), alpha (α_ext),
  a, a_hat, P, N, kind

This is directly compatible with your existing field-line tracer.

Usage
-----

  python mfs_vacuum_exterior.py interior_solution.npz \
      --k-nn 48 \
      --lambda-candidates 1e-6 1e-4 1e-2 \
      --out mfs_exterior_solution.npz

You can then use the resulting *_exterior_solution.npz with your tracer to
compute vacuum field lines in the *exterior* region, e.g. for divertor
studies, provided your traced trajectories do not cross the interior
source surface.

Dependencies
------------

  - numpy
  - jax, jaxlib (with 64-bit enabled)
  - scikit-learn (NearestNeighbors)
  - matplotlib (optional, for diagnostics if --plot is given)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, vmap, jacrev

from sklearn.neighbors import NearestNeighbors
from typing import Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ---------------------------------------------------------------------------
# JAX configuration
# ---------------------------------------------------------------------------

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Paths and utils
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Simple Pytree dataclass for geometry scale info
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass
class ScaleInfo:
    """Stores geometry center and scale used for normalization.

    World coordinates X are mapped to normalized coordinates Xn via
        Xn = (X - center) * scale
    """
    center: jnp.ndarray   # (3,)
    scale: jnp.ndarray    # scalar array, shape ()

    def tree_flatten(self):
        return (self.center, jnp.asarray(self.scale)), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        center, scale = children
        return cls(center=center, scale=jnp.asarray(scale))


# ---------------------------------------------------------------------------
# Laplace fundamental solution and its gradient
# ---------------------------------------------------------------------------

@jit
def green_G(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Fundamental solution G(x,y) = 1/(4π |x - y|)."""
    r = jnp.linalg.norm(x - y, axis=-1)
    return 1.0 / (4.0 * jnp.pi * jnp.maximum(1e-30, r))


@jit
def grad_green_x(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Gradient ∇_x G(x,y) of the Laplace kernel."""
    r = x - y
    r2 = jnp.sum(r * r, axis=-1)
    r3 = jnp.maximum(1e-30, r2 * jnp.sqrt(r2))
    return -r / (4.0 * jnp.pi * r3[..., None])


# ---------------------------------------------------------------------------
# Multivalued bases around a given axis a_hat
# ---------------------------------------------------------------------------

@jit
def grad_azimuth_about_axis(Xn: jnp.ndarray, a_hat: jnp.ndarray) -> jnp.ndarray:
    """∇ϕ_a for azimuth around arbitrary unit axis a_hat in normalized space.

    For each point x, decompose x into parallel and perpendicular parts w.r.t a_hat:
        r_perp = x - (x·a_hat)a_hat
        ∇ϕ_a = (a_hat × r_perp) / |r_perp|^2
    """
    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par = jnp.sum(Xn * a[None, :], axis=1, keepdims=True) * a[None, :]
    r_perp = Xn - r_par
    r2 = jnp.maximum(1e-30, jnp.sum(r_perp * r_perp, axis=1, keepdims=True))
    cross = jnp.cross(a[None, :], r_perp)
    return cross / r2


def multivalued_bases_about_axis(Pn: jnp.ndarray,
                                 Nn: jnp.ndarray,
                                 a_hat: jnp.ndarray):
    """Construct multivalued basis gradients around axis a_hat.

    Returns two gradient functions in *normalized* coordinates:

      grad_t(Xn): "toroidal-like" multivalued basis ∇ϕ_a (azimuth around a_hat)
      grad_p(Xn): "poloidal-like" tangent basis constructed from local geometry

    These are the same types of bases used in the interior solver to represent
    toroidal and poloidal flux degrees of freedom.
    """
    Pn_ref = jnp.asarray(Pn)
    Nn_ref = jnp.asarray(Nn)
    a_hat = jnp.asarray(a_hat)

    @jit
    def _nearest_normal_jax(Xn_eval: jnp.ndarray) -> jnp.ndarray:
        """Nearest neighbor normals (brute-force) for use in grad_p."""
        X2 = jnp.sum(Xn_eval * Xn_eval, axis=1, keepdims=True)
        P2 = jnp.sum(Pn_ref * Pn_ref, axis=1, keepdims=True)
        dist2 = X2 + P2.T - 2.0 * (Xn_eval @ Pn_ref.T)
        idx = jnp.argmin(dist2, axis=1)
        return Nn_ref[idx, :]

    @jit
    def _unit(v: jnp.ndarray, eps: float = 1e-30) -> jnp.ndarray:
        nrm = jnp.linalg.norm(v, axis=1, keepdims=True)
        return v / jnp.maximum(eps, nrm)

    @jit
    def _project_tangent(v: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
        return v - jnp.sum(v * n, axis=1, keepdims=True) * n

    def grad_t(Xn: jnp.ndarray) -> jnp.ndarray:
        """Toroidal-like multivalued gradient ∇ϕ_a in normalized coords."""
        return grad_azimuth_about_axis(Xn, a_hat)

    def grad_p(Xn: jnp.ndarray) -> jnp.ndarray:
        """Poloidal-like tangent direction from axis and surface normals."""
        # Nearest boundary normal
        n = _nearest_normal_jax(Xn)
        # Build φ̂_a = unit(a × r_perp)
        a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
        r_par = jnp.sum(Xn * a[None, :], axis=1, keepdims=True) * a[None, :]
        r_perp = Xn - r_par
        phi_hat = _unit(jnp.cross(a[None, :], r_perp))
        phi_tan = _unit(_project_tangent(phi_hat, n))
        theta_hat = _unit(jnp.cross(n, phi_tan))
        return theta_hat

    return grad_t, grad_p


# ---------------------------------------------------------------------------
# kNN-based quadrature weights on the boundary
# ---------------------------------------------------------------------------

def kNN_geometry_stats(Pn: np.ndarray, k: int = 48):
    """Compute crude area weights W and local length scale rk from kNN in 3D.

    Parameters
    ----------
    Pn : (N,3) array
        Normalized boundary nodes.
    k : int
        Number of neighbors for local radius estimate.

    Returns
    -------
    W : (N,) jnp.ndarray
        Approximate area weights W_i ≈ π * r_k,i^2.
    rk : (N,) np.ndarray
        Local k-NN radius (in normalized coordinates).
    """
    P_np = np.asarray(Pn)
    k_eff = min(k + 1, len(P_np))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree").fit(P_np)
    dists, _ = nbrs.kneighbors(P_np)
    rk = dists[:, -1]  # k-th neighbor radius in 3D
    W = jnp.asarray(np.pi * rk**2, dtype=jnp.float64)
    return W, rk


# ---------------------------------------------------------------------------
# MFS system assembly (Neumann BC) for exterior problem
# ---------------------------------------------------------------------------

def build_mfs_sources_interior(Pn: jnp.ndarray,
                               Nn: jnp.ndarray,
                               Yn_interior: jnp.ndarray) -> jnp.ndarray:
    """Build interior (for exterior problem) fictitious sources.

    We mirror the interior solver's source distances:

      δ_i = ||Yn_in[i] - Pn[i]|| (in normalized coords)
      y_i^int = Pn[i] - δ_i Nn[i]

    Here Yn_interior is the *exterior* source cloud used for the interior solve.
    """
    Pn_np = np.asarray(Pn)
    Nn_np = np.asarray(Nn)
    Y_in_np = np.asarray(Yn_interior)

    delta_n = np.linalg.norm(Y_in_np - Pn_np, axis=1)  # (N,) normalized distances
    Yn_int_np = Pn_np - delta_n[:, None] * Nn_np       # move inward

    return jnp.asarray(Yn_int_np, dtype=jnp.float64)


@jit
def build_system_matrix_neumann(Pn: jnp.ndarray,
                                Nn: jnp.ndarray,
                                Yn_int: jnp.ndarray,
                                scinfo: ScaleInfo) -> jnp.ndarray:
    """Build collocation matrix A for Neumann BC:

        A_ij = n_i · ∇_x G(x_i, y_j), x_i = Pn[i], y_j = Yn_int[j]

    The result is scaled by scinfo.scale so that it corresponds to world units.
    """
    X = Pn

    @jit
    def row_kernel(xi, ni):
        grads = vmap(lambda yj: grad_green_x(xi, yj))(Yn_int)  # (M,3)
        return jnp.dot(grads, ni)                              # (M,)

    A = vmap(row_kernel)(X, Nn)  # (N,M)
    return scinfo.scale * A


# ---------------------------------------------------------------------------
# Regularized weighted least-squares solve for α_ext
# ---------------------------------------------------------------------------

def solve_alpha_tikhonov(A: jnp.ndarray,
                         W: jnp.ndarray,
                         h_raw: jnp.ndarray,
                         lam: float,
                         verbose: bool = True):
    """Solve min || W^{1/2}(A α + h_raw)||_2^2 + λ^2 ||α||_2^2 for α.

    Here h_raw encodes the mismatch between multivalued Neumann contribution
    and the target Neumann data from the interior solution:

        n·∇φ_out = n·∇φ_in  ⇒  A α_ext + g_mv ≈ Bn_target
        ⇒ A α_ext + (g_mv - Bn_target) ≈ 0  ⇒ h_raw = g_mv - Bn_target
    """
    Wsqrt = jnp.sqrt(W)
    Aw = Wsqrt[:, None] * A
    hw = Wsqrt * h_raw

    ATA = Aw.T @ Aw
    ATb = Aw.T @ hw

    NE = ATA + (lam**2) * jnp.eye(A.shape[1], dtype=A.dtype)
    rhs = -ATb

    # Cholesky solve
    L = jnp.linalg.cholesky(NE)
    y = jax.scipy.linalg.solve_triangular(L, rhs, lower=True)
    alpha = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)

    res_w = Aw @ alpha + hw
    res_norm = float(jnp.linalg.norm(res_w))

    if verbose:
        condNE = float(np.linalg.cond(np.asarray(NE)))
        print(f"[EXT-LS] λ={lam:.3e}, cond(NE)≈{condNE:.3e}, "
              f"||W^{0.5}(Aα+h)||₂≈{res_norm:.3e}")

    return alpha, res_norm


# ---------------------------------------------------------------------------
# Evaluator construction: φ and ∇φ from MFS + MV part (world coordinates)
# ---------------------------------------------------------------------------

def build_evaluators_mfs(Pn: jnp.ndarray,
                         Yn: jnp.ndarray,
                         alpha: jnp.ndarray,
                         a: jnp.ndarray,
                         grad_t_fn,
                         grad_p_fn,
                         scinfo: ScaleInfo):
    """Build φ(X) and ∇φ(X) evaluators in world coordinates for φ(x) = φ_mv + ψ.

    ψ(x) = Σ_j α_j G(Xn(x), Yn_j) where Xn = (X - center)*scale, Yn normalized.

    φ_mv(x) is represented only by its gradient through the multivalued bases:

      ∇φ_mv(X) = scale * [ a_t grad_t(Xn) + a_p grad_p(Xn) ].
    """
    Y = Yn

    @jit
    def S_alpha_at(xn: jnp.ndarray) -> jnp.ndarray:
        Gvals = vmap(lambda y: green_G(xn, y))(Y)
        return jnp.dot(Gvals, alpha)

    @jit
    def grad_S_alpha_at(xn: jnp.ndarray) -> jnp.ndarray:
        Grads = vmap(lambda y: grad_green_x(xn, y))(Y)  # (M,3)
        return jnp.sum(Grads * alpha[:, None], axis=0)

    S_batch = vmap(S_alpha_at)
    dS_batch = vmap(grad_S_alpha_at)

    def grad_mv_world(X: jnp.ndarray) -> jnp.ndarray:
        Xn = (X - scinfo.center) * scinfo.scale
        return scinfo.scale * (a[0] * grad_t_fn(Xn) +
                               a[1] * grad_p_fn(Xn))

    @jit
    def psi_world(X: jnp.ndarray) -> jnp.ndarray:
        Xn = (X - scinfo.center) * scinfo.scale
        return S_batch(Xn)

    @jit
    def grad_psi_world(X: jnp.ndarray) -> jnp.ndarray:
        Xn = (X - scinfo.center) * scinfo.scale
        return scinfo.scale * dS_batch(Xn)

    @jit
    def phi_world(X: jnp.ndarray) -> jnp.ndarray:
        # MV part only contributes a constant in φ (we keep it implicit),
        # so we only return ψ(x) here. In practice only ∇φ is used for B.
        return psi_world(X)

    def grad_phi_world(X: jnp.ndarray) -> jnp.ndarray:
        return grad_mv_world(X) + grad_psi_world(X)

    return phi_world, grad_phi_world


# ---------------------------------------------------------------------------
# Interior evaluator reconstruction from checkpoint
# ---------------------------------------------------------------------------

def build_interior_evaluator_from_checkpoint(center: np.ndarray,
                                             scale: float,
                                             Yn_in: np.ndarray,
                                             alpha_in: np.ndarray,
                                             a: np.ndarray,
                                             a_hat: np.ndarray,
                                             P: np.ndarray,
                                             N: np.ndarray):
    """Reconstruct the interior field evaluators from a checkpoint.

    This uses the same data representation as main.py: Yn_in are normalized
    source points, center/scale are used to normalize world coordinates, and
    a,a_hat define the multivalued basis.

    Returns
    -------
    scinfo : ScaleInfo
    phi_in_fn : callable(X) -> φ_in(X)
    grad_in_fn : callable(X) -> ∇φ_in(X)
    """
    center_j = jnp.asarray(center, dtype=jnp.float64)
    scale_j = jnp.asarray(scale, dtype=jnp.float64)
    scinfo = ScaleInfo(center=center_j, scale=scale_j)

    Pn = (jnp.asarray(P) - scinfo.center) * scinfo.scale
    Nn = jnp.asarray(N)
    a_j = jnp.asarray(a, dtype=jnp.float64)
    a_hat_j = jnp.asarray(a_hat, dtype=jnp.float64)
    Yn_in_j = jnp.asarray(Yn_in, dtype=jnp.float64)

    grad_t, grad_p = multivalued_bases_about_axis(Pn, Nn, a_hat_j)
    phi_in_fn, grad_in_fn = build_evaluators_mfs(
        Pn=Pn,
        Yn=Yn_in_j,
        alpha=jnp.asarray(alpha_in, dtype=jnp.float64),
        a=a_j,
        grad_t_fn=grad_t,
        grad_p_fn=grad_p,
        scinfo=scinfo
    )

    return scinfo, phi_in_fn, grad_in_fn, grad_t, grad_p


# ---------------------------------------------------------------------------
# Optional diagnostics
# ---------------------------------------------------------------------------

def diagnostics_plot_boundary_residual(P, N, grad_in, grad_out):
    if not HAVE_MPL:
        print("[DIAG] Matplotlib not available; skipping plots.")
        return

    P_np = np.asarray(P)
    N_np = np.asarray(N)
    gin = np.sum(N_np * np.asarray(grad_in), axis=1)
    gout = np.sum(N_np * np.asarray(grad_out), axis=1)
    diff = gout - gin

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax1.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                     c=np.abs(diff), s=6, cmap='magma')
    fig.colorbar(sc, ax=ax1, shrink=0.7,
                 label=r"$|n\cdot(\nabla\phi_{\rm out}-\nabla\phi_{\rm in})|$")
    ax1.set_title("Normal-component mismatch on Γ")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(np.abs(diff), bins=50, alpha=0.9)
    ax2.set_xlabel(r"$|n\cdot(\nabla\phi_{\rm out}-\nabla\phi_{\rm in})|$")
    ax2.set_ylabel("count")
    ax2.set_title("Histogram of normal-component mismatch")

    plt.tight_layout()
    
def _orthonormal_complement(a_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a_hat, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-30)
    t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = t - np.dot(t, a) * a
    e1 /= (np.linalg.norm(e1) + 1e-30)
    e2 = np.cross(a, e1)
    e2 /= (np.linalg.norm(e2) + 1e-30)
    return e1, e2

def plot_neumann_profile_along_chord(P, N,
                                     grad_in_on_Gamma,
                                     grad_out_on_Gamma,
                                     center,
                                     a_hat,
                                     kind: str = "torus"):
    """
    1D profile of n·B_in, n·B_out, and their difference along a chord
    around the device (good for papers: clean, readable line plot).

    The chord direction is chosen orthogonal to a_hat, and the
    abscissa is the coordinate u1 along that direction.
    """
    if not HAVE_MPL:
        print("[DIAG] Matplotlib not available; skipping Neumann profile plot.")
        return

    P = np.asarray(P); N = np.asarray(N)
    g_in  = np.asarray(grad_in_on_Gamma)
    g_out = np.asarray(grad_out_on_Gamma)

    # Normal components
    Bn_in  = np.sum(N * g_in,  axis=1)
    Bn_out = np.sum(N * g_out, axis=1)
    dBn    = Bn_out - Bn_in

    # Build local axes ⟂ a_hat and project boundary points
    center = np.asarray(center)
    a_hat  = np.asarray(a_hat)
    e1, e2 = _orthonormal_complement(a_hat)

    X  = P - center[None, :]
    u1 = X @ e1
    u2 = X @ e2
    s  = X @ (a_hat / (np.linalg.norm(a_hat) + 1e-30))

    # Restrict to a thin strip around the chord (same logic as seeds)
    u2_span = np.percentile(np.abs(u2), 99.0) + 1e-12
    s_span  = np.percentile(np.abs(s),  99.0) + 1e-12
    u2_tol  = 0.03 * u2_span
    if kind.lower() == "torus":
        s_tol = 0.10 * s_span
        mask  = (np.abs(u2) <= u2_tol) & (np.abs(s) <= s_tol)
    else:
        mask  = (np.abs(u2) <= u2_tol)

    if not np.any(mask):
        mask = np.ones_like(u1, dtype=bool)

    u1_sel   = u1[mask]
    Bn_in_sel  = Bn_in[mask]
    Bn_out_sel = Bn_out[mask]
    dBn_sel    = dBn[mask]

    # Sort by u1 (gives a nice smooth-ish curve)
    order = np.argsort(u1_sel)
    u1s   = u1_sel[order]
    BnI   = Bn_in_sel[order]
    BnO   = Bn_out_sel[order]
    dBnO  = dBn_sel[order]

    apply_paper_style()
    fig, ax = plt.subplots()

    ax.plot(u1s, BnI,  lw=1.2, label=r"$n\cdot B_{\rm in}$")
    ax.plot(u1s, BnO,  lw=1.2, ls="--", label=r"$n\cdot B_{\rm out}$")
    ax.plot(u1s, dBnO, lw=1.0, ls=":",  label=r"$n\cdot(B_{\rm out}-B_{\rm in})$")

    ax.set_xlabel(r"Chord coordinate $u_1$ (a.u.)")
    ax.set_ylabel(r"$n\cdot B$")
    ax.set_title(r"Normal field continuity along a chord")
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("neumann_profile_along_chord.png", dpi=300)
    print("[PLOT] Saved neumann_profile_along_chord.png")

def plot_neumann_relative_error_stats(P,
                                      grad_in_on_Gamma,
                                      grad_out_on_Gamma,
                                      eps_rel: float = 1e-12):
    """
    Publication-ready histogram + CDF of pointwise relative Neumann mismatch:

        err_rel = |n·(B_out - B_in)| / max(|n·B_in|, eps_rel).

    Shows log10(err_rel) histogram and its cumulative distribution.
    """
    if not HAVE_MPL:
        print("[DIAG] Matplotlib not available; skipping error stats plot.")
        return

    P = np.asarray(P)
    g_in  = np.asarray(grad_in_on_Gamma)
    g_out = np.asarray(grad_out_on_Gamma)

    # Here we only need normal direction; reconstruct N from g_in if desired,
    # but better to pass N explicitly if you like. For now assume we only
    # care about mismatch magnitude projected along the *local* field direction
    # of B_in (or you can pass N explicitly if you prefer).
    # If you want explicit normals, just add N as argument and use that:
    #   Bn_in = sum(N*g_in, axis=1)
    # Here we just project mismatch on g_in direction as a proxy:
    B_in_mag = np.linalg.norm(g_in, axis=1) + eps_rel
    # component of mismatch along B_in direction:
    dB = g_out - g_in
    proj = np.sum(dB * g_in, axis=1) / B_in_mag
    err_rel = np.abs(proj) / B_in_mag

    err_rel = np.maximum(err_rel, eps_rel)
    log_err = np.log10(err_rel)

    apply_paper_style()
    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Histogram of log10(err_rel)
    bins = 40
    ax_hist.hist(log_err, bins=bins, density=True, alpha=0.8, edgecolor="k")
    ax_hist.set_xlabel(r"$\log_{10} \varepsilon_{\rm rel}$")
    ax_hist.set_ylabel("PDF")
    ax_hist.set_title("Relative Neumann mismatch (PDF)")

    # CDF
    log_sorted = np.sort(log_err)
    y = np.linspace(0.0, 1.0, log_sorted.size, endpoint=False)
    ax_cdf.plot(log_sorted, y, lw=1.4)
    ax_cdf.set_xlabel(r"$\log_{10} \varepsilon_{\rm rel}$")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("Relative Neumann mismatch (CDF)")
    ax_cdf.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("neumann_relative_error_stats.png", dpi=300)
    print("[PLOT] Saved neumann_relative_error_stats.png")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    default_solution = "wout_precise_QA_solution.npz"
    # default_solution = "wout_precise_QH_solution.npz"
    # default_solution = "wout_SLAM_4_coils_solution.npz"
    # default_solution = "wout_SLAM_6_coils_solution.npz"
    # default_solution = "knot_tube_solution.npz"

    parser = argparse.ArgumentParser(
        description="Exterior vacuum-field MFS solver consistent with "
                    "an interior MFS solution checkpoint."
    )
    parser.add_argument("mfs_in", nargs="?", default=resolve_npz_file_location(default_solution),
                        help="MFS solution checkpoint (*.npz) containing center, scale, Yn, alpha, a, a_hat, P, N")
    parser.add_argument("--k-nn", type=int, default=48,
                        help="k for kNN geometry stats (boundary weights)")
    parser.add_argument("--lambda-candidates", type=float, nargs="+",
                        default=[1e-6, 1e-4, 1e-2],
                        help="Candidate λ values for Tikhonov regularization")
    parser.add_argument("--out", default=None,
                        help="Output .npz for exterior solution "
                             "(default: <input_stem>_exterior_solution.npz)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable diagnostic plots even if matplotlib "
                             "is available.")
    args = parser.parse_args()

    in_path = Path(args.mfs_in).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {in_path}")

    if args.out is None:
        out_path = in_path.with_name(in_path.stem + "_exterior_solution.npz")
    else:
        out_path = Path(args.out).resolve()

    print(f"[IO] Reading interior checkpoint: {in_path}")
    data = np.load(in_path, allow_pickle=True)

    center = data["center"]
    scale = float(data["scale"])
    Yn_in = data["Yn"]
    alpha_in = data["alpha"]
    a = data["a"]
    a_hat = data["a_hat"]
    P = data["P"]
    N = data["N"]
    kind = str(data["kind"])

    print(f"[IO] Loaded: center={center}, scale={scale:.6g}")
    print(f"[IO] Boundary points: P.shape={P.shape}, normals N.shape={N.shape}")
    print(f"[IO] Interior sources: Yn.shape={Yn_in.shape}, alpha.shape={alpha_in.shape}")
    print(f"[IO] Multivalued a={a}, a_hat={a_hat}, kind={kind}")

    # 1) Reconstruct interior evaluators and boundary Neumann data
    scinfo, phi_in_fn, grad_in_fn, grad_t, grad_p = \
        build_interior_evaluator_from_checkpoint(center, scale,
                                                 Yn_in, alpha_in,
                                                 a, a_hat, P, N)

    P_j = jnp.asarray(P, dtype=jnp.float64)
    N_j = jnp.asarray(N, dtype=jnp.float64)

    print("[INT] Evaluating interior gradient on Γ ...")
    grad_in_on_Gamma = grad_in_fn(P_j)
    Bn_target = jnp.sum(N_j * grad_in_on_Gamma, axis=1)

    # 2) Normalize geometry and build weights
    Pn = (P_j - scinfo.center) * scinfo.scale
    Nn = N_j
    W, rk = kNN_geometry_stats(np.asarray(Pn), k=args.k_nn)

    print(f"[GEOM] k-NN area weights built with k={args.k_nn}. "
          f"Total area ≈ {float(jnp.sum(W)):.6g}")

    # 3) Build interior fictitious sources for the *exterior* problem
    print("[SRC] Building interior fictitious source surface for exterior φ ...")
    Yn_ext = build_mfs_sources_interior(Pn, Nn, jnp.asarray(Yn_in))

    # 4) Build Neumann system matrix for exterior ψ_out
    print("[SYS] Assembling Neumann collocation matrix A for exterior problem ...")
    A_ext = build_system_matrix_neumann(Pn, Nn, Yn_ext, scinfo)

    # 5) Compute multivalued contribution to Neumann data on Γ
    print("[MV] Evaluating multivalued Neumann contribution on Γ ...")
    grad_t_bdry = grad_t(Pn)
    grad_p_bdry = grad_p(Pn)
    g_mv = scinfo.scale * jnp.sum(Nn * (a[0] * grad_t_bdry +
                                        a[1] * grad_p_bdry), axis=1)

    # 6) Build right-hand mismatch h_raw = g_mv - Bn_target
    h_raw = g_mv - Bn_target

    # 7) Solve for α_ext using a small set of λ candidates
    print("[SOL] Solving exterior MFS system for α_ext ...")
    best_alpha = None
    best_res = np.inf
    best_lam = None

    for lam in args.lambda_candidates:
        alpha_ext, res_norm = solve_alpha_tikhonov(A_ext, W, h_raw,
                                                   lam=lam, verbose=True)
        if res_norm < best_res:
            best_res = res_norm
            best_alpha = alpha_ext
            best_lam = lam

    alpha_ext = best_alpha
    print(f"[SOL] Selected λ*={best_lam:.3e} with ||W^0.5(Aα+h)||₂≈{best_res:.3e}")
    print(f"[SOL] ||α_ext||₂={float(jnp.linalg.norm(alpha_ext)):.3e}")

    # 8) Build exterior evaluators and check consistency on Γ
    phi_out_fn, grad_out_fn = build_evaluators_mfs(
        Pn=Pn,
        Yn=Yn_ext,
        alpha=alpha_ext,
        a=jnp.asarray(a, dtype=jnp.float64),
        grad_t_fn=grad_t,
        grad_p_fn=grad_p,
        scinfo=scinfo
    )

    print("[CHK] Evaluating exterior gradient on Γ ...")
    grad_out_on_Gamma = grad_out_fn(P_j)
    Bn_out = jnp.sum(N_j * grad_out_on_Gamma, axis=1)

    mismatch = Bn_out - Bn_target
    mis_l2 = float(jnp.linalg.norm(mismatch))
    mis_linf = float(jnp.max(jnp.abs(mismatch)))
    print(f"[CHK] Neumann mismatch on Γ: L2≈{mis_l2:.3e}, Linf≈{mis_linf:.3e}")

    # Optional diagnostics plots
    if not args.no_plot:
        diagnostics_plot_boundary_residual(P, N, grad_in_on_Gamma, grad_out_on_Gamma)
        plot_neumann_profile_along_chord(
            P=np.asarray(P),
            N=np.asarray(N),
            grad_in_on_Gamma=np.asarray(grad_in_on_Gamma),
            grad_out_on_Gamma=np.asarray(grad_out_on_Gamma),
            center=np.asarray(center),
            a_hat=np.asarray(a_hat),
            kind=kind,
        )
        plot_neumann_relative_error_stats(
            P=np.asarray(P),
            grad_in_on_Gamma=np.asarray(grad_in_on_Gamma),
            grad_out_on_Gamma=np.asarray(grad_out_on_Gamma),
        )
        plt.show()


    # 9) Save exterior checkpoint (same layout as interior solver)
    out_dir = out_path.parent
    os.makedirs(out_dir, exist_ok=True)
    print(f"[SAVE] Writing exterior MFS checkpoint → {out_path}")
    np.savez(
        out_path,
        center=np.asarray(scinfo.center, dtype=float),
        scale=float(np.asarray(scinfo.scale)),
        Yn=np.asarray(Yn_ext, dtype=float),   # interior sources for exterior φ
        alpha=np.asarray(alpha_ext, dtype=float),
        a=np.asarray(a, dtype=float),
        a_hat=np.asarray(a_hat, dtype=float),
        P=np.asarray(P, dtype=float),
        N=np.asarray(N, dtype=float),
        kind=str(kind),
    )
    print("[DONE] Exterior vacuum-field MFS solution written successfully.")


if __name__ == "__main__":
    main()
