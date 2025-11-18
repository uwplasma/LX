#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy-minimizing vacuum field via Method of Fundamental Solutions (MFS)
=======================================================================

Check README.md

Usage
-----

  python bim_mfs.py \
      --xyz inputs/wout_precise_QH.csv \
      --normals inputs/wout_precise_QH_normals.csv \
      --phiedge 1.0

All outputs (checkpoint .npz and PNG figures) are written to ../outputs.

Author: (your name / affiliation)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from functools import partial
import numpy as np
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from jax import debug as jdbg

from main_helpers import (diagnostics_on_inner_shell, compute_axis_coordinates,
                          numerical_laplacian_phi_s, vec_stats, get_candidates)

# ----------------------------------------------------------------------
# Geometry loading & normalization (I/O: numpy ok, solver: jax)
# ----------------------------------------------------------------------

def load_surface_xyz_normals(xyz_csv, normals_csv, verbose=True):
    """
    Load surface point cloud P and normals N from CSV, and normalize normals.

    Expected CSV format: first row header, then 3 columns (x,y,z) or (nx,ny,nz).
    """
    P = np.loadtxt(xyz_csv, delimiter=",", skiprows=1)
    N = np.loadtxt(normals_csv, delimiter=",", skiprows=1)
    assert P.shape[1] == 3 and N.shape[1] == 3, "CSV must have 3 columns"
    nrm = N / np.maximum(1e-15, np.linalg.norm(N, axis=1, keepdims=True))
    if verbose:
        print(f"[LOAD] points:   {P.shape}")
        print(f"[LOAD] normals:  {N.shape}")
        print(f"[LOAD] extents (min..max) per axis:")
        for k, nm in enumerate("xyz"):
            print(f"       {nm}: {P[:,k].min():.6g} .. {P[:,k].max():.6g}")
        nlen = np.linalg.norm(nrm, axis=1)
        print(f"[LOAD] normal lengths: min={nlen.min():.3g}, "
              f"max={nlen.max():.3g}, mean={nlen.mean():.3g}")
    return jnp.asarray(P, dtype=jnp.float64), jnp.asarray(nrm, dtype=jnp.float64)


@jax.tree_util.register_pytree_node_class
class ScaleInfo:
    """
    Center + scale as a PyTree, for easy passing into JAX-jitted code.

    We normalize coordinates so that the median radius from the center
    is O(1). This is used for multivalued bases; the MFS itself is done
    in *physical* coordinates.
    """
    def __init__(self, center, scale):
        self.center = jnp.asarray(center)
        self.scale  = jnp.asarray(scale)

    def tree_flatten(self):
        return ((self.center, self.scale), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        center, scale = children
        return cls(center, scale)


def normalize_geometry(P, verbose=True):
    """
    Translate + scale so that median radius from center is ~1 (normalized space).

    We retain both world coordinates (P) and normalized coordinates (Pn) for:
      - MFS in world coordinates
      - multivalued bases built in normalized coordinates.
    """
    c = jnp.mean(P, axis=0)
    r = jnp.linalg.norm(P - c, axis=1)
    r_med = jnp.median(r)
    s = 1.0 / jnp.maximum(r_med, 1e-12)
    Pn = (P - c) * s
    if verbose:
        print(f"[SCALE] center = {np.array(c)}")
        print(f"[SCALE] median radius = {float(r_med):.6g} -> scale={float(s):.6g}")
    return Pn, ScaleInfo(center=c, scale=s)


def maybe_flip_normals(P, N, verbose=True):
    """
    Ensure outward normals: require <(P-c)·N> > 0 on average.

    If average (P-c)·N is negative, flip all normals.
    """
    c = jnp.mean(P, axis=0)
    s = jnp.sum((P - c) * N, axis=1)
    avg = float(jnp.mean(s))
    if avg < 0:
        if verbose: print(f"[ORIENT] Normals inward on average (⟨(P-c)·N⟩≈{avg:.3e}) → flipping.")
        return -N
    if verbose: print(f"[ORIENT] Normals seem outward (⟨(P-c)·N⟩≈{avg:.3e}).")
    return N


# ----------------------------------------------------------------------
# Pairwise geometry & area weights (for diagnostics, flux integrals)
# ----------------------------------------------------------------------

@jit
def pairwise_dist2(P):
    """Pairwise squared distances D_ij = |P_i - P_j|^2, P∈R^{N×3}."""
    Pi = P[:, None, :]   # (N,1,3)
    Pj = P[None, :, :]   # (1,N,3)
    diff = Pi - Pj       # (N,N,3)
    return jnp.sum(diff*diff, axis=-1)  # (N,N)


@partial(jit, static_argnames=("verbose"))
def estimate_area_weights_knn(P, k=32, verbose=True):
    """
    Estimate patch areas and local spacing from a k-NN heuristic.

    Returns
    -------
    W : (N,) array
        Crude patch areas.
    h : (N,) array
        Local spacing ~ distance to k-th nearest neighbor.

    This is used ONLY for flux integrals and diagnostics, not to define
    the MFS operator itself.
    """
    k_int = jnp.asarray(k, dtype=jnp.int32)

    Npts = P.shape[0]
    D2 = pairwise_dist2(P)              # (N,N)
    big = jnp.max(D2) + 1.0
    D2 = D2 + jnp.eye(Npts, dtype=D2.dtype) * big  # kill self-distance

    D2_sorted = jnp.sort(D2, axis=1)
    h = jnp.sqrt(D2_sorted[:, k_int - 1])     # (N,)

    # Each disk of radius h contains ~k points ⇒ area ≈ k * (true patch area).
    # Divide by k so that ∑ W ≈ total surface area.
    k_float = k_int.astype(P.dtype)
    W = (jnp.pi * h * h) / k_float

    if verbose:
        jdbg.print("[QUAD] k={k}, h stats: min={mn:.3e}, med={md:.3e}, max={mx:.3e}",
                k=k_int, mn=jnp.min(h), md=jnp.median(h), mx=jnp.max(h))
        jdbg.print("[QUAD] area weights: sum W ≈ {sw:.3e}", sw=jnp.sum(W))
    return W, h


# ----------------------------------------------------------------------
# MFS: fictitious outer source surface and Neumann matrix
# ----------------------------------------------------------------------

@partial(jit, static_argnames=("verbose"))
def build_mfs_sources(P, N, h_world, source_offset_factor=2.0, verbose=True):
    """
    Construct MFS source points Y on a fictitious outer surface Γ^d:

        y_i = x_i + d_i n_i,   d_i = source_offset_factor * h_i,

    where h_i is a local spacing estimate in *world* coordinates.

    For strictly interior problems with outward normals, this places
    sources outside the domain, avoiding singularities in G and ∇G.
    """
    P = jnp.asarray(P)
    N = jnp.asarray(N)
    h_world = jnp.asarray(h_world)

    d = source_offset_factor * h_world    # (N,)
    Y = P + d[:, None] * N

    if verbose:
        jdbg.print("[MFS] source_offset_factor={f:.3f}", f=source_offset_factor)
        jdbg.print("[MFS] d stats: min={mn:.3e}, med={md:.3e}, max={mx:.3e}",
                mn=jnp.min(d), md=jnp.median(d), mx=jnp.max(d))
    return Y

@partial(jit, static_argnames=("verbose",))
def build_mfs_neumann_matrix(P, N, Y, verbose=True):
    """
    Build dense MFS Neumann matrix A for n·∇φ_s:

        A_ij = n_i · ∇_x G(x_i, y_j),

    where   G(x,y) = 1/(4π|x-y|),
            ∇_x G = -(x - y)/(4π|x-y|^3).

    Inputs
    ------
    P : (N,3)
        Boundary points x_i (world coordinates).
    N : (N,3)
        Outward unit normals n_i.
    Y : (M,3)
        MFS source points y_j on fictitious outer surface.

    Returns
    -------
    A : (N,M)
        Dense Neumann MFS matrix.
    """
    X = jnp.asarray(P)   # (N,3)
    Nw = jnp.asarray(N)  # (N,3)
    Y_src = jnp.asarray(Y)  # (M,3)

    Xi = X[:, None, :]       # (N,1,3)
    Yj = Y_src[None, :, :]   # (1,M,3)
    diff = Xi - Yj           # (N,M,3), x_i - y_j

    r2 = jnp.sum(diff * diff, axis=-1)      # (N,M)
    r = jnp.sqrt(jnp.maximum(r2, 1e-24))    # avoid r=0
    r3 = r2 * r                             # (N,M)

    # ∇_x G = -(x - y)/(4π |x-y|^3)
    gradG = -diff / (4.0 * jnp.pi * r3[..., None])  # (N,M,3)

    # n_i · ∇G at each (i,j)
    n_dot_grad = jnp.sum(Nw[:, None, :] * gradG, axis=-1)  # (N,M)

    if verbose:
        jdbg.print("[MFS] A shape = ({n},{m})", n=n_dot_grad.shape[0], m=n_dot_grad.shape[1])
        jdbg.print("[MFS] |A| stats: min={mn:.3e}, med={md:.3e}, max={mx:.3e}",
                mn=jnp.min(jnp.abs(n_dot_grad)),
                md=jnp.median(jnp.abs(n_dot_grad)),
                mx=jnp.max(jnp.abs(n_dot_grad)))
    return n_dot_grad

# ----------------------------------------------------------------------
# MFS field evaluators: φ_s and B_s
# ----------------------------------------------------------------------

def make_mfs_single_layer_evaluators(Y_src, coeffs):
    """
    Build batched evaluators φ_s(X) and ∇φ_s(X) for an MFS solution:

        φ_s(x) = ∑_j c_j G(x,y_j),
        B_s(x) = ∇φ_s(x) = ∑_j c_j ∇_x G(x,y_j).

    Inputs
    ------
    Y_src  : (M,3)
        Source points y_j.
    coeffs : (M,)
        Coefficients c_j.

    Returns
    -------
    phi_s_batch(X) : (K,)
        Values of φ_s at K points.
    grad_s_batch(X): (K,3)
        Values of ∇φ_s at K points.
    """
    Y_src = jnp.asarray(Y_src)
    c = jnp.asarray(coeffs)

    @jit
    def phi_s_at_point(x):
        diff = x[None, :] - Y_src      # (M,3)
        r = jnp.linalg.norm(diff, axis=-1)
        r = jnp.maximum(r, 1e-24)
        Gvals = 1.0 / (4.0 * jnp.pi * r)
        return jnp.dot(c, Gvals)

    @jit
    def grad_phi_s_at_point(x):
        diff = x[None, :] - Y_src      # (M,3)
        r2 = jnp.sum(diff * diff, axis=-1)
        r2 = jnp.maximum(r2, 1e-24)
        r = jnp.sqrt(r2)
        r3 = r2 * r
        gradG = -diff / (4.0 * jnp.pi * r3[:, None])  # (M,3)
        return jnp.einsum("m,mk->k", c, gradG)        # (3,)

    phi_s_batch  = jit(vmap(phi_s_at_point, in_axes=(0,)))
    grad_s_batch = jit(vmap(grad_phi_s_at_point, in_axes=(0,)))

    return phi_s_batch, grad_s_batch


def make_B_mv_evaluator(a, grad_t_fn, grad_p_fn, scinfo: ScaleInfo):
    """
    Build the multivalued field evaluator B_mv for a given coefficient a:

        B_mv(x) = scale * [ a_t grad_t(Xn) + a_p grad_p(Xn) ],
        Xn = (x - center) * scale,
    """
    a = jnp.asarray(a)
    center = scinfo.center
    scale = scinfo.scale

    @jit
    def B_mv_fn(X):
        Xn = (X - center[None, :]) * scale
        Gt = grad_t_fn(Xn)
        Gp = grad_p_fn(Xn)
        return scale * (a[0] * Gt + a[1] * Gp)

    return B_mv_fn


def make_total_field_evaluators_for_fixed_a_mfs(
    Y_src, coeffs,
    scinfo: ScaleInfo,
    a, grad_t_fn, grad_p_fn
):
    """
    Given MFS sources and coefficients, plus multivalued bases with
    coefficient a, build total field evaluators:

        phi_s_fn(X): single-valued MFS potential φ_s(X)
        B_fn(X):     full B = B_mv + B_s
        B_mv_fn(X):  multivalued part only
        B_s_fn(X):   MFS part only
    """
    phi_s_fn, grad_s_fn = make_mfs_single_layer_evaluators(Y_src, coeffs)
    B_mv_fn = make_B_mv_evaluator(a, grad_t_fn, grad_p_fn, scinfo)

    @jit
    def B_s_fn(X):
        return grad_s_fn(X)

    @jit
    def B_fn(X):
        return B_mv_fn(X) + B_s_fn(X)

    return phi_s_fn, B_fn, B_mv_fn, B_s_fn


# ----------------------------------------------------------------------
# Basis solves and energy/flux functionals
# ----------------------------------------------------------------------

def solve_basis_fields_mfs(
    P, N, Pn, scinfo: ScaleInfo,
    grad_t_fn, grad_p_fn,
    Y_src, W, reg=1e-8,
    use_weighted_ls: bool = True,
    verbose: bool = True,
):
    """
    Solve the MFS Neumann system for the two multivalued bases:

      a^(1) = (1,0), a^(2) = (0,1)

    We compute n·B_mv on the boundary for each basis, then solve

        A c^(k) = -g^(k),   k=1,2,

    where A is the MFS Neumann matrix (n·∇G).

    Returns
    -------
      (Dt, Dp)             : boundary data n·B_t, n·B_p
      (c1, c2)             : MFS coefficients for the two bases
      (B_t_bdry, B_p_bdry) : multivalued fields on boundary (for diagnostics)
    """
    # Multivalued basis fields on boundary (normalized -> world)
    Gt_bdry = grad_t_fn(Pn)   # (N,3) in normalized coords
    Gp_bdry = grad_p_fn(Pn)
    B_t_bdry = scinfo.scale * Gt_bdry
    B_p_bdry = scinfo.scale * Gp_bdry

    # Raw normal fluxes on the boundary
    Dt_raw = jnp.sum(N * B_t_bdry, axis=1)
    Dp_raw = jnp.sum(N * B_p_bdry, axis=1)

    # Enforce Neumann compatibility: ∫ n·B dS = 0 (weighted by W)
    Wsum = jnp.sum(W)
    Dt_mean = jnp.dot(W, Dt_raw) / jnp.maximum(Wsum, 1e-30)
    Dp_mean = jnp.dot(W, Dp_raw) / jnp.maximum(Wsum, 1e-30)
    Dt = Dt_raw - Dt_mean
    Dp = Dp_raw - Dp_mean

    if verbose:
        vec_stats("[BASIS] g_t_raw = n·B_t", Dt_raw)
        vec_stats("[BASIS] g_p_raw = n·B_p", Dp_raw)
        vec_stats("[BASIS] g_t (zero-flux)", Dt)
        vec_stats("[BASIS] g_p (zero-flux)", Dp)
        print(f"[BASIS] weighted mean g_t_raw = {float(Dt_mean):.3e}")
        print(f"[BASIS] weighted mean g_p_raw = {float(Dp_mean):.3e}")

    # Assemble MFS matrix once
    A = build_mfs_neumann_matrix(P, N, Y_src, verbose=verbose)
    Npts, Msrc = A.shape
    assert Npts == Msrc, "[BASIS] Expect square MFS matrix (one source per boundary point)."

    # Optionally use area-weighted least squares:
    #   minimize ∑_i W_i (A c + g)_i^2 + reg ||c||^2
    #   ⇔ minimize ||A_w c + g_w||^2 + reg ||c||^2
    #   where A_w = diag(√W) A, g_w = diag(√W) g.
    if use_weighted_ls:
        sqrtW = jnp.sqrt(W)
        A_w = sqrtW[:, None] * A
        Dt_w = sqrtW * Dt
        Dp_w = sqrtW * Dp

        At = A_w.T
        AtA = At @ A_w
        rhs1 = -At @ Dt_w
        rhs2 = -At @ Dp_w
    else:
        At = A.T
        AtA = At @ A
        rhs1 = -At @ Dt
        rhs2 = -At @ Dp

    I = jnp.eye(Msrc, dtype=A.dtype)
    AtA_reg = AtA + reg * I

    if verbose:
        jdbg.print("[BASIS] MFS A shape   = ({n},{m})", n=Npts, m=Msrc)
        jdbg.print("[BASIS] MFS AtA shape = ({n},{m})", n=AtA.shape[0], m=AtA.shape[1])

    c1 = jnp.linalg.solve(AtA_reg, rhs1)
    c2 = jnp.linalg.solve(AtA_reg, rhs2)

    if verbose:
        jdbg.print("[BASIS] ||c1||_2 = {n1:.3e}, ||c2||_2 = {n2:.3e}",
                   n1=jnp.linalg.norm(c1), n2=jnp.linalg.norm(c2))

    return (Dt, Dp), (c1, c2), (B_t_bdry, B_p_bdry)


def build_basis_evaluators_mfs(
    P, N, Y_src,
    scinfo: ScaleInfo,
    grad_t_fn, grad_p_fn,
    c1, c2
):
    """
    Build field evaluators for the two MFS-based basis fields:

      Basis 1: a = (1,0)
      Basis 2: a = (0,1)

    Returns
    -------
      (phi1_fn, B1_fn, B1_mv_fn, B1_s_fn),
      (phi2_fn, B2_fn, B2_mv_fn, B2_s_fn)
    """
    a1 = jnp.array([1.0, 0.0])
    a2 = jnp.array([0.0, 1.0])

    phi1_fn, B1_fn, B1_mv_fn, B1_s_fn = make_total_field_evaluators_for_fixed_a_mfs(
        Y_src, c1, scinfo, a1, grad_t_fn, grad_p_fn
    )
    phi2_fn, B2_fn, B2_mv_fn, B2_s_fn = make_total_field_evaluators_for_fixed_a_mfs(
        Y_src, c2, scinfo, a2, grad_t_fn, grad_p_fn
    )

    return (phi1_fn, B1_fn, B1_mv_fn, B1_s_fn), (phi2_fn, B2_fn, B2_mv_fn, B2_s_fn)

# -----------------------------------------------------------------------------#
# Axis detection and axis-aware multivalued bases (for torus / knot geometries)
# -----------------------------------------------------------------------------#

def _orthonormal_complement(a_hat: np.ndarray):
    """
    Return two orthonormal vectors spanning the plane ⟂ a_hat.
    Deterministic construction using NumPy (host-side only).
    """
    a = np.asarray(a_hat, dtype=float)
    a /= np.linalg.norm(a) + 1e-30
    # pick any vector not parallel to a
    if abs(a[0]) < 0.9:
        t = np.array([1.0, 0.0, 0.0])
    else:
        t = np.array([0.0, 1.0, 0.0])
    e1 = t - np.dot(t, a)*a
    e1 /= np.linalg.norm(e1) + 1e-30
    e2 = np.cross(a, e1)
    e2 /= np.linalg.norm(e2) + 1e-30
    return e1, e2

def detect_geometry_and_axis(Pn, verbose=True):
    """
    PCA-based geometry classification and axis detection.

    Input:
      Pn : (N,3) normalized points on the surface.

    Output:
      kind : "torus" or "mirror" (heuristic)
      a_hat: (3,) unit vector for geometric axis
      c    : (3,) center used for PCA
      s    : (3,) singular values (desc)
    """
    P = np.asarray(Pn)
    c = P.mean(axis=0)
    X = P - c
    # SVD: X ≈ U diag(S) V^T, rows of V^T are principal directions
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Largest → smallest variance directions:
    e1, e2, e3 = Vt  # already in descending singular-value order

    ratio_long = S[0] / max(S[1], 1e-12)
    ratio_thin = S[1] / max(S[2], 1e-12)

    if ratio_long > 2.0 and ratio_thin < 1.8:
        kind = "mirror"   # one long direction >> two comparable transverse
        axis = e1
    elif ratio_thin > 2.0 and ratio_long < 1.8:
        kind = "torus"    # thin shell: two wide directions >> one thin
        axis = e3
    else:
        # Ambiguous: default to torus-like choice with smallest-variance normal
        kind = "torus"
        axis = e3

    a_hat = axis / (np.linalg.norm(axis) + 1e-30)

    if verbose:
        print(f"[GEOM] s (desc) = {S}")
        print(f"[GEOM] ratio_long={ratio_long:.2f}, ratio_thin={ratio_thin:.2f}")
        print(f"[GEOM] kind={kind}, a_hat={a_hat}")

    return (
        kind,
        jnp.asarray(a_hat, dtype=jnp.float64),
        jnp.asarray(c, dtype=jnp.float64),
        jnp.asarray(S, dtype=jnp.float64),
        jnp.asarray(Vt.T, dtype=jnp.float64),
    )

@jit
def grad_azimuth_about_axis(Xn, a_hat):
    """
    ∇ϕ_a for azimuth around an arbitrary unit axis a_hat.
      r_perp = X - (X·a)a
      ∇ϕ_a ≈ (a × r_perp) / max(|r_perp|^2, r2_min)
    """
    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par   = jnp.sum(Xn * a[None,:], axis=1, keepdims=True) * a[None,:]
    r_perp  = Xn - r_par

    # We normalized so that median radius ~1, so clipping at ρ_min ≈ 0.2 is reasonable.
    r2 = jnp.sum(r_perp*r_perp, axis=1, keepdims=True)
    r2_min = 0.2**2  # tune if needed, but 0.1–0.3 is a good range
    r2_clipped = jnp.maximum(r2, r2_min)

    cross   = jnp.cross(a[None,:], r_perp)
    return cross / r2_clipped  # (N,3)

def multivalued_bases_about_axis(Pn, Nn, a_hat, verbose=True):
    """
    Build axis-aware multivalued bases (φ_t,∇φ_t) and (φ_p,∇φ_p) around a_hat.

      - grad_t(X) = ∇ϕ_a(X): azimuth around axis a_hat (like toroidal direction).
      - grad_p(X) = θ̂(X): poloidal-like tangent direction built from normals.

    They are NOT required to be gradients of single-valued scalar fields globally
    (grad_p especially), which is what makes them robust on knots and tubes.
    """
    Pn_ref = jnp.asarray(Pn, dtype=jnp.float64)
    Nn_ref = jnp.asarray(Nn, dtype=jnp.float64)
    a_hat  = jnp.asarray(a_hat, dtype=jnp.float64)

    @jit
    def _nearest_normal_jax(Xn):
        # Brute-force nearest neighbor in normalized coordinates (OK at N~O(10^3))
        X2 = jnp.sum(Xn*Xn, axis=1, keepdims=True)
        P2 = jnp.sum(Pn_ref*Pn_ref, axis=1, keepdims=True)
        dist2 = X2 + P2.T - 2.0 * (Xn @ Pn_ref.T)
        idx = jnp.argmin(dist2, axis=1)
        return Nn_ref[idx, :]

    @jit
    def _unit(v, eps=1e-30):
        nrm = jnp.linalg.norm(v, axis=1, keepdims=True)
        return v / jnp.maximum(eps, nrm)

    @jit
    def _project_tangent(v, n):
        # project v onto tangent plane defined by normal n
        return v - jnp.sum(v*n, axis=1, keepdims=True)*n

    @jit
    def grad_t(Xn):
        # azimuthal gradient around axis a_hat
        return grad_azimuth_about_axis(Xn, a_hat)

    @jit
    def grad_p(Xn):
        """
        Poloidal-like direction:
          1) build φ̂_a = unit(a × r_perp)
          2) project φ̂_a to tangent plane → φ_tan
          3) θ̂ = unit(n × φ_tan)
        """
        n = _nearest_normal_jax(Xn)
        a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
        r_par   = jnp.sum(Xn * a[None,:], axis=1, keepdims=True) * a[None,:]
        r_perp  = Xn - r_par
        phi_hat = _unit(jnp.cross(a[None,:], r_perp))
        phi_tan = _unit(_project_tangent(phi_hat, n))
        theta_hat = _unit(jnp.cross(n, phi_tan))

        # Remove any component along the axis to suppress toroidal flux from B_p
        a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
        theta_hat = theta_hat - jnp.sum(theta_hat * a[None, :], axis=1, keepdims=True) * a[None, :]
        theta_hat = _unit(theta_hat)

        return theta_hat

    def phi_t(Xn):
        # Scalar MV value is unused in current pipeline; keep zero for interface.
        return jnp.zeros((Xn.shape[0],), dtype=jnp.float64)

    def phi_p(Xn):
        return jnp.zeros((Xn.shape[0],), dtype=jnp.float64)

    if verbose:
        print("[MV] Using axis-aware multivalued bases around detected axis a_hat.")

    return phi_t, grad_t, phi_p, grad_p


# ----------------------------------------------------------------------
# VMEC-like poloidal cross-section quadrature
# ----------------------------------------------------------------------

def build_poloidal_cross_section_quadrature(P, a_hat, E, scinfo,
                                            n_r=32, n_theta=64,
                                            margin_frac=0.15, verbose=True):
    """
    Build nodes and weights on a poloidal cross-section:

      - Plane orthogonal to a_hat, through the geometric center.
      - Polar grid (r, θ) inside the torus footprint, with
        r in [r_in, r_out] based on radial extent of boundary.

    Returns
    -------
      X_cs   : (Nq,3) cross-section points.
      w_area : (Nq,) area weights dS = r dr dθ.
      rho    : (Nq,) distances to the axis (>= r_in).
    """
    P = jnp.asarray(P)
    a = a_hat / jnp.maximum(jnp.linalg.norm(a_hat), 1e-30)
    center = scinfo.center

    # Use PCA directions e1,e2 as in-plane axes
    e1 = E[:, 0]
    e2 = E[:, 1]

    # Radial distances of boundary points from axis
    r_vec = P - center[None, :]
    r_par = jnp.sum(r_vec * a[None, :], axis=1, keepdims=True) * a[None, :]
    r_perp = r_vec - r_par
    rho_bdry = jnp.linalg.norm(r_perp, axis=1)

    # Inner/outer radii with some margin to stay well inside the boundary
    r_min = float(np.percentile(np.asarray(rho_bdry), margin_frac * 100.0))
    r_max = float(np.percentile(np.asarray(rho_bdry), (1.0 - margin_frac) * 100.0))
    if r_max <= r_min:
        raise RuntimeError("Degenerate radial range for cross-section.")

    n_r = int(n_r)
    n_theta = int(n_theta)
    dr = (r_max - r_min) / float(n_r)

    r_vals = r_min + (jnp.arange(n_r) + 0.5) * dr  # midpoints
    dtheta = 2.0 * np.pi / float(n_theta)
    theta_vals = jnp.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    rr, tt = jnp.meshgrid(r_vals, theta_vals, indexing="ij")  # (n_r,n_theta)

    rr_flat = rr.ravel()
    tt_flat = tt.ravel()

    cos_t = jnp.cos(tt_flat)
    sin_t = jnp.sin(tt_flat)

    u_hat = e1[None, :]
    v_hat = e2[None, :]

    offsets = rr_flat[:, None] * (cos_t[:, None] * u_hat + sin_t[:, None] * v_hat)
    X_cs = center[None, :] + offsets  # (Nq,3)

    # Area weights and distances to axis
    w_area = rr_flat * dr * dtheta
    rho = rr_flat

    if verbose:
        jdbg.print("[XS] r_min={rmin:.3e}, r_max={rmax:.3e}, n_r={nr}, n_theta={nt}",
                rmin=r_min, rmax=r_max, nr=n_r, nt=n_theta)
        jdbg.print("[XS] total area ≈ {A:.3e}", A=jnp.sum(w_area))

    return X_cs, w_area, rho


def build_energy_flux_matrices_on_cross_section(
    P,
    a_hat,
    E,
    scinfo: ScaleInfo,
    B1_fn, B2_fn,
    n_r=32, n_theta=64,
    margin_frac=0.15,
    verbose=True
):
    """
    Construct M (2x2) and c (2,) using a VMEC-like poloidal cross-section.

      - M encodes the 3D magnetic energy via volume weights
        w_vol = 2π ρ w_area.
      - c encodes the toroidal flux via Φ_tor = ∫_S B·a_hat dS.

    Returns
    -------
      M, c, X_cs, B1_cs, B2_cs
    """
    X_cs, w_area, rho = build_poloidal_cross_section_quadrature(
        P, a_hat, E, scinfo,
        n_r=n_r, n_theta=n_theta, margin_frac=margin_frac,
        verbose=verbose
    )

    B1_cs = B1_fn(X_cs)   # (Nq,3)
    B2_cs = B2_fn(X_cs)   # (Nq,3)

    # Volume weights for energy integral
    w_vol = 2.0 * np.pi * rho * w_area

    dot11 = jnp.sum(B1_cs * B1_cs, axis=1)
    dot22 = jnp.sum(B2_cs * B2_cs, axis=1)
    dot12 = jnp.sum(B1_cs * B2_cs, axis=1)

    M11 = jnp.dot(w_vol, dot11)
    M22 = jnp.dot(w_vol, dot22)
    M12 = jnp.dot(w_vol, dot12)
    M = jnp.array([[M11, M12],
                   [M12, M22]])

    # Toroidal flux vector c
    a_hat = a_hat / jnp.maximum(jnp.linalg.norm(a_hat), 1e-30)
    proj1 = jnp.sum(B1_cs * a_hat[None, :], axis=1)
    proj2 = jnp.sum(B2_cs * a_hat[None, :], axis=1)

    c1 = jnp.dot(w_area, proj1)
    c2 = jnp.dot(w_area, proj2)
    c = jnp.array([c1, c2])

    if verbose:
        vec_stats("[ENERGY] M matrix entries", np.array([M11, M22, M12]))
        vec_stats("[FLUX] c vector", np.array([c1, c2]))

    return M, c, X_cs, B1_cs, B2_cs


def solve_energy_minimizing_coeffs(M, c, phiedge, verbose=True):
    """
    Solve the 2×2 constrained minimization:

      minimize  ½ aᵀ M a
      s.t.      cᵀ a = phiedge.
    """
    Minv = jnp.linalg.inv(M)
    denom = float(c.T @ (Minv @ c))
    if abs(denom) < 1e-14:
        print("[WARN] cᵀ M⁻¹ c is very small; using fallback a=(phiedge,0).")
        return jnp.array([phiedge, 0.0])

    a_star = (phiedge / denom) * (Minv @ c)
    if verbose:
        print(f"[ENERGY] denom cᵀ M⁻¹ c = {denom:.6e}")
        print(f"[ENERGY] a* = (a_t, a_p) = ({float(a_star[0]):.6e}, {float(a_star[1]):.6e})")
    return a_star

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main(xyz_csv, normals_csv, k_nn=32, reg=1e-8, phiedge=1.0,
         source_offset_factor=2.0, mfs_out=None, verbose=True,
         xs_n_r=32, xs_n_theta=64, xs_margin_frac=0.15, eps_factor=0.3):

    print("========================================================")
    print(" MFS-based Neumann Laplace Solver (B = ∇φ)")
    print(" VMEC-like energy-minimizing vacuum field with fixed phiedge")
    print("========================================================")
    print(f"[PARAM] k_nn                = {k_nn}")
    print(f"[PARAM] reg (Tikhonov)      = {reg:.1e}")
    print(f"[PARAM] phiedge (Φ_tor)     = {phiedge:.6g}")
    print(f"[PARAM] source_offset_factor= {source_offset_factor:.3f}")
    print(f"[PARAM] eps_factor          = {eps_factor:.3f}")

    # Resolve paths and output directory
    script_dir = Path(__file__).resolve().parent
    xyz_path = Path(xyz_csv).resolve()
    base_name = xyz_path.stem
    out_dir = (script_dir / ".." / "outputs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PATHS] Output directory   : {out_dir}")
    
    # Geometry hint: treat any file with "knot" in the name as a knotatron
    base_lower = base_name.lower()
    geom_hint = "knot" if "knot" in base_lower else None
    if verbose and geom_hint == "knot":
        print(f"[GEOM] Detected 'knot' in base name '{base_name}' → using knot mode.")

    # Load surface geometry
    P, N = load_surface_xyz_normals(xyz_csv, normals_csv, verbose=verbose)
    N = maybe_flip_normals(P, N, verbose=verbose)
    Nn = N
    print(f"[PARAM] Total number of surface points= {len(P)}")

    # Normalize geometry (for multivalued bases)
    Pn, scinfo = normalize_geometry(P, verbose=verbose)

    # Area weights & spacings from normalized coords, rescaled to world units
    Wn, h_n = estimate_area_weights_knn(Pn, k=k_nn, verbose=verbose)
    scale = float(scinfo.scale)

    W = Wn / (scale**2)    # world-area weights
    h = h_n / scale        # world spacings
    h_min = float(jnp.min(h))
    h_med = float(jnp.median(h))
    if verbose:
        print(f"[QUAD] h_min={h_min:.3e}, h_med={h_med:.3e}")
        vec_stats("[QUAD] W (world)", W)

    # Decide geometry and axis using PCA on normalized surface
    kind, a_hat, center_pca, svals, E = detect_geometry_and_axis(Pn, verbose=verbose)

    # If the file name hinted this is a knot, still treat it as torus-like for bases
    if geom_hint == "knot":
        print(f"[GEOM] geom_hint='knot' → overriding kind to 'torus' but keeping PCA axis.")
        kind = "torus"  # it's a tube-like shell, just knotted
    else:
        print(f"[GEOM] geom_hint='{geom_hint}', detected kind='{kind}'.")
        
    # For knots, we will use a simpler prescription for a* (no VMEC-like minimization)
    use_simple_a_for_knot = (geom_hint == "knot")
    if use_simple_a_for_knot and verbose:
        print("[GEOM] Knot geometry: will use simple a* = (phiedge, 0) instead of VMEC-like minimization.")

    # Build axis-aware multivalued bases around a_hat
    phi_t, grad_t, phi_p, grad_p = multivalued_bases_about_axis(Pn, Nn, a_hat, verbose=verbose)

    # MFS sources on fictitious outer surface
    Y_src = build_mfs_sources(P, N, h, source_offset_factor=source_offset_factor, verbose=verbose)

    # --- Step 1: basis solves for (1,0) and (0,1) with MFS ---
    (Dt, Dp), (c1, c2), (B_t_bdry, B_p_bdry) = solve_basis_fields_mfs(
        P, N, Pn, scinfo,
        grad_t, grad_p,
        Y_src, W,
        reg=reg, verbose=verbose
    )
    if verbose:
        print(f"[BASIS] ||B_t||_avg = {jnp.linalg.norm(B_t_bdry)/float(B_t_bdry.shape[0]):.6e}, "
          f"||B_p||_avg = {jnp.linalg.norm(B_p_bdry)/float(B_p_bdry.shape[0]):.6e}")

    # Diagnostics: boundary residuals for each basis
    A = build_mfs_neumann_matrix(P, N, Y_src, verbose=verbose)
    n_dot_Bs1 = A @ c1
    n_dot_Bs2 = A @ c2
    res1 = Dt + n_dot_Bs1
    res2 = Dp + n_dot_Bs2
    if verbose:
        vec_stats("[BASIS] BC residual for basis 1 (n·B_mv + n·B_s)", res1)
        vec_stats("[BASIS] BC residual for basis 2 (n·B_mv + n·B_s)", res2)

    # --- Step 2: build basis field evaluators ---

    (phi1_fn, B1_fn, B1_mv_fn, B1_s_fn), (phi2_fn, B2_fn, B2_mv_fn, B2_s_fn) = \
        build_basis_evaluators_mfs(P, N, Y_src, scinfo, grad_t, grad_p, c1, c2)

    # --- Step 3: energy matrix M and flux vector c on poloidal cross-section ---

    M, c_vec, X_cs, B1_cs, B2_cs = build_energy_flux_matrices_on_cross_section(
        P, a_hat, E, scinfo,
        B1_fn, B2_fn,
        n_r=xs_n_r, n_theta=xs_n_theta, margin_frac=xs_margin_frac,
        verbose=verbose
    )

    # --- Step 4: solve constrained minimization for a* ---

    if use_simple_a_for_knot:
        # For knot geometries, avoid VMEC-like cross-section minimization,
        # and simply take a* to be purely "toroidal-like".
        a_star = jnp.array([phiedge, 0.0])
        if verbose:
            print(f"[ENERGY] (knot) overriding a* = (a_t, a_p) = ({phiedge:.6e}, 0.0)")
    else:
        a_star = solve_energy_minimizing_coeffs(M, c_vec, phiedge=phiedge, verbose=verbose)

    a_t, a_p = float(a_star[0]), float(a_star[1])

    # --- Step 5: build final coefficients and field evaluators for a* ---

    c_star = a_star[0] * c1 + a_star[1] * c2
    B_mv_star_fn = make_B_mv_evaluator(a_star, grad_t, grad_p, scinfo)
    phi_s_star_fn, B_star_fn, _, B_s_star_fn = make_total_field_evaluators_for_fixed_a_mfs(
        Y_src, c_star, scinfo, a_star, grad_t, grad_p
    )

    # Diagnostics: BC residual for a*
    if verbose:
        n_dot_Bs_star = A @ c_star
        g_star = a_star[0] * Dt + a_star[1] * Dp
        res_star = g_star + n_dot_Bs_star
        vec_stats("[BC] BC residual for a* (n·B_mv + n·B_s)", res_star)
        print(f"[BC] Mean BC residual = {float(jnp.mean(res_star)):.3e}")

    # --- Step 6: boundary diagnostics on Γ ---

    B_bdry = B_star_fn(P)
    B_mv_bdry = B_mv_star_fn(P)
    B_s_bdry = B_s_star_fn(P)

    Bmag_bdry = jnp.linalg.norm(B_bdry, axis=1)
    Bmag_mv_bdry = jnp.linalg.norm(B_mv_bdry, axis=1)
    Bmag_s_bdry = jnp.linalg.norm(B_s_bdry, axis=1)

    n_hat = N / jnp.maximum(jnp.linalg.norm(N, axis=1, keepdims=True), 1e-30)
    n_dot_B_bdry = jnp.sum(n_hat * B_bdry, axis=1)
    q_bdry = n_dot_B_bdry / jnp.maximum(Bmag_bdry, 1e-12)

    if verbose:
        vec_stats("[BΓ] |B| on Γ", Bmag_bdry)
        vec_stats("[BΓ] |B_mv| on Γ", Bmag_mv_bdry)
        vec_stats("[BΓ] |B_s| on Γ", Bmag_s_bdry)
        vec_stats("[BΓ] n·B on Γ", n_dot_B_bdry)
        vec_stats("[BΓ] n·B/|B| on Γ", q_bdry)
        print(f"[BΓ] Flux through Γ: Φ ≈ {float(jnp.dot(W, n_dot_B_bdry)):.6e}")
        print(f"     L2 norm of n·B/|B| on Γ: {jnp.linalg.norm(q_bdry):.6e}")
        print(f"     Linf norm of n·B/|B| on Γ: {jnp.max(jnp.abs(q_bdry)):.6e}")
        print(f"     L2 norm of n·B on Γ: {jnp.linalg.norm(n_dot_B_bdry):.6e}")
        print(f"     Linf norm of n·B on Γ: {jnp.max(jnp.abs(n_dot_B_bdry)):.6e}")

    # --- Step 6b: boundary coordinates relative to axis (ρ, φ) ---

    rho_bdry, phi_bdry = compute_axis_coordinates(P, a_hat, E, scinfo.center)
    if verbose: vec_stats("ρ (distance to axis) on Γ", rho_bdry)

    # --- Step 7: inner-shell diagnostics and Laplacian of φ_s ---

    X_shell, B_shell, n_dot_B_shell, Bmag_shell = diagnostics_on_inner_shell(
        P, N, W, B_star_fn, h_min=h_min, eps_factor=eps_factor, label="inner shell",
        verbose=verbose
    )

    h_fd = 0.25 * h_min
    lap_phi_s_inner = numerical_laplacian_phi_s(phi_s_star_fn, X_shell, h_fd=h_fd)
    if verbose: vec_stats("∇²φ_s on inner shell (samples)", lap_phi_s_inner)

    if mfs_out is None:
        mfs_out_path = out_dir / f"{base_name}_mfs_energymin_vmec_like_solution.npz"
    else:
        mfs_out_path = Path(mfs_out).resolve()

    try:
        np.savez(
            mfs_out_path,
            center=np.asarray(scinfo.center),
            scale=float(np.asarray(scinfo.scale)),
            P=np.asarray(P),
            N=np.asarray(N),
            W=np.asarray(W),
            Y_src=np.asarray(Y_src),
            c_star=np.asarray(c_star),
            a=np.asarray(a_star),
            a_hat=np.asarray(a_hat),
            kind=str(kind),
            phiedge=float(phiedge),
            base_name=base_name,
        )
        if verbose: print(f"[SAVE] Wrote MFS energy-minimizing solution checkpoint → {mfs_out_path}")
    except Exception as e:
        if verbose: print("[WARN] Could not save checkpoint:", e)

    return {"base_name": str(out_dir / base_name), "P": P, "N": N, "W": W,
        "h": h, "Y_src": Y_src, "kind": kind, "mfs_out_path": str(mfs_out_path), "center": scinfo.center,
        "scale": scinfo.scale, "c_star": c_star, "a_star": a_star, "a_hat": a_hat, "phiedge": phiedge,
        "scinfo": scinfo, "P_n": Pn, "Bmag_bdry": Bmag_bdry, "q_bdry": q_bdry,
        "phi_bdry": phi_bdry, "rho_bdry": rho_bdry, "Bmag_mv_bdry": Bmag_mv_bdry,
        "Bmag_s_bdry": Bmag_s_bdry, "n_dot_B_bdry": n_dot_B_bdry, "lap_phi_s_inner": lap_phi_s_inner,
    }

if __name__ == "__main__":
    # Default example (adjust file_name to your case)
    # file_name = "wout_precise_QH"
    # file_name = "wout_precise_QA"
    # file_name = "wout_SLAM_6_coils"
    # file_name = "wout_SLAM_4_coils"
    # file_name = "sflm_rm4"
    file_name = "knot_tube"
    
    candidate_xyz, candidate_normals = get_candidates(file_name, subdir="inputs", verbose=True)

    parser = argparse.ArgumentParser(
        description="MFS-based Neumann Laplace solver (vacuum B = ∇φ) "
                    "with VMEC-like energy minimization and fixed toroidal flux."
    )
    parser.add_argument("xyz", nargs="?", default=candidate_xyz,
                        help="CSV file with x,y,z columns (positional or --xyz)")
    parser.add_argument("normals", nargs="?", default=candidate_normals,
                        help="CSV file with nx,ny,nz columns (positional or --normals)")
    parser.add_argument("--k-nn", type=int, default=8,
                        help="k for kNN-based area weights (default: 32)")
    parser.add_argument("--source-offset-factor", type=float, default=0.9,
                        help="Factor for MFS source offset distance: d_i = factor * h_i")
    parser.add_argument("--reg", type=float, default=1e-8,
                        help="Small Tikhonov regularization in normal equations")
    parser.add_argument("--phiedge", type=float, default=1.0,
                        help="Toroidal flux (phiedge) for energy minimization")
    parser.add_argument("--out", dest="mfs_out", default=None,
                        help="Output .npz path for checkpoint (optional)")
    parser.add_argument("--xs-n-r", type=int, default=32,
                        help="Number of radial points in the poloidal cross-section")
    parser.add_argument("--xs-n-theta", type=int, default=64,
                        help="Number of angular points in the poloidal cross-section")
    parser.add_argument("--xs-margin-frac", type=float, default=0.15,
                        help="Fractional radial margin inside boundary for cross-section")
    parser.add_argument("--eps-factor", type=float, default=0.3)
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose debug printing")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots at the end")

    args = parser.parse_args()

    time0 = time.time()
    out = main(
        xyz_csv=args.xyz,
        normals_csv=args.normals,
        k_nn=args.k_nn,
        reg=args.reg,
        phiedge=args.phiedge,
        mfs_out=args.mfs_out,
        verbose=args.verbose,
        source_offset_factor=args.source_offset_factor,
        xs_n_r=args.xs_n_r,
        xs_n_theta=args.xs_n_theta,
        xs_margin_frac=args.xs_margin_frac,
        eps_factor=args.eps_factor
    )
    print("========================================================")
    print(f" Done. Total runtime: {time.time() - time0:.2f} seconds")
    print("========================================================")

    if args.plot:
        import matplotlib.pyplot as plt
        from main_helpers import (make_3d_boundary_plots,
            make_1d_residual_plots, make_boundary_decomposition_vs_phi,
            make_boundary_decomposition_vs_rho, make_boundary_geometry_plots)
        # --- Step 8: plots (all saved under ../outputs with base_name prefix) ---
        base_name = out["base_name"].split("/")[-1]
        boundary_png = f"{base_name}_boundary_diagnostics.png"
        residual_png = f"{base_name}_1d_diagnostics.png"
        decomp_phi_png = f"{base_name}_boundary_decomposition_vs_phi.png"
        decomp_rho_png = f"{base_name}_boundary_decomposition_vs_rho.png"
        geom_png = f"{base_name}_boundary_geometry_weights.png"
        make_3d_boundary_plots(out["P"], out["Bmag_bdry"], out["q_bdry"], boundary_png)
        make_1d_residual_plots(out["q_bdry"], out["lap_phi_s_inner"], residual_png)
        make_boundary_decomposition_vs_phi(
            out["phi_bdry"], out["Bmag_bdry"], out["Bmag_mv_bdry"], out["Bmag_s_bdry"],
            out["n_dot_B_bdry"], out["rho_bdry"], decomp_phi_png)
        make_boundary_decomposition_vs_rho(
            out["rho_bdry"], out["Bmag_bdry"], out["Bmag_mv_bdry"], out["Bmag_s_bdry"],
            out["n_dot_B_bdry"], decomp_rho_png)
        make_boundary_geometry_plots(out["W"], out["h"], out["rho_bdry"], geom_png)

        print(f"[PLOTS] Saved 3D boundary diagnostics to      {boundary_png}")
        print(f"[PLOTS] Saved 1D residual diagnostics to     {residual_png}")
        print(f"[PLOTS] Saved field decomposition vs φ to    {decomp_phi_png}")
        print(f"[PLOTS] Saved field decomposition vs ρ to    {decomp_rho_png}")
        print(f"[PLOTS] Saved geometry/weights diagnostics to {geom_png}")
        
        plt.show()