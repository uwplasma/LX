#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary-Integral Laplace Solver with Multi-valued Potential (B = ∇φ)
=====================================================================

This script computes a vacuum magnetic field B = ∇φ inside a closed 3D
surface Γ given by a point cloud (x,y,z) and outward normals n, enforcing

    n · B = 0  on Γ  (perfect conductor, field lines tangent to Γ),

for a *non-trivial* harmonic field that includes *multi-valued* pieces
(toroidal / poloidal) determined self-consistently from the surface.

Mathematical formulation
------------------------
We write the potential as

    φ(x) = φ_mv(x) + φ_s(x),

where
  - φ_mv is a (multi-valued) harmonic potential generating long-range,
    topologically nontrivial fields (toroidal/poloidal fluxes).
  - φ_s is a *single-valued* harmonic potential represented via a
    single-layer potential on Γ,

        φ_s(x) = - ∫_Γ σ(y) G(x,y) dS_y,
        G(x,y) = 1/(4π|x-y|).

The corresponding fields are

    B_mv = ∇φ_mv,    B_s = ∇φ_s,    B = B_mv + B_s.

On the boundary, the Neumann condition n·B = 0 becomes

    n · B_mv(x0) + ∂_n φ_s(x0) = 0,     x0 ∈ Γ.

For the single-layer, the classical interior Neumann jump relation yields

    ∂_n φ_s(x0) = -½ σ(x0) - ∫_Γ σ(y) ∂_{n_x}G(x0,y) dS_y.

Discretizing Γ with nodes (x_i, n_i, W_i), we obtain the dense linear system

    (½ I + K') σ = g,    g_i = n_i · B_mv(x_i),

where

    K'_{ij} ≈ ∂_{n_x} G(x_i,x_j) W_j.

The *multi-valued* field is constructed as

    B_mv(x) = a_t b_t(x) + a_p b_p(x),

where b_t, b_p are geometry-based axis-aware harmonic basis fields (toroidal
and poloidal), and a = (a_t,a_p)ᵀ are determined from the geometry via a
robust weighted SVD of their normal components n·b_t, n·b_p on Γ.

All core numerics (area weights, axis detection, K', SVD, solve, evaluation)
are implemented in JAX, so φ, B are differentiable w.r.t. the geometry.

Usage
-----
  python bie_neumann_vacuum_mv.py \
      --xyz inputs/wout_precise_QA.csv \
      --normals inputs/wout_precise_QA_normals.csv

Command-line arguments mirror your MFS scripts (xyz/normals, k-NN, etc.).

Outputs
-------
  - Rich console diagnostics:
      * PCA/axis info, geometry kind (torus/mirror)
      * area weights stats
      * multivalued coefficients a_t, a_p and their effect on n·B_mv
      * conditioning of (½I + K'), ||σ||₂
      * Neumann residual n·B on Γ, flux neutrality
  - Plots: surface colored by |B| and by |n·B|.
  - Checkpoint .npz with keys:
        center, scale, P, N, W, sigma, a, a_hat, kind

Dependencies
------------
  - jax, jaxlib  (64-bit enabled)
  - matplotlib   (for plots only)
  - numpy        (I/O & plotting only; NOT used in core solver)

Author: (your name / affiliation)
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np              # ONLY for I/O & plotting
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from jax import debug as jdbg

# -------------------------- Paths and utils ------------------------- #

script_dir = Path(__file__).resolve().parent

def get_candidates(file_name, subdir="inputs"):
    """
    Convenience helper: resolve ../inputs/<file_name>.csv and normals.
    """
    try:
        candidate_xyz = (script_dir / ".." / subdir / (file_name + ".csv")).resolve()
        candidate_normals = (script_dir / ".." / subdir / (file_name + "_normals.csv")).resolve()
        if candidate_xyz.exists():
            print(f"Resolved checkpoint path -> {candidate_xyz}")
            candidate_xyz = str(candidate_xyz)
        else:
            print(f"[WARN] Expected xyz at {candidate_xyz}; using literal path.")
        if candidate_normals.exists():
            print(f"Resolved checkpoint path -> {candidate_normals}")
            candidate_normals = str(candidate_normals)
        else:
            print(f"[WARN] Expected normals at {candidate_normals}; using literal path.")
    except Exception as e:
        print(f"[WARN] Failed to resolve ../{subdir} path: {e}")
    return candidate_xyz, candidate_normals

# ------------------------------ Utilities --------------------------- #

def pct(a, p):
    return float(np.percentile(np.asarray(a), p))

def vec_stats(label, v):
    v_np = np.asarray(v)
    print(f"[STATS] {label}: "
          f"L2={np.linalg.norm(v_np):.3e}, "
          f"Linf={np.max(np.abs(v_np)):.3e}, "
          f"mean={np.mean(v_np):.3e}")

def fix_matplotlib_3d(ax):
    """Equal aspect ratio in 3D plot."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1]-x_limits[0]); x_mid = 0.5*(x_limits[0]+x_limits[1])
    y_range = abs(y_limits[1]-y_limits[0]); y_mid = 0.5*(y_limits[0]+y_limits[1])
    z_range = abs(z_limits[1]-z_limits[0]); z_mid = 0.5*(z_limits[0]+z_limits[1])
    R = 0.5 * max(x_range, y_range, z_range)
    ax.set_xlim3d([x_mid-R, x_mid+R])
    ax.set_ylim3d([y_mid-R, y_mid+R])
    ax.set_zlim3d([z_mid-R, z_mid+R])

# ----------------------------------------------------------------------
# Geometry loading & normalization (I/O: numpy ok, solver: jax)
# ----------------------------------------------------------------------

def load_surface_xyz_normals(xyz_csv, normals_csv, verbose=True):
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
    """Center + scale as a PyTree, for easy passing into JAX-jitted code."""
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
      - BIE kernels in world coordinates
      - multi-valued bases built in normalized coordinates
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

def maybe_flip_normals(P, N):
    """
    Ensure outward normals: require <(P-c)·N> > 0.
    """
    c = jnp.mean(P, axis=0)
    s = jnp.sum((P - c) * N, axis=1)
    avg = float(jnp.mean(s))
    if avg < 0:
        print(f"[ORIENT] Normals inward on average (⟨(P-c)·N⟩≈{avg:.3e}) → flipping.")
        return -N
    print(f"[ORIENT] Normals seem outward (⟨(P-c)·N⟩≈{avg:.3e}).")
    return N

# ----------------------------------------------------------------------
# Pairwise geometry & area weights in pure JAX
# ----------------------------------------------------------------------

@jit
def pairwise_dist2(P):
    """Pairwise squared distances D_ij = |P_i - P_j|^2, P∈R^{N×3}."""
    Pi = P[:, None, :]   # (N,1,3)
    Pj = P[None, :, :]   # (1,N,3)
    diff = Pi - Pj       # (N,N,3)
    return jnp.sum(diff*diff, axis=-1)  # (N,N)

@jit
def estimate_area_weights_knn(P, k=32):
    """
    Return:
      W : (N,) crude patch areas
      h : (N,) local spacing ~ distance to k-th nearest neighbor

    We also use h to regularize near-singular kernels in build_Kprime.
    """
    N = P.shape[0]
    D2 = pairwise_dist2(P)              # (N,N)
    big = jnp.max(D2) + 1.0
    D2 = D2 + jnp.eye(N) * big          # kill self-distance

    D2_sorted = jnp.sort(D2, axis=1)
    h = jnp.sqrt(D2_sorted[:, k-1])     # (N,)
    W = jnp.pi * h*h

    jdbg.print("[QUAD] k={k}, h stats: min={mn:.3e}, med={md:.3e}, max={mx:.3e}",
               k=k, mn=jnp.min(h), md=jnp.median(h), mx=jnp.max(h))
    jdbg.print("[QUAD] area weights: sum W ≈ {sw:.3e}", sw=jnp.sum(W))
    return W, h

# ----------------------------------------------------------------------
# PCA-based axis detection (pure JAX)
# ----------------------------------------------------------------------

def detect_geometry_and_axis(Pn, verbose=True):
    """
    PCA on normalized coordinates Pn to choose an axis a_hat and classify
    the geometry as 'torus' or 'mirror', in the spirit of your MFS script.

      - 'torus': two large singular values, one much smaller (thin shell).
      - 'mirror': one very large singular value, two comparable smaller.

    Returns:
      kind: 'torus' or 'mirror'
      a_hat: unit axis vector in world coordinates of Pn (normalized space)
      E:     3x3 matrix of principal directions (columns)
      svals: singular values (descending)
    """
    X = Pn - jnp.mean(Pn, axis=0)
    U, S, Vt = jnp.linalg.svd(X, full_matrices=False)
    E = Vt.T  # principal directions as columns
    s = S     # already descending

    e1, e2, e3 = E[:,0], E[:,1], E[:,2]
    ratio_long = float(s[0] / jnp.maximum(s[1], 1e-12))
    ratio_thin = float(s[1] / jnp.maximum(s[2], 1e-12))

    if ratio_long > 2.0 and ratio_thin < 1.8:
        kind = "mirror"
        a_hat = e1
    elif ratio_thin > 2.0 and ratio_long < 1.8:
        kind = "torus"
        a_hat = e3
    else:
        kind = "torus"
        a_hat = e3

    a_hat = a_hat / jnp.maximum(jnp.linalg.norm(a_hat), 1e-30)

    if verbose:
        print(f"[PCA] singular values (desc) = {np.array(s)}")
        print(f"[PCA] ratio_long={ratio_long:.2f}, ratio_thin={ratio_thin:.2f}")
        print(f"[GEOM] kind={kind}, axis a_hat={np.array(a_hat)}")

    return kind, a_hat, E, s

# ----------------------------------------------------------------------
# Multivalued bases: toroidal & poloidal gradients in normalized coords
# ----------------------------------------------------------------------

@jit
def grad_azimuth_about_axis(Xn, a_hat):
    """
    ∇ϕ_a for azimuth around an arbitrary unit axis a_hat:

        r_perp = X - (X·a)a
        ∇ϕ_a = (a × r_perp) / |r_perp|^2

    Xn: (M,3) points in normalized coordinates
    """
    a = a_hat / jnp.maximum(jnp.linalg.norm(a_hat), 1e-30)
    r_par  = jnp.sum(Xn * a[None,:], axis=1, keepdims=True) * a[None,:]
    r_perp = Xn - r_par
    r2 = jnp.maximum(jnp.sum(r_perp*r_perp, axis=1, keepdims=True), 1e-30)
    cross = jnp.cross(a[None,:], r_perp)
    return cross / r2

def multivalued_bases_about_axis(Pn_ref, N_ref, a_hat, verbose=True):
    """
    Build axis-aware multivalued basis gradients in normalized coordinates:

      - grad_t(Xn): toroidal-like basis ∇ϕ_a around axis a_hat.
      - grad_p(Xn): poloidal-like basis built from local tangent geometry.

    Pn_ref: (N,3) normalized boundary nodes
    N_ref:  (N,3) normals at those nodes (world and normalized coincide in direction)
    """
    Pn_ref = jnp.asarray(Pn_ref)
    N_ref  = jnp.asarray(N_ref)
    a_hat  = jnp.asarray(a_hat)

    @jit
    def _nearest_normal_jax(Xn):
        """
        For each Xn[i], pick nearest boundary node in Pn_ref and return its normal.
        O(M*N) but fully vectorized; OK for N~O(10^3).
        """
        X2 = jnp.sum(Xn*Xn, axis=1, keepdims=True)              # (M,1)
        P2 = jnp.sum(Pn_ref*Pn_ref, axis=1, keepdims=True)      # (N,1)
        # dist^2 = |x|^2 + |p|^2 - 2 x·p
        dist2 = X2 + P2.T - 2.0 * (Xn @ Pn_ref.T)               # (M,N)
        idx = jnp.argmin(dist2, axis=1)                         # (M,)
        return N_ref[idx, :]                                    # (M,3)

    @jit
    def _unit(v, eps=1e-30):
        nrm = jnp.linalg.norm(v, axis=1, keepdims=True)
        return v / jnp.maximum(nrm, eps)

    @jit
    def _project_tangent(v, n):
        return v - jnp.sum(v*n, axis=1, keepdims=True)*n

    def grad_t(Xn):
        return grad_azimuth_about_axis(Xn, a_hat)

    def grad_p(Xn):
        """
        Poloidal-like basis:

          - use azimuthal direction ϕ̂_a = unit(a × r_perp)
          - project to tangent plane, renormalize to φ̃
          - define θ̂ = unit(n × φ̃)

        and use θ̂ as a proxy for ∇θ_p. This is divergence-free only up
        to geometric errors but works well as a second multivalued direction.
        """
        n = _nearest_normal_jax(Xn)      # (M,3)
        a = a_hat / jnp.maximum(jnp.linalg.norm(a_hat), 1e-30)
        r_par  = jnp.sum(Xn * a[None,:], axis=1, keepdims=True) * a[None,:]
        r_perp = Xn - r_par
        phi_hat = _unit(jnp.cross(a[None,:], r_perp))
        phi_tan = _unit(_project_tangent(phi_hat, n))
        theta_hat = _unit(jnp.cross(n, phi_tan))
        return theta_hat

    if verbose:
        print("[MV] Using axis-aware multivalued bases (toroidal+poloidal).")

    return grad_t, grad_p

# ----------------------------------------------------------------------
# Fit multivalued coefficients a = (a_t, a_p) from n·B_mv on Γ
# ----------------------------------------------------------------------

def fit_mv_coeffs_minimize_rhs(N, W, B_t_bdry, B_p_bdry, verbose=True):
    """
    Weighted SVD-based fit for the multivalued coefficients a = (a_t, a_p):

       B_mv = a_t B_t + a_p B_p
       g_mv = n · B_mv = D a,  D = [Dt, Dp]

    We *do not* solve a standard LS with g=0 (which would give a=0).
    Instead, we:

      1. Form D = [Dt, Dp] and weighted-center its columns.
      2. Compute leading right singular vector of W^{1/2}D_0.
      3. Scale so that ||D_0 a||_W hits a reasonable target relative to
         the column norms (~0.5 * median).

    This yields a robust, geometry-informed a that spans the dominant
    multivalued direction, without prescribing fluxes a priori.
    """
    Dt = jnp.sum(N * B_t_bdry, axis=1)   # (N,)
    Dp = jnp.sum(N * B_p_bdry, axis=1)   # (N,)
    D = jnp.stack([Dt, Dp], axis=1)      # (N,2)

    W = jnp.asarray(W)
    Wsum = jnp.sum(W) + 1e-30
    mu = (W @ D) / Wsum                  # (2,)
    D0 = D - mu[None, :]                 # weighted-centered

    Wsqrt = jnp.sqrt(W)
    Dw0 = D0 * Wsqrt[:, None]            # (N,2)

    # Leading singular vector of Dw0:
    U, S, Vt = jnp.linalg.svd(Dw0, full_matrices=False)
    a_dir = -Vt[0, :]                    # direction in (a_t,a_p) space

    # Scale so that ||D0 a||_W ≈ 0.5 * median column norm
    col_w2 = jnp.array([
        jnp.sqrt(jnp.dot(W, D0[:,0]**2)),
        jnp.sqrt(jnp.dot(W, D0[:,1]**2)),
    ])
    target = 0.5 * float(jnp.median(col_w2))
    g_dir = D0 @ a_dir
    denom = float(jnp.sqrt(jnp.dot(W, g_dir**2)) + 1e-30)
    scale = target / denom
    a = scale * a_dir

    if verbose:
        g_mv = D @ a
        vec_stats("[MV] n·B_mv (before σ)", g_mv)
        print(f"[MV] a_t={float(a[0]):.6g}, a_p={float(a[1]):.6g}")
    return a, Dt, Dp

# ----------------------------------------------------------------------
# Kernels: Green's function and its normal derivative (single-layer)
# ----------------------------------------------------------------------

@jit
def green_G(x, y):
    """G(x,y) = 1/(4π|x-y|). x,y ∈ R^3."""
    r = jnp.linalg.norm(x - y)
    r = jnp.maximum(r, 1e-12)
    return 1.0 / (4.0 * jnp.pi * r)

@jit
def grad_green_x(x, y):
    """∇_x G(x,y), x,y ∈ R^3."""
    r_vec = x - y
    r2 = jnp.sum(r_vec*r_vec)
    r2 = jnp.maximum(r2, 1e-18)
    r3 = r2 * jnp.sqrt(r2)
    return -r_vec / (4.0 * jnp.pi * r3)

@jit
def build_Kprime(P, N, W, h, clip_factor=0.2):
    """
    Build K' with near-singular regularization:

        r_eff^2 = max(r^2, (clip_factor * h_min)^2)

    where h_min is the global min spacing. This mimics the effect of
    integrating over a finite patch instead of a point singularity.
    """
    X = P
    Ni = N
    Wj = W

    Xi = X[:, None, :]  # (N,1,3)
    Xj = X[None, :, :]  # (1,N,3)
    diff = Xi - Xj      # (N,N,3)
    r2 = jnp.sum(diff*diff, axis=-1)  # (N,N)

    # Global minimum spacing as a simple scale:
    h_min = jnp.min(h)
    r2_clip = (clip_factor * h_min)**2

    mask = ~jnp.eye(X.shape[0], dtype=bool)

    # Clip all off-diagonal distances from below:
    r2_clipped = jnp.maximum(r2, r2_clip)
    r2_safe = jnp.where(mask, r2_clipped, 1.0)  # dummy diagonal

    r3 = r2_safe * jnp.sqrt(r2_safe)
    gradG = -diff / (4.0 * jnp.pi * r3[..., None])     # (N,N,3)

    n_dot_grad = jnp.sum(gradG * Ni[:, None, :], axis=-1)   # (N,N)
    n_dot_grad = jnp.where(mask, n_dot_grad, 0.0)

    Kprime = n_dot_grad * Wj[None, :]

    jdbg.print("[K'] clip_factor={cf:.3f}, h_min={hm:.3e}", cf=clip_factor, hm=h_min)
    jdbg.print("[K'] Kprime shape = ({n},{m})", n=Kprime.shape[0], m=Kprime.shape[1])
    jdbg.print("[K'] |K'| stats: min={mn:.3e}, med={md:.3e}, max={mx:.3e}",
               mn=jnp.min(jnp.abs(Kprime)),
               md=jnp.median(jnp.abs(Kprime)),
               mx=jnp.max(jnp.abs(Kprime)))
    return Kprime

# ----------------------------------------------------------------------
# Solve (½ I + K') σ = g in pure JAX
# ----------------------------------------------------------------------

@jit
def solve_density_sigma(P, N, W, h, g, reg=1e-6):
    """
    Solve (½ I + K') σ = g with clipped near-field and a slightly larger
    Tikhonov regularization reg * I.
    """
    Npts = P.shape[0]
    Kprime = build_Kprime(P, N, W, h)
    I = jnp.eye(Npts)
    A = 0.5 * I + Kprime + reg * I

    jdbg.print("[SOLVE] Assembling A, shape=({n},{m})", n=A.shape[0], m=A.shape[1])
    jdbg.print("[SOLVE] rhs g stats: L2={l2:.3e}, Linf={linf:.3e}",
               l2=jnp.linalg.norm(g), linf=jnp.max(jnp.abs(g)))

    sigma = jnp.linalg.solve(A, g)
    jdbg.print("[SOLVE] ||sigma||_2 = {ns:.3e}", ns=jnp.linalg.norm(sigma))
    return sigma

# ----------------------------------------------------------------------
# Single-layer potential evaluation (φ_s, B_s) in JAX
# ----------------------------------------------------------------------

def make_single_layer_evaluators(P_src, W_src, sigma, h_min, clip_factor=0.2):
    P_src = jnp.asarray(P_src)
    W_src = jnp.asarray(W_src)
    sigma = jnp.asarray(sigma)
    weight = sigma * W_src   # (N,)

    @jit
    def phi_s_at_point(x):
        diff = x[None, :] - P_src      # (N,3)
        r = jnp.linalg.norm(diff, axis=-1)
        r = jnp.maximum(r, 1e-12)
        Gvals = 1.0 / (4.0 * jnp.pi * r)
        return -jnp.sum(Gvals * weight)

    @jit
    def grad_phi_s_at_point(x):
        diff = x[None, :] - P_src      # (N,3)
        r2 = jnp.sum(diff*diff, axis=-1)

        # --- NEW: clip near field like in build_Kprime ---
        r2_clip = (clip_factor * h_min)**2
        r2 = jnp.maximum(r2, r2_clip)
        # --------------------------------------------------

        r3 = r2 * jnp.sqrt(r2)
        gradG = -diff / (4.0 * jnp.pi * r3[:, None])  # (N,3)
        return -jnp.sum(gradG * weight[:, None], axis=0)

    phi_s_batch  = jit(vmap(phi_s_at_point, in_axes=(0,)))
    grad_s_batch = jit(vmap(grad_phi_s_at_point, in_axes=(0,)))

    return phi_s_batch, grad_s_batch


def make_total_field_evaluators(P_src, W_src, sigma,
                                scinfo: ScaleInfo,
                                a, grad_t_fn, grad_p_fn,
                                h_min, clip_factor=0.2):
    """
    Return JIT-compiled evaluators:

        phi_fn(X): "single-valued" part φ_s(X) (we ignore multi-valued jumps)
        B_fn(X):   full B = B_mv + B_s

    Multi-valued gradient in world coordinates:

        Xn = (X - center) * scale
        B_mv = scale * (a_t grad_t(Xn) + a_p grad_p(Xn))
    """
    phi_s_fn, grad_s_fn = make_single_layer_evaluators(P_src, W_src, sigma,
                                                       h_min, clip_factor)
    a = jnp.asarray(a)
    center = scinfo.center
    scale  = scinfo.scale

    @jit
    def B_mv_fn(X):
        Xn = (X - center[None,:]) * scale
        Gt = grad_t_fn(Xn)
        Gp = grad_p_fn(Xn)
        return scale * (a[0]*Gt + a[1]*Gp)

    @jit
    def B_fn(X):
        return B_mv_fn(X) + grad_s_fn(X)

    @jit
    def phi_fn(X):
        # Multi-valued part omitted; φ_s is single-valued and sufficient
        return phi_s_fn(X)

    return phi_fn, B_fn, B_mv_fn

# ----------------------------------------------------------------------
# Diagnostics & plotting
# ----------------------------------------------------------------------

def diagnostics_on_boundary(P, N, W, B_fn, h_min=None):
    X = jnp.asarray(P)
    Nw = jnp.asarray(N)
    Ww = jnp.asarray(W)

    if h_min is None:
        h_min = jnp.min(jnp.sqrt(jnp.sum((X[1:] - X[:-1])**2, axis=1)))

    eps = 0.3 * h_min
    X_eval = X - eps * Nw

    B_on_Gamma = B_fn(X_eval)          # (N,3) at slightly interior points
    n_dot_B = jnp.sum(Nw * B_on_Gamma, axis=1)
    Bmag = jnp.linalg.norm(B_on_Gamma, axis=1)

    vec_stats("B|Γ magnitude", Bmag)
    vec_stats("n·B|Γ", n_dot_B)

    flux = float(jnp.dot(Ww, n_dot_B))
    area = float(jnp.sum(Ww))
    print(f"[CHK] Flux through Γ from B_total: Φ ≈ {flux:.6e}, avg n·B ≈ {flux/area:.3e}")

    return np.asarray(B_on_Gamma), np.asarray(n_dot_B), np.asarray(Bmag)

def make_plots(P, N, Bmag, n_dot_B):
    """
    Simple diagnostic plots:
      - Surface colored by |B|
      - Surface colored by |n·B|
    """
    P_np = np.asarray(P)
    Bmag_np = np.asarray(Bmag)
    ndot_np = np.asarray(np.abs(n_dot_B))

    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    vmin = pct(Bmag_np, 1); vmax = pct(Bmag_np, 99)
    sc1 = ax1.scatter(P_np[:,0], P_np[:,1], P_np[:,2],
                      c=Bmag_np, s=6, cmap='viridis',
                      vmin=vmin, vmax=vmax)
    fig.colorbar(sc1, ax=ax1, shrink=0.7, label="|B| on Γ")
    ax1.set_title("Boundary colored by |B|")
    fix_matplotlib_3d(ax1)

    ax2 = fig.add_subplot(1,2,2, projection='3d')
    vmin2 = pct(ndot_np, 1); vmax2 = pct(ndot_np, 99)
    sc2 = ax2.scatter(P_np[:,0], P_np[:,1], P_np[:,2],
                      c=ndot_np, s=6, cmap='magma',
                      vmin=vmin2, vmax=vmax2)
    fig.colorbar(sc2, ax=ax2, shrink=0.7, label="|n·B| on Γ")
    ax2.set_title("Boundary colored by |n·B| (Neumann residual)")
    fix_matplotlib_3d(ax2)

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main(xyz_csv,
         normals_csv,
         k_nn=32,
         reg=1e-10,
         mfs_out=None,
         verbose=True):

    print("========================================================")
    print(" Boundary-Integral Neumann Laplace Solver (B = ∇φ)")
    print(" with self-consistent multi-valued (toroidal/poloidal) pieces")
    print("========================================================")
    print(f"[PARAM] k_nn = {k_nn}")
    print(f"[PARAM] reg  = {reg:.1e} (small Tikhonov)")

    # Load surface geometry
    P, N = load_surface_xyz_normals(xyz_csv, normals_csv, verbose=verbose)
    N = maybe_flip_normals(P, N)

    # Normalize geometry (for multi-valued bases)
    Pn, scinfo = normalize_geometry(P, verbose=verbose)

    # Area weights & spacings from normalized coords, rescaled to world units
    Wn, h_n = estimate_area_weights_knn(Pn, k=k_nn)  # in normalized coordinates
    scale = float(scinfo.scale)

    # Areas scale like length^2, distances like length
    W = Wn / (scale**2)      # world-area weights
    h = h_n / scale          # world spacings
    h_min = float(jnp.min(h))

    vec_stats("[QUAD] W (world)", W)


    # Geometry classification and axis detection
    kind, a_hat, E, svals = detect_geometry_and_axis(Pn, verbose=verbose)

    # Multivalued bases in normalized coords
    grad_t_fn, grad_p_fn = multivalued_bases_about_axis(Pn, N, a_hat, verbose=verbose)

    # Gradients at boundary (normalized); convert to world via scale
    Gt_bdry = grad_t_fn(Pn)
    Gp_bdry = grad_p_fn(Pn)
    B_t_bdry = scinfo.scale * Gt_bdry
    B_p_bdry = scinfo.scale * Gp_bdry

    # Fit multivalued coefficients a = (a_t, a_p)
    a, Dt, Dp = fit_mv_coeffs_minimize_rhs(N, W, B_t_bdry, B_p_bdry, verbose=verbose)

    # Boundary data g = n·B_mv
    g = Dt * a[0] + Dp * a[1]
    vec_stats("[BC] g (raw) = n·B_mv (rhs for (½I+K')σ=g)", g)

    # Enforce Neumann compatibility: ⟨g⟩_W = 0
    g_mean = jnp.dot(W, g) / jnp.sum(W)
    g = g - g_mean
    jdbg.print("[BC] projected g to zero weighted mean, g_mean={gm:.3e}", gm=g_mean)
    vec_stats("[BC] g (after mean removal)", g)

    # Solve BIE for σ
    sigma = solve_density_sigma(P, N, W, h, g, reg=reg)

    # Build evaluators φ, B
    phi_fn, B_fn, B_mv_fn = make_total_field_evaluators(
        P_src=P, W_src=W, sigma=sigma,
        scinfo=scinfo, a=a,
        grad_t_fn=grad_t_fn, grad_p_fn=grad_p_fn,
        h_min=h_min, clip_factor=0.2,
    )
    
    # --- Extra diagnostics: separate B_mv and B_s on a slightly interior shell ---
    X = jnp.asarray(P)
    Nw = jnp.asarray(N)
    eps = 0.3 * h_min
    X_int = X - eps * Nw

    B_mv_int = B_mv_fn(X_int)
    B_tot_int = B_fn(X_int)
    B_s_int = B_tot_int - B_mv_int

    n_dot_B_mv = jnp.sum(Nw * B_mv_int, axis=1)
    n_dot_B_s  = jnp.sum(Nw * B_s_int,  axis=1)

    vec_stats("[DIAG] n·B_mv (interior shell)", n_dot_B_mv)
    vec_stats("[DIAG] n·B_s  (interior shell)", n_dot_B_s)

    # Diagnostics on the boundary
    B_on_Gamma, n_dot_B, Bmag = diagnostics_on_boundary(P, N, W, B_fn, h_min=h_min)

    # Plots
    make_plots(P, N, Bmag, n_dot_B)

    # Save checkpoint for downstream use
    if mfs_out is None:
        mfs_out = xyz_csv.replace(".csv", "_bie_mv_solution.npz")
        mfs_out = str((Path(xyz_csv).parent / mfs_out).resolve())
    try:
        np.savez(
            mfs_out,
            center=np.asarray(scinfo.center),
            scale=float(np.asarray(scinfo.scale)),
            P=np.asarray(P),
            N=np.asarray(N),
            W=np.asarray(W),
            sigma=np.asarray(sigma),
            a=np.asarray(a),
            a_hat=np.asarray(a_hat),
            kind=str(kind),
        )
        print(f"[SAVE] Wrote BIE+MV solution checkpoint → {mfs_out}")
    except Exception as e:
        print("[WARN] Could not save checkpoint:", e)

    print("========================================================")
    print(" Done.")
    print("========================================================")

if __name__ == "__main__":
    file_name = "wout_precise_QA"
    # file_name = "wout_precise_QH"
    # file_name = "wout_SLAM_6_coils"
    # file_name = "wout_SLAM_4_coils"
    # file_name = "sflm_rm4"
    # file_name = "knot_tube"
    candidate_xyz, candidate_normals = get_candidates(file_name, subdir="inputs")

    parser = argparse.ArgumentParser(
        description="Boundary-integral Neumann Laplace solver (vacuum B = ∇φ) "
                    "with self-consistent multi-valued pieces."
    )
    parser.add_argument("xyz", nargs="?", default=candidate_xyz,
                        help="CSV file with x,y,z columns (positional or --xyz)")
    parser.add_argument("normals", nargs="?", default=candidate_normals,
                        help="CSV file with nx,ny,nz columns (positional or --normals)")
    parser.add_argument("--k-nn", type=int, default=32,
                        help="k for kNN-based area weights (default: 32)")
    parser.add_argument("--reg", type=float, default=1e-6,
                        help="Small Tikhonov regularization for Neumann system")
    parser.add_argument("--out", dest="mfs_out", default=None,
                        help="Output .npz path for checkpoint (optional)")

    args = parser.parse_args()

    main(
        xyz_csv=args.xyz,
        normals_csv=args.normals,
        k_nn=args.k_nn,
        reg=args.reg,
        mfs_out=args.mfs_out,
        verbose=True,
    )
