#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy-minimizing vacuum field via Method of Fundamental Solutions (MFS)
=======================================================================

We compute a vacuum magnetic field B = ∇φ inside a closed 3D surface Γ
given by a point cloud (x,y,z) and outward normals n, enforcing

    n · B = 0  on Γ   (perfect conductor, field lines tangent to Γ),

and selecting among all such harmonic fields the one that

  (i)  has a prescribed *toroidal flux* phiedge, defined as
       Φ_tor = ∫_S B·a_hat dS through a poloidal cross-section S
       orthogonal to the PCA-based axis a_hat; and

  (ii) minimizes the *magnetic energy*

          E = ½ ∫_Ω |B|² dV,

       approximated by revolving that cross-section around the
       straight axis line:

          dV ≈ (2π ρ) dS,  where ρ is the distance to the axis.

Representation: MFS + multivalued bases
---------------------------------------

We represent φ as

    φ(x) = φ_mv(x) + φ_s(x),

where

  - φ_mv is a multi-valued harmonic potential whose gradient
    B_mv = ∇φ_mv spans the topological (toroidal / poloidal) space.
    We use two axis-aware multivalued basis fields:

        B_mv(x; a) = a_t b_t(x) + a_p b_p(x),  a = (a_t, a_p)^T.

  - φ_s is a *single-valued* harmonic potential represented by a sum
    of fundamental solutions (Method of Fundamental Solutions, MFS):

        φ_s(x) = ∑_j c_j G(x, y_j),
        G(x,y) = 1/(4π|x-y|),

    where the source points {y_j} lie on a *fictitious outer surface*
    Γ^d constructed by offsetting Γ along its outward normals:

        y_j = x_j + d_j n_j,   d_j ≈ source_offset_factor * h_j,

    with h_j a local spacing estimate.

The field is

    B(x) = ∇φ(x) = B_mv(x) + B_s(x),    B_s = ∇φ_s.

No singular integrals appear in the MFS formulation because all sources
are strictly outside the physical domain.

Boundary condition and basis solves
-----------------------------------

On the boundary Γ:

    n · B = 0  ⇒  n·B_mv + n·B_s = 0.

Using the MFS representation of B_s, this becomes

    ∑_j c_j [n_i · ∇_x G(x_i, y_j)] = - g_i,
    g_i = n_i · B_mv(x_i),

for boundary points x_i ∈ Γ with unit outward normals n_i.

We define the dense MFS Neumann matrix

    A_ij = n_i · ∇_x G(x_i, y_j),

and solve

    A c = -g.

To parameterize the topological space of vacuum fields, we choose two
multivalued basis fields:

  - grad_t(x) : constant harmonic field along the PCA axis (toroidal).
  - grad_p(x) : azimuthal harmonic field looping poloidally around the axis.

For each basis choice a^(1)=(1,0), a^(2)=(0,1), we solve:

  n·B_mv^{(k)}(x_i) = g^{(k)}_i,
  A c^{(k)} = -g^{(k)},   k = 1,2,

to obtain two full vacuum fields:

  B^{(k)}(x) = B_mv^{(k)}(x) + B_s^{(k)}(x),   k=1,2.

Any linear combination

    B(a) = a_1 B^{(1)} + a_2 B^{(2)}

is a divergence-free, harmonic field with n·B = 0 on Γ.

VMEC-like energy and toroidal flux
----------------------------------

We then construct a *poloidal cross-section* S:

  - Axis direction a_hat from a PCA of normalized coordinates (kind="torus").
  - Center at the geometric center of the point cloud.
  - Plane orthogonal to a_hat.
  - Polar grid (r,θ) inside the torus footprint.

On this cross-section, we define:

  - Area quadrature weights: dS = r dr dθ.
  - Toroidal flux:

        Φ_tor(a) = ∫_S B(a)·a_hat dS
                  ≈ ∑_q w_q (B(a,x_q)·a_hat).

  - Energy matrix M via an approximate volume integral:

        E(a) = ½ ∫_Ω |B(a)|² dV
              ≈ ½ ∑_q (2π ρ_q w_q) |B(a,x_q)|²
              = ½ aᵀ M a,

    where ρ_q is the distance from x_q to the axis line, and
    M_{ij} = ∑_q 2π ρ_q w_q [B^{(i)}(x_q)·B^{(j)}(x_q)].

Given a user-specified phiedge = Φ_tor target, we solve the quadratic
program

    minimize  E(a) = ½ aᵀ M a
    subject to   cᵀ a = phiedge,

with c_i = Φ_tor(B^{(i)}). The solution is analytic:

    a* = (phiedge / (cᵀ M⁻¹ c)) M⁻¹ c.

Diagnostics and plotting
------------------------

The script prints:

  - Geometry and PCA diagnostics
  - kNN-based quadrature / spacing stats
  - MFS source offset diagnostics
  - Condition numbers and norms for MFS systems
  - Boundary residuals n·B and n·B/|B| on Γ
  - Flux checks on boundary and on an inner shell
  - Laplacian residual ∇²φ_s on an inner shell (FD approximation)
  - Decomposition of B into multivalued and MFS parts on Γ
  - Correlations of |B| with distance to axis ρ and toroidal angle φ

and produces publication-ready plots (all saved under ../outputs):

  1. 3D: boundary colored by |B| and n·B/|B|.
  2. 1D: n·B/|B| on Γ; ∇²φ_s on inner shell.
  3. 1D: |B|, |B_mv|, |B_s| vs toroidal angle φ.
  4. 1D: |B|, |B_mv|, |B_s| vs distance to axis ρ.
  5. 1D: geometric/quad diagnostics (ρ, W, h) vs boundary index.

JAX and differentiability
-------------------------

All core pieces (MFS assembly, solves, field evaluation, energy and
flux integrals) are written in JAX using jnp, jit, and vmap. This makes
the vacuum field and energy functional differentiable with respect to
geometry (point cloud P, normals N, etc.) for use in shape optimization
and design.

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
            print(f"Resolved xyz path -> {candidate_xyz}")
            candidate_xyz = str(candidate_xyz)
        else:
            print(f"[WARN] Expected xyz at {candidate_xyz}; using literal path.")
        if candidate_normals.exists():
            print(f"Resolved normals path -> {candidate_normals}")
            candidate_normals = str(candidate_normals)
        else:
            print(f"[WARN] Expected normals at {candidate_normals}; using literal path.")
    except Exception as e:
        print(f"[WARN] Failed to resolve ../{subdir} path: {e}")
    return candidate_xyz, candidate_normals


def pct(a, p):
    """Percentile helper using NumPy (for diagnostics only)."""
    return float(np.percentile(np.asarray(a), p))


def vec_stats(label, v):
    """
    Simple L2/Linf/mean stats for 1D arrays.

    Parameters
    ----------
    label : str
        Label printed before the statistics.
    v : array_like
        1D array (NumPy or JAX) whose norms are reported.
    """
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


def maybe_flip_normals(P, N):
    """
    Ensure outward normals: require <(P-c)·N> > 0 on average.

    If average (P-c)·N is negative, flip all normals.
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
# Pairwise geometry & area weights (for diagnostics, flux integrals)
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

    jdbg.print("[QUAD] k={k}, h stats: min={mn:.3e}, med={md:.3e}, max={mx:.3e}",
               k=k_int, mn=jnp.min(h), md=jnp.median(h), mx=jnp.max(h))
    jdbg.print("[QUAD] area weights: sum W ≈ {sw:.3e}", sw=jnp.sum(W))
    return W, h


# ----------------------------------------------------------------------
# PCA-based axis detection (pure JAX)
# ----------------------------------------------------------------------

def detect_geometry_and_axis(Pn, verbose=True):
    """
    PCA on normalized coordinates Pn to choose an axis a_hat and classify
    the geometry as 'torus' or 'mirror'.

    For now we primarily target toroidal geometries; 'mirror' is noted but
    we always take the thin direction as the axis for grad_t / grad_p.
    """
    X = Pn - jnp.mean(Pn, axis=0)
    U, S, Vt = jnp.linalg.svd(X, full_matrices=False)
    E = Vt.T  # principal directions as columns
    s = S     # descending

    e1, e2, e3 = E[:, 0], E[:, 1], E[:, 2]
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

        r_par  = (X·a)a
        r_perp = X - r_par
        ∇ϕ_a   = (a × r_perp) / |r_perp|^2

    Xn: (M,3) points in normalized coordinates.

    This is the gradient of a multi-valued harmonic scalar away from
    the axis line, giving poloidal "looping" around the axis.
    """
    a = a_hat / jnp.maximum(jnp.linalg.norm(a_hat), 1e-30)
    r_par  = jnp.sum(Xn * a[None, :], axis=1, keepdims=True) * a[None, :]
    r_perp = Xn - r_par
    r2 = jnp.maximum(jnp.sum(r_perp * r_perp, axis=1, keepdims=True), 1e-30)
    cross = jnp.cross(a[None, :], r_perp)
    return cross / r2


def multivalued_bases_about_axis(Pn_ref, N_ref, a_hat, verbose=True):
    """
    Build *harmonic* basis gradients in normalized coordinates:

      - grad_t(Xn): toroidal basis, ∇φ_tor with φ_tor = a_hat·x  (constant field).
      - grad_p(Xn): poloidal basis, ∇φ_pol = ∇ϕ_a (azimuth around axis a_hat),
                    harmonic and multi-valued, singular only on the axis line.
    """
    del N_ref  # not used yet, but kept in signature for future extensions
    Pn_ref = jnp.asarray(Pn_ref)
    _ = Pn_ref
    a_hat  = jnp.asarray(a_hat)

    a_unit = a_hat / jnp.maximum(jnp.linalg.norm(a_hat), 1e-30)

    def grad_t(Xn):
        """
        Toroidal-like basis: constant harmonic field along the PCA axis:

          φ_t(X) = a_hat·X  ⇒  ∇φ_t = a_hat (constant).
        """
        Xn = jnp.asarray(Xn)
        return jnp.broadcast_to(a_unit, Xn.shape)

    def grad_p(Xn):
        """
        Poloidal-like basis: azimuth around the axis a_hat.
        """
        return grad_azimuth_about_axis(Xn, a_hat)

    if verbose:
        print("[MV] Using *harmonic* multivalued bases:")
        print("     - grad_t: constant along axis (toroidal flux carrier).")
        print("     - grad_p: azimuth around axis (poloidal loops).")

    return grad_t, grad_p


# ----------------------------------------------------------------------
# MFS: fictitious outer source surface and Neumann matrix
# ----------------------------------------------------------------------

@jit
def build_mfs_sources(P, N, h_world, source_offset_factor=2.0):
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

    jdbg.print("[MFS] source_offset_factor={f:.3f}", f=source_offset_factor)
    jdbg.print("[MFS] d stats: min={mn:.3e}, med={md:.3e}, max={mx:.3e}",
               mn=jnp.min(d), md=jnp.median(d), mx=jnp.max(d))
    return Y


@jit
def build_mfs_neumann_matrix(P, N, Y):
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

    jdbg.print("[MFS] A shape = ({n},{m})", n=n_dot_grad.shape[0], m=n_dot_grad.shape[1])
    jdbg.print("[MFS] |A| stats: min={mn:.3e}, med={md:.3e}, max={mx:.3e}",
               mn=jnp.min(jnp.abs(n_dot_grad)),
               md=jnp.median(jnp.abs(n_dot_grad)),
               mx=jnp.max(jnp.abs(n_dot_grad)))
    return n_dot_grad


@jit
def solve_mfs_neumann(P, N, Y, g, reg=0.0):
    """
    Solve the MFS Neumann system for a *given* multivalued boundary data:

        n · B_s(x_i) = -g_i,

    with

        B_s = ∇φ_s,
        φ_s(x) = ∑_j c_j G(x,y_j),

    which leads to

        A c = -g,    A_ij = n_i · ∇_x G(x_i, y_j).

    Inputs
    ------
    P, N : (N,3)
        Boundary points and normals.
    Y    : (M,3)
        MFS source points.
    g    : (N,)
        Boundary data n·B_mv.
    reg  : float
        Small Tikhonov regularization parameter (added to diag).

    Returns
    -------
    c : (M,)
        MFS coefficients.
    """
    A = build_mfs_neumann_matrix(P, N, Y)   # (N,M)
    Npts, Msrc = A.shape

    # Overdetermined or square systems: we use normal equations with reg
    # to keep things robust. For N=M it's just a stabilized square solve.
    At = A.T
    AtA = At @ A
    rhs = -At @ g

    if reg != 0.0:
        AtA = AtA + reg * jnp.eye(Msrc, dtype=AtA.dtype)

    jdbg.print("[MFS] Solving normal equations (AtA size={n}x{n})", n=Msrc)
    c = jnp.linalg.solve(AtA, rhs)

    jdbg.print("[MFS] ||c||_2 = {nc:.3e}", nc=jnp.linalg.norm(c))
    return c


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
    Y_src, reg=1e-8
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

    Dt = jnp.sum(N * B_t_bdry, axis=1)
    Dp = jnp.sum(N * B_p_bdry, axis=1)

    vec_stats("[BASIS] g_t = n·B_t", Dt)
    vec_stats("[BASIS] g_p = n·B_p", Dp)

    # Assemble MFS matrix once
    A = build_mfs_neumann_matrix(P, N, Y_src)
    At = A.T
    AtA = At @ A

    Npts, Msrc = A.shape
    jdbg.print("[BASIS] MFS AtA shape = ({n},{m})", n=Msrc, m=Msrc)

    # Solve for basis 1 and 2 with shared AtA
    rhs1 = -At @ Dt
    rhs2 = -At @ Dp
    if reg != 0.0:
        AtA_reg = AtA + reg * jnp.eye(Msrc, dtype=AtA.dtype)
    else:
        AtA_reg = AtA

    c1 = jnp.linalg.solve(AtA_reg, rhs1)
    c2 = jnp.linalg.solve(AtA_reg, rhs2)

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


# ----------------------------------------------------------------------
# VMEC-like poloidal cross-section quadrature
# ----------------------------------------------------------------------

def build_poloidal_cross_section_quadrature(P, a_hat, E, scinfo,
                                            n_r=32, n_theta=64,
                                            margin_frac=0.15):
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
    margin_frac=0.15
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
        n_r=n_r, n_theta=n_theta, margin_frac=margin_frac
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

    vec_stats("[ENERGY] M matrix entries", np.array([M11, M22, M12]))
    vec_stats("[FLUX] c vector", np.array([c1, c2]))

    return M, c, X_cs, B1_cs, B2_cs


def solve_energy_minimizing_coeffs(M, c, phiedge):
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
    print(f"[ENERGY] denom cᵀ M⁻¹ c = {denom:.6e}")
    print(f"[ENERGY] a* = (a_t, a_p) = ({float(a_star[0]):.6e}, {float(a_star[1]):.6e})")
    return a_star


# ----------------------------------------------------------------------
# Diagnostics: inner-shell field, Laplacian of φ_s, boundary B
# ----------------------------------------------------------------------

def diagnostics_on_inner_shell(P, N, W, B_fn,
                               h_min=None, eps_factor=0.3, label="inner shell"):
    """
    Sample B slightly inside the boundary along normals, and report
    magnitude and normal component, plus flux through the shell.
    """
    X = jnp.asarray(P)
    Nw = jnp.asarray(N)
    Ww = jnp.asarray(W)

    if h_min is None:
        # crude spacing estimate if not provided
        h_min = jnp.min(jnp.sqrt(jnp.sum((X[1:] - X[:-1])**2, axis=1)))

    eps = eps_factor * h_min
    X_eval = X - eps * Nw

    B_on_shell = B_fn(X_eval)
    n_dot_B = jnp.sum(Nw * B_on_shell, axis=1)
    Bmag = jnp.linalg.norm(B_on_shell, axis=1)

    vec_stats(f"B|{label} magnitude", Bmag)
    vec_stats(f"n·B|{label}", n_dot_B)

    flux = float(jnp.dot(Ww, n_dot_B))
    area = float(jnp.sum(Ww))
    print(f"[CHK] Flux through {label}: Φ ≈ {flux:.6e}, avg n·B ≈ {flux/area:.3e}")

    return np.asarray(X_eval), np.asarray(B_on_shell), np.asarray(n_dot_B), np.asarray(Bmag)


def numerical_laplacian_phi_s(phi_s_fn, X_inner, h_fd):
    """
    Approximate Laplacian ∇²φ_s at inner-shell points via a 7-point stencil.
    """
    X = jnp.asarray(X_inner)
    h = h_fd

    ex = jnp.array([1.0, 0.0, 0.0])
    ey = jnp.array([0.0, 1.0, 0.0])
    ez = jnp.array([0.0, 0.0, 1.0])

    def phi_at_offset(X, direction):
        return phi_s_fn(X + h * direction[None, :]), phi_s_fn(X - h * direction[None, :])

    phi0 = phi_s_fn(X)

    phi_px_x, phi_mx_x = phi_at_offset(X, ex)
    phi_px_y, phi_mx_y = phi_at_offset(X, ey)
    phi_px_z, phi_mx_z = phi_at_offset(X, ez)

    lap = (phi_px_x + phi_mx_x - 2.0 * phi0
           + phi_px_y + phi_mx_y - 2.0 * phi0
           + phi_px_z + phi_mx_z - 2.0 * phi0) / (h * h)

    return np.asarray(lap)


# ----------------------------------------------------------------------
# Boundary coordinates relative to axis (ρ, φ)
# ----------------------------------------------------------------------

def compute_axis_coordinates(P, scinfo: ScaleInfo, a_hat, E):
    """
    Compute distance to axis ρ and toroidal angle φ for each boundary point.

    Parameters
    ----------
    P : (N,3) array-like
        Boundary points in world coordinates.
    scinfo : ScaleInfo
        Contains center used for normalization.
    a_hat : (3,) array-like
        Unit vector along the PCA-based axis.
    E : (3,3) array-like
        PCA eigenvectors as columns. e1,e2 span the poloidal plane.

    Returns
    -------
    rho : (N,) ndarray
        Distances from the axis line.
    phi : (N,) ndarray
        Toroidal angles in [0, 2π), defined using projection onto (e1,e2).
    """
    Pj = jnp.asarray(P)
    center = scinfo.center
    a = a_hat / jnp.maximum(jnp.linalg.norm(a_hat), 1e-30)

    e1 = E[:, 0]
    e2 = E[:, 1]

    r_vec = Pj - center[None, :]
    r_par = jnp.sum(r_vec * a[None, :], axis=1, keepdims=True) * a[None, :]
    r_perp = r_vec - r_par

    rho = jnp.linalg.norm(r_perp, axis=1)

    x1 = jnp.sum(r_perp * e1[None, :], axis=1)
    x2 = jnp.sum(r_perp * e2[None, :], axis=1)
    phi = jnp.arctan2(x2, x1)
    # Map to [0, 2π)
    two_pi = 2.0 * jnp.pi
    phi = jnp.where(phi < 0.0, phi + two_pi, phi)

    return np.asarray(rho), np.asarray(phi)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def make_3d_boundary_plots(P, Bmag, n_dot_B_norm, outfile):
    """
    3D diagnostic plots:
      - Surface colored by |B|
      - Surface colored by n·B/|B|
    """
    P_np = np.asarray(P)
    Bmag_np = np.asarray(Bmag)
    ndot_np = np.asarray(n_dot_B_norm)

    fig = plt.figure(figsize=(14, 6))

    # |B| on Γ
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    vmin = pct(Bmag_np, 1)
    vmax = pct(Bmag_np, 99)
    sc1 = ax1.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                      c=Bmag_np, s=6, cmap='viridis',
                      vmin=vmin, vmax=vmax)
    cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.7)
    cb1.set_label(r"$|B|$ on $\Gamma$")
    ax1.set_title("Boundary colored by $|B|$")
    fix_matplotlib_3d(ax1)

    # n·B/|B| on Γ
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    maxabs = np.max(np.abs(ndot_np))
    vmin2, vmax2 = -maxabs, maxabs
    sc2 = ax2.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                      c=ndot_np, s=6, cmap='magma',
                      vmin=vmin2, vmax=vmax2)
    cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.7)
    cb2.set_label(r"$\mathbf{n}\!\cdot\!\mathbf{B}/|B|$ on $\Gamma$")
    ax2.set_title(r"Boundary colored by $\mathbf{n}\!\cdot\!\mathbf{B}/|B|$")
    fix_matplotlib_3d(ax2)

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_1d_residual_plots(q_bdry, lap_inner, outfile):
    """
    1D residual plots:
      - q_bdry = n·B/|B| on Γ vs index
      - ∇²φ_s on inner shell vs index
    """
    q_np = np.asarray(q_bdry)
    lap_np = np.asarray(lap_inner)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(q_np, ".", ms=2)
    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[0].set_xlabel("Boundary point index")
    axes[0].set_ylabel(r"$\mathbf{n}\!\cdot\!\mathbf{B}/|B|$")
    axes[0].set_title(r"Boundary residual $\mathbf{n}\!\cdot\!\mathbf{B}/|B|$")

    axes[1].plot(lap_np, ".", ms=2)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("Inner-shell point index")
    axes[1].set_ylabel(r"$\nabla^2 \phi_s$ (FD)")
    axes[1].set_title(r"Laplacian residual $\nabla^2 \phi_s$ on inner shell")

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_boundary_decomposition_vs_phi(phi, Bmag_tot, Bmag_mv, Bmag_s,
                                       n_dot_B, rho, outfile):
    """
    Plot |B|, |B_mv|, |B_s| and n·B vs toroidal angle φ.
    """
    phi_np = np.asarray(phi)
    Btot = np.asarray(Bmag_tot)
    Bmv = np.asarray(Bmag_mv)
    Bs  = np.asarray(Bmag_s)
    ndot = np.asarray(n_dot_B)
    rho_np = np.asarray(rho)

    # Sort by φ for clearer trends
    idx = np.argsort(phi_np)
    phi_s = phi_np[idx]
    Btot_s = Btot[idx]
    Bmv_s = Bmv[idx]
    Bs_s = Bs[idx]
    ndot_s = ndot[idx]
    rho_s = rho_np[idx]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(phi_s, Btot_s, ".", ms=2, label=r"$|B|$")
    axes[0].plot(phi_s, Bmv_s, ".", ms=2, label=r"$|B_{\mathrm{mv}}|$")
    axes[0].plot(phi_s, Bs_s, ".", ms=2, label=r"$|B_{\mathrm{MFS}}|$")
    axes[0].set_ylabel(r"$|B|$")
    axes[0].set_title(r"Boundary field decomposition vs toroidal angle $\varphi$")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(phi_s, ndot_s, ".", ms=2)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_ylabel(r"$\mathbf{n}\!\cdot\!\mathbf{B}$")

    axes[2].plot(phi_s, rho_s, ".", ms=2, color="tab:green")
    axes[2].set_xlabel(r"Toroidal angle $\varphi$ [rad]")
    axes[2].set_ylabel(r"$\rho$ (distance to axis)")

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_boundary_decomposition_vs_rho(rho, Bmag_tot, Bmag_mv, Bmag_s,
                                       n_dot_B, outfile):
    """
    Plot |B|, |B_mv|, |B_s| and n·B vs distance to axis ρ.
    """
    rho_np = np.asarray(rho)
    Btot = np.asarray(Bmag_tot)
    Bmv = np.asarray(Bmag_mv)
    Bs  = np.asarray(Bmag_s)
    ndot = np.asarray(n_dot_B)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(rho_np, Btot, ".", ms=2, label=r"$|B|$")
    axes[0].plot(rho_np, Bmv, ".", ms=2, label=r"$|B_{\mathrm{mv}}|$")
    axes[0].plot(rho_np, Bs,  ".", ms=2, label=r"$|B_{\mathrm{MFS}}|$")
    axes[0].set_ylabel(r"$|B|$")
    axes[0].set_title(r"Boundary field decomposition vs distance to axis $\rho$")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(rho_np, ndot, ".", ms=2)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel(r"$\rho$ (distance to axis)")
    axes[1].set_ylabel(r"$\mathbf{n}\!\cdot\!\mathbf{B}$")

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_boundary_geometry_plots(W, h, rho, outfile):
    """
    Plot geometric / quadrature diagnostics vs boundary index:
      - distance to axis ρ
      - area weights W
      - spacing h
    """
    W_np = np.asarray(W)
    h_np = np.asarray(h)
    rho_np = np.asarray(rho)
    idx = np.arange(len(W_np))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(idx, rho_np, ".", ms=2)
    axes[0].set_ylabel(r"$\rho$")

    axes[1].plot(idx, W_np, ".", ms=2)
    axes[1].set_ylabel(r"$W$ (area weight)")

    axes[2].plot(idx, h_np, ".", ms=2)
    axes[2].set_ylabel(r"$h$ (spacing)")
    axes[2].set_xlabel("Boundary point index")

    axes[0].set_title("Boundary geometry / quadrature diagnostics")

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main(xyz_csv,
         normals_csv,
         k_nn=32,
         reg=1e-8,
         phiedge=1.0,
         source_offset_factor=2.0,
         mfs_out=None,
         verbose=True,
         xs_n_r=32, xs_n_theta=64, xs_margin_frac=0.15):

    print("========================================================")
    print(" MFS-based Neumann Laplace Solver (B = ∇φ)")
    print(" VMEC-like energy-minimizing vacuum field with fixed phiedge")
    print("========================================================")
    print(f"[PARAM] k_nn                = {k_nn}")
    print(f"[PARAM] reg (normal eq.)    = {reg:.1e}")
    print(f"[PARAM] phiedge (Φ_tor)     = {phiedge:.6g}")
    print(f"[PARAM] source_offset_factor= {source_offset_factor:.3f}")

    # Resolve paths and output directory
    xyz_path = Path(xyz_csv).resolve()
    base_name = xyz_path.stem
    out_dir = (script_dir / ".." / "outputs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PATHS] Output directory   : {out_dir}")

    # Load surface geometry
    P, N = load_surface_xyz_normals(xyz_csv, normals_csv, verbose=verbose)
    N = maybe_flip_normals(P, N)

    # Normalize geometry (for multivalued bases)
    Pn, scinfo = normalize_geometry(P, verbose=verbose)

    # Area weights & spacings from normalized coords, rescaled to world units
    Wn, h_n = estimate_area_weights_knn(Pn, k=k_nn)
    scale = float(scinfo.scale)

    W = Wn / (scale**2)    # world-area weights
    h = h_n / scale        # world spacings
    h_min = float(jnp.min(h))
    h_med = float(jnp.median(h))
    print(f"[QUAD] h_min={h_min:.3e}, h_med={h_med:.3e}")
    vec_stats("[QUAD] W (world)", W)

    # Geometry classification and axis detection
    kind, a_hat, E, svals = detect_geometry_and_axis(Pn, verbose=verbose)

    # Multivalued bases in normalized coords
    grad_t_fn, grad_p_fn = multivalued_bases_about_axis(Pn, N, a_hat, verbose=verbose)

    # MFS sources on fictitious outer surface
    Y_src = build_mfs_sources(P, N, h, source_offset_factor=source_offset_factor)

    # --- Step 1: basis solves for (1,0) and (0,1) with MFS ---

    (Dt, Dp), (c1, c2), (B_t_bdry, B_p_bdry) = solve_basis_fields_mfs(
        P, N, Pn, scinfo, grad_t_fn, grad_p_fn, Y_src, reg=reg
    )
    print(f"[BASIS] ||B_t||_avg = {jnp.linalg.norm(B_t_bdry)/float(B_t_bdry.shape[0]):.6e}, "
          f"||B_p||_avg = {jnp.linalg.norm(B_p_bdry)/float(B_p_bdry.shape[0]):.6e}")

    # Diagnostics: boundary residuals for each basis
    A = build_mfs_neumann_matrix(P, N, Y_src)
    n_dot_Bs1 = A @ c1
    n_dot_Bs2 = A @ c2
    res1 = Dt + n_dot_Bs1
    res2 = Dp + n_dot_Bs2
    vec_stats("[BASIS] BC residual for basis 1 (n·B_mv + n·B_s)", res1)
    vec_stats("[BASIS] BC residual for basis 2 (n·B_mv + n·B_s)", res2)

    # --- Step 2: build basis field evaluators ---

    (phi1_fn, B1_fn, B1_mv_fn, B1_s_fn), (phi2_fn, B2_fn, B2_mv_fn, B2_s_fn) = \
        build_basis_evaluators_mfs(
            P, N, Y_src, scinfo, grad_t_fn, grad_p_fn, c1, c2
        )

    # --- Step 3: energy matrix M and flux vector c on poloidal cross-section ---

    M, c_vec, X_cs, B1_cs, B2_cs = build_energy_flux_matrices_on_cross_section(
        P, a_hat, E, scinfo,
        B1_fn, B2_fn,
        n_r=xs_n_r, n_theta=xs_n_theta, margin_frac=xs_margin_frac
    )

    # --- Step 4: solve constrained minimization for a* ---

    a_star = solve_energy_minimizing_coeffs(M, c_vec, phiedge=phiedge)
    a_t, a_p = float(a_star[0]), float(a_star[1])

    # --- Step 5: build final coefficients and field evaluators for a* ---

    c_star = a_star[0] * c1 + a_star[1] * c2
    B_mv_star_fn = make_B_mv_evaluator(a_star, grad_t_fn, grad_p_fn, scinfo)
    phi_s_star_fn, B_star_fn, _, B_s_star_fn = make_total_field_evaluators_for_fixed_a_mfs(
        Y_src, c_star, scinfo, a_star, grad_t_fn, grad_p_fn
    )

    # Diagnostics: BC residual for a*
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

    rho_bdry, phi_bdry = compute_axis_coordinates(P, scinfo, a_hat, E)
    vec_stats("ρ (distance to axis) on Γ", rho_bdry)

    # --- Step 7: inner-shell diagnostics and Laplacian of φ_s ---

    X_shell, B_shell, n_dot_B_shell, Bmag_shell = diagnostics_on_inner_shell(
        P, N, W, B_star_fn, h_min=h_min, eps_factor=0.3, label="inner shell"
    )

    h_fd = 0.5 * h_min
    lap_phi_s_inner = numerical_laplacian_phi_s(phi_s_star_fn, X_shell, h_fd=h_fd)
    vec_stats("∇²φ_s on inner shell (samples)", lap_phi_s_inner)

    # --- Step 8: plots (all saved under ../outputs with base_name prefix) ---

    boundary_png = out_dir / f"{base_name}_boundary_diagnostics.png"
    residual_png = out_dir / f"{base_name}_1d_diagnostics.png"
    decomp_phi_png = out_dir / f"{base_name}_boundary_decomposition_vs_phi.png"
    decomp_rho_png = out_dir / f"{base_name}_boundary_decomposition_vs_rho.png"
    geom_png = out_dir / f"{base_name}_boundary_geometry_weights.png"

    make_3d_boundary_plots(P, Bmag_bdry, q_bdry, boundary_png)
    make_1d_residual_plots(q_bdry, lap_phi_s_inner, residual_png)
    make_boundary_decomposition_vs_phi(
        phi_bdry, Bmag_bdry, Bmag_mv_bdry, Bmag_s_bdry,
        n_dot_B_bdry, rho_bdry, decomp_phi_png
    )
    make_boundary_decomposition_vs_rho(
        rho_bdry, Bmag_bdry, Bmag_mv_bdry, Bmag_s_bdry,
        n_dot_B_bdry, decomp_rho_png
    )
    make_boundary_geometry_plots(W, h, rho_bdry, geom_png)

    print(f"[PLOTS] Saved 3D boundary diagnostics to      {boundary_png}")
    print(f"[PLOTS] Saved 1D residual diagnostics to     {residual_png}")
    print(f"[PLOTS] Saved field decomposition vs φ to    {decomp_phi_png}")
    print(f"[PLOTS] Saved field decomposition vs ρ to    {decomp_rho_png}")
    print(f"[PLOTS] Saved geometry/weights diagnostics to {geom_png}")

    # --- Step 9: save checkpoint for downstream use ---

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
        )
        print(f"[SAVE] Wrote MFS energy-minimizing solution checkpoint → {mfs_out_path}")
    except Exception as e:
        print("[WARN] Could not save checkpoint:", e)

    print("========================================================")
    print(" Done.")
    print("========================================================")


if __name__ == "__main__":
    # Default example (adjust file_name to your case)
    file_name = "wout_precise_QH"
    # file_name = "wout_precise_QA"
    candidate_xyz, candidate_normals = get_candidates(file_name, subdir="inputs")

    parser = argparse.ArgumentParser(
        description="MFS-based Neumann Laplace solver (vacuum B = ∇φ) "
                    "with VMEC-like energy minimization and fixed toroidal flux."
    )
    parser.add_argument("xyz", nargs="?", default=candidate_xyz,
                        help="CSV file with x,y,z columns (positional or --xyz)")
    parser.add_argument("normals", nargs="?", default=candidate_normals,
                        help="CSV file with nx,ny,nz columns (positional or --normals)")
    parser.add_argument("--k-nn", type=int, default=32,
                        help="k for kNN-based area weights (default: 32)")
    parser.add_argument("--source-offset-factor", type=float, default=2.0,
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

    args = parser.parse_args()

    main(
        xyz_csv=args.xyz,
        normals_csv=args.normals,
        k_nn=args.k_nn,
        reg=args.reg,
        phiedge=args.phiedge,
        mfs_out=args.mfs_out,
        verbose=True,
        source_offset_factor=args.source_offset_factor,
        xs_n_r=args.xs_n_r,
        xs_n_theta=args.xs_n_theta,
        xs_margin_frac=args.xs_margin_frac,
    )
