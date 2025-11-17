#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy-minimizing vacuum field with VMEC-like constraints
=========================================================

We compute a vacuum magnetic field B = ∇φ inside a closed 3D surface Γ
given by a point cloud (x,y,z) and outward normals n, enforcing

    n · B = 0  on Γ   (perfect conductor, field lines tangent to Γ),

and selecting among all such harmonic fields the one that

  (i)  has a prescribed *toroidal flux* phiedge, defined as
       Φ_tor = ∫_S B·a_hat dS through a poloidal cross-section S
       orthogonal to the PCA-based axis a_hat; and

  (ii) minimizes the *magnetic energy*

          E = ½ ∫_Ω |B|² dV

       approximated by revolving that cross-section around the
       straight axis line:

          dV ≈ (2π ρ) dS,  where ρ is the distance to the axis.

Representation
--------------

We split the potential as

    φ(x) = φ_mv(x) + φ_s(x),

where
  - φ_mv is a multi-valued harmonic potential whose gradient
    B_mv = ∇φ_mv spans the topological (toroidal / poloidal) space.
  - φ_s is a single-valued harmonic potential represented by a
    single-layer potential over Γ:

        φ_s(x) = - ∫_Γ σ(y) G(x,y) dS_y,   G(x,y) = 1/(4π|x-y|).

The field is

    B(x) = B_mv(x) + B_s(x),    B_s = ∇φ_s.

On Γ we impose

    n · B = 0  ⇒  n·B_mv + ∂_n φ_s = 0.

Using the interior Neumann jump relation for the single-layer potential,

    ∂_n φ_s(x0) = -½ σ(x0) - ∫_Γ σ(y) ∂_{n_x}G(x0,y) dS_y,

we obtain the boundary integral equation

    (½ I + K') σ = g,      g_i = n_i · B_mv(x_i),

where

    K'_{ij} ≈ ∂_{n_x} G(x_i, x_j) W_j.

We construct B_mv as a linear combination of two axis-aware multivalued
basis fields (toroidal/poloidal):

    B_mv(x) = a_t b_t(x) + a_p b_p(x).

The coefficients a = (a_t, a_p)ᵀ parametrize the 2D space of harmonic
fields in a toroidal domain.

Energy and toroidal flux (VMEC-like)
------------------------------------

For two basis choices a^(1)=(1,0), a^(2)=(0,1), we solve the BIE twice:

  A σ^(k) = g^(k),  k=1,2,

to obtain two full vacuum fields

  B^(k)(x) = B_mv^(k)(x) + B_s^(k)(x)      (k = 1,2).

Any linear combination B(a) = a_1 B^(1) + a_2 B^(2) satisfies Maxwell
and the boundary condition n·B = 0.

We then construct a *poloidal cross-section* S:

  - Axis direction a_hat from PCA (kind="torus").
  - Center at the geometric center of the point cloud.
  - Plane orthogonal to a_hat.
  - Polar grid (r,θ) inside the torus footprint in that plane.

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
    M_{ij} = ∑_q 2π ρ_q w_q [B^(i)(x_q)·B^(j)(x_q)].

This is VMEC-like: toroidal flux is a flux through a poloidal
cross-section, and energy is a 3D integral under an axisymmetry
assumption around the straight axis a_hat.

Given a user-specified phiedge = Φ_tor target, we solve the quadratic
program

    minimize  E(a) = ½ aᵀ M a
    subject to   cᵀ a = phiedge,

with c_i = Φ_tor(B^(i)). The solution is analytic:

    a* = (phiedge / (cᵀ M⁻¹ c)) M⁻¹ c.

JAX and differentiability
-------------------------

All core numerics (kernels, BIE assembly, solves, multivalued bases,
field evaluation, energy and flux functionals) are implemented in JAX,
so that φ, B, and the energy functional are differentiable with respect
to geometry (point cloud P, normals N, etc).

Diagnostics and plots
---------------------

The script prints:

  - Quadrature and geometry stats
  - Integral equation residual ‖(½I+K')σ−g‖
  - Boundary-normal residual analytically from the jump relation
  - |B|, n·B, and n·B/|B| *on Γ* (regularized singularity subtraction)
  - Laplacian residual ∇²φ_s at an inner shell (via finite differences)

and produces plots:

  1. 3D: |B| on Γ, and n·B/|B| on Γ.
  2. 1D: n·B/|B| at each boundary point (index vs residual).
  3. 1D: ∇²φ_s at inner-shell sample points (index vs residual).

Usage
-----

  python bie_vacuum_energy_min_vmec_like.py \
      --xyz inputs/wout_precise_QH.csv \
      --normals inputs/wout_precise_QH_normals.csv \
      --phiedge 1.0

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
      - multivalued bases built in normalized coordinates
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
    Npts = P.shape[0]
    D2 = pairwise_dist2(P)              # (N,N)
    big = jnp.max(D2) + 1.0
    D2 = D2 + jnp.eye(Npts) * big       # kill self-distance

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
    the geometry as 'torus' or 'mirror'.

      - 'torus': two large singular values, one much smaller.
      - 'mirror': one very large singular value, two comparable smaller.

    Returns:
      kind: 'torus' or 'mirror'
      a_hat: unit axis vector in normalized-space coordinates
      E:     3x3 matrix of principal directions (columns)
      svals: singular values (descending)
    """
    X = Pn - jnp.mean(Pn, axis=0)
    U, S, Vt = jnp.linalg.svd(X, full_matrices=False)
    E = Vt.T  # principal directions as columns
    s = S     # descending

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
    Build *harmonic* basis gradients in normalized coordinates:

      - grad_t(Xn): toroidal basis, ∇φ_tor with φ_tor = a_hat·x  (constant field).
      - grad_p(Xn): poloidal basis, ∇φ_pol = ∇ϕ_a (azimuth around axis a_hat),
                    harmonic and multi-valued, singular only on the axis line.

    Both are gradients of harmonic scalars, so any linear combination
    a_t grad_t + a_p grad_p is a vacuum field (up to the axis singularity),
    and adding ∇φ_s (single-layer) preserves the vacuum property.
    """
    Pn_ref = jnp.asarray(Pn_ref)
    N_ref  = jnp.asarray(N_ref)
    a_hat  = jnp.asarray(a_hat)

    a_unit = a_hat / jnp.maximum(jnp.linalg.norm(a_hat), 1e-30)

    def grad_t(Xn):
        """
        Toroidal-like basis: constant harmonic field along the PCA axis.

          φ_t(X) = a_hat·X  ⇒  ∇φ_t = a_hat (constant).

        In world coordinates, B_t_mv = scale * a_hat; since scale is
        applied outside, here we just broadcast a_hat in normalized coords.
        """
        Xn = jnp.asarray(Xn)
        # shape (M,3), each row = a_unit
        return jnp.broadcast_to(a_unit, Xn.shape)

    def grad_p(Xn):
        """
        Poloidal-like basis: azimuth around the axis a_hat.

          r_par  = (X·a_hat)a_hat
          r_perp = X - r_par
          ∇ϕ_a = (a_hat × r_perp)/|r_perp|^2

        This is the gradient of a multi-valued harmonic scalar away from
        the axis line, giving the "looping" poloidal direction.
        """
        return grad_azimuth_about_axis(Xn, a_hat)

    if verbose:
        print("[MV] Using *harmonic* multivalued bases:")
        print("     - grad_t: constant along axis (toroidal flux carrier).")
        print("     - grad_p: azimuth around axis (poloidal loops).")

    return grad_t, grad_p

# ----------------------------------------------------------------------
# Kernels: Green's function and its normal derivative (single-layer)
# ----------------------------------------------------------------------

@jit
def build_Kprime(P, N, W, h, clip_factor=0.2):
    """
    Build K' with near-singular regularization:

        r_eff^2 = max(r^2, (clip_factor * h_min)^2).

    where h_min is the global min spacing. This mimics integrating over
    a finite patch instead of a point singularity.
    """
    X = P
    Ni = N
    Wj = W

    Xi = X[:, None, :]  # (N,1,3)
    Xj = X[None, :, :]  # (1,N,3)
    diff = Xi - Xj      # (N,N,3)
    r2 = jnp.sum(diff*diff, axis=-1)  # (N,N)

    h_min = jnp.min(h)
    r2_clip = (clip_factor * h_min)**2

    mask = ~jnp.eye(X.shape[0], dtype=bool)

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
    Solve (½ I + K') σ = g with clipped near-field and small
    Tikhonov regularization reg * I.
    """
    Npts = P.shape[0]
    Kprime = build_Kprime(P, N, W, h)
    I = jnp.eye(Npts)
    A = -0.5 * I + Kprime + reg * I

    # correct RHS:
    rhs = -g

    jdbg.print("[SOLVE] Assembling A, shape=({n},{m})", n=A.shape[0], m=A.shape[1])
    jdbg.print("[SOLVE] rhs g stats: L2={l2:.3e}, Linf={linf:.3e}",
               l2=jnp.linalg.norm(g), linf=jnp.max(jnp.abs(g)))

    sigma = jnp.linalg.solve(A, rhs)
    jdbg.print("[SOLVE] ||sigma||_2 = {ns:.3e}", ns=jnp.linalg.norm(sigma))
    return sigma

def diagnostics_integral_eq(P, N, W, h, g, sigma):
    """
    Diagnostic for the integral equation (½I + K')σ = g.
    """
    Kprime = build_Kprime(P, N, W, h)
    Npts = P.shape[0]
    A = -0.5 * jnp.eye(Npts) + Kprime
    r = A @ sigma + g   # (-1/2 I + K')σ + g = 0 ideally

    vec_stats("[IE] residual r = (-½I + K')σ + g", r)
    return np.asarray(r)

def diagnostics_normal_on_boundary(P, N, W, h, g, sigma):
    """
    Compute n·B on Γ using the analytic jump relation:

        n·B = g - (½ σ + K'σ),

    where g = n·B_mv was used in (½I + K')σ = g.
    """
    Kprime = build_Kprime(P, N, W, h)
    sigma = jnp.asarray(sigma)
    g = jnp.asarray(g)

    Ksigma = Kprime @ sigma
    n_dot_B = g + (-0.5 * sigma + Ksigma)  # g + ∂n φ_s

    vec_stats("[BC-analytic] n·B on Γ (jump formula)", n_dot_B)

    flux = float(jnp.dot(W, n_dot_B))
    area = float(jnp.sum(W))
    print(f"[BC-analytic] Flux through Γ: Φ ≈ {flux:.6e}, avg n·B ≈ {flux/area:.3e}")

    return np.asarray(n_dot_B)

# ----------------------------------------------------------------------
# Single-layer potential evaluation (φ_s, B_s) in JAX (interior points)
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
        return jnp.sum(Gvals * weight)

    @jit
    def grad_phi_s_at_point(x):
        diff = x[None, :] - P_src      # (N,3)
        r2 = jnp.sum(diff*diff, axis=-1)

        r2_clip = (clip_factor * h_min)**2
        r2 = jnp.maximum(r2, r2_clip)

        r3 = r2 * jnp.sqrt(r2)
        gradG = -diff / (4.0 * jnp.pi * r3[:, None])  # (N,3)
        return jnp.sum(gradG * weight[:, None], axis=0)

    phi_s_batch  = jit(vmap(phi_s_at_point, in_axes=(0,)))
    grad_s_batch = jit(vmap(grad_phi_s_at_point, in_axes=(0,)))

    return phi_s_batch, grad_s_batch

def make_total_field_evaluators_for_fixed_a(P_src, W_src, sigma,
                                            scinfo: ScaleInfo,
                                            a, grad_t_fn, grad_p_fn,
                                            h_min, clip_factor=0.2):
    """
    Return JIT-compiled evaluators for a *fixed* coefficient vector a:

        phi_s_fn(X): single-layer potential φ_s(X)
        B_fn(X):     full B = B_mv + B_s
        B_mv_fn(X):  multivalued part only
    """
    phi_s_fn, grad_s_fn = make_single_layer_evaluators(
        P_src, W_src, sigma, h_min, clip_factor
    )
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

    return phi_s_fn, B_fn, B_mv_fn

# ----------------------------------------------------------------------
# Basis solves and energy/flux functionals
# ----------------------------------------------------------------------

def solve_basis_fields(P, N, W, h,
                       Pn, scinfo: ScaleInfo,
                       grad_t_fn, grad_p_fn,
                       reg=1e-6):
    """
    Solve the BIE twice for basis coefficients:

      a^(1) = (1,0), a^(2) = (0,1),
    """
    Gt_bdry = grad_t_fn(Pn)
    Gp_bdry = grad_p_fn(Pn)

    B_t_bdry = scinfo.scale * Gt_bdry
    B_p_bdry = scinfo.scale * Gp_bdry

    Dt = jnp.sum(N * B_t_bdry, axis=1)
    Dp = jnp.sum(N * B_p_bdry, axis=1)

    vec_stats("[BASIS] g_t = n·B_t", Dt)
    vec_stats("[BASIS] g_p = n·B_p", Dp)

    sigma1 = solve_density_sigma(P, N, W, h, Dt, reg=reg)
    sigma2 = solve_density_sigma(P, N, W, h, Dp, reg=reg)

    return (Dt, Dp), (sigma1, sigma2), (B_t_bdry, B_p_bdry)

def build_basis_evaluators(P, W, h_min,
                           scinfo: ScaleInfo,
                           grad_t_fn, grad_p_fn,
                           sigma1, sigma2):
    """
    Build JAX field evaluators for the two basis fields:

      B^(1) : a = (1,0)
      B^(2) : a = (0,1)
    """
    a1 = jnp.array([1.0, 0.0])
    a2 = jnp.array([0.0, 1.0])

    phi1_fn, B1_fn, B1_mv_fn = make_total_field_evaluators_for_fixed_a(
        P, W, sigma1, scinfo, a1, grad_t_fn, grad_p_fn, h_min
    )
    phi2_fn, B2_fn, B2_mv_fn = make_total_field_evaluators_for_fixed_a(
        P, W, sigma2, scinfo, a2, grad_t_fn, grad_p_fn, h_min
    )

    return (phi1_fn, B1_fn, B1_mv_fn), (phi2_fn, B2_fn, B2_mv_fn)

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

    Returns:
      X_cs : (Nq,3) cross-section points
      w_area : (Nq,) area weights dS = r dr dθ
      rho : (Nq,) distances to the axis (>= r_in)
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

def build_energy_flux_matrices_on_cross_section(P,
                                                a_hat,
                                                E,
                                                scinfo: ScaleInfo,
                                                B1_fn, B2_fn,
                                                n_r=32, n_theta=64,
                                                margin_frac=0.15):
    """
    Construct M (2x2) and c (2,) using a VMEC-like poloidal cross-section.

      - M encodes the 3D magnetic energy via volume weights
        w_vol = 2π ρ w_area.
      - c encodes the toroidal flux via Φ_tor = ∫_S B·a_hat dS.

    Returns:
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
    proj1 = jnp.sum(B1_cs * a_hat[None,:], axis=1)
    proj2 = jnp.sum(B2_cs * a_hat[None,:], axis=1)

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
    X = jnp.asarray(P)
    Nw = jnp.asarray(N)
    Ww = jnp.asarray(W)

    if h_min is None:
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
        return phi_s_fn(X + h * direction[None,:]), phi_s_fn(X - h * direction[None,:])

    phi0 = phi_s_fn(X)

    phi_px_x, phi_mx_x = phi_at_offset(X, ex)
    phi_px_y, phi_mx_y = phi_at_offset(X, ey)
    phi_px_z, phi_mx_z = phi_at_offset(X, ez)

    lap = (phi_px_x + phi_mx_x - 2.0*phi0
         + phi_px_y + phi_mx_y - 2.0*phi0
         + phi_px_z + phi_mx_z - 2.0*phi0) / (h*h)

    return np.asarray(lap)

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def make_3d_boundary_plots(P, Bmag, n_dot_B_norm):
    """
    3D diagnostic plots:
      - Surface colored by |B|
      - Surface colored by n·B/|B|
    """
    P_np = np.asarray(P)
    Bmag_np = np.asarray(Bmag)
    ndot_np = np.asarray(n_dot_B_norm)

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
    maxabs = np.max(np.abs(ndot_np))
    vmin2, vmax2 = -maxabs, maxabs
    sc2 = ax2.scatter(P_np[:,0], P_np[:,1], P_np[:,2],
                      c=ndot_np, s=6, cmap='magma',
                      vmin=vmin2, vmax=vmax2)
    fig.colorbar(sc2, ax=ax2, shrink=0.7, label="n·B/|B| on Γ")
    ax2.set_title("Boundary colored by n·B/|B|")
    fix_matplotlib_3d(ax2)

    plt.tight_layout()
    plt.savefig("boundary_diagnostics.png", dpi=150)

def make_1d_residual_plots(q_bdry, lap_inner):
    """
    1D plots:
      - q_bdry = n·B/|B| on Γ vs index
      - ∇²φ_s on inner shell vs index
    """
    q_np = np.asarray(q_bdry)
    lap_np = np.asarray(lap_inner)

    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    axes[0].plot(q_np, ".", ms=2)
    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[0].set_xlabel("Boundary point index")
    axes[0].set_ylabel("n·B/|B|")
    axes[0].set_title("Boundary residual n·B/|B|")

    axes[1].plot(lap_np, ".", ms=2)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("Inner-shell point index")
    axes[1].set_ylabel("∇²φ_s (FD)")
    axes[1].set_title("Laplacian residual ∇²φ_s on inner shell")

    plt.tight_layout()
    plt.savefig("1d_diagnostics.png", dpi=150)

# ----------------------------------------------------------------------
# Boundary field reconstruction with singularity subtraction
# ----------------------------------------------------------------------

@jit
def compute_Bs_tan_on_boundary(P, N, W, sigma, h, clip_factor=0.2):
    """
    Tangential part of B_s = ∇φ_s on Γ using singularity subtraction.

    For a constant σ, the tangential component is zero, so we subtract σ_i:
        B_s^tan(x_i) = - sum_j (σ_j - σ_i) W_j ∇G(x_i, x_j), projected to tangent.
    """
    X = jnp.asarray(P)
    Nw = jnp.asarray(N)
    Ww = jnp.asarray(W)
    sigma = jnp.asarray(sigma)

    Npts = X.shape[0]

    Xi = X[:, None, :]      # (N,1,3)
    Xj = X[None, :, :]      # (1,N,3)
    diff = Xi - Xj          # (N,N,3)
    r2 = jnp.sum(diff * diff, axis=-1)   # (N,N)

    h_min = jnp.min(h)
    r2_clip = (clip_factor * h_min)**2
    mask = ~jnp.eye(Npts, dtype=bool)

    r2 = jnp.where(mask, jnp.maximum(r2, r2_clip), 1.0)
    r3 = r2 * jnp.sqrt(r2)
    gradG = -diff / (4.0 * jnp.pi * r3[..., None])   # (N,N,3)

    Sigma_j = sigma[None, :]        # (1,N)
    Sigma_i = sigma[:, None]        # (N,1)
    weight = (Sigma_j - Sigma_i) * Ww[None, :]   # (N,N)

    Bs = -jnp.einsum("ij,ijk->ik", weight, gradG)   # (N,3)

    # Project to tangent plane
    n_hat = Nw / jnp.maximum(
        jnp.linalg.norm(Nw, axis=1, keepdims=True), 1e-30
    )
    n_dot_Bs = jnp.sum(n_hat * Bs, axis=1, keepdims=True)
    Bs_tan = Bs - n_hat * n_dot_Bs
    return Bs_tan  # (N,3)


def build_B_on_boundary_with_jump(P, N, W, h,
                                  scinfo: ScaleInfo,
                                  a, grad_t_fn, grad_p_fn,
                                  sigma, g,
                                  clip_factor=0.2):
    """
    Construct B on Γ with:
      - B_mv from multivalued bases (world coords);
      - B_s^tan from singularity subtraction;
      - B_s^n from analytic jump relation.
    """
    P = jnp.asarray(P)
    N = jnp.asarray(N)
    W = jnp.asarray(W)
    sigma = jnp.asarray(sigma)
    g = jnp.asarray(g)

    center = scinfo.center
    scale  = scinfo.scale
    a = jnp.asarray(a)

    # 1) multivalued part on boundary
    Xn = (P - center[None, :]) * scale
    Gt = grad_t_fn(Xn)
    Gp = grad_p_fn(Xn)
    B_mv = scale * (a[0] * Gt + a[1] * Gp)   # (N,3)

    # 2) tangential part of B_s from singularity subtraction
    B_s_tan = compute_Bs_tan_on_boundary(P, N, W, sigma, h, clip_factor=clip_factor)

    # 3) normal part of B_s from jump relation
    Kprime = build_Kprime(P, N, W, h)
    Ksigma = Kprime @ sigma
    n_dot_B_s = (-0.5 * sigma + Ksigma)      # (N,)

    n_hat = N / jnp.maximum(jnp.linalg.norm(N, axis=1, keepdims=True), 1e-30)
    B_s_n = n_hat * n_dot_B_s[:, None]       # (N,3)

    B_s = B_s_tan + B_s_n
    B_tot = B_mv + B_s
    return B_tot

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main(xyz_csv,
         normals_csv,
         k_nn=32,
         reg=1e-10,
         phiedge=1.0,
         mfs_out=None,
         verbose=True):

    print("========================================================")
    print(" Boundary-Integral Neumann Laplace Solver (B = ∇φ)")
    print(" VMEC-like energy-minimizing vacuum field with fixed phiedge")
    print("========================================================")
    print(f"[PARAM] k_nn   = {k_nn}")
    print(f"[PARAM] reg    = {reg:.1e} (small Tikhonov)")
    print(f"[PARAM] phiedge (toroidal flux) = {phiedge:.6g}")

    # Load surface geometry
    P, N = load_surface_xyz_normals(xyz_csv, normals_csv, verbose=verbose)
    N = maybe_flip_normals(P, N)

    # Normalize geometry (for multivalued bases)
    Pn, scinfo = normalize_geometry(P, verbose=verbose)

    # Area weights & spacings from normalized coords, rescaled to world units
    Wn, h_n = estimate_area_weights_knn(Pn, k=k_nn)
    scale = float(scinfo.scale)

    W = Wn / (scale**2)          # world-area weights
    h = h_n / scale              # world spacings
    h_min = float(jnp.min(h))

    vec_stats("[QUAD] W (world)", W)

    # Geometry classification and axis detection
    kind, a_hat, E, svals = detect_geometry_and_axis(Pn, verbose=verbose)

    # Multivalued bases in normalized coords
    grad_t_fn, grad_p_fn = multivalued_bases_about_axis(Pn, N, a_hat, verbose=verbose)

    # --- Step 1: basis solves for (1,0) and (0,1) ---

    (Dt, Dp), (sigma1, sigma2), (B_t_bdry, B_p_bdry) = solve_basis_fields(
        P, N, W, h, Pn, scinfo, grad_t_fn, grad_p_fn, reg=reg
    )
    print(f"[BASIS] ||σ₁||₂ = {jnp.linalg.norm(sigma1):.6e}, ||σ₂||₂ = {jnp.linalg.norm(sigma2):.6e}")
    print(f"[BASIS] ||B_t||_avg = {jnp.linalg.norm(B_t_bdry)/float(B_t_bdry.shape[0]):.6e}, "
          f"||B_p||_avg = {jnp.linalg.norm(B_p_bdry)/float(B_p_bdry.shape[0]):.6e}")
    print(f"[BASIS] Flux_t ≈ {jnp.dot(W, Dt):.6e}, Flux_p ≈ {jnp.dot(W, Dp):.6e}")

    # Integral-equation residuals for each basis
    _ = diagnostics_integral_eq(P, N, W, h, Dt, sigma1)
    _ = diagnostics_integral_eq(P, N, W, h, Dp, sigma2)

    # --- Step 2: build basis field evaluators ---

    (phi1_fn, B1_fn, B1_mv_fn), (phi2_fn, B2_fn, B2_mv_fn) = build_basis_evaluators(
        P, W, h_min, scinfo, grad_t_fn, grad_p_fn, sigma1, sigma2
    )

    # --- Step 3: energy matrix M and flux vector c on poloidal cross-section ---

    M, c, X_cs, B1_cs, B2_cs = build_energy_flux_matrices_on_cross_section(
        P, a_hat, E, scinfo,
        B1_fn, B2_fn,
        n_r=32, n_theta=64, margin_frac=0.15
    )

    # --- Step 4: solve constrained minimization for a* ---

    a_star = solve_energy_minimizing_coeffs(M, c, phiedge=phiedge)
    a_t, a_p = float(a_star[0]), float(a_star[1])

    # --- Step 5: build final σ* and field evaluators for a* ---

    sigma_star = a_star[0] * sigma1 + a_star[1] * sigma2
    g_star = Dt * a_star[0] + Dp * a_star[1]

    # Diagnostics: integral equation residual for σ*
    _ = diagnostics_integral_eq(P, N, W, h, g_star, sigma_star)

    # Analytic BC diagnostic using jump relation
    _ = diagnostics_normal_on_boundary(P, N, W, h, g_star, sigma_star)

    # Final field evaluators for a*
    phi_s_star_fn, B_star_fn, B_mv_star_fn = make_total_field_evaluators_for_fixed_a(
        P, W, sigma_star, scinfo, a_star, grad_t_fn, grad_p_fn,
        h_min, clip_factor=0.2
    )

    # --- Step 6: boundary diagnostics on Γ with analytic regularization ---

    B_bdry = build_B_on_boundary_with_jump(
        P, N, W, h,
        scinfo=scinfo,
        a=a_star,
        grad_t_fn=grad_t_fn,
        grad_p_fn=grad_p_fn,
        sigma=sigma_star,
        g=g_star,
        clip_factor=0.2,
    )

    Bmag_bdry = jnp.linalg.norm(B_bdry, axis=1)
    n_hat = N / jnp.maximum(jnp.linalg.norm(N, axis=1, keepdims=True), 1e-30)
    n_dot_B_bdry = jnp.sum(n_hat * B_bdry, axis=1)
    q_bdry = n_dot_B_bdry / jnp.maximum(Bmag_bdry, 1e-12)

    vec_stats("[BΓ] |B| on Γ", Bmag_bdry)
    vec_stats("[BΓ] n·B on Γ", n_dot_B_bdry)
    vec_stats("[BΓ] n·B/|B| on Γ", q_bdry)
    print(f"[BΓ] Flux through Γ: Φ ≈ {float(jnp.dot(W, n_dot_B_bdry)):.6e}")
    print(f"L2 norm of n·B/|B| on Γ: {jnp.linalg.norm(q_bdry):.6e}")
    print(f"Linf norm of n·B/|B| on Γ: {jnp.max(jnp.abs(q_bdry)):.6e}")
    print(f"L2 norm of n·B on Γ: {jnp.linalg.norm(n_dot_B_bdry):.6e}")
    print(f"Linf norm of n·B on Γ: {jnp.max(jnp.abs(n_dot_B_bdry)):.6e}")

    # --- Step 7: inner-shell diagnostics and Laplacian of φ_s ---

    X_shell, B_shell, n_dot_B_shell, Bmag_shell = diagnostics_on_inner_shell(
        P, N, W, B_star_fn, h_min=h_min, eps_factor=0.3, label="inner shell"
    )

    h_fd = 0.5 * h_min
    lap_phi_s_inner = numerical_laplacian_phi_s(phi_s_star_fn, X_shell, h_fd=h_fd)
    vec_stats("∇²φ_s on inner shell (samples)", lap_phi_s_inner)

    # --- Step 8: plots ---

    make_3d_boundary_plots(P, Bmag_bdry, q_bdry)
    make_1d_residual_plots(q_bdry, lap_phi_s_inner)
    plt.show()

    # --- Step 9: save checkpoint for downstream use ---

    if mfs_out is None:
        mfs_out = xyz_csv.replace(".csv", "_bie_energymin_vmec_like_solution.npz")
        mfs_out = str((Path(xyz_csv).parent / mfs_out).resolve())
    try:
        np.savez(
            mfs_out,
            center=np.asarray(scinfo.center),
            scale=float(np.asarray(scinfo.scale)),
            P=np.asarray(P),
            N=np.asarray(N),
            W=np.asarray(W),
            sigma=np.asarray(sigma_star),
            a=np.asarray(a_star),
            a_hat=np.asarray(a_hat),
            kind=str(kind),
            phiedge=float(phiedge),
        )
        print(f"[SAVE] Wrote energy-minimizing BIE solution checkpoint → {mfs_out}")
    except Exception as e:
        print("[WARN] Could not save checkpoint:", e)

    print("========================================================")
    print(" Done.")
    print("========================================================")

if __name__ == "__main__":
    # file_name = "wout_precise_QH"
    file_name = "wout_precise_QA"
    candidate_xyz, candidate_normals = get_candidates(file_name, subdir="inputs")

    parser = argparse.ArgumentParser(
        description="Boundary-integral Neumann Laplace solver (vacuum B = ∇φ) "
                    "with VMEC-like energy minimization and fixed toroidal flux."
    )
    parser.add_argument("xyz", nargs="?", default=candidate_xyz,
                        help="CSV file with x,y,z columns (positional or --xyz)")
    parser.add_argument("normals", nargs="?", default=candidate_normals,
                        help="CSV file with nx,ny,nz columns (positional or --normals)")
    parser.add_argument("--k-nn", type=int, default=32,
                        help="k for kNN-based area weights (default: 32)")
    parser.add_argument("--reg", type=float, default=1e-6,
                        help="Small Tikhonov regularization for Neumann system")
    parser.add_argument("--phiedge", type=float, default=1.0,
                        help="Toroidal flux (phiedge) for energy minimization")
    parser.add_argument("--out", dest="mfs_out", default=None,
                        help="Output .npz path for checkpoint (optional)")

    args = parser.parse_args()

    main(
        xyz_csv=args.xyz,
        normals_csv=args.normals,
        k_nn=args.k_nn,
        reg=args.reg,
        phiedge=args.phiedge,
        mfs_out=args.mfs_out,
        verbose=True,
    )
