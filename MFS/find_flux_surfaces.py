#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quadratic-Flux-Minimizing (QFMin) surfaces inside a 3D closed surface,
using the Laplace solution checkpoint produced by your main.py.

We DO NOT solve Laplace here. We reconstruct φ, ∇φ from:
  ψ(x) = Σ_j α_j G((x-c)*s, y_j),  ∇φ = a_t ∇ϕ_a + a_p θ̂ + ∇ψ.

Main upgrades vs. previous version:
  • Bound-method jit/vmap traps removed (pure functions only).
  • Stable graph-Laplacian smoothing (correct descent sign).
  • Iteratively updated surface normals from kNN covariance (JAX).
  • JITed inner loops; vmapped evaluators; small adaptive step cap.
  • Identical I/O contract (consumes …_solution.npz from main.py).

Usage:
  python qfm_surfaces.py mfs_checkpoint_solution.npz \
      --n-surfaces 3 --dt 0.25 --iters 60 --beta 0.03 --k-nn 64 --tol 1e-3 \
      --save-prefix qfm --plot
"""

from __future__ import annotations
import argparse
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap, jacrev, lax
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# ----------------------------- small utils ----------------------------- #

def pct(a, p): return float(np.percentile(np.asarray(a), p))

@jit
def _step_stats(step_n, step_s, X, Pb, Nb, margin):
    # norms
    n_norm_med = jnp.median(jnp.linalg.norm(step_n, axis=1))
    s_norm_med = jnp.median(jnp.linalg.norm(step_s, axis=1))
    # boundary distances
    s_here, _ = signed_dist_and_nout(X, Pb, Nb)
    s_min = jnp.min(s_here); s_med = jnp.median(s_here)
    near_frac = jnp.mean((s_here >= -2.0*margin).astype(jnp.float32))
    return n_norm_med, s_norm_med, s_min, s_med, near_frac

# ----------------------------- geometry ----------------------------- #

def best_fit_axis(points_np: np.ndarray):
    """PCA plane used to build (u,v) metrics and weights (NumPy once)."""
    c = np.mean(points_np, axis=0)
    X = points_np - c
    pca = PCA(n_components=3).fit(X)
    order = np.argsort(-pca.singular_values_)
    comps = pca.components_[order]
    e3 = comps[2]
    e1 = comps[0] - np.dot(comps[0], e3) * e3
    e1 /= np.linalg.norm(e1) + 1e-30
    e2 = np.cross(e3, e1)
    E = np.stack([e1, e2, e3], axis=1)
    return jnp.asarray(c), jnp.asarray(E), pca

@jit
def project_to_local(P: jnp.ndarray, c: jnp.ndarray, E: jnp.ndarray) -> jnp.ndarray:
    return (P - c) @ E

def knn_area_weights(P_like_np: np.ndarray, P_boundary_np: np.ndarray, k: int = 48) -> jnp.ndarray:
    """kNN area weights in boundary (u,v) plane; NumPy neighbors, JAX values."""
    c, E, _ = best_fit_axis(P_boundary_np)
    XY = np.asarray(project_to_local(jnp.asarray(P_like_np), c, E)[:, :2])
    k_eff = min(k + 1, len(XY))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree").fit(XY)
    dists, _ = nbrs.kneighbors(XY)
    rk = dists[:, -1]
    return jnp.asarray(np.pi * rk**2, dtype=jnp.float64)

def knn_graph_indices(P_np: np.ndarray, k: int = 16):
    """kNN graph (indices + weights) in ambient R^3 (NumPy once)."""
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(P_np)), algorithm="kd_tree").fit(P_np)
    dists, idxs = nbrs.kneighbors(P_np)
    idxs = idxs[:, 1:]  # drop self
    # weights: Gaussian based on neighborhood scale
    sigma = np.median(dists[:, -1]) + 1e-12
    W = np.exp(-(dists[:, 1:] ** 2) / (2 * sigma * sigma))
    return idxs.astype(np.int32), W.astype(np.float64)

# --------------------------- scaling (pytree) --------------------------- #

@jax.tree_util.register_pytree_node_class
@dataclass
class ScaleInfo:
    center: jnp.ndarray  # (3,)
    scale: jnp.ndarray   # scalar () array

    def normalize(self, X: jnp.ndarray) -> jnp.ndarray:
        return (X - self.center) * self.scale

    def denormalize(self, Xn: jnp.ndarray) -> jnp.ndarray:
        return Xn / self.scale + self.center

    # pytree hooks
    def tree_flatten(self):
        return ((self.center, self.scale), None)
    @classmethod
    def tree_unflatten(cls, aux, children):
        c, s = children
        return cls(center=c, scale=s)

# ---------------------------- field kernels ---------------------------- #

@jit
def green_G(x, y):
    r = jnp.linalg.norm(x - y, axis=-1)
    return 1.0 / (4.0 * jnp.pi * jnp.maximum(1e-30, r))

@jit
def grad_green_x(x, y):
    r = x - y
    r2 = jnp.sum(r * r, axis=-1)
    r3 = jnp.maximum(1e-30, r2 * jnp.sqrt(r2))
    return -r / (4.0 * jnp.pi * r3[..., None])

@jit
def grad_azimuth_about_axis(Xn, a_hat):
    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par = jnp.sum(Xn * a[None, :], axis=1, keepdims=True) * a[None, :]
    r_perp = Xn - r_par
    r2 = jnp.maximum(1e-30, jnp.sum(r_perp * r_perp, axis=1, keepdims=True))
    cross = jnp.cross(a[None, :], r_perp)
    return cross / r2

@jit
def nearest_normal_bruteforce(P_ref: jnp.ndarray, N_ref: jnp.ndarray, Xq: jnp.ndarray) -> jnp.ndarray:
    """Brute-force nearest neighbor in normalized coords (vmapped, OK for N~1e3)."""
    def nn1(x):
        d2 = jnp.sum((P_ref - x[None, :]) ** 2, axis=1)
        idx = jnp.argmin(d2)
        return N_ref[idx]
    return vmap(nn1)(Xq)

# --------------------------- Field container --------------------------- #

@jax.tree_util.register_pytree_node_class
@dataclass
class Field:
    sc: ScaleInfo
    Yn: jnp.ndarray       # (M,3) normalized sources
    alpha: jnp.ndarray    # (M,)
    a: jnp.ndarray        # (2,) [a_t, a_p]
    a_hat: jnp.ndarray    # (3,)
    Pn_ref: jnp.ndarray   # (Nb,3) normalized boundary points
    Nn_ref: jnp.ndarray   # (Nb,3)

    # pytree
    def tree_flatten(self):
        children = (self.sc, self.Yn, self.alpha, self.a, self.a_hat, self.Pn_ref, self.Nn_ref)
        return (children, None)
    @classmethod
    def tree_unflatten(cls, aux, children):
        sc, Yn, alpha, a, a_hat, Pn_ref, Nn_ref = children
        return cls(sc, Yn, alpha, a, a_hat, Pn_ref, Nn_ref)

# MV bases (free functions so JAX can trace them cleanly)
@jit
def grad_t_fn(Xn, a_hat):
    return grad_azimuth_about_axis(Xn, a_hat)

@jit
def grad_p_fn(Xn, a_hat, Pn_ref, Nn_ref):
    # θ̂ from (n × ϕ̂_tan)
    n = nearest_normal_bruteforce(Pn_ref, Nn_ref, Xn)
    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par = jnp.sum(Xn * a[None, :], axis=1, keepdims=True) * a[None, :]
    r_perp = Xn - r_par
    phi_hat = jnp.cross(a[None, :], r_perp)
    phi_hat = phi_hat / jnp.maximum(1e-30, jnp.linalg.norm(phi_hat, axis=1, keepdims=True))
    phi_tan = phi_hat - jnp.sum(phi_hat * n, axis=1, keepdims=True) * n
    phi_tan = phi_tan / jnp.maximum(1e-30, jnp.linalg.norm(phi_tan, axis=1, keepdims=True))
    theta_hat = jnp.cross(n, phi_tan)
    theta_hat = theta_hat / jnp.maximum(1e-30, jnp.linalg.norm(theta_hat, axis=1, keepdims=True))
    return theta_hat

# ψ and ∇ψ evaluators
@jit
def _S_alpha_at(xn, Yn, alpha):
    G = vmap(lambda y: green_G(xn, y))(Yn)
    return jnp.dot(G, alpha)

@jit
def _grad_S_alpha_at(xn, Yn, alpha):
    Gr = vmap(lambda y: grad_green_x(xn, y))(Yn)   # (M,3)
    return jnp.sum(Gr * alpha[:, None], axis=0)    # (3,)

@jit
def field_phi(field: Field, X: jnp.ndarray) -> jnp.ndarray:
    Xn = field.sc.normalize(X)
    return vmap(_S_alpha_at, in_axes=(0, None, None))(Xn, field.Yn, field.alpha)

@jit
def field_grad(field: Field, X: jnp.ndarray) -> jnp.ndarray:
    Xn = field.sc.normalize(X)
    dS = vmap(_grad_S_alpha_at, in_axes=(0, None, None))(Xn, field.Yn, field.alpha) * field.sc.scale
    Gt = grad_t_fn(Xn, field.a_hat)
    Gp = grad_p_fn(Xn, field.a_hat, field.Pn_ref, field.Nn_ref)
    mv = field.sc.scale * (field.a[0] * Gt + field.a[1] * Gp)
    return mv + dS

# ---------------------- surface normals (JAX) ---------------------- #

def build_fixed_knn(P0_np: np.ndarray, k_norm: int = 16):
    """Neighbor indices (NumPy) reused across iterations for speed."""
    nbrs = NearestNeighbors(n_neighbors=min(k_norm, len(P0_np)), algorithm="kd_tree").fit(P0_np)
    dists, idxs = nbrs.kneighbors(P0_np)
    return idxs.astype(np.int32)

@jit
def _normals_from_cov(X: jnp.ndarray, idxs: jnp.ndarray) -> jnp.ndarray:
    """
    Estimate normals as the smallest-eigenvector of neighbor covariance.
    idxs: (N,k) integer array of neighbor indices (fixed).
    """
    N, k = idxs.shape
    Xnbr = X[idxs, :]                    # (N,k,3)
    Xm = jnp.mean(Xnbr, axis=1, keepdims=True)
    Y = Xnbr - Xm                        # (N,k,3)
    # Cov = Y^T Y  (3x3) per point
    Cov = jnp.einsum('nki,nkj->nij', Y, Y) / (k + 1e-9)  # (N,3,3)
    # eigen-decomp; smallest eigenvector
    evals, evecs = jnp.linalg.eigh(Cov)  # ascending evals
    n = evecs[:, :, 0]                   # (N,3) eigenvector for smallest eigenvalue
    # make normals consistent/outward-ish by pointing roughly like initial mean
    n = n / jnp.maximum(1e-30, jnp.linalg.norm(n, axis=1, keepdims=True))
    return n

# ---------------------- quadratic flux & metrics ---------------------- #

@jit
def qflux_rms_linf(W: jnp.ndarray, Nsurf: jnp.ndarray, grad_on: jnp.ndarray):
    n_dot = jnp.sum(Nsurf * grad_on, axis=1)
    rms = jnp.sqrt(jnp.dot(W, n_dot * n_dot) / (jnp.sum(W) + 1e-30))
    linf = jnp.max(jnp.abs(n_dot))
    return rms, linf, n_dot

@jit
def signed_dist_and_nout(X: jnp.ndarray, Pb: jnp.ndarray, Nb: jnp.ndarray):
    """
    For each row of X:
      - find nearest boundary point P_nn and its outward normal N_out_nn
      - return signed distance s = (X - P_nn)·N_out_nn   (>0 => outside)
      - also return N_out_nn for clamping/alignment
    """
    x2 = jnp.sum(X * X, axis=1, keepdims=True)          # (N,1)
    p2 = jnp.sum(Pb * Pb, axis=1, keepdims=True).T       # (1,Nb)
    dist2 = x2 + p2 - 2.0 * (X @ Pb.T)                   # (N,Nb)
    idx = jnp.argmin(dist2, axis=1)                      # (N,)
    Pnn = Pb[idx]                                        # (N,3)
    Nnn = Nb[idx]                                        # (N,3)
    s = jnp.sum((X - Pnn) * Nnn, axis=1)                 # (N,)
    return s, Nnn


# ----------------------- QFMin evolution (JIT) ----------------------- #

def evolve_qfmin_surface(P0_world: jnp.ndarray,
                         field: Field,
                         P_boundary_ref_np: np.ndarray,
                         N_boundary_out_np: np.ndarray,
                         is_inside_func=None,
                         k_knn_weights: int = 48,
                         k_knn_normals: int = 16,
                         dt: float = 0.25,
                         beta: float = 0.05,
                         iters: int = 40,
                         tol: float = 1e-3,
                         verbose: bool = True,
                         margin_world: float = None):
    """
    Steepest-descent on ∫ (n·∇φ)^2 dS with graph-Laplacian smoothing.
    All heavy ops are JAX; neighbors are fixed (NumPy) for speed.
    """
    P0_np = np.asarray(P0_world)
    # static weights on initial geometry projected to boundary (u,v)
    W = knn_area_weights(P0_np, np.asarray(P_boundary_ref_np), k=k_knn_weights)
    # fixed neighbors for normals and smoothing
    idxs_norm = build_fixed_knn(P0_np, k_norm=k_knn_normals)
    idxs_smooth, Wg_np = knn_graph_indices(P0_np, k=max(6, k_knn_normals))
    # build symmetric dense normalized Laplacian once (NumPy) → JAX array
    Npts, k = idxs_smooth.shape
    L = np.zeros((Npts, Npts), dtype=np.float64)
    for i in range(Npts):
        wi = Wg_np[i]
        nbrs = idxs_smooth[i]
        for w, j in zip(wi, nbrs):
            L[i, j] -= w
            L[j, i] -= w
        L[i, i] += wi.sum()
    # normalized: L_norm = D^{-1/2} L D^{-1/2}
    d = np.clip(np.diag(L), 1e-12, None)
    Dmh = 1.0 / np.sqrt(d)
    L = (Dmh[:, None] * L) * Dmh[None, :]
    L = 0.5 * (L + L.T)
    L = jnp.asarray(L)


    W = jnp.asarray(W)
    idxs_norm = jnp.asarray(idxs_norm)
    X = jnp.asarray(P0_world)

    # Host-side references for inside checks / projections
    Pb_ref_np = np.asarray(P_boundary_ref_np)
    Nb_out_np = np.asarray(N_boundary_out_np)
    Pb = jnp.asarray(Pb_ref_np)      # (Nb,3)
    Nb = jnp.asarray(Nb_out_np)      # (Nb,3)
    is_inside_host = is_inside_func if is_inside_func is not None else build_inside_tester(Pb_ref_np, Nb_out_np)

    # Debug: initial inside stats
    inside_init = is_inside_host(np.asarray(X))
    print(f"[EVOLVE:init] Inside points: {inside_init.sum()}/{len(inside_init)}. "
        f"Min|X-P_nn|≈{np.min(np.linalg.norm(X - Pb_ref_np[NearestNeighbors(n_neighbors=1).fit(Pb_ref_np).kneighbors(np.asarray(X))[1][:, 0]], axis=1)):.3e} (rough)")

    # Convert world -> normalized coordinates (Field.sc.scale)
    # We’ll enforce the clamp in the same space we evolve X (world).
    if margin_world is None:
        # Fallback (rare): tie to a conservative fraction of a rough spacing
        # NOTE: h_world is already computed in main(); we prefer using margin_world from there.
        # If it wasn't provided, pick 0.1% of the median NN distance to boundary as a tiny buffer.
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(Pb_ref_np)
        dmin, _ = nn.kneighbors(np.asarray(P0_world))
        margin_world = float(np.median(dmin)) * 0.1

    margin = float(margin_world)  # keep clamp in world units

    # helper: one iteration (JITed)
    @jit
    def one_step(X, Pb, Nb, L, W, dt, beta, margin):
        """
        One descent step with:
        - inward alignment of geometry normals (vs nearest boundary outward normal)
        - tangent projection to remove outward motion when close to the wall
        - per-iteration clamp to keep all points inside (s <= -margin)
        """
        # Geometry normals from covariance (world coords)
        Nsurf_raw = _normals_from_cov(X, idxs_norm)

        # Align geometry normals to be *inward*
        s_here, Nout_here = signed_dist_and_nout(X, Pb, Nb)                 # (N,), (N,3)
        dot_no = jnp.sum(Nsurf_raw * Nout_here, axis=1, keepdims=True)      # (N,1)
        Nsurf = jnp.where(dot_no > 0.0, -Nsurf_raw, Nsurf_raw)              # inward-oriented

        # Field gradient and quadratic-flux terms
        G = field_grad(field, X)
        rms, linf, n_dot = qflux_rms_linf(W, Nsurf, G)

        # Steepest descent (along geometry normal) + smoothing
        step_n = (n_dot[:, None]) * Nsurf                                   # (N,3)
        step_s = L @ X                                                       # (N,3) graph-Laplacian smoothing

        # Gentle cap on the normal step to avoid spikes
        cap = jnp.maximum(5e-4, 0.2 * jnp.median(jnp.linalg.norm(step_n, axis=1)))
        sn_norm = jnp.linalg.norm(step_n, axis=1, keepdims=True) + 1e-30
        scale = jnp.minimum(1.0, cap / sn_norm)
        step_n = step_n * scale

        n_norm_med, s_norm_med, s_min, s_med, near_frac = _step_stats(step_n, step_s, X, Pb, Nb, margin)

        # adaptive smoothing scale to match magnitudes
        sn_med = jnp.median(jnp.linalg.norm(step_n, axis=1)) + 1e-12
        ss_med = jnp.median(jnp.linalg.norm(step_s, axis=1)) + 1e-12
        beta_eff = beta * (sn_med / ss_med)   # keeps smoothing “in range” of step_n
        dX = -dt * (step_n + beta_eff * step_s)

        # --- Tangent projection barrier near the wall ---
        # If point is closer than 2*margin to boundary, remove any outward component from dX
        # outward component amount:
        dX_out = jnp.sum(dX * Nout_here, axis=1, keepdims=True)             # (N,1)
        # mask of "near" points
        near = (s_here > -2.0 * margin).astype(dX.dtype)[:, None]           # (N,1)
        # only subtract outward (positive) component when near
        dX_corrected = dX - near * jnp.maximum(0.0, dX_out) * Nout_here

        # Tentative update
        X_new = X + dX_corrected

        # --- Hard inside clamp (backstop) ---
        s_new, Nout_new = signed_dist_and_nout(X_new, Pb, Nb)               # (N,), (N,3)
        overshoot = jnp.maximum(0.0, s_new + margin)[:, None]               # (N,1)
        X_new = X_new - overshoot * Nout_new

        return X_new, (rms, linf, jnp.mean(jnp.abs(n_dot)), jnp.max(jnp.abs(n_dot)),
               n_norm_med, s_norm_med, s_min, s_med, near_frac)

    def body_fun(it, carry):
        X, dt_curr = carry

        X_trial, aux_t = one_step(X, Pb, Nb, L, W, dt_curr, beta, margin)
        rms_t, linf_t, ndm_t, ndx_t, nnmed_t, smed_t, smin_t, smid_t, nfrac_t = aux_t

        # Current metrics (no step)
        G_now = field_grad(field, X)
        N_now = _normals_from_cov(X, idxs_norm)
        rms_now, linf_now, _ = qflux_rms_linf(W, N_now, G_now)

        improve = (rms_t < rms_now) | ((jnp.abs(rms_t - rms_now) < 1e-4 * (1.0 + rms_now)) & (linf_t <= linf_now))

        X_next = jnp.where(improve, X_trial, X)        # <- keep previous geometry if no improvement
        dt_next = jnp.where(improve,
                            jnp.minimum(1.10 * dt_curr, 1.5 * dt),
                            jnp.maximum(0.5 * dt_curr, 0.02 * dt))

        s_after, _ = signed_dist_and_nout(X_next, Pb, Nb)
        outs = jnp.sum((s_after >= -margin).astype(jnp.int32))
        r_log = jnp.where(improve, rms_t, rms_now)
        l_log = jnp.where(improve, linf_t, linf_now)

        # Option A: print boolean directly
        jax.debug.print(
            "[EVOLVE@{it}] rms={r:.3e} linf={l:.3e} outs={o} dt={d:.3e} "
            "||step_n||_med={nn:.3e} ||step_s||_med={sn:.3e} s_min={smin:.3e} "
            "s_med={smed:.3e} near%={pf:.1f} improve={imp}",
            it=it, r=r_log, l=l_log, o=outs, d=dt_next,
            nn=nnmed_t, sn=smed_t, smin=smin_t, smed=smid_t, pf=100.0*nfrac_t,
            imp=improve
        )

        return (X_next, dt_next)

    X_final, _dt_final = lax.fori_loop(0, iters, body_fun, (X, dt))

    # ensure strict inside at the end
    X_np = np.asarray(X_final)
    X_np = project_inside(np.asarray(Pb), np.asarray(Nb), X_np, eps=0.05*float(margin_world), max_iters=4)
    X_final = jnp.asarray(X_np)

    # Final metrics (non-jitted print friendly)
    Nsurf_final = _normals_from_cov(X_final, idxs_norm)
    G_final = field_grad(field, X_final)
    rms_f, linf_f, _ = qflux_rms_linf(W, Nsurf_final, G_final)
    if verbose:
        print(f"[QFM] done  RMS(n·∇φ)≈{float(rms_f):.3e}  Linf≈{float(linf_f):.3e}")

    # after evolve
    Nf = _normals_from_cov(X_final, idxs_norm)
    Gf = field_grad(field, X_final)
    n_dot = jnp.sum(Nf * Gf, axis=1)
    G_norm = jnp.linalg.norm(Gf, axis=1) + 1e-30
    sin_angle = jnp.abs(n_dot) / G_norm
    print("[DIAG] mean|n·G|/|G| =", float(jnp.mean(sin_angle)),
        "max =", float(jnp.max(sin_angle)))

    return np.asarray(X_final), np.asarray(Nsurf_final)

# ------------------------------ I/O ------------------------------ #

def load_mfs_checkpoint(path: str):
    data = np.load(path, allow_pickle=True)
    center = jnp.asarray(data["center"], dtype=jnp.float64)
    scale = jnp.asarray(data["scale"], dtype=jnp.float64)
    Yn = jnp.asarray(data["Yn"], dtype=jnp.float64)
    alpha = jnp.asarray(data["alpha"], dtype=jnp.float64)
    a = jnp.asarray(data["a"], dtype=jnp.float64)            # [a_t, a_p]
    a_hat = jnp.asarray(data["a_hat"], dtype=jnp.float64)
    P = jnp.asarray(data["P"], dtype=jnp.float64)
    N = jnp.asarray(data["N"], dtype=jnp.float64)
    kind = str(data["kind"]) if "kind" in data else "torus"
    sc = ScaleInfo(center=center, scale=scale)
    # normalized refs for MV bases
    Pn = (P - center) * scale
    Nn = N
    field = Field(sc=sc, Yn=Yn, alpha=alpha, a=a, a_hat=a_hat, Pn_ref=Pn, Nn_ref=Nn)
    return field, np.asarray(P), np.asarray(N), kind

# -------------------------- orientation & inside tests -------------------------- #

def make_normals_globally_outward(P: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Heuristic: flip any boundary normal whose average radial projection is inward.
    Uses barycenter c and (P - c) · N. Then propagate consistency by kNN smoothing.
    """
    c = P.mean(axis=0)
    s = (P - c) * N
    score = s.sum(axis=1)
    N_fix = N.copy()
    # Flip those obviously wrong relative to barycenter
    flips0 = score < 0.0
    N_fix[flips0] *= -1.0

    # Consistency propagation with kNN majority vote
    from sklearn.neighbors import NearestNeighbors
    k = min(16, len(P))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(P)
    _, idxs = nbrs.kneighbors(P)
    votes = np.sign((N_fix[idxs] * N_fix[:, None, :]).sum(axis=2))  # (N,k) pairwise cos
    maj = (votes.sum(axis=1) >= 0.0)
    flips1 = ~maj
    N_fix[flips1] *= -1.0

    print(f"[ORIENT] Boundary normal fixes: flips0={flips0.sum()}, flips1={flips1.sum()} (after propagation)")
    # Final check
    avg = ((P - c) * N_fix).sum(axis=1).mean()
    print(f"[ORIENT] ⟨(P-c)·N_out⟩≈{avg:.3e} (should be > 0 for outward).")
    return N_fix

def build_inside_tester(Pb: np.ndarray, Nb_out: np.ndarray):
    """
    Returns a function is_inside(X) using nearest boundary normal test:
      inside if (X - P_nn) · N_out_nn < 0.
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(Pb)
    def is_inside(X: np.ndarray) -> np.ndarray:
        dists, idx = nbrs.kneighbors(X)
        idx = idx[:, 0]
        v = X - Pb[idx]
        s = (v * Nb_out[idx]).sum(axis=1)
        return s < 0.0
    return is_inside

def project_inside(Pb: np.ndarray, Nb_out: np.ndarray, X: np.ndarray, eps: float = 1e-4,
                   max_iters: int = 10, growth: float = 1.5) -> np.ndarray:
    """
    Iteratively nudge points that are outside back inside along the nearest boundary
    outward normal (negative direction). Start with 'eps' and grow by 'growth' each try.
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(Pb)

    Xw = X.copy()
    step = float(eps)
    for it in range(max_iters):
        dists, idx = nbrs.kneighbors(Xw)
        idx = idx[:, 0]
        v = Xw - Pb[idx]
        s = (v * Nb_out[idx]).sum(axis=1)
        mask = s >= 0.0   # outside if dot >= 0
        n_out = Nb_out[idx][mask]
        if mask.any():
            Xw[mask] = Xw[mask] - step * n_out
            print(f"[PROJ] Iter {it+1}: projected {mask.sum()} points by {step:g}.")
            step *= growth
        else:
            print(f"[PROJ] All projected inside after {it+1} iteration(s).")
            break
    # Final report
    dists, idx = nbrs.kneighbors(Xw)
    idx = idx[:, 0]
    v = Xw - Pb[idx]
    s = (v * Nb_out[idx]).sum(axis=1)
    still_out = int((s >= 0.0).sum())
    if still_out:
        print(f"[PROJ] WARNING: {still_out} points still outside after max_iters={max_iters}.")
    return Xw

# ----------------------------- plotting ----------------------------- #

def plot_surfaces(boundary_P: np.ndarray, surfaces: list[np.ndarray], title: str = "QFMin surfaces"):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(boundary_P[:, 0], boundary_P[:, 1], boundary_P[:, 2], s=3, alpha=0.25, label='Boundary')
    for i, S in enumerate(surfaces):
        ax.scatter(S[:, 0], S[:, 1], S[:, 2], s=5, label=f'QFM {i+1}')
    ax.legend(loc='upper left')
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    plt.tight_layout()

def plot_poincare_slice(boundary_P: np.ndarray,
                        surfaces: list[np.ndarray],
                        phi0: float,
                        dphi: float):
    """
    Plot R-Z of points whose cylindrical angle φ is within [phi0-dphi, phi0+dphi].
    Shows boundary and all QFM surfaces.
    """
    def cyl_phi(X):
        return np.arctan2(X[:,1], X[:,0])

    def wrap_diff(a, b):
        # shortest signed diff between angles
        d = a - b
        d = (d + np.pi) % (2*np.pi) - np.pi
        return d

    # boundary slice
    phi_b = cyl_phi(boundary_P)
    mask_b = np.abs(wrap_diff(phi_b, phi0)) <= dphi
    Rb = np.sqrt(boundary_P[mask_b,0]**2 + boundary_P[mask_b,1]**2)
    Zb = boundary_P[mask_b,2]

    plt.figure(figsize=(7,6))
    if Rb.size:
        plt.scatter(Rb, Zb, s=6, alpha=0.6, label='Boundary')

    # surfaces
    for i, S in enumerate(surfaces):
        phi_s = cyl_phi(S)
        mask_s = np.abs(wrap_diff(phi_s, phi0)) <= dphi
        if not np.any(mask_s): continue
        Rs = np.sqrt(S[mask_s,0]**2 + S[mask_s,1]**2)
        Zs = S[mask_s,2]
        plt.scatter(Rs, Zs, s=8, label=f'QFM {i+1}')

    plt.xlabel('R')
    plt.ylabel('Z')
    plt.title(f'R-Z slice at φ={phi0:.3f} ± {dphi:.3f}')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()

# ------------------------------- main ------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mfs_npz', help='Checkpoint produced by main.py (…_solution.npz)')
    ap.add_argument('--n-surfaces', type=int, default=1)
    ap.add_argument('--offset-min', type=float, default=0.25,
                    help='Min inward offset as fraction of median h (normalized)')
    ap.add_argument('--offset-max', type=float, default=1.2,
                    help='Max inward offset as fraction of median h (normalized)')
    ap.add_argument('--k-nn', type=int, default=64, help='k for kNN area weights')
    ap.add_argument('--dt', type=float, default=0.08)
    ap.add_argument('--beta', type=float, default=0.12,
                    help='Graph-Laplacian smoothing strength')
    ap.add_argument('--iters', type=int, default=25)
    ap.add_argument('--tol', type=float, default=1e-4)
    ap.add_argument('--save-prefix', default=None)
    ap.add_argument('--plot', action='store_true', default=True)
    ap.add_argument('--poincare-phi', type=float, default=0,
                    help='If set (radians), plot R-Z points with |phi-phi0| < dphi.')
    ap.add_argument('--poincare-dphi', type=float, default=0.02,
                    help='Half-width tolerance around phi0 (radians).')
    args = ap.parse_args()

    field, P, N_raw, kind = load_mfs_checkpoint(args.mfs_npz)

    # Ensure boundary normals are globally outward and consistent
    N = make_normals_globally_outward(P, N_raw)

    # Inside tester & quick sanity on boundary itself (should be “not inside”)
    is_inside = build_inside_tester(P, N)
    print("[CHK] Boundary inside test on P (should be False for most points):")
    inside_on_P = is_inside(P)
    print(f"      inside count on boundary: {inside_on_P.sum()} / {len(P)}")

    # boundary spacing (median) to pick offsets in *world* units
    Wb = knn_area_weights(P, P, k=args.k_nn)
    h_med = float(np.median(np.sqrt(np.asarray(Wb) / np.pi)))
    # normalized h_med is in boundary's (u,v) metric; convert to world via 1/sc
    h_world = h_med / float(np.asarray(field.sc.scale))
    print(f"[SCALE] median h_world≈{h_world:.6g}. "
        f"offsets range: [{args.offset_min*h_world:.3e}, {args.offset_max*h_world:.3e}] (world units)")
    margin_world = 0.05 * h_world   # <-- stronger safety buffer in world units

    offsets = np.linspace(args.offset_min, args.offset_max, max(1, args.n_surfaces))
    surfaces = []
    for j, tau in enumerate(offsets):
        X0 = P - (tau * h_world) * N   # inward if N is outward
        # DEBUG: how many seeds ended up outside by accident?
        inside0 = is_inside(X0)
        print(f"[SEED] tau={tau:.4g}: inside count {inside0.sum()}/{len(P)} "
            f"(expected all True). If not, we’ll project.")
        if not np.all(inside0):
            # Start with ~10% of the nominal inward offset and allow growth
            X0 = project_inside(P, N, X0,
                                eps=0.10 * tau * h_world,
                                max_iters=8,
                                growth=1.5)
            inside0b = is_inside(X0)
            print(f"[SEED] After projection: inside count {inside0b.sum()}/{len(P)}.")
        print(f"[QFM] Evolving QFMin surface {j+1}/{len(offsets)} with tau={tau}...")
        Xj, Nj = evolve_qfmin_surface(
            jnp.asarray(X0),
            field,
            P_boundary_ref_np=P,
            N_boundary_out_np=N,          # NEW
            is_inside_func=is_inside,     # NEW
            k_knn_weights=args.k_nn,
            k_knn_normals=max(12, args.k_nn // 4),
            dt=args.dt,
            beta=args.beta,
            iters=args.iters,
            tol=args.tol,
            verbose=True,
            margin_world=margin_world,
        )
        surfaces.append(Xj)
        inside_end = is_inside(Xj)
        print(f"[END] tau={tau:.4g}: inside {inside_end.sum()}/{len(inside_end)}; "
            f"min/max distance to boundary (rough NN): ", end="")
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)
        dmin, idx = nn.kneighbors(Xj)
        print(f"{dmin.min():.3e} / {dmin.max():.3e}")

    if args.save_prefix:
        out_npz = f"{args.save_prefix}_qfm.npz"
        np.savez(out_npz,
                 boundary_P=P, boundary_N=N,
                 surfaces=np.array(surfaces, dtype=object), kind=kind)
        print(f"[SAVE] Wrote QFMin bundle → {out_npz}")
        for i, S in enumerate(surfaces):
            np.savetxt(f"{args.save_prefix}_qfm_{i+1}.csv", S, delimiter=",", header="x,y,z", comments="")
        print(f"[SAVE] Wrote {len(surfaces)} QFMin CSVs with prefix {args.save_prefix}_qfm_*.csv")

    if args.plot:
        plot_surfaces(P, surfaces)
        plot_poincare_slice(P, surfaces, args.poincare_phi, args.poincare_dphi)
        plt.show()

if __name__ == '__main__':
    main()
