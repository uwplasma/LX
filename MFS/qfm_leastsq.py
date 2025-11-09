#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QFMin via global least-squares:
    minimize  ⟨(∇φ · ∇ψ)^2⟩ + μ ⟨(Δψ)^2⟩
subject to  ψ|_{boundary}=1,  ψ|_{axis-seed}=0.

Outputs: node cloud (X), scalar ψ(X), and a few ψ=const isosurfaces (projected).
"""

from __future__ import annotations
import argparse, sys, math
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import lsqr, cg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.sparse import spdiags


# -------------------- Load φ from MFS checkpoint -------------------- #

@jax.tree_util.register_pytree_node_class
@dataclass
class ScaleInfo:
    center: jnp.ndarray  # (3,)
    scale: jnp.ndarray   # ()

    def normalize(self, X: jnp.ndarray) -> jnp.ndarray:
        return (X - self.center) * self.scale

    def tree_flatten(self):
        return ((self.center, self.scale), None)
    @classmethod
    def tree_unflatten(cls, aux, children):
        c, s = children
        return cls(c, s)

@jit
def green_G(x, y):
    r = jnp.linalg.norm(x - y, axis=-1)
    return 1.0 / (4.0 * jnp.pi * jnp.maximum(1e-30, r))

@jit
def grad_green_x(x, y):
    r = x - y
    r2 = jnp.sum(r*r, axis=-1)
    r3 = jnp.maximum(1e-30, r2*jnp.sqrt(r2))
    return -r / (4.0*jnp.pi*r3[..., None])

@jit
def grad_azimuth_about_axis(Xn, a_hat):
    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par  = jnp.sum(Xn*a, axis=1, keepdims=True)*a
    r_perp = Xn - r_par
    r2     = jnp.maximum(1e-30, jnp.sum(r_perp*r_perp, axis=1, keepdims=True))
    return jnp.cross(a[None,:], r_perp) / r2

def _pct(a, p): 
    a = np.asarray(a).ravel()
    return float(np.percentile(a, p))

def load_mfs_checkpoint(path: str):
    data = np.load(path, allow_pickle=True)
    center = jnp.asarray(data["center"], dtype=jnp.float64)
    scale  = jnp.asarray(data["scale"],  dtype=jnp.float64)
    Yn     = jnp.asarray(data["Yn"],     dtype=jnp.float64)     # (M,3) normalized sources
    alpha  = jnp.asarray(data["alpha"],  dtype=jnp.float64)     # (M,)
    a      = jnp.asarray(data["a"],      dtype=jnp.float64)     # [a_t, a_p]
    a_hat  = jnp.asarray(data["a_hat"],  dtype=jnp.float64)     # (3,)
    P      = np.asarray(data["P"],       dtype=np.float64)      # boundary points
    N      = np.asarray(data["N"],       dtype=np.float64)      # boundary outward normals
    kind   = str(data["kind"]) if "kind" in data else "torus"
    sc     = ScaleInfo(center=center, scale=scale)
    return sc, Yn, alpha, a, a_hat, P, N, kind

@jit
def _S_alpha_at(xn, Yn, alpha):
    G = vmap(lambda y: green_G(xn, y))(Yn)               # (M,)
    return jnp.dot(G, alpha)                              # ()

@jit
def _grad_S_alpha_at(xn, Yn, alpha):
    Gr = vmap(lambda y: grad_green_x(xn, y))(Yn)         # (M,3)
    return jnp.sum(Gr * alpha[:, None], axis=0)          # (3,)

@jit
def field_grad_total(sc: ScaleInfo, Yn, alpha, a, a_hat, Pn_ref, Nn_ref, X: jnp.ndarray) -> jnp.ndarray:
    """∇φ = a_t ϕ̂_t + a_p θ̂  + ∇ψsingle; θ̂ via nearest boundary normals at X."""
    Xn = sc.normalize(X)
    # single-valued Green part
    dS = vmap(_grad_S_alpha_at, in_axes=(0, None, None))(Xn, Yn, alpha) * sc.scale
    # toroidal part
    Gt = grad_azimuth_about_axis(Xn, a_hat) * sc.scale
    # poloidal direction from boundary normals
    # nearest normal (quick NN on host passed in)
    # here we approximate θ̂ with ϕ̂_t projected to tangent of nearest boundary normal:
    aunit = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par = jnp.sum(Xn*aunit, axis=1, keepdims=True)*aunit
    r_perp = Xn - r_par
    phi_hat = jnp.cross(aunit[None,:], r_perp)
    phi_hat = phi_hat / jnp.maximum(1e-30, jnp.linalg.norm(phi_hat, axis=1, keepdims=True))
    # use boundary normals table (nearest neighbor indices are precomputed outside JIT)
    # caller supplies Nnear(X) as an array
    # we instead require caller to provide Nnear separately to avoid NN search in JIT
    raise NotImplementedError("Use field_grad_at_points() below with supplied nearest boundary normals.")

def field_grad_at_points(sc, Yn, alpha, a, a_hat, X_np: np.ndarray, P_bndry: np.ndarray, N_bndry: np.ndarray):
    """Host helper: ∇φ(X) using nearest boundary normals for the θ̂ piece."""
    # NN for boundary normals
    nbr = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P_bndry)
    _, idx = nbr.kneighbors(X_np)
    idx = idx[:, 0]
    Nnear = N_bndry[idx]                       # (N,3)

    X = jnp.asarray(X_np)
    Xn = sc.normalize(X)
    dS = vmap(_grad_S_alpha_at, in_axes=(0, None, None))(Xn, Yn, alpha) * sc.scale
    aunit = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par = jnp.sum(Xn*aunit, axis=1, keepdims=True)*aunit
    r_perp = Xn - r_par
    phi_hat = jnp.cross(aunit[None,:], r_perp)
    phi_hat = phi_hat / jnp.maximum(1e-30, jnp.linalg.norm(phi_hat, axis=1, keepdims=True))
    # project to tangent plane of boundary at the nearest point
    Nsurf = jnp.asarray(Nnear)
    phi_tan = phi_hat - jnp.sum(phi_hat*Nsurf, axis=1, keepdims=True)*Nsurf
    phi_tan = phi_tan / jnp.maximum(1e-30, jnp.linalg.norm(phi_tan, axis=1, keepdims=True))
    Gt = grad_azimuth_about_axis(Xn, a_hat) * sc.scale
    mv = sc.scale * (a[0]*Gt + a[1]*phi_tan)
    return np.asarray(mv + dS)                 # (N,3)

# -------------------- RBF-FD stencils (kNN, p=1) -------------------- #

def rbf_ph(r, eps):      # polyharmonic spline r^3 (works well)
    return r**3
def rbf_dph_dr(r, eps):  # derivative wrt r
    return 3*r**2

def rbf_fd_weights_grad_and_lap(Xc: np.ndarray, Xn: np.ndarray, eps=1.0):
    m = Xn.shape[0]
    # Pair vectors and distances from center -> neighbors
    V = Xc[None, :] - Xn          # (m,3)  (note Xc - Xj)
    r = np.linalg.norm(V, axis=1) # (m,)
    # RBF block (PHS r^3)
    R = np.linalg.norm(Xn[None,:,:] - Xn[:,None,:], axis=2)  # (m,m)
    A = rbf_ph(R, eps)
    # Polynomial augmentation
    P = np.c_[np.ones(m), Xn - Xc[None,:]]                   # (m,4)
    Z = np.zeros((4,4))
    LHS = np.block([[A, P],
                    [P.T, Z]])                               # (m+4, m+4)

    # ---- RHS for operators applied to basis at center ----
    # Gradient components (use ∇φ_j(Xc) = 3 r_j (Xc - Xj))
    gx_top = 3.0 * r * V[:, 0]                               # (m,)
    gy_top = 3.0 * r * V[:, 1]
    gz_top = 3.0 * r * V[:, 2]
    # Polynomial parts: d/dx,y,z at center
    rhs_gx = np.r_[gx_top, [0.0, 1.0, 0.0, 0.0]]
    rhs_gy = np.r_[gy_top, [0.0, 0.0, 1.0, 0.0]]
    rhs_gz = np.r_[gz_top, [0.0, 0.0, 0.0, 1.0]]

    # Laplacian (Δφ_j(Xc) = 6 r_j ; Δ poly = 0)
    lap_top = 6.0 * r
    rhs_lap = np.r_[lap_top, [0.0, 0.0, 0.0, 0.0]]

    try:
        from numpy.linalg import solve
        wx = solve(LHS, rhs_gx)[:m]
        wy = solve(LHS, rhs_gy)[:m]
        wz = solve(LHS, rhs_gz)[:m]
        wl = solve(LHS, rhs_lap)[:m]
    except np.linalg.LinAlgError:
        LHS_reg = LHS + 1e-12*np.eye(LHS.shape[0])
        wx = np.linalg.solve(LHS_reg, rhs_gx)[:m]
        wy = np.linalg.solve(LHS_reg, rhs_gy)[:m]
        wz = np.linalg.solve(LHS_reg, rhs_gz)[:m]
        wl = np.linalg.solve(LHS_reg, rhs_lap)[:m]

    Wgrad = np.vstack([wx, wy, wz])    # (3,m)
    Wlap  = wl                         # (m,)
    return Wgrad, Wlap

def build_all_stencils(X: np.ndarray, k: int = 30):
    """
    For each node i, build indices of neighbors and gradient/Laplacian weights
    mapping psi[neighbors] -> grad psi at i, and -> Laplacian psi at i.
    Returns lists suitable to assemble sparse rows.
    """
    N = X.shape[0]
    k = min(k, max(8, N-1))
    nbr = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(X)
    _, idxs = nbr.kneighbors(X)                            # (N,k)

    G_wts = []   # list of (i, idxs[i], Wgrad(3,k))
    L_wts = []   # list of (i, idxs[i], Wlap(k,))
    for i in range(N):
        J = idxs[i]
        Wg, Wl = rbf_fd_weights_grad_and_lap(X[i], X[J])
        G_wts.append((i, J, Wg))
        L_wts.append((i, J, Wl))
    return G_wts, L_wts

def _debug_print_geometry_and_stencils(X, P_b, N_b, G_wts, L_wts, title="[DBG:geom]"):
    print(f"{title} N_nodes={len(X)}  N_boundary={len(P_b)}")
    # neighbor sizes
    ks = [len(J) for (_, J, _) in G_wts]
    print(f"{title} stencil k: min={min(ks)} med={np.median(ks):.1f} max={max(ks)}")
    # quick boundary distance stats
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P_b)
    d, idx = nn.kneighbors(X)
    print(f"{title} dist(X, boundary): p1={_pct(d,1):.3e}  p50={_pct(d,50):.3e}  p99={_pct(d,99):.3e}  max={np.max(d):.3e}")

# -------------------- System assembly: min ||A ψ - b|| -------------------- #

def assemble_system(X: np.ndarray,
                    P_b: np.ndarray,
                    N_b: np.ndarray,
                    sc, Yn, alpha, a, a_hat,
                    mu: float = 1e-2,
                    w_boundary: float = 50.0,
                    w_axis: float = 50.0,
                    n_axis_seeds: int = 32,
                    k: int = 30):
    """
    Build sparse A,b for least squares:
        rows:   w_t (∇φ·∇ψ)    at interior nodes
                sqrt(mu) Δψ     at interior nodes
                w_b (ψ-1)       at boundary samples
                w_0 (ψ-0)       at axis seeds
    """
    # Keep per-row scalings so diagnostics can match the system actually solved
    dot_row_scales = []   # length = #interior rows (one per interior node)
    lap_row_scales = []   # length = #interior rows (one per interior node), if mu>0

    # 1) Build stencils on *all* nodes (X includes interior + boundary samples)
    G_wts, L_wts = build_all_stencils(X, k=k)

    # 2) Evaluate ∇φ at nodes
    Gphi = field_grad_at_points(sc, Yn, alpha, a, a_hat, X, P_b, N_b)   # (N,3)

    rows_i, cols_j, vals, rhs = [], [], [], []

    Nn = X.shape[0]
    # flags: mark boundary nodes and interior nodes
    nbr_b = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P_b)
    dmin, _ = nbr_b.kneighbors(X)
    # a node is boundary-sampled if it's *very* close to P_b; we will also explicitly add boundary set later
    boundary_mask = (dmin[:,0] < 1e-10)  # exact if we reused P_b in X
    interior_idx = np.where(~boundary_mask)[0]
    boundary_idx = np.where(boundary_mask)[0]

    print(f"[ASM] interior nodes: {len(interior_idx)} / {Nn}  boundary nodes in X: {len(boundary_idx)}")

    # 3) (∇φ·∇ψ) rows for interior
    for i in interior_idx:
        irow = len(rhs)
        i_center, J, Wg = G_wts[i]
        gx, gy, gz = Wg    # each (k,)
        g = Gphi[i]        # (3,)
        # dot(∇φ, ∇ψ) = g_x*(∑ wx ψ_j) + g_y*(∑ wy ψ_j) + g_z*(∑ wz ψ_j)
        # => coefficients over ψ[J]:  g_x*wx + g_y*wy + g_z*wz
        coeff = g[0]*gx + g[1]*gy + g[2]*gz
        # Row normalization (store the scale used so diagnostics can divide by it)
        rnorm = float(np.linalg.norm(coeff))
        if rnorm < 1e-30:
            rnorm = 1e-30
        dot_row_scales.append(rnorm)   # <—— record
        coeff = coeff / rnorm
        for c, j in zip(coeff, J):
            rows_i.append(irow); cols_j.append(j); vals.append(c)
        rhs.append(0.0)  # 0 stays 0 after scaling

    # 4) sqrt(μ) Δψ rows (interior)
    rt_mu = math.sqrt(max(mu, 0.0))
    if rt_mu > 0:
        for i in interior_idx:
            irow = len(rhs)
            _, J, Wl = L_wts[i]
            row = rt_mu * Wl
            rnorm = float(np.linalg.norm(row))
            if rnorm < 1e-30:
                rnorm = 1e-30
            lap_row_scales.append(rnorm)   # <—— record
            row = row / rnorm
            for c, j in zip(row, J):
                rows_i.append(irow); cols_j.append(j); vals.append(c)
            rhs.append(0.0)

    # 5) Boundary constraints ψ=1 on a subset of boundary samples
    # take unique P_b indices if X stacked [interior; boundary]
    if boundary_idx.size == 0:
        # if the provided X had no exact boundary nodes, append some from P_b
        start = len(X)
        Xb = P_b
        w_boundary_eff = float(w_boundary)  # after normalization, this is an actual strength
        for ii, x in enumerate(Xb):
            irow = len(rhs)
            rows_i.append(irow); cols_j.append(start+ii); vals.append(w_boundary_eff)
            rhs.append(w_boundary_eff*1.0)
        # we must also expand matrix size; easiest is to concatenate at the end
        X = np.vstack([X, Xb])
    else:
        w_boundary_eff = float(w_boundary)
        for i in boundary_idx:
            irow = len(rhs)
            rows_i.append(irow); cols_j.append(i); vals.append(w_boundary_eff)
            rhs.append(w_boundary_eff*1.0)

    # 6) Axis-seed constraints ψ=0 near a local minimum of |∇φ|
    # pick n_axis_seeds nodes with smallest |∇φ|
    gnorm = np.linalg.norm(Gphi, axis=1)
    # pick seeds only among interior nodes
    interior_sorted = interior_idx[np.argsort(gnorm[interior_idx])]
    seeds = interior_sorted[:n_axis_seeds]
    for i in seeds:
        irow = len(rhs)
        rows_i.append(irow); cols_j.append(i); vals.append(w_axis)
        rhs.append(0.0)

    A = coo_matrix((vals, (rows_i, cols_j)), shape=(len(rhs), X.shape[0])).tocsr()
    b = np.asarray(rhs, dtype=float)

    print(f"[ASM] rows={len(rhs)}  cols={X.shape[0]}  nnz≈{len(vals)} "
          f"(dot rows={len(interior_idx)}  lap rows={len(interior_idx) if rt_mu>0 else 0}  "
          f"boundary rows~{(len(boundary_idx) if boundary_idx.size>0 else len(P_b))}  axis rows={n_axis_seeds})")

    return (A, b, X,
            G_wts, L_wts,              # stencils actually used
            Gphi,                      # ∇φ used in assembly
            interior_idx, boundary_idx,
            np.asarray(dot_row_scales, float),
            np.asarray(lap_row_scales, float) if rt_mu > 0 else None)

# -------------------- Iso-surface extraction on nodes -------------------- #

def choose_levels(psi, interior_idx, boundary_idx, seeds, target_quants):
    """
    Pick isosurface levels robustly from the *interior* ψ without clipping,
    then clamp to (ε, 1-ε) using boundary/seeds as anchors.
    """
    # Anchor values (for safety if affine fit ever drifts)
    psi_axis  = float(np.median(psi[seeds])) if len(seeds) else float(np.min(psi))
    psi_bound = float(np.median(psi[boundary_idx])) if len(boundary_idx) else float(np.max(psi))
    if psi_bound == psi_axis:
        psi_axis, psi_bound = 0.0, 1.0

    # Normalize interior ψ to [0,1] using the *empirical* anchors
    psi_int = psi[interior_idx]
    psi01 = (psi_int - psi_axis) / (psi_bound - psi_axis)
    # No clipping here; compute quantiles on the distribution as-is
    raw = np.quantile(psi01, target_quants)

    # Clamp and uniquify
    eps = 1e-3
    raw = np.clip(raw, eps, 1.0 - eps)
    # Remove near-duplicates
    uniq = [raw[0]]
    for r in raw[1:]:
        if abs(r - uniq[-1]) > 2e-3:
            uniq.append(r)
    # Map back to the ψ scale
    levels = [float(psi_axis + u * (psi_bound - psi_axis)) for u in uniq]
    return levels

def rbf_fd_weights_interp_identity(Xc: np.ndarray, Xn: np.ndarray, eps=1.0):
    """
    Weights w such that  sum_j w_j f(Xn_j) ≈ f(Xc)   (PHS r^3 with linear poly aug).
    """
    m = Xn.shape[0]
    R = np.linalg.norm(Xn[None,:,:] - Xn[:,None,:], axis=2)
    A = rbf_ph(R, eps)
    P = np.c_[np.ones(m), Xn - Xc[None, :]]
    Z = np.zeros((4,4))
    LHS = np.block([[A, P],
                    [P.T, Z]])
    # RHS is the basis evaluated at Xc:
    # rbf at Xc relative to centers Xn: r = |Xc - Xn|
    r = np.linalg.norm(Xc[None,:] - Xn, axis=1)
    rhs = np.r_[rbf_ph(r, eps), [1.0, 0.0, 0.0, 0.0]]
    try:
        w = np.linalg.solve(LHS, rhs)[:m]
    except np.linalg.LinAlgError:
        w = np.linalg.solve(LHS + 1e-12*np.eye(LHS.shape[0]), rhs)[:m]
    return w

def _stratified_seeds_by_phi(X, psi, c, nbins=64, per_bin=40):
    """Return ~nbins*per_bin seed indices, balanced across cylindrical φ bins."""
    phi = np.arctan2(X[:,1], X[:,0])
    # Map to [0, 2π)
    phi = (phi + 2*np.pi) % (2*np.pi)
    bins = np.linspace(0.0, 2*np.pi, nbins+1)
    which = np.digitize(phi, bins) - 1
    which = np.clip(which, 0, nbins-1)

    idxs = []
    resid = np.abs(psi - c)
    for b in range(nbins):
        J = np.where(which == b)[0]
        if J.size == 0:
            continue
        take = min(per_bin, J.size)
        # smallest |ψ-c| in this φ bin
        pick = J[np.argpartition(resid[J], take-1)[:take]]
        idxs.append(pick)
    if not idxs:
        return np.argsort(resid)[:nbins*per_bin]
    return np.concatenate(idxs, axis=0)

def newton_project_to_isosurface(X: np.ndarray,
                                 psi: np.ndarray,
                                 levels,
                                 Gpsi_stencils,          # kept for API compatibility (unused)
                                 max_iters=3,
                                 n_seed_per_level=6000,
                                 k_local=40):
    """
    True projection using local RBF-FD with per-iteration kNN rebuilt around the
    *current* points Xc. At each Newton step:
        ψ(Xc) ≈ w_id^T ψ[J],  ∇ψ(Xc) ≈ Wgrad @ ψ[J],
    where J are the k nearest neighbors of Xc in the *global* cloud X.
    Prints convergence diagnostics per level per iteration.
    """
    nodes = []

    for c in levels:
        # pick seeds closest to the level c from the *global* nodes
        # k_local now comes from the function argument
        nbins = max(8, int(np.sqrt(n_seed_per_level)))  # heuristic
        per_bin = max(1, n_seed_per_level // nbins)
        idx0 = _stratified_seeds_by_phi(X, psi, c, nbins=nbins, per_bin=per_bin)
        Xc = X[idx0].copy()

        for it in range(max_iters):
            # Fit a single KD-tree to the global cloud; query at current Xc
            nbr = NearestNeighbors(n_neighbors=k_local, algorithm="kd_tree").fit(X)
            _, idxs = nbr.kneighbors(Xc)  # shape (m, k_local)

            F = np.empty(len(Xc), dtype=float)
            Gp = np.empty((len(Xc), 3), dtype=float)

            # Build local RBF-FD weights at each moved point Xc[i] against its current neighbors X[J]
            for i_loc in range(len(Xc)):
                J = idxs[i_loc]
                XJ = X[J]

                # interpolation weight to evaluate ψ(Xc[i])
                w_id = rbf_fd_weights_interp_identity(Xc[i_loc], XJ)

                # gradient weights at Xc[i]
                Wg_local, _ = rbf_fd_weights_grad_and_lap(Xc[i_loc], XJ)

                psi_xc = float(np.dot(w_id, psi[J]))
                gx = float(np.dot(Wg_local[0], psi[J]))
                gy = float(np.dot(Wg_local[1], psi[J]))
                gz = float(np.dot(Wg_local[2], psi[J]))

                F[i_loc] = psi_xc - c
                Gp[i_loc, :] = (gx, gy, gz)

            g2 = np.sum(Gp * Gp, axis=1) + 1e-16
            Xc -= (F / g2)[:, None] * Gp

            # ---- projector convergence print (per iteration) ----
            print(f"[ISO-NEWTON] level={c:.6f}  iter={it+1}/{max_iters}  "
                  f"max|ψ-c|={np.max(np.abs(F)):.3e}  median|ψ-c|={np.median(np.abs(F)):.3e}")

        nodes.append(Xc)

    return nodes

# -------------------- Driver -------------------- #

def make_interior_cloud(P: np.ndarray, N: np.ndarray,
                        step_h: float = 0.6,
                        n_layers: int = 80,
                        max_depth_factor: float = 8.0):
    """
    March inward along outward normals by depths t = (1..n_layers)*step_h*h,
    up to ~max_depth_factor*h, and keep points that are still inside.
    This fills the *whole* volume instead of a thin shell.
    """
    # Estimate boundary spacing h
    nbr = NearestNeighbors(n_neighbors=8, algorithm="kd_tree").fit(P)
    d, _ = nbr.kneighbors(P)
    h = np.median(d[:, 1])

    depths = (np.arange(1, n_layers + 1) * step_h) * h
    depths = depths[depths <= max_depth_factor * h]
    clouds = []
    for t in depths:
        X = P - t * N
        clouds.append(X)
    Xcand = np.vstack(clouds)

    # Keep only points that truly lie inside (<=0 along outward normal)
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)
    _, idx = nn.kneighbors(Xcand)
    idx = idx[:, 0]
    Pn = P[idx]; Nn = N[idx]
    signed = np.einsum('ij,ij->i', Xcand - Pn, Nn)  # >0 means outside
    Xin = Xcand[signed <= 1e-9]

    # Optional: light thinning to avoid massive duplication
    # (bucket into a grid in xyz or use farthest-point thinning if needed)
    return Xin

def _cyl_phi(X):
    return np.arctan2(X[:, 1], X[:, 0])

def _wrap_diff(a, b):
    d = a - b
    d = (d + np.pi) % (2*np.pi) - np.pi
    return d

def inside_mask_wrt_boundary(X: np.ndarray,
                             P_b: np.ndarray,
                             N_b: np.ndarray,
                             tol: float = 1e-9) -> np.ndarray:
    """
    Returns boolean mask of points X that lie inside the closed surface defined by (P_b, N_b),
    using the sign of (X - P_near)·N_near with nearest boundary point/normals.
    Convention: outward normals in N_b. 'Inside' means dot <= tol (i.e., not outside).
    """
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P_b)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 0]
    Pn = P_b[idx]
    Nn = N_b[idx]
    signed = np.einsum('ij,ij->i', X - Pn, Nn)  # >0 -> outside (along outward normal)
    return signed <= tol

def plot_poincare_slice(boundary_P: np.ndarray,
                        surfaces: list[np.ndarray],
                        phi0: float = 0.0,
                        dphi: float = 0.02,
                        title: str = "Poincaré slice (R–Z)",
                        colors: list = None,
                        ms_surf: float = 3.0,
                        ms_bdry: float = 4.0,
                        alpha_bdry: float = 0.5,
                        alpha_surf: float = 0.9):
    """
    Plot R–Z of points with cylindrical angle φ within [phi0-dphi, phi0+dphi].
    Uses a consistent color per surface across slices.
    """
    if colors is None:
        # deterministic palette with wraparound
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(surfaces))]

    phi_b = _cyl_phi(boundary_P)
    mask_b = np.abs(_wrap_diff(phi_b, phi0)) <= dphi
    Rb = np.sqrt(boundary_P[mask_b, 0]**2 + boundary_P[mask_b, 1]**2)
    Zb = boundary_P[mask_b, 2]
    nb = int(mask_b.sum())
    print(f"[PC] boundary points in slice φ≈{phi0:.3f}: {nb}")

    if Rb.size:
        plt.scatter(Rb, Zb, s=ms_bdry, alpha=alpha_bdry, label='Boundary')

    for i, S in enumerate(surfaces):
        if S is None or len(S) == 0:
            continue
        phi_s = _cyl_phi(S)
        mask_s = np.abs(_wrap_diff(phi_s, phi0)) <= dphi
        ns = int(mask_s.sum())
        print(f"[PC] iso #{i} points in slice φ≈{phi0:.3f}: {ns}")
        if not np.any(mask_s):
            continue
        Rs = np.sqrt(S[mask_s, 0]**2 + S[mask_s, 1]**2)
        Zs = S[mask_s, 2]
        plt.scatter(Rs, Zs, s=ms_surf, alpha=alpha_surf, color=colors[i], label=None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mfs_npz")
    ap.add_argument("--mu", type=float, default=5e-3, help="Laplacian stabilizer weight")
    ap.add_argument("--k", type=int, default=30, help="kNN for RBF-FD stencils")
    ap.add_argument("--axis-seeds", type=int, default=48, help="number of axis anchor points (ψ=0)")
    ap.add_argument("--levels", type=str, default="0.05, 0.15, 0.35, 0.55, 0.75", help="ψ levels to extract as surfaces")
    ap.add_argument("--poincare-nphi", type=int, default=4, help="If set, overrides --poincare-phi and plots Nphi*nfp evenly spaced Poincaré slices.")
    ap.add_argument("--nfp", type=int, default=2, help="Number of field periods; used with --poincare-nphi to tile full 2π.")
    ap.add_argument("--poincare-dphi", type=float, default=0.02, help="Half-width tolerance around φ0 for the slice (radians).")
    ap.add_argument("--plot", action="store_true", default=True, help="show a quick 3D scatter plot")
    ap.add_argument("--proj-seeds", type=int, default=2500, help="Seeds per ψ-level for Newton projection")
    ap.add_argument("--proj-k",     type=int, default=24,   help="Local kNN size for RBF-FD during projection")
    ap.add_argument("--proj-iters", type=int, default=2,    help="Newton iterations for projection")
    ap.add_argument("--lsqr-iter-lim", type=int, default=1500, help="max iterations for lsqr solver")
    args = ap.parse_args()

    args.save_prefix = args.mfs_npz.replace(".npz", "_qfm")
    sc, Yn, alpha, a, a_hat, P, N, kind = load_mfs_checkpoint(args.mfs_npz)

    # build node cloud: interior + boundary
    Xin = make_interior_cloud(P, N, step_h=0.6, n_layers=80, max_depth_factor=8.0)
    X = np.vstack([Xin, P])     # include boundary nodes explicitly (ψ=1 rows will hit exactly these)

    # ---- Pre-assembly ∇φ diagnostics ----
    Gphi_probe = field_grad_at_points(sc, Yn, alpha, a, a_hat, X, P, N)
    gnorm = np.linalg.norm(Gphi_probe, axis=1)
    imin = int(np.argmin(gnorm))
    print(f"[GRADφ] |∇φ| stats: min={gnorm.min():.3e} (at idx {imin})  "
          f"p1={_pct(gnorm,1):.3e} p25={_pct(gnorm,25):.3e} p50={_pct(gnorm,50):.3e} "
          f"p75={_pct(gnorm,75):.3e} p99={_pct(gnorm,99):.3e} max={gnorm.max():.3e}")

    # assemble least-squares system
    (A, b, X, G_wts, L_wts, Gphi_asm,
     interior_idx, boundary_idx,
     dot_scales, lap_scales) = assemble_system(
         X, P, N, sc, Yn, alpha, a, a_hat,
         mu=args.mu, w_boundary=50.0, w_axis=150.0,
         n_axis_seeds=args.axis_seeds, k=args.k)

    # Quick block-norm audit on the assembled (already normalized) rows
    try:
        # Sample ~1000 rows to keep it cheap
        import numpy.random as npr
        m = A.shape[0]
        take = min(1000, m)
        I = npr.default_rng(0).choice(m, size=take, replace=False)
        # row_norms ≈ sqrt(sum_j A[i,j]^2) since rhs already scaled
        row_norms = np.sqrt(A[I].multiply(A[I]).sum(axis=1)).A.ravel()
        print(f"[ASM] row-norms (post-normalization) on sample: "
              f"min={row_norms.min():.3e} med={np.median(row_norms):.3e} max={row_norms.max():.3e}")
    except Exception as _e:
        pass

    # solve
    print(f"[LSQ] Assembled A with shape {A.shape}, nnz={A.nnz}. Solving ...")

    # --- Optional: Jacobi right-preconditioner (column scaling) ---
    # Scale columns so each has unit 2-norm (up to floor), then solve for psi_hat; recover psi = Dinv * psi_hat.
    col_norms = np.sqrt(A.power(2).sum(axis=0)).A.ravel()
    col_norms[col_norms < 1e-12] = 1.0
    Dinv = 1.0 / col_norms

    Dinv_sparse = spdiags(Dinv, 0, A.shape[1], A.shape[1])
    A_scaled = A @ Dinv_sparse      # same as A @ diag(Dinv), but sparse

    sol = lsqr(A_scaled, b, damp=0, atol=1e-12, btol=1e-12, iter_lim=args.lsqr_iter_lim, show=True)
    psi_hat = sol[0]
    psi = Dinv * psi_hat     # undo the right-preconditioning
    print(f"[LSQ] done. iters={sol[2]}  residual norm={sol[3]:.3e}")

    # --- Affine normalization so (axis ≈ 0, boundary ≈ 1) in least squares ---
    targets = []
    design  = []

    # boundary targets
    if boundary_idx.size > 0:
        design.append(np.c_[psi[boundary_idx], np.ones_like(boundary_idx, dtype=float)])
        targets.append(np.ones(boundary_idx.size, float))

    # axis seeds (same seeds used in assemble_system)
    gnorm_all = np.linalg.norm(Gphi_asm, axis=1)
    interior_sorted = interior_idx[np.argsort(gnorm_all[interior_idx])]
    seeds = interior_sorted[:args.axis_seeds]
    design.append(np.c_[psi[seeds], np.ones_like(seeds, dtype=float)])
    targets.append(np.zeros(seeds.size, float))

    Afit = np.vstack(design)
    bfit = np.concatenate(targets)
    alpha, beta = np.linalg.lstsq(Afit, bfit, rcond=None)[0]

    psi = alpha * psi + beta
    print(f"[NORM] ψ <- {alpha:.6g} * ψ + {beta:.6g}")

    # ---- Detailed residual diagnostics (using the exact rows we assembled) ----
    # r_dot_normed[i] corresponds 1:1 to interior_idx[i]
    r_dot_normed = []
    r_dot_raw    = []
    for pos, i in enumerate(interior_idx):
        _, J, Wg = G_wts[i]
        gx = np.dot(Wg[0], psi[J]); gy = np.dot(Wg[1], psi[J]); gz = np.dot(Wg[2], psi[J])
        val = float(np.dot(Gphi_asm[i], [gx, gy, gz]))    # unnormalized row value
        r_dot_raw.append(val)
        r_dot_normed.append(val / dot_scales[pos])        # exactly what LSQR minimized

    r_dot_normed = np.asarray(r_dot_normed)
    r_dot_raw    = np.asarray(r_dot_raw)

    print(f"[RES] (normalized) ∇φ·∇ψ: RMS={np.sqrt(np.mean(r_dot_normed**2)):.3e}  "
          f"Linf={np.max(np.abs(r_dot_normed)):.3e}  "
          f"p95={_pct(np.abs(r_dot_normed),95):.3e} p99={_pct(np.abs(r_dot_normed),99):.3e}")
    print(f"[RES] (raw units)  ∇φ·∇ψ: RMS={np.sqrt(np.mean(r_dot_raw**2)):.3e}  "
          f"Linf={np.max(np.abs(r_dot_raw)):.3e}  "
          f"p95={_pct(np.abs(r_dot_raw),95):.3e} p99={_pct(np.abs(r_dot_raw),99):.3e}")

    # Laplacian residuals (normalized exactly as assembled)
    if lap_scales is not None and lap_scales.size:
        r_lap_normed = []
        r_lap_raw    = []
        for pos, i in enumerate(interior_idx):
            _, J, Wl = L_wts[i]
            val = float(np.dot(Wl, psi[J])) * math.sqrt(max(args.mu, 0.0))  # same as assembly before scaling
            r_lap_raw.append(val)
            r_lap_normed.append(val / lap_scales[pos])

        r_lap_normed = np.asarray(r_lap_normed)
        r_lap_raw    = np.asarray(r_lap_raw)

        print(f"[RES] (normalized) sqrt(mu)Δψ: RMS={np.sqrt(np.mean(r_lap_normed**2)):.3e}  "
              f"Linf={np.max(np.abs(r_lap_normed)):.3e}  "
              f"p95={_pct(np.abs(r_lap_normed),95):.3e} p99={_pct(np.abs(r_lap_normed),99):.3e}")
        print(f"[RES] (raw units)  sqrt(mu)Δψ: RMS={np.sqrt(np.mean(r_lap_raw**2)):.3e}  "
              f"Linf={np.max(np.abs(r_lap_raw)):.3e}  "
              f"p95={_pct(np.abs(r_lap_raw),95):.3e} p99={_pct(np.abs(r_lap_raw),99):.3e}")
    else:
        r_lap_normed = np.array([]); r_lap_raw = np.array([])

    combined_norm = np.sqrt(
        len(interior_idx) * (np.sqrt(np.mean(r_dot_normed**2)))**2 +
        len(interior_idx) * (np.sqrt(np.mean(r_lap_normed**2)))**2
    )
    print(f"[RES] (normalized) combined 2-norm ≈ {combined_norm:.3e}  (compare to LSQR r1norm)")

    # 6.3 Boundary constraint ψ≈1
    if boundary_idx.size > 0:
        rb = psi[boundary_idx] - 1.0
        print(f"[RES] boundary ψ-1: RMS={np.sqrt(np.mean(rb*rb)):.3e}  Linf={np.max(np.abs(rb)):.3e}  "
              f"p99={_pct(np.abs(rb),99):.3e}")
    else:
        print("[RES] boundary ψ-1: enforced via augmented nodes; see A rows summary above.")

    # 6.4 Axis seeds ψ≈0 (chosen as min |∇φ|)
    # recompute the same selection used in assemble_system:
    gnorm_all = np.linalg.norm(Gphi_asm, axis=1)
    interior_sorted = interior_idx[np.argsort(gnorm_all[interior_idx])]
    seeds = interior_sorted[:args.axis_seeds]
    r0 = psi[seeds] - 0.0
    print(f"[RES] axis ψ-0: RMS={np.sqrt(np.mean(r0*r0)):.3e}  Linf={np.max(np.abs(r0)):.3e}  "
          f"min|∇φ| among seeds={gnorm_all[seeds].min():.3e}, med={np.median(gnorm_all[seeds]):.3e}, max={gnorm_all[seeds].max():.3e}")

    # 6.5 Scalar-range & percentile summary (useful for sanity)
    print(f"[ψ] range: min={psi.min():.6f}, p1={_pct(psi,1):.6f}, p25={_pct(psi,25):.6f}, "
          f"med={_pct(psi,50):.6f}, p75={_pct(psi,75):.6f}, p99={_pct(psi,99):.6f}, max={psi.max():.6f}")

    _debug_print_geometry_and_stencils(X, P, N, G_wts, L_wts)

    # extract surfaces on given levels
    qs = [float(s) for s in args.levels.split(",")]  # e.g. 0.05,0.15,...
    levels = choose_levels(psi, interior_idx, boundary_idx, seeds, qs)
    print(f"[LVL] robust levels (quantiles {qs}) → {np.array(levels)}")

    surfaces = newton_project_to_isosurface(
        X, psi, levels, (G_wts, L_wts),
        max_iters=args.proj_iters,
        n_seed_per_level=args.proj_seeds,
        k_local=args.proj_k
    )

    # --- define per-surface colors once (used by 3D and Poincaré) ---
    cmap = plt.get_cmap('tab10')
    surf_colors = [cmap(i % 10) for i in range(len(surfaces))]

    # Filter iso nodes to the interior and report fractions
    filtered_surfaces = []
    for c, S in zip(levels, surfaces):
        if S is None or len(S) == 0:
            print(f"[ISO] ψ={c:.3f}: points=0")
            filtered_surfaces.append(S)
            continue
        mask_in = inside_mask_wrt_boundary(S, P, N, tol=1e-9)
        nin = int(np.count_nonzero(mask_in))
        nout = int(len(S) - nin)
        frac_in = nin / max(1, len(S))
        if nout > 0:
            print(f"[ISO] ψ={c:.3f}: points={len(S)}  inside={nin} ({100*frac_in:.1f}%)  outside={nout}  [clipped]")
        else:
            print(f"[ISO] ψ={c:.3f}: points={len(S)}  inside=all")
        filtered_surfaces.append(S[mask_in])
    surfaces = filtered_surfaces

    # save
    out = f"{args.save_prefix}_bundle.npz"
    np.savez(out, X=X, psi=psi, boundary_P=P, boundary_N=N, levels=np.array(levels, float),
             surfaces=np.array(surfaces, dtype=object), kind=kind)
    print(f"[SAVE] Wrote {out}")
    for i, S in enumerate(surfaces, 1):
        np.savetxt(f"{args.save_prefix}_psi{levels[i-1]:.3f}.csv", S, delimiter=",", header="x,y,z", comments="")
    print(f"[SAVE] Wrote {len(surfaces)} CSV point clouds for ψ-levels.")

    if args.plot:
        try:
            fig = plt.figure(figsize=(9,7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(P[:,0], P[:,1], P[:,2], s=3, alpha=0.25, label="Boundary")
            sca = ax.scatter(X[:,0], X[:,1], X[:,2], c=psi, s=3, cmap="viridis", alpha=0.6)
            for i, (S, c) in enumerate(zip(surfaces, levels)):
                ax.scatter(S[:,0], S[:,1], S[:,2], s=6, color=surf_colors[i], label=f"ψ={c}")
            fig.colorbar(sca, ax=ax, shrink=0.7, label="ψ")
            ax.legend(); ax.set_title("QFMin via least-squares ψ")
            plt.tight_layout()
            
            # --- Single or multiple Poincaré slices ---
            phis = np.linspace(0.0, 2*np.pi/args.nfp, args.poincare_nphi, endpoint=False).tolist()
            print(f"[PLOT] Multi-Poincaré: nphi={args.poincare_nphi}, nfp={args.nfp} -> {len(phis)} slices")
            plt.figure(figsize=(7, 6))
            for kphi, phi0 in enumerate(phis):
                plot_poincare_slice(
                    P, surfaces, phi0=float(phi0), dphi=args.poincare_dphi,
                    colors=surf_colors, ms_surf=2.0, ms_bdry=3.0, alpha_bdry=0.4, alpha_surf=0.9
                )
            plt.xlabel('R');plt.ylabel('Z');plt.axis('equal');plt.legend();plt.tight_layout()

            fig2, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
            # Left panel: histograms
            axL.hist(np.abs(r_dot_normed), bins=60, density=True, alpha=0.7, label=r'$|\nabla\phi\cdot\nabla\psi|$')
            axL.hist(np.abs(r_lap_normed), bins=60, density=True, alpha=0.6, label=r'$\sqrt{\mu}\,|\Delta\psi|$')
            axL.set_xlabel('Residual magnitude');axL.set_ylabel('PDF (normalized)');
            axL.set_title('Residual distributions');axL.set_yscale('log');axL.legend()
            # Right panel: cumulative RMS vs quantile
            def cum_rms(x):
                x = np.sort(x**2)
                cs = np.cumsum(x) / max(1, len(x))
                return np.sqrt(cs)
            q = np.linspace(0, 1, 200, endpoint=True)
            axR.plot(np.linspace(0, 1, len(r_dot_normed), endpoint=True), cum_rms(np.abs(r_dot_normed)), label=r'$|\nabla\phi\cdot\nabla\psi|$')
            print(f"cum_rms dot", cum_rms(np.abs(r_dot_normed)))
            axR.plot(np.linspace(0, 1, len(r_lap_normed), endpoint=True), cum_rms(np.abs(r_lap_normed)), label=r'$\sqrt{\mu}\,|\Delta\psi|$')
            print(f"cum_rms lap", cum_rms(np.abs(r_lap_normed)))
            axR.set_yscale('log');axR.set_xlabel('Quantile');axR.set_ylabel('Cumulative RMS');
            axR.set_title('Convergence proxy (lower = better)');axR.legend()
            fig2.tight_layout()

            plt.show()
        except Exception as e:
            print("[PLOT] skipped:", e)

if __name__ == "__main__":
    main()
