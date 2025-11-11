#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global field-aligned flux function ψ via anisotropic diffusion:
    div( D(x) ∇ψ ) = 0,  D = P_perp + eps * P_par,  t = grad φ / |grad φ|
BCs: ψ=1 on Γ (thin boundary band), ψ=0 on axis band (detected by short gradient-flow collapse).
This makes ψ ~ constant along grad φ while diffusing across it → nested level sets.

Refs (theory & numerics):
- Weickert, "Anisotropic Diffusion in Image Processing" (Teubner, 1998); CE diffusion (1999). 
- Field-aligned diffusion solvers in plasma (FCI/field-line-map & anisotropic diffusion stability).
"""

from __future__ import annotations
import argparse, time, sys, math, os
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap, jacrev
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from scipy.sparse.linalg import spilu, LinearOperator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------- Debug utils ---------------------------- #
def pct(a, p): return float(np.percentile(np.asarray(a), p))
def pinfo(msg): print(f"[INFO] {msg}")
def pstat(msg, v):
    v=np.asarray(v); print(f"[STAT] {msg}: min={v.min():.3e} med={np.median(v):.3e} max={v.max():.3e} L2={np.linalg.norm(v):.3e}")

# ------------------- Green's function & gradient (JAX) ------------------- #
@jit
def green_G(x, y):  # 1/(4π r)
    r = jnp.linalg.norm(x - y, axis=-1)
    return 1.0/(4.0*jnp.pi*jnp.maximum(1e-30, r))

@jit
def grad_green_x(x, y):  # ∇_x G = -(x-y)/(4π r^3)
    r = x - y
    r2 = jnp.sum(r*r, axis=-1)
    r3 = jnp.maximum(1e-30, r2*jnp.sqrt(r2))
    return -r/(4.0*jnp.pi*r3[...,None])

def fix_matplotlib_3d(ax):
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1]-x_limits[0]); x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1]-y_limits[0]); y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1]-z_limits[0]); z_mid = np.mean(z_limits)
    R = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid-R, x_mid+R]); ax.set_ylim3d([y_mid-R, y_mid+R]); ax.set_zlim3d([z_mid-R, z_mid+R])

# --------------------- Multivalued gradient pieces (JAX) ------------------ #
@jit
def grad_azimuth_about_axis(Xn, a_hat):
    a = a_hat/jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par = jnp.sum(Xn*a[None,:], axis=1, keepdims=True)*a[None,:]
    r_perp = Xn - r_par
    r2 = jnp.maximum(1e-30, jnp.sum(r_perp*r_perp, axis=1, keepdims=True))
    return jnp.cross(a[None,:], r_perp)/r2

def make_mv_grads(a_vec, a_hat, sc_center, sc_scale):
    a_vec = jnp.asarray(a_vec); a_hat=jnp.asarray(a_hat)
    sc_center=jnp.asarray(sc_center); sc_scale=jnp.asarray(sc_scale)
    @jit
    def grad_t(Xn): return grad_azimuth_about_axis(Xn, a_hat)
    @jit
    def grad_p(Xn):  # simple θ-hat surrogate; robust enough for band near Γ
        n = jnp.zeros_like(Xn);  # fallback (we won't use θ-hat heavily inside)
        return jnp.zeros_like(Xn)
    def grad_mv_world(X):
        Xn = (X - sc_center)*sc_scale
        return sc_scale*(a_vec[0]*grad_t(Xn) + a_vec[1]*grad_p(Xn))
    return grad_mv_world

# ----------------------- Rebuild evaluators from NPZ ---------------------- #
@dataclass
class Evaluators:
    center: jnp.ndarray
    scale: float
    Yn: jnp.ndarray
    alpha: jnp.ndarray
    a: jnp.ndarray
    a_hat: jnp.ndarray

    def build(self):
        sc_c   = jnp.asarray(self.center)
        sc_s   = jnp.asarray(self.scale)
        Yn_c   = jnp.asarray(self.Yn)
        alpha_c= jnp.asarray(self.alpha)
        a_c    = jnp.asarray(self.a)
        a_hatc = jnp.asarray(self.a_hat)

        # --- closures depend only on arrays captured above (no `self` arg) ---

        @jit
        def S_batch(Xn):
            # returns Σ_j α_j G(Xn, Y_j) for each Xn row
            def S_at(xn):
                Gv = vmap(lambda y: green_G(xn, y))(Yn_c)
                return jnp.dot(Gv, alpha_c)
            return vmap(S_at)(Xn)

        @jit
        def dS_batch(Xn):
            # returns ∑_j α_j ∇_x G(Xn, Y_j) for each Xn row
            def dS_at(xn):
                Gg = vmap(lambda y: grad_green_x(xn, y))(Yn_c)  # (M,3)
                return jnp.sum(Gg * alpha_c[:, None], axis=0)    # (3,)
            return vmap(dS_at)(Xn)

        # your multivalued gradient builder (uses center/scale internally)
        grad_mv = make_mv_grads(a_c, a_hatc, sc_c, sc_s)

        @jit
        def phi_fn(X):
            Xn = (X - sc_c) * sc_s
            return S_batch(Xn)

        @jit
        def grad_phi_fn(X):
            Xn = (X - sc_c) * sc_s
            return grad_mv(X) + sc_s * dS_batch(Xn)

        return phi_fn, grad_phi_fn

# ----------------------- Geometry: inside mask & bands -------------------- #
def inside_mask_from_surface(P_surf, N_surf, Xq):
    """
    Inside if (x - p_nn)·n_nn < 0 (outward normals).
    Return: inside(bool), idx(int), signed distance s (float)
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P_surf)
    d, idx = nbrs.kneighbors(Xq)
    p = P_surf[idx[:,0], :]
    n = N_surf[idx[:,0], :]
    s = np.sum((Xq - p)*n, axis=1)  # >0 outside
    return (s < 0.0), idx[:,0], s


# ---------------------- Axis seeds by gradient collapse ------------------- #
def collapse_to_axis(grad_phi, X0, step=0.02, iters=400, tol=1e-6):
    X = jnp.asarray(X0, dtype=jnp.float64)
    @jit
    def one_step(X):
        g = grad_phi(X[None, :])[0]
        n = jnp.linalg.norm(g) + 1e-30
        return X + step * (g / n)
    # do the fixed-iteration loop in Python; it's fine since one_step is jitted
    for _ in range(iters):
        X_new = one_step(X)
        if float(jnp.linalg.norm(X_new - X)) < tol:
            break
        X = X_new
    return np.array(X)

def axis_band_mask(P_axis, Xq, rad):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P_axis)
    d, _ = nbrs.kneighbors(Xq)
    return (d[:,0] < rad)

# ------------------------- Diffusion tensor and stencil ------------------- #
def diffusion_tensor(gradphi, eps):
    # gradphi: (N,3)
    n = np.linalg.norm(gradphi, axis=-1, keepdims=True)        # (N,1)
    # t = gradphi / n when n>thr, else 0; this makes D=I in low-|∇φ| regions
    t = np.divide(gradphi, n, out=np.zeros_like(gradphi), where=n > 1e-10)
    tt = t[..., :, None] * t[..., None, :]                     # (N,3,3)
    I  = np.eye(3)[None, :, :]                                 # (1,3,3) -> broadcast
    D  = (I - tt) + eps * tt
    return D

def build_sparse_operator(nx, ny, nz, dx, dy, dz, inside, Dfield):
    """
    27-point finite-volume stencil for ∇·(D ∇ψ) on a uniform grid.
    Dfield[k,j,i] is the 3x3 tensor at cell centers (we store as Dfield[p,3,3]).
    We assemble using:
      - face-centered <D_ii> by averaging adjacent cell tensors,
      - edge/quad-centered <D_ij> (i≠j) by averaging the 4 adjacent cell tensors,
    and the standard cross-derivative 4-point stencil per mixed term.

    Notes:
      * We only couple nodes where 'inside' is True. Outside remains identity rows (Dirichlet stamped later).
      * Result is symmetric (we add both (p,q) and (q,p) contributions).
    """
    def idx3(i, j, k): return (k*ny + j)*nx + i

    N = nx * ny * nz
    rows, cols, vals = [], [], []

    # convenience: reshape flat arrays to 3D
    inside_3 = inside.reshape(nx, ny, nz)
    D3 = Dfield.reshape(nx, ny, nz, 3, 3)

    hx, hy, hz = dx, dy, dz
    hx2, hy2, hz2 = dx*dx, dy*dy, dz*dz
    inv_hx2, inv_hy2, inv_hz2 = 1.0/hx2, 1.0/hy2, 1.0/hz2
    inv_hxhy = 1.0/(hx*hy)
    inv_hxhz = 1.0/(hx*hz)
    inv_hyhz = 1.0/(hy*hz)

    # helper to push a symmetric pair
    def add(p, q, w):
        rows.append(p); cols.append(q); vals.append(w)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = idx3(i, j, k)
                if not inside_3[i, j, k]:
                    # identity row (will be Dirichlet-stamped anyway)
                    rows.append(p); cols.append(p); vals.append(1.0)
                    continue

                # ---- gather local tensors around (i,j,k) ----
                Dc = D3[i, j, k]  # 3x3 at center

                # Face-centered (for principal parts): average adjacent cells
                def D_face_x(iL, j0, k0, iR, j1, k1):
                    return 0.5 * (D3[iL, j0, k0] + D3[iR, j1, k1])
                def D_face_y(i0, jL, k0, i1, jR, k1):
                    return 0.5 * (D3[i0, jL, k0] + D3[i1, jR, k1])
                def D_face_z(i0, j0, kL, i1, j1, kR):
                    return 0.5 * (D3[i0, j0, kL] + D3[i1, j1, kR])

                # Edge/quad-centered (for mixed terms): average of 4 neighboring cells
                def D_quad(i0, j0, k0, i1, j1, k1, i2, j2, k2, i3, j3, k3):
                    return 0.25*(D3[i0,j0,k0] + D3[i1,j1,k1] + D3[i2,j2,k2] + D3[i3,j3,k3])

                diag = 0.0

                # =========================
                # (1) Principal parts ∂x(Dxx ∂x u), ∂y(Dyy ∂y u), ∂z(Dzz ∂z u)
                # =========================
                # x-direction faces
                if i > 0 and inside_3[i-1, j, k]:
                    Dxm = D_face_x(i-1, j, k, i, j, k)
                    w = Dxm[0, 0] * inv_hx2
                    add(p, idx3(i-1, j, k), -w)
                    diag += w
                if i < nx-1 and inside_3[i+1, j, k]:
                    Dxp = D_face_x(i, j, k, i+1, j, k)
                    w = Dxp[0, 0] * inv_hx2
                    add(p, idx3(i+1, j, k), -w)
                    diag += w

                # y-direction faces
                if j > 0 and inside_3[i, j-1, k]:
                    Dym = D_face_y(i, j-1, k, i, j, k)
                    w = Dym[1, 1] * inv_hy2
                    add(p, idx3(i, j-1, k), -w)
                    diag += w
                if j < ny-1 and inside_3[i, j+1, k]:
                    Dyp = D_face_y(i, j, k, i, j+1, k)
                    w = Dyp[1, 1] * inv_hy2
                    add(p, idx3(i, j+1, k), -w)
                    diag += w

                # z-direction faces
                if k > 0 and inside_3[i, j, k-1]:
                    Dzm = D_face_z(i, j, k-1, i, j, k)
                    w = Dzm[2, 2] * inv_hz2
                    add(p, idx3(i, j, k-1), -w)
                    diag += w
                if k < nz-1 and inside_3[i, j, k+1]:
                    Dzp = D_face_z(i, j, k, i, j, k+1)
                    w = Dzp[2, 2] * inv_hz2
                    add(p, idx3(i, j, k+1), -w)
                    diag += w

                # =========================
                # (2) Mixed terms for ∂x(Dxy ∂y u) + ∂y(Dxy ∂x u)
                #     (xy-plane edges/corners)
                # =========================
                def add_xy_block(sign_i, sign_j, coeff):
                    ii = i + sign_i
                    jj = j + sign_j
                    if 0 <= ii < nx and 0 <= jj < ny and inside_3[ii, jj, k]:
                        add(p, idx3(ii, jj, k), coeff)

                if 0 < i < nx-1 and 0 < j < ny-1:
                    # four quads around (i±1/2, j±1/2, k)
                    Dpp = D_quad(i, j, k, i+1, j, k, i, j+1, k, i+1, j+1, k)   # +x,+y
                    Dpm = D_quad(i, j, k, i+1, j, k, i, j-1, k, i+1, j-1, k)   # +x,-y
                    Dmp = D_quad(i, j, k, i-1, j, k, i, j+1, k, i-1, j+1, k)   # -x,+y
                    Dmm = D_quad(i, j, k, i-1, j, k, i, j-1, k, i-1, j-1, k)   # -x,-y

                    # Coeffs from the standard 4-point mixed derivative stencil:
                    c_pp =  + Dpp[0,1] * ( +1.0 ) * 0.25 * ( 4.0 * inv_hxhy )
                    c_pm =  + Dpm[0,1] * ( -1.0 ) * 0.25 * ( 4.0 * inv_hxhy )
                    c_mp =  + Dmp[0,1] * ( -1.0 ) * 0.25 * ( 4.0 * inv_hxhy )
                    c_mm =  + Dmm[0,1] * ( +1.0 ) * 0.25 * ( 4.0 * inv_hxhy )

                    # corners (i±1, j±1, k)
                    add_xy_block(+1, +1,  c_pp)
                    add_xy_block(+1, -1,  c_pm)
                    add_xy_block(-1, +1,  c_mp)
                    add_xy_block(-1, -1,  c_mm)

                    # center couplings of the mixed term add negative mass at face nodes:
                    # they come in implicitly when you expand (u_{i±1,j±1} - u_{i±1,j} - u_{i,j±1} + u_{i,j})
                    # We account for the -u_{i±1,j} and -u_{i,j±1} with these edge weights:
                    def add_edge(i_off, j_off, coeff_sum):
                        ii = i + i_off; jj = j + j_off
                        if 0 <= ii < nx and 0 <= jj < ny and inside_3[ii, jj, k]:
                            add(p, idx3(ii, jj, k), coeff_sum)

                    # coefficients on (i±1,j, k) and (i, j±1, k)
                    c_i_p = -(c_pp + c_pm)  # sum over +y and -y
                    c_i_m = -(c_mp + c_mm)  # sum over +y and -y with -x
                    c_j_p = -(c_pp + c_mp)  # sum over +x and -x
                    c_j_m = -(c_pm + c_mm)

                    add_edge(+1,  0, c_i_p)
                    add_edge(-1,  0, c_i_m)
                    add_edge( 0, +1, c_j_p)
                    add_edge( 0, -1, c_j_m)

                    # center contributes the remaining (−) mass of mixed terms:
                    diag += - (c_i_p + c_i_m + c_j_p + c_j_m + c_pp + c_pm + c_mp + c_mm)

                # =========================
                # (3) Mixed terms for xz-plane: ∂x(Dxz ∂z u)+∂z(Dxz ∂x u)
                # =========================
                def add_xz_block(sign_i, sign_k, coeff):
                    ii = i + sign_i
                    kk = k + sign_k
                    if 0 <= ii < nx and 0 <= kk < nz and inside_3[ii, j, kk]:
                        add(p, idx3(ii, j, kk), coeff)

                if 0 < i < nx-1 and 0 < k < nz-1:
                    Dpp = D_quad(i, j, k, i+1, j, k, i, j, k+1, i+1, j, k+1)
                    Dpm = D_quad(i, j, k, i+1, j, k, i, j, k-1, i+1, j, k-1)
                    Dmp = D_quad(i, j, k, i-1, j, k, i, j, k+1, i-1, j, k+1)
                    Dmm = D_quad(i, j, k, i-1, j, k, i, j, k-1, i-1, j, k-1)

                    c_pp = + Dpp[0,2] * (+1.0) * 0.25 * (4.0 * inv_hxhz)
                    c_pm = + Dpm[0,2] * (-1.0) * 0.25 * (4.0 * inv_hxhz)
                    c_mp = + Dmp[0,2] * (-1.0) * 0.25 * (4.0 * inv_hxhz)
                    c_mm = + Dmm[0,2] * (+1.0) * 0.25 * (4.0 * inv_hxhz)

                    add_xz_block(+1, +1,  c_pp)
                    add_xz_block(+1, -1,  c_pm)
                    add_xz_block(-1, +1,  c_mp)
                    add_xz_block(-1, -1,  c_mm)

                    c_i_p = -(c_pp + c_pm)
                    c_i_m = -(c_mp + c_mm)
                    c_k_p = -(c_pp + c_mp)
                    c_k_m = -(c_pm + c_mm)

                    # edges (i±1, j, k) and (i, j, k±1)
                    if 0 < i < nx-1:
                        if inside_3[i+1, j, k]: add(p, idx3(i+1, j, k), c_i_p)
                        if inside_3[i-1, j, k]: add(p, idx3(i-1, j, k), c_i_m)
                    if 0 < k < nz-1:
                        if inside_3[i, j, k+1]: add(p, idx3(i, j, k+1), c_k_p)
                        if inside_3[i, j, k-1]: add(p, idx3(i, j, k-1), c_k_m)

                    diag += - (c_i_p + c_i_m + c_k_p + c_k_m + c_pp + c_pm + c_mp + c_mm)

                # =========================
                # (4) Mixed terms for yz-plane: ∂y(Dyz ∂z u)+∂z(Dyz ∂y u)
                # =========================
                def add_yz_block(sign_j, sign_k, coeff):
                    jj = j + sign_j
                    kk = k + sign_k
                    if 0 <= jj < ny and 0 <= kk < nz and inside_3[i, jj, kk]:
                        add(p, idx3(i, jj, kk), coeff)

                if 0 < j < ny-1 and 0 < k < nz-1:
                    Dpp = D_quad(i, j, k, i, j+1, k, i, j, k+1, i, j+1, k+1)
                    Dpm = D_quad(i, j, k, i, j+1, k, i, j, k-1, i, j+1, k-1)
                    Dmp = D_quad(i, j, k, i, j-1, k, i, j, k+1, i, j-1, k+1)
                    Dmm = D_quad(i, j, k, i, j-1, k, i, j, k-1, i, j-1, k-1)

                    c_pp = + Dpp[1,2] * (+1.0) * 0.25 * (4.0 * inv_hyhz)
                    c_pm = + Dpm[1,2] * (-1.0) * 0.25 * (4.0 * inv_hyhz)
                    c_mp = + Dmp[1,2] * (-1.0) * 0.25 * (4.0 * inv_hyhz)
                    c_mm = + Dmm[1,2] * (+1.0) * 0.25 * (4.0 * inv_hyhz)

                    add_yz_block(+1, +1,  c_pp)
                    add_yz_block(+1, -1,  c_pm)
                    add_yz_block(-1, +1,  c_mp)
                    add_yz_block(-1, -1,  c_mm)

                    c_j_p = -(c_pp + c_mp)
                    c_j_m = -(c_pm + c_mm)
                    c_k_p = -(c_pp + c_pm)
                    c_k_m = -(c_mp + c_mm)

                    if 0 < j < ny-1:
                        if inside_3[i, j+1, k]: add(p, idx3(i, j+1, k), c_j_p)
                        if inside_3[i, j-1, k]: add(p, idx3(i, j-1, k), c_j_m)
                    if 0 < k < nz-1:
                        if inside_3[i, j, k+1]: add(p, idx3(i, j, k+1), c_k_p)
                        if inside_3[i, j, k-1]: add(p, idx3(i, j, k-1), c_k_m)

                    diag += - (c_j_p + c_j_m + c_k_p + c_k_m + c_pp + c_pm + c_mp + c_mm)

                # finally center diagonal
                rows.append(p); cols.append(p); vals.append(diag if diag > 0 else 1.0)

    L = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    rhs = np.zeros(N, dtype=float)
    return L, rhs

def march_iso_surface(psi_n, inside, nx, ny, nz, mins, maxs, level):
    vol = psi_n.reshape(nx, ny, nz).copy()
    vol[~inside.reshape(nx,ny,nz)] = np.nan
    vol = np.nan_to_num(vol, nan=-1.0)
    verts, faces, norm, val = marching_cubes(vol, level=level)
    # index -> physical coords
    sx = (maxs[0]-mins[0])/(nx-1); sy = (maxs[1]-mins[1])/(ny-1); sz = (maxs[2]-mins[2])/(nz-1)
    vx = mins[0] + verts[:,0]*sx
    vy = mins[1] + verts[:,1]*sy
    vz = mins[2] + verts[:,2]*sz
    V = np.column_stack([vx, vy, vz])
    return V, faces

def intersect_iso_with_phi_plane(verts, faces, phi0, eps=1e-12):
    # plane: nφ · X = 0 with nφ=(sinφ0, -cosφ0, 0)
    n = np.array([np.sin(phi0), -np.cos(phi0), 0.0])
    d = 0.0
    segs = []
    for tri in faces:
        P = verts[tri]                # (3,3)
        s = P @ n - d                 # signed distances to plane
        # collect edges that cross the plane (opposite signs)
        pairs = [(0,1), (1,2), (2,0)]
        pts = []
        for i,j in pairs:
            si, sj = s[i], s[j]
            if si*sj < 0.0:           # strict cross
                t = si / (si - sj)    # in (0,1)
                pts.append(P[i] + t*(P[j] - P[i]))
            elif abs(si) < eps and abs(sj) < eps:
                # rare: triangle lies in plane → skip to avoid long degenerate polylines
                pts = []
                break
        if len(pts) == 2:
            segs.append(np.vstack(pts))  # 2x3
    return segs  # list of 2x3 segments

def draw_poincare_from_segments(ax, segs):
    if not segs:
        return 0
    cnt = 0
    for S in segs:
        x, y, z = S[:,0], S[:,1], S[:,2]
        R = np.sqrt(x*x + y*y)
        ax.plot(R, z, lw=1.0)
        cnt += 1
    return cnt


# ------------------------------- Main flow ------------------------------- #
def main(npz_file, grid_N=96, eps=1e-3, band_h=1.5, axis_seed_count=64, axis_band_radius=0.02,
         cg_tol=1e-8, cg_maxit=2000, verbose=True, plot=True, psi0=0.3):

    pinfo(f"Loading MFS checkpoint: {npz_file}")
    dat = np.load(npz_file, allow_pickle=True)
    center = dat["center"]; scale = float(dat["scale"])
    Yn = dat["Yn"]; alpha = dat["alpha"]; a = dat["a"]; a_hat = dat["a_hat"]
    P = dat["P"]; N = dat["N"]; kind = str(dat["kind"])
    ev = Evaluators(center=jnp.asarray(center), scale=scale,
                    Yn=jnp.asarray(Yn), alpha=jnp.asarray(alpha),
                    a=jnp.asarray(a), a_hat=jnp.asarray(a_hat))
    phi_fn, grad_phi = ev.build()
    pinfo(f"NPZ loaded. N_surf={P.shape[0]}, N_sources={Yn.shape[0]}, kind={kind}")

    # Grid bounding box from surface extents (+ margin)
    mins = P.min(axis=0); maxs = P.max(axis=0); span = maxs - mins
    mins -= 0.05*span; maxs += 0.05*span
    nx = ny = nz = int(grid_N)
    xs = np.linspace(mins[0], maxs[0], nx)
    ys = np.linspace(mins[1], maxs[1], ny)
    zs = np.linspace(mins[2], maxs[2], nz)
    dx, dy, dz = xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]
    # build grid with consistent in-memory layout for idx3
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")      # (ny, nx, nz)
    XX = XX.transpose(1,0,2)  # -> (nx,ny,nz)
    YY = YY.transpose(1,0,2)
    ZZ = ZZ.transpose(1,0,2)

    # build Xq in the SAME order the operator expects: i fastest, then j, then k
    Xq = np.column_stack([XX.ravel(order="C"),
                        YY.ravel(order="C"),
                        ZZ.ravel(order="C")])
    pinfo(f"Grid: {nx}x{ny}x{nz} ~ {Xq.shape[0]} nodes; spacing dx≈{dx:.3g},dy≈{dy:.3g},dz≈{dz:.3g}")

    voxel = min(dx, dy, dz)

    # Inside & boundary bands
    # After loading Xq and computing inside, also get 's'
    inside, nn_idx, s_signed = inside_mask_from_surface(P, N, Xq)
    pstat("Inside mask", inside.astype(float))
    # Estimate surface spacing from P (median NN distance on the surface)
    nbrs_surf = NearestNeighbors(n_neighbors=4, algorithm="kd_tree").fit(P)
    d_surf, _ = nbrs_surf.kneighbors(P)            # (Ns, 4)
    surf_h = float(np.median(d_surf[:,1]))         # use first non-zero neighbor
    # Start from user/voxel suggestion
    h_band = float(band_h) * voxel
    band = (np.abs(s_signed) <= h_band) & inside

    # Adapt to target fraction of inside (aim ~10%, tighten if >15%)
    
    target_frac = 0.10
    max_frac    = 0.15
    n_in = int(inside.sum())
    if n_in > 0:
        abs_s_inside = np.abs(s_signed[inside])
        # If current band too fat, shrink by quantile down to >= 0.5 voxel
        for _ in range(3):
            frac = band.sum() / n_in
            if frac <= max_frac:
                break
            kth = int(max(1, target_frac * n_in))
            kth = min(kth, abs_s_inside.size - 1)
            h_band_new = float(np.partition(abs_s_inside, kth)[kth])
            h_band = max(0.5 * voxel, h_band_new)
            band = (np.abs(s_signed) <= h_band) & inside

    pstat("Boundary band fraction", band.astype(float))

    # Axis seeds via collapsing random interior points along +grad φ
    rng = np.random.default_rng(0)
    candidates = Xq[inside]
    if candidates.shape[0] < axis_seed_count:
        axis_seed_count = max(8, candidates.shape[0]//20)
    picks = candidates[rng.choice(candidates.shape[0], size=axis_seed_count, replace=False)]
    axis_pts = np.array([collapse_to_axis(grad_phi, x0, step=0.1*min(dx,dy,dz), iters=600, tol=1e-6) for x0 in picks])
    # Cluster axis points to a 1D set (thin tube): use kNN thinning
    nbrs = NearestNeighbors(n_neighbors=8).fit(axis_pts)
    # Just keep them all; band selection will take radius
    # axis_band = axis_band_mask(axis_pts, Xq, rad=axis_band_radius*np.linalg.norm(span))
    # axis_band = np.logical_and(axis_band, inside)
    # pick radius as a few voxels wide
    # Downsample axis points to avoid "everywhere near-axis" tubes
    if axis_pts.shape[0] > 64:
        # quick thinning: keep every k-th point in k-NN order
        keep = np.linspace(0, axis_pts.shape[0]-1, 64, dtype=int)
        axis_pts_ds = axis_pts[keep]
    else:
        axis_pts_ds = axis_pts

    # Radius ~ 1–1.5 voxels
    axis_band_radius = float(axis_band_radius) * np.linalg.norm(span)
    axis_band = axis_band_mask(axis_pts_ds, Xq, rad=axis_band_radius) & inside
    pstat("Axis band fraction", axis_band.astype(float))

    # Evaluate grad φ everywhere inside (vectorized in chunks to save memory)
    pinfo("Evaluating ∇φ on grid (chunked)...")
    def eval_grad_chunk(Xchunk): return np.asarray(grad_phi(jnp.asarray(Xchunk)))
    G = np.zeros_like(Xq)
    chunk = 50000
    for s in range(0, Xq.shape[0], chunk):
        G[s:s+chunk] = eval_grad_chunk(Xq[s:s+chunk])
    eps_t = 1e-12
    gnorm = np.linalg.norm(G, axis=1)
    mask_t = (gnorm > eps_t)
    t_hat = np.zeros_like(G)
    t_hat[mask_t] = (G[mask_t].T / gnorm[mask_t]).T
    D = diffusion_tensor(G, eps=eps)

    Dxx = D[...,0,0]; Dyy = D[...,1,1]; Dzz = D[...,2,2]
    Dxy = D[...,0,1]; Dxz = D[...,0,2]; Dyz = D[...,1,2]
    pstat("Dxx", Dxx); pstat("Dyy", Dyy); pstat("Dzz", Dzz)
    pstat("Dxy", Dxy); pstat("Dxz", Dxz); pstat("Dyz", Dyz)

    n_tot = Xq.shape[0]
    n_in  = int(inside.sum())
    n_bnd = int(band.sum())
    n_ax  = int(axis_band.sum())

    # Nodes we actually solve for (unfixed interior)
    fixed = np.zeros(n_tot, dtype=bool)
    fixed[band] = True
    fixed[axis_band] = True
    fixed[~inside] = True
    n_fixed = int(fixed.sum())
    n_free  = int(n_tot - n_fixed)

    pinfo(f"[COUNT] total={n_tot} inside={n_in} band={n_bnd} axis={n_ax} fixed={n_fixed} free={n_free}")
    if (n_bnd / max(1, n_in)) > 0.25:
        pinfo("[WARN] boundary band too thick; try --band-h ≈ 1.0–2.0 (voxel units)")
    if (n_ax / max(1, n_in)) > 0.05:
        pinfo("[WARN] axis band too thick; try axis radius ≈ 1–2 voxels")
    if n_free < 0.2 * n_in:
        pinfo("[WARN] too few free interior nodes; shrink bands or refine grid (--N)")
    if n_free <= 0:
        pinfo("[WARN] No free interior unknowns! Your boundary/axis bands are covering everything.")
    pinfo(f"[BAND] surf_h={surf_h:.3e}, h_band={h_band:.3e}, "
        f"band/inside={(n_bnd/max(1,n_in))*100:.1f}% of interior")
    pinfo(f"[AXIS] axis_band_radius={axis_band_radius:.3e}, "
        f"axis/inside={(n_ax/max(1,n_in))*100:.1f}% of interior")
    pinfo(f"[AXIS] seeds={axis_seed_count}, kept={axis_pts_ds.shape[0]} (thin tube)")
    
    # Assemble SPD operator
    pinfo("Assembling sparse operator L ...")
    L, rhs = build_sparse_operator(nx, ny, nz, dx, dy, dz, inside, D)

    # Dirichlet rows for boundary and axis bands
    Ntot = Xq.shape[0]
    fixed = np.zeros(Ntot, dtype=bool)
    val = np.zeros(Ntot, dtype=float)
    fixed[band] = True; val[band] = 1.0
    fixed[axis_band] = True; val[axis_band] = 0.0
    # Outside nodes also fixed (identity rows already in L)
    fixed[np.logical_not(inside)] = True; val[np.logical_not(inside)] = 0.0

    pinfo(f"[L] nnz={L.nnz}, density={L.nnz/(L.shape[0]**2):.3e}")

    # Impose Dirichlet on rows (standard row stamping)
    pinfo("Imposing Dirichlet rows ...")
    L = L.tolil()
    rows = np.where(fixed)[0]
    L[rows, :] = 0.0
    L[rows, rows] = 1.0
    rhs[rows] = val[rows]
    # (optional) also stamp columns for stricter SPD full operator
    # L = L.tocsc(); L[:, rows] = 0.0; L = L.tolil(); L[rows, rows] = 1.0; L = L.tocsr()
    # Convert once to CSR and proceed; avoid zeroing columns (slow) and global symmetrization
    L = L.tocsr()

    # --- NEW: extract the reduced free-free system ---
    free  = ~fixed
    fidx  = np.where(free)[0]
    cidx  = np.where(fixed)[0]   # complement (fixed nodes)

    n_free = int(free.sum())
    if n_free == 0:
        raise RuntimeError("No free interior unknowns after band selection.")

    # Known Dirichlet values on fixed nodes
    psi_fixed = val[fixed]  # shape (n_fixed,)

    # Blocks
    L_ff = L[free][:, free].tocsr()
    L_fc = L[free][:, fixed].tocsr()

    symerr = (L_ff - L_ff.T).power(1).sum()
    pinfo(f"[L_ff] symmetry check (||A-A^T||_1): {symerr:.2e}")

    # Correct RHS for fixed contributions:
    # L_ff * psi_f = rhs_f  with  rhs_f = rhs[free] - L_fc * psi_fixed
    rhs_f = rhs[free] - L_fc.dot(psi_fixed)

    diagL = np.array(L_ff.diagonal())
    pstat("L_ff diag", diagL)
    pinfo(f"[L_ff] nnz={L_ff.nnz}, avg row nnz≈{L_ff.nnz/L_ff.shape[0]:.1f}")

    # Try ILU(0) on the reduced system; if it fails, try AMG (if available),
    # else fall back to Jacobi.
    try:
        ilu = spilu(
            L_ff.tocsc(),
            drop_tol=1e-4,      # allow dropping tiny fill
            fill_factor=5.0,    # more fill → better PC
            diag_pivot_thresh=0.1
        )
        M = LinearOperator(L_ff.shape, ilu.solve)
        pinfo("[PCG] Using ILU(0) preconditioner on reduced system")
    except Exception as e:
        pinfo(f"[PCG] ILU failed on reduced system ({e}); trying AMG if available...")
        try:
            import pyamg  # optional dependency
            ml = pyamg.ruge_stuben_solver(L_ff)  # classical AMG
            M = LinearOperator(L_ff.shape, lambda x: ml.aspreconditioner()(x))
            pinfo("[PCG] Using AMG preconditioner (pyamg)")
        except Exception as _:
            pinfo("[PCG] Falling back to Jacobi")
            from scipy.sparse import diags as spdiags
            diagL = np.array(L_ff.diagonal(), dtype=float)
            inv_diag = np.where(diagL > 0, 1.0/diagL, 1.0)
            M = spdiags(inv_diag, offsets=0)

    # Solve
    pinfo("CG solve on reduced system ...")
    t0 = time.time()
    try:
        psi_f, info = cg(L_ff, rhs_f, rtol=cg_tol, atol=0.0, maxiter=cg_maxit, M=M)
    except TypeError:
        psi_f, info = cg(L_ff, rhs_f, tol=cg_tol, maxiter=cg_maxit, M=M)
    t1 = time.time()
    
    # Report true reduced residual
    # Residuals computed against the SAME (unsymmetrized) blocks:
    r_free = (L_ff @ psi_f) + (L_fc @ psi_fixed) - rhs[free]
    pinfo(f"[PCG] reduced residual: L2={np.linalg.norm(r_free):.3e}, Linf={np.max(np.abs(r_free)):.3e}")
    pinfo(f"CG done (reduced): info={info}, iters<= {cg_maxit}, wall={t1-t0:.2f}s")

    # Scatter back to full psi vector
    psi = np.array(rhs)  # start with rhs so Dirichlet values are in place
    psi[free] = psi_f

    # Normalize ψ on the inside so [0,1] roughly spans interior
    psi_in = psi[inside]
    p01, p50, p99 = np.percentile(psi_in, [1, 50, 99])
    scale = max(p99 - p01, 1e-12)
    psi_n = np.clip((psi - p01)/scale, 0.0, 1.0)
    psi0_n = np.clip((psi0 - p01)/scale, 0.0, 1.0)

    pinfo(f"[ψ] inside percentiles: p1={p01:.3e}, p50={p50:.3e}, p99={p99:.3e}; using normalized ψ in [0,1]")

    # Quality metrics
    # parallel derivative proxy: |t_hat·∇ψ|
    # compute ∇ψ by central differences (skip edges)
    # x,y,z indexing
    psi3 = psi.reshape(nx, ny, nz)

    # Centered differences on the common interior core -> (nx-2, ny-2, nz-2)
    dpsidx = (psi3[2:,   1:-1, 1:-1] - psi3[:-2, 1:-1, 1:-1]) / (2*dx)
    dpsidy = (psi3[1:-1, 2:,   1:-1] - psi3[1:-1, :-2, 1:-1]) / (2*dy)
    dpsidz = (psi3[1:-1, 1:-1, 2:  ] - psi3[1:-1, 1:-1, :-2 ]) / (2*dz)

    # Match t_hat to the same interior core
    t_hat_x = t_hat[:,0].reshape(nx, ny, nz)[1:-1,1:-1,1:-1]
    t_hat_y = t_hat[:,1].reshape(nx, ny, nz)[1:-1,1:-1,1:-1]
    t_hat_z = t_hat[:,2].reshape(nx, ny, nz)[1:-1,1:-1,1:-1]

    par = (t_hat_x*dpsidx + t_hat_y*dpsidy + t_hat_z*dpsidz).ravel()
    pstat("|t·∇ψ| (interior)", par)

    # build boolean voxel masks
    band_vox  = band.reshape(nx,ny,nz)
    axis_vox  = axis_band.reshape(nx,ny,nz)
    inside_3 = inside.reshape(nx, ny, nz)
    inside_vox = inside_3

    # dilate bands by one cell to avoid numerical mixing (simple pad + max)
    def dilate1(b):
        # cheap 6-neighbor dilation
        bb = b.copy()
        bb[1:,:,:] |= b[:-1,:,:]; bb[:-1,:,:] |= b[1:,:,:]
        bb[:,1:,:] |= b[:,:-1,:]; bb[:,:-1,:] |= b[:,1:,:]
        bb[:,:,1:] |= b[:,:,:-1]; bb[:,:,:-1] |= b[:,:,1:]
        return bb

    band_dil  = dilate1(band_vox)
    axis_dil  = dilate1(axis_vox)

    core = inside_vox.copy()
    core[band_dil] = False
    core[axis_dil] = False

    # interior core slice to match (nx-2,ny-2,nz-2)
    core_mid = core[1:-1,1:-1,1:-1]
    gn_mid   = gnorm.reshape(nx,ny,nz)[1:-1,1:-1,1:-1]
    mask_mid = core_mid & (gn_mid > 1e-10)

    par_core = (t_hat_x*dpsidx + t_hat_y*dpsidy + t_hat_z*dpsidz)
    par = par_core[mask_mid].ravel()
    if par.size < 100:
        pinfo("[CORE] very small core sample; shrink bands or increase --N")
    pstat("|t·∇ψ| (interior core)", par)
    pinfo(f"[CORE] kept {par.size} samples ({100.0*par.size/max(1,core_mid.size):.1f}% of interior core)")

    # Flux neutrality check inside: ∫ div(D∇ψ) dV ≈ 0 → sample residual
    # quick residual r = Lψ - rhs (dense-free via multiply)
    # Full residual (again vs the stamped L)
    r = L @ psi - rhs
    pstat("Residual Lψ - rhs", r)

    # split residual sanity
    free = ~fixed
    pstat("Residual (free rows)", (L[free] @ psi - rhs[free]))
    pstat("Residual (fixed rows)", (L[fixed] @ psi - rhs[fixed]))

    pinfo(f"[PCG] iterations: {info}")
    cond_est = np.max(np.abs(diagL)) / max(1e-16, np.min(np.abs(diagL)))
    pinfo(f"[L_ff] crude cond. diag ratio ~ {cond_est:.2e}")

    # ------------------------------ Plots ------------------------------ #
    if plot:
        # 3D isosurfaces
        # Compute grad φ on a stride of boundary points and quiver
        stride = max(1, P.shape[0] // 1200)   # keep it light
        Pb = P[::stride]
        Gb = np.asarray(grad_phi(jnp.asarray(Pb)))
        # scale arrows by a visually reasonable factor
        gscale = 0.15 * max(maxs - mins).max() * 0.03

        fig = plt.figure(figsize=(14,6))
        ax = fig.add_subplot(1,2,1, projection='3d')
        # take only inside region
        vol = psi.reshape(nx,ny,nz).copy()
        vol[~inside.reshape(nx,ny,nz)] = np.nan
        for lv in [0.3, 0.6]:
            try:
                verts, faces, norm, val = marching_cubes(np.nan_to_num(vol, nan=-1.0), level=lv)
                # map verts from index to real coords
                vx = mins[0] + verts[:,0]*(maxs[0]-mins[0])/(nx-1)
                vy = mins[1] + verts[:,1]*(maxs[1]-mins[1])/(ny-1)
                vz = mins[2] + verts[:,2]*(maxs[2]-mins[2])/(nz-1)
                ax.plot_trisurf(vx, vy, faces, vz, alpha=0.35, linewidth=0.1)
            except Exception as e:
                pinfo(f"Marching cubes failed at level {lv}: {e}")
        ax.scatter(Pb[:,0], Pb[:,1], Pb[:,2], s=2, c='k', alpha=0.35)
        ax.quiver(Pb[:,0], Pb[:,1], Pb[:,2], Gb[:,0], Gb[:,1], Gb[:,2],
                    length=gscale, normalize=True, linewidth=0.5, color='tab:red', alpha=0.6)
        ax.set_title("Isosurfaces of ψ (0.2, 0.4) + boundary points")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        fix_matplotlib_3d(ax)

        # Quality dashboard
        ax2 = fig.add_subplot(1,2,2)
        ax2.hist(np.abs(par), bins=60, alpha=0.8)
        ax2.set_yscale('log'); ax2.set_xlabel(r"$|\,\hat t\cdot\nabla\psi\,|$")
        ax2.set_title("Field-aligned derivative magnitude (smaller is better)")
        plt.tight_layout()

        # Residual scatter & slices
        fig2, axa = plt.subplots(1,3, figsize=(14,4))
        r3 = r.reshape(nx,ny,nz)
        im0 = axa[0].imshow(r3[:,:,nz//2].T, origin='lower', aspect='equal'); plt.colorbar(im0, ax=axa[0]); axa[0].set_title("residual @ z mid")
        im1 = axa[1].imshow(r3[:,ny//2,:].T, origin='lower', aspect='equal'); plt.colorbar(im1, ax=axa[1]); axa[1].set_title("residual @ y mid")
        im2 = axa[2].imshow(r3[nx//2,:,:].T, origin='lower', aspect='equal'); plt.colorbar(im2, ax=axa[2]); axa[2].set_title("residual @ x mid")
        plt.tight_layout()

        # --------- Poincaré-like cross-sections of iso-surface ψ=psi0 ----------
        # --- Build iso-surface once (normalized ψ) ---
        try:
            Viso, Fiso = march_iso_surface(psi_n, inside, nx, ny, nz, mins, maxs, level=psi0_n)
        except Exception as e:
            pinfo(f"Marching cubes (normalized) failed at ψ≈{psi0_n:.2f}: {e}")
            Viso = Fiso = None

        phi_list = np.linspace(-np.pi, np.pi, 8, endpoint=False)
        figp, axes = plt.subplots(2, 4, figsize=(14,7), constrained_layout=True)
        axes = axes.ravel()

        for aidx, phi0 in enumerate(phi_list):
            axp = axes[aidx]
            if Viso is None:
                axp.set_title(f"φ={phi0:+.2f} (no iso)"); axp.axis('off'); continue
            segs = intersect_iso_with_phi_plane(Viso, Fiso, phi0)
            nseg = draw_poincare_from_segments(axp, segs)
            pinfo(f"[POI] φ={phi0:+.2f}: segments={nseg}")
            axp.set_aspect('equal', 'box'); axp.set_xlabel("R"); axp.set_ylabel("Z")
            axp.set_title(f"φ={phi0:+.2f} (ψ≈{psi0_n:.2f})")

        for k in range(len(phi_list), len(axes)):
            axes[k].axis('off')
        
        plt.show()

    return dict(psi=psi, grid=(xs,ys,zs), inside=inside, quality=dict(parallel_dot_grad=par, residual=r))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="MFS solution checkpoint (*.npz) containing center, scale, Yn, alpha, a, a_hat, P, N")
    ap.add_argument("--N", type=int, default=26, help="grid resolution per axis")
    ap.add_argument("--eps", type=float, default=1e-2, help="parallel diffusion weight (smaller => more field-aligned)")
    ap.add_argument("--band-h", type=float, default=1.0, help="boundary band thickness multiplier")
    ap.add_argument("--cg-maxit", type=int, default=500)
    ap.add_argument("--axis-seed-count", type=int, default=64, help="number of interior seeds to collapse onto axis")
    ap.add_argument("--axis-band-radius", type=float, default=0.012, help="axis band radius as fraction of bbox size")
    ap.add_argument("--cg-tol", type=float, default=1e-8)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--psi0", type=float, default=0.3, help="iso-surface level for plotting")
    args = ap.parse_args()
    out = main(args.npz, grid_N=args.N, eps=args.eps, band_h=args.band_h,
               axis_seed_count=args.axis_seed_count, axis_band_radius=args.axis_band_radius,
               cg_tol=args.cg_tol, cg_maxit=args.cg_maxit, plot=(not args.no_plot), psi0=args.psi0)
