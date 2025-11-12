#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global field-aligned flux function ψ via anisotropic diffusion:
    div( D(x) ∇ψ ) = 0,  D = eps P_perp + 1·P_par,  t = grad φ / |grad φ|
BCs: ψ=1 on Γ (thin boundary band), ψ=0 on axis band (detected by short gradient-flow collapse).
This makes ψ ~ constant along grad φ while diffusing across it → nested level sets.

Refs (theory & numerics):
- Weickert, "Anisotropic Diffusion in Image Processing" (Teubner, 1998); CE diffusion (1999). 
- Field-aligned diffusion solvers in plasma (FCI/field-line-map & anisotropic diffusion stability).
"""

from __future__ import annotations
import argparse, time
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, diags as spdiags
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from scipy.sparse.linalg import LinearOperator, cg, minres, gmres
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pyamg import smoothed_aggregation_solver
from sklearn.cluster import DBSCAN
from scipy.interpolate import RegularGridInterpolator as RGI

# ---------------------------- Debug utils ---------------------------- #
def pct(a, p): return float(np.percentile(np.asarray(a), p))
def pinfo(msg): print(f"[INFO] {msg}")
def pstat(msg, v):
    v=np.asarray(v); print(f"[STAT] {msg}: min={v.min():.3e} med={np.median(v):.3e} max={v.max():.3e} L2={np.linalg.norm(v):.3e}")

def probe_symmetry(A, n=3, seed=0):
    rng = np.random.default_rng(seed)
    rels = []
    for _ in range(n):
        x = rng.standard_normal(A.shape[0])
        y = rng.standard_normal(A.shape[0])
        Ax, Ay = A @ x, A @ y
        num = float(x @ Ay - Ax @ y)
        denom = float(max(1e-16, abs(x @ Ay) + abs(Ax @ y)))
        rels.append(abs(num)/denom)
    return max(rels)

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
def collapse_to_axis(grad_phi, X0, step=0.02, iters=400, tol=1e-6, dir_sign=+1.0):
    X = jnp.asarray(X0, dtype=jnp.float64)
    @jit
    def one_step(X):
        g = grad_phi(X[None, :])[0]
        n = jnp.linalg.norm(g) + 1e-30
        return X - step * dir_sign * (g / n)
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
def diffusion_tensor(gradphi, eps, delta=1e-2):
    """
    Build D = R diag(1, eps, eps) R^T + delta I, where the first column of R is t̂.
    This is equivalent to eps*(I - t̂t̂^T) + t̂t̂^T + delta I but constructed via a
    numerically stable local frame (t̂, b̂1, b̂2).
    """
    I = np.eye(3)[None, :, :]
    n = np.linalg.norm(gradphi, axis=-1, keepdims=True)
    t = np.divide(gradphi, n, out=np.zeros_like(gradphi), where=n > 1e-12)  # (N,3)

    # Build b1 robustly: pick an anchor not collinear with t
    # anchor = e_x unless |t_x| is too large, then use e_y
    ax = np.abs(t[..., 0]) > 0.70710678
    anchor = np.zeros_like(t)
    anchor[~ax, 0] = 1.0  # use ex
    anchor[ ax, 1] = 1.0  # use ey
    b1 = np.cross(t, anchor)                                    # (N,3)
    nb1 = np.linalg.norm(b1, axis=-1, keepdims=True)
    ok = nb1[:, 0] > 1e-15
    b1[ok] /= nb1[ok]
    b1[~ok] = 0.0
    b2 = np.cross(t, b1)

    # Orthonormal basis matrix R = [t, b1, b2]
    R = np.stack([t, b1, b2], axis=-1)                          # (N,3,3)
    # Efficient diag construction (no shape shenanigans):
    Lam = np.zeros((1, 3, 3), dtype=float)
    Lam[..., 0, 0] = 1.0
    Lam[..., 1, 1] = float(eps)
    Lam[..., 2, 2] = float(eps)

    D = R @ Lam @ np.swapaxes(R, -1, -2) + delta * I            # (N,3,3)
    return D

# ===================== Matrix-free operator & simple AMG PC =====================

def _harmonic(a, b, eps=1e-30):
    return 2.0 * a * b / np.maximum(a + b, eps)

def _face_T(D3, dx, dy, dz):
    """
    Two-point flux transmissibility using the face-normal component n·D·n.
    For axis-aligned faces, n is e_x, e_y, or e_z, hence this equals the
    corresponding diagonal entry of D in world coords.
    """
    Dxx = D3[..., 0, 0]; Dyy = D3[..., 1, 1]; Dzz = D3[..., 2, 2]

    # x-faces between i-1 and i  -> shape (nx-1,ny,nz)
    kx = _harmonic(Dxx[1:, :, :], Dxx[:-1, :, :]) / (dx*dx)

    # y-faces between j-1 and j  -> shape (nx,ny-1,nz)
    ky = _harmonic(Dyy[:, 1:, :], Dyy[:, :-1, :]) / (dy*dy)

    # z-faces between k-1 and k  -> shape (nx,ny,nz-1)
    kz = _harmonic(Dzz[:, :, 1:], Dzz[:, :, :-1]) / (dz*dz)

    return kx, ky, kz

def build_aniso_csr_free(nx, ny, nz, dx, dy, dz, inside, fixed, Dfield, val):
    inside_3 = inside.reshape(nx,ny,nz)
    fixed_1d = fixed.ravel(order="C")
    val_1d   = val.ravel(order="C")
    D3 = Dfield.reshape(nx,ny,nz,3,3)

    Tx, Ty, Tz = _face_T(D3, dx, dy, dz)  # harmonic face coeffs

    idx = np.arange(nx*ny*nz).reshape(nx,ny,nz)
    rows, cols, vals = [], [], []
    b = np.zeros(nx*ny*nz, dtype=float)

    def add(rc, cc, vv):
        rows.append(rc); cols.append(cc); vals.append(vv)

    diag = np.zeros(nx*ny*nz, dtype=float)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not inside_3[i,j,k]:
                    continue
                p = idx[i,j,k]
                if fixed_1d[p]:
                    # fixed rows are dropped later by slicing (no entries here)
                    continue

                # x- neighbor (face between i-1 and i -> Tx[i-1,j,k])
                if i > 0 and inside_3[i-1,j,k]:
                    q = idx[i-1,j,k]
                    K = Tx[i-1,j,k]
                    if fixed_1d[q]:
                        b[p] += K * val_1d[q]
                    else:
                        add(p, q, -K)
                    diag[p] += K

                # x+ neighbor (face between i and i+1 -> Tx[i,j,k])
                if i+1 < nx and inside_3[i+1,j,k]:
                    q = idx[i+1,j,k]
                    K = Tx[i,j,k]
                    if fixed_1d[q]:
                        b[p] += K * val_1d[q]
                    else:
                        add(p, q, -K)
                    diag[p] += K

                # y- (face j-1/j -> Ty[i,j-1,k])
                if j > 0 and inside_3[i,j-1,k]:
                    q = idx[i,j-1,k]
                    K = Ty[i,j-1,k]
                    if fixed_1d[q]:
                        b[p] += K * val_1d[q]
                    else:
                        add(p, q, -K)
                    diag[p] += K

                # y+ (face j/j+1 -> Ty[i,j,k])
                if j+1 < ny and inside_3[i,j+1,k]:
                    q = idx[i,j+1,k]
                    K = Ty[i,j,k]
                    if fixed_1d[q]:
                        b[p] += K * val_1d[q]
                    else:
                        add(p, q, -K)
                    diag[p] += K

                # z- (face k-1/k -> Tz[i,j,k-1])
                if k > 0 and inside_3[i,j,k-1]:
                    q = idx[i,j,k-1]
                    K = Tz[i,j,k-1]
                    if fixed_1d[q]:
                        b[p] += K * val_1d[q]
                    else:
                        add(p, q, -K)
                    diag[p] += K

                # z+ (face k/k+1 -> Tz[i,j,k])
                if k+1 < nz and inside_3[i,j,k+1]:
                    q = idx[i,j,k+1]
                    K = Tz[i,j,k]
                    if fixed_1d[q]:
                        b[p] += K * val_1d[q]
                    else:
                        add(p, q, -K)
                    diag[p] += K

    # add diagonals only for free, inside rows
    free = (inside) & (~fixed)
    p_idx = np.where(free)[0]
    rows += list(p_idx); cols += list(p_idx); vals += list(diag[p_idx])

    A = coo_matrix((vals, (rows, cols)), shape=(nx*ny*nz, nx*ny*nz)).tocsr()
    A_ff = A[free][:, free].tocsr()
    b_f  = b[free]
    fidx = np.where(free)[0]
    return A_ff, b_f, fidx

def _matvec_full(u, nx, ny, nz, dx, dy, dz, inside_3, D3):
    u3 = u.reshape(nx, ny, nz)

    Tx, Ty, Tz = _face_T(D3, dx, dy, dz)

    mx = inside_3[1:,:,:] & inside_3[:-1,:,:]
    my = inside_3[:,1:,:] & inside_3[:,:-1,:]
    mz = inside_3[:,:,1:] & inside_3[:,:,:-1]

    duf_x = np.zeros_like(Tx); duf_x[mx] = (u3[1:,:,:] - u3[:-1,:,:])[mx]
    duf_y = np.zeros_like(Ty); duf_y[my] = (u3[:,1:,:] - u3[:,:-1,:])[my]
    duf_z = np.zeros_like(Tz); duf_z[mz] = (u3[:,:,1:] - u3[:,:,:-1])[mz]

    Fx = np.zeros_like(Tx); Fx[mx] = Tx[mx] * duf_x[mx]
    Fy = np.zeros_like(Ty); Fy[my] = Ty[my] * duf_y[my]
    Fz = np.zeros_like(Tz); Fz[mz] = Tz[mz] * duf_z[mz]

    divF = np.zeros((nx, ny, nz))

    # x-direction
    divF[1:-1,:,:] += (Fx[1:,:,:] - Fx[:-1,:,:])

    # y-direction
    divF[:,1:-1,:] += (Fy[:,1:,:] - Fy[:,:-1,:])

    # z-direction
    divF[:,:,1:-1] += (Fz[:,:,1:] - Fz[:,:,:-1])

    divF[~inside_3] = 0.0
    return (-divF).ravel(order="C")

def apply_fci_spd(u, fci, alpha_par=1.0):
    """
    SPD 'normal-equations' form for the parallel operator:
      A_par u = α * [ (W+ - I)^T (W+ - I) + (W- - I)^T (W- - I) ] u / step^2
    This is guaranteed symmetric positive semidefinite even with masking.
    """
    idxp, wp, in_p = fci["idxp"], fci["wp"], fci["in_p"]
    idxm, wm, in_m = fci["idxm"], fci["wm"], fci["in_m"]
    step = np.maximum(1e-30, fci["step"])

    def apply_one(W_idx, W_w, in_mask):
        # Wu (interpolation)
        # Wu = np.sum(u[W_idx] * W_w, axis=1)
        # Wu = np.where(in_mask, Wu, u)   # if footpoint invalid, treat as identity (W->I)
        # v  = Wu - u                     # (W - I) u
        Wu = np.sum(u[W_idx] * W_w, axis=1)
        Wu = np.where(in_mask, Wu, u)   # current
        # better: if invalid, drop the whole parallel term for that row
        v  = np.where(in_mask, Wu - u, 0.0)

        # y = (W - I)^T v
        y = -v.copy()
        np.add.at(y, W_idx, W_w * v[:, None])  # (N,8) broadcast, no manual repeat
        return y

    r = apply_one(idxp, wp, in_p) + apply_one(idxm, wm, in_m)
    return alpha_par * (r / (step**2))


def smooth_vec3_box(nx, ny, nz, V, passes=1):
    """
    3D 6-neighbor box smoother for a vector field V of shape (N,3) laid out as (nx,ny,nz).
    Returns smoothed and renormalized field (zero where original magnitude ~0).
    """
    V3 = V.reshape(nx, ny, nz, 3).copy()
    for _ in range(max(0, passes)):
        W = np.zeros_like(V3)
        C = np.ones((nx,ny,nz,1))  # center weight 1
        # 6-neighbors (no diagonals)
        W[1:,:,:]   += V3[:-1,:,:];   C[1:,:,:]   += 1
        W[:-1,:,:]  += V3[1:,:,:];    C[:-1,:,:]  += 1
        W[:,1:,:]   += V3[:,:-1,:];   C[:,1:,:]   += 1
        W[:,:-1,:]  += V3[:,1:,:];    C[:,:-1,:]  += 1
        W[:,:,1:]   += V3[:,:,:-1];   C[:,:,1:]   += 1
        W[:,:,:-1]  += V3[:,:,1:];    C[:,:,:-1]  += 1
        V3 = (V3 + W) / np.maximum(C, 1.0)

    Vflat = V3.reshape(-1,3)
    nrm = np.linalg.norm(Vflat, axis=1, keepdims=True)
    good = (nrm[:,0] > 1e-12)
    Vout = np.zeros_like(Vflat)
    Vout[good] = Vflat[good] / nrm[good]
    return Vout


def make_linear_operator(nx, ny, nz, dx, dy, dz, inside, Dperp, fixed_mask, fixed_val, fci=None, alpha_par=1.0):
    """
    Build a LinearOperator A_full such that A_full @ u_full returns
    the residual of the stamped system:
      - free rows: div(D grad u)
      - fixed rows: u - fixed_val
    All vectors are full-sized (N=nx*ny*nz).
    """
    inside_3 = inside.reshape(nx,ny,nz)
    # D3 = Dfield.reshape(nx,ny,nz,3,3)
    D3 = Dperp.reshape(nx,ny,nz,3,3)
    rows_fixed = np.where(fixed_mask)[0]
    inside_rows = np.where(inside.ravel(order="C"))[0]  # flat indices of interior

    def matvec(u):
        # ⊥ part (matrix-free face flux with D_perp)
        out = _matvec_full(u, nx, ny, nz, dx, dy, dz, inside_3, D3)
        # ∥ part via FCI (discrete 1D Laplacian along field lines)
        if fci is not None:
            r_par_full = apply_fci_spd(u, fci, alpha_par=alpha_par)
            out += r_par_full
            
        # Dirichlet stamping last (overrides any accumulation on fixed rows)
        out[rows_fixed] = u[rows_fixed] - fixed_val[rows_fixed]

        # safety: zero outside-of-domain rows
        out[~inside.ravel(order="C")] = 0.0
        return out

    N = nx*ny*nz
    return LinearOperator((N, N), matvec=matvec, rmatvec=matvec, dtype=float)

# ---------------- FCI helpers: trilinear & footpoints ---------------- #
def _search_indices(coord, grid):
    """For each value in coord, return lower cell index i with grid[i] <= x <= grid[i+1].
    Clamped to [0, len(grid)-2]. Vectorized."""
    i = np.searchsorted(grid, coord, side="right") - 1
    return np.clip(i, 0, len(grid) - 2)

def _trilinear_weights_indices(Xp, xs, ys, zs, nx, ny, nz):
    """Given footpoints Xp=(N,3), return (idx8, w8) where:
       - idx8: (N,8) flat indices into the (nx,ny,nz) C-ordered grid
       - w8  : (N,8) weights that sum to 1 for each row
    Points outside the bbox are clamped to nearest cell for stability."""
    x, y, z = Xp[:,0], Xp[:,1], Xp[:,2]
    i = _search_indices(x, xs); j = _search_indices(y, ys); k = _search_indices(z, zs)
    x0, x1 = xs[i], xs[i+1]; y0, y1 = ys[j], ys[j+1]; z0, z1 = zs[k], zs[k+1]
    with np.errstate(divide="ignore", invalid="ignore"):
        fx = (x - x0) / np.maximum(x1 - x0, 1e-30)
        fy = (y - y0) / np.maximum(y1 - y0, 1e-30)
        fz = (z - z0) / np.maximum(z1 - z0, 1e-30)
    fx = np.clip(fx, 0.0, 1.0); fy = np.clip(fy, 0.0, 1.0); fz = np.clip(fz, 0.0, 1.0)
    def flat(ii, jj, kk): return (ii + nx*(jj + ny*kk)).astype(np.int64)
    i0, j0, k0 = i, j, k; i1, j1, k1 = i+1, j+1, k+1
    idx8 = np.stack([flat(i0,j0,k0), flat(i1,j0,k0), flat(i0,j1,k0), flat(i1,j1,k0),
                     flat(i0,j0,k1), flat(i1,j0,k1), flat(i0,j1,k1), flat(i1,j1,k1)], axis=1)
    wx0, wx1 = (1.0 - fx), fx; wy0, wy1 = (1.0 - fy), fy; wz0, wz1 = (1.0 - fz), fz
    w8 = np.stack([wx0*wy0*wz0, wx1*wy0*wz0, wx0*wy1*wz0, wx1*wy1*wz0,
                   wx0*wy0*wz1, wx1*wy0*wz1, wx0*wy1*wz1, wx1*wy1*wz1], axis=1)
    return idx8, w8

def precompute_fci(xs, ys, zs, Xq, t_hat, step_vec, inside_bool, substeps_max=6):
    """
    Compute +/- footpoints with adaptive substepping so the endpoints remain inside.
    If a full step would leave the 'effective inside' set, we bisect until valid or
    we hit substeps_max. Returns the same dict keys as before.
    """
    xs = np.asarray(xs); ys = np.asarray(ys); zs = np.asarray(zs)
    Xq = np.asarray(Xq); th = np.asarray(t_hat)
    inside = np.asarray(inside_bool).astype(bool)
    nx, ny, nz = len(xs), len(ys), len(zs)

    # utility: trilinear lookups for a batch of points
    def idx_w_for(points):
        return _trilinear_weights_indices(points, xs, ys, zs, nx, ny, nz)

    # helper to test if the *cell center* of the containing voxel is inside
    inside_3 = inside.reshape(nx, ny, nz)
    def center_inside(idx8):
        i = idx8[:, 0] % nx
        j = (idx8[:, 0] // nx) % ny
        k = (idx8[:, 0] // (nx * ny))
        return inside_3.ravel(order="C")[i + nx*(j + ny*k)]

    step = np.asarray(step_vec).reshape(-1, 1)
    Xp_try = Xq + step * th
    Xm_try = Xq - step * th

    # If a footpoint lands outside, shorten the step (bisect) up to substeps_max times
    def repair(points, sign):
        P = points.copy()
        s = step.copy()
        ok_idx, w = idx_w_for(P)
        ok = center_inside(ok_idx)
        tries = 0
        while (not ok.all()) and (tries < substeps_max):
            # halve the remaining step where not ok
            s[~ok] *= 0.5
            P[~ok] = Xq[~ok] + sign * s[~ok] * th[~ok]
            ok_idx, w = idx_w_for(P)
            ok = center_inside(ok_idx)
            tries += 1
        return P, ok

    Xp, in_p = repair(Xp_try, +1.0)
    Xm, in_m = repair(Xm_try, -1.0)

    idxp, wp = _trilinear_weights_indices(Xp, xs, ys, zs, nx, ny, nz)
    idxm, wm = _trilinear_weights_indices(Xm, xs, ys, zs, nx, ny, nz)

    # Final per-node effective step (used in operator scaling)
    # (note: even when invalid, we keep step so SPD form can fall back to I)
    step_eff = step.flatten()

    return dict(idxp=idxp, wp=wp, idxm=idxm, wm=wm, step=step_eff, in_p=in_p, in_m=in_m)


def diffusion_tensor_perp(gradphi, eps, delta=1e-2):
    """D_perp = eps*(I - t t^T) + delta*I."""
    I = np.eye(3)[None,:,:]
    n = np.linalg.norm(gradphi, axis=-1, keepdims=True)
    t = np.divide(gradphi, n, out=np.zeros_like(gradphi), where=n > 1e-12)
    tt = np.einsum("...i,...j->...ij", t, t)
    D = eps*(I - tt) + delta*I
    return D

def diffusion_tensor_perp_world_diagonal(gradphi, eps, delta=1e-2):
    """
    World-diagonal version consistent with the 6-face 2-point stencil:
      D_diag = eps * diag(1 - t_x^2, 1 - t_y^2, 1 - t_z^2) + delta * I
    """
    n = np.linalg.norm(gradphi, axis=-1, keepdims=True)
    t = np.divide(gradphi, n, out=np.zeros_like(gradphi), where=n > 1e-12)
    tx2, ty2, tz2 = t[...,0]**2, t[...,1]**2, t[...,2]**2
    Dxx = eps*(1.0 - tx2) + delta
    Dyy = eps*(1.0 - ty2) + delta
    Dzz = eps*(1.0 - tz2) + delta
    # pack as (nx,ny,nz,3,3) with zeros off-diagonal
    D = np.zeros(gradphi.shape[:-1] + (3,3), dtype=float)
    D[...,0,0] = Dxx; D[...,1,1] = Dyy; D[...,2,2] = Dzz
    return D


def eps_schedule(final_eps):
    base = [0.3, 0.12, 0.06]
    if final_eps >= base[0]:
        return [final_eps]
    seq = [e for e in base if e > final_eps] + [final_eps]
    return seq

def _decide_axis_sign(grad_phi, x0):
    x0 = jnp.asarray(x0, dtype=jnp.float64)

    def advance(x_init, sgn, K=10, step=1e-2):
        x = jnp.asarray(x_init, dtype=jnp.float64)
        # initialize g at the starting point
        g = grad_phi(x[None, :])[0]
        n = jnp.linalg.norm(g) + 1e-30
        g = g / n
        for _ in range(K):
            # move first using current gradient
            x = x + sgn * step * g
            # then refresh gradient at the new location
            g = grad_phi(x[None, :])[0]
            n = jnp.linalg.norm(g) + 1e-30
            g = g / n
        return np.array(x)

    x_plus  = advance(x0, +1.0)
    x_minus = advance(x0, -1.0)
    return "+", x_plus, "-", x_minus

def march_iso_surface(psi_n, inside, nx, ny, nz, mins, maxs, level):
    vol = psi_n.reshape(nx, ny, nz).copy()
    mask = inside.reshape(nx, ny, nz)
    vol[~mask] = np.nan
    vol = np.nan_to_num(vol, nan=-1.0)
    verts, faces, norm, val = marching_cubes(vol, level=level, mask=mask)

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
        ax.plot(R, z, lw=1.0, alpha=0.9)
        ax.plot(R[::10], z[::10], '.', ms=1.8, alpha=0.7)
        cnt += 1
    return cnt

def boundary_curve_RZ(P, phi0, K_min=None):
    # P: (Ns,3) boundary points
    Phib = np.arctan2(P[:,1], P[:,0])
    # pick K nearest-by-angle (wrap-aware)
    dphi = ( (Phib - float(phi0) + np.pi) % (2*np.pi) ) - np.pi
    if K_min is None:
        K_min = max(50, P.shape[0]//200)   # ~0.5% or ≥50 points
    keep = np.argsort(np.abs(dphi))[:K_min]

    # orthonormal basis in the φ-plane
    c, s = np.cos(phi0), np.sin(phi0)
    eR   = np.array([ c,  s, 0.0])   # radial unit vector in plane
    ephi = np.array([-s,  c, 0.0])   # azimuthal unit normal of the plane

    # project exactly to φ=const plane: subtract (n·r) n with n = eφ
    dist = P[keep] @ ephi
    r_proj = P[keep] - dist[:,None] * ephi[None,:]

    Rcut = r_proj @ eR
    Zcut = r_proj[:,2]
    return Rcut, Zcut

def _snap_points(P, tol):
    """
    Snap nearly-identical 3D points within 'tol' together and return:
      - P_snapped: (m,3) unique points after snapping
      - labels: len(P) integers mapping each original row to a snapped index
    """
    if len(P) == 0:
        return P, np.array([], dtype=int)

    # grid-hash for O(n) expected
    key = np.floor(P / tol).astype(np.int64)
    # lexicographic key for grouping
    order = np.lexsort((key[:,2], key[:,1], key[:,0]))
    key_sorted = key[order]
    P_sorted = P[order]

    uniq_idx = [0]
    for i in range(1, len(P_sorted)):
        if not np.all(key_sorted[i] == key_sorted[uniq_idx[-1]]):
            uniq_idx.append(i)
    uniq_idx = np.array(uniq_idx, dtype=int)

    centers = []
    labels = np.empty(len(P_sorted), dtype=int)
    start = 0
    for ui, s in enumerate(uniq_idx):
        e = uniq_idx[ui+1] if ui+1 < len(uniq_idx) else len(P_sorted)
        block = P_sorted[s:e]
        c = block.mean(axis=0)
        centers.append(c)
        labels[s:e] = ui
        start = e

    centers = np.array(centers)
    # invert permutation
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return centers, labels[inv]

def segments_to_polylines(segs, tol):
    """
    Convert a list of 2x3 segments into polylines by endpoint stitching.
    Returns a list of arrays with shape (m_i, 3) each (closed loops will have first!=last, but are contiguous).
    """
    if not segs:
        return []

    # Collect endpoints
    P0 = np.vstack([s[0] for s in segs])
    P1 = np.vstack([s[1] for s in segs])
    Pend = np.vstack([P0, P1])

    # Snap endpoints
    centers, lab = _snap_points(Pend, tol)
    l0 = lab[:len(P0)]
    l1 = lab[len(P0):]

    # Build adjacency (undirected multigraph)
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for a, b in zip(l0, l1):
        if a == b:
            continue
        adj[a].append(b)
        adj[b].append(a)

    visited_edge = defaultdict(int)
    polylines = []

    # For each node, walk unused edges to build chains
    for start in range(len(centers)):
        if start not in adj:
            continue
        # try every neighbor to start a new chain
        for nb in list(adj[start]):
            # Edge key (ordered)
            ek = (min(start, nb), max(start, nb))
            if visited_edge[ek] > 0:
                continue

            # walk forward
            chain = [start, nb]
            visited_edge[ek] += 1

            # extend at the tail
            cur = nb
            prev = start
            while True:
                nxts = [u for u in adj[cur] if u != prev]
                nxt = None
                for u in nxts:
                    e2 = (min(cur, u), max(cur, u))
                    if visited_edge[e2] == 0:
                        nxt = u
                        visited_edge[e2] += 1
                        break
                if nxt is None:
                    break
                chain.append(nxt)
                prev, cur = cur, nxt

            # extend at the head
            cur = start
            prev = nb
            while True:
                nxts = [u for u in adj[cur] if u != prev]
                nxt = None
                for u in nxts:
                    e2 = (min(cur, u), max(cur, u))
                    if visited_edge[e2] == 0:
                        nxt = u
                        visited_edge[e2] += 1
                        break
                if nxt is None:
                    break
                chain = [nxt] + chain
                prev, cur = cur, nxt

            # Map to coordinates
            poly = centers[np.array(chain)]
            # Optional: close tiny gaps
            if np.linalg.norm(poly[0] - poly[-1]) < 2*tol and len(poly) > 3:
                poly = poly  # already effectively closed (we keep it open-form)
            polylines.append(poly)

    return polylines

def draw_poincare_from_polylines(ax, polylines):
    cnt = 0
    for P in polylines:
        R = np.sqrt(P[:,0]**2 + P[:,1]**2)
        Z = P[:,2]
        ax.plot(R, Z, lw=1.2, alpha=0.95)
        cnt += 1
    return cnt

# Choose a reasonable R,Z box per φ from the boundary points
def rz_box_for_phi(P, phi0, pad=0.05):
    c, s = np.cos(phi0), np.sin(phi0)
    eR = np.array([ c,  s, 0.0])
    eφ = np.array([-s,  c, 0.0])
    # project boundary to the plane: subtract (eφ·r)eφ
    dist = P @ eφ
    rproj = P - dist[:,None]*eφ[None,:]
    Rb = rproj @ eR
    Zb = rproj[:,2]
    # pad the min/max a bit
    Rmin, Rmax = Rb.min(), Rb.max()
    Zmin, Zmax = Zb.min(), Zb.max()
    dR, dZ = (Rmax-Rmin), (Zmax-Zmin)
    return (Rmin-pad*dR, Rmax+pad*dR), (Zmin-pad*dZ, Zmax+pad*dZ)

# ------------------------------- Main flow ------------------------------- #
def main(npz_file, grid_N=96, eps=1e-3, band_h=1.5, axis_seed_count=0, axis_band_radius=0.0,
         cg_tol=1e-8, cg_maxit=2000, verbose=True, plot=True, psi0=0.3, nfp=2, psi_levels="",
         save_figures=True, fci_step_cells=1.0, alpha_par=1.0, delta_mult=1e-2):

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
    target_frac = 0.08
    max_frac    = 0.12
    n_in = int(inside.sum())
    if n_in > 0:
        abs_s_inside = np.abs(s_signed[inside])
        # If current band too fat, shrink by quantile down to >= 0.5 voxel
        for _ in range(6):
            frac = band.sum() / n_in
            if frac <= max_frac:
                break
            kth = int(max(1, target_frac * n_in))
            kth = min(kth, abs_s_inside.size - 1)
            h_band_new = float(np.partition(abs_s_inside, kth)[kth])
            h_band = max(0.5 * voxel, h_band_new)
            band = (np.abs(s_signed) <= h_band) & inside

    pstat("Boundary band fraction", band.astype(float))
    
    pinfo("_decide_axis_sign outputs:")
    if not np.any(inside):
        raise RuntimeError("inside mask is empty; check surface normals or grid bounds.")
    x0 = Xq[inside][0]
    sp, xp, sm, xm = _decide_axis_sign(grad_phi, x0)
    print(f"  {sp}: {xp}\n  {sm}: {xm}")

    # quick φ angles to verify intuition
    phi0 = np.arctan2(x0[1], x0[0])
    phip = np.arctan2(xp[1], xp[0])
    phim = np.arctan2(xm[1], xm[0])

    # distances first (so they exist before Δ print)
    d_plus  = float(np.linalg.norm(xp - x0))
    d_minus = float(np.linalg.norm(xm - x0))

    pinfo(f"[AXIS-DIR] φ0={phi0:+.3f}, φ+= {phip:+.3f}, φ-= {phim:+.3f}")
    pinfo(f"[AXIS-DIR] Δ={abs(d_plus - d_minus):.3e}")

    # ---------- Choose a consistent toroidal direction ----------
    dir_sign = +1.0 if d_plus > d_minus else -1.0 if abs(d_plus - d_minus) > 1e-6 else 1.0
    pinfo(f"[AXIS-DIR] dir_sign={dir_sign:+.0f}  (d+= {d_plus:.3e}, d-= {d_minus:.3e})")

    # Rebuild evaluators so the multivalued toroidal term matches dir_sign
    a_flipped = np.array([float(dir_sign)*float(a[0]), float(a[1])], dtype=float)
    ev = Evaluators(center=jnp.asarray(center), scale=scale,
                    Yn=jnp.asarray(Yn), alpha=jnp.asarray(alpha),
                    a=jnp.asarray(a_flipped), a_hat=jnp.asarray(a_hat))
    phi_fn, grad_phi = ev.build()  # refresh with the signed MV piece

    # Axis seeds via collapsing random interior points along ±grad φ
    rng = np.random.default_rng(0)
    candidates = Xq[inside]
    if axis_seed_count <= 0:
        # Auto: ~2% of interior, clamped into [16, 128]
        axis_seed_count = int(np.clip(candidates.shape[0] // 50, 16, 128))
    axis_seed_count = min(axis_seed_count, max(16, candidates.shape[0]))
    picks = candidates[rng.choice(candidates.shape[0], size=axis_seed_count, replace=False)]
    axis_pts = np.array([
        collapse_to_axis(grad_phi, x0, step=0.1*min(dx,dy,dz),
                         iters=600, tol=1e-6, dir_sign=dir_sign)
        for x0 in picks
    ])
    # Thinning for stability/efficiency
    keepN = min(96, axis_pts.shape[0])
    axis_pts_ds = axis_pts[np.linspace(0, axis_pts.shape[0]-1, keepN, dtype=int)]

    # Interpret --axis-band-radius:
    #   0      -> auto (1.5 * voxel)
    #   (0,1)  -> fraction of bbox diagonal
    #   >=1    -> absolute length in world units
    voxel = min(dx, dy, dz)
    if axis_band_radius == 0.0:
        # auto: ~1.5 voxels wide tube (resolution-invariant)
        axis_band_radius_eff = 1.5 * voxel
    elif axis_band_radius < 1.0:
        # interpret as "multiples of voxel" (not fraction of bbox size)
        axis_band_radius_eff = max(1.0 * voxel, float(axis_band_radius) * voxel / 0.05)
        # if you want 0.05 to keep today's behavior, delete "/ 0.05"
    else:
        # absolute world units
        axis_band_radius_eff = max(1.0 * voxel, float(axis_band_radius))


    axis_band = axis_band_mask(axis_pts_ds, Xq, rad=axis_band_radius_eff) & inside
    pstat("Axis band fraction", axis_band.astype(float))

    # Evaluate grad φ everywhere inside (vectorized in chunks to save memory)
    pinfo("Evaluating ∇φ on grid (chunked)...")
    def eval_grad_chunk(Xchunk): return np.asarray(grad_phi(jnp.asarray(Xchunk)))
    G = np.zeros_like(Xq)
    chunk = 50000
    for s in range(0, Xq.shape[0], chunk):
        Gi = eval_grad_chunk(Xq[s:s+chunk])
        # NEW: sanitize bad values from evaluator
        bad = ~np.isfinite(Gi).all(axis=1)
        if np.any(bad):
            Gi[bad] = 0.0
        G[s:s+chunk] = Gi
    eps_t = 1e-12
    gnorm = np.linalg.norm(G, axis=1, keepdims=True)
    # NEW: explicit zeroing when ||∇φ|| is tiny
    t_hat = np.zeros_like(G)
    good = (gnorm > eps_t)[:, 0]
    t_hat[good] = (G[good] / gnorm[good])
    # D = diffusion_tensor(G, eps=eps, delta=1e-2)
    # Split tensor: D_perp only (parallel handled by FCI)
    delta_val = delta_mult * (dx*dx + dy*dy + dz*dz) / 3.0
    D_perp = diffusion_tensor_perp_world_diagonal(G, eps=eps, delta=delta_val)
    
    # report tiny-gradient fraction
    tiny = (gnorm[:,0] <= 1e-10)
    pinfo(f"[∇φ] tiny-norm fraction = {100.0*tiny.mean():.2f}%  (<=1e-10)")

    # OPTIONAL smoothing passes on t̂ to suppress noise-driven cross diffusion
    smooth_passes = 1  # try 1–2; set to 0 to disable
    if smooth_passes > 0:
        t_hat_sm = smooth_vec3_box(nx, ny, nz, t_hat, passes=smooth_passes)
        # where grad was tiny, keep zeros; elsewhere use the smoothed dir
        t_hat[~tiny] = t_hat_sm[~tiny]
        pinfo(f"[t̂] smoothed with {smooth_passes} pass(es)")
    
    # Fallback: build an axis band from low-|∇φ| skeleton if empty
    if not np.any(axis_band):
        pinfo("[AXIS] collapse-based axis band empty; using low-|∇φ| fallback.")
        gmag = np.linalg.norm(G, axis=1)
        gmag_inside = gmag[inside]
        if gmag_inside.size:
            thr = np.percentile(gmag_inside, 3.0 if gmag_inside.size > 500 else 1.0)
            cand = Xq[inside & (gmag <= thr)]
            # thin to ~2k points if very dense
            if cand.shape[0] > 2000:
                cand = cand[np.linspace(0, cand.shape[0]-1, 2000, dtype=int)]
            if cand.shape[0] >= 8:
                # cluster into a 1D tube; radius ~ 2 * voxel
                db = DBSCAN(eps=2.0*min(dx,dy,dz), min_samples=5).fit(cand)
                centers = []
                for lbl in set(db.labels_) - {-1}:
                    pts = cand[db.labels_ == lbl]
                    centers.append(pts.mean(axis=0))
                centers = np.array(centers) if len(centers) else cand
            else:
                centers = cand
            axis_band = axis_band_mask(centers, Xq, rad=axis_band_radius_eff) & inside
            pstat("[AXIS fallback] Axis band fraction", axis_band.astype(float))

    # Dxx = D[...,0,0]; Dyy = D[...,1,1]; Dzz = D[...,2,2]
    # Dxy = D[...,0,1]; Dxz = D[...,0,2]; Dyz = D[...,1,2]
    # pstat("Dxx", Dxx); pstat("Dyy", Dyy); pstat("Dzz", Dzz)
    # pstat("Dxy", Dxy); pstat("Dxz", Dxz); pstat("Dyz", Dyz)
    Dxx = D_perp[...,0,0]; Dyy = D_perp[...,1,1]; Dzz = D_perp[...,2,2]
    Dxy = D_perp[...,0,1]; Dxz = D_perp[...,0,2]; Dyz = D_perp[...,1,2]
    pstat("D⊥xx", Dxx); pstat("D⊥yy", Dyy); pstat("D⊥zz", Dzz)
    pstat("D⊥xy", Dxy); pstat("D⊥xz", Dxz); pstat("D⊥yz", Dyz)

    n_tot = Xq.shape[0]
    n_in  = int(inside.sum())
    n_bnd = int(band.sum())
    n_ax  = int(axis_band.sum())

    # For reporting only: fixed/free **inside** the domain
    fixed_report = np.zeros(n_tot, dtype=bool)
    fixed_report[band] = True
    fixed_report[axis_band] = True
    n_fixed = int((fixed_report & inside).sum())
    n_free  = int((~fixed_report & inside).sum())

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
    pinfo(f"[AXIS] axis_band_radius={axis_band_radius_eff:.3e}, "
        f"axis/inside={(n_ax/max(1,n_in))*100:.1f}% of interior")
    pinfo(f"[AXIS] seeds={axis_seed_count}, kept={axis_pts_ds.shape[0]} (thin tube)")
    
    # =================== Matrix-free + AMG PCG (replaces assembly) ===================
    pinfo("Preparing Dirichlet masks & values ...")
    Ntot = Xq.shape[0]
    
    fixed = np.zeros(Ntot, dtype=bool)
    val   = np.zeros(Ntot, dtype=float)
    # Dirichlet on boundary band and axis band
    fixed[band] = True;      val[band] = 1.0
    fixed[axis_band] = True; val[axis_band] = 0.0

    free = inside & (~fixed)
    if not np.any(free):
        raise RuntimeError("No free interior unknowns inside domain.")

    pinfo("Building matrix-free LinearOperator ...")
    # A_full = make_linear_operator(nx, ny, nz, dx, dy, dz, inside, D, fixed, val)
    # per-node "voxel length" along t-hat (Pythagorean in grid coords)
    hx = dx*np.abs(t_hat[:,0]); hy = dy*np.abs(t_hat[:,1]); hz = dz*np.abs(t_hat[:,2])
    h_par = np.maximum(1e-12, np.sqrt(hx*hx + hy*hy + hz*hz))
    # user knob; keep ~1 cell along field by default
    step_cells = float(fci_step_cells)
    def build_fci_with(step_cells_try):
        step_vec_try = np.maximum(1e-12, step_cells_try * np.sqrt((dx*np.abs(t_hat[:,0]))**2 +
                                                                   (dy*np.abs(t_hat[:,1]))**2 +
                                                                   (dz*np.abs(t_hat[:,2]))**2))
        base = min(dx, dy, dz)
        step_vec_try = np.minimum(step_vec_try, 0.8 * base)
        return precompute_fci(xs, ys, zs, Xq, t_hat, step_vec_try,
                              inside & (~band) & (~axis_band)), step_vec_try

    target_valid = 0.60  # 60% of nodes with valid +/- footpoints
    for attempt in range(8):
        fci, step_vec = build_fci_with(step_cells)
        valrate = 0.5*(np.mean(fci["in_p"]) + np.mean(fci["in_m"]))
        pinfo(f"[FCI] step_cells={step_cells:.3f} → valid ± average={100*valrate:.1f}%")
        if valrate >= target_valid or step_cells < 0.05:
            break
        step_cells *= 0.6  # shrink and retry
    step_cells = max(step_cells, 0.05)   # don’t let it get too tiny

    # FCI precompute (masked) using INSIDE to prevent cross-boundary leakage
    effective_inside = inside & (~band) & (~axis_band)
    # pass effective_inside to precompute_fci instead of inside
    # NEW: a one-voxel dilation just for FCI feasibility
    # dilate bands by one cell to avoid numerical mixing (simple pad + max)
    def dilate1(b):
        # cheap 6-neighbor dilation
        bb = b.copy()
        bb[1:,:,:] |= b[:-1,:,:]; bb[:-1,:,:] |= b[1:,:,:]
        bb[:,1:,:] |= b[:,:-1,:]; bb[:,:-1,:] |= b[:,1:,:]
        bb[:,:,1:] |= b[:,:,:-1]; bb[:,:,:-1] |= b[:,:,1:]
        return bb
    eff3 = effective_inside.reshape(nx,ny,nz)
    def dilate_k(b, k=2):
        bb = b.copy()
        for _ in range(k):
            bb = dilate1(bb)
        return bb
    eff3 = dilate_k(eff3, k=2)  # try 2 or 3
    effective_inside = eff3.ravel(order="C")

    fci = precompute_fci(xs, ys, zs, Xq, t_hat, step_vec, effective_inside, substeps_max=10)
    pinfo(f"[FCI] step_cells={step_cells:.2f}; "
        f"median step={np.median(step_vec):.3e}  min/max={step_vec.min():.3e}/{step_vec.max():.3e}")
    pinfo(f"[FCI] valid +footpoints: {100.0*np.mean(fci['in_p']):.1f}%   "
        f"valid -footpoints: {100.0*np.mean(fci['in_m']):.1f}%")
    
    # Cheap leakage probe: how often are the +/− footpoints crossing the boundary band?
    in_band = band
    cross_p = np.any(in_band[fci["idxp"]], axis=1)  # any of the 8 corners lies in band
    cross_m = np.any(in_band[fci["idxm"]], axis=1)
    pinfo(f"[FCI] crosses boundary band: + {100.0*np.mean(cross_p):.1f}% , - {100.0*np.mean(cross_m):.1f}%")
    
    bad = cross_p | cross_m
    fci["wp"][bad] *= 0.0; fci["wm"][bad] *= 0.0
    fci["in_p"][bad] = False; fci["in_m"][bad] = False

    # Report alignment per-decile (after solve we do a histogram already, this is pre-solve geometry)
    gcos = np.abs( (t_hat * (G/np.maximum(gnorm,1e-30))).sum(axis=1) )
    gcos = gcos[inside]
    for p in (50, 75, 90, 95, 99):
        print(f"[pre] |t̂·ĝ| perc {p:2d}: {np.percentile(gcos, p):.3e}")
    
    valrate = 0.5*(np.mean(fci["in_p"]) + np.mean(fci["in_m"]))
    alpha_par_eff = alpha_par * float(valrate)
    A_full = make_linear_operator(nx, ny, nz, dx, dy, dz, inside, D_perp, fixed, val,
                              fci=fci, alpha_par=alpha_par_eff)
    pinfo(f"[FCI] using alpha_par_eff={alpha_par_eff:.3f}")
    
    def Afree_matvec(x_f):
        x_full = np.array(val, copy=True); x_full[free] = x_f
        r_full = A_full @ x_full
        r_free = r_full[free]
        if not np.all(np.isfinite(r_free)):
            raise FloatingPointError("Non-finite residual: increase delta or fix D/inside.")
        return r_free

    # Solve only on free rows by wrapping A_full with a 'free-slice' operator:
    fidx = np.where(free)[0]
    def Afree_linear_matvec(x_f):
        # strictly linear operator: zero on fixed rows
        x_full = np.zeros_like(val)
        x_full[free] = x_f
        r_full = A_full @ x_full          # free-row residual = L_ff x_f (linear)
        return r_full[free]
    # Build linear operator (no constant term inside)
    A_free = LinearOperator((fidx.size, fidx.size), matvec=Afree_linear_matvec, rmatvec=Afree_linear_matvec, dtype=float)
    # Build RHS from fixed values ONCE:
    def residual_with_fixed_only():
        x_full = np.array(val, copy=True)  # fixed=val, free=0
        r_full = A_full @ x_full
        return r_full[free]                # this equals -b_free (by definition)
    b_free = -residual_with_fixed_only()
    
    # 1) Ones test: with axis band empty and boundary band at 1, a constant 1 should solve ⇒ A*1 == b
    if not np.any(axis_band):
        ones = np.ones_like(b_free)
        print("||A_free*1 - b_free|| / ||b_free|| =",
            np.linalg.norm((A_free @ ones) - b_free) / max(1e-16, np.linalg.norm(b_free)))

    # consistency of CSR vs matvec on the same vector
    # 2) CSR vs matvec: compare linear parts (no RHS)
    A_ff_csr, _, _ = build_aniso_csr_free(nx, ny, nz, dx, dy, dz, inside, fixed, D_perp, val)
    # Preconditioner on ⊥ block only
    A_full_no_fci = make_linear_operator(nx, ny, nz, dx, dy, dz, inside, D_perp, fixed, val, fci=None)
    def Afree_linear_matvec_no_fci(x_f):
        x_full = np.zeros_like(val)
        x_full[free] = x_f
        r_full = A_full_no_fci @ x_full
        return r_full[free]
    A_free_no_fci = LinearOperator((fidx.size, fidx.size), matvec=Afree_linear_matvec_no_fci,
                                rmatvec=Afree_linear_matvec_no_fci, dtype=float)
    psi_test = np.random.default_rng(2).standard_normal(fidx.size)
    print("||A_no_fci_matvec - A_csr|| / ||A_csr|| =",
        np.linalg.norm((A_free_no_fci @ psi_test) - (A_ff_csr @ psi_test)) /
        max(1e-16, np.linalg.norm(A_ff_csr @ psi_test)))

    # Check SPD-ish numerics
    eig_diag = np.min(np.stack([Dxx, Dyy, Dzz], axis=-1), axis=-1)
    assert np.all(np.isfinite(eig_diag))
    assert np.percentile(eig_diag, 0.01) > 1e-8  # after delta=1e-2 this will hold

    # 3) Symmetry probe now measures the true linear operator
    print("[A_free] antisymmetry probe =", probe_symmetry(A_free))
    
    _probe = Afree_matvec(np.zeros(fidx.size))
    if not np.all(np.isfinite(_probe)):
        pinfo("[Afree] Non-finite residual on zero vector; increase delta or eps and retry.")

    # Rayleigh quotient on free rows (crude λ_max/λ_min feel)
    z = np.random.default_rng(0).standard_normal(fidx.size)
    Az = A_free @ z
    rq = float(z @ Az) / max(1e-16, float(z @ z))
    pinfo(f"[Rayleigh ~⟨z,Az⟩/⟨z,z⟩] {rq:.3e}")
    
    # Diagnostics
    bnorm = float(np.linalg.norm(b_free))
    pinfo(f"[A_free] n_free={fidx.size}, ||b_free||_2={bnorm:.3e}")
    if not np.isfinite(bnorm) or bnorm == 0.0:
        pinfo("[WARN] b_free is zero or NaN; check bands/inside mask overlap and Dirichlet stamping.")

    # Preconditioner: AMG built on a cheap isotropic Laplacian over free nodes (setup once)
    try:
        pinfo("[PCG] Building AMG on the true anisotropic operator (free rows) ...")

        # --- Ensure CSR float64 and strictly positive diagonal
        A_ff_csr = A_ff_csr.astype(np.float64).tocsr(copy=True)
        d = A_ff_csr.diagonal().copy()
        neg_or_zero = d <= 0.0
        if np.any(neg_or_zero):
            A_ff_csr = A_ff_csr + coo_matrix(( (1e-12 - d[neg_or_zero]),
                                               (np.where(neg_or_zero)[0],
                                                np.where(neg_or_zero)[0]) ),
                                              shape=A_ff_csr.shape).tocsr()

        # --- Add simple ∥ diagonal to better match A_free (still SPD & symmetric)
        step_free = np.maximum(1e-30, fci["step"][free])  # (n_free,)
        diag_par  = 2.0 / (step_free**2)
        A_ff_csr_plus = A_ff_csr.copy()
        A_ff_csr_plus.setdiag(A_ff_csr_plus.diagonal() + diag_par)

        # --- Prefer SA; if it fails anywhere, try RS; else Jacobi
        ml = None
        try:
            from pyamg import smoothed_aggregation_solver as SA
            ml = SA(A_ff_csr_plus, max_levels=20, max_coarse=200)
            which = "SA"
        except Exception as e1:
            pinfo(f"[PCG] SA failed: {e1}  → trying Ruge–Stüben")
            from pyamg import ruge_stuben_solver as RS
            ml = RS(A_ff_csr_plus, max_levels=20, max_coarse=200)
            which = "RS"

        M_par = ml.aspreconditioner()

        # Symmetric Jacobi scaling around AMG
        Dfine = np.maximum(1e-12, np.asarray(ml.levels[0].A.diagonal()))
        Dm12  = 1.0/np.sqrt(Dfine)
        def M_apply(x):
            y = Dm12 * x
            y = M_par @ y
            y = Dm12 * y
            return y

        M = LinearOperator((b_free.size, b_free.size), matvec=M_apply, rmatvec=M_apply, dtype=float)
        # Sanity check
        ytest = M @ np.ones_like(b_free)
        if not np.all(np.isfinite(ytest)):
            raise RuntimeError("AMG preconditioner produced non-finite values.")
        pinfo(f"[PCG] {which} AMG preconditioner ready")
    except Exception as e:
        pinfo(f"[PCG] Using Jacobi preconditioner ({e})")
        diag_approx = np.maximum(1e-8, np.ones_like(b_free))
        def M_apply(x): return x / diag_approx
        M = LinearOperator((b_free.size, b_free.size), matvec=M_apply, rmatvec=M_apply, dtype=float)
        
    # Initial guess: M @ b_free
    x0_f = M @ b_free   # way better than zeros on hard cases
    if not np.all(np.isfinite(x0_f)):
        pinfo("[Init] Non-finite x0 from preconditioner; falling back to zeros.")
        x0_f = np.zeros_like(b_free)
    
    pinfo("CG solve (matrix-free, SPD) ...")
    t0 = time.time()
    
    # psi_f, info = cg(A_free, b_free, rtol=cg_tol, atol=0.0, maxiter=cg_maxit, M=M, x0=x0_f)
    psi_f = None
    x0_f  = M @ b_free if 'M' in locals() else np.zeros_like(b_free)  # initial warm start
    for e in eps_schedule(eps):                   # e.g. [0.3, 0.12, final]
        D_perp = diffusion_tensor_perp_world_diagonal(G, eps=e, delta=delta_val)

        # *** keep the same alpha_par_eff you computed earlier ***
        A_full = make_linear_operator(
            nx, ny, nz, dx, dy, dz,
            inside, D_perp, fixed, val,
            fci=fci, alpha_par=alpha_par_eff
        )

        free = inside & (~fixed)                 # (unchanged, but keep local)
        fidx = np.where(free)[0]

        def Afree_linear_matvec(x_f):
            x_full = np.zeros_like(val); x_full[free] = x_f
            return (A_full @ x_full)[free]

        A_free = LinearOperator(
            (fidx.size, fidx.size),
            matvec=Afree_linear_matvec,
            rmatvec=Afree_linear_matvec,
            dtype=float
        )

        # *** RHS MUST be rebuilt whenever A_full changes ***
        def residual_with_fixed_only():
            x_full = np.array(val, copy=True)    # fixed rows = val, free = 0
            return (A_full @ x_full)[free]
        b_free = -residual_with_fixed_only()

        # warm start
        if psi_f is not None:
            x0_f = psi_f

        psi_f, info = cg(
            A_free, b_free,
            rtol=cg_tol, atol=0.0, maxiter=cg_maxit,
            M=M, x0=x0_f
        )
    
    if (not np.all(np.isfinite(psi_f))) or (info > 0):
        pinfo(f"[CG] status={info}. Retrying CG with Jacobi (symmetric) preconditioner.")
        # symmetric Jacobi:
        diag_j = np.maximum(1e-10, np.ones_like(b_free))
        M_jac = LinearOperator((b_free.size, b_free.size),
                            matvec=lambda v: v/diag_j,
                            rmatvec=lambda v: v/diag_j, dtype=float)
        x0_f2 = M_jac @ b_free
        psi_f, info = cg(A_free, b_free, rtol=cg_tol, atol=0.0, maxiter=cg_maxit, M=M_jac, x0=x0_f2)
        if (not np.all(np.isfinite(psi_f))) or (info > 0):
            pinfo(f"[CG] still struggling (status={info}). MINRES (no/nice M).")
            # MINRES requires symmetric A and symmetric positive (semi)definite M.
            # Use none or Jacobi only:
            psi_f, info = minres(A_free, b_free, rtol=cg_tol, maxiter=cg_maxit,
                                M=M_jac, x0=x0_f2)
    t1 = time.time()
    pinfo(f"CG done: info={info}, iters≈{cg_maxit if info>0 else 'converged'} , wall={t1-t0:.2f}s")

    print("[A_free] antisymmetry probe (final) =", probe_symmetry(A_free))
        
    def sym_check(A, n=5, seed=0):
        rng = np.random.default_rng(seed)
        s = []
        for _ in range(n):
            x = rng.standard_normal(A.shape[0]); y = rng.standard_normal(A.shape[0])
            Ax, Ay = A @ x, A @ y
            num = abs(x @ Ay - y @ Ax)
            den = np.linalg.norm(Ax)*np.linalg.norm(y) + np.linalg.norm(Ay)*np.linalg.norm(x) + 1e-16
            s.append(num / den)
        return max(s)
    print("[A_free] sym_check =", sym_check(A_free))

    # Scatter back to full vector
    psi = np.array(val)    # Dirichlet in place
    psi[free] = psi_f

    # Optional: full-operator residual should be ~0 everywhere (incl. fixed rows)
    r_full = A_full @ psi
    pstat("Residual Aψ (full)", r_full)
        
    # True reduced residual:
    r_free_full = (A_full @ psi)[free]
    print("[RES] ||r_free_full||/||b_free|| =",
        np.linalg.norm(r_free_full) / max(1e-16, np.linalg.norm(b_free)))

    r_reduced = (A_free @ psi_f) - b_free
    pinfo(f"[RES(reduced)] {np.linalg.norm(r_reduced)/max(1e-16, np.linalg.norm(b_free)):.3e}")
    
    # Normalize ψ on the inside so [0,1] roughly spans interior
    psi_in = psi[inside]
    p01, p50, p99 = np.percentile(psi_in, [1, 50, 99])
    scale = max(p99 - p01, 1e-12)
    psi_n = np.clip((psi - p01)/scale, 0.0, 1.0)
    psi0_n = np.clip((psi0 - p01)/scale, 0.0, 1.0)

    pinfo(f"[ψ] inside percentiles: p1={p01:.3e}, p50={p50:.3e}, p99={p99:.3e}; using normalized ψ in [0,1]")

    levels = []
    if psi_levels is not None and psi_levels.strip() != "":
        levels = [float(s) for s in psi_levels.split(",") if s.strip()]
    else:
        levels = [psi0_n]

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

    # Split residual sanity (free vs fixed rows)
    free_mask = inside & (~fixed)
    pstat("Residual (free rows)", r_full[free_mask])
    pstat("Residual (fixed rows)", r_full[fixed])

    # Crude conditioning proxy from the AMG proxy Laplacian (finest-level diag)
    try:
        Lfine = ml.levels[0].A  # pyamg hierarchy's finest operator
        diag = np.abs(Lfine.diagonal())
        cond_est = np.max(diag) / max(1e-16, np.min(diag))
        pinfo(f"[AMG proxy] crude diag ratio ~ {cond_est:.2e}")
    except Exception as _:
        pinfo("[AMG proxy] could not estimate diag ratio")
        
    # ------------------------------ Plots ------------------------------ #
    if plot:
        # 3D isosurfaces
        # Compute grad φ on a stride of boundary points and quiver
        stride = max(1, P.shape[0] // 1200)   # keep it light
        Pb = P[::stride]
        Gb = np.asarray(grad_phi(jnp.asarray(Pb)))
        # scale arrows by a visually reasonable factor
        gscale = 0.03 * float(np.linalg.norm(maxs - mins))

        fig = plt.figure(figsize=(14,6))
        ax = fig.add_subplot(1,2,1, projection='3d')
        # take only inside region
        # 3D isosurfaces (normalized ψ)
        vol = psi_n.reshape(nx,ny,nz).copy()
        vol[~inside.reshape(nx,ny,nz)] = np.nan
        for lv in [0.1, 0.5]:
            try:
                vol = psi_n.reshape(nx,ny,nz)
                mask = inside.reshape(nx,ny,nz)
                verts, faces, _, _ = marching_cubes(np.nan_to_num(vol, nan=-1.0), level=lv, mask=mask)
                vx = mins[0] + verts[:,0]*(maxs[0]-mins[0])/(nx-1)
                vy = mins[1] + verts[:,1]*(maxs[1]-mins[1])/(ny-1)
                vz = mins[2] + verts[:,2]*(maxs[2]-mins[2])/(nz-1)
                ax.plot_trisurf(vx, vy, vz, triangles=faces, alpha=0.35, linewidth=0.1)
            except Exception as e:
                pinfo(f"Marching cubes failed at level {lv}: {e}")
        ax.scatter(Pb[:,0], Pb[:,1], Pb[:,2], s=2, c='k', alpha=0.15)
        ax.quiver(Pb[:,0], Pb[:,1], Pb[:,2], Gb[:,0], Gb[:,1], Gb[:,2],
                    length=gscale, normalize=True, linewidth=0.5, color='tab:red', alpha=0.3)
        ax.set_title("Isosurfaces of ψ (0.1, 0.5) + boundary points")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        fix_matplotlib_3d(ax)

        # Quality dashboard: |t·∇ψ| / |∇ψ|  (dimensionless in [0,1])
        ax2 = fig.add_subplot(1,2,2)
        # center differences already computed: dpsidx, dpsidy, dpsidz on the (nx-2,ny-2,nz-2) core
        grad_mag = np.sqrt(dpsidx**2 + dpsidy**2 + dpsidz**2)
        par_core = (t_hat_x*dpsidx + t_hat_y*dpsidy + t_hat_z*dpsidz)  # same shape
        with np.errstate(divide='ignore', invalid='ignore'):
            q = np.abs(par_core) / np.maximum(grad_mag, 1e-14)
            
        qv = np.abs(par_core)[mask_mid].ravel() / np.maximum(grad_mag[mask_mid].ravel(), 1e-14)
        for p in (50, 75, 90, 95, 99):
            print(f"q perc {p:2d}: {np.percentile(qv, p):.3e}")
            
        # Per-shell (R bins) median of alignment in the core
        Rmid3 = np.sqrt(XX[1:-1,1:-1,1:-1]**2 + YY[1:-1,1:-1,1:-1]**2)              # keep 3-D
        rm    = Rmid3[mask_mid]                                                     # → 1-D
        qcore = q[mask_mid]                                                         # → 1-D
        if qcore.size:
            bins = np.quantile(rm, [0.0, 0.25, 0.5, 0.75, 1.0])
            for b0, b1 in zip(bins[:-1], bins[1:]):
                sel = (rm >= b0) & (rm < b1)
                if sel.any():
                    print(f"[core] R∈[{b0:.3f},{b1:.3f})  median(q)={np.median(qcore[sel]):.3e}  n={sel.sum()}")
        if qcore.size:
            bins = np.quantile(rm, [0.0, 0.25, 0.5, 0.75, 1.0])
            for b0, b1 in zip(bins[:-1], bins[1:]):
                sel = (rm>=b0) & (rm<b1)
                if sel.any():
                    print(f"[core] R∈[{b0:.3f},{b1:.3f})  median(q)={np.median(qcore[sel]):.3e}  n={sel.sum()}")
            
        # sample only meaningful interior (mask_mid)
        qv = q[mask_mid].ravel()
        ax2.hist(qv, bins=60, alpha=0.8)
        ax2.set_yscale('log')
        ax2.set_xlabel(r"$|\,\hat t\cdot\nabla\psi\,|/|\nabla\psi|$")
        ax2.set_title("Field-alignment metric (smaller is better)")
        plt.tight_layout()
        if save_figures:
            fig.savefig("solve_flux_3d_iso_and_quality.png", dpi=150)

        # Residual scatter & slices
        fig2, axa = plt.subplots(1,3, figsize=(14,4))
        r3 = r_full.reshape(nx,ny,nz)
        im0 = axa[0].imshow(r3[:,:,nz//2].T, origin='lower', aspect='equal'); plt.colorbar(im0, ax=axa[0]); axa[0].set_title("residual @ z mid")
        im1 = axa[1].imshow(r3[:,ny//2,:].T, origin='lower', aspect='equal'); plt.colorbar(im1, ax=axa[1]); axa[1].set_title("residual @ y mid")
        im2 = axa[2].imshow(r3[nx//2,:,:].T, origin='lower', aspect='equal'); plt.colorbar(im2, ax=axa[2]); axa[2].set_title("residual @ x mid")
        plt.tight_layout()
        if save_figures:
            fig2.savefig("solve_flux_residual_slices.png", dpi=150)

        # --------- Poincaré-like cross-sections of iso-surface ψ=psi0 ----------
        # --- Build iso-surface once (normalized ψ) ---
        iso_cache = []
        for lv in levels:
            try:
                Viso, Fiso = march_iso_surface(psi_n, inside, nx, ny, nz, mins, maxs, level=lv)
                iso_cache.append((lv, Viso, Fiso))
            except Exception as e:
                pinfo(f"Marching cubes failed at level {lv:.2f}: {e}")
                iso_cache.append((lv, None, None))

        phi_list = jnp.linspace(0, 2*jnp.pi/nfp, 4, endpoint=False).tolist()
        figp, axes = plt.subplots(1, 4, figsize=(14,7), constrained_layout=True)
        axes = axes.ravel()

        for aidx, phi0 in enumerate(phi_list):
            axp = axes[aidx]
            # boundary curve (see B below) before plotting cuts:
            Rb, Zb = boundary_curve_RZ(P, phi0)
            if Rb.size:
                axp.scatter(Rb, Zb, s=4, alpha=0.8, edgecolors='none')  # boundary first

            for (lv, Viso, Fiso) in iso_cache:
                if Viso is None:
                    continue
                segs = intersect_iso_with_phi_plane(Viso, Fiso, phi0)
                # stitch segments into polylines; tolerance ~ voxel scale
                tol = 0.5 * min(dx, dy, dz)
                polys = segments_to_polylines(segs, tol=tol)
                nseg = draw_poincare_from_polylines(axp, polys)
            axp.set_aspect('equal', 'box')
            axp.set_xlabel("R"); axp.set_ylabel("Z")
            axp.set_title(f"φ={phi0:+.2f}")


        for k in range(len(phi_list), len(axes)):
            axes[k].axis('off')
        
        if save_figures:
            figp.savefig("solve_flux_poincare_cuts.png", dpi=150)
        
        # --- Interpolator for ψ and for 'inside' mask (nearest for mask) ---
        psi3 = psi.reshape(nx, ny, nz)        # (nx,ny,nz)
        inside3 = inside.reshape(nx, ny, nz)  # (nx,ny,nz)

        psi_interp = RGI((xs, ys, zs), psi3, bounds_error=False, fill_value=np.nan)
        ins_interp = RGI((xs, ys, zs), inside3.astype(float), method="nearest",
                        bounds_error=False, fill_value=0.0)

        figrz, axrz = plt.subplots(1, 4, figsize=(14,5), constrained_layout=True)
        axrz = axrz.ravel()

        for aidx, phi0 in enumerate(phi_list):
            axp = axrz[aidx]
            (Rlo,Rhi), (Zlo,Zhi) = rz_box_for_phi(P, phi0, pad=0.08)

            # build a modest R–Z sampling grid
            nR, nZ = 240, 160
            Rg = np.linspace(Rlo, Rhi, nR)
            Zg = np.linspace(Zlo, Zhi, nZ)
            RR, ZZ = np.meshgrid(Rg, Zg, indexing="xy")

            # map to Cartesian at this φ-plane
            c, s = np.cos(phi0), np.sin(phi0)
            XXp = RR*c
            YYp = RR*s
            ZZp = ZZ

            pts = np.column_stack([XXp.ravel(), YYp.ravel(), ZZp.ravel()])
            Psi_plane = psi_interp(pts).reshape(nR, nZ).T
            Ins_plane = ins_interp(pts).reshape(nR, nZ).T > 0.5  # boolean

            # mask outside-of-domain pixels
            Psi_plane_masked = np.ma.array(Psi_plane, mask=~Ins_plane)

            im = axp.imshow(Psi_plane_masked,
                            origin="lower", aspect="equal",
                            extent=[Rlo, Rhi, Zlo, Zhi])
            # overlay a few ψ contours to aid reading
            try:
                cs = axp.contour(Rg, Zg, Psi_plane_masked, levels=[p50, p01, p99], linewidths=0.8)
                axp.clabel(cs, fmt="ψ=%.2f", inline=True, fontsize=8)
            except Exception:
                pass

            # also overlay boundary points projected in this plane for reference
            Rb, Zb = boundary_curve_RZ(P, phi0)
            axp.plot(Rb, Zb, '.', ms=2, alpha=0.6, color='k')

            axp.set_title(f"ψ(R,Z) @ φ={phi0:+.2f}")
            axp.set_xlabel("R"); axp.set_ylabel("Z")

        figrz.colorbar(im, ax=axrz.tolist(), shrink=0.85, pad=0.02)
        if save_figures:
            figrz.savefig("solve_flux_maps_RZ_planes.png", dpi=150)
        
        plt.show()
        
        if save_figures:
            np.savez("flux_solution.npz", psi=psi, xs=xs, ys=ys, zs=zs, inside=inside)

    return dict(psi=psi, grid=(xs,ys,zs), inside=inside, quality=dict(parallel_dot_grad=par, residual=r_full))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default="wout_precise_QA_solution.npz",
                    help="MFS solution checkpoint (*.npz) containing center, scale, Yn, alpha, a, a_hat, P, N")
    ap.add_argument("--nfp", type=int, default=2, help="number of field periods (for plotting)")
    ap.add_argument("--N", type=int, default=32, help="grid resolution per axis")
    ap.add_argument("--eps", type=float, default=0.12, help="perpendicular diffusion coefficient (smaller ⇒ more field-aligned)")
    ap.add_argument("--band-h", type=float, default=0.25, help="boundary band thickness multiplier")
    ap.add_argument("--cg-maxit", type=int, default=1000)
    ap.add_argument("--axis-seed-count", type=int, default=0,
                    help="number of interior seeds to collapse onto axis; 0 = auto (~2% of interior, clamped to [16,128])")
    ap.add_argument("--axis-band-radius", type=float, default=0,
                    help="axis band radius: 0=auto (1.5*voxel); (0,1)=fraction of bbox diagonal; >=1=absolute units")
    ap.add_argument("--cg-tol", type=float, default=1e-8)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--psi0", type=float, default=0.1, help="iso-surface level for plotting")
    ap.add_argument("--psi-levels", type=str, default="0.1,0.3,0.5,0.8",
                    help="comma-separated normalized levels for iso-surface cuts (e.g. '0.2,0.5,0.8'). If empty, use --psi0 only.")
    ap.add_argument("--save-figures", action="store_true", default=True, help="save figures instead of showing interactively")
    ap.add_argument("--fci-step-cells", type=float, default=0.4,
                    help="FCI step length along t̂ in multiples of the smallest voxel size")
    ap.add_argument("--alpha-par", type=float, default=1.0, help="scale factor multiplying the parallel (FCI) operator")
    ap.add_argument("--delta-mult", type=float, default=1e-2, help="regularization delta multiplier (larger ⇒ more stable, less anisotropic)")
    args = ap.parse_args()
    out = main(args.npz, grid_N=args.N, eps=args.eps, band_h=args.band_h,
               axis_seed_count=args.axis_seed_count, axis_band_radius=args.axis_band_radius,
               cg_tol=args.cg_tol, cg_maxit=args.cg_maxit, plot=(not args.no_plot), psi0=args.psi0,
               nfp=args.nfp, psi_levels=args.psi_levels, save_figures=args.save_figures,
               fci_step_cells=args.fci_step_cells, alpha_par=args.alpha_par, delta_mult=args.delta_mult)
