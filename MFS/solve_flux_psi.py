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
from matplotlib.patches import Rectangle
import itertools

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

def stat01(v): 
    v = np.asarray(v)
    p = np.percentile(v, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(f"[ψ stats] min={p[0]:.3e} p1={p[1]:.3e} p5={p[2]:.3e} "
          f"p50={p[4]:.3e} p95={p[6]:.3e} p99={p[7]:.3e} max={p[8]:.3e}")

def frac_above(v, thrs=(1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3)):
    v = np.asarray(v); m = v.size
    print("[ψ volume fractions > τ]")
    for τ in thrs:
        print(f"  τ={τ:g}: {np.count_nonzero(v>τ)/max(1,m):.3%}")

def per_phi_percentiles(psi3, inside3):
    # psi3: (NR, Nphi, NZ) for torus; inside3 same shape (bool)
    NR, Nphi, NZ = psi3.shape
    p50 = np.zeros(Nphi); p90 = np.zeros(Nphi); p99 = np.zeros(Nphi)
    for j in range(Nphi):
        w = psi3[:, j, :][inside3[:, j, :]]
        if w.size:
            p50[j] = np.percentile(w, 50)
            p90[j] = np.percentile(w, 90)
            p99[j] = np.percentile(w, 99)
    print("[ψ per-φ] med@φ[0:4]=", p50[:4], "  p99@φ[0:4]=", p99[:4])
    print(f"[ψ per-φ] global med({np.nanmedian(p50):.3e})  p99({np.nanmedian(p99):.3e})")


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
    # smooth taper 0→1 as |∇φ| grows: s≈0 → isotropic, s≈1 → anisotropic
    n0 = 5e-3 * np.nanmax(n) + 1e-30
    s  = (n / n0); s = np.clip(s, 0.0, 1.0)            # linear is sufficient here
    t = np.divide(gradphi, np.maximum(n, 1e-30), out=np.zeros_like(gradphi))  # (N,3)

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

    Daniso = R @ Lam @ np.swapaxes(R, -1, -2)                   # anisotropic part
    Diso   = np.eye(3)[None, :, :]                              # isotropic fallback
    D = (s[..., None] * Daniso) + ((1.0 - s[..., None]) * Diso) + delta * I
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

def build_cyl_grid(P, nfp, NR, NZ, Nphi):
    # tight (R,Z) box from surface
    Rb = np.sqrt(P[:,0]**2 + P[:,1]**2)
    Rmin, Rmax = float(Rb.min()), float(Rb.max())
    Zmin, Zmax = float(P[:,2].min()), float(P[:,2].max())
    padR = 0.03*(Rmax-Rmin + 1e-12); padZ = 0.03*(Zmax-Zmin + 1e-12)
    Rmin -= padR; Rmax += padR; Zmin -= padZ; Zmax += padZ

    Rs   = np.linspace(Rmin, Rmax, NR)
    Zs   = np.linspace(Zmin, Zmax, NZ)
    phis = np.linspace(0.0, 2*np.pi/nfp, Nphi, endpoint=False)

    RR, PHI, ZZ = np.meshgrid(Rs, phis, Zs, indexing="ij")  # (NR, Nphi, NZ)
    XX = RR*np.cos(PHI); YY = RR*np.sin(PHI)

    Xq = np.column_stack([XX.ravel(order="C"),
                          YY.ravel(order="C"),
                          ZZ.ravel(order="C")])
    dR = float(Rs[1]-Rs[0]); dZ = float(Zs[1]-Zs[0]); dphi = float(phis[1]-phis[0]) if Nphi>1 else 2*np.pi/nfp
    return (Rs, phis, Zs), (dR, dphi, dZ), Xq

def _face_k_cyl(D3, PHI3):
    """
    Return face-normal 'conductivities' at faces via n^T D n where:
      n_R = e_R(φ), n_phi = e_phi(φ), n_Z = e_Z.
    Shapes:
      D3: (NR, Nphi, NZ, 3, 3)
      PHI3: (NR, Nphi, NZ)
    Outputs:
      kR at R-faces: shape (NR-1, Nphi, NZ)   (harmonic avg across the face)
      kphi at φ-faces: (NR, Nphi, NZ) but will use periodic neighbor j±1
      kZ at Z-faces: (NR, Nphi, NZ-1)
    """
    c = np.cos(PHI3); s = np.sin(PHI3)
    eR   = np.stack([c, s, 0*PHI3], axis=-1)      # (...,3)
    ephi = np.stack([-s, c, 0*PHI3], axis=-1)
    eZ   = np.zeros_like(eR); eZ[...,2] = 1.0

    def quad(D, n):  # n^T D n
        tmp = np.einsum("...ij,...j->...i", D, n)
        return np.sum(n*tmp, axis=-1)

    kR_cell   = quad(D3, eR)     # (NR, Nphi, NZ)
    kphi_cell = quad(D3, ephi)
    kZ_cell   = quad(D3, eZ)

    # harmonic average to faces in R and Z
    kR_face = _harmonic(kR_cell[1:,:,:], kR_cell[:-1,:,:])   # (NR-1, Nphi, NZ)
    kZ_face = _harmonic(kZ_cell[:,:,1:], kZ_cell[:,:,:-1])   # (NR, Nphi, NZ-1)

    # φ uses cell-centered k with periodic neighbors; simple arithmetic mean is OK
    # kphi_face = 0.5*(kphi_cell + np.roll(kphi_cell, shift=-1, axis=1))  # periodic in φ
    def _harm(a,b,eps=1e-30): return 2*a*b/np.maximum(a+b, eps)
    kphi_face = _harm(kphi_cell, np.roll(kphi_cell, -1, axis=1))
    return kR_face, kphi_face, kZ_face

def make_linear_operator_cyl(Rs, phis, Zs, dR, dphi, dZ, inside, Dfield,
                             fixed_mask=None, fixed_val=None):
    """
    Matrix-free LinearOperator for cylindrical FV with metric factors.
    Grid is (NR, Nphi, NZ) in (R,φ,Z).
    inside: boolean mask on full nodes (NR*Nphi*NZ, order='C' with (i,j,k))
    Dfield: (NR*Nphi*NZ, 3, 3) in world coords.
    """
    NR, Nphi, NZ = len(Rs), len(phis), len(Zs)
    inside_3 = inside.reshape(NR, Nphi, NZ)
    D3 = Dfield.reshape(NR, Nphi, NZ, 3, 3)
    PHI3 = np.broadcast_to(phis[None, :, None], (NR, Nphi, NZ))

    kR_face, kphi_face, kZ_face = _face_k_cyl(D3, PHI3)

    if fixed_mask is None:
        fixed_mask = np.zeros(NR*Nphi*NZ, dtype=bool)
    if fixed_val is None:
        fixed_val = np.zeros(NR*Nphi*NZ, dtype=float)
    rows_fixed = np.where(fixed_mask)[0]
    fixed_3 = fixed_mask.reshape(NR, Nphi, NZ)

    # --- IMPORTANT: face masks depend ONLY on "inside", not on "fixed/free" ---
    mR   = (inside_3[1:, :, :] & inside_3[:-1, :, :])          # (NR-1, Nphi, NZ)
    mphi = (inside_3 & np.roll(inside_3, -1, axis=1))           # (NR,   Nphi, NZ)
    mZ   = (inside_3[:, :, 1:] & inside_3[:, :, :-1])          # (NR,   Nphi, NZ-1)

    R3 = np.broadcast_to(Rs[:, None, None], (NR, Nphi, NZ))
    Rphi_face = 0.5 * (R3 + np.roll(R3, -1, axis=1))         # (NR, Nphi, NZ)
    Aphi = dR * dZ

    def matvec(u):
        U = u.reshape(NR, Nphi, NZ)
        # cell volumes
        V  = R3 * dR * dphi * dZ

        # R faces
        Rf = 0.5*(np.broadcast_to(Rs[:-1, None, None], (NR-1, Nphi, NZ)) +
                  np.broadcast_to(Rs[1:,  None, None], (NR-1, Nphi, NZ)))
        AR = Rf * dphi * dZ
        dU_R_lo = np.zeros_like(kR_face)
        dU_R_lo[mR] = (U[1:, :, :] - U[:-1, :, :])[mR] / dR
        FR = np.zeros_like(kR_face)
        FR[mR] = kR_face[mR] * AR[mR] * dU_R_lo[mR]

        # φ faces (periodic, shared metric)
        dU_phi = np.roll(U, -1, axis=1) - U
        Gphi   = np.zeros_like(dU_phi)
        Gphi[mphi] = dU_phi[mphi] / (Rphi_face[mphi] * dphi)
        Fphi   = np.zeros_like(dU_phi)
        Fphi[mphi] = kphi_face[mphi] * Aphi * Gphi[mphi]
        Fphi_bwd = np.roll(Fphi, +1, axis=1)

        # Z faces
        AZ = R3 * dR * dphi
        dU_Z_lo = np.zeros_like(kZ_face)
        dU_Z_lo[mZ] = (U[:, :, 1:] - U[:, :, :-1])[mZ] / dZ
        FZ = np.zeros_like(kZ_face)
        FZ[mZ] = kZ_face[mZ] * AZ[:, :, :-1][mZ] * dU_Z_lo[mZ]

        # divergence on inside cells
        div = np.zeros_like(U)

        # R
        div[1:-1, :, :] += (FR[1:, :, :] - FR[:-1, :, :]) / V[1:-1, :, :]

        # φ
        div += (Fphi - Fphi_bwd) / V

        # Z
        div[:, :, 1:-1] += (FZ[:, :, 1:] - FZ[:, :, :-1]) / V[:, :, 1:-1]

        # zero rows outside domain
        div[~inside_3] = 0.0

        out = (-div).ravel(order="C")
        # Dirichlet rows (stamp after divergence so neighbors still see flux)
        out[rows_fixed] = u[rows_fixed] - fixed_val[rows_fixed]
        return out

    N = NR * Nphi * NZ
    return LinearOperator((N, N), matvec=matvec, rmatvec=matvec, dtype=float)



def build_aniso_csr_free_cyl(Rs, phis, Zs, dR, dphi, dZ, inside, fixed, Dfield, val):
    NR, Nphi, NZ = len(Rs), len(phis), len(Zs)
    inside_3 = inside.reshape(NR, Nphi, NZ)
    fixed_1d = fixed.ravel(order="C")
    val_1d   = val.ravel(order="C")
    D3 = Dfield.reshape(NR, Nphi, NZ, 3, 3)

    # n(φ) directions
    PHI3 = np.broadcast_to(phis[None,:,None], (NR, Nphi, NZ))
    kR_face, kphi_face, kZ_face = _face_k_cyl(D3, PHI3)

    idx = np.arange(NR*Nphi*NZ).reshape(NR, Nphi, NZ)
    rows, cols, vals = [], [], []
    b = np.zeros(NR*Nphi*NZ, dtype=float)

    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    # geometric measures
    R3 = np.broadcast_to(Rs[:,None,None], (NR, Nphi, NZ))
    V  = R3 * dR * dphi * dZ

    # R-face areas at i±1/2
    Rf = 0.5*(np.broadcast_to(Rs[:-1,None,None], (NR-1, Nphi, NZ)) +
              np.broadcast_to(Rs[1:, None,None], (NR-1, Nphi, NZ)))
    AR = Rf * dphi * dZ
    Aphi = dR * dZ
    AZ = R3 * dR * dphi
    Rphi_face = 0.5 * (R3 + np.roll(R3, -1, axis=1))            # face metric

    for i in range(NR):
      for j in range(Nphi):
        for k in range(NZ):
          if not inside_3[i,j,k]: 
              continue
          p = idx[i,j,k]
          if fixed_1d[p]:
              continue

          diag = 0.0

          # R- neighbor (i-1)
          # NOTE: all K below are divided by local cell volume V[i,j,k]
          # so this CSR matches the matrix-free divergence form.
          # R- neighbor (i-1)
          if i > 0 and inside_3[i-1,j,k]:
              q = idx[i-1,j,k]
              K = (kR_face[i-1,j,k] * AR[i-1,j,k] / dR) / V[i,j,k]
              if fixed_1d[q]: b[p] += K * val_1d[q]
              else: add(p, q, -K)
              diag += K

          # R+ neighbor (i+1)
          if i+1 < NR and inside_3[i+1,j,k]:
              q = idx[i+1,j,k]
              K = (kR_face[i,j,k] * AR[i,j,k] / dR) / V[i,j,k]
              if fixed_1d[q]: b[p] += K * val_1d[q]
              else: add(p, q, -K)
              diag += K

          # φ- neighbor (j-1) periodic  --> use conductivity on the *backward* face (j-1)
          jm = (j-1) % Nphi
          if inside_3[i,jm,k]:
              q = idx[i,jm,k]
              K = (kphi_face[i,jm,k] * Aphi / (Rphi_face[i,jm,k]*dphi)) / V[i,j,k]
              if fixed_1d[q]: b[p] += K * val_1d[q]
              else: add(p, q, -K)
              diag += K

          # φ+ neighbor (j+1) periodic  --> use conductivity on the *forward* face (j)
          jp = (j + 1) % Nphi
          if inside_3[i,jp,k]:
              q = idx[i,jp,k]
              K = (kphi_face[i,j,k] * Aphi / (Rphi_face[i,j,k]*dphi)) / V[i,j,k]
              if fixed_1d[q]: b[p] += K * val_1d[q]
              else: add(p, q, -K)
              diag += K

          # Z- neighbor (k-1)
          if k > 0 and inside_3[i,j,k-1]:
              q = idx[i,j,k-1]
              K = (kZ_face[i,j,k-1] * AZ[i,j,k-1] / dZ) / V[i,j,k]
              if fixed_1d[q]: b[p] += K * val_1d[q]
              else: add(p, q, -K)
              diag += K

          # Z+ neighbor (k+1)
          if k+1 < NZ and inside_3[i,j,k+1]:
              q = idx[i,j,k+1]
              K = (kZ_face[i,j,k] * AZ[i,j,k] / dZ) / V[i,j,k]
              if fixed_1d[q]: b[p] += K * val_1d[q]
              else: add(p, q, -K)
              diag += K

          # diagonal
          add(p, p, diag)

    from scipy.sparse import coo_matrix
    Ntot = NR*Nphi*NZ
    A = coo_matrix((vals, (rows, cols)), shape=(Ntot, Ntot)).tocsr()
    # slice to "free" rows/cols just like the Cartesian path
    free = (inside) & (~fixed)
    fidx = np.where(free)[0]
    A_ff = A[free][:, free].tocsr()
    b_f  = b[free]
    return A_ff, b_f, fidx


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

def make_linear_operator(nx, ny, nz, dx, dy, dz, inside, Dfield, fixed_mask, fixed_val):
    """
    Build a LinearOperator A_full such that A_full @ u_full returns
    the residual of the stamped system:
      - free rows: div(D grad u)
      - fixed rows: u - fixed_val
    All vectors are full-sized (N=nx*ny*nz).
    """
    inside_3 = inside.reshape(nx,ny,nz)
    D3 = Dfield.reshape(nx,ny,nz,3,3)

    fixed_mask = fixed_mask.astype(bool)
    free_mask  = ~fixed_mask
    rows_fixed = np.where(fixed_mask)[0]

    def matvec(u):
        out = _matvec_full(u, nx, ny, nz, dx, dy, dz, inside_3, D3)
        # impose Dirichlet rows in the operator: (u - val)
        out[rows_fixed] = u[rows_fixed] - fixed_val[rows_fixed]
        return out

    N = nx*ny*nz
    return LinearOperator((N, N), matvec=matvec, rmatvec=matvec, dtype=float)

def eps_schedule(final_eps):
    # Start coarse; end sharper
    base = [0.6, 0.3, 0.15, 0.08, 0.04]
    return [e for e in base if e > final_eps] + [final_eps]

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

def march_iso_surface_cyl(psi_n, inside, Rs, phis, Zs, level):
    """
    Marching cubes on a cylindrical grid ψ(R,φ,Z).
    Returns world vertices mapped via (x,y,z)=(R cosφ, R sinφ, Z).
    """
    NR, Nphi, NZ = len(Rs), len(phis), len(Zs)
    vol = psi_n.reshape(NR, Nphi, NZ).copy()
    mask = inside.reshape(NR, Nphi, NZ)
    vol[~mask] = np.nan
    vol = np.nan_to_num(vol, nan=-1.0)
    verts, faces, _, _ = marching_cubes(vol, level=level, mask=mask)

    # Map index space → physical (R,φ,Z). Rs/phis/Zs are linspace here.
    # verts[:,0] ~ iR, verts[:,1] ~ iPhi, verts[:,2] ~ iZ
    # Use linear map like in Cartesian path:
    if NR > 1:
        Rv = Rs[0] + (Rs[-1]-Rs[0]) * (verts[:,0] / (NR-1))
    else:
        Rv = np.full(len(verts), Rs[0])
    if Nphi > 1:
        Pv = phis[0] + (phis[-1]-phis[0]) * (verts[:,1] / (Nphi-1))
    else:
        Pv = np.full(len(verts), phis[0])
    if NZ > 1:
        Zv = Zs[0] + (Zs[-1]-Zs[0]) * (verts[:,2] / (NZ-1))
    else:
        Zv = np.full(len(verts), Zs[0])

    Xv = Rv * np.cos(Pv)
    Yv = Rv * np.sin(Pv)
    Zv = Zv
    V = np.column_stack([Xv, Yv, Zv])
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

def axis_band_from_points_cyl(axis_pts, Rs, phis, Zs, rad):
    NR, Nphi, NZ = len(Rs), len(phis), len(Zs)
    band = np.zeros((NR, Nphi, NZ), dtype=bool)
    RR, ZZ = np.meshgrid(Rs, Zs, indexing="ij")
    for jj, phi0 in enumerate(phis):
        # points near this φ
        c, s = np.cos(phi0), np.sin(phi0)
        ephi = np.array([-s, c, 0.0])
        dφ = np.abs((np.arctan2(axis_pts[:,1], axis_pts[:,0]) - phi0 + np.pi) % (2*np.pi) - np.pi)
        Q = axis_pts[dφ <= (2*np.pi/len(phis))]      # ~local slice
        if len(Q)==0: continue
        Rq = np.hypot(Q[:,0], Q[:,1]); Zq = Q[:,2]
        nbr = NearestNeighbors(n_neighbors=1).fit(np.c_[Rq, Zq])
        d,_ = nbr.kneighbors(np.c_[RR.ravel(), ZZ.ravel()])
        band[:, jj, :] = (d[:,0].reshape(NR, NZ) <= rad)
    return band

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

def boundary_band_cylindrical(P, N, Rs, phis, Zs, h_band_vox, j_smooth=2):
    """
    Return boolean mask 'band' of shape (NR, Nphi, NZ) selecting a continuous
    ribbon of Dirichlet=1 cells near the boundary in each φ-slice.
    - P, N: boundary points and outward normals (Ns,3)
    - h_band_vox: half-thickness in *world* units (recommend 2–3× voxel)
    """
    NR, Nphi, NZ = len(Rs), len(phis), len(Zs)
    band = np.zeros((NR, Nphi, NZ), dtype=bool)
    RR, ZZ = np.meshgrid(Rs, Zs, indexing="ij")

    for jj, phi0 in enumerate(phis):
        # project boundary *points and normals* to this φ-plane
        c, s = np.cos(phi0), np.sin(phi0)
        eR   = np.array([ c,  s, 0.0])
        ephi = np.array([-s,  c, 0.0])  # plane normal
        # exact projection onto the plane
        dist = P @ ephi
        Pp   = P - dist[:,None]*ephi[None,:]
        Np   = N - (N @ ephi)[:,None]*ephi[None,:]      # normals projected into plane
        # normalize 2D normals in (R,Z)
        Rb = Pp @ eR
        Zb = Pp[:,2]
        NbR = Np @ eR
        NbZ = Np[:,2]
        n2 = np.hypot(NbR, NbZ) + 1e-30
        NbR /= n2; NbZ /= n2

        # nearest neighbor in-plane
        nbr = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(np.c_[Rb, Zb])
        d, idx = nbr.kneighbors(np.c_[RR.ravel(), ZZ.ravel()])
        dmin = d[:,0].reshape(NR, NZ)

        # signed by projected normal
        Q = idx[:,0]
        sign = np.sign((RR.ravel()-Rb[Q]) * NbR[Q] + (ZZ.ravel()-Zb[Q]) * NbZ[Q]).reshape(NR, NZ)
        dsigned = dmin * sign  # >0 outside if normals are outward

        # thin interior ribbon: dsigned < 0 and |dsigned| <= h_band_vox
        m = (dsigned < 0.0) & (np.abs(dsigned) <= h_band_vox)
        if j_smooth > 0:
            from scipy.ndimage import binary_erosion
            # micro-erosion to avoid fat overlap and keep a 1–2 voxel stripe
            m = binary_erosion(m, iterations=1, border_value=0)
        band[:, jj, :] = m

    return band


def maybe_flip_normals(P, N):
    c = np.mean(P, axis=0)
    s = np.sum((P - c) * N, axis=1)
    avg = float(np.mean(s))
    if avg < 0:
        print(f"[ORIENT] Normals inward on average (⟨(P-c)·N⟩≈{avg:.3e}) → flipping.")
        return -N, True
    print(f"[ORIENT] Normals seem outward (⟨(P-c)·N⟩≈{avg:.3e}).")
    return N, False

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

def draw_bbox(ax, mins, maxs, **kw):
    xs = [mins[0], maxs[0]]; ys = [mins[1], maxs[1]]; zs = [mins[2], maxs[2]]
    corners = np.array(list(itertools.product(xs, ys, zs)))
    edges = [(0,1),(0,2),(0,4),(3,1),(3,2),(3,7),(5,1),(5,4),(5,7),(6,2),(6,4),(6,7)]
    for i,j in edges:
        xi, yi, zi = corners[i]; xj, yj, zj = corners[j]
        ax.plot([xi,xj],[yi,yj],[zi,zj], **({"color":"k","lw":0.8,"alpha":0.5} | kw))
        
def draw_cyl_bbox(ax, Rs, Zs, phis, n=72, **kw):
    Rmin, Rmax = float(Rs.min()), float(Rs.max())
    Zmin, Zmax = float(Zs.min()), float(Zs.max())
    Phi = np.linspace(phis[0], phis[-1] + (phis[1]-phis[0]), n, endpoint=True)
    for R in (Rmin, Rmax):
        x = R*np.cos(Phi); y = R*np.sin(Phi)
        ax.plot(x, y, Zmin*np.ones_like(x), **({"color":"k","lw":0.8,"alpha":0.5}|kw))
        ax.plot(x, y, Zmax*np.ones_like(x), **({"color":"k","lw":0.8,"alpha":0.5}|kw))
    # vertical generatrices at few φs
    for phi in np.linspace(Phi[0], Phi[-1], 8):
        c, s = np.cos(phi), np.sin(phi)
        for R in (Rmin, Rmax):
            ax.plot([R*c,R*c],[R*s,R*s],[Zmin,Zmax],
                    **({"color":"k","lw":0.6,"alpha":0.4}|kw))
     
def draw_RZ_box(ax, Rs, Zs, pad_frac=0.01):
    """Outline the cylindrical grid window in the R–Z plane, inset a hair so it doesn't coincide with the axes frame."""
    Rmin, Rmax = float(np.min(Rs)), float(np.max(Rs))
    Zmin, Zmax = float(np.min(Zs)), float(np.max(Zs))
    dR = Rmax - Rmin; dZ = Zmax - Zmin
    # Slightly expand the *axes limits* so the rectangle isn't on the frame
    ax.set_xlim(Rmin - pad_frac*dR, Rmax + pad_frac*dR)
    ax.set_ylim(Zmin - pad_frac*dZ, Zmax + pad_frac*dZ)
    # Draw rectangle slightly *inside* the true window
    epsR = 0.002*dR; epsZ = 0.002*dZ
    from matplotlib.patches import Rectangle
    rect = Rectangle((Rmin+epsR, Zmin+epsZ),
                     dR-2*epsR, dZ-2*epsZ,
                     fill=False, linewidth=1.5,
                     edgecolor='k', alpha=0.95, zorder=50)
    ax.add_patch(rect)

            
def plot_maps_RZ_planes_overlaid(P, psi_interp, ins_interp, phi_list, nfp, *,
                                 Rs, Zs, title="ψ(R,Z) @ multiple φ", nR=300, nZ=200):
    # Use the actual cylindrical grid window
    Rlo, Rhi = float(np.min(Rs)), float(np.max(Rs))
    Zlo, Zhi = float(np.min(Zs)), float(np.max(Zs))

    Rg = np.linspace(Rlo, Rhi, nR)
    Zg = np.linspace(Zlo, Zhi, nZ)
    RR, ZZ = np.meshgrid(Rg, Zg, indexing="xy")

    fig, ax = plt.subplots(figsize=(9,7))
    # show a faint background for the first plane (purely for color scale context)
    phi0 = float(phi_list[0])
    pts0 = np.column_stack([RR.ravel(), np.full(RR.size, phi0), ZZ.ravel()])
    Psi0 = psi_interp(pts0).reshape(nR, nZ).T
    Psi0 = np.nan_to_num(Psi0, nan=np.nanmin(Psi0))
    im = ax.imshow(Psi0, origin="lower", aspect="equal",
                   extent=[Rlo, Rhi, Zlo, Zhi], alpha=0.25)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85); cbar.set_label("ψ (interior)")

    # overlay contours from ALL φ on the same axes
    for idx, phi in enumerate(phi_list):
        pts = np.column_stack([RR.ravel(), np.full(RR.size, float(phi)), ZZ.ravel()])
        Psi = psi_interp(pts).reshape(nR, nZ).T
        Psi = np.nan_to_num(Psi, nan=np.nanmin(Psi))
        try:
            ax.contour(Rg, Zg, Psi, levels=6, linewidths=1.1, alpha=0.9)
        except Exception:
            pass
        Rb, Zb = boundary_curve_RZ(P, float(phi))
        ax.plot(Rb, Zb, '.', ms=2, alpha=0.6)

    ax.set_xlabel("R"); ax.set_ylabel("Z")
    ax.set_title(title + f"  (φ×{len(phi_list)})")
    ax.set_aspect('equal', 'box')
    draw_RZ_box(ax, Rs, Zs)
    return fig, ax

def plot_psi_maps_RZ_panels(psi3, Rs, phis, Zs, jj_list, title="ψ(R,Z)"):
    fig, axa = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axa = axa.ravel()
    Rmin, Rmax = float(np.min(Rs)), float(np.max(Rs))
    Zmin, Zmax = float(np.min(Zs)), float(np.max(Zs))
    extent = [Rmin, Rmax, Zmin, Zmax]
    for kk, jj in enumerate(jj_list):
        im = axa[kk].imshow(psi3[:, jj, :].T, origin='lower', aspect='equal', extent=extent)
        axa[kk].set_title(f"{title} @ φ≈{phis[jj]:+.2f}")
        axa[kk].set_xlabel("R"); axa[kk].set_ylabel("Z")
        draw_RZ_box(axa[kk], Rs, Zs)
        plt.colorbar(im, ax=axa[kk], shrink=0.85)
    return fig

def mid_lineout(psi3, Rs, phis, Zs):
    jj = len(phis)//8  # arbitrary slice, not special
    kk = len(Zs)//2
    v = psi3[:, jj, kk]
    print(f"[lineout @ φ≈{phis[jj]:+.2f}, Z≈{Zs[kk]:+.3f}] "
        f"min={v.min():.3e} med={np.median(v):.3e} max={v.max():.3e}")

def plot_poincare_from_slice_contours(P, psi3, Rs, phis, Zs, levels, jj_list):
    fig, ax = plt.subplots(figsize=(12,6))
    Rg, Zg = np.meshgrid(Rs, Zs, indexing="ij")
    for jj in jj_list:
        for lv in levels:
            try:
                cs = ax.contour(Rg, Zg, psi3[:, jj, :], levels=[lv], linewidths=1.2, alpha=0.9)
                # break cs into segments & draw; matplotlib already draws them in R–Z
            except Exception:
                pass
        # boundary projection for reference
        Rb, Zb = boundary_curve_RZ(P, float(phis[jj]))
        ax.plot(Rb, Zb, '.', ms=2, alpha=0.35, color='k')
    ax.set_aspect('equal', 'box'); ax.set_xlabel("R"); ax.set_ylabel("Z")
    ax.set_title("Poincaré-like cuts from 2-D contours (per φ slices)")
    draw_RZ_box(ax, Rs, Zs)
    return fig, ax

def plot_poincare_cuts_overlaid(P, iso_cache, phi_list, voxel, *, Rs=None, Zs=None):
    fig, ax = plt.subplots(figsize=(12,6))
    for phi0 in phi_list:
        # boundary first (faint)
        Rb, Zb = boundary_curve_RZ(P, phi0)
        if Rb.size:
            ax.plot(Rb, Zb, '.', ms=2, alpha=0.35, color='k')
        # overlay all iso levels at this φ
        for (lv, Viso, Fiso) in iso_cache:
            if Viso is None: 
                continue
            segs = intersect_iso_with_phi_plane(Viso, Fiso, phi0)
            polys = segments_to_polylines(segs, tol=0.3*voxel)
            for Pseg in polys:
                R = np.sqrt(Pseg[:,0]**2 + Pseg[:,1]**2)
                Z = Pseg[:,2]
                ax.plot(R, Z, lw=1.2, alpha=0.9)
    # global box from all boundary points
    if (Rs is not None) and (Zs is not None):
        # lock axes to the true cylindrical window and draw it
        ax.set_xlim(float(np.min(Rs)), float(np.max(Rs)))
        ax.set_ylim(float(np.min(Zs)), float(np.max(Zs)))
        draw_RZ_box(ax, Rs, Zs)
    else:
        eR = np.sqrt(P[:,0]**2 + P[:,1]**2)
        ax.set_xlim(eR.min()-0.05*(np.ptp(eR)+1e-9), eR.max()+0.05*(np.ptp(eR)+1e-9))
        ax.set_ylim(P[:,2].min()-0.05*(np.ptp(P[:,2])+1e-9), P[:,2].max()+0.05*(np.ptp(P[:,2])+1e-9))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("R"); ax.set_ylabel("Z")
    ax.set_title("Poincaré-like cuts (all φ, all levels)"); 
    return fig, ax

def plot_RZ_at_phi0(psi3, inside3, Rs, phis, Zs, title="ψ(R,Z) @ φ=0"):
    # choose the φ-index closest to 0 (accounting for periodicity)
    j0 = int(np.argmin(np.abs(((phis + np.pi) % (2*np.pi)) - np.pi)))
    Rmin, Rmax = float(np.min(Rs)), float(np.max(Rs))
    Zmin, Zmax = float(np.min(Zs)), float(np.max(Zs))
    extent = [Rmin, Rmax, Zmin, Zmax]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    # mask outside
    slice_psi = psi3[:, j0, :].T.copy()
    slice_msk = inside3[:, j0, :].T
    slice_psi[~slice_msk] = np.nan
    im = ax.imshow(slice_psi, origin="lower", aspect="equal", extent=extent)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9); cbar.set_label("ψ")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(f"{title} (φ≈{phis[j0]:+.3f})")
    draw_RZ_box(ax, Rs, Zs)
    return fig, ax


# ------------------------------- Main flow ------------------------------- #
def main(npz_file, grid_N=96, eps=1e-3, band_h=1.5, axis_seed_count=0, axis_band_radius=0.0,
         cg_tol=1e-8, cg_maxit=2000, verbose=True, plot=True, psi0=0.3, nfp=2, psi_levels="",
         save_figures=True, NR=12, NZ=12, Nphi=32):

    pinfo(f"Loading MFS checkpoint: {npz_file}")
    dat = np.load(npz_file, allow_pickle=True)
    center = dat["center"]; scale = float(dat["scale"])
    Yn = dat["Yn"]; alpha = dat["alpha"]; a = dat["a"]; a_hat = dat["a_hat"]
    P = dat["P"]; N = dat["N"]; kind = str(dat["kind"])
    # Normalize 'kind' once and use a robust boolean everywhere.
    kind_str = str(kind).strip().lower()
    kind_is_torus = (kind_str == "torus")
    ev = Evaluators(center=jnp.asarray(center), scale=scale,
                    Yn=jnp.asarray(Yn), alpha=jnp.asarray(alpha),
                    a=jnp.asarray(a), a_hat=jnp.asarray(a_hat))
    phi_fn, grad_phi = ev.build()
    pinfo(f"NPZ loaded. N_surf={P.shape[0]}, N_sources={Yn.shape[0]}, kind={kind_str}")

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
    
    if kind_is_torus:
        (Rs, phis, Zs), (dR, dphi, dZ), Xq = build_cyl_grid(P, nfp, NR, NZ, Nphi)
        NR, Nphi, NZ = len(Rs), len(phis), len(Zs)
        voxel = min(dR, dZ, Rs.mean()*dphi)  # for band sizing
    else:
        # existing Cartesian path (your current xs,ys,zs, XX/YY/ZZ, Xq, dx,dy,dz)
        NR = Nphi = NZ = None  # unused in mirror case
    
    N, flipped = maybe_flip_normals(P, N)
    if flipped:
        pinfo("[ORIENT] flipped boundary normals to outward.")
    else:
        pinfo("[ORIENT] boundary normals appear outward.")

    # Inside & boundary bands
    # After loading Xq and computing inside, also get 's'
    inside, nn_idx, s_signed = inside_mask_from_surface(P, N, Xq)
    pstat("Inside mask", inside.astype(float))
    
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
    if abs(d_plus - d_minus) <= 5e-5 * max(d_plus, d_minus):  # relative tie
        dir_sign = +1.0  # deterministic default
    else:
        dir_sign = +1.0 if d_plus > d_minus else -1.0
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
    bbox_diag = float(np.linalg.norm(maxs - mins))
    if axis_band_radius == 0.0:
        axis_band_radius_eff = 1.5 * min(dx, dy, dz)
    elif axis_band_radius < 1.0:
        axis_band_radius_eff = max(1.0 * min(dx, dy, dz), axis_band_radius * bbox_diag)
    else:
        axis_band_radius_eff = max(1.0 * min(dx, dy, dz), float(axis_band_radius))

    Ntot = Xq.shape[0]
    h_band_vox = max(2.0*voxel, float(band_h)*voxel)

    # compute bands once
    band3 = boundary_band_cylindrical(P, N, Rs, phis, Zs, h_band_vox)
    band3 &= inside.reshape(NR, Nphi, NZ)
    axis_band3 = axis_band_from_points_cyl(axis_pts_ds, Rs, phis, Zs, rad=axis_band_radius_eff)
    axis_band3 &= inside.reshape(NR, Nphi, NZ)
    # wrap periodicity
    band3[:, -1, :]      |= band3[:,  0, :]
    axis_band3[:, -1, :] |= axis_band3[:, 0, :]
    # ravel once
    band      = band3.ravel(order="C")
    axis_band = axis_band3.ravel(order="C")

    # free = inside & (~fixed)
    # if not np.any(free):
    #     raise RuntimeError("No free interior unknowns inside domain.")

    pstat("Boundary band fraction", band.astype(float))
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
    # =================== Dirichlet sets (independent of ε) ===================
    pinfo("Preparing Dirichlet masks & values ...")
    Ntot = Xq.shape[0]
    fixed = np.zeros(Ntot, dtype=bool)
    val   = np.zeros(Ntot, dtype=float)
    fixed[band] = True;      val[band] = 1.0

    # axis_band = axis_band_from_points_cyl(axis_pts_ds, Rs, phis, Zs, rad=axis_band_radius_eff)
    # axis_band = (axis_band & inside.reshape(NR, Nphi, NZ)).ravel(order="C")
    # val[axis_band] = 0.0; fixed[axis_band] = True
    
    free = inside & (~fixed)
    if not np.any(free):
        raise RuntimeError("No free interior unknowns inside domain.")
    # --- ε continuation: coarse → sharp alignment ---
    # Here we only update D to the sharpest stage and postpone solving
    # until the main CG section (where M is actually defined).
    D = diffusion_tensor(G, eps=eps, delta=1e-2)
    if kind_is_torus:
        A_full = make_linear_operator_cyl(Rs, phis, Zs, dR, dphi, dZ, inside, D,
                                            fixed_mask=fixed, fixed_val=val)
    else:
        A_full = make_linear_operator(nx, ny, nz, dx, dy, dz, inside, D, fixed, val)
    # After the loop, D and A_full correspond to the sharpest eps.
    
    # Fallback: build an axis band from low-|∇φ| skeleton if empty
    if not np.any(axis_band):
        pinfo("[AXIS] collapse-based axis band empty; using low-|∇φ| fallback.")
        gmag = np.linalg.norm(G, axis=1)
        gmag_inside = gmag[inside]
        if gmag_inside.size:
            thr = np.percentile(gmag_inside, 3.0)  # lowest 3%
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
            if centers.size == 0:
                centers = cand  # absolute last resort: use the raw candidates
            # axis_band = axis_band_mask(centers, Xq, rad=axis_band_radius_eff) & inside
            # pstat("[AXIS fallback] Axis band fraction", axis_band.astype(float))

    Dxx = D[...,0,0]; Dyy = D[...,1,1]; Dzz = D[...,2,2]
    Dxy = D[...,0,1]; Dxz = D[...,0,2]; Dyz = D[...,1,2]
    pstat("Dxx", Dxx); pstat("Dyy", Dyy); pstat("Dzz", Dzz)
    pstat("Dxy", Dxy); pstat("Dxz", Dxz); pstat("Dyz", Dyz)

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
    # pinfo(f"[BAND] surf_h={surf_h:.3e}, h_band={h_band:.3e}, "
    #     f"band/inside={(n_bnd/max(1,n_in))*100:.1f}% of interior")
    pinfo(f"[AXIS] axis_band_radius={axis_band_radius_eff:.3e}, "
        f"axis/inside={(n_ax/max(1,n_in))*100:.1f}% of interior")
    pinfo(f"[AXIS] seeds={axis_seed_count}, kept={axis_pts_ds.shape[0]} (thin tube)")
    
    # =================== Matrix-free + AMG PCG (replaces assembly) ===================
    pinfo("Building matrix-free LinearOperator ...")
    if kind_is_torus:
        A_full = make_linear_operator_cyl(
            Rs, phis, Zs, dR, dphi, dZ, inside, D,
            fixed_mask=fixed, fixed_val=val
        )
        # --- cell volumes for cylindrical grid (R dR dφ dZ) ---
        R3 = np.broadcast_to(Rs[:, None, None], (len(Rs), len(phis), len(Zs)))
        V_cells = (R3 * dR * dphi * dZ).ravel(order="C")
    else:
        A_full = make_linear_operator(nx, ny, nz, dx, dy, dz, inside, D, fixed, val)
        # Cartesian: constant volumes
        V_cells = np.full(Xq.shape[0], dx*dy*dz, dtype=float)
    
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
    
    # ---- Euclidean-SPD wrapper via symmetric volume scaling ----
    sqrtV_free    = np.sqrt(np.maximum(1e-30, V_cells[free]))
    invsqrtV_free = 1.0 / sqrtV_free

    def A_free_sym_matvec(y):
        # y = sqrt(V) * u_free
        u_free = invsqrtV_free * y
        r_free = Afree_linear_matvec(u_free)            # = A_ff * u_free
        return invsqrtV_free * r_free                   # = V^{-1/2} A_ff V^{-1/2} y

    A_free = LinearOperator((fidx.size, fidx.size),
                            matvec=A_free_sym_matvec,
                            rmatvec=A_free_sym_matvec,
                            dtype=float)
    b_free = sqrtV_free * b_free                        # b_s = sqrt(V) * b
    
    print("||b_free||_2 =", np.linalg.norm(b_free))
    if kind_is_torus:
        print("rows touching boundary band:",
              int(((inside & ~fixed) & np.roll(band.reshape(NR,Nphi,NZ), +1, 0).ravel()).sum()))
        print("rows touching axis band:",
              int(((inside & ~fixed) & np.roll(axis_band.reshape(NR,Nphi,NZ), +1, 0).ravel()).sum()))

    # 1) Ones test: with axis band empty and boundary band at 1, a constant 1 should solve ⇒ A*1 == b
    if not np.any(axis_band):
        ones = np.ones_like(b_free)
        print("||A_free*1 - b_free|| / ||b_free|| =",
            np.linalg.norm((A_free @ ones) - b_free) / max(1e-16, np.linalg.norm(b_free)))


    # after A_full/A_free are built
    z = np.random.default_rng(1).standard_normal(fidx.size)
    lhs = z @ (A_free @ z)
    rhs = (A_free @ z) @ z
    print("Energy symmetry check (free):", abs(lhs - rhs) / max(1e-16, abs(lhs) + abs(rhs)))
    
    z = np.random.default_rng(0).standard_normal(fidx.size)
    lhs = z @ (A_free @ z)
    rhs = (A_free @ z) @ z
    print("Energy symmetry check (scaled):", abs(lhs-rhs)/max(1e-16, abs(lhs)+abs(rhs)))

    # consistency of CSR vs matvec on the same vector
    # 2) CSR vs matvec: compare linear parts (no RHS)
    if kind_is_torus:
        A_ff_csr, _, _ = build_aniso_csr_free_cyl(Rs, phis, Zs, dR, dphi, dZ, inside, fixed, D, val)
    else:
        A_ff_csr, _, _ = build_aniso_csr_free(nx, ny, nz, dx, dy, dz, inside, fixed, D, val)
    # apply symmetric scaling to CSR for diagnostics / AMG build
    Dsca = spdiags(invsqrtV_free, 0, shape=(fidx.size, fidx.size))
    A_ff_csr = Dsca @ A_ff_csr @ Dsca
    psi_test = np.random.default_rng(2).standard_normal(fidx.size)
    Apsi_matvec = A_free @ psi_test
    Apsi_csr    = A_ff_csr @ psi_test
    diff_rel = np.linalg.norm(Apsi_matvec - Apsi_csr) / max(1e-16, np.linalg.norm(Apsi_csr))
    print("||A_matvec - A_csr|| / ||A||:", diff_rel)
    if diff_rel > 1e-8 and verbose:
        print("[WARN] CSR vs matvec mismatch is large; stencil/metrics likely off.")
        
    # Check SPD-ish numerics
    eig_diag = np.min(np.stack([Dxx, Dyy, Dzz], axis=-1), axis=-1)
    assert np.all(np.isfinite(eig_diag))
    assert np.percentile(eig_diag, 0.01) > 1e-8  # after delta=1e-2 this will hold

    # 3) Symmetry probe now measures the true linear operator
    asp = probe_symmetry(A_free)
    print("[A_free] antisymmetry probe =", asp)
    if asp > 1e-8:
        print("[WARN] A_free shows noticeable antisymmetry; check stencil indexing / BC stamping.")

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

    # Preconditioner: AMG on the *scaled* operator (symmetric)
    try:
        pinfo("[PCG] Building AMG on the scaled anisotropic operator (free rows) ...")
        def probe_symmetry_csr(A):
            rng = np.random.default_rng(0)
            x = rng.standard_normal(A.shape[0]); y = rng.standard_normal(A.shape[0])
            lhs = float(x @ (A @ y)); rhs = float((A @ x) @ y)
            den = max(1e-16, abs(lhs) + abs(rhs))
            return abs(lhs - rhs)/den
        pinfo(f"[A_ff_csr symmetry probe] {probe_symmetry_csr(A_ff_csr):.3e}")
        ml = smoothed_aggregation_solver(
            A_ff_csr,
            symmetry='symmetric',
            strength=('symmetric', {'theta': 0.04}),
            smooth=('energy', {'krylov': 'cg', 'degree': 2}),
            presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
            postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
            max_coarse=400,
            aggregate='standard',
            improve_candidates=None,
            B=np.ones((A_ff_csr.shape[0], 1))
        )
        M_amg = ml.aspreconditioner(cycle='W')

        # simple symmetric (left/right) Jacobi around AMG V-cycle
        dfine = np.maximum(1e-12, np.asarray(ml.levels[0].A.diagonal()))
        Dm12  = 1.0 / np.sqrt(dfine)
        def M_apply(x):
            y = Dm12 * x
            y = M_amg @ y
            return Dm12 * y

        # Wrap as LinearOperator
        M = LinearOperator((b_free.size, b_free.size), matvec=M_apply, rmatvec=M_apply, dtype=float)

        # ---------- sanity probe ----------
        vtest = np.random.default_rng(0).standard_normal(b_free.size)
        ytest = M @ vtest
        if not np.all(np.isfinite(ytest)):
            raise RuntimeError("AMG preconditioner produced non-finite values; falling back.")
        pinfo("[PCG] AMG preconditioner ready (finite check OK)")
    except Exception as e:
        pinfo(f"[PCG] Using Jacobi preconditioner ({e})")
        diag_approx = np.maximum(1e-8, np.ones_like(b_free))  # crude but safe
        def M_apply(x): return x / diag_approx
        M = LinearOperator((b_free.size, b_free.size), matvec=M_apply, rmatvec=M_apply, dtype=float)
        
    # Initial guess in scaled space
    x0_f = M @ b_free
    if not np.all(np.isfinite(x0_f)):
        pinfo("[Init] Non-finite x0 from preconditioner; falling back to zeros.")
        x0_f = np.zeros_like(b_free)
    
    pinfo("CG solve (matrix-free, SPD) ...")
    t0 = time.time()
    psi_f_scaled, info = cg(A_free, b_free, rtol=cg_tol, atol=0.0, maxiter=cg_maxit, M=M, x0=x0_f)
    if (not np.all(np.isfinite(psi_f_scaled))) or (info > 0):
        pinfo(f"[CG] status={info}. Retrying CG with Jacobi (symmetric) preconditioner.")
        # symmetric Jacobi:
        diag_j = np.maximum(1e-10, np.ones_like(b_free))
        M_jac = LinearOperator((b_free.size, b_free.size),
                            matvec=lambda v: v/diag_j,
                            rmatvec=lambda v: v/diag_j, dtype=float)
        x0_f2 = M_jac @ b_free
        psi_f_scaled, info = cg(A_free, b_free, rtol=cg_tol, atol=0.0, maxiter=cg_maxit, M=M_jac, x0=x0_f2)
        if (not np.all(np.isfinite(psi_f_scaled))) or (info > 0):
            pinfo(f"[CG] still struggling (status={info}). MINRES (no/nice M).")
            # MINRES requires symmetric A and symmetric positive (semi)definite M.
            # Use none or Jacobi only:
            psi_f_scaled, info = minres(A_free, b_free, rtol=cg_tol, maxiter=cg_maxit, M=M_jac, x0=x0_f2)
    t1 = time.time()
    pinfo(f"CG done: info={info}, iters≈{cg_maxit if info>0 else 'converged'} , wall={t1-t0:.2f}s")

    # Scatter back to full vector
    # psi = np.array(val)    # Dirichlet in place
    # psi[free] = psi_f
    # Scatter back to full vector using the last (sharpest) stage
    # undo scaling: u_free = V^{-1/2} y
    psi = np.array(val); psi[free] = invsqrtV_free * psi_f_scaled

    # Optional: full-operator residual should be ~0 everywhere (incl. fixed rows)
    r_full = A_full @ psi
    pstat("Residual Aψ (full)", r_full)
        
    # True reduced residual:
    r_reduced = (A_free @ (sqrtV_free * psi[free])) - b_free
    pinfo(f"[RES(reduced)] {np.linalg.norm(r_reduced)/max(1e-16, bnorm):.3e}")
    
    print("\n===== ψ distribution (inside) =====")
    psi_in = psi[inside];  stat01(psi_in);  frac_above(psi_in)
    if kind_is_torus:
        psi3_c = psi.reshape(NR, Nphi, NZ); inside3_c = inside.reshape(NR, Nphi, NZ)
        per_phi_percentiles(psi3_c, inside3_c)

    # Sanity on Dirichlet stamping (means should be ≈1.0 on band and ≈0 on axis)
    print(f"[BC check] ⟨ψ⟩ on boundary band={psi[band].mean() if band.any() else np.nan:.6f}, "
        f"on axis band={psi[axis_band].mean() if axis_band.any() else np.nan:.6f}")


    # Normalize ψ on the inside so [0,1] roughly spans interior
    mask_free_inside = inside & (~band) & (~axis_band)
    psi_in = psi[mask_free_inside]
    p01, p50, p99 = np.percentile(psi_in, [1, 50, 99]) if psi_in.size else (0.0, 0.0, 1.0)
    scale = max(p99 - p01, 1e-12)
    psi_n = np.clip((psi - p01)/scale, 0.0, 1.0)

    pinfo(f"[ψ] inside percentiles: p1={p01:.3e}, p50={p50:.3e}, p99={p99:.3e}; using normalized ψ in [0,1]")

# Levels: if user provided, respect; else pick quantiles that actually exist.
    if (psi_levels is not None) and (psi_levels.strip() != ""):
        levels = [float(s) for s in psi_levels.split(",") if s.strip()]
    else:
        # pick interior quantiles in normalized space but keep them small if ψ is concentrated near 0
        q = np.quantile(psi_n[inside], [0.02, 0.05, 0.1, 0.2, 0.4])
        levels = sorted(set([float(v) for v in q]))
    print("CLAMPING!! [iso-levels] pre-clamp", levels)
    levels = [v for v in levels if v <= 0.85]
    print("CLAMPING!! [iso-levels] post-clamp", levels)

    # Quality metrics (cylindrical vs Cartesian)
    if kind_is_torus:
        # Cylindrical core derivatives on (NR,Nphi,NZ)
        psi_c = psi.reshape(NR, Nphi, NZ)
        # central in R and Z; periodic in phi
        dpsi_dR  = (psi_c[2:, 1:-1, 1:-1] - psi_c[:-2, 1:-1, 1:-1]) / (2*dR)
        dpsi_dZ  = (psi_c[1:-1, 1:-1, 2:  ] - psi_c[1:-1, 1:-1, :-2 ]) / (2*dZ)
        dpsi_dph = (np.roll(psi_c[1:-1,:,:], -1, axis=1) - np.roll(psi_c[1:-1,:,:], +1, axis=1))[ :, 1:-1, 1:-1] / (2*dphi)
        # geometry on the same (NR-2, Nphi-2, NZ-2) core
        RR = np.broadcast_to(Rs[1:-1,None,None], dpsi_dR.shape)
        PH = np.broadcast_to(phis[None,1:-1,None], dpsi_dR.shape)
        cph, sph = np.cos(PH), np.sin(PH)
        invR = 1.0/np.maximum(RR, 1e-6)   # be slightly more conservative near axis
        # cylindrical → Cartesian components of ∇ψ
        gx = dpsi_dR*cph - (invR*dpsi_dph)*sph
        gy = dpsi_dR*sph + (invR*dpsi_dph)*cph
        gz = dpsi_dZ
        # match t_hat to same core
        th = t_hat.reshape(NR, Nphi, NZ, 3)
        thx = th[1:-1,1:-1,1:-1,0]
        thy = th[1:-1,1:-1,1:-1,1]
        thz = th[1:-1,1:-1,1:-1,2]
        par_core = thx*gx + thy*gy + thz*gz
        # interior core mask: drop voxels with tiny |∇φ|
        gnorm3 = gnorm.reshape(NR, Nphi, NZ)[1:-1,1:-1,1:-1]
        mask_mid = (gnorm3 > 1e-10)
        par = par_core[mask_mid].ravel()
    else:
        # Cartesian (nx,ny,nz) path as before
        psi3 = psi.reshape(nx, ny, nz)
        dpsidx = (psi3[2:,   1:-1, 1:-1] - psi3[:-2, 1:-1, 1:-1]) / (2*dx)
        dpsidy = (psi3[1:-1, 2:,   1:-1] - psi3[1:-1, :-2, 1:-1]) / (2*dy)
        dpsidz = (psi3[1:-1, 1:-1, 2:  ] - psi3[1:-1, 1:-1, :-2 ]) / (2*dz)
        t_hat_x = t_hat[:,0].reshape(nx, ny, nz)[1:-1,1:-1,1:-1]
        t_hat_y = t_hat[:,1].reshape(nx, ny, nz)[1:-1,1:-1,1:-1]
        t_hat_z = t_hat[:,2].reshape(nx, ny, nz)[1:-1,1:-1,1:-1]
        par = (t_hat_x*dpsidx + t_hat_y*dpsidy + t_hat_z*dpsidz).ravel()

    # build boolean voxel masks (per grid kind)
    if kind_is_torus:
        band_vox  = band.reshape(NR, Nphi, NZ)
        axis_vox  = axis_band.reshape(NR, Nphi, NZ)
        inside_3  = inside.reshape(NR, Nphi, NZ)
    else:
        band_vox  = band.reshape(nx, ny, nz)
        axis_vox  = axis_band.reshape(nx, ny, nz)
        inside_3  = inside.reshape(nx, ny, nz)
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

    # interior core slice to match derivative core
    if kind_is_torus:
        core_mid = core[1:-1,1:-1,1:-1]
        gn_mid   = gnorm.reshape(NR, Nphi, NZ)[1:-1,1:-1,1:-1]
        par_core = thx*gx + thy*gy + thz*gz
    else:
        core_mid = core[1:-1,1:-1,1:-1]
        gn_mid   = gnorm.reshape(nx, ny, nz)[1:-1,1:-1,1:-1]
        par_core = (t_hat_x*dpsidx + t_hat_y*dpsidy + t_hat_z*dpsidz)
    mask_mid = core_mid & (gn_mid > 1e-10)
    par = par_core[mask_mid].ravel()
    if par.size < 100:
        pinfo("[CORE] very small core sample; shrink bands or increase --N")
    pstat("|t·∇ψ| (core, mid, cyl-frame)", par)
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
        # Compute grad φ on a stride of boundary points and quiver (both kinds)
        stride = max(1, P.shape[0] // 1200)   # keep it light
        Pb = P[::stride]
        Gb = np.asarray(grad_phi(jnp.asarray(Pb)))
        # scale arrows by a visually reasonable factor
        gscale = 0.03 * float(np.linalg.norm(maxs - mins))

        fig = plt.figure(figsize=(14,6))
        ax = fig.add_subplot(1,2,1, projection='3d')
        # 3D isosurfaces (normalized ψ) — mirror: Cartesian; torus: cylindrical mapping
        iso_levels = levels[:2] if len(levels) >= 2 else levels
        for lv in iso_levels:
            try:
                if kind_is_torus:
                    draw_cyl_bbox(ax, Rs, Zs, phis)
                    Viso, Fiso = march_iso_surface_cyl(psi_n, inside, Rs, phis, Zs, level=lv)
                    ax.plot_trisurf(Viso[:,0], Viso[:,1], Viso[:,2], triangles=Fiso,
                                    alpha=0.35, linewidth=0.1)
                    print(f"[iso] level={lv:.3g}  verts={0 if Viso is None else len(Viso)}  tris={0 if Fiso is None else len(Fiso)}")
                else:
                    draw_bbox(ax, mins, maxs)
                    vol = psi_n.reshape(nx,ny,nz)
                    mask = inside.reshape(nx,ny,nz)
                    verts, faces, _, _ = marching_cubes(np.nan_to_num(vol, nan=-1.0), level=lv, mask=mask)
                    vx = mins[0] + verts[:,0]*(maxs[0]-mins[0])/(nx-1)
                    vy = mins[1] + verts[:,1]*(maxs[1]-mins[1])/(ny-1)
                    vz = mins[2] + verts[:,2]*(maxs[2]-mins[2])/(nz-1)
                    ax.plot_trisurf(vx, vy, vz, triangles=faces, alpha=0.35, linewidth=0.1)
            except Exception as e:
                pinfo(f"Marching cubes failed at level {lv}: {e}")
        ax.scatter(Pb[:,0], Pb[:,1], Pb[:,2], s=2, c='k', alpha=0.3)
        ax.quiver(Pb[:,0], Pb[:,1], Pb[:,2], Gb[:,0], Gb[:,1], Gb[:,2],
                  length=gscale, normalize=True, linewidth=0.5, color='tab:red', alpha=0.1)
        ax.set_title("Isosurfaces of ψ (0.1, 0.5) + boundary points")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        fix_matplotlib_3d(ax)

        # Quality dashboard: |t·∇ψ| / |∇ψ|  (dimensionless in [0,1])
        ax2 = fig.add_subplot(1,2,2)
        if kind_is_torus:
            # already have gx,gy,gz and thx,thy,thz on the (NR-2,Nphi-2,NZ-2) core
            grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
            par_core = thx*gx + thy*gy + thz*gz
        else:
            # Cartesian core
            grad_mag = np.sqrt(dpsidx**2 + dpsidy**2 + dpsidz**2)
            par_core = t_hat_x*dpsidx + t_hat_y*dpsidy + t_hat_z*dpsidz

        with np.errstate(divide='ignore', invalid='ignore'):
            q = np.abs(par_core) / np.maximum(grad_mag, 1e-14)
        qv = q[mask_mid].ravel()
        ax2.hist(qv, bins=60, alpha=0.8)
        ax2.set_yscale('log')
        ax2.set_xlabel(r"$|\,\hat t\cdot\nabla\psi\,|/|\nabla\psi|$")
        ax2.set_title("Field-alignment metric (smaller is better)")
        plt.tight_layout()
        if save_figures:
            fig.savefig("solve_flux_3d_iso_and_quality.png", dpi=150)
            
        # Residual slices / maps
        if not kind_is_torus:
            fig2, axa = plt.subplots(1,3, figsize=(14,4))
            r3 = r_full.reshape(nx,ny,nz)
            im0 = axa[0].imshow(r3[:,:,nz//2].T, origin='lower', aspect='equal'); plt.colorbar(im0, ax=axa[0]); axa[0].set_title("residual @ z mid")
            im1 = axa[1].imshow(r3[:,ny//2,:].T, origin='lower', aspect='equal'); plt.colorbar(im1, ax=axa[1]); axa[1].set_title("residual @ y mid")
            im2 = axa[2].imshow(r3[nx//2,:,:].T, origin='lower', aspect='equal'); plt.colorbar(im2, ax=axa[2]); axa[2].set_title("residual @ x mid")
            plt.tight_layout()
            if save_figures:
                fig2.savefig("solve_flux_residual_slices.png", dpi=150)
        else:
            # Torus: show residual maps in R–Z at a few φ
            r3 = r_full.reshape(NR, Nphi, NZ)
            phi_idx_list = [0, Nphi//4, Nphi//2, (3*Nphi)//4]
            fig2, axa = plt.subplots(2, 2, figsize=(12,8), constrained_layout=True)
            axa = axa.ravel(order="C")
            for kk, jj in enumerate(phi_idx_list):
                Rmin, Rmax = float(np.min(Rs)), float(np.max(Rs))
                Zmin, Zmax = float(np.min(Zs)), float(np.max(Zs))
                extent = [Rmin, Rmax, Zmin, Zmax]
                im = axa[kk].imshow(r3[:, jj, :].T, origin='lower', aspect='equal', extent=extent)
                axa[kk].set_xlabel("R"); axa[kk].set_ylabel("Z")
                axa[kk].set_title(f"residual @ φ≈{phis[jj]:+.2f}")
                plt.colorbar(im, ax=axa[kk], shrink=0.85)
                # overlay the true window outline
                draw_RZ_box(axa[kk], Rs, Zs)
            if save_figures:
                fig2.savefig("solve_flux_residual_maps_cyl.png", dpi=150)
                
        # --------- Poincaré-like cross-sections (both kinds) ----------
        iso_cache = []
        for lv in levels:
            try:
                if kind_is_torus:
                    Viso, Fiso = march_iso_surface_cyl(psi_n, inside, Rs, phis, Zs, level=lv)
                else:
                    Viso, Fiso = march_iso_surface(psi_n, inside, nx, ny, nz, mins, maxs, level=lv)
                iso_cache.append((lv, Viso, Fiso))
            except Exception as e:
                pinfo(f"Marching cubes failed at level {lv:.2f}: {e}")
                iso_cache.append((lv, None, None))

        phi_list = jnp.linspace(0, 2*jnp.pi/nfp, 4, endpoint=False).tolist()
        
        plot_poincare_cuts_overlaid(P, iso_cache, phi_list, voxel, Rs=Rs, Zs=Zs)
        if save_figures:
            plt.gcf().savefig("solve_flux_poincare_overlaid.png", dpi=150)
        
        # --- Interpolators for ψ and 'inside' (nearest for mask) ---
        if kind_is_torus:
            inside3_c = inside.reshape(NR, Nphi, NZ)
            psi3_c = psi.reshape(NR, Nphi, NZ)
            mask_in = inside3_c.astype(bool)
            free3   = (mask_in & (~band3) & (~axis_band3))
            both_in = free3[:, 0, :] & free3[:, -1, :]
            wrap_err = np.nanmax(np.abs(psi3_c[:,0,:][both_in] - psi3_c[:,-1,:][both_in]))
            print(f"[φ-periodicity | free∩free] max = {wrap_err:.3e}")
            # --- enforce 2π/NFP periodicity by wrapping the φ axis (last slice = first slice) ---
            dphi = float(phis[1]-phis[0]) if Nphi > 1 else 2*np.pi/nfp
            phis_ext = np.concatenate([phis, [phis[-1] + dphi]])
            psi_ext  = np.concatenate([psi3_c,  psi3_c[:, :1, :]], axis=1)
            ins_ext  = np.concatenate([inside3_c.astype(float), inside3_c[:, :1, :].astype(float)], axis=1)
            psi_interp = RGI((Rs, phis_ext, Zs), psi_ext, bounds_error=False, fill_value=np.nan)
            ins_interp = RGI((Rs, phis_ext, Zs), ins_ext, method="nearest",
                             bounds_error=False, fill_value=0.0)
            mid_lineout(psi3_c, Rs, phis, Zs)
        else:
            psi3 = psi.reshape(nx, ny, nz)
            inside3 = inside.reshape(nx, ny, nz)
            psi_interp = RGI((xs, ys, zs), psi3, bounds_error=False, fill_value=np.nan)
            ins_interp = RGI((xs, ys, zs), inside3.astype(float),
                             method="nearest", bounds_error=False, fill_value=0.0)
            
    if kind_is_torus:
        mid_lineout(psi3_c, Rs, phis, Zs)
        if kind_is_torus:
            plot_maps_RZ_planes_overlaid(P, psi_interp, ins_interp, phi_list, nfp,
                                         Rs=Rs, Zs=Zs, title="ψ(R,Z) overlaid")
            if save_figures:
                plt.gcf().savefig("solve_flux_maps_RZ_overlaid.png", dpi=150)
        
        
        if kind_is_torus and plot:
            jj_list = [0, Nphi//4, Nphi//2, (3*Nphi)//4]
            figP = plot_psi_maps_RZ_panels(psi3_c, Rs, phis, Zs, jj_list, title="ψ(R,Z)")
            if save_figures:
                figP.savefig("solve_flux_maps_RZ_panels.png", dpi=150)
                
            inside_vals = psi3_c[inside3_c]
            qs = np.quantile(inside_vals, [0.02, 0.05, 0.1, 0.2]) if inside_vals.size else [0.05]
            jj_list = [0, Nphi//4, Nphi//2, (3*Nphi)//4]
            figC, _ = plot_poincare_from_slice_contours(P, psi3_c, Rs, phis, Zs, qs, jj_list)
            if save_figures:
                figC.savefig("solve_flux_poincare_from_slices.png", dpi=150)
                
            inside3_c = inside.reshape(NR, Nphi, NZ)
            fig_phi0, _ = plot_RZ_at_phi0(psi3_c, inside3_c, Rs, phis, Zs)
            if save_figures:
                fig_phi0.savefig("solve_flux_RZ_phi0.png", dpi=150)
        
        plt.show()

    if kind_is_torus:
        grid = dict(kind="cyl", Rs=Rs, phis=phis, Zs=Zs)
    else:
        grid = dict(kind="cart", xs=xs, ys=ys, zs=zs)

    return dict(psi=psi, grid=grid, inside=inside, quality=dict(parallel_dot_grad=par, residual=r_full))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default="wout_precise_QA_solution.npz",
                    help="MFS solution checkpoint (*.npz) containing center, scale, Yn, alpha, a, a_hat, P, N")
    ap.add_argument("--nfp", type=int, default=2, help="number of field periods (for plotting)")
    ap.add_argument("--N", type=int, default=48, help="grid resolution per axis")
    ap.add_argument("--eps", type=float, default=0.1, help="parallel diffusion weight (smaller => more field-aligned)")
    ap.add_argument("--band-h", type=float, default=2.0, help="boundary band thickness multiplier")
    ap.add_argument("--cg-maxit", type=int, default=1000)
    ap.add_argument("--axis-seed-count", type=int, default=128,
                    help="number of interior seeds to collapse onto axis; 0 = auto (~2% of interior, clamped to [16,128])")
    ap.add_argument("--axis-band-radius", type=float, default=0.05,
                    help="axis band radius: 0=auto (1.5*voxel); (0,1)=fraction of bbox diagonal; >=1=absolute units")
    ap.add_argument("--cg-tol", type=float, default=1e-8)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--psi0", type=float, default=0.1, help="iso-surface level for plotting")
    ap.add_argument("--psi-levels", type=str, default="0.1, 0.2, 0.5, 0.8",
                    help="comma-separated normalized levels for iso-surface cuts (e.g. '0.2,0.5,0.8'). If empty, use --psi0 only.")
    ap.add_argument("--save-figures", action="store_true", default=True, help="save figures instead of showing interactively")
    ap.add_argument("--NR", type=int, default=28)
    ap.add_argument("--NZ", type=int, default=28)
    ap.add_argument("--Nphi", type=int, default=48)
    args = ap.parse_args()
    out = main(args.npz, grid_N=args.N, eps=args.eps, band_h=args.band_h,
               axis_seed_count=args.axis_seed_count, axis_band_radius=args.axis_band_radius,
               cg_tol=args.cg_tol, cg_maxit=args.cg_maxit, plot=(not args.no_plot), psi0=args.psi0,
               nfp=args.nfp, psi_levels=args.psi_levels, save_figures=args.save_figures,
               NR=args.NR, NZ=args.NZ, Nphi=args.Nphi)
