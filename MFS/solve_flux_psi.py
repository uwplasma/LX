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
    Sign test via nearest neighbor: inside if (x - p_nn)·n_nn < 0 (outward normals).
    Robust for reasonably smooth closed surfaces.
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P_surf)
    d, idx = nbrs.kneighbors(Xq)
    p = P_surf[idx[:,0], :]
    n = N_surf[idx[:,0], :]
    s = np.sum((Xq - p)*n, axis=1)  # >0 outside (along outward normals)
    return (s < 0.0), idx[:,0]

def boundary_band_mask(P_surf, Xq, k=4, band_h=2.0):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(P_surf)
    d, _ = nbrs.kneighbors(Xq)
    # band if mean distance to nearest neighbors < band_h * local spacing estimate
    dn = np.mean(d, axis=1)
    thr = band_h*np.median(dn)
    return dn < thr

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
    t = gradphi / (np.linalg.norm(gradphi, axis=-1, keepdims=True) + 1e-30)
    # P_perp = I - t t^T; P_par = t t^T
    I = np.eye(3)[None,:,:]
    tt = t[..., :, None]*t[..., None, :]
    D = (I - tt) + eps*tt
    return D

def build_sparse_operator(nx, ny, nz, dx, dy, dz, inside, Dfield):
    """
    3D 6-point stencil for div(D ∇ψ). Tensor handled by projecting along axes.
    Returns CSR matrix L and rhs (zeros).
    """
    def idx3(i,j,k): return (k*ny + j)*nx + i
    rows, cols, vals = [], [], []
    N = nx*ny*nz
    # Precompute directional conductivities κx,κy,κz from tensor (coordinate-wise projection)
    kx = Dfield[...,0,0]; ky = Dfield[...,1,1]; kz = Dfield[...,2,2]
    # Off-diagonal coupling is ignored in this simple 6-point FV (robust if grid aligns reasonably with t);
    # You can add cross-derivative terms by using a 27-point stencil if needed.

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = idx3(i,j,k)
                if not inside[p]:
                    rows.append(p); cols.append(p); vals.append(1.0)  # outside -> identity (ψ won't be used)
                    continue
                diag = 0.0
                # x- neighbors
                if i>0 and inside[idx3(i-1,j,k)]:
                    w = 0.5*(kx[p] + kx[idx3(i-1,j,k)]) / dx**2
                    rows += [p]; cols += [idx3(i-1,j,k)]; vals += [-w]; diag += w
                if i<nx-1 and inside[idx3(i+1,j,k)]:
                    w = 0.5*(kx[p] + kx[idx3(i+1,j,k)]) / dx**2
                    rows += [p]; cols += [idx3(i+1,j,k)]; vals += [-w]; diag += w
                # y- neighbors
                if j>0 and inside[idx3(i,j-1,k)]:
                    w = 0.5*(ky[p] + ky[idx3(i,j-1,k)]) / dy**2
                    rows += [p]; cols += [idx3(i,j-1,k)]; vals += [-w]; diag += w
                if j<ny-1 and inside[idx3(i,j+1,k)]:
                    w = 0.5*(ky[p] + ky[idx3(i,j+1,k)]) / dy**2
                    rows += [p]; cols += [idx3(i,j+1,k)]; vals += [-w]; diag += w
                # z- neighbors
                if k>0 and inside[idx3(i,j,k-1)]:
                    w = 0.5*(kz[p] + kz[idx3(i,j,k-1)]) / dz**2
                    rows += [p]; cols += [idx3(i,j,k-1)]; vals += [-w]; diag += w
                if k<nz-1 and inside[idx3(i,j,k+1)]:
                    w = 0.5*(kz[p] + kz[idx3(i,j,k+1)]) / dz**2
                    rows += [p]; cols += [idx3(i,j,k+1)]; vals += [-w]; diag += w
                rows.append(p); cols.append(p); vals.append(diag if diag>0 else 1.0)

    L = coo_matrix((vals,(rows,cols)), shape=(N,N)).tocsr()
    rhs = np.zeros(N, dtype=float)
    return L, rhs

# ------------------------------- Main flow ------------------------------- #
def main(npz_file, grid_N=96, eps=1e-3, band_h=1.5, axis_seed_count=64, axis_band_radius=0.02,
         cg_tol=1e-8, cg_maxit=2000, verbose=True, plot=True):

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
    XX,YY,ZZ = np.meshgrid(xs,ys,zs, indexing="xy")
    Xq = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    pinfo(f"Grid: {nx}x{ny}x{nz} ~ {Xq.shape[0]} nodes; spacing dx≈{dx:.3g},dy≈{dy:.3g},dz≈{dz:.3g}")

    # Inside & boundary bands
    inside, nn_idx = inside_mask_from_surface(P, N, Xq)
    pstat("Inside mask", inside.astype(float))
    band = boundary_band_mask(P, Xq, k=4, band_h=band_h)
    band = np.logical_and(band, inside)
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
    axis_band = axis_band_mask(axis_pts, Xq, rad=axis_band_radius*np.linalg.norm(span))
    axis_band = np.logical_and(axis_band, inside)
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

    # Impose Dirichlet: overwrite rows
    pinfo("Imposing Dirichlet rows ...")
    L = L.tolil()
    rows = np.where(fixed)[0]
    L[rows, :] = 0.0
    L[rows, rows] = 1.0
    rhs[rows] = val[rows]
    L = L.tocsr()

    # Solve
    pinfo("CG solve ...")
    t0 = time.time()

    # Cheap diagonal (Jacobi) preconditioner
    from scipy.sparse import diags as spdiags
    diagL = np.array(L.diagonal(), dtype=float)
    # Guard against zeros on Dirichlet rows:
    inv_diag = np.where(diagL > 0, 1.0/diagL, 1.0)
    M = spdiags(inv_diag, offsets=0)

    # Symmetrize to reduce tiny non-SPD artifacts from row stamping
    L = (L + L.T) * 0.5
    # Ensure CSR for fast matvec
    L = L.tocsr()

    # SciPy API compatibility: prefer rtol/atol, fallback to tol on older versions
    try:
        psi, info = cg(L, rhs, rtol=cg_tol, atol=0.0, maxiter=cg_maxit, M=M)
    except TypeError:
        # Older SciPy (uses 'tol' only)
        psi, info = cg(L, rhs, tol=cg_tol, maxiter=cg_maxit, M=M)

    t1 = time.time()
    pinfo(f"CG done: info={info}, wall={t1-t0:.2f}s")

    # Quality metrics
    # parallel derivative proxy: |t_hat·∇ψ|
    # compute ∇ψ by central differences (skip edges)
    # x,y,z indexing
    psi3 = psi.reshape(ny, nx, nz).transpose(1, 0, 2)  # (nx, ny, nz)

    # Centered differences on the common interior core -> (nx-2, ny-2, nz-2)
    dpsidx = (psi3[2:,   1:-1, 1:-1] - psi3[:-2, 1:-1, 1:-1]) / (2*dx)
    dpsidy = (psi3[1:-1, 2:,   1:-1] - psi3[1:-1, :-2, 1:-1]) / (2*dy)
    dpsidz = (psi3[1:-1, 1:-1, 2:  ] - psi3[1:-1, 1:-1, :-2 ]) / (2*dz)

    # Match t_hat to the same interior core
    t_hat_x = t_hat[:,0].reshape(nx,ny,nz)[1:-1,1:-1,1:-1]
    t_hat_y = t_hat[:,1].reshape(nx,ny,nz)[1:-1,1:-1,1:-1]
    t_hat_z = t_hat[:,2].reshape(nx,ny,nz)[1:-1,1:-1,1:-1]

    par = (t_hat_x*dpsidx + t_hat_y*dpsidy + t_hat_z*dpsidz).ravel()
    pstat("|t·∇ψ| (interior)", par)

    # Flux neutrality check inside: ∫ div(D∇ψ) dV ≈ 0 → sample residual
    # quick residual r = Lψ - rhs (dense-free via multiply)
    r = L @ psi - rhs
    pstat("Residual Lψ - rhs", r)

    # ------------------------------ Plots ------------------------------ #
    if plot:
        # 3D isosurfaces
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(14,6))
        ax = fig.add_subplot(1,2,1, projection='3d')
        # take only inside region
        vol = psi.reshape(nx,ny,nz).copy()
        vol[~inside.reshape(nx,ny,nz)] = np.nan
        for lv in [0.2, 0.4, 0.6, 0.8]:
            try:
                verts, faces, norm, val = marching_cubes(np.nan_to_num(vol, nan=-1.0), level=lv)
                # map verts from index to real coords
                vx = mins[0] + verts[:,0]*(maxs[0]-mins[0])/(nx-1)
                vy = mins[1] + verts[:,1]*(maxs[1]-mins[1])/(ny-1)
                vz = mins[2] + verts[:,2]*(maxs[2]-mins[2])/(nz-1)
                ax.plot_trisurf(vx, vy, faces, vz, alpha=0.35, linewidth=0.1)
            except Exception as e:
                pinfo(f"Marching cubes failed at level {lv}: {e}")
        ax.scatter(P[::8,0], P[::8,1], P[::8,2], s=2, c='k', alpha=0.2)
        ax.set_title("Isosurfaces of ψ (0.2, 0.4, 0.6, 0.8) + boundary points")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

        # Quality dashboard
        ax2 = fig.add_subplot(1,2,2)
        ax2.hist(np.abs(par), bins=60, alpha=0.8)
        ax2.set_yscale('log'); ax2.set_xlabel(r"$|\,\hat t\cdot\nabla\psi\,|$")
        ax2.set_title("Field-aligned derivative magnitude (smaller is better)")
        plt.tight_layout(); plt.show()

        # Residual scatter & slices
        fig2, axa = plt.subplots(1,3, figsize=(14,4))
        r3 = r.reshape(nx,ny,nz)
        im0 = axa[0].imshow(r3[:,:,nz//2].T, origin='lower', aspect='equal'); plt.colorbar(im0, ax=axa[0]); axa[0].set_title("residual @ z mid")
        im1 = axa[1].imshow(r3[:,ny//2,:].T, origin='lower', aspect='equal'); plt.colorbar(im1, ax=axa[1]); axa[1].set_title("residual @ y mid")
        im2 = axa[2].imshow(r3[nx//2,:,:].T, origin='lower', aspect='equal'); plt.colorbar(im2, ax=axa[2]); axa[2].set_title("residual @ x mid")
        plt.tight_layout(); plt.show()

    return dict(psi=psi, grid=(xs,ys,zs), inside=inside, quality=dict(parallel_dot_grad=par, residual=r))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="MFS solution checkpoint (*.npz) containing center, scale, Yn, alpha, a, a_hat, P, N")
    ap.add_argument("--N", type=int, default=26, help="grid resolution per axis")
    ap.add_argument("--eps", type=float, default=1e-3, help="parallel diffusion weight (smaller => more field-aligned)")
    ap.add_argument("--band-h", type=float, default=1.5, help="boundary band thickness multiplier")
    ap.add_argument("--axis-seed-count", type=int, default=64, help="number of interior seeds to collapse onto axis")
    ap.add_argument("--axis-band-radius", type=float, default=0.02, help="axis band radius as fraction of bbox size")
    ap.add_argument("--cg-tol", type=float, default=1e-8)
    ap.add_argument("--cg-maxit", type=int, default=300)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    out = main(args.npz, grid_N=args.N, eps=args.eps, band_h=args.band_h,
               axis_seed_count=args.axis_seed_count, axis_band_radius=args.axis_band_radius,
               cg_tol=args.cg_tol, cg_maxit=args.cg_maxit, plot=(not args.no_plot))
