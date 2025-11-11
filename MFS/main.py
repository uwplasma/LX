#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## Run as python main.py --sf_min 0.5 --lbfgs-maxiter 50 --k-nn 128 for high-accuracy solve
## Run as python main.py --sf_min 1.5 --lbfgs-maxiter 5 --k-nn 32 for fast solve
### Example python main.py wout_precise_QA.csv wout_precise_QA_normals.csv
### SLAM surface might need --sf_min 3.0 for stability
"""
Laplace (∇²φ = 0) inside a closed 3D surface with Neumann BC n·∇φ = 0 (total)
via Method of Fundamental Solutions (MFS) with multivalued pieces (toroidal+poloidal).

Representation:
  φ(x) = φ_mv(x) + ψ(x),  ψ(x) = Σ_j α_j G(x, y_j),   G = 1/(4π|x - y|)
where the sources y_j are on an OUTER fictitious surface built by offsetting the
boundary points along outward normals by a distance δ.

We solve a Tikhonov-regularized weighted least-squares for [α, a_t, a_p]:
  minimize || W^{1/2} ( A α + D a - g ) ||_2^2 + λ^2 ( ||α||_2^2 + γ^2 ||a||_2^2 )
where:
  - A_ij = n_i · ∇_x G(x_i, y_j)
  - D has 2 columns: D[:,0] = n·grad_t(x_i), D[:,1] = n·grad_p(x_i)
  - g = 0  (we solve for total Neumann = 0)   equivalently Aα + D a ≈ 0
    (this automatically cancels the multivalued normal component)
We enforce compatibility by centering D columns under W (optional but helpful).

Diagnostics and plots:
  • Rich prints: kNN scales, δ choice, λ choice, system sizes, condition surrogates,
    residual norms, |∇φ| stats on Γ and on Γ₋ (interior-offset ring), flux neutrality on Γ₋,
    and |∇²ψ| near boundary (should be tiny).
  • Plots: geometry+normals, |∇φ| on Γ₋, BC error |n·∇φ| on Γ₋, Laplacian histogram.

Inputs:
  slam_surface.csv           columns: x,y,z
  slam_surface_normals.csv   columns: nx,ny,nz

Deps: jax (64-bit), jaxlib, numpy, matplotlib, scikit-learn
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, jacrev
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
import jax.scipy.linalg as jsp_linalg
from scipy.interpolate import griddata
from jax import debug as jax_debug
from functools import partial
import argparse

# ------------------------- JAX: 64-bit ------------------------- #
jax.config.update("jax_enable_x64", True)

# -------------------------- Small utils ------------------------ #
def pct(a, p): return float(np.percentile(np.asarray(a), p))

def vec_stats(title, v, W=None):
    v_np = np.asarray(v)
    if W is None:
        print(f"[STATS] {title}: L2={np.linalg.norm(v_np):.3e}, "
              f"Linf={np.max(np.abs(v_np)):.3e}, mean={np.mean(v_np):.3e}")
    else:
        W_np = np.asarray(W)
        lw2 = np.sqrt(np.dot(W_np, v_np**2))
        mean_w = np.dot(W_np, v_np) / (np.sum(W_np) + 1e-30)
        print(f"[STATS] {title}: ||·||_W2={lw2:.3e}, Linf={np.max(np.abs(v_np)):.3e}, <W,·>={mean_w:.3e}")

def fix_matplotlib_3d(ax):
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1]-x_limits[0]); x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1]-y_limits[0]); y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1]-z_limits[0]); z_mid = np.mean(z_limits)
    R = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid-R, x_mid+R]); ax.set_ylim3d([y_mid-R, y_mid+R]); ax.set_zlim3d([z_mid-R, z_mid+R])


@jax.tree_util.register_pytree_node_class
@dataclass
class ScaleInfo:
    center: jnp.ndarray   # (3,)
    scale: jnp.ndarray    # scalar array, shape ()

    def tree_flatten(self):
        # ensure scale is a JAX scalar
        return ((self.center, jnp.asarray(self.scale)), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        center, scale = children
        # IMPORTANT: do NOT call float(scale) here
        return cls(center=center, scale=jnp.asarray(scale))
    
# --- NEW: small dense BFGS in N-D with projection box ---
def bfgs_nd(value_and_grad_fn, p0, project_fn, *, max_iter=25, tol=1e-6):
    p = p0
    H = jnp.eye(p0.shape[0], dtype=p0.dtype)
    f_prev = jnp.inf
    done = False
    it = 0

    def step(state):
        p, H, f_prev, it, done = state
        def _advance(_):
            p_proj = project_fn(p)
            f, g = value_and_grad_fn(p_proj)

            d = - H @ g  # descent
            # Armijo backtracking (up to 4 shrinks)
            def bt_body(carry, _):
                step, f_curr = carry
                p_try = project_fn(p_proj + step * d)
                f_try, _ = value_and_grad_fn(p_try)
                ok = f_try <= f + 1e-4 * step * (g @ d)
                step = jnp.where(ok, step, 0.5 * step)
                f_curr = jnp.where(ok, f_try, f_curr)
                return (step, f_curr), ok
            (step_final, f_new), _ = jax.lax.scan(
                bt_body, (jnp.array(1.0, p.dtype), f), jnp.arange(4)
            )
            p_new = project_fn(p_proj + step_final * d)
            _, g_new = value_and_grad_fn(p_new)

            s = p_new - p_proj
            y = g_new - g
            ys = y @ s
            # Powell damping
            Hy = H @ s
            syHs = s @ Hy
            theta = jnp.where(ys < 0.2 * syHs, 0.8 * syHs / (syHs - ys + 1e-30), 1.0)
            y_tilde = theta * y + (1 - theta) * Hy
            rho = 1.0 / (y_tilde @ s + 1e-30)
            I = jnp.eye(p.shape[0], dtype=p.dtype)
            V = I - rho * jnp.outer(s, y_tilde)
            H_new = V @ H @ V.T + rho * jnp.outer(s, s)

            rel = jnp.abs((f_new - f) / (jnp.abs(f) + 1e-30))
            done_new = jnp.logical_or(jnp.linalg.norm(g_new) < tol, rel < 1e-3)
            return (p_new, H_new, f_new, it + 1, done_new)

        def _pass(_):
            return (p, H, f_prev, it + 1, done)

        return jax.lax.cond(done, _pass, _advance, operand=None)

    def cond(state):
        _, _, _, it, done = state
        return jnp.logical_and(it < max_iter, jnp.logical_not(done))

    state0 = (p, H, f_prev, jnp.array(0, jnp.int32), jnp.array(False))
    p_star, H_star, f_star, _, _ = jax.lax.while_loop(cond, step, state0)
    f_star, g_star = value_and_grad_fn(project_fn(p_star))
    return p_star, f_star, g_star, H_star


# ----------------------------- Solvers ----------------------------- #
def build_ring_weights(P_in, Pn, k=32):
    """
    kNN area weights W_ring on the interior ring Γ₋.
    Reuses the best-fit (u,v)-plane from the boundary Pn to define distances.
    """
    # Plane from boundary (stays constant for the surface)
    c_plane, E_plane, _ = best_fit_axis(np.array(Pn), verbose=False)
    # Project ring to that plane
    Ploc_ring = project_to_local(P_in, c_plane, E_plane)
    XY = np.asarray(Ploc_ring[:, :2])

    k_eff = min(k+1, len(XY))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree").fit(XY)
    dists, _ = nbrs.kneighbors(XY)
    rk_ring = dists[:, -1]  # k-th neighbor radius
    W_ring = jnp.asarray(np.pi * rk_ring**2, dtype=jnp.float64)
    return W_ring

def autotune(P, N, Pn, Nn, W, rk, scinfo,
             use_mv=True,
             interior_eps_factor=5e-3,
             verbose=True,
             sf_min=1.0, sf_max=6.5,
             lbfgs_maxiter=15, lbfgs_tol=1e-8):
    
    # harmonic pack (fixed, tiny)
    h_vals, h_grads = get_harmonic_pack()

    # 1) MV bases and a (once)
    if use_mv:
        kind, a_hat, E_axes, c_axes, svals = detect_geometry_and_axis(Pn, verbose=True)
        phi_t, grad_t, phi_p, grad_p = multivalued_bases_about_axis(Pn, Nn, a_hat, verbose=True)
        gt_b, gp_b = grad_t(Pn), grad_p(Pn)
        a, _, _ = fit_mv_coeffs_minimize_rhs(Nn, W, gt_b, gp_b, verbose=False)
        g_raw = scinfo.scale * jnp.sum(Nn * (a[0]*gt_b + a[1]*gp_b), axis=1)
    else:
        def phi_t(Xn): return jnp.zeros((Xn.shape[0],), dtype=jnp.float64)
        def grad_t(Xn): return jnp.zeros_like(Xn)
        def phi_p(Xn): return jnp.zeros((Xn.shape[0],), dtype=jnp.float64)
        def grad_p(Xn): return jnp.zeros_like(Xn)
        a = jnp.zeros((2,), dtype=jnp.float64)
        g_raw = jnp.zeros((Pn.shape[0],), dtype=jnp.float64)

    obj, eps_bounds = make_objective_for_sf_lam_eps(
        P, N, Pn, Nn, W, rk, scinfo, grad_t, grad_p, a, g_raw,
        eps_bounds=(2e-3, 5e-2), h_grads=h_grads, debug_opt=True
    )

    # Start near your current defaults
    p0 = jnp.array([jnp.log(1.5), jnp.log(5e-3), jnp.log(5e-3)], dtype=jnp.float64)

    # Projection box in log-space
    log_sf_lo, log_sf_hi   = jnp.log(sf_min), jnp.log(sf_max)
    log_lam_lo, log_lam_hi = jnp.log(1e-4),   jnp.log(5e-2)
    log_eps_lo, log_eps_hi = jnp.log(eps_bounds[0]), jnp.log(eps_bounds[1])

    def project_fn(p):
        return jnp.array([
            jnp.clip(p[0], log_sf_lo,  log_sf_hi),
            jnp.clip(p[1], log_lam_lo, log_lam_hi),
            jnp.clip(p[2], log_eps_lo, log_eps_hi),
        ], dtype=p.dtype)

    # === Coarse seeding (stable, tiny budget) ===
    # Scan a few sensible candidates to avoid a bad basin:
    sf_grid  = jnp.array([1.2, 2.0, 3.5, 5.0, sf_max], dtype=jnp.float64)
    lam_grid = jnp.array([3e-3, 5e-3, 1e-2], dtype=jnp.float64)
    eps_grid = jnp.array([2e-3, 3e-3, 5e-3], dtype=jnp.float64)

    best_seed = None
    best_val  = jnp.inf

    obj_valonly = lambda p: obj(p)[0]  # obj returns (val, grad)
    for sf0 in sf_grid:
        for lam0 in lam_grid:
            for eps0 in eps_grid:
                p_try = jnp.log(jnp.array([sf0, lam0, eps0], dtype=jnp.float64))
                v = obj_valonly(p_try)
                if v < best_val:
                    best_val, best_seed = v, p_try

    # Start BFGS from the best coarse seed
    p0 = best_seed

    p_star, f_star, g_star, H_star = bfgs_nd(obj, p0, project_fn, max_iter=lbfgs_maxiter, tol=lbfgs_tol)
    log_sf_star, log_lam_star, log_eps_star = project_fn(p_star)
    sf_star   = float(jnp.exp(log_sf_star))
    lam_star  = float(jnp.exp(log_lam_star))
    epsn_star = float(jnp.exp(log_eps_star))

    # finalize with sources at optimum and return epsn_star for later diagnostics
    Yn, _ = build_mfs_sources(Pn, Nn, rk, scinfo, source_factor=sf_star, verbose=False)
    if verbose:
        print(f"[OPT-3D] δ*={sf_star:.3f}, λ*={lam_star:.3e}, ε_n*={epsn_star:.3e}, J*={float(f_star):.3e}")

    return dict(
        source_factor=sf_star, lambda_reg=lam_star,
        eps_n_star=epsn_star,
        a=a, phi_t=phi_t, grad_t=grad_t, phi_p=phi_p, grad_p=grad_p, Yn=Yn,
        a_hat=a_hat, kind=kind,
        h_vals=h_vals, h_grads=h_grads
    )

def autotune_outer(P, N, Pn, Nn, W, rk, scinfo,
                   use_mv=True, verbose=False,
                   sf_min=1.0, sf_max=6.5,
                   lbfgs_maxiter=15, lbfgs_tol=1e-8):
    """
    Thin wrapper around `autotune` for the refinement block, returning the same dict.
    """
    return autotune(
        P, N, Pn, Nn, W, rk, scinfo,
        use_mv=use_mv, interior_eps_factor=5e-3,  # keep same default you used earlier
        verbose=verbose,
        sf_min=sf_min, sf_max=sf_max,
        lbfgs_maxiter=lbfgs_maxiter, lbfgs_tol=lbfgs_tol
    )

# --- NEW: D columns for MV in WORLD units on arbitrary points/normals ---
@partial(jax.jit, static_argnames=("grad_t_fn","grad_p_fn"))
def build_D_columns_at_points(Xn_eval, N_eval, scinfo, grad_t_fn, grad_p_fn):
    """
    Returns D=(N,2) with D[:,0]=n·grad_t(Xn_eval), D[:,1]=n·grad_p(Xn_eval) in WORLD units.
    """
    Gt = grad_t_fn(Xn_eval)   # (N,3) normalized-coord grads
    Gp = grad_p_fn(Xn_eval)   # (N,3)
    # multiply by scinfo.scale to convert ∇ in normalized coords to WORLD coords
    Dt = scinfo.scale * jnp.sum(N_eval * Gt, axis=1)
    Dp = scinfo.scale * jnp.sum(N_eval * Gp, axis=1)
    return jnp.stack([Dt, Dp], axis=1)  # (N,2)

# --- NEW: augmented rows [A | H | D] at arbitrary points ---
@partial(jax.jit, static_argnames=("grad_t_fn","grad_p_fn","h_grads"))
def build_aug_rows_at_points(Xn_eval, N_eval, Yn, scinfo, grad_t_fn, grad_p_fn, h_grads=None):
    A_in = build_A_rows_at_points(Xn_eval, N_eval, Yn, scinfo, h_grads=h_grads)   # (N, M_mfs[+K])
    D_in = build_D_columns_at_points(Xn_eval, N_eval, scinfo, grad_t_fn, grad_p_fn)  # (N,2)
    return jnp.concatenate([A_in, D_in], axis=1)  # (N, M_mfs[+K]+2)

# --- NEW: flux constraint on augmented matrices (zero RHS) ---
def make_zero_rhs_flux_constraint(A_like, W_like):
    """
    For augmented matrix Â with rows over points and W_like quadrature weights,
    return (c_vec, d=0) with c_vec = Â^T W 1.
    """
    Wv = jnp.asarray(W_like).reshape(-1)
    A  = jnp.asarray(A_like)
    if A.ndim != 2 or A.shape[0] != Wv.shape[0]:
        raise ValueError(f"make_zero_rhs_flux_constraint expects A.shape=(N,M), len(W)=N; got {A.shape}, {Wv.shape[0]}")
    c_vec = (A.T * Wv[None, :]).sum(axis=1)  # (M,)
    d_val = jnp.array(0.0, dtype=A.dtype)
    return c_vec, d_val

# --- NEW: joint solve for x=[alpha; a] with constraints and per-column reg ---
def solve_alpha_a_with_constraints(A_aug, W, lam, reg_gamma_a=1e-3, constraints=None, verbose=True):
    """
    A_aug: (N, M_tot) with columns [MFS (+harmonic) | 2x MV]
    Regularization: lam on the first (M_tot-2) columns, lam*reg_gamma_a on the last 2 columns.
    Constraints: list of (c_vec, d_scalar) with c_vec length M_tot.
    """
    Nrows, Mtot = A_aug.shape
    Wv  = jnp.asarray(W).reshape(-1)
    ATW = A_aug.T * Wv[None, :]
    # diag reg
    reg = (lam**2) * jnp.ones((Mtot,), dtype=A_aug.dtype)
    reg = reg.at[-2:].set((lam*reg_gamma_a)**2)   # gentler reg on 'a'
    NE  = ATW @ A_aug + jnp.diag(reg)
    rhs = jnp.zeros((Mtot,), dtype=A_aug.dtype)   # zero RHS now

    if not constraints:
        L = jnp.linalg.cholesky(NE)
        y = jsp_linalg.solve_triangular(L, rhs, lower=True)
        x = jsp_linalg.solve_triangular(L.T, y, lower=False)
        if verbose:
            condNE = float(np.linalg.cond(np.asarray(NE)))
            lw2 = float(jnp.sqrt(jnp.dot(Wv, (A_aug @ x)**2)))
            print(f"[LS-(α,a)] size={NE.shape}, cond≈{condNE:.3e}, λ={lam:.3e}, ||W^1/2 res||={lw2:.3e}")
        return x

    # Schur complement for constraints: minimize with C^T x = d
    Ccols, dlist = [], []
    for (c_vec, d_k) in constraints:
        c = jnp.asarray(c_vec).reshape(-1)
        if c.shape[0] != Mtot:
            raise ValueError(f"[constraints] length {c.shape[0]} != columns {Mtot}")
        Ccols.append(c[:, None]); dlist.append(d_k)
    C = jnp.concatenate(Ccols, axis=1)           # (Mtot, K)
    d = jnp.asarray(dlist, dtype=A_aug.dtype)    # (K,)

    L = jnp.linalg.cholesky(NE)
    def solve_NE(b):
        y = jsp_linalg.solve_triangular(L, b, lower=True)
        return jsp_linalg.solve_triangular(L.T, y, lower=False)

    z1 = solve_NE(rhs)      # NE^{-1} rhs = 0 → z1 = 0 (kept explicit for clarity)
    Z  = solve_NE(C)        # NE^{-1} C
    S  = C.T @ Z            # C^T NE^{-1} C
    rhsμ = d - (C.T @ z1)   # = d
    μ = jnp.linalg.solve(S, rhsμ)
    x = z1 - Z @ μ          # = - Z μ

    if verbose:
        res = A_aug @ x
        lw2 = float(jnp.sqrt(jnp.dot(Wv, res**2)))
        condNE = float(np.linalg.cond(np.asarray(NE)))
        condS  = float(np.linalg.cond(np.asarray(S)))
        crel = float(jnp.linalg.norm(C.T @ x - d) / (jnp.linalg.norm(d) + 1e-30))
        print(f"[LS-(α,a):Schur] NE cond≈{condNE:.3e}, S cond≈{condS:.3e}, λ={lam:.3e}, ||W^1/2 res||={lw2:.3e}, ||C^T x - d||/||d||={crel:.3e}")
    return x


def make_objective_for_sf_lam_eps(P, N, Pn, Nn, W, rk, scinfo,
                                  grad_t, grad_p, a, g_raw,
                                  eps_bounds=(2e-3, 5e-2),
                                  h_grads=None, debug_opt=True):

    # Precompute: normalized X and a ring weight that does NOT depend on eps (cheap, stable)
    Xn_bdry = (P - scinfo.center) * scinfo.scale
    # Use boundary quadrature W as a proxy on the ring objective (keeps objective smooth)
    W_ring_fixed = W

    def objective(p):
        log_sf, log_lam, log_eps = p
        sf   = jnp.exp(log_sf)
        lam  = jnp.exp(log_lam)
        epsn = jnp.exp(log_eps)  # normalized offset (h units)
        # clamp epsn softly by projecting p (handled outside); here just use epsn directly

        Yn, _ = build_mfs_sources(Pn, Nn, rk, scinfo, source_factor=sf, verbose=False)

        # Boundary collocation A
        A, _ = build_system_matrices(
            Pn, Nn, Yn, W, grad_t, grad_p, scinfo,
            use_mv=True, center_D=True, verbose=False, h_grads=h_grads
        )
        c_bdry, d_bdry = make_flux_constraint(A, W, g_raw)

        # Interior ring: P_in depends smoothly on epsn
        eps_w = epsn / scinfo.scale
        P_in  = P - eps_w * N
        Xn_in = (P_in - scinfo.center) * scinfo.scale
        A_in  = build_A_rows_at_points(Xn_in, N, Yn, scinfo, h_grads=h_grads)
        # MV on the ring
        Gt_in = grad_t(Xn_in)
        Gp_in = grad_p(Xn_in)
        g_in  = scinfo.scale * jnp.sum(N * (a[0]*Gt_in + a[1]*Gp_in), axis=1)
        c_int, d_int = make_flux_constraint(A_in, W_ring_fixed, g_in)

        # α solving with both flux constraints
        alpha = solve_alpha_with_rhs_hard_flux_multi(
            A, W, g_raw, lam=lam, constraints=[(c_int, -d_int), (c_bdry, -d_bdry)], verbose=False
        )

        # 1) Boundary residual (weighted)
        res_w = jnp.sqrt(W) * (A @ alpha + g_raw)
        term_res = res_w @ res_w

        # 2) Interior BC penalty (n·∇φ on ring)
        n_dot_in = A_in @ alpha + g_in
        term_bc  = jnp.dot(W_ring_fixed, n_dot_in**2)

        # Tiny reg on logs
        reg = 1e-6 * (log_sf**2 + log_lam**2 + log_eps**2)
        total = term_res + term_bc + reg

        if debug_opt:
            jax_debug.print(
                "[OPT-3D] sf={sf:.3f} lam={lam:.3e} eps_n={eps:.3e}  "
                "res={res:.3e}  bc={bc:.3e}  total={tot:.3e}",
                sf=sf, lam=lam, eps=epsn, res=term_res, bc=term_bc, tot=total
            )
        return total

    # return value_and_grad
    return jit(jax.value_and_grad(objective)), eps_bounds

# ----------------------------- I/O ----------------------------- #
def load_surface_xyz_normals(xyz_csv, nrm_csv, verbose=True):
    P = np.loadtxt(xyz_csv, delimiter=",", skiprows=1)
    N = np.loadtxt(nrm_csv, delimiter=",", skiprows=1)
    assert P.shape[1] == 3 and N.shape[1] == 3, "CSV must have 3 columns"
    nrm = N / np.maximum(1e-15, np.linalg.norm(N, axis=1, keepdims=True))
    if verbose:
        print(f"[LOAD] points: {P.shape}, normals: {N.shape}")
        print(f"[LOAD] point extents (min..max) per axis:")
        for k, nm in enumerate("xyz"):
            print(f"       {nm}: {P[:,k].min():.6g} .. {P[:,k].max():.6g}")
        nlen = np.linalg.norm(nrm, axis=1)
        print(f"[LOAD] normal lengths: min={nlen.min():.3g}, max={nlen.max():.3g}, mean={nlen.mean():.3g}")
    return jnp.asarray(P, dtype=jnp.float64), jnp.asarray(nrm, dtype=jnp.float64)

# --------------------- Normalization (scale) -------------------- #
def normalize_geometry(P, verbose=True):
    c = jnp.mean(P, axis=0)
    r = jnp.linalg.norm(P - c, axis=1)
    r_med = jnp.median(r)                   # keep as JAX scalar
    s = 1.0 / jnp.maximum(r_med, 1e-12)     # JAX scalar (shape ())
    Pn = (P - c) * s
    if verbose:
        print(f"[SCALE] center={np.array(c)}, median radius={float(np.asarray(r_med)):.6g}, "
              f"scale={float(np.asarray(s)):.6g} (so median radius→1)")
    return Pn, ScaleInfo(center=c, scale=s)

# -------------------- Geometry and angles ---------------------- #
def detect_geometry_and_axis(Pn, verbose=True):
    """
    Use PCA singular values to pick an axis and a geometry kind:
      - 'torus': s1 ~ s2 >> s3  → axis = e3  (smallest variance; surface normal to best-fit plane)
      - 'mirror': s1 >> s2 ~ s3 → axis = e1  (largest variance; long direction)
    If ambiguous, default to torus behavior (axis=e3).
    Returns (kind, a_hat [3,], E [3x3], singvals [3], center c).
    """
    c_np, E_np, pca = best_fit_axis(np.array(Pn), verbose=False)
    svals = np.array(pca.singular_values_)      # already sorted in best_fit_axis
    # We sorted desc in best_fit_axis via 'order'; reconstruct the same order here:
    order = np.argsort(-pca.singular_values_)
    s = svals[order]  # s[0]>=s[1]>=s[2]
    E = np.asarray(E_np)                         # columns e1,e2,e3
    e1, e2, e3 = E[:,0], E[:,1], E[:,2]

    # Heuristics with margins (tune if needed)
    ratio_long = s[0] / max(s[1], 1e-12)
    ratio_thin = s[1] / max(s[2], 1e-12)

    if ratio_long > 2.0 and ratio_thin < 1.8:
        kind = "mirror"   # one long direction >> two comparable transverse
        a_hat = jnp.asarray(e1 / (np.linalg.norm(e1) + 1e-30))
    elif ratio_thin > 2.0 and ratio_long < 1.8:
        kind = "torus"    # thin shell: two wide directions >> one thin
        a_hat = jnp.asarray(e3 / (np.linalg.norm(e3) + 1e-30))
    else:
        # Ambiguous → fall back to torus-like choice
        kind = "torus"
        a_hat = jnp.asarray(e3 / (np.linalg.norm(e3) + 1e-30))

    if verbose:
        print(f"[GEOM] s (desc) = {s}; ratio_long={ratio_long:.2f}, ratio_thin={ratio_thin:.2f}")
        print(f"[GEOM] kind={kind}, axis=a_hat = {np.array(a_hat)}")

    return kind, a_hat, jnp.asarray(E_np), jnp.asarray(c_np), jnp.asarray(s)

def best_fit_axis(points, verbose=True):
    c = np.mean(points, axis=0)
    X = points - c
    pca = PCA(n_components=3).fit(X)
    # Sort by singular value DESC:
    order = np.argsort(-pca.singular_values_)
    comps = pca.components_[order]  # comps[0], comps[1] = in-plane; comps[2] = smallest
    # e3 = plane normal (smallest variance)
    e3 = comps[2]
    # make a right-handed in-plane frame
    e1 = comps[0] - np.dot(comps[0], e3) * e3
    e1 /= np.linalg.norm(e1) + 1e-30
    e2 = np.cross(e3, e1)
    E = np.stack([e1, e2, e3], axis=1)
    if verbose:
        print(f"[PCA] singular values: {pca.singular_values_}")
        print(f"[PCA] var ratios: {pca.explained_variance_ratio_}")
        print("[AXES] Using e3 = smallest-singular-vector (plane normal).")
    return jnp.asarray(c), jnp.asarray(E), pca

@jit
def project_to_local(P, c, E): return (P - c) @ E

@jit
def cylindrical_angle_and_radius(local_pts):
    x, y = local_pts[:,0], local_pts[:,1]
    theta = jnp.arctan2(y, x)
    rho   = jnp.sqrt(jnp.maximum(1e-30, x*x + y*y))
    return theta, rho

def detect_if_angle_is_meaningful(theta, label):
    th = np.unwrap(np.array(theta)); span = np.max(th) - np.min(th)
    print(f"[MV] {label}: angular span ~ {span:.3f} rad (~{span*180/np.pi:.1f} deg)")
    return span > (200.0*np.pi/180.0)

def order_points_by_knn_chain(P_loop, k=8):
    """Greedy k-NN chaining to order a near-1D band of points into a polyline."""
    if len(P_loop) < 3: return P_loop
    nbrs = NearestNeighbors(n_neighbors=min(k, len(P_loop))).fit(P_loop)
    visited = np.zeros(len(P_loop), dtype=bool)
    path = [0]; visited[0] = True
    for _ in range(1, len(P_loop)):
        i = path[-1]
        dists, idxs = nbrs.kneighbors(P_loop[i:i+1], return_distance=True)
        # pick nearest unvisited
        for j in idxs[0]:
            if not visited[j]:
                path.append(j); visited[j] = True; break
        else:
            # fallback: pick any unvisited
            j = int(np.where(~visited)[0][0])
            path.append(j); visited[j] = True
    return P_loop[path]

def estimate_periods_on_cycles(P, N, grad_fn, a_hat, scinfo, n_bins=60):
    """
    Build two near-closed paths from the boundary cloud:
      - Toroidal loop: points near constant azimuth φ_a ≈ φ0 around a_hat
      - Poloidal loop: points near constant poloidal θ ≈ θ0 (built w.r.t. a_hat frame)
    Then do midpoint-rule ∮ grad·dl on each ordered polyline.
    """
    P_np = np.asarray(P); N_np = np.asarray(N)
    # Axis-aware angles
    phi_a, theta, s = angles_for_axis(jnp.asarray(P_np), jnp.asarray(N_np), jnp.asarray(a_hat),
                                      center=jnp.asarray(scinfo.center))
    phi_a = np.asarray(phi_a); theta = np.asarray(theta)

    # Choose bins centered near 0 for both angles
    def select_band(vals, half_width):
        # unwrap to be stable near 0
        v = np.unwrap(vals)
        return np.abs(v - np.median(v)) <= half_width

    # band widths
    dphi = np.pi / n_bins
    dth  = np.pi / n_bins

    # Toroidal loop: narrow band in φ_a
    mask_t = select_band(phi_a, dphi/2)
    P_t = P_np[mask_t]
    # Poloidal loop: narrow band in θ
    mask_p = select_band(theta, dth/2)
    P_p = P_np[mask_p]

    # Order points into polylines to avoid zig-zags
    P_t = order_points_by_knn_chain(P_t, k=8)
    P_p = order_points_by_knn_chain(P_p, k=8)

    def line_integral(P_loop):
        if len(P_loop) < 4:
            return 0.0
        G = np.asarray(grad_fn(jnp.asarray(P_loop)))
        dP = np.diff(P_loop, axis=0)
        Gmid = 0.5*(G[1:,:] + G[:-1,:])
        return float(np.sum(np.einsum('ij,ij->i', Gmid, dP)))

    Pi_t = line_integral(P_t)
    Pi_p = line_integral(P_p)
    print(f"[PERIOD] line integrals: toroidal≈{Pi_t:.6e}, poloidal≈{Pi_p:.6e}")
    return Pi_t, Pi_p

def solve_ap_at_from_periods(Pi_t, Pi_p):
    """
    For the multivalued basis grad φ ≈ a_t grad_t + a_p grad_p + single-valued,
    we lock periods with a 2x2 identity mapping (assumes period_t ~ 2π a_t, period_p ~ 2π a_p).
    """
    A = np.array([[2*np.pi, 0.0],[0.0, 2*np.pi]])
    b = np.array([Pi_t, Pi_p])
    a_t, a_p = np.linalg.solve(A, b)
    print(f"[PERIOD] locked (a_t,a_p)=({a_t:.6e},{a_p:.6e}) from periods.")
    return jnp.array([a_t, a_p], dtype=jnp.float64)

@jit
def grad_theta_world_from_plane(local_pts, E_plane_cols, rho_floor):
    e1 = E_plane_cols[:,0]; e2 = E_plane_cols[:,1]
    x, y = local_pts[:,0], local_pts[:,1]
    rho2 = x*x + y*y
    # soft clamp: rho_eff^2 = rho^2 + rho_floor^2  (keeps direction, limits magnitude)
    rho2_eff = rho2 + (rho_floor * rho_floor)
    a = -y / rho2_eff
    b =  x / rho2_eff
    return a[:,None]*e1[None,:] + b[:,None]*e2[None,:]

# ------------------ Multivalued bases (2D) ---------------------- #
def fit_mv_coeffs_minimize_rhs(Nn, W, grad_t_bdry, grad_p_bdry, verbose=True):
    """
    Weighted LS *directional* fit for a=(a_t,a_p), using centered columns
    to reduce g = n·(a_t grad_t + a_p grad_p). Returns a and centered D.
    """
    Dt = jnp.sum(Nn * grad_t_bdry, axis=1)
    Dp = jnp.sum(Nn * grad_p_bdry, axis=1)
    D = jnp.stack([Dt, Dp], axis=1)  # (N,2)

    # Weighted centering → columns zero-mean w.r.t. W
    Wsum = jnp.sum(W) + 1e-30
    mu = (W @ D) / Wsum               # (2,)
    D0 = D - mu[None, :]

    # Work with weighted data
    Wsqrt = jnp.sqrt(W)
    Dw0 = D0 * Wsqrt[:, None]

    # Leading singular vector: direction that reduces magnitude fastest
    U, S, Vt = jnp.linalg.svd(Dw0, full_matrices=False)
    a_dir = -Vt[0, :]

    # Scale so ||D0 a||_W2 hits a reasonable target (median column norm / 2)
    col_w2 = jnp.array([jnp.sqrt(jnp.dot(W, D0[:,0]**2)), jnp.sqrt(jnp.dot(W, D0[:,1]**2))])
    target = 0.5 * float(jnp.median(col_w2))
    denom = float(jnp.sqrt(jnp.dot(W, (D0 @ a_dir)**2)) + 1e-30)
    scale = target / denom
    a = scale * a_dir

    if verbose:
        g_fit = D0 @ a
        vec_stats("[MV-FIT] g(a)=n·∇φ_mv (centered)", g_fit, W)
        print(f"[MV-FIT] a_t={float(a[0]):.6g}, a_p={float(a[1]):.6g}")
    return a, D, D0

@jit
def grad_azimuth_about_axis(Xn, a_hat):
    """
    ∇ϕ_a for azimuth around an arbitrary unit axis a_hat.
      r_perp = X - (X·a)a
      ∇ϕ_a = (a × r_perp) / |r_perp|^2
    """
    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par   = jnp.sum(Xn * a[None,:], axis=1, keepdims=True) * a[None,:]
    r_perp  = Xn - r_par
    r2      = jnp.maximum(1e-30, jnp.sum(r_perp*r_perp, axis=1, keepdims=True))
    cross   = jnp.cross(a[None,:], r_perp)
    return cross / r2

def multivalued_bases_about_axis(Pn, Nn, a_hat, verbose=True):
    """
    Toroidal-like multivalued set defined around the chosen axis a_hat.
      - grad_t(X) = ∇ϕ_a(X) with azimuth around a_hat
      - grad_p(X) = θ̂(X): build from n × ϕ̂_tan, where ϕ̂_tan is azimuth unit projected to tangent plane
    This reduces to your old bases when a_hat = ẑ.
    """
    Pn_ref = jnp.asarray(Pn)
    Nn_ref = jnp.asarray(Nn)
    a_hat  = jnp.asarray(a_hat)

    @jit
    def _nearest_normal_jax(Xn):
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
        return v - jnp.sum(v*n, axis=1, keepdims=True)*n

    def grad_t(Xn):
        return grad_azimuth_about_axis(Xn, a_hat)

    def grad_p(Xn):
        # Build φ̂_a (azimuth unit) then project to tangent and make θ̂
        n = _nearest_normal_jax(Xn)
        # φ̂_a = unit(a × r_perp)
        a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
        r_par   = jnp.sum(Xn * a[None,:], axis=1, keepdims=True) * a[None,:]
        r_perp  = Xn - r_par
        phi_hat = _unit(jnp.cross(a[None,:], r_perp))
        phi_tan = _unit(_project_tangent(phi_hat, n))
        theta_hat = _unit(jnp.cross(n, phi_tan))
        return theta_hat

    def phi_t(Xn):
        # Unused value basis for consistency; keeping interface
        return jnp.zeros((Xn.shape[0],), dtype=jnp.float64)

    def phi_p(Xn):
        return jnp.zeros((Xn.shape[0],), dtype=jnp.float64)

    if verbose:
        print("[MV] Using axis-aware multivalued bases around detected axis a_hat.")

    return (phi_t, grad_t, phi_p, grad_p)

# --- NEW: a fallback nontrivial multivalued amplitude if everything looks zero-ish ---
def default_axis_aware_a(a_hat, prefer_toroidal=True):
    """
    Pick a deterministic, geometry-aware 'a' so φ_mv is nontrivial even when
    periods are not prescribed. For tori, set a_t = 1, a_p = 0 by default.
    For 'mirror' shapes you can flip the preference if desired.
    """
    if prefer_toroidal:
        return jnp.array([1.0, 0.0], dtype=jnp.float64)
    else:
        return jnp.array([0.0, 1.0], dtype=jnp.float64)


# -------------------------- kNN scales & weights ------------------------ #
def kNN_geometry_stats(Pn, k=48, verbose=True):
    c, E, _ = best_fit_axis(np.array(Pn), verbose=False)
    Ploc = project_to_local(Pn, c, E)
    XY = np.asarray(Ploc[:, :2])
    k_eff = min(k+1, len(XY))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree").fit(XY)
    dists, _ = nbrs.kneighbors(XY)
    rk = dists[:, -1]                              # k-th neighbor radius in local plane
    W = jnp.asarray(np.pi * rk**2, dtype=jnp.float64)
    if verbose:
        print(f"[QUAD] k-NN k={k}, area stats: min={float(W.min()):.3g}, max={float(W.max()):.3g}, median={float(jnp.median(W)):.3g}")
        print(f"[QUAD] k-NN radius stats: min={float(rk.min()):.3g}, max={float(rk.max()):.3g}, median={float(np.median(rk)):.3g}")
        print(f"[QUAD] total area estimate (sum W)≈{float(jnp.sum(W)):.3f}")
    return W, rk

# ----------------------------- Kernels -------------------------- #
@jit
def green_G(x, y):
    r = jnp.linalg.norm(x - y, axis=-1)
    return 1.0 / (4.0 * jnp.pi * jnp.maximum(1e-30, r))

@jit
def grad_green_x(x, y):
    r = x - y
    r2 = jnp.sum(r*r, axis=-1)
    r3 = jnp.maximum(1e-30, r2 * jnp.sqrt(r2))
    return - r / (4.0 * jnp.pi * r3[..., None])

# ------------------ Tiny harmonic polynomial pack ------------------ #
# We use 6 linearly independent trace-free quadratics (all ∇² h_k = 0):
#   h1 = x*y,  h2 = y*z,  h3 = z*x,
#   h4 = x^2 - y^2,  h5 = y^2 - z^2,  h6 = z^2 - x^2  (only 2 of these are independent, but
#   the set of 3 is fine; the pack is kept tiny and stabilized by Tikhonov)
# We also provide their gradients. Everything is in *normalized* coordinates Xn.
def get_harmonic_pack():
    @jit
    def h_vals(Xn):  # (N,3) -> (N, K=6)
        x, y, z = Xn[:,0], Xn[:,1], Xn[:,2]
        return jnp.stack([
            x*y,          # h1
            y*z,          # h2
            z*x,          # h3
            x*x - y*y,    # h4
            y*y - z*z,    # h5
            z*z - x*x,    # h6
        ], axis=1)
    @jit
    def h_grads(Xn):  # (N,3) -> (N,6,3)
        x, y, z = Xn[:,0], Xn[:,1], Xn[:,2]
        # gradients per column (K=6)
        g1 = jnp.stack([y, x, jnp.zeros_like(x)], axis=1)          # ∇(x y)
        g2 = jnp.stack([jnp.zeros_like(x), z, y], axis=1)          # ∇(y z)
        g3 = jnp.stack([z, jnp.zeros_like(x), x], axis=1)          # ∇(z x)
        g4 = jnp.stack([2*x, -2*y, jnp.zeros_like(x)], axis=1)     # ∇(x^2 - y^2)
        g5 = jnp.stack([jnp.zeros_like(x), 2*y, -2*z], axis=1)     # ∇(y^2 - z^2)
        g6 = jnp.stack([-2*x, jnp.zeros_like(x), 2*z], axis=1)     # ∇(z^2 - x^2)
        return jnp.stack([g1, g2, g3, g4, g5, g6], axis=1)         # (N,6,3)
    return h_vals, h_grads

# ------------------------- MFS source cloud ---------------------- #
def build_mfs_sources(Pn, Nn, rk, scale_info, source_factor=2.0, verbose=True):
    """
    Adaptive MFS sources: Y_i = P_i + δ_i N_i with δ_i = source_factor * rk_i
    where rk_i is the local kNN radius in the best-fit (u,v) plane.
    This makes δ smaller in the figure-8 neck and larger in the lobes.
    """
    # per-point δ_i
    delta_n_i = source_factor * rk.reshape(-1)         # (N,)
    Yn = Pn + delta_n_i[:, None] * Nn                  # (N,3)

    if verbose:
        dn_med = float(np.median(np.asarray(delta_n_i)))
        dn_min = float(np.min(np.asarray(delta_n_i)))
        dn_max = float(np.max(np.asarray(delta_n_i)))
        print(f"[MFS] Using adaptive source offsets δ_i (normalized): "
              f"median={dn_med:.4g}, min={dn_min:.4g}, max={dn_max:.4g}")

    return Yn, delta_n_i

# ----------------------------- System build ---------------------- #
@partial(jax.jit, static_argnames=("grad_t", "grad_p", "use_mv", "center_D", "verbose", "h_grads"))
def build_system_matrices(Pn, Nn, Yn, W, grad_t, grad_p, scinfo,
                          use_mv=True, center_D=True, verbose=True,
                          h_grads=None):
    """
    Build collocation matrix for Neumann BC:
      A_ij = n_i · ∇_x G( x_i , y_j ),  x_i = Pn[i], y_j = Yn[j]
    If h_grads is not None, append the K harmonic-gradient columns exactly
    like build_A_rows_at_points does, so A has shape (N, M+K).
    Returns:
      A  : (N, M[+K]) in WORLD units
      D  : kept for interface compatibility (zeros)
    """
    X = Pn

    @jit
    def row_kernel(xi, ni):
        grads = vmap(lambda yj: grad_green_x(xi, yj))(Yn)   # (M,3)
        return jnp.dot(grads, ni)                           # (M,)

    # Base MFS block (N, M)
    A_mfs = vmap(row_kernel)(X, Nn)
    A_mfs = scinfo.scale * A_mfs

    # Optionally append harmonic columns (N, K)
    if h_grads is not None:
        G = h_grads(X)                                     # (N, K, 3)  NOTE: X is normalized coords
        H = jnp.sum(G * Nn[:, None, :], axis=2)            # (N, K)
        A = jnp.concatenate([A_mfs, scinfo.scale * H], axis=1)
    else:
        A = A_mfs

    if verbose:
        n, m = A.shape
        Amin = jnp.min(jnp.abs(A)); Amed = jnp.median(jnp.abs(A)); Amax = jnp.max(jnp.abs(A))
        jax_debug.print(
            "[SYS] A shape=({n},{m}), |A| stats: min={mn:.3e}, median={md:.3e}, max={mx:.3e}",
            n=n, m=m, mn=Amin, md=Amed, mx=Amax)

    D = jnp.zeros((Pn.shape[0], 2), dtype=jnp.float64)
    return A, D

# ----------------------- Regularized weighted LS ------------------------ #
def solve_alpha_with_rhs(A, W, g_raw, lam=1e-3, verbose=True):
    """ Solve min || W^{1/2}(A α + g_raw)||_2^2 + λ^2 ||α||_2^2 for α only. """
    Wsqrt = jnp.sqrt(W)
    Aw = Wsqrt[:, None] * A
    gw = Wsqrt * g_raw
    ATA = Aw.T @ Aw
    ATg = Aw.T @ gw
    NE = ATA + (lam**2) * jnp.eye(A.shape[1], dtype=A.dtype)
    rhs = -ATg
    condNE = np.linalg.cond(np.asarray(NE))
    if verbose:
        print(f"[LS-α] size={NE.shape}, cond≈{condNE:.3e}, λ={lam:.3e}")
    L = jnp.linalg.cholesky(NE)
    y = jsp_linalg.solve_triangular(L, rhs, lower=True)
    alpha = jsp_linalg.solve_triangular(L.T, y, lower=False)
    res = Aw @ alpha + gw
    if verbose:
        vec_stats("[LS-α] weighted residual", res)
    return alpha

def solve_alpha_with_rhs_hard_flux_multi(A, W, g_raw, lam=1e-3, constraints=None, verbose=True):
    """
    Solve min ||W^{1/2}(A α + g)||^2 + λ^2||α||^2  s.t. C^T α = d
    using Schur complement: μ from (C^T NE^{-1} C)μ = d - C^T NE^{-1} rhs1,
    then α = NE^{-1}(rhs1 - C μ), where NE = A^T W A + λ^2 I, rhs1 = -A^T W g.
    """
    if not constraints:
        return solve_alpha_with_rhs(A, W, g_raw, lam=lam, verbose=verbose)

    Wv  = jnp.asarray(W).reshape(-1)
    ATW = A.T * Wv[None, :]
    NE  = ATW @ A + (lam**2) * jnp.eye(A.shape[1], dtype=A.dtype)
    rhs1 = - (ATW @ g_raw)

    # Build C (M x K) and d (K,)
    Ccols, dlist = [], []
    for (c_vec, d_k) in constraints:
        Ccols.append(c_vec[:, None])
        dlist.append(d_k)
    C = jnp.concatenate(Ccols, axis=1)             # (M,K)
    d = jnp.asarray(dlist, dtype=NE.dtype)         # (K,)

    # Cholesky of NE
    L = jnp.linalg.cholesky(NE)

    # Solve NE^{-1} rhs1 and NE^{-1} C (columns)
    y1 = jsp_linalg.solve_triangular(L, rhs1, lower=True)
    z1 = jsp_linalg.solve_triangular(L.T, y1, lower=False)      # z1 = NE^{-1} rhs1

    Y  = jsp_linalg.solve_triangular(L, C, lower=True)
    Z  = jsp_linalg.solve_triangular(L.T, Y, lower=False)       # Z = NE^{-1} C

    # Schur system: S μ = d - C^T z1,  where S = C^T Z = C^T NE^{-1} C
    S   = C.T @ Z                                               # (K,K)
    rhsμ = d - (C.T @ z1)
    μ   = jnp.linalg.solve(S, rhsμ)

    # α = NE^{-1}(rhs1 - C μ) = z1 - Z μ
    alpha = z1 - Z @ μ

    if verbose:
        res   = A @ alpha + g_raw
        lw2   = float(jnp.sqrt(jnp.dot(Wv, res**2)))
        condNE = float(np.linalg.cond(np.asarray(NE)))
        condS  = float(np.linalg.cond(np.asarray(S)))
        print(f"[LS-α:Schur] NE cond≈{condNE:.3e}, S cond≈{condS:.3e}, λ={lam:.3g}, ||W^0.5 res||={lw2:.3e}")
    return alpha

def _prefactor_constrained_system(A, W, lam, constraints):
    """
    Precompute objects we can reuse when only the RHS changes:
      NE_base = A^T W A + λ^2 I,  L_base = chol(NE_base),
      C, Z = NE_base^{-1} C, and G = C^T Z.
    """
    Wv  = jnp.asarray(W).reshape(-1)
    ATW = A.T * Wv[None, :]
    ATA = ATW @ A
    NE_base = ATA + (lam**2) * jnp.eye(A.shape[1], dtype=A.dtype)
    L_base  = jnp.linalg.cholesky(NE_base)

    # Pack constraints into a tall matrix C (M x K) and vector d (K,)
    Ccols, dlist = [], []
    for (c_vec, d_k) in (constraints or []):
        Ccols.append(jnp.asarray(c_vec).reshape(-1, 1))
        dlist.append(d_k)
    C = jnp.concatenate(Ccols, axis=1) if Ccols else jnp.zeros((A.shape[1],0), dtype=A.dtype)
    d = jnp.asarray(dlist, dtype=A.dtype) if dlist else jnp.zeros((0,), dtype=A.dtype)

    # helpers using the factor
    def solve_base(rhs):
        y = jsp_linalg.solve_triangular(L_base, rhs, lower=True)
        return jsp_linalg.solve_triangular(L_base.T, y, lower=False)

    Z = solve_base(C) if C.shape[1] > 0 else C
    G = C.T @ Z if C.shape[1] > 0 else jnp.zeros((0,0), dtype=A.dtype)

    return dict(L_base=L_base, C=C, d=d, Z=Z, G=G, ATW=ATW)

def fast_constrained_resolve_with_new_g(A, W, g_raw, lam, prefac):
    """
    Given the same A, W, λ, constraints, and prefactor (from _prefactor_constrained_system),
    quickly solve the constrained Tikhonov system for a NEW g_raw (e.g., after MV-LOCK).
    This is a single Schur solve; no AL iterations.
    """
    L_base = prefac["L_base"]; C = prefac["C"]; d = prefac["d"]; Z = prefac["Z"]; G = prefac["G"]; ATW = prefac["ATW"]

    rhs1 = - (ATW @ jnp.asarray(g_raw))
    # NE^{-1} rhs1
    y1 = jsp_linalg.solve_triangular(L_base, rhs1, lower=True)
    z1 = jsp_linalg.solve_triangular(L_base.T, y1, lower=False)

    if C.shape[1] == 0:
        return z1

    # Schur solve: μ from G μ = d - C^T z1
    rhsμ = d - (C.T @ z1)
    μ    = jnp.linalg.solve(G, rhsμ)

    # α = z1 - Z μ
    return z1 - Z @ μ

def augmented_lagrangian_solve(
    A, W, g_raw, lam, constraints,
    rho0=1e-2, rho_max=1e3, iters=5, verbose=True,
    tol_c=1e-3, tol_res_rel=1e-3
):
    # Pack constraints
    M = A.shape[1]
    Ccols, dlist = [], []
    for (c_vec, d_k) in (constraints or []):
        c_vec = jnp.asarray(c_vec).reshape(-1)
        if c_vec.shape[0] != M:
            raise ValueError(
                f"[constraints] c_vec has length {c_vec.shape[0]} but must match A.shape[1] = {M}. "
                f"One of the constraints likely passed a transposed A_like into make_flux_constraint."
            )
        Ccols.append(c_vec[:, None])
        dlist.append(d_k)
    C = jnp.concatenate(Ccols, axis=1)  # (M, K)
    d = jnp.asarray(dlist, dtype=A.dtype)      # (K,)

    Wv  = jnp.asarray(W).reshape(-1)
    ATW = A.T * Wv[None, :]
    ATA = ATW @ A
    ATg = ATW @ g_raw

    # Pre-factor base normal equations
    NE_base = ATA + (lam**2) * jnp.eye(A.shape[1], dtype=A.dtype)
    L_base  = jnp.linalg.cholesky(NE_base)

    def solve_base(rhs):
        y = jsp_linalg.solve_triangular(L_base, rhs, lower=True)
        return jsp_linalg.solve_triangular(L_base.T, y, lower=False)

    # Precompute Z = NE_base^{-1} C  and the small Gram G = C^T Z
    Z = solve_base(C)                # (M, K)
    G = C.T @ Z                      # (K, K)

    mu = jnp.zeros((C.shape[1],), dtype=A.dtype)
    rho = rho0
    alpha = jnp.zeros((A.shape[1],), dtype=A.dtype)

    prev_lw2 = None
    for it in range(iters):
        # Target rhs: NE(ρ) α = -ATg - C (mu + ρ d)
        rhs = -ATg - C @ (mu + rho * d)

        # Woodbury: (NE_base + ρ C C^T)^{-1} rhs
        # = solve_base(rhs) - Z * ( (I/ρ + G)^{-1} * (C^T * solve_base(rhs)) ) / 1
        z1 = solve_base(rhs)                      # (M,)
        Ct_z1 = C.T @ z1                          # (K,)
        Sρ = (G + (1.0 / rho) * jnp.eye(G.shape[0], dtype=G.dtype))  # (K,K)
        y_small = jnp.linalg.solve(Sρ, Ct_z1)     # (K,)
        alpha = z1 - Z @ y_small                  # (M,)

        # Constraint residual and multiplier update
        r = (C.T @ alpha) - d
        mu = mu + rho * r

        # Diagnostics
        lw2 = float(jnp.sqrt(jnp.dot(Wv, (A @ alpha + g_raw)**2)))
        crel = float(jnp.linalg.norm(r) / (jnp.linalg.norm(d) + 1e-30))

        if verbose:
            print(f"[AL] it={it}  ||W^1/2 res||={lw2:.3e}  ||C^Tα-d||/||d||={crel:.3e}  rho={rho:.2e}")

        # --- Early exit: small constraint violation and diminishing residual improvement ---
        if crel < tol_c and (prev_lw2 is not None):
            rel_impr = abs(lw2 - prev_lw2) / (abs(prev_lw2) + 1e-30)
            if rel_impr < tol_res_rel:
                if verbose:
                    print(f"[AL] early-exit: crel={crel:.3e}, Δres/res={rel_impr:.3e}")
                break

        prev_lw2 = lw2
        rho = min(rho * 10.0, rho_max)

    return alpha

@partial(jax.jit, static_argnames=("h_grads",))
def build_A_rows_at_points(Xn_eval, N_eval, Yn, scinfo, h_grads=None):
    @jit
    def row_kernel(xi, ni):
        grads = vmap(lambda yj: grad_green_x(xi, yj))(Yn)   # (M,3)
        return jnp.dot(grads, ni)                           # (M,)
    A_in = vmap(row_kernel)(Xn_eval, N_eval)                # (N,M_mfs)
    A_in = scinfo.scale * A_in
    if h_grads is not None:
        G = h_grads(Xn_eval)                                # (N,K,3)
        H = jnp.sum(G * N_eval[:, None, :], axis=2)         # (N,K)
        A_in = jnp.concatenate([A_in, scinfo.scale * H], axis=1)  # (N, M_mfs+K)
    return A_in

def make_flux_constraint(A_like, W_like, g_like):
    """
    Return (c_vec, d_scalar) for flux-compatibility:
      c_vec = A^T W 1   (shape M,)
      d     = W·g       (scalar)
    Robust to A_like being (N,M) or (M,N).
    """
    Wv = jnp.asarray(W_like).reshape(-1)
    A  = jnp.asarray(A_like)
    g  = jnp.asarray(g_like)

    # Case 1: A is (N, M) and len(W) == N  → expected
    if A.ndim == 2 and A.shape[0] == Wv.shape[0]:
        c_vec = (A.T * Wv[None, :]).sum(axis=1)       # (M,)
        d_val = jnp.dot(Wv, g)
        return c_vec, d_val

    # Case 2: A is (M, N) and len(W) == N  → transposed input; handle gracefully
    if A.ndim == 2 and A.shape[1] == Wv.shape[0]:
        c_vec = (A * Wv[None, :]).sum(axis=1)         # (M,)
        d_val = jnp.dot(Wv, g)
        return c_vec, d_val

    # Otherwise, shapes are inconsistent; raise a clear error
    raise ValueError(
        f"make_flux_constraint: incompatible shapes. "
        f"A_like shape={A.shape}, len(W_like)={Wv.shape[0]}, len(g_like)={g.shape[0]} "
        f"(expected len(W)=rows(A) or cols(A))"
    )

# ----------------- Evaluators & Laplacian(ψ) ------------------- #
def build_evaluators_mfs_plusH(Pn, Yn, alpha_full, phi_t, phi_p, a, scinfo: ScaleInfo,
                               grad_t_fn, grad_p_fn, h_vals=None, h_grads=None):
    Y = Yn
    # sizes inferred from closed-over tensors/functions (static wrt trace)
    M_mfs = Y.shape[0]
    if h_grads is None:
        K_harm = 0
    else:
        # infer K by calling once on a dummy point (shape-safe and cheap)
        K_harm = int(h_grads(jnp.zeros((1, 3), dtype=Y.dtype)).shape[1])

    @jax.jit
    def split_alpha(a_vec):
        alpha_mfs = jax.lax.dynamic_slice_in_dim(a_vec, 0, M_mfs)
        beta_h = (jax.lax.dynamic_slice_in_dim(a_vec, M_mfs, K_harm)
                  if K_harm > 0 else jnp.zeros((0,), dtype=a_vec.dtype))
        return alpha_mfs, beta_h

    @jit
    def S_alpha_at(xn, alpha_mfs, beta_h):
        Gvals = vmap(lambda y: green_G(xn, y))(Y)                # (M_mfs,)
        val = jnp.dot(Gvals, alpha_mfs)
        if K_harm > 0 and h_vals is not None:
            hv = h_vals(xn[None, :])[0]                          # (K,)
            val = val + jnp.dot(hv, beta_h)
        return val

    @jit
    def grad_S_alpha_at(xn, alpha_mfs, beta_h):
        Grads = vmap(lambda y: grad_green_x(xn, y))(Y)          # (M_mfs,3)
        g = jnp.sum(Grads * alpha_mfs[:, None], axis=0)
        if K_harm > 0 and h_grads is not None:
            Hg = h_grads(xn[None, :])[0]                        # (K,3)
            g = g + jnp.dot(beta_h[None, :], Hg).reshape(3)
        return g

    def S_batch(Xn, alpha_mfs, beta_h):
        return vmap(lambda x: S_alpha_at(x, alpha_mfs, beta_h))(Xn)
    def dS_batch(Xn, alpha_mfs, beta_h):
        return vmap(lambda x: grad_S_alpha_at(x, alpha_mfs, beta_h))(Xn)

    @jit
    def phi_mv_world(X):
        Xn = (X - scinfo.center) * scinfo.scale
        return a[0]*phi_t(Xn) + a[1]*phi_p(Xn)

    # Multivalued gradients supplied by caller (already in WORLD basis)
    def grad_mv_world_batch(X):
        Xn = (X - scinfo.center) * scinfo.scale
        return scinfo.scale * (a[0]*grad_t_fn(Xn) + a[1]*grad_p_fn(Xn))

    @jit
    def psi_fn_world(X):
        Xn = (X - scinfo.center) * scinfo.scale
        alpha_mfs, beta_h = split_alpha(alpha_full)
        return S_batch(Xn, alpha_mfs, beta_h)

    @jit
    def grad_psi_fn_world(X):
        Xn = (X - scinfo.center) * scinfo.scale
        alpha_mfs, beta_h = split_alpha(alpha_full)
        return (scinfo.scale) * dS_batch(Xn, alpha_mfs, beta_h)

    @jit
    def phi_fn_world(X):
        return phi_mv_world(X) + psi_fn_world(X)

    def grad_fn_world(X):
        return grad_mv_world_batch(X) + grad_psi_fn_world(X)

    def laplacian_psi_world(X):
        def grad_at_point(x):
            return grad_psi_fn_world(x[None, :])[0]
        J = vmap(jacrev(grad_at_point))(X)   # (M,3,3)
        return jnp.trace(J, axis1=1, axis2=2)

    return phi_fn_world, grad_fn_world, psi_fn_world, grad_psi_fn_world, laplacian_psi_world, grad_mv_world_batch

# -------------------------- Orientation check -------------------- #
def maybe_flip_normals(P, N):
    c = jnp.mean(P, axis=0)
    s = jnp.sum((P - c) * N, axis=1)
    avg = float(jnp.mean(s))
    if avg < 0:
        print(f"[ORIENT] Normals inward on average (⟨(P-c)·N⟩≈{avg:.3e}) → flipping.")
        return -N, True
    print(f"[ORIENT] Normals seem outward (⟨(P-c)·N⟩≈{avg:.3e}).")
    return N, False

# --- NEW: final diagnostics dumper ---
def print_final_diagnostics(
    tag, P, N, W, W_ring, scinfo,
    eps_n, lam, source_factor, a, alpha,
    grad_fn, lap_psi_fn,
    A_aug_bdry=None, constraints=None,
    recompute_W_ring=False, k_nn_for_ring=64, Pn=None,
    Yn=None, h_grads=None, grad_t_fn=None, grad_p_fn=None, verbose_rebuild=True
):
    print(f"\n===== FINAL DIAGNOSTICS: {tag} =====")
    # ring geometry
    if recompute_W_ring:
        W_ring = build_ring_weights(P_in, Pn, k=k_nn_for_ring)
    eps_w = float(eps_n) / float(np.asarray(scinfo.scale))
    P_in  = P - eps_w * N
    grad_on_Gamma   = grad_fn(P)
    grad_on_ring    = grad_fn(P_in)
    lap_on_ring     = lap_psi_fn(P_in)

    # magnitudes
    gΓ  = jnp.linalg.norm(grad_on_Gamma, axis=1)
    gR  = jnp.linalg.norm(grad_on_ring,  axis=1)
    n·g = jnp.sum(N * grad_on_ring, axis=1)
    rn  = jnp.abs(n·g) / jnp.maximum(1e-30, gR)

    # flux neutrality on ring
    flux = float(jnp.dot(W_ring, n·g))
    area = float(jnp.sum(W_ring))

    # report
    vec_stats("[Γ] |∇φ|", gΓ)
    print(f"[Γ]   dimless 50/90/99 pct(|∇φ|/median(|∇φ|_Γ)): "
          f"{pct(gΓ/np.median(np.asarray(gΓ)),50):.3f}/"
          f"{pct(gΓ/np.median(np.asarray(gΓ)),90):.3f}/"
          f"{pct(gΓ/np.median(np.asarray(gΓ)),99):.3f}")
    vec_stats("[Γ₋] |∇φ|", gR)
    vec_stats("[Γ₋] |n·∇φ|/|∇φ|", rn)
    vec_stats("[Γ₋] |∇²ψ|", jnp.abs(lap_on_ring))
    print(f"[Γ₋] Flux neutrality: ∫ n·∇φ dS ≈ {flux:.6e} (avg={flux/area:.3e})")
    print(f"[PARAM] δ~{source_factor:.3f} (median offset), λ={lam:.3e}, ε_n={eps_n:.3e}")
    print(f"[MV] a_t={float(a[0]):.6e}, a_p={float(a[1]):.6e}, ||α||₂={float(jnp.linalg.norm(alpha)):.3e}")

    # --- Matrix condition check (robust to shape changes) ---
    if A_aug_bdry is not None:
        try:
            Wv  = jnp.asarray(W).reshape(-1)
            if A_aug_bdry.shape[0] != Wv.shape[0]:
                # Try to rebuild with CURRENT geometry to match W
                if (Pn is not None) and (Yn is not None):
                    A_re = build_A_rows_at_points(Pn, N, Yn, scinfo, h_grads=h_grads)
                    if verbose_rebuild:
                        print(f"[MATRIX] Rebuilt A for diagnostics: old rows={A_aug_bdry.shape[0]}, new rows={A_re.shape[0]}")
                    A_aug_bdry = A_re
                else:
                    print("[MATRIX] Skipping condition estimate (shape mismatch and cannot rebuild).")
                    A_aug_bdry = None
        except Exception as e:
            print("[MATRIX] Skipping condition estimate (rebuild failed):", e)
            A_aug_bdry = None

    if A_aug_bdry is not None:
        Wv  = jnp.asarray(W).reshape(-1)
        ATW = A_aug_bdry.T * Wv[None, :]
        NE  = ATW @ A_aug_bdry + 1e-30*jnp.eye(A_aug_bdry.shape[1])
        condNE = float(np.linalg.cond(np.asarray(NE)))
        print(f"[MATRIX] cond(NormalEq)≈{condNE:.3e}")

    if constraints:
        # check constraint residuals on x=[α;a]
        # We don't have x here, but you can pass it if you want exact numbers.
        print(f"[CONSTR] Count={len(constraints)} (Γ, Γ₋, caps).")
    print("===== END FINAL DIAGNOSTICS =====\n")


# -------------------------- Diagnostics/plots -------------------- #
def _angles_phi_theta(P, N=None, verbose=False):
    P = np.asarray(P)
    x,y,z = P[:,0], P[:,1], P[:,2]
    phi = np.arctan2(y, x)

    if N is None:
        # Fallback: define θ by projecting ẑ to the local plane orthogonal to ϕ̂
        r2 = np.maximum(1e-30, x*x + y*y)
        phi_hat = np.stack([-y/np.sqrt(r2), x/np.sqrt(r2), np.zeros_like(x)], axis=1)
        zhat = np.array([0.0, 0.0, 1.0])[None,:]
        # Build an orthonormal frame {θ̂, ϕ̂, ϵ̂} with θ̂ ⟂ ϕ̂ and as close to ẑ as possible
        z_proj = zhat - (np.sum(zhat*phi_hat, axis=1, keepdims=True))*phi_hat
        z_proj /= (np.linalg.norm(z_proj, axis=1, keepdims=True) + 1e-30)
        theta_hat = z_proj
        # θ is angle of ẑ projection in (θ̂, ϕ̂) basis → 0 by construction
        theta = np.zeros_like(phi)
        return phi, theta

    # Existing path with true surface normals
    Nw = np.asarray(N)
    r2 = np.maximum(1e-30, x*x + y*y)
    phi_hat = np.stack([-y/np.sqrt(r2), x/np.sqrt(r2), np.zeros_like(x)], axis=1)
    phi_tan = phi_hat - np.sum(phi_hat*Nw, axis=1, keepdims=True)*Nw
    phi_tan /= (np.linalg.norm(phi_tan, axis=1, keepdims=True) + 1e-30)
    theta_hat = np.cross(Nw, phi_tan)
    theta_hat /= (np.linalg.norm(theta_hat, axis=1, keepdims=True) + 1e-30)
    zhat = np.array([0.0, 0.0, 1.0])[None,:]
    z_tan = zhat - np.sum(zhat*Nw, axis=1, keepdims=True)*Nw
    z_tan /= (np.linalg.norm(z_tan, axis=1, keepdims=True) + 1e-30)
    num = np.sum(z_tan * phi_tan, axis=1)
    den = np.sum(z_tan * theta_hat, axis=1)
    theta = np.arctan2(num, den)
    return phi, theta

def angles_for_axis(P, N, a_hat, center=None):
    """
    Azimuth ϕ_a and poloidal θ w.r.t. axis a_hat, using (e1,e2) ⟂ a_hat
    as a stable reference. Uses (X - center) everywhere.
    Returns (phi_a, theta, s).
    """
    Xw = jnp.asarray(P)
    Nw = jnp.asarray(N)
    if center is None:
        # fall back to geometric mean if not provided
        center = jnp.mean(Xw, axis=0)
    c = jnp.asarray(center)

    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    Xc = Xw - c[None, :]

    # orthonormal complement to a:
    e1_np, e2_np = _orthonormal_complement(np.array(a_hat))
    e1 = jnp.asarray(e1_np); e2 = jnp.asarray(e2_np)

    # decompose r_perp in (e1,e2)
    r_par   = jnp.sum(Xc * a[None,:], axis=1, keepdims=True) * a[None,:]
    r_perp  = Xc - r_par
    u1 = jnp.sum(r_perp * e1[None,:], axis=1)
    u2 = jnp.sum(r_perp * e2[None,:], axis=1)
    phi_a = jnp.arctan2(u2, u1)  # azimuth around a

    # build φ̂_a and a tangent frame:
    r2 = jnp.maximum(1e-30, u1*u1 + u2*u2)[:, None]
    phi_hat = (u2[:,None]*e1[None,:] - u1[:,None]*e2[None,:]) / jnp.sqrt(r2)
    # project φ̂ to tangent plane and orthonormalize with N:
    phi_tan = phi_hat - jnp.sum(phi_hat * Nw, axis=1, keepdims=True) * Nw
    phi_tan = phi_tan / jnp.maximum(1e-30, jnp.linalg.norm(phi_tan, axis=1, keepdims=True))
    theta_hat = jnp.cross(Nw, phi_tan)
    theta_hat = theta_hat / jnp.maximum(1e-30, jnp.linalg.norm(theta_hat, axis=1, keepdims=True))

    # reference direction in tangent: project e1 to tangent
    e1_tan = e1[None,:] - jnp.sum(e1[None,:]*Nw, axis=1, keepdims=True)*Nw
    e1_tan = e1_tan / jnp.maximum(1e-30, jnp.linalg.norm(e1_tan, axis=1, keepdims=True))

    # θ is angle from e1_tan in the (phi_tan, theta_hat) basis
    num = jnp.sum(e1_tan * phi_tan, axis=1)
    den = jnp.sum(e1_tan * theta_hat, axis=1)
    theta = jnp.arctan2(num, den)

    # axial coordinate along a:
    s = jnp.sum(Xc * a[None,:], axis=1)
    return phi_a, theta, s

def _orthonormal_complement(a_hat):
    """
    Return two orthonormal vectors spanning the plane ⟂ a_hat.
    Deterministic construction.
    """
    a = np.asarray(a_hat) / (np.linalg.norm(a_hat) + 1e-30)
    # pick any vector not parallel to a
    t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = t - np.dot(t, a)*a
    e1 /= (np.linalg.norm(e1) + 1e-30)
    e2 = np.cross(a, e1)
    e2 /= (np.linalg.norm(e2) + 1e-30)
    return e1, e2

def plot_geometry_and_solution(P, N, grads_where, title_suffix="", show_normals=True, kind="torus", a_hat=None):
    """
    Left: surface + normals + scatter colored by |∇φ|
    Right: imshow of |∇φ| projected to best-fit (u,v) plane.
    """
    grad_mag = np.linalg.norm(np.asarray(grads_where), axis=1)

    fig = plt.figure(figsize=(14, 6))
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    axR = fig.add_subplot(1, 2, 2)

    # --- Left: 3D with normals and colored points ---
    center = np.mean(np.asarray(P), axis=0)
    median_radius = np.median(np.linalg.norm(np.asarray(P) - center, axis=1))
    q_len = float(0.1 * median_radius)

    # Points colored by |∇φ|
    vmin = pct(grad_mag, 1); vmax = pct(grad_mag, 90)
    sc = axL.scatter(P[:,0], P[:,1], P[:,2],
                     c=grad_mag, s=6, cmap="viridis", vmin=vmin, vmax=vmax)
    # Normals
    if show_normals and N is not None:
        axL.quiver(P[:,0], P[:,1], P[:,2], N[:,0], N[:,1], N[:,2],
                   length=q_len, linewidth=0.5, normalize=True, color="k", alpha=0.7)
    axL.set_title(r"Surface & normals, colored by $|\nabla\phi|$" + title_suffix)
    axL.set_xlabel("x"); axL.set_ylabel("y"); axL.set_zlabel("z")
    axL.view_init(elev=20, azim=35); fix_matplotlib_3d(axL)
    cbar = fig.colorbar(sc, ax=axL, shrink=0.7, label=r"$|\nabla \phi|$ on $\Gamma_-$")

    # --- Right panel: coordinates for imshow ---
    # For torus: x = TRUE cylindrical φ := atan2(y,x); y = poloidal θ (tokamak-style)
    # For mirror: keep (s, θ) as before.
    if kind == "mirror":
        # keep axis-aware (s, θ)
        phi_a, theta, s = angles_for_axis(
            jnp.asarray(P), jnp.asarray(N), jnp.asarray(a_hat) if a_hat is not None else jnp.array([0.0,0.0,1.0]),
            center=jnp.asarray(np.mean(np.asarray(P), axis=0))
        )
        x_axis_vals = np.asarray(s) - np.median(np.asarray(s))
        x_label = "axial coordinate s"
        y_axis_vals = np.asarray(theta)
        y_label = "poloidal angle θ (rad)"
    else:
        # TORUS: true cylindrical φ on x-axis
        Pnp = np.asarray(P)
        x_axis_vals = np.arctan2(Pnp[:,1], Pnp[:,0])  # TRUE φ
        x_axis_vals = np.unwrap(x_axis_vals)
        x_label = "toroidal angle ϕ (rad)"
        # Build θ the same robust way you already do (tokamak-style)
        phi, theta = _angles_phi_theta(P, N)
        y_axis_vals = np.asarray(theta)
        y_label = "poloidal angle θ (rad)"

    # Rasterize |∇φ| to a grid in (x_axis, θ)
    nX = 360 if kind != "mirror" else 240
    nY = 180
    Xu = np.linspace(-np.pi, np.pi, nX) if kind != "mirror" else np.linspace(x_axis_vals.min(), x_axis_vals.max(), nX)
    Yu = np.linspace(-np.pi, np.pi, nY)
    XX, YY = np.meshgrid(Xu, Yu, indexing="xy")

    pts = np.column_stack([x_axis_vals, y_axis_vals])
    grid = griddata(points=pts, values=grad_mag, xi=(XX, YY), method='linear')

    im = axR.imshow(grid, origin='lower',
                    extent=(Xu[0], Xu[-1], Yu[0], Yu[-1]),
                    aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axR.set_xlabel(x_label)
    axR.set_ylabel(y_label)
    axR.set_title(r"$|\nabla \phi|$ in axis-aware coords")
    fig.colorbar(im, ax=axR, shrink=0.8, label=r"$|\nabla \phi|$")

    plt.tight_layout(); plt.show()

def plot_boundary_condition_errors(P, N, grad_on_ring):
    n_dot_grad = jnp.sum(N * grad_on_ring, axis=1)
    grad_norm  = jnp.linalg.norm(grad_on_ring, axis=1)
    rn = jnp.abs(n_dot_grad) / jnp.maximum(1e-30, grad_norm)  # dimensionless
    ang_deg = jnp.degrees(jnp.arccos(jnp.clip(1.0 - 2.0*rn**2, -1.0, 1.0)))  # optional alt metric

    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(1,2,1, projection='3d')
    vmin = pct(rn, 1); vmax = pct(rn, 99)
    s1 = ax.scatter(P[:,0], P[:,1], P[:,2], c=np.array(rn), s=6, cmap='magma', vmin=vmin, vmax=vmax)
    fig.colorbar(s1, ax=ax, shrink=0.7, label=r"$|n\cdot\nabla\phi|/|\nabla\phi|$ on $\Gamma_-$")
    ax.set_title("Normalized Neumann residual on Γ₋")
    fix_matplotlib_3d(ax)

    ax2 = fig.add_subplot(1,2,2)
    idx = np.arange(P.shape[0])
    ax2.plot(idx, np.asarray(rn), lw=0.8, label=r"$|n\cdot\nabla\phi|/|\nabla\phi|$")
    ax2.set_xlabel("point index"); ax2.set_ylabel("dimensionless"); ax2.legend()
    ax2.set_title("BC residuals (line scan) on Γ₋")
    plt.tight_layout(); plt.show()


def plot_laplacian_errors_on_interior_band(P, lap_psi_at_interior, eps):
    P = np.asarray(P)
    lap = np.asarray(lap_psi_at_interior)
    assert P.shape[0] == lap.shape[0], \
        f"plot_laplacian_errors_on_interior_band: |P|={P.shape[0]} vs |lap|={lap.shape[0]}."
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,2,1, projection='3d')
    vmin = pct(jnp.abs(lap_psi_at_interior), 1); vmax = pct(jnp.abs(lap_psi_at_interior), 99)
    s = ax.scatter(P[:,0], P[:,1], P[:,2],
                   c=np.array(jnp.abs(lap_psi_at_interior)), s=6, cmap='inferno',
                   vmin=vmin, vmax=vmax)
    fig.colorbar(s, ax=ax, shrink=0.7, label=rf"$|\nabla^2 \psi|(x - \varepsilon n),\ \varepsilon={eps:g}$")
    ax.set_title(r"Near-boundary Laplacian of $\psi$ (should be ≈0 inside)"); fix_matplotlib_3d(ax)
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist(np.asarray(jnp.abs(lap_psi_at_interior)), bins=60, alpha=0.9)
    ax2.set_title(r"Histogram of $|\nabla^2 \psi|$ at interior offsets")
    ax2.set_xlabel("|∇²ψ|"); ax2.set_ylabel("count")
    plt.tight_layout(); plt.show()

def build_cap_flux_constraint(P, N, Pn, Nn, scinfo, Yn,
                              grad_t, grad_p, a, a_hat,
                              q=0.02, ds_frac=0.02, k_cap=32, side="low",
                              h_grads=None):
    """
    Build (c_cap, d_cap) for a virtual end-cap perpendicular to a_hat.
    - side: "low" or "high": chooses s-quantile q or 1-q
    - ds_frac: half-thickness in axial coordinate as a fraction of axial span
    """
    X = jnp.asarray(P)
    a = jnp.asarray(a_hat) / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    c = jnp.asarray(scinfo.center)

    # axial coordinate s and span
    s = jnp.sum((X - c[None,:]) * a[None,:], axis=1)
    s_min, s_max = float(jnp.min(s)), float(jnp.max(s))
    s_span = max(1e-9, s_max - s_min)

    s0 = np.quantile(np.asarray(s), q if side == "low" else 1.0 - q)
    ds = ds_frac * s_span
    mask = np.abs(np.asarray(s) - s0) <= ds
    if not np.any(mask):
        raise RuntimeError(f"No points found for {side} cap; try increasing ds_frac.")

    # cap points and normals (constant ±a_hat)
    P_cap = X[mask, :]
    N_cap = ( -a if side == "low" else a )[None, :].repeat(P_cap.shape[0], axis=0)

    # weights on cap in plane ⟂ a_hat via kNN
    # build plane basis (e1,e2) ⟂ a_hat
    e1_np, e2_np = _orthonormal_complement(np.array(a_hat))
    e1 = jnp.asarray(e1_np); e2 = jnp.asarray(e2_np)
    Xc = P_cap - c[None, :]
    u1 = np.asarray(jnp.sum(Xc * e1[None,:], axis=1))
    u2 = np.asarray(jnp.sum(Xc * e2[None,:], axis=1))
    UV = np.column_stack([u1, u2])

    k_eff = min(k_cap+1, len(UV))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree").fit(UV)
    dists, _ = nbrs.kneighbors(UV)
    rk_cap = dists[:, -1]
    W_cap = jnp.asarray(np.pi * rk_cap**2, dtype=jnp.float64)

    # collocation on the cap (use normalized coords for grad_t/grad_p)
    Xn_cap = (P_cap - scinfo.center) * scinfo.scale

    # ⟵ IMPORTANT: include harmonic columns so A_cap has the same width as A
    A_cap = build_A_rows_at_points(Xn_cap, N_cap, Yn, scinfo, h_grads=h_grads)

    # MV contribution on the cap
    Gt_cap = grad_t(Xn_cap)
    Gp_cap = grad_p(Xn_cap)
    g_cap  = scinfo.scale * jnp.sum(N_cap * (a[0]*Gt_cap + a[1]*Gp_cap), axis=1)

    # flux constraint vector for the cap
    c_cap, d_cap = make_flux_constraint(A_cap, W_cap, g_cap)
    return c_cap, d_cap


# ------------------------------- main ------------------------------- #
def main(xyz_csv="slam_surface.csv", nrm_csv="slam_surface_normals.csv",
         use_mv=True, k_nn=48,
         interior_eps_factor=5e-3,  # ε ~ interior offset for evaluation, in *normalized* h units
         verbose=True, show_normals=False,
         toroidal_flux=None,          # prescribe Φ_t (sets a_t = Φ_t/(2π)) if not None
         sf_min=1.0, sf_max=6.5, lbfgs_maxiter=30, lbfgs_tol=1e-8, seed=None, lbfgs_maxiter_final=30,
         reg_gamma_a=1e-2
        ):

    # Load & orient
    P, N = load_surface_xyz_normals(xyz_csv, nrm_csv, verbose=verbose)
    print(f"[INFO] Npoints={P.shape[0]}")
    N, flipped = maybe_flip_normals(P, N)

    # Normalize geometry (for numerics)
    Pn, scinfo = normalize_geometry(P, verbose=verbose)
    Nn = N  # direction unchanged by uniform scaling/translation for normals

    # kNN scales & crude surface weights
    W, rk = kNN_geometry_stats(Pn, k=k_nn, verbose=verbose)
    h_med = float(np.median(rk))
    print(f"[SCALE] median local spacing h_med≈{h_med:.4g} (normalized units)")
    
    # --- Auto-tune ---
    best = autotune(P, N, Pn, Nn, W, rk, scinfo, interior_eps_factor=interior_eps_factor,
                          use_mv=use_mv,
                          verbose=True, sf_min=sf_min, sf_max=sf_max,
                          lbfgs_maxiter=lbfgs_maxiter, lbfgs_tol=lbfgs_tol)
    eps_n = best["eps_n_star"]
    source_factor_opt = best["source_factor"]
    lambda_reg_opt    = best["lambda_reg"]
    # interior_eps_factor    = best["interior_eps_factor"]
    # Reuse pre-built things to avoid recompute:
    phi_t, grad_t, phi_p, grad_p, Yn, kind = best["phi_t"], best["grad_t"], best["phi_p"], best["grad_p"], best["Yn"], best["kind"]
    delta_n = float(np.median(np.linalg.norm(np.asarray(Yn) - np.asarray(Pn), axis=1)))

    # include harmonic pack everywhere
    h_vals, h_grads = best["h_vals"], best["h_grads"]

    # --- BOUNDARY: use MFS(+H) only; MV goes to RHS g_raw ---
    A_bdry, _ = build_system_matrices(
        Pn, Nn, Yn, W, grad_t, grad_p, scinfo,
        use_mv=False, center_D=False, verbose=verbose, h_grads=h_grads
    )

    # --- Choose a nontrivial 'a' ---
    a = best["a"]
    if (jnp.linalg.norm(a) < 1e-12):
        # If the SVD-based fit came out ~zero, pick a deterministic non-zero default
        a = default_axis_aware_a(best["a_hat"], prefer_toroidal=(kind=="torus"))

    # Build g_raw on the boundary for this 'a'
    grad_t_bdry = grad_t(Pn)
    grad_p_bdry = grad_p(Pn)
    g_raw = scinfo.scale * jnp.sum(Nn * (a[0]*grad_t_bdry + a[1]*grad_p_bdry), axis=1)

    # --- Interior ring Γ₋ (at tuned eps): MFS(+H) rows only ---
    eps_w = float(eps_n) / float(np.asarray(scinfo.scale))
    P_in  = P - eps_w * N
    Xn_in = (P_in - scinfo.center) * scinfo.scale
    A_in  = build_A_rows_at_points(Xn_in, N, Yn, scinfo, h_grads=h_grads)

    # MV RHS on the ring
    Gt_in = grad_t(Xn_in)
    Gp_in = grad_p(Xn_in)
    g_in  = scinfo.scale * jnp.sum(N * (a[0]*Gt_in + a[1]*Gp_in), axis=1)

    # --- Flux constraints on Γ and Γ₋ with MV RHS included ---
    W_ring = build_ring_weights(P_in, Pn, k=k_nn)
    c_bdry, d_bdry = make_flux_constraint(A_bdry, W, g_raw)
    c_int,  d_int  = make_flux_constraint(A_in,   W_ring, g_in)

    # --- Optional end-caps: make caps augmented as well (H + D) ---
    cap_constraints = []
    try:
        # Use previous cap selection logic but build augmented rows there:
        def cap_aug(side):
            # select cap points/normals + W_cap exactly as build_cap_flux_constraint
            a_hat = best["a_hat"]
            X = jnp.asarray(P)
            a = jnp.asarray(a_hat) / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
            c0 = jnp.asarray(scinfo.center)
            s = jnp.sum((X - c0[None,:]) * a[None,:], axis=1)
            s_min, s_max = float(jnp.min(s)), float(jnp.max(s))
            s_span = max(1e-9, s_max - s_min)
            s0 = np.quantile(np.asarray(s), 0.02 if side=="low" else 0.98)
            ds = 0.02 * s_span
            mask = np.abs(np.asarray(s) - s0) <= ds
            P_cap = X[mask, :]
            N_cap = ( -a if side=="low" else a )[None,:].repeat(P_cap.shape[0], axis=0)

            # weights in plane ⟂ a_hat via kNN (same as before)
            e1_np, e2_np = _orthonormal_complement(np.array(a_hat))
            e1 = jnp.asarray(e1_np); e2 = jnp.asarray(e2_np)
            Xc = P_cap - c0[None, :]
            u1 = np.asarray(jnp.sum(Xc * e1[None,:], axis=1))
            u2 = np.asarray(jnp.sum(Xc * e2[None,:], axis=1))
            UV = np.column_stack([u1, u2])
            k_eff = min(k_nn+1, len(UV))
            nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree").fit(UV)
            dists, _ = nbrs.kneighbors(UV)
            rk_cap = dists[:, -1]
            W_cap = jnp.asarray(np.pi * rk_cap**2, dtype=jnp.float64)

            Xn_cap = (P_cap - scinfo.center) * scinfo.scale
            A_cap = build_A_rows_at_points(Xn_cap, N_cap, Yn, scinfo, h_grads=h_grads)
            Gt_cap = grad_t(Xn_cap)
            Gp_cap = grad_p(Xn_cap)
            g_cap  = scinfo.scale * jnp.sum(N_cap * (a[0]*Gt_cap + a[1]*Gp_cap), axis=1)
            c_cap, d_cap = make_flux_constraint(A_cap, W_cap, g_cap)
            return (c_cap, d_cap)

        c_cap_low,  d_cap_low  = cap_aug("low")
        c_cap_high, d_cap_high = cap_aug("high")
        cap_constraints = [(c_cap_low, d_cap_low), (c_cap_high, d_cap_high)]
    except Exception as e:
        print("[WARN] Cap constraints (augmented) failed; continuing without them:", e)
        cap_constraints = []

    # --- Build full constraint set (Γ, Γ₋, caps) ---
    constraints = [(c_bdry, d_bdry), (c_int, d_int)] + cap_constraints

    # --- SOLVE α ONLY with hard flux constraints and non-zero RHS ---
    alpha = solve_alpha_with_rhs_hard_flux_multi(
        A_bdry, W, g_raw, lam=lambda_reg_opt, constraints=constraints, verbose=verbose
    )
    print(f"[SOL-α] ||alpha||₂={float(jnp.linalg.norm(alpha)):.3e}, a_t={float(a[0]):.6g}, a_p={float(a[1]):.6g}")    
    print(f"[SOL] a_t={float(a[0]):.6g}, a_p={float(a[1]):.6g}, ||alpha||₂={float(jnp.linalg.norm(alpha)):.3e}")

    phi_fn, grad_fn, psi_fn, grad_psi_fn, lap_psi_fn, _ = build_evaluators_mfs_plusH(
        Pn, Yn, alpha, phi_t, phi_p, a, scinfo, grad_t, grad_p,
        h_vals=h_vals, h_grads=h_grads
    )
    
    if use_mv and (toroidal_flux is None):
        # If the estimated periods are ~zero (degenerate selection), keep the default 'a'
        Pi_t, Pi_p = estimate_periods_on_cycles(P, N, lambda X: grad_fn(X), best["a_hat"], scinfo)
        if (abs(Pi_t) > 1e-8) or (abs(Pi_p) > 1e-8):
            a_locked = solve_ap_at_from_periods(Pi_t, Pi_p)
            a = a_locked
            grad_t_bdry = grad_t(Pn); grad_p_bdry = grad_p(Pn)
            g_raw = scinfo.scale * jnp.sum(Nn * (a[0]*grad_t_bdry + a[1]*grad_p_bdry), axis=1)
            prefac = _prefactor_constrained_system(A_bdry, W, lambda_reg_opt, constraints)
            alpha  = fast_constrained_resolve_with_new_g(A_bdry, W, g_raw, lambda_reg_opt, prefac)
            # rebuild evaluators
            phi_fn, grad_fn, psi_fn, grad_psi_fn, lap_psi_fn, _ = build_evaluators_mfs_plusH(
                Pn, Yn, alpha, phi_t, phi_p, a, scinfo, grad_t, grad_p,
                h_vals=h_vals, h_grads=h_grads
            )
            print(f"[MV-LOCK] Period lock applied. a_t={float(a[0]):.6e}, a_p={float(a[1]):.6e}")
        else:
            print("[MV-LOCK] Periods ~0; kept default nontrivial 'a'.")


    # === Diagnostics on Γ (illustrative; gradients may still be large due to proximity to sources) ===
    grad_on_Gamma = grad_fn(P)
    vec_stats("[EVAL Γ] |∇φ|", jnp.linalg.norm(grad_on_Gamma, axis=1))

    # === Diagnostics on Γ₋ (interior offset): use tuned eps ===
    eps_w = float(eps_n) / float(np.asarray(scinfo.scale))
    P_in  = P - eps_w * N
    print(f"[DIAG] Using interior-offset ring Γ₋ with eps_n={eps_n:.3g} (normalized), eps_world={eps_w:.3g}.")

    grad_on_ring = grad_fn(P_in)
    grad_mag_ring = jnp.linalg.norm(grad_on_ring, axis=1)
    vec_stats("[EVAL Γ₋] |∇φ|", grad_mag_ring)
    
    n_dot_grad_ring = jnp.sum(N * grad_on_ring, axis=1)
    rn = jnp.abs(n_dot_grad_ring) / jnp.maximum(1e-30, grad_mag_ring)
    vec_stats("[EVAL Γ₋] normalized BC |n·∇φ|/|∇φ|", rn)

    # Flux neutrality on Γ₋ (should be ~0)
    # reuse the ring weights from the constraint stage
    flux = float(jnp.dot(W_ring, n_dot_grad_ring))
    area_ring = float(jnp.sum(W_ring))
    print(f"[CHK] Flux neutrality on Γ₋: ∫ n·∇φ dS ≈ {flux:.6e}  (avg={flux/area_ring:.3e})")
    d = jnp.linalg.norm(grad_fn(P) - grad_fn(P - (eps_n / scinfo.scale) * N)) / (jnp.linalg.norm(grad_fn(P)) + 1e-30)
    print("[CHK] rel change of grad between Γ and Γ_-:", float(d))
    
    # === Adaptive up-sampling of boundary where BC error is high ===
    rn = jnp.abs(jnp.sum(N * grad_on_ring, axis=1)) / jnp.maximum(1e-30, jnp.linalg.norm(grad_on_ring, axis=1))
    thresh = float(np.percentile(np.asarray(rn), 90))
    mask_refine = np.asarray(rn) > thresh
    print(f"[ADAPT] refining {mask_refine.sum()} / {len(mask_refine)} boundary points with high BC error (>p90).")
    if mask_refine.sum() > 0:
        # simple duplication + small jitter in tangent plane (keeps normals)
        Pref = np.asarray(P)[mask_refine]
        Nref = np.asarray(N)[mask_refine]
        # jitter in tangent plane to avoid duplicates
        rng = np.random.default_rng(seed)
        jitter = rng.normal(scale=0.01*np.median(np.linalg.norm(P,axis=1)), size=Pref.shape)
        jitter -= (np.sum(jitter*Nref,axis=1,keepdims=True))*Nref  # project tangent
        Pref2 = Pref + 0.2*jitter
        Nref2 = Nref  # reuse normals
        P_aug = jnp.asarray(np.vstack([P, Pref2]))
        N_aug = jnp.asarray(np.vstack([N, Nref2]))
        # rebuild everything quick:
        Pn_aug, scinfo_aug = normalize_geometry(P_aug, verbose=False)
        W_aug, rk_aug = kNN_geometry_stats(Pn_aug, k=k_nn, verbose=False)
        best_aug = autotune_outer(P_aug, N_aug, Pn_aug, N_aug, W_aug, rk_aug, scinfo_aug,
                                  use_mv=use_mv, verbose=False,
                                  sf_min=sf_min, sf_max=sf_max,
                                  lbfgs_maxiter=lbfgs_maxiter_final, lbfgs_tol=lbfgs_tol)
        Yn_aug, _ = build_mfs_sources(Pn_aug, N_aug, rk_aug, scinfo_aug, source_factor=best_aug["source_factor"], verbose=False)
        A_aug, _ = build_system_matrices(
            Pn_aug, N_aug, Yn_aug, W_aug,
            best_aug["grad_t"], best_aug["grad_p"], scinfo_aug,
            use_mv=True, center_D=True, verbose=False,
            h_grads=best_aug["h_grads"]
        )
        gt_b = best_aug["grad_t"](Pn_aug); gp_b = best_aug["grad_p"](Pn_aug)
        a_aug = best_aug["a"]
        g_aug = scinfo_aug.scale * jnp.sum(N_aug * (a_aug[0]*gt_b + a_aug[1]*gp_b), axis=1)
        alpha_aug = solve_alpha_with_rhs(A_aug, W_aug, g_aug, lam=best_aug["lambda_reg"], verbose=True)
        print("[ADAPT] Re-solve done on augmented boundary.")
        # overwrite current solution objects if you want to proceed with augmented solve
        P, N, Pn, scinfo, W, rk, Yn = P_aug, N_aug, Pn_aug, scinfo_aug, W_aug, rk_aug, Yn_aug
        phi_fn, grad_fn, psi_fn, grad_psi_fn, lap_psi_fn, _ = build_evaluators_mfs_plusH(
            Pn, Yn, alpha_aug,
            best_aug["phi_t"], best_aug["phi_p"], a_aug, scinfo,
            best_aug["grad_t"], best_aug["grad_p"],
            h_vals=best_aug["h_vals"], h_grads=best_aug["h_grads"]
        )
        alpha = alpha_aug
        # overwrite current solution objects
        P, N, Pn, scinfo, W, rk, Yn = P_aug, N_aug, Pn_aug, scinfo_aug, W_aug, rk_aug, Yn_aug

        # keep h_med consistent with the new boundary
        h_med = float(np.median(np.asarray(rk)))

        # use *optimized* eps from the new autotune
        eps_n = float(best_aug["eps_n_star"])
        eps_w = eps_n / float(np.asarray(scinfo.scale))
        P_in  = P - eps_w * N

        # recompute diagnostics on the new ring
        grad_on_ring = grad_fn(P_in)
        lap_in       = lap_psi_fn(P_in)

    def grad_cyl_about_axis(P, a_hat):
        # ∇ϕ_a = (a × r_perp)/|r_perp|^2 with r_perp = (X - c) - ((X - c)·a)a
        a = a_hat / np.linalg.norm(a_hat)
        X = np.asarray(P)
        c = np.asarray(scinfo.center)          # <<<<<< use the same center
        Xc = X - c
        r_par  = (Xc @ a)[:,None] * a[None,:]
        r_perp = Xc - r_par
        r2     = np.sum(r_perp*r_perp, axis=1, keepdims=True)
        return np.cross(a[None,:], r_perp) / np.maximum(1e-30, r2)

    Gt = np.asarray(grad_t(Pn))          # normalized-space grad_t
    Gc = grad_cyl_about_axis(P, np.array(best["a_hat"]))  # world-space, but only direction matters
    # compare directions only
    c = np.sum(Gt*Gc, axis=1)/(np.linalg.norm(Gt,axis=1)*np.linalg.norm(Gc,axis=1)+1e-30)
    print("median cos(angle(grad_t, axis-aware cylindrical)) ≈", np.median(c))

    # Laplacian(ψ) near boundary (independent check)
    eps_w = float(eps_n) / float(np.asarray(scinfo.scale))
    P_in  = P - eps_w * N

    # (RE)build ring weights to match the CURRENT P_in
    W_ring = build_ring_weights(P_in, Pn, k=k_nn)

    # now safe to run diagnostics
    grad_on_ring = grad_fn(P_in)
    lap_in       = lap_psi_fn(P_in)

    vec_stats("[LAP Γ₋] |∇²ψ|", jnp.abs(lap_in))

    print_final_diagnostics(
        tag="post-solve",
        P=P, N=N, W=W, W_ring=W_ring, scinfo=scinfo,
        eps_n=eps_n, lam=lambda_reg_opt, source_factor=source_factor_opt,
        a=a, alpha=alpha, grad_fn=grad_fn, lap_psi_fn=lap_psi_fn,
        A_aug_bdry=A_bdry, constraints=constraints, recompute_W_ring=False,
        Pn=Pn, Yn=Yn, h_grads=h_grads, grad_t_fn=grad_t, grad_p_fn=grad_p
    )


    plot_geometry_and_solution(P, N, grad_on_ring, title_suffix="",
                           show_normals=show_normals, kind=kind, a_hat=best.get("a_hat", None))
    plot_boundary_condition_errors(P, N, grad_on_ring)
    plot_laplacian_errors_on_interior_band(P, lap_in, eps_w)

    return dict(
        P=P, N=N, Pn=Pn, W=W, rk=rk, h_med=h_med,
        alpha=alpha, a=a, delta_n=delta_n, eps_n=eps_n,
        phi_fn=phi_fn, grad_fn=grad_fn, psi_fn=psi_fn,
        grad_psi_fn=grad_psi_fn, laplacian_psi_fn=lap_psi_fn,
        scinfo=scinfo, Yn=Yn, a_hat=best["a_hat"], kind=kind
    )

if __name__ == "__main__":
    # default_xyz = "wout_precise_QA.csv"
    # default_normals = "wout_precise_QA_normals.csv"
    default_xyz = "slam_surface.csv"
    default_normals = "slam_surface_normals.csv"
    # default_xyz = "sflm_rm4.csv"
    # default_normals = "sflm_rm4_normals.csv"
    # default_xyz = "wout_precise_QH.csv"
    # default_normals = "wout_precise_QH_normals.csv"
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", nargs="?", default=default_xyz,
                    help="CSV file with x,y,z columns (positional or --xyz)")
    ap.add_argument("normals", nargs="?", default=default_normals,
                    help="CSV file with nx,ny,nz columns (positional or --nrm)")
    ap.add_argument("--xyz", dest="xyz_flag", default=default_xyz,
                    help="CSV file with x,y,z columns (alternative to positional)")
    ap.add_argument("--normals", dest="nrm_flag", default=default_normals,
                    help="CSV file with nx,ny,nz columns (alternative to positional)")
    ap.add_argument("--sf_min", type=float, default=1.0, help="Min source factor for autotuning")
    ap.add_argument("--sf_max", type=float, default=6.5, help="Max source factor for autotuning")
    ap.add_argument("--lbfgs-maxiter", type=int, default=5, help="Max iterations for L-BFGS")
    ap.add_argument("--lbfgs-tol", type=float, default=1e-8, help="Tolerance for L-BFGS")
    ap.add_argument("--k-nn", type=int, default=64) # this sets the k for kNN weights & scales (computational cost)
    ap.add_argument("--no-mv",  dest="use_mv", action="store_false", help="Disable multivalued harmonic field")
    ap.add_argument("--interior-eps-factor", type=float, default=5e-3)
    ap.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    ap.add_argument("--show-normals", action="store_true", help="Show surface normals in plots")
    ap.add_argument("--toroidal-flux", type=float, default=None,
                    help="If set, prescribes the toroidal flux Φ_t (sets a_t = Φ_t/(2π))")
    ap.add_argument("--mfs-out", default=None,
                    help="Write portable MFS solution to this .npz (center,scale,Yn,alpha,a,a_hat,P,N,kind)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for refinement jitter")
    ap.add_argument("--lbfgs-maxiter-final", type=int, default=10,
                    help="Max iterations for L-BFGS on final solve after refinement")
    ap.add_argument("--reg-gamma-a", type=float, default=1e-2,
                    help="Relative reg on a: uses (λ*reg_gamma_a)^2 on the last 2 columns")
    args = ap.parse_args()

    if args.mfs_out == None:
        args.mfs_out = args.xyz.replace(".csv", "_solution.npz")

    out = main(
        xyz_csv=args.xyz, nrm_csv=args.normals,
        use_mv=args.use_mv, k_nn=args.k_nn,
        interior_eps_factor=args.interior_eps_factor,
        verbose=args.verbose,
        show_normals=args.show_normals,
        toroidal_flux=args.toroidal_flux,
        sf_min=args.sf_min, sf_max=args.sf_max,
        lbfgs_maxiter=args.lbfgs_maxiter, lbfgs_tol=args.lbfgs_tol,
        seed=args.seed, lbfgs_maxiter_final=args.lbfgs_maxiter_final,
        reg_gamma_a=args.reg_gamma_a
    )

    # --- SAVE a portable checkpoint for the tracer ---
    # Pull everything we need out of 'out' and the local scope
    try:
        # Objects available at end of main():
        #   P, N, Pn, W, rk, h_med, alpha, a, delta_n, eps_n,
        #   phi_fn, grad_fn, psi_fn, grad_psi_fn, laplacian_psi_fn
        # Local names still in scope here: scinfo, Yn, best/kind, best["a_hat"]
        np.savez(
            args.mfs_out,
            center=np.asarray(out["scinfo"].center, dtype=float),
            scale=float(np.asarray(out["scinfo"].scale)),
            Yn=np.asarray(out["Yn"], dtype=float),
            alpha=np.asarray(out["alpha"], dtype=float),
            a=np.asarray(out["a"], dtype=float),
            a_hat=np.asarray(out["a_hat"], dtype=float),
            P=np.asarray(out["P"], dtype=float),
            N=np.asarray(out["N"], dtype=float),
            kind=str(out["kind"]),
        )
        print(f"[SAVE] Wrote MFS checkpoint → {args.mfs_out}")
    except Exception as e:
        print("[WARN] Could not save MFS checkpoint:", e)