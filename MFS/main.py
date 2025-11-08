#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    
# ----------------------- 2D quasi-Newton (BFGS) ----------------------- #
def bfgs_2d(value_and_grad_fn, p0, *,
            sf_min=1.0, sf_max=4.0,
            max_iter=25, tol=1e-6):
    """
    Tiny 2D BFGS in JAX over p=[log_sf, log_lambda] with early stopping.
    Early stop when ||g|| < tol or iterations reach max_iter.
    """
    def project(p):
        # p = [log_sf, log_lam]
        log_sf, log_lam = p
        sf = jnp.exp(log_sf)
        sf_clamped = jnp.clip(sf, sf_min, sf_max)
        log_lam_clamped = jnp.clip(log_lam, -9.0, -3.0)  # λ in [e^-9, e^-3] ≈ [1e-4, 5e-2]
        return jnp.array([jnp.log(sf_clamped), log_lam_clamped], dtype=p.dtype)

    def one_step(state):
        # state = (p, H, f_prev, it, done)
        p, H, f_prev, it, done = state

        # If already done, just return the same state (no-ops) to keep shapes static
        def _advance(_):
            p_proj = project(p)
            f, g = value_and_grad_fn(p_proj)

            jax_debug.print("[BFGS] it={it}  f={f:.3e}  ||g||={gn:.2e}  sf={sf:.3f}  lam={lam:.3e}",
                            it=it, f=f, gn=jnp.linalg.norm(g),
                            sf=jnp.exp(p_proj[0]), lam=jnp.exp(p_proj[1]))

            # descent direction
            d = - H @ g

            # Backtracking line search (Armijo)
            def bt_body(carry, _):
                step, f_curr = carry
                p_try = project(p_proj + step * d)
                f_try, _ = value_and_grad_fn(p_try)
                ok = f_try <= f + 1e-4 * step * (g @ d)
                step = jnp.where(ok, step, 0.5 * step)
                f_curr = jnp.where(ok, f_try, f_curr)
                return (step, f_curr), ok

            (step_final, f_new), _ = jax.lax.scan(
                bt_body,
                (jnp.array(1.0, dtype=p.dtype), f),
                jnp.arange(4)   # up to 4 backtracks
            )
            p_new = project(p_proj + step_final * d)

            # BFGS update with mild Powell damping to keep H SPD
            _, g_new = value_and_grad_fn(p_new)
            s = p_new - p_proj
            y = g_new - g
            ys = y @ s
            # Powell damping: ensure y^T s is not too small
            theta = jnp.where(ys < 0.2 * (s @ (H @ s)), 
                            (0.8 * (s @ (H @ s))) / (s @ (H @ s) - ys + 1e-30),
                            1.0)
            y_tilde = theta * y + (1 - theta) * (H @ s)
            rho = 1.0 / (y_tilde @ s + 1e-30)
            I = jnp.eye(2, dtype=p.dtype)
            V = I - rho * jnp.outer(s, y_tilde)
            H_new = V @ H @ V.T + rho * jnp.outer(s, s)

            # early stop: gradient small OR relative f change small
            rel = jnp.abs((f_new - f) / (jnp.abs(f) + 1e-30))
            done_new = jnp.logical_or(jnp.linalg.norm(g_new) < tol, rel < 1e-3)
            return (p_new, H_new, f_new, it + 1, done_new)

        def _passthrough(_):
            return (p, H, f_prev, it + 1, done)

        # If done, do passthrough; else, advance one BFGS step
        return jax.lax.cond(done, _passthrough, _advance, operand=None)

    # Initial state
    H0 = jnp.eye(2, dtype=p0.dtype)
    f0 = jnp.inf
    state0 = (p0, H0, f0, jnp.array(0, dtype=jnp.int32), jnp.array(False))

    def cond_fun(state):
        _, _, f_prev, it, done = state
        # stop if done or max_iter; also stop if relative change in f is tiny
        return jnp.logical_and(it < max_iter, jnp.logical_not(done))

    p_star, H_star, f_star, _, _ = jax.lax.while_loop(cond_fun, one_step, state0)
    f_star, g_star = value_and_grad_fn(project(p_star))
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

def solve_once(P, N, Pn, Nn, W, rk, scinfo,
               use_mv, k_nn, source_factor, lambda_reg, mv_weight,
               interior_eps_factor, verbose=True):
    # Build sources and system
    Yn, delta_n = build_mfs_sources(Pn, Nn, rk, scinfo, source_factor=source_factor, verbose=verbose)
    # sanity: ensure δ points outward (dot > 0 for most points)
    dot = jnp.sum((Yn - Pn) * Nn, axis=1)
    print(f"[CHK] (Yn - Pn)·Nn: min={float(dot.min()):.3e}, median={float(jnp.median(dot)):.3e}")
    kind, a_hat, E_axes, c_axes, svals = detect_geometry_and_axis(Pn, verbose=True)
    phi_t, grad_t, phi_p, grad_p = multivalued_bases_about_axis(Pn, Nn, a_hat, verbose=True) if use_mv else (
        (lambda Xn: jnp.zeros((Xn.shape[0],))), (lambda Xn: jnp.zeros_like(Xn)),
        (lambda Xn: jnp.zeros((Xn.shape[0],))), (lambda Xn: jnp.zeros_like(Xn))
    )
    A, D = build_system_matrices(Pn, Nn, Yn, W, grad_t, grad_p, scinfo,
                                 use_mv=use_mv, center_D=True, verbose=verbose)

    # Fit multivalued coefficients a and rhs
    if use_mv:
        grad_t_bdry, grad_p_bdry = grad_t(Pn), grad_p(Pn)
        a, D_raw, D0 = fit_mv_coeffs_minimize_rhs(Nn, W, grad_t_bdry, grad_p_bdry, verbose=verbose)
        g_raw = jnp.sum(Nn * (a[0]*grad_t_bdry + a[1]*grad_p_bdry), axis=1)
    else:
        a = jnp.zeros((2,), dtype=jnp.float64)
        g_raw = jnp.zeros((Pn.shape[0],), dtype=jnp.float64)

    # Solve for alpha
    alpha = solve_alpha_with_rhs(A, W, g_raw, lam=lambda_reg, verbose=verbose)

    # Build evaluators
    phi_fn, grad_fn, psi_fn, grad_psi_fn, lap_psi_fn, _ = build_evaluators_mfs(
        Pn, Yn, alpha, phi_t, phi_p, a, scinfo, grad_t, grad_p
    )

    # Interior ring diagnostics
    Wsum = float(jnp.sum(W))
    Wsqrt = jnp.sqrt(W)
    h_med = float(np.median(rk))
    eps_n = max(1e-6, interior_eps_factor * h_med)
    eps_w = eps_n / scinfo.scale
    P_in  = P - eps_w * N
    grad_on_ring = grad_fn(P_in)
    n_dot_grad   = jnp.sum(N * grad_on_ring, axis=1)
    bc_w2 = float(jnp.sqrt(jnp.dot(W, n_dot_grad**2)))          # ||n·∇φ||_W2 on Γ₋
    flux = float(jnp.dot(W, n_dot_grad))                        # ∫Γ₋ n·∇φ dS
    grad_mag_ring = jnp.linalg.norm(grad_on_ring, axis=1)
    lap_in = lap_psi_fn(P_in)
    lap_l2 = float(jnp.linalg.norm(lap_in))                     # ||∇²ψ||_2 on Γ₋
    alpha_norm = float(jnp.linalg.norm(alpha))

    # Rough normal-equations condition proxy (reuse LS-α build)
    Aw = Wsqrt[:, None] * A
    ATA = Aw.T @ Aw
    NE  = ATA + (lambda_reg**2) * jnp.eye(A.shape[1], dtype=A.dtype)
    condNE = float(np.linalg.cond(np.asarray(NE)))

    metrics = dict(
        bc_w2=bc_w2,
        flux_abs=abs(flux),
        lap_l2=lap_l2,
        alpha_norm=alpha_norm,
        condNE=condNE,
        a=a,
        delta_n=delta_n,
        eps_n=eps_n,
        eps_w=eps_w,
        grad_on_ring=grad_on_ring,
        lap_in=lap_in,
        phi_fn=phi_fn,
        grad_fn=grad_fn
    )
    return alpha, metrics

def autotune(P, N, Pn, Nn, W, rk, scinfo,
             use_mv=True,
             mv_weight=0.5,
             interior_eps_factor=5e-3,
             verbose=True):
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

    # ---- 2D optimization over (log_sf, log_lambda) ----
    h_med = float(np.median(rk))
    obj = make_objective_for_delta_lambda(
        P, N, Pn, Nn, W, rk, scinfo,
        grad_t, grad_p, a, g_raw,
        interior_eps_factor=interior_eps_factor, h_med=h_med
    )

    # robust starting point (you were scanning 1.5..2.5)
    log_sf0  = jnp.log(2.0)
    log_lam0 = jnp.log(1e-2)
    p0 = jnp.array([log_sf0, log_lam0], dtype=jnp.float64)

    # small box for sf, keep λ unbounded in log-space (still positive)
    p_star, f_star, g_star, H_star = bfgs_2d(
        obj, p0, sf_min=3.5, sf_max=6.0, max_iter=25, tol=1e-7
    )
    log_sf_star, log_lam_star = p_star
    sf_star  = float(jnp.exp(log_sf_star))
    lam_star = float(jnp.exp(log_lam_star))

    # Build final sources at optimum and reuse downstream
    Yn, _ = build_mfs_sources(Pn, Nn, rk, scinfo, source_factor=sf_star, verbose=False)

    if verbose:
        print(f"[OPT] δ/source_factor* = {sf_star:.3f}, λ* = {lam_star:.3e}, J* = {float(f_star):.3e}")

    return dict(
        source_factor=sf_star, lambda_reg=lam_star,
        a=a, phi_t=phi_t, grad_t=grad_t, phi_p=phi_p, grad_p=grad_p, Yn=Yn,
        a_hat=a_hat, kind=kind
    )

def make_objective_for_delta_lambda(P, N, Pn, Nn, W, rk, scinfo,
                                    grad_t, grad_p, a, g_raw,
                                    interior_eps_factor, h_med):
    eps_factor_obj = jnp.maximum(2e-2, interior_eps_factor)
    Wsqrt = jnp.sqrt(W)
    eps_n = jnp.maximum(1e-6, eps_factor_obj * h_med)
    eps_w = eps_n / scinfo.scale
    P_in  = P - eps_w * N

    # --- NEW: precompute ring weights for the objective (constant wrt p) ---
    # Use a modest k to keep it cheap; it does not depend on (sf, λ).
    W_ring_obj = build_ring_weights(P_in, Pn, k=32)

    def objective(p):
        log_sf, log_lam = p
        sf  = jnp.exp(log_sf)
        lam = jnp.exp(log_lam)

        Yn, _ = build_mfs_sources(Pn, Nn, rk, scinfo, source_factor=sf, verbose=False)
        A, _  = build_system_matrices(Pn, Nn, Yn, W, grad_t, grad_p, scinfo,
                                      use_mv=True, center_D=True, verbose=False)

        # Boundary quantities (unchanged)
        c_bdry, d_bdry = make_flux_constraint(A, W, g_raw)

        # Interior-ring constraint — use the precomputed ring weights
        Xn_in = (P_in - scinfo.center) * scinfo.scale
        A_in  = build_A_rows_at_points(Xn_in, N, Yn, scinfo)
        Gt_in = grad_t(Xn_in); Gp_in = grad_p(Xn_in)
        g_in  = scinfo.scale * jnp.sum(N * (a[0]*Gt_in + a[1]*Gp_in), axis=1)
        c_int, d_int = make_flux_constraint(A_in, W_ring_obj, g_in)

        alpha = solve_alpha_with_rhs_hard_flux_multi(
            A, W, g_raw, lam=lam, constraints=[(c_int, -d_int,), (c_bdry, -d_bdry)],
            verbose=False
        )

        res_w = Wsqrt * (A @ alpha + g_raw)
        term_res = jnp.dot(res_w, res_w)

        zero_like = lambda Xn: jnp.zeros((Xn.shape[0],), dtype=Xn.dtype)
        phi_t0, grad_t_fn, phi_p0, grad_p_fn = zero_like, grad_t, zero_like, grad_p
        phi_fn, grad_fn, _, _, _, _ = build_evaluators_mfs(
            Pn, Yn, alpha, phi_t0, phi_p0, a, scinfo, grad_t_fn, grad_p_fn
        )
        grad_in = grad_fn(P_in)
        n_dot   = jnp.sum(N * grad_in, axis=1)
        term_bc = jnp.dot(W, n_dot**2)

        reg = 1e-6 * (log_sf**2 + log_lam**2)
        return term_res + term_bc + reg

    return jit(jax.value_and_grad(objective))

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

def _rho_floor_from_points(Pn, c, E, frac=0.02):
    # project once and take a robust floor: a few percent of median radius
    Ploc = (Pn - c) @ E
    _, rho = cylindrical_angle_and_radius(Ploc[:, :2])
    rho_med = float(jnp.median(rho))
    return max(1e-3 * rho_med, frac * rho_med)  # safety + user-tunable

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

def kNN_weights_for_points(Xn_like, k=32):
    # Xn_like: points in *normalized* coords whose weights you want
    # Build weights in the best-fit (u,v) plane using the same PCA as boundary
    # We re-use best_fit_axis on the *boundary* Pn to get the plane.
    # (Pass E,c down if you prefer.)
    return None  # placeholder; we’ll inline below instead

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

# ------------------------- MFS source cloud ---------------------- #
# --- replace your current build_mfs_sources with this ---
def build_mfs_sources(Pn, Nn, rk, scale_info, source_factor=2.0, verbose=True):
    """
    Place MFS sources outside the domain along outward normals:
      Y = P + δ N, with δ = source_factor * median(rk) in *normalized* units.
    JAX-safe for JIT when verbose=False.
    """
    # JAX-friendly median (works both traced and eager)
    rk_med = jnp.median(rk)
    delta_n = source_factor * rk_med              # stays as a JAX value if traced
    Yn = Pn + delta_n * Nn                        # broadcast in normalized coords

    if verbose:
        # Convert to Python floats only for printing (outside JIT)
        dn = float(np.asarray(delta_n))
        print(f"[MFS] Using source offset δ_n={dn:.4g} (normalized units).")
        print(f"[MFS] Sources count: {int(np.asarray(Yn.shape[0]))} (one per boundary point).")

    return Yn, delta_n

# ----------------------------- System build ---------------------- #
@partial(jit,static_argnames=("grad_t", "grad_p", "use_mv", "center_D", "verbose"))
def build_system_matrices(Pn, Nn, Yn, W, grad_t, grad_p, scinfo, use_mv=True, center_D=True, verbose=True):
    """
    Build collocation matrix for Neumann BC:
      A_ij = n_i · ∇_x G( x_i , y_j ),  x_i = Pn[i], y_j = Yn[j]
    Returns A (world units). D is not needed in the current solve path.
    """
    X = Pn

    @jit
    def row_kernel(xi, ni):
        grads = vmap(lambda yj: grad_green_x(xi, yj))(Yn)   # (M,3)
        return jnp.dot(grads, ni)                           # (M,)

    A = vmap(row_kernel)(X, Nn)                              # (N, M)
    A = scinfo.scale * A

    if verbose:
        # All JAX ops; no numpy, no Python formatting of tracers
        shape0 = A.shape[0]; shape1 = A.shape[1]
        Amin = jnp.min(jnp.abs(A)); Amed = jnp.median(jnp.abs(A)); Amax = jnp.max(jnp.abs(A))
        jax_debug.print(
            "[SYS] A shape=({n},{m}), |A| stats: min={mn:.3e}, median={md:.3e}, max={mx:.3e}",
            n=shape0, m=shape1, mn=Amin, md=Amed, mx=Amax)
    # keep old signature but return dummy D to avoid changing callers
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

@jit
def build_A_rows_at_points(Xn_eval, N_eval, Yn, scinfo):
    @jit
    def row_kernel(xi, ni):
        grads = vmap(lambda yj: grad_green_x(xi, yj))(Yn)   # (M,3)
        return jnp.dot(grads, ni)                           # (M,)
    A_in = vmap(row_kernel)(Xn_eval, N_eval)                # (N,M)
    return scinfo.scale * A_in                              # to world units

def make_flux_constraint(A_like, W_like, g_like):
    # Return (c_vec, d_scalar) with c_vec = A_like^T W_like 1, d = W_like·g_like
    Wv = jnp.asarray(W_like).reshape(-1)
    c_vec = (A_like.T * Wv[None, :]).sum(axis=1)   # (M,)
    d_val = jnp.dot(Wv, g_like)                    # scalar
    return c_vec, d_val

# ----------------- Evaluators & Laplacian(ψ) ------------------- #
def build_evaluators_mfs(Pn, Yn, alpha, phi_t, phi_p, a, scinfo: ScaleInfo,
                         grad_t_fn, grad_p_fn):
    Y = Yn

    @jit
    def S_alpha_at(xn):
        Gvals = vmap(lambda y: green_G(xn, y))(Y)
        return jnp.dot(Gvals, alpha)

    @jit
    def grad_S_alpha_at(xn):
        Grads = vmap(lambda y: grad_green_x(xn, y))(Y)  # (M,3)
        return jnp.sum(Grads * alpha[:, None], axis=0)

    S_batch  = vmap(S_alpha_at)
    dS_batch = vmap(grad_S_alpha_at)

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
        return S_batch(Xn)

    @jit
    def grad_psi_fn_world(X):
        Xn = (X - scinfo.center) * scinfo.scale
        return (scinfo.scale) * dS_batch(Xn)

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
                              q=0.02, ds_frac=0.02, k_cap=32, side="low"):
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
    A_cap = build_A_rows_at_points(Xn_cap, N_cap, Yn, scinfo)

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
         source_factor=2.0,           # δ_n = source_factor * median(rk)
         lambda_reg=1e-3,             # Tikhonov λ
         mv_weight=0.5,               # regularization weight for [a_t,a_p]
         interior_eps_factor=5e-3,  # ε ~ interior offset for evaluation, in *normalized* h units
         verbose=True, show_normals=False,
         toroidal_flux=None,          # NEW: prescribe Φ_t (sets a_t = Φ_t/(2π)) if not None
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
    best = autotune(P, N, Pn, Nn, W, rk, scinfo, use_mv=use_mv, mv_weight=mv_weight, interior_eps_factor=interior_eps_factor, verbose=True)
    source_factor_opt = best["source_factor"]
    lambda_reg_opt    = best["lambda_reg"]
    # Reuse pre-built things to avoid recompute:
    phi_t, grad_t, phi_p, grad_p, Yn, kind = best["phi_t"], best["grad_t"], best["phi_p"], best["grad_p"], best["Yn"], best["kind"]
    delta_n = float(np.median(np.linalg.norm(np.asarray(Yn) - np.asarray(Pn), axis=1)))

    A, D = build_system_matrices(Pn, Nn, Yn, W, grad_t, grad_p, scinfo,
                                 use_mv=use_mv, center_D=True, verbose=verbose)

    if use_mv:
        grad_t_bdry, grad_p_bdry = grad_t(Pn), grad_p(Pn)
        a, D_raw, D0 = fit_mv_coeffs_minimize_rhs(Nn, W, grad_t_bdry, grad_p_bdry, verbose=verbose)
        g_raw = scinfo.scale * jnp.sum(Nn * (a[0]*grad_t_bdry + a[1]*grad_p_bdry), axis=1)
    else:
        a = jnp.zeros((2,), dtype=jnp.float64)
        g_raw = jnp.zeros((Pn.shape[0],), dtype=jnp.float64)

    # --- OVERRIDE a_t if a toroidal flux is prescribed ---
    if use_mv and (toroidal_flux is not None):
        a_t_fixed = float(toroidal_flux) / (2.0 * np.pi)
        a = jnp.array([a_t_fixed, 0.0], dtype=jnp.float64)
        if verbose:
            print(f"[MV-FIX] Prescribing toroidal flux Φ_t={toroidal_flux:.6g} ⇒ a_t={a_t_fixed:.6g}; setting a_p=0.")
        # Rebuild g_raw with fixed a
        grad_t_bdry = grad_t(Pn)
        grad_p_bdry = grad_p(Pn)
        g_raw = scinfo.scale * jnp.sum(Nn * (a[0]*grad_t_bdry + a[1]*grad_p_bdry), axis=1)

    # === Prepare constraints ===
    # Boundary constraint:
    c_bdry, d_bdry = make_flux_constraint(A, W, g_raw)

    # Interior ring for *constraint* (same ring you use for diagnostics)
    eps_n = max(1e-6, interior_eps_factor * h_med)    # normalized
    eps_w = eps_n / scinfo.scale
    P_in  = P - eps_w * N
    Xn_in = (P_in - scinfo.center) * scinfo.scale
    A_in  = build_A_rows_at_points(Xn_in, N, Yn, scinfo)

    # g_in from MV on interior:
    # OPTIONAL speed path: reuse boundary normals for ring; ϕ̂_tan, θ̂ remain consistent for tiny offsets
    use_fast_ring = (kind == "torus")  # mirrors need accurate θ̂
    Gt_in = grad_t(Xn_in)
    if use_fast_ring:
        # Build poloidal direction with boundary normals to avoid nearest-neighbor search
        x, y = Xn_in[:, 0], Xn_in[:, 1]
        r2   = jnp.maximum(1e-30, x*x + y*y)
        phi_hat   = jnp.stack([-y / jnp.sqrt(r2), x / jnp.sqrt(r2), jnp.zeros_like(x)], axis=1)
        phi_tan   = phi_hat - jnp.sum(phi_hat * N, axis=1, keepdims=True) * N
        phi_tan   = phi_tan / jnp.maximum(1e-30, jnp.linalg.norm(phi_tan, axis=1, keepdims=True))
        theta_hat = jnp.cross(N, phi_tan)
        theta_hat = theta_hat / jnp.maximum(1e-30, jnp.linalg.norm(theta_hat, axis=1, keepdims=True))
        Gp_in = theta_hat
    else:
        Gp_in = grad_p(Xn_in)

    g_in  = scinfo.scale * jnp.sum(N * (a[0]*Gt_in + a[1]*Gp_in), axis=1)

    # Build ring weights once and use them for the constraint
    W_ring = build_ring_weights(P_in, Pn, k=k_nn)
    c_int, d_int = make_flux_constraint(A_in, W_ring, g_in)

    # --- Add two virtual end-cap flux constraints (close the open sleeve) ---
    try:
        c_cap_low,  d_cap_low  = build_cap_flux_constraint(P, N, Pn, Nn, scinfo, Yn,
                                                        grad_t, grad_p, a, best["a_hat"],
                                                        q=0.02, ds_frac=0.02, k_cap=k_nn, side="low")
        c_cap_high, d_cap_high = build_cap_flux_constraint(P, N, Pn, Nn, scinfo, Yn,
                                                        grad_t, grad_p, a, best["a_hat"],
                                                        q=0.02, ds_frac=0.02, k_cap=k_nn, side="high")
        cap_constraints = [(c_cap_low,  -d_cap_low),
                        (c_cap_high, -d_cap_high)]
    except Exception as e:
        print("[WARN] Cap constraints failed; continuing without them:", e)
        cap_constraints = []

    # Choose your policy:
    # 1) Most robust in practice: enforce on Γ₋ only
    # constraints = [(c_int, -d_int)]
    # 2) Enforce on both Γ and Γ₋ (two constraints):
    # constraints = [(c_bdry, -d_bdry), (c_int, -d_int)]
    # 3) Enforce on Γ, Γ₋, and end-caps (four constraints):
    constraints = [(c_bdry, -d_bdry), (c_int, -d_int)] + cap_constraints  # <-- NEW

    alpha = solve_alpha_with_rhs_hard_flux_multi(
        A, W, g_raw, lam=lambda_reg_opt, constraints=constraints, verbose=verbose
    )
    
    print(f"[SOL] a_t={float(a[0]):.6g}, a_p={float(a[1]):.6g}, ||alpha||₂={float(jnp.linalg.norm(alpha)):.3e}")

    phi_fn, grad_fn, psi_fn, grad_psi_fn, lap_psi_fn, _ = build_evaluators_mfs(
        Pn, Yn, alpha, phi_t, phi_p, a, scinfo, grad_t, grad_p
    )

    # === Diagnostics on Γ (illustrative; gradients may still be large due to proximity to sources) ===
    grad_on_Gamma = grad_fn(P)
    vec_stats("[EVAL Γ] |∇φ|", jnp.linalg.norm(grad_on_Gamma, axis=1))

    # === Diagnostics on Γ₋ (interior offset): reliable ===
    eps_n = max(1e-6, interior_eps_factor * h_med)  # normalized offset inward
    eps_w = eps_n / scinfo.scale                     # world offset
    P_in = P - eps_w * N
    print(f"[DIAG] Using interior-offset ring Γ₋ with eps_n={eps_n:.3g} (normalized), eps_world={eps_w:.3g}.")

    grad_on_ring = grad_fn(P_in)
    grad_mag_ring = jnp.linalg.norm(grad_on_ring, axis=1)
    vec_stats("[EVAL Γ₋] |∇φ|", grad_mag_ring)
    
    n_dot_grad_ring = jnp.sum(N * grad_on_ring, axis=1)
    grad_mag_ring   = jnp.linalg.norm(grad_on_ring, axis=1)
    rn = jnp.abs(n_dot_grad_ring) / jnp.maximum(1e-30, grad_mag_ring)
    vec_stats("[EVAL Γ₋] normalized BC |n·∇φ|/|∇φ|", rn)

    # Flux neutrality on Γ₋ (should be ~0)
    # reuse the ring weights from the constraint stage
    n_dot_grad_ring = jnp.sum(N * grad_on_ring, axis=1)
    flux = float(jnp.dot(W_ring, n_dot_grad_ring))
    area_ring = float(jnp.sum(W_ring))
    print(f"[CHK] Flux neutrality on Γ₋: ∫ n·∇φ dS ≈ {flux:.6e}  (avg={flux/area_ring:.3e})")

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
    lap_in = lap_psi_fn(P_in)
    vec_stats("[LAP Γ₋] |∇²ψ|", jnp.abs(lap_in))

    # Plots (all on Γ₋)
    plot_geometry_and_solution(P, N, grad_on_ring, title_suffix="",
                           show_normals=show_normals, kind=kind, a_hat=best.get("a_hat", None))
    plot_boundary_condition_errors(P, N, grad_on_ring)
    plot_laplacian_errors_on_interior_band(P, lap_in, eps_w)

    return dict(
        P=P, N=N, Pn=Pn, W=W, rk=rk, h_med=h_med,
        alpha=alpha, a=a, delta_n=delta_n, eps_n=eps_n,
        phi_fn=phi_fn, grad_fn=grad_fn, psi_fn=psi_fn, grad_psi_fn=grad_psi_fn, laplacian_psi_fn=lap_psi_fn
    )

if __name__ == "__main__":
    # Set these to your file names as needed:
    xyz_csv = "wout_precise_QA.csv"
    nrm_csv = "wout_precise_QA_normals.csv"
    # xyz_csv = "wout_precise_QH.csv"
    # nrm_csv = "wout_precise_QH_normals.csv"
    # xyz_csv = "slam_surface.csv"
    # nrm_csv = "slam_surface_normals.csv"
    # xyz_csv = "sflm_rm4.csv"
    # nrm_csv = "sflm_rm4_normals.csv"
    _ = main(
        xyz_csv=xyz_csv, nrm_csv=nrm_csv,
        use_mv=True, k_nn=64,
        source_factor=2.0,        # try 1.5–3.0 if needed
        lambda_reg=1e-3,          # try 3e-4..3e-3 depending on noise
        mv_weight=0.5,            # 0.2..1.0 is reasonable
        interior_eps_factor=5e-3,
        verbose=True,
        toroidal_flux=1.0
    )
