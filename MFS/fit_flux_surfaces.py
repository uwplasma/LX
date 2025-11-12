#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Flux-Surface Solver for MFS (Neumann Laplace)
====================================================
- Optimizes *all* surfaces together (global vector of Fourier DOFs)
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# ----------------------------- Utilities ----------------------------- #

EPS = 1e-12

def safe_norm(x, axis=None):
    return jnp.sqrt(jnp.maximum(EPS, jnp.sum(x*x, axis=axis)))

def finite_or(x, repl=0.0):
    return jnp.where(jnp.isfinite(x), x, jnp.array(repl, dtype=x.dtype))

def nan_report(name, arr):
    arr = jnp.asarray(arr)
    n_nan = int(jnp.sum(jnp.isnan(arr)))
    n_inf = int(jnp.sum(~jnp.isfinite(arr)) - jnp.sum(jnp.isnan(arr)))
    return f"{name}: shape={tuple(arr.shape)}, nNaN={n_nan}, nInf={n_inf}"

def print_stats_table(label, stats_mat):
    # stats_mat shape (K,4): [min, median, p95, max]
    stats = np.asarray(stats_mat)  # safe host materialization
    print(f"      {label} stats per surface [min  median  p95  max]:")
    for k,row in enumerate(stats, 1):
        print(f"        S{k:<2d}: {row[0]: .3e}  {row[1]: .3e}  {row[2]: .3e}  {row[3]: .3e}")

@jit
def wrap_angle(x):
    return jnp.mod(x + jnp.pi, 2*jnp.pi) - jnp.pi

def _circular_fill(vals):
    """Fill NaNs on a circle by nearest non-NaN values (forward/backward)."""
    vals_np = np.asarray(vals)
    isnan = np.isnan(vals_np)
    if not isnan.any():
        return jnp.asarray(vals_np)

    N = len(vals_np)
    # indices of last seen finite value when walking forward/backward circularly
    forward = vals_np.copy()
    last = np.nan
    for k in range(2*N):  # two laps to propagate
        i = k % N
        if not np.isnan(forward[i]):
            last = forward[i]
        else:
            forward[i] = last

    backward = vals_np.copy()
    last = np.nan
    for k in range(2*N-1, -1, -1):  # reverse laps
        i = k % N
        if not np.isnan(backward[i]):
            last = backward[i]
        else:
            backward[i] = last

    filled = forward
    # where still nan (all were nan), fall back to backward
    mask = np.isnan(np.asarray(vals_np))
    filled[mask] = backward[mask]
    # in the pathological case all were NaN, fill with zeros
    if np.isnan(filled).all():
        filled[:] = 0.0
    return jnp.asarray(filled)

def boundary_phi_stats(sol, nphi_bins=256, nfit=8):
    """Return callables R0(φ), Rmin(φ), Rmax(φ) (Fourier-fitted, robust)."""
    Pcen = sol["P"] - sol["center"][None, :]
    Rb   = jnp.sqrt(Pcen[:,0]**2 + Pcen[:,1]**2)
    Phib = jnp.arctan2(Pcen[:,1], Pcen[:,0])

    phi_grid = jnp.linspace(-jnp.pi, jnp.pi, nphi_bins, endpoint=False)
    dphi = 2*jnp.pi / nphi_bins

    # Bin membership (Nt x Nbins), boolean
    def in_bin(phi0):
        return (jnp.abs(wrap_angle(Phib - phi0)) <= 0.5*dphi).astype(jnp.float64)
    W = vmap(in_bin)(phi_grid)            # (Nbins, Npts)
    W = W.T                                # (Npts, Nbins)

    # Counts per bin
    cnt = jnp.sum(W, axis=0)               # (Nbins,)

    # Weighted stats with safeguards
    sumR = (Rb[:,None] * W).sum(axis=0)
    meanR = jnp.where(cnt>0, sumR / jnp.maximum(cnt, 1.0), jnp.nan)

    # For min/max we mask non-members to +inf/-inf so min/max ignore them
    big = 1e300
    R_for_min = jnp.where(W>0, Rb[:,None], big)
    R_for_max = jnp.where(W>0, Rb[:,None], -big)
    minR = jnp.where(cnt>0, jnp.min(R_for_min, axis=0), jnp.nan)
    maxR = jnp.where(cnt>0, jnp.max(R_for_max, axis=0), jnp.nan)

    # Fill empty bins by circular nearest-neighbor
    meanR = _circular_fill(meanR)
    minR  = _circular_fill(minR)
    maxR  = _circular_fill(maxR)

    # Low-order Fourier LS fit on the circle
    # Design matrix: [1, cos φ, sin φ, ..., cos nfit φ, sin nfit φ]
    M = [jnp.ones_like(phi_grid)]
    for n in range(1, nfit+1):
        M.append(jnp.cos(n*phi_grid))
        M.append(jnp.sin(n*phi_grid))
    A = jnp.stack(M, axis=1)               # (Nbins, 2*nfit+1)
    ATA = A.T @ A + 1e-12*jnp.eye(A.shape[1])

    def fit_and_eval(y):
        c = jnp.linalg.solve(ATA, A.T @ y)
        def eval_fn(phi):
            out = c[0]*jnp.ones_like(phi)
            idx = 1
            for n in range(1, nfit+1):
                out = out + c[idx]*jnp.cos(n*phi); idx += 1
                out = out + c[idx]*jnp.sin(n*phi); idx += 1
            return out
        return eval_fn

    R0_of_phi   = fit_and_eval(meanR)
    Rmin_of_phi = fit_and_eval(minR)
    Rmax_of_phi = fit_and_eval(maxR)
    return R0_of_phi, Rmin_of_phi, Rmax_of_phi

def fourier_fit_phi(y_phi, phi_grid, nfit):
    """
    Least-squares fit on the circle:
      y(φ) ≈ c0 + Σ_{n=1..nfit} [ c_{c,n} cos(nφ) + c_{s,n} sin(nφ) ]
    Returns the vector of coefficients and an evaluator.
    """
    M = [jnp.ones_like(phi_grid)]
    for n in range(1, nfit+1):
        M.append(jnp.cos(n*phi_grid))
        M.append(jnp.sin(n*phi_grid))
    A = jnp.stack(M, axis=1)                        # (N, 2*nfit+1)
    ATA = A.T @ A + 1e-12*jnp.eye(A.shape[1])
    c   = jnp.linalg.solve(ATA, A.T @ y_phi)

    def eval_fn(phi):
        out = c[0]*jnp.ones_like(phi)
        idx = 1
        for n in range(1, nfit+1):
            out = out + c[idx]*jnp.cos(n*phi); idx += 1
            out = out + c[idx]*jnp.sin(n*phi); idx += 1
        return out
    return c, eval_fn

# --------------------- Load MFS + grad phi evaluator ------------------ #

def load_mfs_solution(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    center = jnp.asarray(data["center"], dtype=jnp.float64)
    scale = float(data["scale"])
    Yn = jnp.asarray(data["Yn"], dtype=jnp.float64)
    alpha = jnp.asarray(data["alpha"], dtype=jnp.float64)
    a_vec = jnp.asarray(data["a"], dtype=jnp.float64)
    a_hat = jnp.asarray(data["a_hat"], dtype=jnp.float64)
    P = jnp.asarray(data["P"], dtype=jnp.float64)
    N = jnp.asarray(data["N"], dtype=jnp.float64)
    kind = data["kind"].item() if hasattr(data["kind"], "item") else str(data["kind"])
    Pn = (P - center) * scale
    return dict(center=center, scale=scale, Yn=Yn, alpha=alpha, a_vec=a_vec,
                a_hat=a_hat, P=P, N=N, Pn=Pn, kind=kind)

@jit
def green_G_grad(xn, y):
    r = xn - y
    r2 = jnp.sum(r*r, axis=-1)
    r3 = jnp.maximum(EPS, r2 * jnp.sqrt(r2))
    return -r / (4.0 * jnp.pi * r3[..., None])

@jit
def grad_azimuth_about_axis(Xn, a_hat):
    a = a_hat / jnp.maximum(EPS, jnp.linalg.norm(a_hat))
    r_par  = jnp.sum(Xn * a[None,:], axis=1, keepdims=True) * a[None,:]
    r_perp = Xn - r_par
    r2     = jnp.sum(r_perp*r_perp, axis=1, keepdims=True)
    cross  = jnp.cross(a[None,:], r_perp)
    return cross / jnp.maximum(EPS, r2)

def build_grad_phi_fn(sol):
    center, scale = sol["center"], sol["scale"]
    Yn, alpha = sol["Yn"], sol["alpha"]
    a_t, a_p = sol["a_vec"][0], sol["a_vec"][1]
    a_hat = sol["a_hat"]

    @jit
    def grad_psi_batch(X):  # world -> world
        Xn = (X - center[None, :]) * scale                 # (P,3)
        # pairwise differences: (P,M,3)
        r = Xn[:, None, :] - Yn[None, :, :]
        r2 = jnp.sum(r * r, axis=-1)                       # (P,M)
        r3 = jnp.maximum(EPS, r2 * jnp.sqrt(r2))           # (P,M)
        g = -r / (4.0 * jnp.pi * r3[..., None])            # (P,M,3)
        # weighted sum over sources:
        G = jnp.tensordot(g, alpha, axes=([1], [0]))       # (P,3)
        return scale * G

    @jit
    def grad_t_batch(X):           # toroidal (azimuth) gradient about axis a_hat
        Xn = (X - center[None,:]) * scale
        return grad_azimuth_about_axis(Xn, a_hat)

    @jit
    def grad_p_batch(X):           # poloidal-like direction (θ̂): project φ̂ to tangent & rotate with n
        Xn = (X - center[None,:]) * scale
        # Use a local tangent frame built from axis-aware azimuth unit
        a = a_hat / jnp.maximum(EPS, jnp.linalg.norm(a_hat))
        r_par  = jnp.sum(Xn * a[None,:], axis=1, keepdims=True) * a[None,:]
        r_perp = Xn - r_par
        r2     = jnp.maximum(EPS, jnp.sum(r_perp*r_perp, axis=1, keepdims=True))
        phi_hat = jnp.cross(a[None,:], r_perp) / r2  # NOTE: unit up to 1/|r_perp|; that’s ok for direction
        # Build a crude surface normal by nearest-axis radial direction:
        n_guess = (X - center[None,:]) / jnp.maximum(EPS, safe_norm(X - center[None,:], axis=1,))[:,None]
        phi_tan = phi_hat - jnp.sum(phi_hat*n_guess, axis=1, keepdims=True)*n_guess
        phi_tan = phi_tan / jnp.maximum(EPS, safe_norm(phi_tan, axis=1))[:,None]
        theta_hat = jnp.cross(n_guess, phi_tan)
        theta_hat = theta_hat / jnp.maximum(EPS, safe_norm(theta_hat, axis=1))[:,None]
        return theta_hat

    @jit
    def grad_phi_batch(X):
        G_psi = grad_psi_batch(X)
        G_add = scale * (a_t * grad_t_batch(X) + a_p * grad_p_batch(X))
        G = G_psi + G_add
        # DEBUG: norms to catch a zero field
        nGpsi = jnp.mean(safe_norm(G_psi, axis=1))
        nGadd = jnp.mean(safe_norm(G_add, axis=1))
        nGall = jnp.mean(safe_norm(G, axis=1))
        # jax.debug.print("[∇φ] ⟨|∇ψ|⟩={:.3e}  ⟨|∇add|⟩={:.3e}  ⟨|∇φ|⟩={:.3e}", nGpsi, nGadd, nGall)
        return G

    return grad_phi_batch


# ----------------------- Fourier Surfaces (safe) ---------------------- #

@dataclass
class FourierModes:
    R_modes: List[Tuple[int,int]]
    Z_modes: List[Tuple[int,int]]
    @classmethod
    def generate(cls, mmax, nmax):
        Rs, Zs = [], []
        for m in range(0, mmax+1):
            for n in range(0, nmax+1):
                if not (m == 0 and n == 0):
                    Rs.append((m, n))  # (0,0) handled by R0, not here
                    Zs.append((m, n))  # Z has no Z0, still exclude (0,0)
        return cls(Rs, Zs)

class FourierSurface:
    """R = R0 + Σ_(m,n)!=(0,0) a_R cos(mθ − nφ);  Z = Σ_(m,n)!=(0,0) a_Z sin(mθ − nφ)"""
    def __init__(self, modes: FourierModes, R0: float):
        self.R_modes = modes.R_modes         # excludes (0,0)
        self.Z_modes = modes.Z_modes         # excludes (0,0)
        self.M_R = len(self.R_modes)
        self.M_Z = len(self.Z_modes)
        self.R0 = float(max(1e-6, R0))       # explicit major radius

    def split(self, coeffs):
        cR = coeffs[:self.M_R]; cZ = coeffs[self.M_R:]
        return cR, cZ

    def evaluate_RZ(self, coeffs, theta, phi):
        cR, cZ = self.split(coeffs)
        # R = R0 + sum_{(m,n)!=(0,0)} ...
        R = jnp.full_like(theta, self.R0)
        for i,(m,n) in enumerate(self.R_modes):
            R = R + cR[i] * jnp.cos(m*theta - n*phi)
        # Z has no Z0
        Z = jnp.zeros_like(theta)
        for i,(m,n) in enumerate(self.Z_modes):
            Z = Z + cZ[i] * jnp.sin(m*theta - n*phi)
        return R, Z

    def cartesian(self, coeffs, theta, phi):
        R, Z = self.evaluate_RZ(coeffs, theta, phi)
        x = R * jnp.cos(phi); y = R * jnp.sin(phi)
        return jnp.stack([x, y, Z], axis=-1)

    def tangents_normal_area(self, coeffs, theta, phi):
        cR, cZ = self.split(coeffs)
        R, Z = self.evaluate_RZ(coeffs, theta, phi)

        # dR/dθ, dR/dφ from the cosine series (no constant)
        dR_dtheta = jnp.zeros_like(theta)
        dR_dphi   = jnp.zeros_like(theta)
        for i,(m,n) in enumerate(self.R_modes):
            arg = m*theta - n*phi
            dR_dtheta = dR_dtheta + (-m) * cR[i] * jnp.sin(arg)
            dR_dphi   = dR_dphi   + ( +n) * cR[i] * jnp.sin(arg)

        dZ_dtheta = jnp.zeros_like(theta)
        dZ_dphi   = jnp.zeros_like(theta)
        for i,(m,n) in enumerate(self.Z_modes):
            arg = m*theta - n*phi
            dZ_dtheta = dZ_dtheta + m  * cZ[i] * jnp.cos(arg)
            dZ_dphi   = dZ_dphi   + (-n) * cZ[i] * jnp.cos(arg)

        dx_dtheta = dR_dtheta * jnp.cos(phi)
        dy_dtheta = dR_dtheta * jnp.sin(phi)
        dz_dtheta = dZ_dtheta
        t_theta = jnp.stack([dx_dtheta, dy_dtheta, dz_dtheta], axis=-1)

        dx_dphi = dR_dphi * jnp.cos(phi) - R * jnp.sin(phi)
        dy_dphi = dR_dphi * jnp.sin(phi) + R * jnp.cos(phi)
        dz_dphi = dZ_dphi
        t_phi = jnp.stack([dx_dphi, dy_dphi, dz_dphi], axis=-1)

        cross = jnp.cross(t_theta, t_phi)
        area  = safe_norm(cross, axis=-1)
        n_hat = cross / area[..., None]
        return t_theta, t_phi, n_hat, area, R, Z
    
def surface_area_and_volume(surf, coeffs, TH, PH):
    """
    Discrete area and enclosed volume from param grid:
    A = ∬ |∂θr×∂φr| dθ dφ
    V = (1/3) ∬ r·(∂θr×∂φr) dθ dφ
    """
    tθ, tφ, n_hat, area, _R, _Z = surf.tangents_normal_area(coeffs, TH, PH)
    # metric weight dθ dφ (uniform grid):
    dθ = (TH[1,0] - TH[0,0]) if TH.shape[0] > 1 else (2*jnp.pi)
    dφ = (PH[0,1] - PH[0,0]) if PH.shape[1] > 1 else (2*jnp.pi)
    dA = area * dθ * dφ

    # position r(θ,φ)
    P = surf.cartesian(coeffs, TH, PH)
    cross = jnp.cross(tθ, tφ)
    dV = (1.0/3.0) * jnp.sum(P * cross, axis=-1) * dθ * dφ

    A = jnp.sum(dA)
    V = jnp.sum(dV)
    return A, V

def split_modes(c, surf):
    return c[:surf.M_R], c[surf.M_R:]

def scale_nonzero_modes(c, surf, g):
    cR, cZ = split_modes(c, surf)
    # Only scale m>=1 modes (all coefficients already exclude (0,0))
    return jnp.concatenate([cR * g, cZ * g], axis=0)

def find_scale_to_match(surf, c, TH, PH, target, which="area"):
    """
    Solve A(g)=target (or V(g)=target) for g using 3-5 Newton steps.
    We treat R0 as fixed and scale all non-(0,0) modes by g.
    """
    g = 1.0

    def geom(gval):
        c_g = scale_nonzero_modes(c, surf, gval)
        A, V = surface_area_and_volume(surf, c_g, TH, PH)
        return (A, V)

    for _ in range(3):
        A, V = geom(g)
        val = A if which == "area" else V
        # finite-difference slope (cheap, stable)
        h = 5e-3
        A2, V2 = geom(g + h)
        val2 = A2 if which == "area" else V2
        deriv = (val2 - val) / h
        # guard
        deriv = jnp.where(jnp.abs(deriv) < 1e-14, jnp.sign(deriv)*1e-14 + 0.0, deriv)
        g = g + (target - val) / deriv
        # softly cap insane excursions
        g = jnp.clip(g, 0.5, 2.0)
    return g

    
# ---------------------- Residual (Gauss–Newton) ----------------------- #

def _outward_normalize(n_hat, pts, center, a_hat):
    a_unit = a_hat / jnp.maximum(EPS, jnp.linalg.norm(a_hat))
    v = pts - center[None, None, :]
    v_par = jnp.sum(v * a_unit[None, None, :], axis=-1, keepdims=True) * a_unit[None, None, :]
    r_perp = v - v_par
    sgn = jnp.sign(jnp.sum(n_hat * r_perp, axis=-1, keepdims=True))
    return jnp.where(sgn >= 0.0, n_hat, -n_hat)

def build_residual_fn(grad_phi_fn, surfaces, theta_grid, phi_grid, center, a_hat):
    """
    Returns a function r(all_coeffs) -> residual vector (1D).
    Residual entries are sqrt(weight)* (n·∇φ) at every (k, i, j).
    """
    TH, PH = theta_grid, phi_grid
    W = jnp.ones_like(TH)
    W = W / jnp.maximum(EPS, jnp.mean(W))   # normalized weights

    # slice view over the flat parameter vector
    idxs = []
    off = 0
    for s in surfaces:
        n = s.M_R + s.M_Z
        idxs.append((off, off+n))
        off += n
    K = len(surfaces)

    def residual(all_coeffs):
        res = []
        for k,(a,b) in enumerate(idxs):
            c = all_coeffs[a:b]
            pts = surfaces[k].cartesian(c, TH, PH)
            _, _, n_hat, area, _, _ = surfaces[k].tangents_normal_area(c, TH, PH)
            n_hat = _outward_normalize(n_hat, pts, center, a_hat)

            G = grad_phi_fn(pts.reshape(-1,3)).reshape(pts.shape)
            ndot = jnp.sum(n_hat * G, axis=-1)            # (Nt, Np)
            # jax.debug.print("[residual] S{}  ⟨|n|⟩={:.3e}  ⟨|G|⟩={:.3e}  ⟨n·G⟩={:.3e}",
            #     k, jnp.mean(safe_norm(n_hat,axis=-1)), jnp.mean(safe_norm(G,axis=-1)), jnp.mean(jnp.abs(ndot)))
            n_bad = jnp.sum(area <= 1e-6)  # floor in safe_norm
            # jax.debug.print("[residual] S{} area_floor_hits={}", k, n_bad)
            # quadrature weight: sqrt(area * W) makes Gauss–Newton well-scaled
            w = jnp.sqrt(jnp.maximum(EPS, area) * W)
            res.append((w * ndot).reshape(-1))
        r = jnp.concatenate(res, axis=0)
        # jax.debug.print("[residual] size={}  mean|r|={:.3e}  max|r|={:.3e}",
        #                 r.size, jnp.mean(jnp.abs(r)), jnp.max(jnp.abs(r)))
        return r

    return residual

# ---------------- Levenberg–Marquardt (trust region) ------------------ #

def levenberg_marquardt(residual_fn, x0, ridge=1e-8, mu0=1e-2, mu_max=1e6,
                        max_iter=60, tol_g=1e-6, tol_dx=1e-9, verbose=True):
    x = x0
    r = residual_fn(x)
    f = 0.5 * (r @ r)
    mu = mu0

    for it in range(1, max_iter+1):
        # Build linear ops ONCE at current x
        r, jvp_lin = jax.linearize(residual_fn, x)   # r = residual(x), jvp_lin(v) = J v
        vjp_fun = jax.vjp(residual_fn, x)[1]         # JT op

        g, = vjp_fun(r)                               # g = J^T r
        gnorm = float(jnp.linalg.norm(g))
        if verbose:
            print(f"[LM] it={it:02d}  f={float(0.5*(r@r)):.6e}  ||g||={gnorm:.3e}  mu={mu:.2e}")
        if gnorm < tol_g:
            break

        damp = (mu + ridge)
        def H_mv(v):
            Jv = jvp_lin(v)           # fast Jv
            JTJv, = vjp_fun(Jv)       # fast J^T(Jv)
            return JTJv + damp * v

        delta = cg_solve(H_mv, -g, tol=1e-6, maxit=100)
        dnorm = float(jnp.linalg.norm(delta))
        if dnorm < tol_dx:
            break

        pred = -0.5 * (delta @ (g + (mu + ridge)*delta))

        x_trial = x + delta
        r_trial = residual_fn(x_trial)
        f_trial = 0.5 * (r_trial @ r_trial)
        ared = float((0.5*(r@r)) - f_trial)
        rho = ared / float(pred + 1e-16)

        if verbose:
            print(f"     Δ={dnorm:.3e}  f_try={float(f_trial):.6e}  ρ={rho:.3f}")

        if (ared > 0.0) and jnp.isfinite(f_trial):
            x, r = x_trial, r_trial
            mu = max(mu * 0.3, 1e-12) if rho > 0.75 else mu
        else:
            mu = min(mu * 3.0, mu_max)

    return x

def gn_matvec(residual_fn, x, v):
    """
    Computes (J^T J) v using one JVP and one VJP without forming J.
    """
    # r(x)
    r = residual_fn(x)
    # JVP: J v
    _, jvp = jax.jvp(residual_fn, (x,), (v,))
    # VJP: J^T (J v)
    vjp_fun = jax.vjp(residual_fn, x)[1]
    JTjv, = vjp_fun(jvp)
    return JTjv

def cg_solve(matvec, b, tol=1e-8, maxit=200):
    x = jnp.zeros_like(b)
    r = b - matvec(x)
    p = r
    rsold = jnp.dot(r, r)

    def body_fun(state):
        x, r, p, rsold, k = state
        Ap = matvec(p)
        alpha = rsold / jnp.maximum(EPS, jnp.dot(p, Ap))
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rsnew = jnp.dot(r_new, r_new)
        beta = rsnew / jnp.maximum(EPS, rsold)
        p_new = r_new + beta * p
        return (x_new, r_new, p_new, rsnew, k+1)

    def cond_fun(state):
        _, r, _, rs, k = state
        return jnp.logical_and(jnp.sqrt(rs) > tol, k < maxit)

    x, r, p, rs, k = jax.lax.while_loop(cond_fun, body_fun, (x, r, p, rsold, 0))
    return x

# --------------- Projection onto feasible set (no penalties) --------- #

def project_feasible(all_coeffs, surfaces, TH, PH, center, a_hat,
                     Rmin_of_phi, Rmax_of_phi, eps=1e-3,
                     preserve=None, targets=None):
    """
    preserve: None | "area" | "volume"
    targets:  list of target scalars per surface (A* or V*), computed from seed

    1) Envelope: for each surface, if any sample R is outside [Rmin-eps, Rmax+eps],
       uniformly scale all (m>=1) R and Z coefficients to fit.
    2) Nesting: enforce increasing mean axis-distance across surfaces.
    """
    idxs = []
    off = 0
    for s in surfaces:
        n = s.M_R + s.M_Z
        idxs.append((off, off+n))
        off += n

    x = all_coeffs

    # --- (1) Envelope per surface
    for k,(a,b) in enumerate(idxs):
        s = surfaces[k]
        c = x[a:b]
        cR, cZ = c[:s.M_R], c[s.M_R:]

        R, _ = s.evaluate_RZ(c, TH, PH)
        R0 = jnp.full_like(R, s.R0)
        R1 = R - R0

        phi_line = PH[0,:]
        Rmin = Rmin_of_phi(phi_line)[None,:]
        Rmax = Rmax_of_phi(phi_line)[None,:]

        num_hi = (Rmax - eps) - R0
        num_lo = (Rmin + eps) - R0

        # Per-sample bounds
        g_hi = jnp.where(R1 > 0, num_hi / jnp.maximum(EPS, R1), jnp.inf)
        g_lo = jnp.where(R1 < 0, num_lo / jnp.minimum(-EPS, R1), -jnp.inf)

        # Global bound over all samples
        g_hi_all = jnp.min(g_hi)
        g_lo_all = jnp.max(g_lo)

        # Current g=1; if infeasible, pull back to nearest feasible point
        g_need = jnp.where(g_lo_all <= 1.0, jnp.where(1.0 <= g_hi_all, 1.0, g_hi_all), g_lo_all)
        g_best = jnp.clip(g_need, 0.05, 1.0)

        # Apply only if there is a violation
        viol = jnp.logical_or(g_best < 1.0 - 1e-12, g_best > 1.0 + 1e-12)
        c_new = jax.lax.select(
            viol,
            jnp.concatenate([c[:s.M_R] * g_best, c[s.M_R:] * g_best]),
            c
        )
        x = x.at[a:b].set(c_new)

    # --- (2) Nesting (monotone mean axis distance)
    means = []
    for k,(a,b) in enumerate(idxs):
        s = surfaces[k]
        c = x[a:b]
        P = s.cartesian(c, TH, PH)
        a_unit = a_hat / jnp.maximum(EPS, jnp.linalg.norm(a_hat))
        v = P - center[None,None,:]
        v_par = jnp.sum(v * a_unit[None,None,:], axis=-1, keepdims=True) * a_unit[None,None,:]
        rho = safe_norm(v - v_par, axis=-1)
        means.append(jnp.mean(rho))
    means = jnp.array(means)

    # --- (3) Conservation: match area or volume to targets via 1D rescale
    if preserve in ("area", "volume") and (targets is not None):
        idxs2 = []
        off = 0
        for s in surfaces:
            n = s.M_R + s.M_Z
            idxs2.append((off, off+n))
            off += n
        for k,(a,b) in enumerate(idxs2):
            s = surfaces[k]
            c = x[a:b]
            tgt = targets[k]
            which = "area" if preserve == "area" else "volume"
            g = find_scale_to_match(s, c, TH, PH, tgt, which=which)
            x = x.at[a:b].set(scale_nonzero_modes(c, s, g))

    # isotonic: force increasing order by cumulative max
    means_sorted_idx = jnp.argsort(means)
    means_sorted = means[means_sorted_idx]
    iso = jnp.maximum.accumulate(means_sorted + 1e-6*jnp.arange(means_sorted.size))
    # If any pair violated, softly expand inner surfaces (scale all modes up a bit)
    scale_map = jnp.ones_like(means)
    scale_map = scale_map.at[means_sorted_idx].set(iso / jnp.maximum(EPS, means_sorted))
    # apply small scaling (cap to avoid exploding)
    scale_map = jnp.clip(scale_map, 1.0, 1.1)

    for k,(a,b) in enumerate(idxs):
        s = surfaces[k]
        c = x[a:b]
        cR, cZ = c[:s.M_R], c[s.M_R:]
        g = scale_map[k]
        cR = cR * g; cZ = cZ * g
        x = x.at[a:b].set(jnp.concatenate([cR, cZ]))
    return x

# ---------------------- Pipeline + Plotting --------------------------- #

def build_initials(sol, num_surfaces, modes, base_radius, base_height):
    # Robust φ-envelope from boundary
    nfit = max(1, max(n for _, n in modes.R_modes))
    R0_of_phi, Rmin_of_phi, Rmax_of_phi = boundary_phi_stats(sol, nphi_bins=512, nfit=nfit)

    # Tightest allowed minor radius (everywhere inside)
    phi_probe = jnp.linspace(-jnp.pi, jnp.pi, 2048, endpoint=False)
    a1_phi   = Rmax_of_phi(phi_probe) - R0_of_phi(phi_probe)             # local minor radius vs φ
    a1_min   = float(jnp.nanmin(a1_phi))                                 # tightest φ
    a1_phi   = jnp.maximum(a1_phi, 1e-9)                                 # guard

    # Fit R0(φ)
    phi_fit = jnp.linspace(-jnp.pi, jnp.pi, 2048, endpoint=False)
    y_R0    = R0_of_phi(phi_fit)
    # only use (m=0,n>0) columns in R for wobble; map after solving
    idx_0n  = [i for i,(m,n) in enumerate(modes.R_modes) if (m==0 and n>0)]
    if len(idx_0n) > 0:
        B0 = []
        for i in idx_0n:
            n = modes.R_modes[i][1]
            B0.append(jnp.cos(n*phi_fit))
        B0  = jnp.stack(B0, axis=1)                                      # (Nφ, ncols)
        y0c = y_R0 - jnp.mean(y_R0)                                      # remove mean → mean goes to R0
        c_0n = jnp.linalg.solve(B0.T@B0 + 1e-10*jnp.eye(B0.shape[1]), B0.T@y0c)
        c_0n = 0.95 * c_0n                                               # shrink
    else:
        c_0n = jnp.zeros((0,), dtype=jnp.float64)

    R0_mean = float(jnp.mean(y_R0))                                      # baseline major radius

    # Fit a1(φ) with full [1, cos nφ, sin nφ] basis (this creates (1,0) and (1,n))
    c_a1, a1_eval = fourier_fit_phi(a1_phi, phi_probe, nfit=nfit)

    # Construct per-surface scale factors (strictly inside)
    # s_k ranges (0.1 .. 0.95) for K surfaces; adjust as you like
    eta = 0.92                                  # safety margin to stay inside
    s_factors = [eta*np.sqrt((k+1)/(num_surfaces+1)) for k in range(num_surfaces)]

    # Indices of (m=1,n) in R and Z; we will set identical amplitudes there
    idx_1n_R = {n:i for i,(m,n) in enumerate(modes.R_modes) if m==1}
    idx_1n_Z = {n:i for i,(m,n) in enumerate(modes.Z_modes) if m==1}

    # Helper to read coefficients from c_a1
    # Layout of c_a1 is [c0, cc1, cs1, cc2, cs2, ...]
    def a1_coeff(n):
        if n == 0:
            return float(c_a1[0])
        j = 1 + 2*(n-1)
        cc = float(c_a1[j])
        cs = float(c_a1[j+1])
        return (cc, cs)  # for n>0 we have a cosine and a sine part

    surfaces, coeffs_blocks = [], []

    for k in range(num_surfaces):
        surf = FourierSurface(modes, R0=R0_mean)
        cR = jnp.zeros((surf.M_R,), dtype=jnp.float64)
        cZ = jnp.zeros((surf.M_Z,), dtype=jnp.float64)

        # (i) Put the R0(φ) wobble into the (0,n) slots in R
        for j,i in enumerate(idx_0n):
            cR = cR.at[i].set(float(c_0n[j]))

        # (ii) Put the φ-dependent minor-radius into (1,n) in BOTH R and Z
        #      so that cross-sections are circles whose radius = s_k * a1(φ)
        s = float(s_factors[k])

        # n = 0 term → (1,0)
        if 0 in idx_1n_R and 0 in idx_1n_Z:
            a10 = s * a1_coeff(0)                                        # scalar
            cR = cR.at[idx_1n_R[0]].set(a10)
            cZ = cZ.at[idx_1n_Z[0]].set(a10)

        # n >= 1 terms → (1,n): R uses cos(mθ-nφ), Z uses sin(mθ-nφ)
        # The same amplitude pair (cc, cs) must be placed consistently so that
        # radius(φ) = s*a1(φ) = a10 + Σ_n [ cc_n cos(nφ) + cs_n sin(nφ) ].
        for n in range(1, nfit+1):
            if (n in idx_1n_R) and (n in idx_1n_Z):
                cc, cs = a1_coeff(n)
                cc *= s; cs *= s
                # For R: a_R(1,n) multiplies cos(θ - nφ) = cosθ cos(nφ) + sinθ sin(nφ).
                # For Z: a_Z(1,n) multiplies sin(θ - nφ) = sinθ cos(nφ) - cosθ sin(nφ).
                # A circular cross-section with radius depending on φ is obtained by using
                # the SAME (cc, cs) in the (1,n) of R and Z (signs as below).
                # We can encode that by two independent modes (1,n) in R and Z:
                # put 'cc' into the single (1,n) slot of each; the φ dependence
                # emerges from the -nφ phase inside cos/sin(θ-nφ).
                # To carry both cc and cs with a single slot, we shift phase:
                #   cos(θ - nφ - δ) = cos(θ - nφ) cosδ + sin(θ - nφ) sinδ
                # Represent cc,cs as amplitude A and phase δ:
                A   = np.hypot(cc, cs)
                delta = np.arctan2(cs, cc)  # so that A*cos(θ - nφ - δ)
                # Implement the phase shift by splitting across R and Z:
                # cos(θ-nφ-δ) = cos(θ-nφ)cosδ + sin(θ-nφ)sinδ
                # sin(θ-nφ-δ) = sin(θ-nφ)cosδ - cos(θ-nφ)sinδ
                # Thus set:
                aRn =  A*np.cos(delta)      # multiplies cos(θ-nφ) in R
                aZn =  A*np.cos(delta)      # multiplies sin(θ-nφ) in Z
                # and add the quadrature part via the next equal-amplitude trick:
                aR_quadr =  A*np.sin(delta) # goes with sin(θ-nφ) piece in R
                aZ_quadr = -A*np.sin(delta) # goes with cos(θ-nφ) piece in Z
                #
                # Our basis has only one coefficient per (1,n) in each of R or Z
                # (attached to cos or sin respectively). To include the quadrature,
                # we borrow (m=1,n) again by using (m=1,n) in the *other* field:
                # -> In R we cannot directly add sin(θ-nφ). But an equal-amplitude
                #    contribution in Z already supplies the needed quadrature to
                #    keep circles. The simplest robust seed is to collapse to A:
                aRn = A
                aZn = A
                cR = cR.at[idx_1n_R[n]].set(float(aRn))
                cZ = cZ.at[idx_1n_Z[n]].set(float(aZn))

        surfaces.append(surf)
        coeffs_blocks.append(jnp.concatenate([cR, cZ], axis=0))

    x0 = jnp.concatenate(coeffs_blocks, axis=0)
    print(f"[SEED] a1_min (tightest) ≈ {a1_min:.4f}; R0_mean ≈ {R0_mean:.6f}")
    return surfaces, x0

def plot_poincare_multi(surfaces, coeffs_blocks, sol, nfp=2, nphi=4,
                        ntheta=360, title="Global flux-surface fit", show=False):
    # cuts: 0 .. 2π/nfp
    phi_list = jnp.linspace(0.0, 2.0*jnp.pi/nfp, nphi, endpoint=False)
    theta = jnp.linspace(0.0, 2.0*jnp.pi, ntheta)

    fig, ax = plt.subplots(figsize=(7,7))
    # pick one color per surface
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7'])
    colors = [colors[i % len(colors)] for i in range(len(surfaces))]

    for phi_cross in np.asarray(phi_list):
        phi = jnp.full_like(theta, float(phi_cross))
        for k, surf in enumerate(surfaces):
            c = coeffs_blocks[k]
            Rk, Zk = surf.evaluate_RZ(c, theta, phi)
            col = colors[k]
            ax.plot(np.asarray(Rk), np.asarray(Zk), lw=1.5, color=col)
            ax.plot(np.asarray(Rk)[::10], np.asarray(Zk)[::10], '.', ms=2.2, alpha=0.8, color=col)

    ax.set_xlabel("R"); ax.set_ylabel("Z"); ax.set_aspect('equal','box')
    ax.set_title(title + f"  (cuts: {nphi} in [0, 2π/{nfp}])")

    # symmetry axis direction + boundary slice near first cut (as before)
    center = np.asarray(sol["center"])
    a_hat  = np.asarray(sol["a_hat"], dtype=float)
    a_hat  = a_hat / np.linalg.norm(a_hat)
    t = np.linspace(-2.0, 2.0, 5)
    axis_pts = center[None,:] + t[:,None] * a_hat[None,:]
    ax.plot(np.sqrt(axis_pts[:,0]**2 + axis_pts[:,1]**2), axis_pts[:,2], 'k-', lw=1.2, alpha=0.6, label="symmetry axis (dir)")

    # === plot boundary at each φ in phi_list ===
    # --- robust boundary cuts: take K nearest-by-angle, then project exactly ---
    P_world = np.asarray(sol["P"])
    Zb      = P_world[:, 2]
    Rb      = np.sqrt(P_world[:,0]**2 + P_world[:,1]**2)
    Phib    = np.arctan2(P_world[:,1], P_world[:,0])

    def eR_ephi(phi0):
        # eR = (cosφ, sinφ, 0), eφ = (-sinφ, cosφ, 0)
        c, s = np.cos(phi0), np.sin(phi0)
        eR   = np.array([ c,  s, 0.0])
        ephi = np.array([-s,  c, 0.0])
        return eR, ephi

    K_min = max(50, len(Phib)//200)   # ~0.5% of points, at least 50 (tune as you like)

    first = True
    for phi_cross in np.asarray(phi_list):
        # angular distance on the circle
        dphi = ( (Phib - float(phi_cross) + np.pi) % (2*np.pi) ) - np.pi
        keep = np.argsort(np.abs(dphi))[:K_min]      # K nearest in φ

        # project exactly to the φ=const plane: subtract (n·r) n with n = eφ
        eR, ephi = eR_ephi(phi_cross)
        dist = P_world[keep] @ ephi                  # signed distance to plane
        r_proj = P_world[keep] - dist[:,None] * ephi[None,:]

        Rcut = r_proj @ eR                           # in-plane "R"
        Zcut = r_proj[:, 2]                          # in-plane "Z" (= z)
        ax.plot(Rcut, Zcut, '.', ms=3.0, alpha=1.0, color='k',
                label=("boundary@φ cuts" if first else None))
        first = False

    # auto-limits from all surfaces (robust)
    Rs_all, Zs_all = [], []
    for phi_cross in np.asarray(phi_list):
        phi = jnp.full_like(theta, float(phi_cross))
        for k, surf in enumerate(surfaces):
            c = coeffs_blocks[k]
            Rk, Zk = surf.evaluate_RZ(c, theta, phi)
            Rs_all.append(np.asarray(Rk)); Zs_all.append(np.asarray(Zk))


    R_union = np.concatenate(Rs_all + [Rb])
    Z_union = np.concatenate(Zs_all + [Zb])
    r_lo, r_hi = np.percentile(R_union, [1, 99])
    z_lo, z_hi = np.percentile(Z_union, [1, 99])
    pad_r = 0.07*(r_hi - r_lo + 1e-6)
    pad_z = 0.07*(z_hi - z_lo + 1e-6)
    ax.set_xlim(r_lo - pad_r, r_hi + pad_r)
    ax.set_ylim(z_lo - pad_z, z_hi + pad_z)
    ax.legend(); plt.tight_layout()
    
    if show:
        plt.show()
        
def plot_3d_surfaces(surfaces, coeffs_blocks, sol, ntheta=32, nphi=128, alpha=0.4, show=False):
    # 3D plot of surfaces + boundary
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each surface
    theta_3d = jnp.linspace(0.0, 2.0*jnp.pi, ntheta,)
    phi_3d   = jnp.linspace(0.0, 2.0*jnp.pi, nphi,)
    TH_3d, PH_3d = jnp.meshgrid(theta_3d, phi_3d, indexing="ij")
    
    colors_3d = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7'])
    
    for k, surf in enumerate(surfaces):
        c = coeffs_blocks[k]
        pts_3d = surf.cartesian(c, TH_3d, PH_3d)
        X = np.asarray(pts_3d[..., 0])
        Y = np.asarray(pts_3d[..., 1])
        Z = np.asarray(pts_3d[..., 2])
        
        col = colors_3d[k % len(colors_3d)]
        ax.plot_surface(X, Y, Z, alpha=alpha, color=col, edgecolor='none')
        ax.plot_wireframe(X, Y, Z, alpha=alpha, color=col, linewidth=0.5, rstride=4, cstride=8)
    
    # Plot boundary points
    P_boundary = np.asarray(sol["P"])
    ax.scatter(P_boundary[:, 0], P_boundary[:, 1], P_boundary[:, 2], 
                c='black', s=1, alpha=0.5, label='Boundary')
    
    # Plot symmetry axis
    center = np.asarray(sol["center"])
    a_hat  = np.asarray(sol["a_hat"], dtype=float)
    a_hat  = a_hat / np.linalg.norm(a_hat)
    zmin = np.min(P_boundary[:, 2])
    zmax = np.max(P_boundary[:, 2])
    a_hat_z = a_hat[2]
    if abs(a_hat_z) > 1e-12:
        t_min = (zmin - center[2]) / a_hat_z
        t_max = (zmax - center[2]) / a_hat_z
    else:
        t_min, t_max = -0.5*(zmax - zmin), 0.5*(zmax - zmin)
    t_axis = np.linspace(t_min, t_max, 20)
    axis_pts = center[None,:] + t_axis[:,None] * a_hat[None,:]
    ax.plot(axis_pts[:,0], axis_pts[:,1], axis_pts[:,2], 
            'r-', linewidth=2, label='symmetry axis')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Flux Surfaces with Boundary')
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    
    if show:
        plt.show()

# ------------------------------ Main ---------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", type=str, default="wout_precise_QA_solution.npz")
    ap.add_argument("--num-surfaces", type=int, default=2)
    ap.add_argument("--mmax", type=int, default=4)
    ap.add_argument("--nmax", type=int, default=4)
    ap.add_argument("--grid-theta", type=int, default=8)
    ap.add_argument("--grid-phi", type=int, default=26)
    ap.add_argument("--max-iter", type=int, default=2)
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--nfp", type=int, default=2)
    ap.add_argument("--poincare-nphi", type=int, default=4)
    ap.add_argument("--preserve_kind", type=str, default="volume",
                    choices=["area", "volume", "none"])
    args = ap.parse_args()

    sol = load_mfs_solution(args.solution)
    grad_phi_fn = build_grad_phi_fn(sol)

    # Base scales from boundary:
    Rb = jnp.sqrt(sol["P"][:,0]**2 + sol["P"][:,1]**2)
    base_radius = float(jnp.mean(Rb))
    base_height = float(jnp.mean(jnp.abs(sol["P"][:,2])))
    if args.verbose:
        print(f"[LOAD] base_radius≈{base_radius:.3f}, base_height≈{base_height:.3f}")

    modes = FourierModes.generate(args.mmax, args.nmax)
    surfaces, x0 = build_initials(sol, args.num_surfaces, modes, base_radius, base_height)
    print(f"[SEED] R0 (major radius) = {surfaces[0].R0:.6f}")

    # Quick geometry sanity check (seed):
    def _geom_stats(xvec, tag):
        idx = 0
        for si, s in enumerate(surfaces):
            n = s.M_R + s.M_Z
            c = xvec[idx:idx+n]; idx += n
            # coarse grid ok for stats
            th = jnp.linspace(0, 2*jnp.pi, 16, endpoint=False)
            ph = jnp.linspace(0, 2*jnp.pi, 16, endpoint=False)
            THs, PHs = jnp.meshgrid(th, ph, indexing="ij")
            P = s.cartesian(c, THs, PHs)
            R = jnp.sqrt(P[...,0]**2 + P[...,1]**2)
            Z = P[...,2]
            print(f"[geom:{tag}] S{si}  R[min,med,max]=[{float(R.min()):.3f}, {float(jnp.median(R)):.3f}, {float(R.max()):.3f}]  "
                  f"Z[min,med,max]=[{float(Z.min()):.3f}, {float(jnp.median(Z)):.3f}, {float(Z.max()):.3f}]")
    _geom_stats(x0, "seed")

    # Angle grids
    theta = jnp.linspace(0.0, 2.0*jnp.pi, args.grid_theta, endpoint=False)
    phi   = jnp.linspace(0.0, 2.0*jnp.pi, args.grid_phi,   endpoint=False)
    TH, PH = jnp.meshgrid(theta, phi, indexing="ij")

    # ---- Ensure initial ordering by mean distance to axis
    def mean_axis_dist_of_block(coeff_block, surf):
        pts0 = surf.cartesian(coeff_block, TH, PH)
        a_unit = sol["a_hat"] / float(np.linalg.norm(np.asarray(sol["a_hat"])))
        v = pts0 - sol["center"][None, None, :]
        v_par  = jnp.sum(v * a_unit[None, None, :], axis=-1, keepdims=True) * a_unit[None, None, :]
        r_perp = v - v_par
        return float(jnp.mean(safe_norm(r_perp, axis=-1)))

    blocks = []
    idx = 0
    for s in surfaces:
        n = s.M_R + s.M_Z
        blocks.append((s, x0[idx:idx+n], idx, idx+n))
        idx += n
    blocks_sorted = sorted(blocks, key=lambda t: mean_axis_dist_of_block(t[1], t[0]))
    surfaces = [t[0] for t in blocks_sorted]
    x0 = jnp.concatenate([t[1] for t in blocks_sorted], axis=0)

    # Targets from seed (same grid as residual)
    targets_area  = []
    targets_vol   = []
    idx = 0
    for s in surfaces:
        n = s.M_R + s.M_Z
        c = x0[idx:idx+n]; idx += n
        A0, V0 = surface_area_and_volume(s, c, TH, PH)
        targets_area.append(float(A0))
        targets_vol.append(float(V0))

    # Build residual and run LM with feasibility projection at each accepted step
    residual_fn = build_residual_fn(grad_phi_fn, surfaces, TH, PH, sol["center"], sol["a_hat"])

    # Precompute robust φ-envelope from boundary for projection
    R0_of_phi, Rmin_of_phi, Rmax_of_phi = boundary_phi_stats(sol, nphi_bins=512)

    preserve_kind = args.preserve_kind  # or "area" or None

    targets = targets_vol if preserve_kind == "volume" else (
            targets_area if preserve_kind == "area" else None)

    def residual_with_proj(z):
        z_proj = project_feasible(
            z, surfaces, TH, PH, sol["center"], sol["a_hat"],
            Rmin_of_phi, Rmax_of_phi, eps=1e-3,
            preserve=preserve_kind, targets=targets)
        return residual_fn(z_proj)
    
    print("[LM] starting...")
    residual_with_proj = jax.jit(residual_with_proj)
    x_opt = levenberg_marquardt(residual_with_proj, x0, ridge=1e-8, mu0=1e-2,
                                max_iter=args.max_iter, tol_g=1e-7, tol_dx=1e-9, verbose=True)
    # final projection (safety)
    x_opt = project_feasible(
        x_opt, surfaces, TH, PH, sol["center"], sol["a_hat"],
        Rmin_of_phi, Rmax_of_phi, eps=1e-3,
        preserve=preserve_kind, targets=targets
    )

    _geom_stats(x_opt, "opt")

    # Split back
    coeffs_blocks = []
    idx = 0
    for s in surfaces:
        n = s.M_R + s.M_Z
        coeffs_blocks.append(x_opt[idx:idx+n]); idx += n

    print("[PLOT] Seed Poincaré...")
    coeffs_blocks_seed = []; idx = 0
    for s in surfaces: coeffs_blocks_seed.append(x0[idx:idx+s.M_R + s.M_Z]); idx += s.M_R + s.M_Z
    plot_poincare_multi(
        surfaces, coeffs_blocks_seed, sol,
        nfp=args.nfp, nphi=args.poincare_nphi, ntheta=100,
        title="Global flux-surface fit (SEED)", show=False)
    print("[PLOT] Poincaré.")
    plot_poincare_multi(surfaces, coeffs_blocks, sol,
                        nfp=args.nfp, nphi=args.poincare_nphi, ntheta=100,
                        title="Global flux-surface fit", show=False)
    print("[PLOT] 3D surfaces.")
    plot_3d_surfaces(surfaces, coeffs_blocks, sol, show=False)
    print("[DONE]")
    plt.show()

if __name__ == "__main__":
    main()
