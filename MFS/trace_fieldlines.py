#!/usr/bin/env python3
"""
Trace field lines x'(t) = ∇φ(x) from an MFS solution checkpoint (.npz).

Usage:
  python trace_fieldlines.py wout_precise_QH_solution.npz --nfp 4 --save-figure
  python trace_fieldlines.py wout_precise_QA_solution.npz --nfp 2 --save-figure
## The one below for SLAM may take a few minutes to run for that big tfinal
  python trace_fieldlines.py slam_surface_solution.npz --nfp 2 --save-figure --seeds "3.3:0:0,3.25:0:0,3.2:0:0,3.15:0:0,3.1:0:0,3.05:0:0,3.0:0:0" --tfinal 13500

Script is parallelized over multiple devices (set number_of_processors_to_use at top).

The .npz must contain (as saved by main.py):
  center(3,), scale(scalar), Yn(M,3), alpha(M,),
  a(2,), a_hat(3,), P(N,3), N(N,3), kind("torus"/"mirror")
"""

from __future__ import annotations

import os
number_of_processors_to_use = 8 # Parallelization, this should divide nfieldlines
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'

import time, argparse
from fractions import Fraction
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax as dfx

from jax import jit, vmap, tree_util, random, lax, device_put
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

mesh = Mesh(jax.devices(), ("dev",))
spec=PartitionSpec("dev", None)
spec_index=PartitionSpec("dev")
sharding = NamedSharding(mesh, spec)
sharding_index = NamedSharding(mesh, spec_index)
out_sharding = NamedSharding(mesh, PartitionSpec("dev", None, None))

# -------------------------- Paths and utils ------------------------- #
script_dir = Path(__file__).resolve().parent

def resolve_npz_file_location(npz_file, subdir="outputs"):
    try:
        npz_name = os.path.basename(str(npz_file))
        candidate = (script_dir / ".." / subdir / npz_name).resolve()
        if candidate.exists():
            npz_file = str(candidate)
            print(f"Resolved checkpoint path -> {npz_file}")
        else:
            print(f"[WARN] Expected checkpoint not found at {candidate}; using provided path: {npz_file}")
    except Exception as e:
        print(f"[WARN] Failed to resolve ../{subdir} path: {e}; using provided path: {npz_file}")
    return npz_file

# ----------------------------- Styling ----------------------------- #
def fix_matplotlib_3d(ax):
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1]-x_limits[0]); x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1]-y_limits[0]); y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1]-z_limits[0]); z_mid = np.mean(z_limits)
    R = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid-R, x_mid+R]); ax.set_ylim3d([y_mid-R, y_mid+R]); ax.set_zlim3d([z_mid-R, z_mid+R])

def apply_paper_style():
    fig_w = 5.5
    mpl.rcParams.update({"figure.figsize": (fig_w, fig_w),
        "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.linewidth": 0.9, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 4, "ytick.major.size": 4, "xtick.minor.size": 2.5, "ytick.minor.size": 2.5, "text.usetex": False})
    return fig_w

def set_equal_data_aspect(ax, rmin, rmax, zmin, zmax, pad_frac=0.03):
    def _pad(lo, hi, frac):
        span = hi - lo
        if span <= 0:
            span = max(1e-6, abs(hi) if abs(hi) > 0 else 1.0)
        pad = frac * span
        return lo - pad, hi + pad

    rlo, rhi = _pad(float(rmin), float(rmax), pad_frac)
    zlo, zhi = _pad(float(zmin), float(zmax), pad_frac)

    rc = 0.5 * (rlo + rhi)
    zc = 0.5 * (zlo + zhi)
    rspan = rhi - rlo
    zspan = zhi - zlo
    span = max(rspan, zspan)
    rlo, rhi = rc - 0.5*span, rc + 0.5*span
    zlo, zhi = zc - 0.5*span, zc + 0.5*span

    ax.set_xlim(rlo, rhi)
    ax.set_ylim(zlo, zhi)
    ax.set_aspect("equal", adjustable="box")

def _orthonormal_complement(a_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a_hat, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-30)
    t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = t - np.dot(t, a) * a
    e1 /= (np.linalg.norm(e1) + 1e-30)
    e2 = np.cross(a, e1)
    e2 /= (np.linalg.norm(e2) + 1e-30)
    return e1, e2

def phi_label_pi(phi: float, wrap=True, max_den=24) -> str:
    if wrap:
        phi = (phi + np.pi) % (2*np.pi) - np.pi
    r = Fraction(phi / np.pi).limit_denominator(max_den)
    p, q = r.numerator, r.denominator
    def _mul_pi(pp, qq):
        if pp == 0:
            return "0"
        sign = "-" if pp < 0 else ""
        pp = abs(pp)
        if qq == 1:
            coeff = "" if pp == 1 else f"{pp}"
            return f"{sign}{coeff}\\pi"
        else:
            coeff = "" if pp == 1 else f"{pp}"
            return f"{sign}{coeff}\\pi/{qq}"
    return rf"$\phi={_mul_pi(p, q)}$"

# ------------------------ MFS evaluators ------------------------- #
def _green_G(x, Y):  # x:(3,), Y:(M,3)
    r = jnp.linalg.norm(x[None,:] - Y, axis=1)
    return 1.0 / (4.0 * jnp.pi * jnp.maximum(1e-30, r))

def _grad_green_x(x, Y):  # -> (M,3)
    r = x[None,:] - Y
    r2 = jnp.sum(r*r, axis=1)
    r3 = jnp.maximum(1e-30, r2 * jnp.sqrt(r2))
    return - r / (4.0 * jnp.pi * r3)[:, None]

def _unit(v, eps=1e-30):
    n = jnp.linalg.norm(v, axis=1, keepdims=True)
    return v / jnp.maximum(eps, n)

def _nearest_normal_jax(Xn, Pn, Nn):
    X2 = jnp.sum(Xn*Xn, axis=1, keepdims=True)
    P2 = jnp.sum(Pn*Pn, axis=1, keepdims=True)
    dist2 = X2 + P2.T - 2.0 * (Xn @ Pn.T)
    idx = jnp.argmin(dist2, axis=1)
    return Nn[idx, :]

def _grad_azimuth_about_axis(Xn, a_hat):
    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par  = jnp.sum(Xn * a[None,:], axis=1, keepdims=True) * a[None,:]
    r_perp = Xn - r_par
    r2     = jnp.maximum(1e-30, jnp.sum(r_perp*r_perp, axis=1, keepdims=True))
    cr     = jnp.cross(a[None,:], r_perp)
    return cr / r2

def _make_mv_grads(a_hat, P, N, center, scale):
    Pn = (P - center[None,:]) * scale
    Nn = N
    a_hat = jnp.asarray(a_hat)

    def grad_t(Xn):   # ∇ϕ_a, accepts (3,) or (N,3)
        Xn = Xn.reshape((-1, 3))
        return _grad_azimuth_about_axis(Xn, a_hat)

    def grad_p(Xn):   # θ̂, accepts (3,) or (N,3)
        Xn = Xn.reshape((-1, 3))
        n    = _nearest_normal_jax(Xn, Pn, Nn)
        a    = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
        rpar = jnp.sum(Xn * a[None,:], axis=1, keepdims=True) * a[None,:]
        rper = Xn - rpar
        phi_hat  = _unit(jnp.cross(a[None,:], rper))
        phi_tan  = _unit(phi_hat - jnp.sum(phi_hat*n, axis=1, keepdims=True)*n)
        theta_hat = _unit(jnp.cross(n, phi_tan))
        return theta_hat

    return grad_t, grad_p

def load_mfs_solution(npz_path: str):
    d = np.load(npz_path, allow_pickle=False)
    center = jnp.asarray(d["center"], dtype=jnp.float64)    # (3,)
    scale  = jnp.asarray(d["scale"].item() if d["scale"].shape==() else float(d["scale"]), dtype=jnp.float64)
    Yn     = jnp.asarray(d["Yn"], dtype=jnp.float64)        # (M,3) normalized coords
    alpha  = jnp.asarray(d["alpha"], dtype=jnp.float64)     # (M,)
    a      = jnp.asarray(d["a"], dtype=jnp.float64)         # (2,)
    a_hat  = jnp.asarray(d["a_hat"], dtype=jnp.float64)     # (3,)
    P      = jnp.asarray(d["P"], dtype=jnp.float64)         # (N,3) world
    N      = jnp.asarray(d["N"], dtype=jnp.float64)         # (N,3) world
    kind   = str(d["kind"])

    grad_t, grad_p = _make_mv_grads(a_hat, P, N, center, scale)

    @jax.jit
    def psi_point_world(x):     # scalar ψ(x)
        xn = (x - center) * scale
        G  = _green_G(xn, Yn)
        return jnp.dot(G, alpha)

    @jax.jit
    def grad_psi_point_world(x):  # vector ∇ψ(x)
        xn = (x - center) * scale
        dG = _grad_green_x(xn, Yn)
        return scale * jnp.sum(dG * alpha[:,None], axis=0)

    @jax.jit
    def grad_mv_point_world(x):
        xn = (x - center) * scale
        xn_b = xn[None, :]                # add batch dim
        gt = grad_t(xn_b)[0]
        gp = grad_p(xn_b)[0]
        return scale * (a[0] * gt + a[1] * gp)

    # We trace along ∇φ = ∇ψ + ∇φ_mv
    @jax.jit
    def grad_point_fn(x: jnp.ndarray) -> jnp.ndarray:
        return grad_mv_point_world(x) + grad_psi_point_world(x)

    # For surface coloring (value not required, but handy)
    @jax.jit
    def u_fn(xs: jnp.ndarray) -> jnp.ndarray:
        xs = xs.reshape(-1, 3)
        G  = jax.vmap(lambda x: _green_G((x - center) * scale, Yn))(xs)
        vals = jax.vmap(psi_point_world)(xs)
        return vals.reshape((-1,))

    # Seeds by nudging boundary inward a bit
    def seeds_from_boundary(nseed: int = 25, eps: float = 1e-3) -> np.ndarray:
        Pb = np.asarray(P); Nb = np.asarray(N)
        Pi = Pb - eps * Nb
        if Pi.shape[0] > nseed:
            stride = max(1, Pi.shape[0] // nseed)
            Pi = Pi[::stride][:nseed]
        return Pi.astype(np.float64)

    return dict(
        u_fn=u_fn, grad_point_fn=grad_point_fn,
        seeds_from_boundary=seeds_from_boundary,
        P=np.asarray(P), N=np.asarray(N),
        kind=kind, a_hat=np.asarray(a_hat), center=np.asarray(center)
    )

def seeds_along_axis_from_boundary(
    P: np.ndarray,
    N: np.ndarray,
    center: np.ndarray,
    a_hat: np.ndarray,
    kind: str,
    nseed: int = 25,
    strip_tol_frac: float = 0.03,   # width of the “strip” around the axis-line (in e2)
    plane_tol_frac: float = 0.10,   # only for torus: keep |s| small along a_hat
    inward_frac: float = 0.02       # inward nudge as fraction of median neighbor spacing in strip
) -> np.ndarray:
    """
    Build seeds on the chord x = center + τ e1, τ ∈ [τ_min, τ_max], where {e1,e2} ⟂ a_hat.
    Endpoints are taken from boundary points in a thin strip around that chord.
    """
    P = np.asarray(P); N = np.asarray(N)
    c = np.asarray(center); a = np.asarray(a_hat)
    e1, e2 = _orthonormal_complement(a)

    X = P - c[None, :]
    u1 = X @ e1
    u2 = X @ e2
    s  = X @ (a / (np.linalg.norm(a)+1e-30))

    # Robust spans
    u2_span = np.percentile(np.abs(u2), 99.0) + 1e-12
    s_span  = np.percentile(np.abs(s),  99.0) + 1e-12

    # Strip selection: near the axis-line (small u2). For torus, also near midplane along a_hat.
    u2_tol = strip_tol_frac * u2_span
    if kind.lower() == "torus":
        s_tol = plane_tol_frac * s_span
        mask = (np.abs(u2) <= u2_tol) & (np.abs(s) <= s_tol)
    else:
        mask = (np.abs(u2) <= u2_tol)

    if not np.any(mask):
        # Fallback: take whole cloud along e1
        mask = np.ones_like(u1, dtype=bool)

    # Endpoints: min/max along e1 inside the strip
    u1_sel = u1[mask]
    idx = np.where(mask)[0]
    iL = idx[np.argmin(u1_sel)]
    iR = idx[np.argmax(u1_sel)]
    pL, nL = P[iL], N[iL]
    pR, nR = P[iR], N[iR]

    pL = (pL+pR)/2.01
    pR = pR*0.99

    # Estimate a data-driven spacing in the strip to set inward epsilon automatically
    # crude kNN via projection to 1D u1:
    if u1_sel.size >= 8:
        u1_sorted = np.sort(u1_sel)
        du = np.median(np.diff(u1_sorted))
        # translate 1D spacing to world using |e1|=1
        h_med = max(1e-6, float(du))
    else:
        # Fallback: use global cloud scale
        bb = np.max(P, axis=0) - np.min(P, axis=0)
        h_med = max(1e-6, 0.01 * float(np.linalg.norm(bb)))

    eps = inward_frac * h_med

    # Build nseed points equally spaced between the two boundary endpoints (world coords),
    # then nudge inward using the nearest boundary normal.
    τ = np.linspace(0.0, 1.0, max(2, nseed))
    chord = (1.0 - τ)[:, None] * pL[None, :] + τ[:, None] * pR[None, :]

    # nearest boundary normal per chord point (simple L2 argmin)
    # (kept numpy to remain host-side; tiny cost compared to integration)
    def _nearest(i):
        d2 = np.sum((P - chord[i])**2, axis=1)
        j = int(np.argmin(d2))
        return N[j]

    normals = np.stack([_nearest(i) for i in range(chord.shape[0])], axis=0)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-30)

    seeds = chord - eps * normals  # inward
    return seeds.astype(np.float64)


# ------------------------- RHS & integrators ------------------------- #
def make_rhs(grad_u_point: Callable[[jnp.ndarray], jnp.ndarray],
             *, clip_grad: Optional[float]=None, normalize: bool=False):
    @jax.jit
    def f(t, y, args):
        g = grad_u_point(y)
        if normalize:
            n = jnp.linalg.norm(g) + 1e-12
            g = g / n
        if (clip_grad is not None) and (clip_grad > 0):
            n = jnp.linalg.norm(g) + 1e-12
            g = jnp.where(n > clip_grad, g * (clip_grad / n), g)
        return g
    return f

@jax.jit
def _cum_and(mask_t):
    return jax.lax.associative_scan(lambda a, b: a & b, mask_t, axis=0)

@jax.jit
def _keep_entered(mask_t: jnp.ndarray) -> jnp.ndarray:
    def step(carry, m):
        started, alive = carry
        started_new = jnp.logical_or(started, m)
        alive_new   = jnp.where(started_new, jnp.logical_and(alive, m), True)
        keep        = jnp.logical_and(started_new, alive_new)
        return (started_new, alive_new), keep
    (_, _), keep_seq = lax.scan(step, (jnp.bool_(False), jnp.bool_(True)), mask_t)
    return keep_seq

def make_streamline_solver(f, ts, dt0_signed, n_save, rtol, atol):
    solver = dfx.Tsit5()  # or Dopri8, your choice
    stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)
    term = dfx.ODETerm(f)
    saveat = dfx.SaveAt(ts=ts)

    def _solve_one(y0):
        sol = dfx.diffeqsolve(
            term, solver, t0=0.0, t1=ts[-1], dt0=dt0_signed,
            y0=y0, stepsize_controller=stepsize_controller,
            max_steps=200_000, saveat=saveat
        )
        return sol.ys

    return jax.jit(
        jax.vmap(_solve_one),
        in_shardings=sharding,
        out_shardings=out_sharding,
    )

def integrate_streamlines_vmap(
    seeds: np.ndarray,
    f,
    t_final: float = 5.0,
    dt0: float = 1e-2,
    box: Tuple[float,float,float,float,float,float] = (-1.5,1.5,-1.5,1.5,-1.0,1.0),
    *,
    backward: bool = False,
    n_save: int = 2001,
    rtol: float = 1e-5,
    atol: float = 1e-7,
):
    dt0_signed = -abs(dt0) if backward else abs(dt0)
    t_final = -abs(t_final) if backward else abs(t_final)
    ts = jnp.linspace(0, t_final, int(n_save), dtype=jnp.float64)

    vmapped_solver = make_streamline_solver(f, ts, dt0_signed, n_save, rtol, atol)
    ys_all = vmapped_solver(device_put(seeds, sharding))

    x_min, x_max, y_min, y_max, z_min, z_max = box
    X, Y, Z = ys_all[..., 0], ys_all[..., 1], ys_all[..., 2]
    in_box = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max) & (Z >= z_min) & (Z <= z_max)

    def _cum_and(mask_t):
        return jax.lax.associative_scan(lambda a, b: a & b, mask_t, axis=0)

    def _keep_entered(mask_t: jnp.ndarray) -> jnp.ndarray:
        # Keep samples after the first time we enter the box, until the first exit.
        # State machine carried by scan: (started, alive)
        def step(carry, m):
            started, alive = carry              # both booleans
            started_new = jnp.logical_or(started, m)
            # once started, we stay "alive" only while m==True
            alive_new   = jnp.where(started_new, jnp.logical_and(alive, m), True)
            keep        = jnp.logical_and(started_new, alive_new)
            return (started_new, alive_new), keep

        (_, _), keep_seq = lax.scan(step, (jnp.bool_(False), jnp.bool_(True)), mask_t)
        return keep_seq

    if args.mask_mode == "none":
        keep_mask = jnp.ones_like(in_box)
    elif args.mask_mode == "instant":
        keep_mask = in_box
    elif args.mask_mode == "entered":
        keep_mask = jax.vmap(_keep_entered)(in_box)
    else:  # "strict"
        keep_mask = jax.vmap(_cum_and)(in_box)

    if args.mask_report and not backward:
        kept_per_line = jnp.sum(keep_mask, axis=1)
        inside0 = in_box[:, 0]
        print("[MASK] mode:", args.mask_mode, "lines:", int(keep_mask.shape[0]))
        print("[MASK] kept samples per line (min/median/max):",
            int(jnp.min(kept_per_line)), int(jnp.median(kept_per_line)), int(jnp.max(kept_per_line)))
        print("[MASK] seeds inside initial box:", int(jnp.sum(inside0)), "/", int(inside0.shape[0]))

    ys_all = jnp.where(keep_mask[..., None], ys_all, jnp.nan)

    return np.asarray(ts), np.asarray(ys_all)

# ------------------------- Poincaré machinery ------------------------- #
def _angle_wrap_jnp(a):
    return (a + jnp.pi) % (2*jnp.pi) - jnp.pi

def _wrap_diff_jnp(a_minus_b):
    return _angle_wrap_jnp(a_minus_b)

@jax.jit
def poincare_RZ_points_jax_dense(Y_all: jnp.ndarray, phi0: float):
    # Y_all: (S, T, 3)
    valid = ~jnp.any(jnp.isnan(Y_all), axis=-1)            # (S,T)
    X = Y_all[..., 0]; Y = Y_all[..., 1]; Z = Y_all[..., 2]
    phi = jnp.arctan2(Y, X)
    dphi = _wrap_diff_jnp(phi - phi0)
    s = jnp.sign(dphi)
    s = jnp.where(s == 0.0, 1.0, s)

    valid_seg = valid[..., :-1] & valid[..., 1:]
    changed   = (s[..., :-1] * s[..., 1:] < 0.0) & valid_seg

    p0 = Y_all[:, :-1, :]
    p1 = Y_all[:,  1:, :]
    d0 = dphi[:, :-1]
    d1 = dphi[:,  1:]
    t = jnp.clip(d0 / (d0 - d1), 0.0, 1.0)
    p = p0 + t[..., None] * (p1 - p0)

    R  = jnp.linalg.norm(p[..., :2], axis=-1)  # (S, T-1)
    Zc = p[..., 2]                             # (S, T-1)

    # Flatten everything
    S, Tm1 = R.shape
    R_flat    = R.reshape(-1)
    Z_flat    = Zc.reshape(-1)
    mask_flat = changed.reshape(-1)

    # Seed index for each candidate: 0..S-1 repeated along time
    seed_idx  = jnp.tile(jnp.arange(S)[:, None], (1, Tm1))  # (S, T-1)
    seed_flat = seed_idx.reshape(-1)

    return R_flat, Z_flat, mask_flat, seed_flat


def poincare_multi_phi_jax(Y_all: jnp.ndarray, phis: jnp.ndarray):
    R_flat, Z_flat, M_flat, seed_flat = jax.vmap(
        poincare_RZ_points_jax_dense, in_axes=(None, 0))(Y_all, phis)
    return R_flat, Z_flat, M_flat, seed_flat

# ------------------------------- Main ------------------------------- #
def main(mfs_npz: str,
         seeds: Optional[List[Tuple[float,float,float]]] = None,
         t_final=6.0,
         normalize=False,
         clip_grad=None,
         nseed: int = 25,
         eps: float = 1e-3,
         rtol: float = 1e-5,
         atol: float = 1e-7,
         n_save: int = 2001,
         box_pad: float = 0.10,
         poincare_phi: Optional[Sequence[float]] = None,
         poincare_label_pi: bool = False,
         save_figure: bool = False,
         args=None
         ):

    # Load MFS checkpoint & evaluators
    m = load_mfs_solution(mfs_npz)
    u_fn = m["u_fn"]
    grad_point_fn = m["grad_point_fn"]
    seeds_from_boundary = m["seeds_from_boundary"]
    P = m["P"]; N = m["N"]; kind = m["kind"]

    # RHS
    f = make_rhs(grad_point_fn, clip_grad=clip_grad, normalize=normalize)

    # Seeds
    if seeds is None:
        if args.seed_mode == "axis":
            seeds_arr = seeds_along_axis_from_boundary(
                P=P, N=N, center=m["center"], a_hat=m["a_hat"], kind=kind,
                nseed=nseed, strip_tol_frac=args.strip_tol_frac,
                plane_tol_frac=args.plane_tol_frac, inward_frac=args.inward_frac
            )
        else:
            seeds_arr = seeds_from_boundary(nseed=nseed, eps=eps)
        seeds = [tuple(x) for x in seeds_arr]
    seeds_arr = np.asarray(seeds, dtype=np.float64)

    print(f"[SEEDS] Using {seeds_arr.shape[0]} seed points at positions:\n{seeds_arr}")

    # Box from boundary point cloud with padding
    mins = P.min(axis=0); maxs = P.max(axis=0)
    if seeds is not None:
        mins = np.minimum(mins, np.min(seeds_arr, axis=0))
        maxs = np.maximum(maxs, np.max(seeds_arr, axis=0))
    pad = box_pad * float(np.linalg.norm(maxs - mins))
    x_min, x_max = float(mins[0]-pad), float(maxs[0]+pad)
    y_min, y_max = float(mins[1]-pad), float(maxs[1]+pad)
    z_min, z_max = float(mins[2]-pad), float(maxs[2]+pad)
    box = (x_min, x_max, y_min, y_max, z_min, z_max)

    # Quick sanity print:
    inside0 = (
        (seeds_arr[:,0] >= x_min) & (seeds_arr[:,0] <= x_max) &
        (seeds_arr[:,1] >= y_min) & (seeds_arr[:,1] <= y_max) &
        (seeds_arr[:,2] >= z_min) & (seeds_arr[:,2] <= z_max)
    )
    print(f"[DEBUG] Seeds inside initial box: {int(inside0.sum())}/{seeds_arr.shape[0]}")
    print(f"[BOX] Integration box: x[{x_min:.3f}, {x_max:.3f}], y[{y_min:.3f}, {y_max:.3f}], z[{z_min:.3f}, {z_max:.3f}]")

    # Integrate fwd/back in parallel
    print(f"[INTEGRATION] Starting integration with {seeds_arr.shape[0]} seeds, t_final={t_final}, n_save={n_save}")
    t0 = time.time()
    ts_f, Yf = integrate_streamlines_vmap(
        seeds_arr, f, t_final=t_final, box=box,
        backward=False, n_save=n_save, rtol=rtol, atol=atol
    )
    ts_b, Yb = integrate_streamlines_vmap(
        seeds_arr, f, t_final=t_final, box=box,
        backward=True,  n_save=n_save, rtol=rtol, atol=atol
    )
    # Yb = Yf
    Y = np.concatenate([np.flip(Yb, axis=1), Yf], axis=1)  # (S, 2*n_save, 3)
    print(f"[TIME] Total elapsed time: {time.time() - t0:.2f} s")

    # Poincaré (optional)
    if poincare_phi and len(poincare_phi) > 0:
        phis = jnp.asarray(poincare_phi, dtype=jnp.float64)

        # NEW: get flat arrays + seed indices
        R_flat, Z_flat, M_flat, seed_flat = poincare_multi_phi_jax(jnp.asarray(Y), phis)

        apply_paper_style()
        fig_p, ax_p = plt.subplots()
        any_points = False

        S = Y.shape[0]  # number of seeds / field lines
        cmap = mpl.colormaps.get_cmap("tab10")  # or "tab10" if S<=10

        # For axis limits later
        all_R = []
        all_Z = []

        # Loop over Poincaré planes
        for k, phi0 in enumerate(np.asarray(phis)):
            Rf = np.asarray(R_flat[k])
            Zf = np.asarray(Z_flat[k])
            Mf = np.asarray(M_flat[k])
            Sf = np.asarray(seed_flat[k])

            # Keep only valid intersections
            Rk = Rf[Mf]
            Zk = Zf[Mf]
            Sk = Sf[Mf]

            if Rk.size == 0:
                continue

            any_points = True
            all_R.append(Rk)
            all_Z.append(Zk)

            # Plot each seed with its own color
            for s in range(S):
                mask_s = (Sk == s)
                if not np.any(mask_s):
                    continue
                color = cmap(s)
                if poincare_label_pi:
                    # optional: only label once per seed for legend clarity
                    label = f"seed {s}" if k == 0 else None
                    ax_p.scatter(
                        Rk[mask_s], Zk[mask_s],
                        s=0.5, alpha=0.85, rasterized=True,
                        color=color, label=label,
                    )
                else:
                    ax_p.scatter(
                        Rk[mask_s], Zk[mask_s],
                        s=0.5, alpha=0.85, rasterized=True,
                        color=color,
                    )

        ax_p.set_xlabel(r"$R=\sqrt{x^2+y^2}$")
        ax_p.set_ylabel(r"$Z$")
        ax_p.set_title(r"Poincaré section(s): cylindrical $\phi$")

        if any_points:
            R_all = np.concatenate(all_R)
            Z_all = np.concatenate(all_Z)

            if args.poincare_tight:
                p_lo, p_hi = args.poincare_pct
                rlo = float(np.nanpercentile(R_all, p_lo))
                rhi = float(np.nanpercentile(R_all, p_hi))
                zlo = float(np.nanpercentile(Z_all, p_lo))
                zhi = float(np.nanpercentile(Z_all, p_hi))

                # pad a bit
                def _pad(lo, hi, frac):
                    span = max(hi - lo, 1e-12)
                    pad  = frac * span
                    return lo - pad, hi + pad
                rlo, rhi = _pad(rlo, rhi, args.poincare_pad_frac)
                zlo, zhi = _pad(zlo, zhi, args.poincare_pad_frac)

                ax_p.set_xlim(rlo, rhi)
                ax_p.set_ylim(zlo, zhi)
                ax_p.set_aspect("auto")  # key line: don't force equal aspect
            else:
                rmin, rmax = float(np.min(R_all)), float(np.max(R_all))
                zmin, zmax = float(np.min(Z_all)), float(np.max(Z_all))
                set_equal_data_aspect(ax_p, rmin, rmax, zmin, zmax, pad_frac=0.03)
        else:
            R_max_box = float(np.sqrt(x_max**2 + y_max**2))
            set_equal_data_aspect(ax_p, 0.0, R_max_box, z_min, z_max, pad_frac=0.03)

        if poincare_label_pi:
            ax_p.legend(loc="best", frameon=True, framealpha=0.85)

        fig_p.tight_layout()
        if save_figure:
            suffix = "_multi" if len(phis) > 1 else f"_phi{float(phis[0]):.6f}".replace(".", "p").replace("-", "m")
            poincare_out = mfs_npz.replace(".npz", "_poincare")
            fig_p.savefig(f"{poincare_out}{suffix}.png")
            print(f"[POINCARE] Saved {poincare_out}{suffix}.png")

    # -------------------- 3D viewer: boundary + field lines -------------------- #
    # Plot 3D with |∇φ| on boundary points and field lines
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection="3d")

    # Color boundary points by |∇φ|
    Pj = jnp.asarray(P, dtype=jnp.float64)
    G = jax.vmap(grad_point_fn)(Pj)              # (Nb, 3)
    Gm = np.linalg.norm(np.asarray(G), axis=1)   # |∇φ| on boundary

    # Robust color limits (avoid outliers)
    vmin = float(np.nanpercentile(Gm, 1.0))
    vmax = float(np.nanpercentile(Gm, 99.0))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps.get_cmap("viridis")
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = mappable.to_rgba(Gm)

    ax_3d.scatter(
        P[:, 0], P[:, 1], P[:, 2],
        c=colors, s=1, depthshade=False, alpha=0.85
    )
    cb = plt.colorbar(mappable, ax=ax_3d, pad=0.05)
    cb.set_label(r"$|\nabla \phi|$ on $\Gamma$")

    # Field lines
    for line in Y:
        ax_3d.plot(line[:, 0], line[:, 1], line[:, 2], lw=1.1)

    # Seed points
    ax_3d.scatter(
        seeds_arr[:, 0], seeds_arr[:, 1], seeds_arr[:, 2],
        s=18, depthshade=True, color="k"
    )

    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.set_title("Field lines of ∇φ (MFS)")

    fix_matplotlib_3d(ax_3d)

    fig_3d.tight_layout()
    if save_figure:
        figure_out = mfs_npz.replace(".npz", "_fieldlines.png")
        fig_3d.savefig(figure_out)
        print(f"[FIGURE] Saved {figure_out}")

    plt.show()

if __name__ == "__main__":
    ### Solutions available in the repo (uncomment one to use as default)
    # default_solution = "wout_precise_QA_solution.npz";
    # default_solution = "wout_precise_QH_solution.npz";
    default_solution = "wout_SLAM_4_coils_solution.npz"
    # default_solution = "wout_SLAM_6_coils_solution.npz"

    nfp_default = 2; tfinal_default = 1200.0; seeds_default = None; n_save_default = 8
    if 'QH' in default_solution: nfp_default = 4
    if 'SLAM' in default_solution:
        n_save_default = 0.5; tfinal_default = 15000; seeds_default = "2.55:0:0,2.65:0:0,2.75:0:0,2.8:0:0,2.85:0:0,2.9:0:0,2.95:0:0,3.0:0:0"
    ap = argparse.ArgumentParser()
    ### MAIN PARAMETERS TO CHANGE
    ap.add_argument("file", nargs="?", type=str, default=resolve_npz_file_location(default_solution),
                    help="Path to mfs_solution.npz (positional or --file).")
    ap.add_argument("-f", "--file", dest="file", type=str, help="Path to mfs_solution.npz (overrides positional if both given).")
    ap.add_argument("--nfp", type=int, default=nfp_default, help="Number of field periods for Poincaré sampling.")
    ap.add_argument("--tfinal", type=float, default=tfinal_default, help="Final integration time for streamlines.")
    ap.add_argument("--n-save", type=int, default=n_save_default, help="Factor => total output points = n_save * tfinal")
    ### NUMBER OF FIELDLINES, NSEED, WILL BE THE NUMBER OF PROCESSORS USED (DEFINED ON TOP)
    ap.add_argument("--save-figure", action="store_true", default=True, help="Save figures to disk instead of just showing.")
    ap.add_argument("--nseed", type=int, default=None)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--clip", type=float, default=None)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--poincare-label-pi", action="store_true", help="Use π-fraction labels on Poincaré plots.")
    ap.add_argument("--box-pad", type=float, default=0.40)
    ap.add_argument("--poincare-nphi", type=int, default=4)
    ap.add_argument("--seed-mode", choices=["axis", "boundary"], default="axis",
                    help="axis: chord across center using a_hat; boundary: old inward-offset sampling")
    ap.add_argument("--strip-tol-frac", type=float, default=0.03, help="Half-width of selection strip (fraction of spread).")
    ap.add_argument("--plane-tol-frac", type=float, default=0.10, help="For torus: |s| tolerance along a_hat (fraction of span).")
    ap.add_argument("--inward-frac", type=float, default=0.02, help="Inward nudge fraction based on local spacing.")
    ap.add_argument("--poincare-tight", action="store_true", default=True,
                    help="Tight axes based on data percentiles; disables equal aspect.")
    ap.add_argument("--poincare-pad-frac", type=float, default=0.03,
                    help="Padding fraction for tight Poincaré limits.")
    ap.add_argument("--poincare-pct", type=float, nargs=2, default=[0.1, 99.9],
                    help="Low/high percentiles for tight limits (e.g., 1 99).")
    ap.add_argument("--mask-mode", choices=["strict","instant","entered","none"],
                    default="entered", help="Masking policy: ""strict=cumulative AND from t0; "
                        "instant=keep only samples inside; " "entered=keep after first entry until exit; " "none=no masking.")
    ap.add_argument("--mask-report", action="store_true", default=True, help="Print per-line mask stats.")
    ap.add_argument("--seeds", type=str, default=seeds_default,
        help="Comma-separated list of seed points as x:y:z,x:y:z,... (overrides automatic seed computation)")

    args = ap.parse_args()

    # Parse seeds if provided
    user_seeds = None
    if args.seeds is not None:
        try:
            user_seeds = []
            for item in args.seeds.split(","):
                xyz = tuple(float(v) for v in item.split(":"))
                if len(xyz) == 3: user_seeds.append(xyz)
            if len(user_seeds) == 0: user_seeds = None
        except Exception as e:
            print(f"[ERROR] Could not parse --seeds argument: {e}")
            user_seeds = None

    if "SLAM" in args.file:
        args.poincare_phi = [0., 1.57079633]#, 2.6]
    else:
        args.poincare_phi = jnp.linspace(0, 2*jnp.pi/args.nfp, args.poincare_nphi, endpoint=False).tolist()
    args.nseed = number_of_processors_to_use

    n_save = int(args.n_save * args.tfinal)
    main( mfs_npz=args.file, seeds=user_seeds,
        t_final=args.tfinal, normalize=args.normalize, clip_grad=args.clip,
        nseed=args.nseed, eps=args.eps, rtol=args.rtol, atol=args.atol,
        n_save=n_save, box_pad=args.box_pad, poincare_phi=args.poincare_phi,
        save_figure=args.save_figure, poincare_label_pi=args.poincare_label_pi, args=args)
