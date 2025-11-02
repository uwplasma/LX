###
###      Run as python trace_fieldlines.py --ckpt pinn_torus_model.SIREN-sin_FF-1x2x4x8_W\[32x32x32x32\]_w30_R0f1.eqx --tfinal=500 --n-save 1000
### clear;clear;python trace_fieldlines.py --ckpt pinn_torus_model.SIREN-sin_FF-1x2x4x8_W\[64x64x64x64\]_w30_R0f1.eqx --tfinal=1000 --n-save 1000
###
#!/usr/bin/env python3
"""
Trace field lines of ∇u for the trained PINN solution.

We integrate dx/dt = ∇u(x) with Diffrax, starting from user-specified seeds.
Requires: jax, jaxlib, equinox, diffrax, optax, toml (or tomllib on 3.11+), numpy, matplotlib.
"""

from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)
import os, sys, math
from pathlib import Path
from typing import Sequence, Tuple, Callable, List, Optional

import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx

import matplotlib as mpl

from fractions import Fraction


# -----------------------------
# Project-local imports (edit if your paths differ)
# -----------------------------
from _initialization import load_config  # your nested TOML loader
from _physics import u_total as project_u_total  # (params, xyz) -> scalar
from _network_and_loss import PotentialMLP, SirenMLP
from _geometry import a_of_phi, inside_torus_mask
from _state import runtime
from _geometry import build_surface_torus
from _geometry_files import build_surfaces_from_files   # files mode
from _multisurface import SurfaceItem                    # dataclass with P_bdry, N_bdry, inside_mask_fn

# -----------------------------
# JAX Poincaré (dense, JIT-safe; no dynamic indexing)
# -----------------------------
# ---------- Plot helpers (paper-quality & equal aspect) ----------
def apply_paper_style():
    """Good-looking defaults for papers. scale in {"1col","2col"}."""
    # if scale == "1col":
    # fig_w = 3.4  # inches (APS single column ≈ 3.375")
    # else:
    fig_w = 5.5  # inches (two-column width)
    mpl.rcParams.update({
        "figure.figsize": (fig_w, fig_w),   # square Poincaré
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 12,        # base font
        "axes.titlesize": 13,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        # Keep mathtext (no LaTeX dependency); looks good enough for most journals
        "text.usetex": False,
    })
    return fig_w

def set_equal_data_aspect(ax, rmin, rmax, zmin, zmax, pad_frac=0.03):
    """Make x/y have equal scale based on data limits, with symmetric padding."""
    # Initial padded limits
    def _pad(lo, hi, frac):
        span = hi - lo
        if span <= 0:
            span = max(1e-6, abs(hi) if abs(hi) > 0 else 1.0)
        pad = frac * span
        return lo - pad, hi + pad

    rlo, rhi = _pad(float(rmin), float(rmax), pad_frac)
    zlo, zhi = _pad(float(zmin), float(zmax), pad_frac)

    # Enforce equal data span: expand the smaller range to match the larger
    rc = 0.5 * (rlo + rhi)
    zc = 0.5 * (zlo + zhi)
    rspan = rhi - rlo
    zspan = zhi - zlo
    span = max(rspan, zspan)
    rlo, rhi = rc - 0.5*span, rc + 0.5*span
    zlo, zhi = zc - 0.5*span, zc + 0.5*span

    ax.set_xlim(rlo, rhi)
    ax.set_ylim(zlo, zhi)
    ax.set_aspect("equal", adjustable="box")  # 1 unit in R equals 1 unit in Z

def _angle_wrap_jnp(a):
    return (a + jnp.pi) % (2*jnp.pi) - jnp.pi

def _wrap_diff_jnp(a_minus_b):
    return _angle_wrap_jnp(a_minus_b)

@jax.jit
def poincare_RZ_points_jax_dense(Y_all: jnp.ndarray, phi0: float):
    """
    Y_all: (S, T, 3) with possible NaNs after exit masking.
    phi0: scalar (radians).
    Returns:
      R_flat: (S*(T-1),)
      Z_flat: (S*(T-1),)
      mask_flat: bool (S*(T-1),) indicating which entries are true crossings.
    """
    # Valid samples
    valid = ~jnp.any(jnp.isnan(Y_all), axis=-1)            # (S,T)

    # Coordinates & angle
    X = Y_all[..., 0]; Y = Y_all[..., 1]; Z = Y_all[..., 2]
    phi = jnp.arctan2(Y, X)                                # (S,T)
    dphi = _wrap_diff_jnp(phi - phi0)                      # (S,T)
    s = jnp.sign(dphi)
    s = jnp.where(s == 0.0, 1.0, s)

    # Segment endpoints (between t and t+1)
    valid_seg = valid[..., :-1] & valid[..., 1:]           # (S,T-1)
    changed   = (s[..., :-1] * s[..., 1:] < 0.0) & valid_seg

    # Gather all segments densely
    p0 = Y_all[:, :-1, :]                                  # (S,T-1,3)
    p1 = Y_all[:,  1:, :]                                  # (S,T-1,3)
    d0 = dphi[:, :-1]                                      # (S,T-1)
    d1 = dphi[:,  1:]                                      # (S,T-1)

    # Linear interpolation fraction for *every* segment
    t = d0 / (d0 - d1)
    t = jnp.clip(t, 0.0, 1.0)

    # Interpolated points (S,T-1,3)
    p = p0 + t[..., None] * (p1 - p0)

    # Cylindrical R and Z for all candidates
    R = jnp.linalg.norm(p[..., :2], axis=-1)               # (S,T-1)
    Zc = p[..., 2]                                         # (S,T-1)

    # Flatten (static shape) + flatten mask
    R_flat    = R.reshape(-1)
    Z_flat    = Zc.reshape(-1)
    mask_flat = changed.reshape(-1)

    return R_flat, Z_flat, mask_flat

def poincare_multi_phi_jax(Y_all: jnp.ndarray, phis: jnp.ndarray):
    """
    Vectorized over multiple φ values.
    Returns Python lists (ragged after masking) for easy plotting.
    """
    # vmapped dense outputs with static shapes
    R_flat, Z_flat, M_flat = jax.vmap(poincare_RZ_points_jax_dense, in_axes=(None, 0))(Y_all, phis)
    # Convert each φ's flat arrays to NumPy and apply masks on CPU (variable length ok here)
    R_list = [np.asarray(R_flat[k])[np.asarray(M_flat[k])] for k in range(R_flat.shape[0])]
    Z_list = [np.asarray(Z_flat[k])[np.asarray(M_flat[k])] for k in range(Z_flat.shape[0])]
    return R_list, Z_list


# Files helpers

def load_training_surface_from_config(cfg: dict) -> Tuple[str, Optional[SurfaceItem]]:
    """
    Returns (mode, surface_item_or_None).
    - mode == "torus": returns ( "torus", None )
    - mode == "files": returns ( "files", SurfaceItem ) using the FIRST entry
    """
    surfaces_cfg = cfg.get("surfaces", {})
    mode = str(surfaces_cfg.get("mode", "single")).lower()
    if mode == "files":
        files_list = surfaces_cfg.get("files", [])
        if not files_list:
            raise RuntimeError("surfaces.mode='files' but [surfaces].files is empty.")
        dataset = build_surfaces_from_files(files_list)
        if not dataset:
            raise RuntimeError("build_surfaces_from_files returned an empty dataset.")
        surf0 = dataset[0]
        return "files", surf0
    return "torus", None


def seeds_for_files_surface(surf: SurfaceItem, nseed: int = 25, eps: float = 1e-3) -> np.ndarray:
    """
    Build seed points *inside* by moving boundary points along -N by eps.
    Uniform downsample to ~nseed.
    """
    Pb = np.asarray(surf.P_bdry)
    Nb = np.asarray(surf.N_bdry)
    Nb = Nb / (np.linalg.norm(Nb, axis=1, keepdims=True) + 1e-12)
    Pi = Pb - eps * Nb       # nudge inward
    # Downsample uniformly if too many:
    if Pi.shape[0] > nseed:
        stride = max(1, Pi.shape[0] // nseed)
        Pi = Pi[::stride][:nseed]
    return Pi.astype(np.float64)

def phi_label_pi(phi: float, wrap=True, max_den=24) -> str:
    """
    Return LaTeX string like r"$\\phi=\\pi/2$" for a given angle in radians.
    - wrap=True: first wrap phi into (-pi, pi]; set False to use raw value
    - max_den: limit denominator when approximating rational multiples of pi
    """
    if wrap:
        # wrap to (-pi, pi]
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

def add_colored_surface_points(ax, surf: SurfaceItem, u_fn, *, cmap="viridis", alpha=0.9):
    """
    Scatter the *file-based* boundary points colored by |∇u|.
    Returns a ScalarMappable to attach a colorbar.
    """
    P = np.asarray(surf.P_bdry)
    # Evaluate |∇u| at boundary points
    Pj = jnp.asarray(P, dtype=jnp.float64)
    G  = grad_on_points(u_fn, Pj)                  # (Nb,3)
    Gm = jnp.linalg.norm(G, axis=-1)               # (Nb,)

    Gm_np = np.asarray(Gm)
    vmin, vmax = float(np.nanmin(Gm_np)), float(np.nanmax(Gm_np))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = m.to_rgba(Gm_np)
    ax.scatter(P[:,0], P[:,1], P[:,2], c=colors, s=4, depthshade=False, alpha=float(alpha))
    return m, vmin, vmax


# -----------------------------
# Robust arrays-only checkpoint I/O
# -----------------------------
def save_arrays_only(model: eqx.Module, path: str | Path):
    arrs, _ = eqx.partition(model, eqx.is_inexact_array)
    eqx.tree_serialise_leaves(str(path), arrs)

def load_arrays_only(template: eqx.Module, path: str | Path) -> eqx.Module:
    arrs_t, stat_t = eqx.partition(template, eqx.is_inexact_array)
    arrs_l = eqx.tree_deserialise_leaves(str(path), arrs_t)
    return eqx.combine(arrs_l, stat_t)

# -----------------------------
# Build model from config
# -----------------------------
def _act_from_name(name: str) -> Callable:
    name = (name or "tanh").lower()
    return {
        "tanh": jax.nn.tanh,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "silu": jax.nn.silu,
        "softplus": jax.nn.softplus,
        "identity": (lambda x: x),
        "sin": jnp.sin,  # only used for PotentialMLP; SIREN ignores this and uses sin internally
    }.get(name, jax.nn.tanh)

def build_model(cfg: dict, key: jax.Array):
    m = cfg.get("model", {})
    hidden = tuple(int(x) for x in m.get("hidden_sizes", [48,48]))
    use_fourier   = bool(m.get("use_fourier", False))
    fourier_bands = tuple(float(b) for b in m.get("fourier_bands", []))
    fourier_scale = float(m.get("fourier_scale", 2*math.pi))
    R0f           = float(m.get("R0_for_fourier", 1.0))

    if bool(m.get("siren", False)) or str(m.get("activation","")).lower() in ("sin","sine"):
        # SIREN path
        omega0 = float(m.get("siren_omega0", 30.0))
        model = SirenMLP(key, widths=hidden, omega0=omega0,
                         use_fourier=use_fourier, fourier_bands=fourier_bands,
                         fourier_scale=fourier_scale, R0_for_fourier=R0f, add_raw_xyz=True)
    else:
        # Standard MLP
        act = _act_from_name(m.get("activation","tanh"))
        model = PotentialMLP(key, hidden_sizes=hidden, act=act,
                             use_fourier=use_fourier, fourier_bands=fourier_bands,
                             fourier_scale=fourier_scale, R0_for_fourier=R0f)
    return model

# -----------------------------
# Multi-valued + NN total potential u_total
# -----------------------------
def make_u_total(cfg: dict, model: eqx.Module) -> Callable[[jnp.ndarray], jnp.ndarray]:
    # Use the project's authoritative u_total if present (handles batching).
    def u_fn(x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., 3)
        return project_u_total(model, x)  # expects model as "params"
    return u_fn

# -----------------------------
# JIT gradient and ODE RHS
# -----------------------------
def make_grad_u(u_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    # Convert scalar-valued u(x) into grad u(x) for x∈R^3.
    # Accepts (...,3) and returns (...,3).
    def _u_point(x):
        # x: (3,)
        return u_fn(x[None, :]).squeeze()  # scalar
    grad_point = jax.jit(jax.grad(_u_point))  # (3,) -> (3,)

    @jax.jit
    def grad_batch(xs):
        # xs: (...,3)
        return jax.vmap(grad_point)(xs.reshape(-1, 3)).reshape(xs.shape)
    return grad_point  # we’ll use pointwise in the ODE

def make_rhs(grad_u_point: Callable[[jnp.ndarray], jnp.ndarray],
             *, clip_grad: Optional[float]=None, normalize: bool=False):
    # Diffrax ODE: f(t, y, args) = ∇u(y)
    @jax.jit
    def f(t, y, args):
        g = grad_u_point(y)
        if normalize:
            n = jnp.linalg.norm(g) + 1e-12
            g = g / n
        if (clip_grad is not None) and (clip_grad > 0):
            # clip by norm
            n = jnp.linalg.norm(g) + 1e-12
            g = jnp.where(n > clip_grad, g * (clip_grad / n), g)
        return g
    return f

# -----------------------------
# Integrate one streamline forward/backward
# -----------------------------
def integrate_streamline(
    y0: np.ndarray,
    f,
    t_final: float = 5.0,
    dt0: float = 1e-2,
    box: Tuple[float,float,float,float,float,float] = (-1.5,1.5,-1.5,1.5,-1.0,1.0),
    *,
    backward: bool = False,
    save_stride: int = 1, 
    n_save: int = 2001,        # <— NEW (1 = keep every internal step)
    rtol: float = 1e-5,          # <— NEW
    atol: float = 1e-7,          # <— NEW
):
    y0 = jnp.asarray(y0, dtype=jnp.float64)
    t0, t1 = (0.0, -t_final) if backward else (0.0, t_final)
    dt0_signed = -abs(dt0) if backward else abs(dt0)

    solver = dfx.Dopri8()
    stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)   # was fixed

    x_min, x_max, y_min, y_max, z_min, z_max = box

    def out_of_box(t, y, args, **kwargs):
        return ((y[0] < x_min) | (y[0] > x_max)
             |  (y[1] < y_min) | (y[1] > y_max)
             |  (y[2] < z_min) | (y[2] > z_max))

    term = dfx.ODETerm(f)
    event = dfx.Event(out_of_box)
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, int(n_save), dtype=jnp.float64))
    progress_meter = dfx.TqdmProgressMeter()

    sol = dfx.diffeqsolve(
        term, solver, t0=t0, t1=t1, dt0=dt0_signed, y0=y0,
        stepsize_controller=stepsize_controller,
        max_steps=200_000, progress_meter=progress_meter,
        saveat=saveat,
        event=event,
    )
    ys = np.array(sol.ys); ts = np.array(sol.ts)
    return ts, ys

def integrate_streamlines_vmap(
    seeds: np.ndarray,             # (S,3) float64
    f,                             # RHS
    t_final: float = 5.0,
    dt0: float = 1e-2,
    box: Tuple[float,float,float,float,float,float] = (-1.5,1.5,-1.5,1.5,-1.0,1.0),
    *,
    backward: bool = False,
    n_save: int = 2001,
    rtol: float = 1e-5,
    atol: float = 1e-7,
):
    """Integrate all seeds in parallel (no terminal event). Returns (S, n_save, 3)."""
    seeds = jnp.asarray(seeds, dtype=jnp.float64)
    t0, t1 = (0.0, -t_final) if backward else (0.0, t_final)
    dt0_signed = -abs(dt0) if backward else abs(dt0)

    solver = dfx.Dopri8()
    stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)
    term = dfx.ODETerm(f)
    ts = jnp.linspace(t0, t1, int(n_save), dtype=jnp.float64)
    saveat = dfx.SaveAt(ts=ts)

    # Solve one streamline (no event so shapes are static → vmap-friendly)
    def _solve_one(y0):
        sol = dfx.diffeqsolve(
            term, solver, t0=t0, t1=t1, dt0=dt0_signed, y0=y0,
            stepsize_controller=stepsize_controller,
            max_steps=200_000, saveat=saveat
        )
        return sol.ys  # (n_save, 3)

    ys_all = jax.vmap(_solve_one)(seeds)  # (S, n_save, 3)

    # Post-mask anything outside the box to NaN for plotting clarity
    x_min, x_max, y_min, y_max, z_min, z_max = box
    X, Y, Z = ys_all[..., 0], ys_all[..., 1], ys_all[..., 2]
    in_box = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max) & (Z >= z_min) & (Z <= z_max)
    # Optional: stop after first exit. Build a "keep" mask that stays True until first False.
    # This uses a cumulative-and scan along time.
    def _cumkeep(mask_t):
        # mask_t: (n_save,)
        # keep[i] = keep[i-1] & mask_t[i]
        return jax.lax.associative_scan(lambda a, b: a & b, mask_t, axis=0)

    keep_mask = jax.vmap(_cumkeep)(in_box)      # (S, n_save)
    keep_mask = keep_mask[..., None]            # (S, n_save, 1)
    ys_all = jnp.where(keep_mask, ys_all, jnp.nan)

    return np.asarray(ts), np.asarray(ys_all)

def add_torus_surface(ax, cfg, *, alpha: float = 0.25, stride_theta: int = 2, stride_phi: int = 2):
    """Plot the current torus surface with semi-transparency."""
    # Use the same θ×φ resolution you use for boundary sampling (or sensible defaults)
    samp = cfg.get("sampling", {})
    nθ = int(samp.get("N_bdry_theta", 32))
    nφ = int(samp.get("N_bdry_phi", 64))

    Xg, Yg, Zg = build_surface_torus(nθ, nφ)  # uses runtime geometry set by load_config
    Xg = np.asarray(Xg); Yg = np.asarray(Yg); Zg = np.asarray(Zg)

    ax.plot_surface(
        Xg, Yg, Zg,
        rstride=max(1, int(stride_theta)),
        cstride=max(1, int(stride_phi)),
        color="lightgray",
        linewidth=0,
        antialiased=True,
        alpha=float(alpha),
        shade=True,
    )

def apply_runtime_from_config(cfg: dict):
    # geometry
    geom = cfg.get("geometry", {})
    runtime.R0      = float(geom.get("R0", 1.0))
    runtime.a0      = float(geom.get("a0", 0.35))
    runtime.a1      = float(geom.get("a1", 0.15))
    runtime.N_harm  = int(geom.get("N_harm", 4))

    # sampling/batch (optional, but keeps consistency if used by any helper)
    batch   = cfg.get("batch", {})
    runtime.BATCH_IN    = int(batch.get("interior", 2048))
    runtime.BATCH_BDRY  = int(batch.get("boundary", 2048))

    # optimization bits that your network/loss reference via runtime
    opt = cfg.get("optimization", {})
    runtime.lam_bc = float(opt.get("lam_bc", 5.0))

    # regularization
    reg = cfg.get("regularization", {})
    runtime.zero_mean_weight = float(reg.get("zero_mean_weight", 0.1))

    # importance sampling multiplier
    samp = cfg.get("sampling", {})
    runtime.bdry_presample_mult = int(samp.get("bdry_presample_mult", cfg.get("bdry_presample_mult", 16)))

    # multi-valued piece (if you use it inside u_total)
    mv = cfg.get("multi_valued", {})
    runtime.kappa = float(mv.get("kappa", 0.0))

    # optional: set a default box on runtime for geometry utils that read it
    box = cfg.get("box", {})
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    z_min = float(box.get("zmin", -0.8))
    z_max = float(box.get("zmax",  0.8))
    runtime.box_bounds = (x_min, x_max, y_min, y_max, z_min, z_max)

def grad_on_points(u_fn: Callable[[jnp.ndarray], jnp.ndarray], P: jnp.ndarray) -> jnp.ndarray:
    """P: (N,3) → ∇u(P): (N,3)."""
    def _u_point(x):  # (3,) -> scalar
        return u_fn(x[None, :]).squeeze()
    return jax.vmap(jax.grad(_u_point))(P)

def add_colored_surface(ax, cfg, u_fn, *, cmap="viridis", alpha=0.9, stride_theta=1, stride_phi=1):
    """Draw the torus colored by |∇u| with a colorbar."""
    samp = cfg.get("sampling", {})
    nθ = int(samp.get("N_bdry_theta", 32))
    nφ = int(samp.get("N_bdry_phi", 64))
    Xg, Yg, Zg = build_surface_torus(nθ, nφ)  # (nθ,nφ)

    # eval |∇u| on the grid
    P = jnp.stack([Xg, Yg, Zg], axis=-1).reshape(-1, 3)
    G = grad_on_points(u_fn, P)                    # (nθ*nφ, 3)
    Gmag = jnp.linalg.norm(G, axis=-1).reshape(Xg.shape)

    # Normalize colormap
    Gmag_np = np.asarray(Gmag)
    vmin, vmax = float(Gmag_np.min()), float(Gmag_np.max())
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    facecolors = m.to_rgba(Gmag_np)

    surf = ax.plot_surface(
        np.asarray(Xg), np.asarray(Yg), np.asarray(Zg),
        rstride=max(1, int(stride_theta)),
        cstride=max(1, int(stride_phi)),
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=False,      # let facecolors show through
        alpha=float(alpha)
    )
    # Attach a colorbar to the same normalization
    # Return a mappable the colorbar can use
    return m, vmin, vmax

# -----------------------------
# Main
# -----------------------------
def main(config_path="input.toml",
         ckpt_path=None,
         seeds: Optional[List[Tuple[float,float,float]]] = None,
         t_final=6.0,
         normalize=False,
         clip_grad=None,
         nseed: int = 25,
         eps: float = 1e-3,
         save_stride: int = 1,
         rtol: float = 1e-5,
         atol: float = 1e-7,
         n_save: int = 2001,):

    cfg = load_config(config_path)
    apply_runtime_from_config(cfg)   # <<< make geometry use the same runtime as main.py

    mode, file_surface = load_training_surface_from_config(cfg)
    print(f"[SURFACE] mode={mode}{' (files)' if mode=='files' else ''}")

    # Build model template
    key = jax.random.PRNGKey(0)
    model = build_model(cfg, key)

    # Decide/checkpoint name if not provided
    if ckpt_path is None:
        # If your training script already decided a filename, use that.
        # Otherwise point to your last used path.
        ckpt_path = cfg.get("checkpoint", {}).get("path", "pinn_torus_model.eqx")

    # Load arrays-only (robust) if the file exists; else use template
    ckpt_file = Path(ckpt_path)
    if ckpt_file.exists():
        try:
            model = load_arrays_only(model, ckpt_file)
            print(f"[LOAD] Loaded arrays from {ckpt_file}")
        except Exception as e:
            print(f"[LOAD] Failed to load arrays from {ckpt_file}: {e}. Using fresh model.")
    else:
        print(f"[LOAD] No checkpoint at {ckpt_file}; using fresh model.")

    # Build u_total and grad u
    u_fn = make_u_total(cfg, model)
    grad_u_point = make_grad_u(u_fn)
    f = make_rhs(grad_u_point, clip_grad=clip_grad, normalize=normalize)

    # Box from config (fallback to geometry-based cube)
    box_cfg = cfg.get("box", {})
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    z_min = float(box_cfg.get("zmin", -0.8))
    z_max = float(box_cfg.get("zmax",  0.8))
    box = (x_min, x_max, y_min, y_max, z_min, z_max)

    # Default seeds depend on mode:
    if seeds is None:
        if mode == "torus":
            # --- Same as your current logic (x-line at y=z=0) ---
            R0 = float(cfg.get("geometry", {}).get("R0", 1.0))
            a0p = float(a_of_phi(jnp.array([0.0], dtype=jnp.float64))[0])       # φ=0
            api = float(a_of_phi(jnp.array([jnp.pi], dtype=jnp.float64))[0])    # φ=π
            x_right_surf = +(R0 + a0p)
            x_left_surf  = -(R0 + api)
            x_right = x_right_surf - eps
            x_left  = x_left_surf  + eps
            xs = np.linspace(x_left, x_right, int(nseed), dtype=float)
            ys = np.zeros_like(xs)
            zs = np.zeros_like(xs)
            seeds = [(float(x), 0.0, 0.0) for x in xs]
            # safety filter
            mask = np.array(inside_torus_mask(jnp.asarray(xs), jnp.asarray(ys), jnp.asarray(zs)), dtype=bool)
            seeds = [s for s, m in zip(seeds, mask) if m]
            if not seeds:
                raise RuntimeError("No valid seeds found. Increase --eps or check geometry.")
        else:
            # mode == "files": nudge boundary points inward along -N, downsample to nseed
            seeds_arr = seeds_for_files_surface(file_surface, nseed=nseed, eps=eps)
            seeds = [tuple(x) for x in seeds_arr]
            
    # Integrate forward and backward for each seed
    # seeds: List[tuple] → array (S,3)
    seeds_arr = np.asarray(seeds, dtype=np.float64)

    # Forward and backward in parallel
    ts_f, Yf = integrate_streamlines_vmap(
        seeds_arr, f, t_final=t_final, box=box,
        backward=False, n_save=n_save, rtol=rtol, atol=atol
    )
    ts_b, Yb = integrate_streamlines_vmap(
        seeds_arr, f, t_final=t_final, box=box,
        backward=True,  n_save=n_save, rtol=rtol, atol=atol
    )

    # Concatenate backward (reversed) + forward for each seed
    Y = np.concatenate([np.flip(Yb, axis=1), Yf], axis=1)  # (S, 2*n_save, 3)
    
    # -----------------------------
    # Poincaré sections φ = const (single overlaid figure)
    # -----------------------------
    if args.poincare_phi:
        phis = jnp.asarray(args.poincare_phi, dtype=jnp.float64)

        # Compute R,Z for each φ (vectorized in JAX, masked on CPU)
        R_list, Z_list = poincare_multi_phi_jax(jnp.asarray(Y), phis)

        # Build one figure and overlay all sections (paper-style)
        apply_paper_style()  # or "2col" for a larger square
        fig_p, ax_p = plt.subplots()     # uses the rcParams figure size set above

        any_points = False
        for k, phi0 in enumerate(np.asarray(phis)):
            R, Z = R_list[k], Z_list[k]
            if R.size == 0:
                continue
            any_points = True
            # Rasterize points for small PDF size; keep axes/vector text as vector
            ax_p.scatter(R, Z, s=1, alpha=0.85, rasterized=True, label=phi_label_pi(float(phi0), wrap=False))

        ax_p.set_xlabel(r"$R=\sqrt{x^2+y^2}$")
        ax_p.set_ylabel(r"$Z$")
        ax_p.set_title("Poincaré section(s): cylindrical $\phi$")

        if any_points:
            R_all = np.concatenate([r for r in R_list if r.size])
            Z_all = np.concatenate([z for z in Z_list if z.size])
            # (optional) outlier trimming
            # rmin, rmax = np.percentile(R_all, [0.5, 99.5])
            # zmin, zmax = np.percentile(Z_all, [0.5, 99.5])
            rmin, rmax = float(np.min(R_all)), float(np.max(R_all))
            zmin, zmax = float(np.min(Z_all)), float(np.max(Z_all))
            set_equal_data_aspect(ax_p, rmin, rmax, zmin, zmax, pad_frac=0.03)
        else:
            # Fall back to geometry box (also with equal aspect)
            R_max_box = float(np.sqrt(x_max**2 + y_max**2))
            set_equal_data_aspect(ax_p, 0.0, R_max_box, z_min, z_max, pad_frac=0.03)

        ax_p.grid(True, alpha=0.3)
        ax_p.legend(loc="best", frameon=True, framealpha=0.85)
        fig_p.tight_layout()

        if args.poincare_out:
            suffix = "_multi" if len(phis) > 1 else f"_phi{float(phis[0]):.6f}".replace(".", "p").replace("-", "m")
            out_png = f"{args.poincare_out}{suffix}.png"
            out_pdf = f"{args.poincare_out}{suffix}.pdf"
            fig_p.savefig(out_png)  # 300 dpi from rcParams
            fig_p.savefig(out_pdf)  # vector (points rasterized, axes/text vector)
            print(f"[POINCARE] Saved {out_png} and {out_pdf}")
            
    # Plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    if mode == "torus":
        # colored |∇u| surface with a colorbar on param grid
        mappable, vmin, vmax = add_colored_surface(ax, cfg, u_fn, cmap="viridis", alpha=0.40,
                                                stride_theta=1, stride_phi=1)
    else:
        # files mode: color the boundary points
        mappable, vmin, vmax = add_colored_surface_points(ax, file_surface, u_fn, cmap="viridis", alpha=0.90)
    cb = plt.colorbar(mappable, ax=ax, pad=0.05); cb.set_label(r"$|\nabla u|$")
    for line in Y:  # line: (2*n_save, 3)
        ax.plot(line[:, 0], line[:, 1], line[:, 2], lw=1.2)
    ax.scatter([s[0] for s in seeds], [s[1] for s in seeds], [s[2] for s in seeds], s=20, depthshade=True)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Field lines of ∇u")
    # Box limits
    if mode == "files":
        Pb = np.asarray(file_surface.P_bdry)
        mins = Pb.min(axis=0); maxs = Pb.max(axis=0)
        pad = 0.10 * float(np.linalg.norm(maxs - mins))
        x_min, x_max = float(mins[0]-pad), float(maxs[0]+pad)
        y_min, y_max = float(mins[1]-pad), float(maxs[1]+pad)
        z_min, z_max = float(mins[2]-pad), float(maxs[2]+pad)
        box = (x_min, x_max, y_min, y_max, z_min, z_max)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="input.toml", help="Path to config TOML")
    ap.add_argument("--ckpt", default=None, help="Path to checkpoint (.eqx) with arrays-only weights")
    ap.add_argument("--tfinal", type=float, default=2000.0, help="Integration horizon (forward/backward)")
    ap.add_argument("--normalize", action="store_true", help="Follow only direction of ∇u (unit-speed)")
    ap.add_argument("--clip", type=float, default=None, help="Clip ||∇u|| to this max norm (after optional normalization)")
    ap.add_argument("--nseed", type=int, default=5, help="Number of field lines (seed points) along x at y=z=0")
    ap.add_argument("--eps", type=float, default=6e-2, help="Inset from the surface along x so seeds start inside")
    ap.add_argument("--save-stride", type=int, default=1,
                    help="Keep every N-th internal solver step (1=all).")
    ap.add_argument("--rtol", type=float, default=1e-10, help="Solver relative tolerance.")
    ap.add_argument("--atol", type=float, default=1e-10, help="Solver absolute tolerance.")
    ap.add_argument("--n-save", type=int, default=1500,
                    help="Number of evenly spaced save times between t0 and t1.")
    ap.add_argument("--poincare-phi", type=float, nargs="*", default=[0],
                    help="One or more cylindrical angles (in radians) for φ=const Poincaré sections. Example: --poincare-phi 0 1.57079632679")
    ap.add_argument("--poincare-out", default=None,
                    help="If set, base filename to save Poincaré plots (e.g., 'poincare'). Files will be suffixed with the φ value.")
    args = ap.parse_args()
    main(args.config, args.ckpt, t_final=args.tfinal, normalize=args.normalize, clip_grad=args.clip,
        nseed=args.nseed, eps=args.eps, save_stride=args.save_stride,
        rtol=args.rtol, atol=args.atol, n_save=args.n_save)
