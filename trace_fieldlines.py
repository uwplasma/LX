###
### Run as python trace_fieldlines.py --ckpt pinn_torus_model.SIREN-sin_FF-1x2x4x8_W\[32x32x32x32\]_w30_R0f1.eqx --tfinal=500 --n-save 1000
###
#!/usr/bin/env python3
"""
Trace field lines of ∇u for the trained PINN solution.

We integrate dx/dt = ∇u(x) with Diffrax, starting from user-specified seeds.
Requires: jax, jaxlib, equinox, diffrax, optax, toml (or tomllib on 3.11+), numpy, matplotlib.
"""

from __future__ import annotations
import os, sys, math
from pathlib import Path
from typing import Sequence, Tuple, Callable, List, Optional

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx

import matplotlib as mpl

# -----------------------------
# Project-local imports (edit if your paths differ)
# -----------------------------
from _initialization import load_config  # your nested TOML loader
from _physics import u_total as project_u_total  # (params, xyz) -> scalar
from _network_and_loss import PotentialMLP, SirenMLP
from _geometry import a_of_phi, inside_torus_mask
from _state import runtime
from _geometry import build_surface_torus

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
         atol: float = 1e-7):
    jax.config.update("jax_enable_x64", True)

    cfg = load_config(config_path)
    apply_runtime_from_config(cfg)   # <<< make geometry use the same runtime as main.py

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

    # Default seeds: on midplane line z=0, y=0, spanning from left to right surface
    if seeds is None:
        R0 = float(cfg.get("geometry", {}).get("R0", 1.0))
        a0p = float(a_of_phi(jnp.array([0.0], dtype=jnp.float64))[0])       # φ=0 (x>0 midplane)
        api = float(a_of_phi(jnp.array([jnp.pi], dtype=jnp.float64))[0])    # φ=π (x<0 midplane)

        x_right_surf = +(R0 + a0p)
        x_left_surf  = -(R0 + api)

        # inset toward the axis so seeds are inside
        x_right = x_right_surf - eps
        x_left  = x_left_surf  + eps

        xs = np.linspace(x_left, x_right, int(nseed), dtype=float)
        ys = np.zeros_like(xs)
        zs = np.zeros_like(xs)
        seeds = [(float(x), 0.0, 0.0) for x in xs]  # y=z=0

        # safety filter in case extreme shaping puts some seeds out of bounds
        mask = np.array(inside_torus_mask(jnp.asarray(xs), jnp.asarray(ys), jnp.asarray(zs)), dtype=bool)
        seeds = [s for s, m in zip(seeds, mask) if m]
        if not seeds:
            raise RuntimeError("No valid seeds found on z=0, y=0 line. Increase --eps or check geometry.")

    # Integrate forward and backward for each seed
    # seeds: List[tuple] → array (S,3)
    seeds_arr = np.asarray(seeds, dtype=np.float64)

    # Forward and backward in parallel
    ts_f, Yf = integrate_streamlines_vmap(
        seeds_arr, f, t_final=t_final, box=box,
        backward=False, n_save=args.n_save, rtol=rtol, atol=atol
    )
    ts_b, Yb = integrate_streamlines_vmap(
        seeds_arr, f, t_final=t_final, box=box,
        backward=True,  n_save=args.n_save, rtol=rtol, atol=atol
    )

    # Concatenate backward (reversed) + forward for each seed
    Y = np.concatenate([np.flip(Yb, axis=1), Yf], axis=1)  # (S, 2*n_save, 3)

    # Plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    # colored |∇u| surface (with colorbar)
    mappable, vmin, vmax = add_colored_surface(ax, cfg, u_fn, cmap="viridis", alpha=0.4, stride_theta=1, stride_phi=1)
    cb = plt.colorbar(mappable, ax=ax, pad=0.05)
    cb.set_label(r"$|\nabla u|$")
    for line in Y:  # line: (2*n_save, 3)
        ax.plot(line[:, 0], line[:, 1], line[:, 2], lw=1.2)
    ax.scatter([s[0] for s in seeds], [s[1] for s in seeds], [s[2] for s in seeds], s=20, depthshade=True)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Field lines of ∇u")
    # Box limits
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="input.toml", help="Path to config TOML")
    ap.add_argument("--ckpt", default=None, help="Path to checkpoint (.eqx) with arrays-only weights")
    ap.add_argument("--tfinal", type=float, default=6.0, help="Integration horizon (forward/backward)")
    ap.add_argument("--normalize", action="store_true", help="Follow only direction of ∇u (unit-speed)")
    ap.add_argument("--clip", type=float, default=None, help="Clip ||∇u|| to this max norm (after optional normalization)")
    ap.add_argument("--nseed", type=int, default=25, help="Number of field lines (seed points) along x at y=z=0")
    ap.add_argument("--eps", type=float, default=1e-3, help="Inset from the surface along x so seeds start inside")
    ap.add_argument("--save-stride", type=int, default=1,
                    help="Keep every N-th internal solver step (1=all).")
    ap.add_argument("--rtol", type=float, default=1e-5, help="Solver relative tolerance.")
    ap.add_argument("--atol", type=float, default=1e-7, help="Solver absolute tolerance.")
    ap.add_argument("--n-save", type=int, default=2001,
                    help="Number of evenly spaced save times between t0 and t1.")
    args = ap.parse_args()
    main(args.config, args.ckpt, t_final=args.tfinal, normalize=args.normalize, clip_grad=args.clip,
        nseed=args.nseed, eps=args.eps, save_stride=args.save_stride,
        rtol=args.rtol, atol=args.atol)
