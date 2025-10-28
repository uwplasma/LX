"""
Initialization utilities for configuration loading and normalization.
- Uses tomllib (Py3.11+) or falls back to tomli.
- Maps activation names to JAX functions.
- Produces a normalized params dict the main script can apply to globals.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

# ---- TOML loader (tomllib preferred, tomli fallback) ----
try:  # Python 3.11+
    import tomllib as _tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    try:
        import tomli as _tomllib  # type: ignore
    except Exception:  # pragma: no cover
        _tomllib = None  # type: ignore


def load_config(path: str = "input.toml") -> Dict[str, Any]:
    if _tomllib is None:
        raise RuntimeError(
            "Missing TOML reader. Use Python 3.11+ (tomllib) or `pip install tomli`"
        )
    with open(path, "rb") as f:
        return _tomllib.load(f)


def get_activation(name: str):
    name = (name or "tanh").strip().lower()
    mapping = {
        "tanh": jax.nn.tanh,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "sigmoid": jax.nn.sigmoid,
        "silu": jax.nn.silu,  # swish
        "swish": jax.nn.silu,
        "softplus": jax.nn.softplus,
        "identity": (lambda x: x),
        "none": (lambda x: x),
        "sin": jnp.sin,
        "cos": jnp.cos,
    }
    return mapping.get(name, jax.nn.tanh)


def parse_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw TOML into a flat params dict with concrete types."""
    p: Dict[str, Any] = {}

    # checkpoint
    p["checkpoint_path"] = str(cfg.get("checkpoint", {}).get("path", "pinn_torus_model.eqx"))

    # batch
    p["batch_interior"] = int(cfg.get("batch", {}).get("interior", 2048))
    p["batch_boundary"] = int(cfg.get("batch", {}).get("boundary", 2048))

    # geometry
    p["R0"] = float(cfg.get("geometry", {}).get("R0", 1.0))
    p["a0"] = float(cfg.get("geometry", {}).get("a0", 0.35))
    p["a1"] = float(cfg.get("geometry", {}).get("a1", 0.20))
    p["N_harm"] = int(cfg.get("geometry", {}).get("N_harm", 3))

    # multi_valued
    p["kappa"] = float(cfg.get("multi_valued", {}).get("kappa", 1.0))

    # sampling
    p["N_in"] = int(cfg.get("sampling", {}).get("N_in", 10_000))
    p["N_bdry_theta"] = int(cfg.get("sampling", {}).get("N_bdry_theta", 32))
    p["N_bdry_phi"] = int(cfg.get("sampling", {}).get("N_bdry_phi", 64))
    p["rng_seed"] = int(cfg.get("sampling", {}).get("rng_seed", 0))

    # model
    hidden = cfg.get("model", {}).get("hidden_sizes", [32, 32])
    if not isinstance(hidden, (list, tuple)):
        hidden = [32, 32]
    p["mlp_hidden_sizes"] = tuple(int(x) for x in hidden)
    p["mlp_activation"] = get_activation(cfg.get("model", {}).get("activation", "tanh"))
    p["siren"] = bool(cfg.get("model", {}).get("siren", False))
    p["siren_omega0"] = float(cfg.get("model", {}).get("siren_omega0", 30.0))

    # optimization
    p["steps"] = int(cfg.get("optimization", {}).get("steps", 1000))
    p["lr"] = float(cfg.get("optimization", {}).get("lr", 3e-3))
    p["lam_bc"] = float(cfg.get("optimization", {}).get("lam_bc", 5.0))
    p["lam_warm"] = float(cfg.get("optimization", {}).get("lam_warm", 200.0))
    p["log_every"] = int(cfg.get("optimization", {}).get("log_every", max(1, p["steps"] // 20)))
    p["mini_epoch"] = int(cfg.get("optimization", {}).get("mini_epoch", 5))
    opt = cfg.get("optimization", {})
    p["lbfgs_steps"] = int(opt.get("lbfgs_steps", 0))       # 0 disables polish
    p["lbfgs_tol"] = float(opt.get("lbfgs_tol", 1e-7))
    p["lbfgs_print_every"] = int(opt.get("lbfgs_print_every", 25))

    # plot
    p["plot_cmap"] = str(cfg.get("plot", {}).get("cmap", "viridis"))
    figsize = cfg.get("plot", {}).get("figsize", [8.0, 4.5])
    if not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
        figsize = [8.0, 4.5]
    p["figsize"] = (float(figsize[0]), float(figsize[1]))

    # --- box (fixed sampling domain) ---
    # Assume major radius ~ 1 and z ∈ [-0.5, 0.5].
    # We'll compute x,y from geometry (R0 + a0 + |a1|) each run.
    p["box_zmin"] = float(cfg.get("box", {}).get("zmin", -0.5))
    p["box_zmax"] = float(cfg.get("box", {}).get("zmax",  0.5))
    # How many candidate points to generate deterministically in the box
    p["box_points_total"] = int(cfg.get("box", {}).get("points_total", 200_000))
    # PRNG seed for the fixed box points
    p["box_seed"] = int(cfg.get("box", {}).get("seed", 42))

    # --- surfaces list (raw dict; parsed in main) ---
    p["surfaces_cfg"] = cfg.get("surfaces", {})

    # control how many points are used per surface in LBFGS (to cap memory)
    # 0 or missing => use "all" available points
    p["lbfgs_interior"]      = int(opt.get("lbfgs_interior", 0))     # interior points per surface
    p["lbfgs_boundary"]      = int(opt.get("lbfgs_boundary", 0))     # boundary points per surface
    p["lbfgs_weighting"]     = str(opt.get("lbfgs_weighting", "equal"))  # "equal" (default)

    return p


def build_params_from_path(path: str = "input.toml") -> Dict[str, Any]:
    return parse_config(load_config(path))
