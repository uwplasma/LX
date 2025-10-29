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
        "silu": jax.nn.silu,      # swish
        "swish": jax.nn.silu,
        "softplus": jax.nn.softplus,
        "identity": (lambda x: x),
        "none": (lambda x: x),
        "sin": jnp.sin,
        "sine": jnp.sin,
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
    # boundary presample multiplier (used by importance sampling)
    p["bdry_presample_mult"] = int(cfg.get("sampling", {}).get("bdry_presample_mult", 16))

    # regularization
    p["zero_mean_weight"] = float(cfg.get("regularization", {}).get("zero_mean_weight", 0.1))

    # model
    model_cfg = cfg.get("model", {})
    hidden = model_cfg.get("hidden_sizes", [32, 32])
    if not isinstance(hidden, (list, tuple)):
        hidden = [32, 32]
    p["mlp_hidden_sizes"] = tuple(int(x) for x in hidden)
    p["mlp_activation"] = get_activation(model_cfg.get("activation", "tanh"))
    p["siren"] = bool(model_cfg.get("siren", False))
    p["siren_omega0"] = float(model_cfg.get("siren_omega0", 30.0))
    p["use_fourier"] = bool(model_cfg.get("use_fourier", False))
    p["fourier_bands"] = tuple(float(x) for x in model_cfg.get("fourier_bands", [1.0, 2.0, 4.0, 8.0]))
    p["fourier_scale"] = float(model_cfg.get("fourier_scale", 2.0 * 3.141592653589793))
    # this is read in main() with default to R0 if missing
    p["R0_for_fourier"] = float(model_cfg.get("R0_for_fourier", p["R0"]))

    # optimization
    opt = cfg.get("optimization", {})
    p["steps"] = int(opt.get("steps", 1000))
    p["lr"] = float(opt.get("lr", 3e-3))
    p["lam_bc"] = float(opt.get("lam_bc", 5.0))
    p["lam_warm"] = float(opt.get("lam_warm", 200.0))
    p["log_every"] = int(opt.get("log_every", max(1, p["steps"] // 20)))
    p["mini_epoch"] = int(opt.get("mini_epoch", 5))

    # optimizer extras used in main()
    p["grad_clip_norm"] = float(opt.get("grad_clip_norm", 1.0))
    p["weight_decay"]   = float(opt.get("weight_decay", 0.0))
    p["lr_warmup_steps"] = int(opt.get("lr_warmup_steps", 0))
    p["lr_min_ratio"]    = float(opt.get("lr_min_ratio", 0.0))

    # LBFGS (top-level toggles)
    p["lbfgs_steps"] = int(opt.get("lbfgs_steps", 0))       # 0 disables polish
    p["lbfgs_tol"] = float(opt.get("lbfgs_tol", 1e-7))
    p["lbfgs_print_every"] = int(opt.get("lbfgs_print_every", 25))

    # LBFGS (caps per surface)
    p["lbfgs_interior"]  = int(opt.get("lbfgs_interior", 0))
    p["lbfgs_boundary"]  = int(opt.get("lbfgs_boundary", 0))
    p["lbfgs_weighting"] = str(opt.get("lbfgs_weighting", "equal"))

    # LBFGS (nested table) e.g. [optimization.lbfgs]
    opt_lbfgs = opt.get("lbfgs", {})
    p["lbfgs_l2"] = float(opt.get("lbfgs_l2", opt_lbfgs.get("l2", 1e-8)))
    p["lbfgs_include_zero_mean"] = bool(opt.get("lbfgs_include_zero_mean",
                                                opt_lbfgs.get("include_zero_mean", True)))
    p["lbfgs_include_aug_lagrangian"] = bool(opt.get("lbfgs_include_aug_lagrangian",
                                                     opt_lbfgs.get("include_aug_lagrangian", True)))

    # Augmented Lagrangian
    p["use_augmented_lagrangian"] = bool(opt.get("use_augmented_lagrangian", False))
    p["al_rho"] = float(opt.get("al_rho", 1.0))
    p["al_update_every"] = int(opt.get("al_update_every", 10))
    p["al_clip"] = float(opt.get("al_clip", 0.0))  # 0.0 => disabled

    # Lookahead / EMA
    p["use_lookahead"] = bool(opt.get("use_lookahead", False))
    p["lookahead_sync_period"] = int(opt.get("lookahead_sync_period", 5))
    p["lookahead_slow_step"] = float(opt.get("lookahead_slow_step", 0.5))
    p["use_ema"] = bool(opt.get("use_ema", False))
    p["ema_decay"] = float(opt.get("ema_decay", 0.999))
    p["ema_eval"]  = bool(opt.get("ema_eval", True))

    # plot
    plot_cfg = cfg.get("plot", {})
    p["plot_cmap"] = str(plot_cfg.get("cmap", "viridis"))
    figsize = plot_cfg.get("figsize", [8.0, 4.5])
    if not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
        figsize = [8.0, 4.5]
    p["figsize"] = (float(figsize[0]), float(figsize[1]))

    # --- box (fixed sampling domain) ---
    p["box_zmin"] = float(cfg.get("box", {}).get("zmin", -0.5))
    p["box_zmax"] = float(cfg.get("box", {}).get("zmax",  0.5))
    p["box_points_total"] = int(cfg.get("box", {}).get("points_total", 200_000))
    p["box_seed"] = int(cfg.get("box", {}).get("seed", 42))

    # --- surfaces list (raw dict; parsed in main) ---
    p["surfaces_cfg"] = cfg.get("surfaces", {})

    return p


def build_params_from_path(path: str = "input.toml") -> Dict[str, Any]:
    return parse_config(load_config(path))
