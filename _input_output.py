# _io.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple
import os
import math

# We don't import JAX here to keep this module lightweight; we only format.

# ----------------------------- TOML formatting ------------------------------

def _toml_bool(v: bool) -> str:
    return "true" if bool(v) else "false"

def _toml_str(s: str) -> str:
    # Minimal TOML escaping (good enough for our keys/paths)
    s = str(s).replace("\\", "\\\\").replace('"', '\\"')
    return f"\"{s}\""

def _toml_float(x: float) -> str:
    # Keep scientific when helpful, else up to 16 sig figs
    if x == 0 or (abs(x) >= 1e-4 and abs(x) < 1e4):
        return f"{float(x):.16g}"
    return f"{float(x):.6e}"

def _toml_int(n: int) -> str:
    return str(int(n))

def _toml_list(xs: Iterable[Any]) -> str:
    parts = []
    for x in xs:
        if isinstance(x, bool):
            parts.append(_toml_bool(x))
        elif isinstance(x, (int,)):
            parts.append(_toml_int(x))
        elif isinstance(x, float):
            parts.append(_toml_float(x))
        elif isinstance(x, str):
            parts.append(_toml_str(x))
        else:
            parts.append(_toml_str(str(x)))
    return f"[{', '.join(parts)}]"

def _emit_kv(k: str, v: Any) -> str:
    if isinstance(v, bool):
        return f"{k} = {_toml_bool(v)}"
    if isinstance(v, int):
        return f"{k} = {_toml_int(v)}"
    if isinstance(v, float):
        return f"{k} = {_toml_float(v)}"
    if isinstance(v, str):
        return f"{k} = {_toml_str(v)}"
    if isinstance(v, (list, tuple)):
        return f"{k} = {_toml_list(v)}"
    # Fallback to string
    return f"{k} = {_toml_str(str(v))}"

# ----------------------------- Public function ------------------------------

def dump_effective_toml(
    *,
    path: str = "effective_config.dump.toml",
    params: Dict[str, Any],
    runtime,                     # your _state.runtime object
    extra: Dict[str, Any] | None = None,  # optional: anything else to include
) -> str:
    """
    Write a TOML-like file reflecting the *effective* configuration actually used
    by the program (after parsing+defaults + runtime switches), so you can diff
    it against input.toml.

    Arguments
    ---------
    path : output filename
    params : the normalized dict returned by build_params_from_path()
    runtime : your shared runtime state (flags like lam_bc, AL, LBFGS flags)
    extra : optional flat dict to insert under [__derived] for debugging
    """
    # Collect all sections mirroring input.toml + a derived section.
    # We read strictly from `params` (normalized) and from `runtime` (live flags).
    surfaces_cfg = params.get("surfaces_cfg", {}) or {}
    mode = str(surfaces_cfg.get("mode", "single")).lower().strip()

    # Build [[surfaces.torus_list]] blocks if present
    torus_list = surfaces_cfg.get("torus_list", []) or []

    # Activation name (best-effort)
    act = params.get("mlp_activation", None)
    act_name = getattr(act, "__name__", str(act)) if act is not None else "tanh"

    # Compose text
    out: list[str] = []
    out.append("# Auto-generated view of the EFFECTIVE configuration in use.\n"
               "# Diff this against your input.toml to catch silent fallbacks.\n")

    # [surfaces]
    out.append("[surfaces]")
    out.append(_emit_kv("mode", mode))
    out.append("")  # blank line

    # [[surfaces.torus_list]]
    if torus_list:
        for t in torus_list:
            out.append("[[surfaces.torus_list]]")
            for k in ("name", "a0", "a1", "N_harm"):
                if k in t:
                    out.append(_emit_kv(k, t[k]))
            out.append("")

    # [checkpoint]
    out.append("[checkpoint]")
    out.append(_emit_kv("path", params.get("checkpoint_path", "pinn_torus_model.eqx")))
    out.append("")

    # [batch]
    out.append("[batch]")
    out.append(_emit_kv("interior", params.get("batch_interior", 2048)))
    out.append(_emit_kv("boundary", params.get("batch_boundary", 2048)))
    out.append("")

    # [geometry]
    out.append("[geometry]")
    out.append(_emit_kv("R0", params.get("R0", 1.0)))
    out.append(_emit_kv("a0", params.get("a0", 0.35)))
    out.append(_emit_kv("a1", params.get("a1", 0.20)))
    out.append(_emit_kv("N_harm", params.get("N_harm", 3)))
    out.append("")

    # [multi_valued]
    out.append("[multi_valued]")
    out.append(_emit_kv("kappa", params.get("kappa", 0.0)))
    out.append("")

    # [sampling]
    out.append("[sampling]")
    out.append(_emit_kv("N_in", params.get("N_in", 10000)))
    out.append(_emit_kv("N_bdry_theta", params.get("N_bdry_theta", 32)))
    out.append(_emit_kv("N_bdry_phi", params.get("N_bdry_phi", 64)))
    out.append(_emit_kv("rng_seed", params.get("rng_seed", 0)))
    out.append(_emit_kv("bdry_presample_mult", params.get("bdry_presample_mult", 16)))
    out.append("")

    # [regularization]
    out.append("[regularization]")
    out.append(_emit_kv("zero_mean_weight", params.get("zero_mean_weight", 0.1)))
    out.append("")

    # [model]
    out.append("[model]")
    out.append(_emit_kv("hidden_sizes", list(params.get("mlp_hidden_sizes", (32, 32)))))
    out.append(_emit_kv("activation", act_name))
    out.append(_emit_kv("siren", params.get("siren", False)))
    out.append(_emit_kv("siren_omega0", params.get("siren_omega0", 30.0)))
    out.append(_emit_kv("use_fourier", params.get("use_fourier", False)))
    out.append(_emit_kv("fourier_bands", list(params.get("fourier_bands", (1.0, 2.0, 4.0, 8.0)))))
    out.append(_emit_kv("fourier_scale", params.get("fourier_scale", 2.0 * math.pi)))
    out.append(_emit_kv("R0_for_fourier", params.get("R0_for_fourier", params.get("R0", 1.0))))
    out.append("")

    # [optimization]
    out.append("[optimization]")
    out.append(_emit_kv("steps", params.get("steps", 1000)))
    out.append(_emit_kv("lr", params.get("lr", 3e-3)))
    out.append(_emit_kv("lam_bc", params.get("lam_bc", 5.0)))
    out.append(_emit_kv("lam_warm", params.get("lam_warm", 200.0)))
    out.append(_emit_kv("log_every", params.get("log_every", max(1, int(params.get("steps", 1000)) // 20))))
    out.append(_emit_kv("mini_epoch", params.get("mini_epoch", 5)))
    out.append(_emit_kv("grad_clip_norm", params.get("grad_clip_norm", 1.0)))
    out.append(_emit_kv("weight_decay", params.get("weight_decay", 0.0)))
    out.append(_emit_kv("lr_warmup_steps", params.get("lr_warmup_steps", 0)))
    out.append(_emit_kv("lr_min_ratio", params.get("lr_min_ratio", 0.0)))
    out.append(_emit_kv("use_lookahead", params.get("use_lookahead", False)))
    out.append(_emit_kv("lookahead_sync_period", params.get("lookahead_sync_period", 5)))
    out.append(_emit_kv("lookahead_slow_step", params.get("lookahead_slow_step", 0.5)))
    out.append(_emit_kv("use_ema", params.get("use_ema", False)))
    out.append(_emit_kv("ema_decay", params.get("ema_decay", 0.999)))
    out.append(_emit_kv("ema_eval", params.get("ema_eval", True)))
    # Augmented Lagrangian (as configured)
    out.append(_emit_kv("use_augmented_lagrangian", params.get("use_augmented_lagrangian", False)))
    out.append(_emit_kv("al_rho", params.get("al_rho", 1.0)))
    out.append(_emit_kv("al_update_every", params.get("al_update_every", 10)))
    out.append(_emit_kv("al_clip", params.get("al_clip", 0.0)))
    # LBFGS top-level toggles/caps
    out.append(_emit_kv("lbfgs_steps", params.get("lbfgs_steps", 0)))
    out.append(_emit_kv("lbfgs_tol", params.get("lbfgs_tol", 1e-7)))
    out.append(_emit_kv("lbfgs_print_every", params.get("lbfgs_print_every", 25)))
    out.append(_emit_kv("lbfgs_interior", params.get("lbfgs_interior", 0)))
    out.append(_emit_kv("lbfgs_boundary", params.get("lbfgs_boundary", 0)))
    out.append(_emit_kv("lbfgs_weighting", params.get("lbfgs_weighting", "equal")))
    out.append("")

    # [optimization.lbfgs] — resolved flags (what LBFGS will actually use)
    out.append("[optimization.lbfgs]")
    # Prefer runtime flags (these are what the polish code actually consults)
    out.append(_emit_kv("l2", getattr(runtime, "lbfgs_l2", params.get("lbfgs_l2", 1e-8))))
    out.append(_emit_kv("include_zero_mean", getattr(runtime, "lbfgs_include_zero_mean",
                                                     params.get("lbfgs_include_zero_mean", True))))
    out.append(_emit_kv("include_aug_lagrangian", getattr(runtime, "lbfgs_include_aug_lagrangian",
                                                          params.get("lbfgs_include_aug_lagrangian", True))))
    out.append("")

    # [box]
    out.append("[box]")
    out.append(_emit_kv("zmin", params.get("box_zmin", -0.5)))
    out.append(_emit_kv("zmax", params.get("box_zmax",  0.5)))
    out.append(_emit_kv("points_total", params.get("box_points_total", 200000)))
    out.append(_emit_kv("seed", params.get("box_seed", 42)))
    out.append("")

    # [plot]
    out.append("[plot]")
    out.append(_emit_kv("cmap", params.get("plot_cmap", "viridis")))
    out.append(_emit_kv("figsize", list(params.get("figsize", (8.0, 4.5)))))
    out.append("")

    # [__derived] — live runtime snapshot for sanity (safe to ignore for strict diffs)
    out.append("[__derived]")
    out.append(_emit_kv("runtime_lam_bc_current", getattr(runtime, "lam_bc", params.get("lam_bc", 5.0))))
    out.append(_emit_kv("runtime_al_enabled", getattr(runtime, "al_enabled", params.get("use_augmented_lagrangian", False))))
    out.append(_emit_kv("runtime_al_lambda", getattr(runtime, "al_lambda", 0.0)))
    out.append(_emit_kv("runtime_al_rho", getattr(runtime, "al_rho", params.get("al_rho", 1.0))))
    out.append(_emit_kv("runtime_al_update_every", getattr(runtime, "al_update_every", params.get("al_update_every", 10))))
    out.append(_emit_kv("runtime_al_clip", getattr(runtime, "al_clip", params.get("al_clip", 0.0))))
    # box bounds if runtime computed them
    if hasattr(runtime, "box_bounds") and runtime.box_bounds:
        xmin, xmax, ymin, ymax, zmin, zmax = runtime.box_bounds
        out.append(_emit_kv("runtime_box_xmin", xmin))
        out.append(_emit_kv("runtime_box_xmax", xmax))
        out.append(_emit_kv("runtime_box_ymin", ymin))
        out.append(_emit_kv("runtime_box_ymax", ymax))
        out.append(_emit_kv("runtime_box_zmin", zmin))
        out.append(_emit_kv("runtime_box_zmax", zmax))
    # optional extras
    if extra:
        for k, v in extra.items():
            out.append(_emit_kv(k, v))
    out.append("")

    text = "\n".join(out)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[DUMP] Effective config written to: {os.path.abspath(path)}")
    return text
