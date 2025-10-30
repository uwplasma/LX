#!/usr/bin/env python3
"""
Hyperparameter optimization for LX (PINN Laplace on torus) using Optuna.

Design:
- Read the user's *nested* TOML with `_initialization.load_config`.
- For each trial, clone & edit that nested dict (tables: [model], [optimization], etc.).
- Force fast HPO toggles: disable plots/checkpoints/LBFGS unless the base file
  explicitly overrides via [hyperparameter_optimization].
- Dump the trial config to .hpo_runs/trial_XXXX.toml with `_initialization.dump_params_to_toml`.
- Call `main.main(config_path=trial_toml)` which returns (model, score).
- Report the scalar score to Optuna (minimize).

This file never touches the flattened dict produced by parse_config().
"""

from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, Tuple

import numpy as np
import optuna
from optuna.trial import Trial
import re
import time
from filelock import FileLock, Timeout
from contextlib import contextmanager
import shutil
from optuna.trial import TrialState
import warnings
from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# ---- local modules ----
from _initialization import load_config, dump_params_to_toml
from main import main as train_and_score  # returns (model, score)

# ------------------------ utilities ------------------------------------------

@contextmanager
def _file_lock(lock_path: Path, timeout_sec: float = 60.0):
    """
    Cross-platform advisory file lock. Creates <lock_path>.lock and locks it.
    Safe on Windows/macOS/Linux and okay on most cloud filesystems.
    """
    lock_file = str(lock_path) + ".lock"  # filelock expects a path, will create it
    lock = FileLock(lock_file)
    try:
        lock.acquire(timeout=timeout_sec)
        yield
    finally:
        try:
            lock.release()
        except Exception:
            pass

def _save_best_callback(hpo_root: Path, keep_mode: str = "all"):
    """
    On each COMPLETE trial:
      - If it’s the new best, copy its TOML to best_so_far.toml (+ meta).
      - If keep_mode == 'best', delete all non-best, non-running trial_* dirs.
    """
    best_path = hpo_root / "best_so_far.toml"
    best_meta = hpo_root / "best_so_far.txt"
    lock_file = hpo_root / ".cleanup.lock"

    def _cb(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if trial.state != TrialState.COMPLETE:
            return

        # Only act when *this* trial is the best.
        if study.best_trial.number != trial.number:
            return

        # 1) Update rolling best TOML + meta
        src = _trial_dir(hpo_root, trial.number) / "input.trial.toml"
        # Lock around best_so_far writes to avoid concurrent clobbering
        with _file_lock(lock_file):
            if src.exists():
                shutil.copy2(src, best_path)
                with open(best_meta, "w") as f:
                    f.write(
                        f"best_trial={trial.number}\n"
                        f"value={trial.value}\n"
                        f"path={src}\n"
                    )
                print(f"[HPO] Updated best: trial {trial.number} -> {best_path}")

        # 2) Optional: prune other trial_* dirs safely across processes
        if keep_mode != "best":
            return

        running_ids = {
            t.number for t in study.get_trials(deepcopy=False)
            if t.state in (TrialState.RUNNING, TrialState.WAITING)
        }

        pat = re.compile(r"^trial_(\d{4})$")
        with _file_lock(lock_file):
            for p in hpo_root.iterdir():
                if not p.is_dir():
                    continue
                m = pat.match(p.name)
                if not m:
                    continue
                num = int(m.group(1))
                if num == trial.number or num in running_ids:
                    continue
                try:
                    shutil.rmtree(p)
                    print(f"[HPO] Deleted {p} (non-best, not running)")
                except Exception as e:
                    print(f"[HPO] Warning: failed to delete {p}: {e}")

    return _cb

def _ensure_table(d: Dict[str, Any], name: str) -> Dict[str, Any]:
    if name not in d or not isinstance(d[name], dict):
        d[name] = {}
    return d[name]

def _set_if_absent(tbl: Dict[str, Any], key: str, value: Any) -> None:
    if key not in tbl:
        tbl[key] = value

def _trial_dir(root: Path, number: int) -> Path:
    p = root / f"trial_{number:04d}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _is_finite(x: float) -> bool:
    return (x is not None) and np.isfinite(float(x))

# ------------------------ search space ---------------------------------------

def suggest_params(trial: Trial, base_raw_cfg: Dict[str, Any], *, vary_steps: bool = False) -> Dict[str, Any]:
    """
    Take a *nested* TOML dict (as loaded by load_config),
    return a NEW nested dict with trial-suggested values filled in.
    """
    cfg = deepcopy(base_raw_cfg)

    # Ensure tables exist
    model_tbl = _ensure_table(cfg, "model")
    opt_tbl   = _ensure_table(cfg, "optimization")
    samp_tbl  = _ensure_table(cfg, "sampling")
    reg_tbl   = _ensure_table(cfg, "regularization")
    hpo_tbl   = _ensure_table(cfg, "hyperparameter_optimization")

    # --- Speed/robustness toggles for HPO (can be overridden in input.toml) ---
    _set_if_absent(hpo_tbl, "hpo_disable_plots", True)
    _set_if_absent(hpo_tbl, "hpo_disable_ckpt",  True)
    _set_if_absent(hpo_tbl, "hpo_disable_lbfgs", True)

    # ---------------- Model space ----------------
    act = trial.suggest_categorical("model.activation", ["tanh", "relu", "gelu", "silu", "softplus", "sin"])
    model_tbl["activation"] = act

    siren = trial.suggest_categorical("model.siren", [True, False])
    model_tbl["siren"] = siren
    if siren or act in ("sin", "sine"):
        model_tbl["siren_omega0"] = trial.suggest_float("model.siren_omega0", 5.0, 60.0, log=True)

    use_fourier = trial.suggest_categorical("model.use_fourier", [True, False])
    model_tbl["use_fourier"] = use_fourier
    if use_fourier:
        pool = [1.0, 2.0, 4.0, 8.0, 16.0]
        k = trial.suggest_int("model.ff_k", 2, 5)   # scalar
        model_tbl["fourier_bands"] = pool[:k]       # build list from scalar
        model_tbl["fourier_scale"] = 2.0 * math.pi

    # hidden sizes: choose width + depth pattern
    depth = trial.suggest_int("model.depth", 2, 5)
    width = trial.suggest_categorical("model.width", [32, 48, 64, 96, 128])
    model_tbl["hidden_sizes"] = [int(width)] * int(depth)

    # pin R0_for_fourier to geometry.R0 unless user set it
    geom_tbl = _ensure_table(cfg, "geometry")
    _set_if_absent(model_tbl, "R0_for_fourier", float(geom_tbl.get("R0", 1.0)))

    # --------------- Optimization space ---------------
    # Steps: keep modest for HPO; your main uses cosine schedule anyway.
    # ---- steps ----
    if vary_steps:
        opt_tbl["steps"] = int(trial.suggest_int("opt.steps", 600, 1800, step=300))
    else:
        # keep base value (or fall back to 600 if absent)
        opt_tbl["steps"] = int(base_raw_cfg.get("optimization", {}).get("steps", 600))

    # Learning rate (AdamW)
    opt_tbl["lr"] = float(trial.suggest_float("opt.lr", 5e-4, 1e-2, log=True))

    # Boundary weight λ_bc and warm start level
    opt_tbl["lam_bc"]  = float(trial.suggest_float("opt.lam_bc", 1.0, 20.0, log=True))
    opt_tbl["lam_warm"] = float(trial.suggest_float("opt.lam_warm", 5.0, 200.0, log=True))

    # AdamW extras
    opt_tbl["grad_clip_norm"] = float(trial.suggest_float("opt.grad_clip_norm", 0.5, 5.0, log=True))
    # Weight decay: allow an exact zero OR a positive log-uniform value
    if trial.suggest_categorical("opt.weight_decay_is_zero", [True, False]):
        opt_tbl["weight_decay"] = 0.0
    else:
        opt_tbl["weight_decay"] = float(trial.suggest_float("opt.weight_decay_pos", 1e-8, 1e-2, log=True))

    # LR warmup & cosine floor
    local_steps = int(opt_tbl["steps"])  # 300 from your base if --vary-steps is not passed
    # LR warmup & cosine floor (use local_steps)
    opt_tbl["lr_warmup_steps"] = int(
        trial.suggest_int(
            "opt.lr_warmup_steps",
            0,
            max(0, local_steps // 4),
            step=max(1, local_steps // 12) or 1,
        )
    )
    opt_tbl["lr_min_ratio"] = float(trial.suggest_float("opt.lr_min_ratio", 0.0, 0.2))

    # EMA and Lookahead (often help stability)
    opt_tbl["use_ema"]   = bool(trial.suggest_categorical("opt.use_ema", [True, False]))
    if opt_tbl["use_ema"]:
        opt_tbl["ema_decay"] = float(trial.suggest_float("opt.ema_decay", 0.995, 0.9999))
        opt_tbl["ema_eval"]  = True
    else:
        opt_tbl["ema_eval"]  = False

    opt_tbl["use_lookahead"] = bool(trial.suggest_categorical("opt.use_lookahead", [False, True]))
    if opt_tbl["use_lookahead"]:
        opt_tbl["lookahead_sync_period"] = int(trial.suggest_int("opt.lookahead_sync_period", 3, 10))
        opt_tbl["lookahead_slow_step"]   = float(trial.suggest_float("opt.lookahead_slow_step", 0.3, 0.8))

    # Augmented Lagrangian toggles
    opt_tbl["use_augmented_lagrangian"] = bool(trial.suggest_categorical("opt.use_al", [False, True]))
    if opt_tbl["use_augmented_lagrangian"]:
        opt_tbl["al_rho"]          = float(trial.suggest_float("opt.al_rho", 0.1, 5.0, log=True))
        opt_tbl["al_update_every"] = int(trial.suggest_int("opt.al_update_every", 5, 30))
        opt_tbl["al_clip"]         = float(trial.suggest_float("opt.al_clip", 0.0, 10.0))

    # LBFGS polish OFF during HPO by default — keeps trials fast/deterministic.
    # You already have hpo_disable_lbfgs that your main honors; leave lbfgs_steps as-is.
    opt_tbl["lbfgs_steps"]        = int(0)
    opt_tbl["lbfgs_tol"]          = float(base_raw_cfg.get("optimization", {}).get("lbfgs_tol", 1e-7))
    opt_tbl["lbfgs_print_every"]  = int(base_raw_cfg.get("optimization", {}).get("lbfgs_print_every", 25))
    opt_tbl["lbfgs_interior"]     = int(0)
    opt_tbl["lbfgs_boundary"]     = int(0)
    opt_tbl["lbfgs_weighting"]    = "equal"
    _ensure_table(opt_tbl, "lbfgs")["l2"] = float(0.0)

    # --------------- Sampling/regularization ---------------
    # Keep your dataset sizes but allow mild tuning of boundary presample multiplier.
    samp_tbl["bdry_presample_mult"] = int(trial.suggest_int("sampling.bdry_presample_mult", 8, 32, step=8))
    reg_tbl["zero_mean_weight"]     = float(trial.suggest_float("regularization.zero_mean_weight", 0.0, 1.0))

    # --------------- Multi-valued term ---------------
    mv_tbl = _ensure_table(cfg, "multi_valued")
    mv_tbl["kappa"] = float(trial.suggest_float("multi_valued.kappa", 0.0, 1.0))

    # --------------- RNG seed per trial ---------------
    samp_tbl["rng_seed"] = int(trial.suggest_int("sampling.rng_seed", 0, 10_000))

    return cfg

# ------------------------ objective ------------------------------------------

def objective(trial: Trial, base_raw_cfg: Dict[str, Any], vary_steps: bool = False) -> float:
    # Create a working dir per trial
    out_root = Path(".hpo_runs")
    out_root.mkdir(exist_ok=True)
    tdir = _trial_dir(out_root, trial.number)

    # Suggest a nested TOML dict and write it
    trial_cfg = suggest_params(trial, base_raw_cfg, vary_steps=vary_steps)

    # Ensure HPO toggles are set unless the user explicitly disabled them
    hpo_tbl = _ensure_table(trial_cfg, "hyperparameter_optimization")
    hpo_tbl.setdefault("hpo_disable_plots", True)
    hpo_tbl.setdefault("hpo_disable_ckpt",  True)
    hpo_tbl.setdefault("hpo_disable_lbfgs", True)

    trial_toml = tdir / "input.trial.toml"
    dump_params_to_toml(trial_toml, trial_cfg)

    # Optional: reduce XLA startup chatter per trial
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # Run training; your main returns (model, final_score)
    try:
        _, score = train_and_score(config_path=str(trial_toml))
    except Exception as e:
        # Let Optuna record the failure
        raise

    # Guard against NaN/inf
    if not _is_finite(score):
        raise optuna.TrialPruned(f"Non-finite score: {score}")

    # Optuna minimizes by default if we set direction='minimize'
    return float(score)

# ------------------------ runner ---------------------------------------------

def run_study(
    config_path: str,
    n_trials: int,
    storage: str | None = None,
    study_name: str = "LX-PINN-HPO",
    sampler_name: str = "tpe",
    pruner_name: str = "median",
    timeout: int | None = None,
    direction: str = "minimize",
    n_jobs: int = 4,
    vary_steps: bool = False,
    keep_trials: str = "best",
) -> optuna.study.Study:

    # Load the *nested* base TOML (not flattened!)
    base_raw_cfg = load_config(config_path)

    # Sampler
    if sampler_name.lower() == "tpe":
        sampler = optuna.samplers.TPESampler(
            multivariate=True, group=True, seed=42, n_startup_trials=10
        )
    elif sampler_name.lower() == "qmc":
        sampler = optuna.samplers.QMCSampler(seed=42)
    else:
        sampler = optuna.samplers.TPESampler(seed=42)

    # Pruner
    if pruner_name.lower() == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    elif pruner_name.lower() == "nop":
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.MedianPruner()

    hpo_root = Path(".hpo_runs")
    hpo_root.mkdir(exist_ok=True)
    
    def _objective(tr):
        return objective(tr, base_raw_cfg, vary_steps=vary_steps)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        lambda t: _objective(t),
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True,
        catch=(Exception,),
        n_jobs=n_jobs,  # parallelization
        callbacks=[_save_best_callback(hpo_root, keep_mode=keep_trials)],
    )
    return study

# ------------------------ CLI ------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(description="Optuna HPO for LX")
    ap.add_argument("-c", "--config", type=str, default="input.toml", help="Base TOML config.")
    ap.add_argument("-n", "--n-trials", type=int, default=30, help="Number of trials.")
    ap.add_argument("--storage", type=str, default=None, help="Optuna storage, e.g. sqlite:///lx_optuna.db")
    ap.add_argument("--study-name", type=str, default="LX-PINN-HPO")
    ap.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "qmc"])
    ap.add_argument("--pruner", type=str, default="median", choices=["median", "nop"])
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=4, help="Parallel trials (threads). Default: 4")
    ap.add_argument("--vary-steps", action="store_true", help="Let Optuna tune optimization.steps. Default: keep base file value.")
    ap.add_argument( "--keep-trials", choices=["all", "best"], default="best",
        help='Keep "all" trial_* folders or only the "best" (default).')
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    study = run_study(
        config_path=args.config,
        n_trials=args.n_trials,
        storage=args.storage,
        study_name=args.study_name,
        sampler_name=args.sampler,
        pruner_name=args.pruner,
        timeout=args.timeout,
        direction="minimize",
        n_jobs=args.n_jobs,
        vary_steps=args.vary_steps,
        keep_trials=args.keep_trials,
    )

    # Only consider COMPLETED trials
    completed = [t for t in study.get_trials(deepcopy=False)
                 if t.state == optuna.trial.TrialState.COMPLETE]

    print("\n=== Best Trial ===")
    if not completed:
        print("No successful trials yet (all failed/pruned). "
              "Fix errors or increase n_trials and try again.")
    else:
        print(f"Number: {study.best_trial.number}")
        print(f"Value : {study.best_value:.6e}")
        print("Params:")
        for k, v in study.best_trial.params.items():
            print(f"  {k}: {v}")

    # at top of __main__ after study is returned
    hpo_root = Path(".hpo_runs")
    lock_file = hpo_root / ".cleanup.lock"

    best_dump = hpo_root / "best_final.toml"
    try:
        src = _trial_dir(hpo_root, study.best_trial.number) / "input.trial.toml"
        if src.exists():
            with _file_lock(lock_file):
                shutil.copy2(src, best_dump)
            print(f"[HPO] Best final copied to: {best_dump}")
    except Exception:
        pass
    
    # Final sweep using the same safe logic (lock + skip RUNNING/WAITING)
    try:
        if completed:
            _save_best_callback(Path(".hpo_runs"), keep_mode=args.keep_trials)(study, study.best_trial)
    except Exception as e:
        print(f"[HPO] Final sweep skipped due to error: {e}")