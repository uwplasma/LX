#!/usr/bin/env python3
"""
PINN for Laplace's equation ∇²u = 0 inside a (possibly non-axisymmetric) torus
in Cartesian coordinates, with a soft Neumann boundary condition n·∇u ≈ 0.

u_total(x,y,z) = u_mv(x,y,z) + u_nn(x,y,z)
  - u_mv is a multi-valued piece: kappa * atan2(y,x) / R0
  - u_nn is a single-valued MLP built with Equinox

Objective (least squares):
  L = ⟨(∇² u_total)^2⟩_interior  +  λ_bc ⟨(n·∇u_total)^2⟩_boundary

End of training:
  - Prints diagnostics (residuals, grads)
  - Plots |∇u| on the boundary surface (θ×φ grid)

One may replace the surface constructor with any surface provider that returns:
  - boundary points P ∈ ℝ^{Nb×3}
  - outward unit normals N ∈ ℝ^{Nb×3}
For the example we build a torus with a(φ) = a0 + a1 cos(N_harm φ).

Change 10.27.2025: single-surface training OR multi-surface pretraining for
Laplace's equation ∇²u = 0 with Neumann BC (n·∇u = 0 on the surface).

Modes (from input.toml):
  [surfaces]
  mode = "single"  -> one surface (as before) using [geometry] params
  mode = "torus"   -> multi-surface pretraining over [[surfaces.torus_list]]

In all cases, interior points are sampled deterministically from a fixed box,
then masked to the current surface (never train on outside points).
"""


# C. Optional architecture toggles (one-line config changes)
# Activation: activation = "sine" (a SIREN-style MLP) often does better for PDEs with oscillatory/tangential structure. You already allow "sin" in _initialization.py mapping; use it for pretraining.
# Fourier features: prepend concat([xyz, sin(2πBx), cos(2πBx)]) with a small Gaussian B. This is ~15 lines in PotentialMLP and can drastically accelerate convergence on geometric families.
# LBFGS polish: After AdamW, run 200–500 LBFGS steps (scipy LBFGS on filtered params). PINN papers routinely report big gains.

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import optax
import equinox as eqx
import matplotlib.pyplot as plt
from jaxopt import LBFGS

# ---------------- local imports ----------------
from _initialization import build_params_from_path
from _plotting import fix_matplotlib_3d, plot_surface_with_vectors_ax, draw_box_edges
from _geometry import (
    surface_points_and_normals,
    fixed_box_points,
    select_interior_from_fixed,
)
from _network_and_loss import (
    PotentialMLP,
    train_step,
    eval_full,
    load_model_if_exists,
    save_model,
    debug_stats,
    train_step_many,
    _full_objective_like_training
)
from _physics import eval_on_boundary, grad_u_total_batch
from _state import runtime
from _multisurface import build_torus_family
from _input_output import dump_effective_toml

# =============================================================================
# ========================== GLOBAL / PARAM BINDING ===========================
# =============================================================================
CHECKPOINT_PATH: Path
BATCH_IN: int
BATCH_BDRY: int
R0: float
a0: float
a1: float
N_harm: int
kappa: float
N_in: int
N_bdry_theta: int
N_bdry_phi: int
rng_seed: int
MLP_HIDDEN_SIZES: tuple
MLP_ACT: Any
steps: int
lr: float
lam_bc: float
PLOT_CMAP: str
FIGSIZE: Tuple[float, float]
LOG_EVERY: int
MINI_EPOCH: int
LAM_WARM: float
LBFGS_STEPS: int
LBFGS_TOL: float
LBFGS_PRINT_EVERY: int
LBFGS_IN: int
LBFGS_BDRY: int
LBFGS_WEIGHTING: str

# box
box_zmin: float
box_zmax: float
box_points_total: int
box_seed: int

# surfaces config (raw dict from TOML; parsed here)
surfaces_cfg: dict

def _finite_or_big(x, big=1e30):
    return jnp.nan_to_num(x, nan=big, posinf=big, neginf=big)

def _take_first_n(x: jnp.ndarray, n: int) -> jnp.ndarray:
    """Deterministic truncation: if n<=0 or n>=len(x), returns x."""
    if n <= 0 or n >= int(x.shape[0]):
        return x
    return x[:int(n)]

def _stride_take(x: jnp.ndarray, k: int) -> jnp.ndarray:
    """Deterministic approx-uniform sub-sampling without randomness."""
    n = int(x.shape[0])
    if k <= 0 or k >= n:
        return x
    stride = max(n // k, 1)
    return x[::stride][:k]

def _lambda_bc_schedule(it, T):
    # Start very high and decay to lam_bc
    warm, end = LAM_WARM, lam_bc
    # cosine schedule in [0, π]
    cosfac = 0.5 * (1 + jnp.cos(jnp.pi * jnp.clip(it / T, 0., 1.)))
    return float(end + (warm - end) * cosfac)

def _ema_update(old, new, decay):
    return jax.tree_util.tree_map(lambda o, n: decay * o + (1.0 - decay) * n, old, new)

def _lookahead_sync(slow, fast, alpha):
    # slow ← slow + α (fast - slow); then fast ← slow
    new_slow = jax.tree_util.tree_map(lambda s, f: s + alpha * (f - s), slow, fast)
    return new_slow

def _apply_params(params: Dict[str, Any]) -> None:
    global CHECKPOINT_PATH, BATCH_IN, BATCH_BDRY, R0, a0, a1, N_harm, kappa
    global N_in, N_bdry_theta, N_bdry_phi, rng_seed
    global MLP_HIDDEN_SIZES, MLP_ACT
    global steps, lr, lam_bc
    global PLOT_CMAP, FIGSIZE
    global box_zmin, box_zmax, box_points_total, box_seed
    global surfaces_cfg

    CHECKPOINT_PATH = Path(params["checkpoint_path"])          # type: ignore
    BATCH_IN        = int(params["batch_interior"])            # type: ignore
    BATCH_BDRY      = int(params["batch_boundary"])            # type: ignore

    R0     = float(params["R0"])                               # type: ignore
    a0     = float(params["a0"])                               # type: ignore
    a1     = float(params["a1"])                               # type: ignore
    N_harm = int(params["N_harm"])                             # type: ignore

    kappa  = float(params["kappa"])                            # type: ignore

    N_in          = int(params["N_in"])                        # type: ignore
    N_bdry_theta  = int(params["N_bdry_theta"])                # type: ignore
    N_bdry_phi    = int(params["N_bdry_phi"])                  # type: ignore
    rng_seed      = int(params["rng_seed"])                    # type: ignore

    MLP_HIDDEN_SIZES = tuple(params["mlp_hidden_sizes"])       # type: ignore
    MLP_ACT          = params["mlp_activation"]                # type: ignore

    steps  = int(params["steps"])                              # type: ignore
    lr     = float(params["lr"])                               # type: ignore
    lam_bc = float(params["lam_bc"])                           # type: ignore

    global LOG_EVERY, MINI_EPOCH
    LOG_EVERY  = int(params.get("log_every", max(1, steps // 20)))   # e.g., 250
    MINI_EPOCH = int(params.get("mini_epoch", 5))                    # K per surface

    PLOT_CMAP = str(params["plot_cmap"])                       # type: ignore
    FIGSIZE   = tuple(params["figsize"])                       # type: ignore

    # Box controls from TOML
    box_zmin        = float(params.get("box_zmin", -0.5))      # type: ignore
    box_zmax        = float(params.get("box_zmax",  0.5))      # type: ignore
    box_points_total= int(params.get("box_points_total", 200_000))  # type: ignore
    box_seed        = int(params.get("box_seed", 42))          # type: ignore

    # Raw surfaces config blob (parsed in main)
    surfaces_cfg    = params.get("surfaces_cfg", {})           # type: ignore

    global LAM_WARM
    LAM_WARM = float(params.get("lam_warm", 200.0))

    global LBFGS_STEPS, LBFGS_TOL, LBFGS_PRINT_EVERY
    LBFGS_STEPS = int(params.get("lbfgs_steps", 0))
    LBFGS_TOL = float(params.get("lbfgs_tol", 1e-7))
    LBFGS_PRINT_EVERY = int(params.get("lbfgs_print_every", 25))

    global LBFGS_IN, LBFGS_BDRY, LBFGS_WEIGHTING
    LBFGS_IN        = int(params.get("lbfgs_interior", 0))
    LBFGS_BDRY      = int(params.get("lbfgs_boundary", 0))
    LBFGS_WEIGHTING = str(params.get("lbfgs_weighting", "equal"))

    # zero-mean weight & boundary presample multiplier & optimizer knobs
    global ZERO_MEAN_WEIGHT_CONF, BDRY_PRESAMPLE_MULT, GRAD_CLIP_NORM, WEIGHT_DECAY
    ZERO_MEAN_WEIGHT_CONF = float(params.get("zero_mean_weight", 0.1))
    BDRY_PRESAMPLE_MULT   = int(params.get("bdry_presample_mult", 16))
    GRAD_CLIP_NORM        = float(params.get("grad_clip_norm", 1.0))
    WEIGHT_DECAY          = float(params.get("weight_decay", 0.0))

# =============================================================================
# ================================== MAIN =====================================
# =============================================================================
def main(config_path: str = "input.toml"):

    # Load config dict and bind globals
    params = build_params_from_path(config_path)
    _apply_params(params)

    # >>> Set runtime state ONCE so all modules see it <<<
    runtime.R0 = R0
    runtime.a0 = a0
    runtime.a1 = a1
    runtime.N_harm = N_harm
    runtime.kappa = kappa
    runtime.BATCH_IN = BATCH_IN
    runtime.BATCH_BDRY = BATCH_BDRY
    runtime.lam_bc = lam_bc
    runtime.zero_mean_weight = ZERO_MEAN_WEIGHT_CONF
    runtime.bdry_presample_mult = int(params.get("bdry_presample_mult", 16))

    # Augmented Lagrangian config
    runtime.al_enabled = bool(params.get("use_augmented_lagrangian", False))
    runtime.al_lambda = 0.0  # reset each run (or load from ckpt if you want stateful)
    runtime.al_rho = float(params.get("al_rho", 1.0))
    runtime.al_update_every = int(params.get("al_update_every", 10))
    runtime.al_clip = float(params.get("al_clip", 0.0))

    # ---- LBFGS flags into runtime so helpers can read them safely ----
    # Read the flattened keys produced by parse_config()
    runtime.lbfgs_include_zero_mean = bool(params.get("lbfgs_include_zero_mean", True))
    runtime.lbfgs_include_aug_lagrangian = bool(params.get("lbfgs_include_aug_lagrangian", True))
    runtime.lbfgs_l2 = float(params.get("lbfgs_l2", 1e-8))

    dump_effective_toml(
        path="effective_config.dump.toml",
        params=params,
        runtime=runtime,
        extra={"note": "Snapshot taken before training and LBFGS polish."}
    )

    act_name = getattr(MLP_ACT, "__name__", str(MLP_ACT))
    print("=== PINN Laplace (single or multi-surface) ===")
    print(f"Network: hidden={MLP_HIDDEN_SIZES}, act={act_name}, optimizer=AdamW(lr={lr})")
    print(f"Training steps: {steps}, λ_bc={lam_bc}")
    sys.stdout.flush()

    # RNG (for model init and misc)
    key = random.PRNGKey(rng_seed)
    key, k_model = random.split(key)

    # --- training history (Adam/optax) ---
    adam_loss_hist = []
    adam_lin_hist  = []
    adam_lbc_hist  = []
    adam_lap_rms_hist = []
    adam_nbc_rms_hist = []
    lbfgs_hist = []
    adam_eval_full_hist = []
    adam_train_like_hist = []
    lbfgs_eval_full_hist = []
    lbfgs_train_like_hist = []

    # ---------------------- Build fixed box + candidate points -----------------
    # x,y box computed from geometry envelope; z from TOML
    Lxy = float(R0 + (abs(a1) + a0))
    xmin, xmax = -Lxy, Lxy
    ymin, ymax = -Lxy, Lxy
    zmin, zmax = float(box_zmin), float(box_zmax)
    runtime.box_bounds = (xmin, xmax, ymin, ymax, zmin, zmax)

    print(f"[BOX] Using fixed sampling box:")
    print(f"      x∈[{xmin:+.3f},{xmax:+.3f}]  y∈[{ymin:+.3f},{ymax:+.3f}]  z∈[{zmin:+.3f},{zmax:+.3f}]")
    print(f"[BOX] Generating {box_points_total} deterministic points (seed={box_seed})")
    P_box = fixed_box_points(box_seed, box_points_total, runtime.box_bounds)   # [M,3]

    if runtime.al_enabled:
        print(f"[AL] enabled: rho={runtime.al_rho}, update_every={runtime.al_update_every}, clip={runtime.al_clip or 'none'}")

    use_ema   = bool(params.get("use_ema", False))
    ema_decay = float(params.get("ema_decay", 0.999))
    ema_eval  = bool(params.get("ema_eval", True))
    if bool(params.get("use_lookahead", False)):
        print(f"[OPT] Lookahead: k={int(params.get('lookahead_sync_period',5))}, alpha={float(params.get('lookahead_slow_step',0.5))}")
    if use_ema:
        print(f"[OPT] EMA: decay={ema_decay}, eval_with_ema={ema_eval}")

    # ---------------------- Build model + optimizer ----------------------------
    use_siren     = bool(params.get("siren", False))
    siren_omega0  = float(params.get("siren_omega0", 30.0))

    if use_siren and (MLP_ACT.__name__ == "sin" or MLP_ACT is jnp.sin):
        from _network_and_loss import SirenMLP
        model = SirenMLP(k_model, in_size=3, out_size=1,
                         widths=MLP_HIDDEN_SIZES, omega0=siren_omega0,
                         use_fourier=bool(params.get("use_fourier", False)),
                         fourier_bands=tuple(params.get("fourier_bands", ())),
                         fourier_scale=float(params.get("fourier_scale", 2*jnp.pi)),
                         R0_for_fourier=float(params.get("R0_for_fourier", R0)))
    else:
        model = PotentialMLP(k_model, hidden_sizes=MLP_HIDDEN_SIZES, act=MLP_ACT,
                             use_fourier=bool(params.get("use_fourier", False)),
                             fourier_bands=tuple(params.get("fourier_bands", ())),
                             fourier_scale=float(params.get("fourier_scale", 2*jnp.pi)),
                             R0_for_fourier=float(params.get("R0_for_fourier", R0)))

    model = load_model_if_exists(model, CHECKPOINT_PATH)

    # --- LR schedule with optional warmup + cosine decay floor ---
    warmup = int(params.get("lr_warmup_steps", 0))
    min_ratio = float(params.get("lr_min_ratio", 0.0))  # e.g., 0.05
    sched_main = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=steps, alpha=min_ratio
    )
    if warmup > 0:
        sched = optax.join_schedules(
            schedules=[optax.linear_schedule(init_value=lr*0.1, end_value=lr, transition_steps=warmup),
                       sched_main],
            boundaries=[warmup]
        )
    else:
        sched = sched_main

    clip_norm = float(params.get("grad_clip_norm", 1.0))
    wd = float(params.get("weight_decay", 0.0))

    base_opt = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate=sched, weight_decay=wd),
    )

    optimizer = base_opt
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    lookahead_enabled   = bool(params.get("use_lookahead", False))
    lookahead_k         = int(params.get("lookahead_sync_period", 5))
    lookahead_alpha     = float(params.get("lookahead_slow_step", 0.5))

    # slow weights (trainable leaves only)
    params_f_init, _ = eqx.partition(model, eqx.is_inexact_array)
    slow_f = jax.tree_util.tree_map(lambda x: x.copy(), params_f_init) if lookahead_enabled else None
    la_step = 0  # global counter

    # Filter inexact (trainable) leaves
    params_f, params_s = eqx.partition(model, eqx.is_inexact_array)
    ema_f = jax.tree_util.tree_map(lambda x: x.copy(), params_f) if use_ema else None

    # ---------------------- Mode selection ------------------------------------
    mode = str(surfaces_cfg.get("mode", "single")).lower().strip()
    print(f"[MODE] {mode}")

    if mode == "single":
        # ===== SINGLE SURFACE (as before) =====
        # Build boundary grid for the current runtime geometry:
        P_bdry, N_bdry, Xg, Yg, Zg = surface_points_and_normals(N_bdry_theta, N_bdry_phi)

        # Robust outwardness check (centroid test)
        P_grid = jnp.stack([Xg, Yg, Zg], axis=-1)                              # (nθ,nφ,3)
        centroid = jnp.mean(P_grid.reshape(-1, 3), axis=0, keepdims=True)      # (1,3)
        Rvec = P_grid - centroid
        Rhat = Rvec / (jnp.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-12)
        Nhat_grid = N_bdry.reshape(Xg.shape + (3,))
        mean_out = float(jnp.mean(jnp.sum(Nhat_grid * Rhat, axis=-1)))
        print(f"[DEBUG] Mean outwardness of normals: {mean_out:+.4f} (positive expected)")
        if mean_out < 0:
            N_bdry = -N_bdry
            Nhat_grid = -Nhat_grid
            print("[DEBUG] Normals flipped to ensure outward orientation.")

        # Deterministic interior points from fixed box, masked to this surface
        P_in = select_interior_from_fixed(P_box, N_in)
        Ni = int(P_in.shape[0])
        if Ni < N_in:
            print(f"[WARN] Only {Ni} interior points found in box. Increase [box].points_total.")
        else:
            P_in = P_in[:N_in]; Ni = N_in

        # Initial diagnostics
        (L0, (Lin0, Lbc0, lap0, n0)) = eval_full(model, P_in, P_bdry, N_bdry)
        print(f"[INIT] loss={float(L0):.6e}  lap={float(Lin0):.6e}  bc={float(Lbc0):.6e}")

        # Pre-compute initial boundary grad (for side-by-side plot)
        Gvec_init, Gmag_init, Nhat_grid = eval_on_boundary(model, P_bdry, N_bdry, Xg, Yg, Zg)

        # Train
        key_train = random.PRNGKey(1234)
        log_every = LOG_EVERY
        for it in range(1, steps + 1):
            cur_lam = _lambda_bc_schedule(it, steps)
            runtime.lam_bc = float(cur_lam)   # update runtime so loss sees it
            key_train, subkey = random.split(key_train)
            model, opt_state, L, (Lin, Lbc, lap_res, nres, mean_u, c_bc_mean), gnorm, grads = train_step(
                model, opt_state, optimizer, P_in, P_bdry, N_bdry, subkey
            )
            # record per-iter history (Adam)
            adam_loss_hist.append(float(L))
            adam_lin_hist.append(float(Lin))
            adam_lbc_hist.append(float(Lbc))
            adam_lap_rms_hist.append(float(jnp.sqrt(jnp.mean(lap_res**2))))
            adam_nbc_rms_hist.append(float(jnp.sqrt(jnp.mean(nres**2))))
            adam_eval_full_hist.append(float(Lin + runtime.lam_bc * Lbc))
            adam_train_like_hist.append(float(_full_objective_like_training(model, P_in, P_bdry, N_bdry)))
            if use_ema:
                mf, ms = eqx.partition(model, eqx.is_inexact_array)
                ema_f = _ema_update(ema_f, mf, ema_decay)
            if lookahead_enabled:
                la_step += 1
                if (la_step % lookahead_k) == 0:
                    mf, ms = eqx.partition(model, eqx.is_inexact_array)
                    slow_f = _lookahead_sync(slow_f, mf, lookahead_alpha)
                    # overwrite model fast weights with synced slow weights
                    model = eqx.combine(slow_f, ms)
            # --- Augmented Lagrangian update (outer loop) ---
            if runtime.al_enabled and ((it % runtime.al_update_every) == 0):
                # Gradient-ascent on λ: λ ← λ + ρ * c
                runtime.al_lambda = float(runtime.al_lambda + runtime.al_rho * float(c_bc_mean))
                # optional clipping
                if runtime.al_clip and (runtime.al_clip > 0.0):
                    runtime.al_lambda = float(jnp.clip(runtime.al_lambda, -runtime.al_clip, runtime.al_clip))
            if (it % log_every) == 0 or it == 1:
                stats = debug_stats(model, grads, lap_res, nres)
                # compute |∇u| on THIS boundary
                if use_ema and ema_eval:
                    # evaluate with EMA params
                    mf, ms = eqx.partition(model, eqx.is_inexact_array)
                    model_eval = eqx.combine(ema_f, ms)
                else:
                    model_eval = model
                gmag = jnp.linalg.norm(grad_u_total_batch(model_eval, P_bdry), axis=-1)
                mean_g = float(jnp.mean(gmag))
                print(
                    f"[{it:5d}] loss={float(L):.6e}  lin={float(Lin):.3e}  lbc={float(Lbc):.3e}  "
                    f"|lap|_rms={stats['lap_rms']:.3e}  "
                    f"|n·∇u|_rms={stats['nbc_rms']:.3e}  "
                    f"||g||={stats['grad_L2']:.3e}  mean|∇u|={mean_g:.3e}  mean(u)={float(mean_u):.3e}  "
                    f"λ_bc={runtime.lam_bc:.2f}  "
                    f"{'AL ' if runtime.al_enabled else ''}c={float(c_bc_mean):+.3e} λ_AL={runtime.al_lambda:+.3e}  "
                )

        model_to_save = model
        if use_ema and ema_eval:
            mf, ms = eqx.partition(model, eqx.is_inexact_array)
            model_to_save = eqx.combine(ema_f, ms)

        # Optional LBFGS polish on this single surface
        runtime._lbfgs_eval_full_hist_sink = lbfgs_eval_full_hist
        if LBFGS_STEPS > 0:
            model_to_save = _lbfgs_polish(
                model_to_save, P_in, P_bdry, N_bdry,
                steps=LBFGS_STEPS, tol=LBFGS_TOL,
                print_every=LBFGS_PRINT_EVERY, label="single",
                history=lbfgs_hist,
            )
        runtime._lbfgs_eval_full_hist_sink = None

        save_model(model_to_save, CHECKPOINT_PATH)

        # Final diagnostics
        (Lf, (Linf, Lbcf, lapf, nf)) = eval_full(model, P_in, P_bdry, N_bdry)
        print(f"[FINAL] loss={float(Lf):.6e}  lap={float(Linf):.6e}  bc={float(Lbcf):.6e}")
        print(f"[FINAL] |lap|_rms={float(jnp.sqrt(jnp.mean(lapf**2))):.3e}  "
              f"|n·∇u|_rms={float(jnp.sqrt(jnp.mean(nf**2))):.3e}")
        
        nbc_rms  = float(jnp.sqrt(jnp.mean(nf**2)))
        lap_rms  = float(jnp.sqrt(jnp.mean(lapf**2)))
        nbc_max  = float(jnp.max(jnp.abs(nf)))
        lap_max  = float(jnp.max(jnp.abs(lapf)))
        print(f"[CHECK] boundary n·∇u  rms={nbc_rms:.3e}  max={nbc_max:.3e}")
        print(f"[CHECK] interior Δu     rms={lap_rms:.3e}  max={lap_max:.3e}")

        # ===================== Loss evolution figure (Adam + LBFGS) =====================
        try:
            figL = plt.figure(figsize=(10, 4), constrained_layout=True)
            axL  = figL.add_subplot(1,1,1)

            x_adam = jnp.arange(1, len(adam_loss_hist)+1)

            # 1) Show BOTH Adam metrics
            axL.semilogy(x_adam, adam_train_like_hist, label="Adam (train-like)", linewidth=2)
            axL.semilogy(x_adam, adam_eval_full_hist,  label="Adam (eval_full = Lin + λ·Lbc)", alpha=0.8)

            # 2) Append LBFGS immediately after Adam on the same x-axis
            x0 = len(adam_loss_hist)
            if len(lbfgs_train_like_hist) == 0 and len(lbfgs_hist) > 0:
                # keep backward-compat: lbfgs_hist is train-like history
                lbfgs_train_like_hist = lbfgs_hist

            if len(lbfgs_train_like_hist) > 0:
                x_lb1 = jnp.arange(x0+1, x0+len(lbfgs_train_like_hist)+1)
                axL.semilogy(x_lb1, lbfgs_train_like_hist, label="LBFGS (train-like)", linewidth=2)

            if len(lbfgs_eval_full_hist) > 0:
                x_lb2 = jnp.arange(x0+1, x0+len(lbfgs_eval_full_hist)+1)
                axL.semilogy(x_lb2, lbfgs_eval_full_hist, label="LBFGS (eval_full)", linewidth=2, linestyle="--")

            axL.set_xlabel("Iteration")
            axL.set_ylabel("Loss (semilogy)")
            axL.set_title("Objective evolution (Adam → LBFGS)")
            axL.legend()
            plt.show()
        except Exception as _e:
            print(f"[WARN] Could not plot loss curve: {_e}")

        # Compute final boundary grad & plot side-by-side
        mf, ms = eqx.partition(model, eqx.is_inexact_array)
        model_eval = eqx.combine(ema_f, ms) if (use_ema and ema_eval) else model
        Gvec_final, Gmag_final, _ = eval_on_boundary(model_eval, P_bdry, N_bdry, Xg, Yg, Zg)

        vmin_shared = float(jnp.minimum(Gmag_init.min(), Gmag_final.min()))
        vmax_shared = float(jnp.maximum(Gmag_init.max(), Gmag_final.max()))
        fig3d = plt.figure(figsize=(12, 6), constrained_layout=True)
        gs = fig3d.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
        ax1 = fig3d.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig3d.add_subplot(gs[0, 1], projection='3d')
        cax = fig3d.add_subplot(gs[0, 2])

        offset = 0.1 * float(a0 + abs(a1))
        m1 = plot_surface_with_vectors_ax(ax1, Xg, Yg, Zg, Gmag_init, Nhat_grid, Gvec=Gvec_init,
                                           title="Initial |∇u| and vectors (boundary)",
                                           cmap=PLOT_CMAP, quiver_len=0.15,
                                           step_theta=6, step_phi=8, plot_normals=False,
                                           vmin=vmin_shared, vmax=vmax_shared,
                                           surf_offset=offset)
        m2 = plot_surface_with_vectors_ax(ax2, Xg, Yg, Zg, Gmag_final, Nhat_grid, Gvec=Gvec_final,
                                           title="Final |∇u| and vectors (boundary)",
                                           cmap=PLOT_CMAP, quiver_len=0.15,
                                           step_theta=6, step_phi=8, plot_normals=False,
                                           vmin=vmin_shared, vmax=vmax_shared,
                                           surf_offset=offset)

        # draw the fixed box on both plots
        draw_box_edges(ax1, xmin, xmax, ymin, ymax, zmin, zmax, lw=1.0, alpha=0.6)
        draw_box_edges(ax2, xmin, xmax, ymin, ymax, zmin, zmax, lw=1.0, alpha=0.6)

        cb = fig3d.colorbar(m2, cax=cax); cb.set_label(r"$|\nabla u|$")
        fix_matplotlib_3d(ax1); fix_matplotlib_3d(ax2)

        # θ×φ heatmaps
        theta = jnp.linspace(0, 2*jnp.pi, N_bdry_theta, endpoint=True)
        phi   = jnp.linspace(0, 2*jnp.pi, N_bdry_phi,   endpoint=True)
        TH, PH = jnp.meshgrid(theta, phi, indexing='ij')
        GN_init  = Gmag_init
        GN_final = Gmag_final
        vmin_hm = float(jnp.minimum(GN_init.min(), GN_final.min()))
        vmax_hm = float(jnp.maximum(GN_init.max(), GN_final.max()))
        figHM = plt.figure(figsize=(12, 4.5), constrained_layout=True)
        gs = figHM.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
        axL = figHM.add_subplot(gs[0, 0]); axR = figHM.add_subplot(gs[0, 1]); cax = figHM.add_subplot(gs[0, 2])
        imL = axL.pcolormesh(PH, TH, GN_init, shading="auto", cmap=PLOT_CMAP, vmin=vmin_hm, vmax=vmax_hm)
        axL.set_title("Initial  $|\\nabla u|$ on boundary"); axL.set_xlabel(r"$\phi$"); axL.set_ylabel(r"$\theta$")
        imR = axR.pcolormesh(PH, TH, GN_final, shading="auto", cmap=PLOT_CMAP, vmin=vmin_hm, vmax=vmax_hm)
        axR.set_title("Final  $|\\nabla u|$ on boundary"); axR.set_xlabel(r"$\phi$")
        cb = figHM.colorbar(imR, cax=cax); cb.set_label(r"$|\nabla u|$")
        plt.show()

    elif mode == "torus":
        # ===== MULTI-SURFACE PRETRAINING =====
        torus_list = surfaces_cfg.get("torus_list", [])
        if len(torus_list) == 0:
            raise RuntimeError("No torus surfaces listed under [[surfaces.torus_list]] for mode='torus'.")

        dataset = build_torus_family(torus_list, N_bdry_theta, N_bdry_phi)
        print(f"[DATA] Loaded {len(dataset)} torus surfaces for pretraining.")

        # Optional: pick first surface for initial plotting/grids (for visualization later)
        surf0 = dataset[0]
        # Reconstruct Xg,Yg,Zg grids from flattened boundary points (shape nθ×nφ×3)
        Pgrid0 = surf0.P_bdry.reshape(N_bdry_theta, N_bdry_phi, 3)
        Xg0, Yg0, Zg0 = Pgrid0[..., 0], Pgrid0[..., 1], Pgrid0[..., 2]

        # Initial eval on surf0 (use masked interior from the common fixed box)
        mask0 = surf0.inside_mask_fn(P_box)
        ids0  = jnp.nonzero(mask0, size=min(N_in, P_box.shape[0]), fill_value=0)[0]
        P_in0 = P_box[ids0][:N_in]
        (L0, (Lin0, Lbc0, lap0, n0)) = eval_full(model, P_in0, surf0.P_bdry, surf0.N_bdry)
        print(f"[INIT:{surf0.name}] loss={float(L0):.6e}  lap={float(Lin0):.6e}  bc={float(Lbc0):.6e}")

        # Precompute initial boundary grad for surf0
        Gvec_init, Gmag_init, Nhat_grid0 = eval_on_boundary(model, surf0.P_bdry, surf0.N_bdry, Xg0, Yg0, Zg0)

        # Build fixed interior pools ONCE (deterministic) for all surfaces
        packs_all = []
        for surf in dataset:
            mask = surf.inside_mask_fn(P_box)
            ids  = jnp.nonzero(mask, size=min(N_in, P_box.shape[0]), fill_value=0)[0]
            P_in_all = P_box[ids][:N_in]
            packs_all.append((P_in_all, surf.P_bdry, surf.N_bdry))
        packs_all = tuple(packs_all)  # static for JIT

        # ---- Train: aggregate ALL surfaces every step ----
        key_train = random.PRNGKey(1234)
        log_every = LOG_EVERY
        for it in range(1, steps + 1):
            cur_lam = _lambda_bc_schedule(it, steps)
            runtime.lam_bc = float(cur_lam)
            key_train, subkey = random.split(key_train)

            model, opt_state, L, (Lin, Lbc, lap_res, nres, mean_u, c_bc_mean), gnorm, grads = train_step_many(
                model, opt_state, optimizer, packs_all, subkey
            )
            # record per-iter history (Adam) – aggregated stats
            adam_loss_hist.append(float(L))
            adam_lin_hist.append(float(Lin))
            adam_lbc_hist.append(float(Lbc))
            adam_lap_rms_hist.append(float(jnp.sqrt(jnp.mean(lap_res**2))))
            adam_nbc_rms_hist.append(float(jnp.sqrt(jnp.mean(nres**2))))
            adam_eval_full_hist.append(float(Lin + runtime.lam_bc * Lbc))
            adam_train_like_hist.append(float(_full_objective_like_training(model, P_in0, surf0.P_bdry, surf0.N_bdry)))
            if use_ema:
                mf, ms = eqx.partition(model, eqx.is_inexact_array)
                ema_f = _ema_update(ema_f, mf, ema_decay)
            if lookahead_enabled:
                la_step += 1
                if (la_step % lookahead_k) == 0:
                    mf, ms = eqx.partition(model, eqx.is_inexact_array)
                    slow_f = _lookahead_sync(slow_f, mf, lookahead_alpha)
                    model = eqx.combine(slow_f, ms)

            # --- Augmented Lagrangian update (outer loop) ---
            if runtime.al_enabled and ((it % runtime.al_update_every) == 0):
                runtime.al_lambda = float(runtime.al_lambda + runtime.al_rho * float(c_bc_mean))
                if runtime.al_clip and (runtime.al_clip > 0.0):
                    runtime.al_lambda = float(jnp.clip(runtime.al_lambda, -runtime.al_clip, runtime.al_clip))

            if (it % log_every) == 0 or it == 1:
                stats = debug_stats(model, grads, lap_res, nres)
                # evaluate |∇u| on an anchor surface (surf0) for a stable diagnostic
                if use_ema and ema_eval:
                    mf, ms = eqx.partition(model, eqx.is_inexact_array)
                    model_eval = eqx.combine(ema_f, ms)
                else:
                    model_eval = model
                gmag0 = jnp.linalg.norm(grad_u_total_batch(model_eval, dataset[0].P_bdry), axis=-1)
                mean_g0 = float(jnp.mean(gmag0))
                print(f"[{it:5d}] loss={float(L):.6e}  lin={float(Lin):.3e}  lbc={float(Lbc):.3e}  "
                    f"|lap|_rms={stats['lap_rms']:.3e}  "
                    f"|n·∇u|_rms={stats['nbc_rms']:.3e}  "
                    f"||g||={stats['grad_L2']:.3e}  mean|∇u|(surf0)={mean_g0:.3e}  mean(u)={float(mean_u):.3e}  "
                    f"λ_bc={runtime.lam_bc:.2f}  "
                    f"{'AL ' if runtime.al_enabled else ''}c={float(c_bc_mean):+.3e} λ_AL={runtime.al_lambda:+.3e}  "
                    f"surf=ALL")

        model_to_save = model
        if use_ema and ema_eval:
            mf, ms = eqx.partition(model, eqx.is_inexact_array)
            model_to_save = eqx.combine(ema_f, ms)

        # Optional LBFGS polish across **all** surfaces
        runtime._lbfgs_eval_full_hist_sink = lbfgs_eval_full_hist
        if LBFGS_STEPS > 0:
            packs = []
            for surf in dataset:
                # interior from common box (deterministic)
                mask = surf.inside_mask_fn(P_box)
                ids  = jnp.nonzero(mask, size=min(N_in, P_box.shape[0]), fill_value=0)[0]
                P_in_all = P_box[ids][:N_in]
                # Downselect interior for LBFGS if requested
                P_in_use = _take_first_n(P_in_all, LBFGS_IN)

                # boundary: optionally sub-sample deterministically to cap memory
                P_b_all, N_b_all = surf.P_bdry, surf.N_bdry
                P_b_use = _stride_take(P_b_all, LBFGS_BDRY)
                N_b_use = _stride_take(N_b_all, LBFGS_BDRY)

                packs.append((P_in_use, P_b_use, N_b_use, surf.name))

            model_to_save = model
            if use_ema and ema_eval:
                mf, ms = eqx.partition(model, eqx.is_inexact_array)
                model_to_save = eqx.combine(ema_f, ms)

            model_to_save = _lbfgs_polish_many(
                model_to_save, packs,
                steps=LBFGS_STEPS, tol=LBFGS_TOL,
                print_every=LBFGS_PRINT_EVERY, label="torus:ALL",
                history=lbfgs_hist,
            )
        runtime._lbfgs_eval_full_hist_sink = None

        save_model(model_to_save, CHECKPOINT_PATH)

        # Final diagnostics on a couple of surfaces
        for j in range(min(2, len(dataset))):
            surf = dataset[j]
            mask = surf.inside_mask_fn(P_box)
            ids  = jnp.nonzero(mask, size=min(N_in, P_box.shape[0]), fill_value=0)[0]
            P_in = P_box[ids][:N_in]
            (Lf, (Linf, Lbcf, lapf, nf)) = eval_full(model, P_in, surf.P_bdry, surf.N_bdry)
            print(f"[FINAL:{surf.name}] loss={float(Lf):.6e}  lap={float(Linf):.6e}  bc={float(Lbcf):.6e}  "
                  f"|lap|_rms={float(jnp.sqrt(jnp.mean(lapf**2))):.3e}  "
                  f"|n·∇u|_rms={float(jnp.sqrt(jnp.mean(nf**2))):.3e}")

        # ===================== Loss evolution figure (Adam + LBFGS) =====================
        try:
            figL = plt.figure(figsize=(10, 4), constrained_layout=True)
            axL  = figL.add_subplot(1,1,1)

            x_adam = jnp.arange(1, len(adam_loss_hist)+1)

            # 1) Show BOTH Adam metrics
            axL.semilogy(x_adam, adam_train_like_hist, label="Adam (train-like)", linewidth=2)
            axL.semilogy(x_adam, adam_eval_full_hist,  label="Adam (eval_full = Lin + λ·Lbc)", alpha=0.8)

            # 2) Append LBFGS immediately after Adam on the same x-axis
            x0 = len(adam_loss_hist)
            if len(lbfgs_train_like_hist) == 0 and len(lbfgs_hist) > 0:
                # keep backward-compat: lbfgs_hist is train-like history
                lbfgs_train_like_hist = lbfgs_hist

            if len(lbfgs_train_like_hist) > 0:
                x_lb1 = jnp.arange(x0+1, x0+len(lbfgs_train_like_hist)+1)
                axL.semilogy(x_lb1, lbfgs_train_like_hist, label="LBFGS (train-like)", linewidth=2)

            if len(lbfgs_eval_full_hist) > 0:
                x_lb2 = jnp.arange(x0+1, x0+len(lbfgs_eval_full_hist)+1)
                axL.semilogy(x_lb2, lbfgs_eval_full_hist, label="LBFGS (eval_full)", linewidth=2, linestyle="--")

            axL.set_xlabel("Iteration")
            axL.set_ylabel("Loss (semilogy)")
            axL.set_title("Objective evolution (Adam → LBFGS)")
            axL.legend()
            plt.show()
        except Exception as _e:
            print(f"[WARN] Could not plot loss curve: {_e}")

        # Plot surf0 initial vs final
        mf, ms = eqx.partition(model, eqx.is_inexact_array)
        model_eval = eqx.combine(ema_f, ms) if (use_ema and ema_eval) else model
        Gvec_final, Gmag_final, _ = eval_on_boundary(model_eval, surf0.P_bdry, surf0.N_bdry, Xg0, Yg0, Zg0)
        vmin_shared = float(jnp.minimum(Gmag_init.min(), Gmag_final.min()))
        vmax_shared = float(jnp.maximum(Gmag_init.max(), Gmag_final.max()))
        fig3d = plt.figure(figsize=(12, 6), constrained_layout=True)
        gs = fig3d.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
        ax1 = fig3d.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig3d.add_subplot(gs[0, 1], projection='3d')
        cax = fig3d.add_subplot(gs[0, 2])

        offset = 0.1 * float(a0 + abs(a1))
        m1 = plot_surface_with_vectors_ax(ax1, Xg0, Yg0, Zg0, Gmag_init, Nhat_grid0, Gvec=Gvec_init,
                                           title=f"Initial |∇u| ({surf0.name})",
                                           cmap=PLOT_CMAP, quiver_len=0.15,
                                           step_theta=6, step_phi=8, plot_normals=False,
                                           vmin=vmin_shared, vmax=vmax_shared,
                                           surf_offset=offset)
        m2 = plot_surface_with_vectors_ax(ax2, Xg0, Yg0, Zg0, Gmag_final, Nhat_grid0, Gvec=Gvec_final,
                                           title=f"Final |∇u| ({surf0.name})",
                                           cmap=PLOT_CMAP, quiver_len=0.15,
                                           step_theta=6, step_phi=8, plot_normals=False,
                                           vmin=vmin_shared, vmax=vmax_shared,
                                           surf_offset=offset)

        draw_box_edges(ax1, xmin, xmax, ymin, ymax, zmin, zmax, lw=1.0, alpha=0.6)
        draw_box_edges(ax2, xmin, xmax, ymin, ymax, zmin, zmax, lw=1.0, alpha=0.6)

        cb = fig3d.colorbar(m2, cax=cax); cb.set_label(r"$|\nabla u|$")
        fix_matplotlib_3d(ax1); fix_matplotlib_3d(ax2)
        plt.show()

    else:
        raise NotImplementedError(f"Unknown [surfaces].mode={mode!r}. Use 'single' or 'torus'.")

    return model

def _lbfgs_polish(model, P_in, P_bdry, N_bdry, *,
                  steps: int, tol: float, print_every: int, label: str,
                  history: list[float] | None = None):
    """
    Run jaxopt.LBFGS on the full-batch objective with a Python loop so we can
    print diagnostics every few iterations. Optimizes only inexact arrays.
    """
    if steps <= 0:
        print(f"[LBFGS] Skipped (steps={steps}).")
        return model

    # Use the *final* λ_bc during polish
    runtime.lam_bc = float(lam_bc)

    # Partition: optimize only floating leaves
    params_f, params_s = eqx.partition(model, eqx.is_inexact_array)

    # Initial diagnostics
    tot0, (Lin0, Lbc0, lap0, n0) = eval_full(model, P_in, P_bdry, N_bdry)
    print(f"[LBFGS:{label} INIT] f={float(tot0):.6e}  lin={float(Lin0):.3e}  lbc={float(Lbc0):.3e}  "
          f"|lap|_rms={float(jnp.sqrt(jnp.mean(lap0**2))):.3e}  "
          f"|n·∇u|_rms={float(jnp.sqrt(jnp.mean(n0**2))):.3e}")

    # Tiny L2 to prevent runaway in SIREN/Fourier regimes
    include_zm = bool(getattr(runtime, "lbfgs_include_zero_mean", True))
    include_al = bool(getattr(runtime, "lbfgs_include_aug_lagrangian", True))
    l2_weight  = float(getattr(runtime, "lbfgs_l2", 1e-8))

    def obj(params_f_opt):
        m = eqx.combine(params_f_opt, params_s)
        if include_zm or include_al:
            total = _full_objective_like_training(m, P_in, P_bdry, N_bdry)
        else:
            total, _ = eval_full(m, P_in, P_bdry, N_bdry)
        l2 = sum([jnp.sum(w*w) for w in jax.tree_util.tree_leaves(params_f_opt)])
        val = total + l2_weight * l2
        return jnp.asarray(_finite_or_big(val), dtype=jnp.float64)

    solver = LBFGS(
        fun=obj,
        value_and_grad=False,
        has_aux=False,
        tol=tol,
        stepsize=5e-4,             # << smaller initial step
        linesearch="zoom",         # more conservative than Hager–Zhang in practice here
        maxls=80,                  # allow more backtracking
        history_size=30,
        verbose=False,
    )

    # Initialize state
    state = solver.init_state(params_f)

    # Manual loop so we can print every `print_every`
    params_curr = params_f
    for it in range(1, int(steps) + 1):
        params_curr, state = solver.update(params_curr, state)
        # Record training-like objective value every iteration (no L2)
        m_for_hist = eqx.combine(params_curr, params_s)
        total_train_like = _full_objective_like_training(m_for_hist, P_in, P_bdry, N_bdry)
        if history is not None:
            history.append(float(total_train_like))
        # record eval_full:
        if hasattr(runtime, "_lbfgs_eval_full_hist_sink") and runtime._lbfgs_eval_full_hist_sink is not None:
            total_eval, _ = eval_full(m_for_hist, P_in, P_bdry, N_bdry)
            runtime._lbfgs_eval_full_hist_sink.append(float(total_eval))
        if (it % print_every) == 0 or it == 1 or it == steps:
            m = eqx.combine(params_curr, params_s)
            total, (Lin, Lbc, lap, n_dot) = eval_full(m, P_in, P_bdry, N_bdry)
            lap_rms = float(jnp.sqrt(jnp.mean(lap**2)))
            nbc_rms = float(jnp.sqrt(jnp.mean(n_dot**2)))
            grad_norm = getattr(state, "grad_norm", None)
            gtxt = (f"  grad_norm={float(grad_norm):.3e}" if grad_norm is not None else "")
            print(f"[LBFGS:{label} {it:4d}/{steps}] f={float(total):.6e}  "
                  f"lin={float(Lin):.3e}  lbc={float(Lbc):.3e}  "
                  f"|lap|_rms={lap_rms:.3e}  |n·∇u|_rms={nbc_rms:.3e}{gtxt}")

        # Early stop if solver exposes an error metric and we’re below tol
        err = getattr(state, "error", None)
        if (err is not None) and (float(err) <= tol):
            break

    model_opt = eqx.combine(params_curr, params_s)

    # Final diagnostics
    totf, (Linf, Lbcf, lapf, nf) = eval_full(model_opt, P_in, P_bdry, N_bdry)
    print(f"[LBFGS:{label} DONE] f={float(totf):.6e}  lin={float(Linf):.3e}  lbc={float(Lbcf):.3e}  "
          f"|lap|_rms={float(jnp.sqrt(jnp.mean(lapf**2))):.3e}  "
          f"|n·∇u|_rms={float(jnp.sqrt(jnp.mean(nf**2))):.3e}")
    return model_opt

def _lbfgs_polish_many(model, packs, *, steps: int, tol: float, print_every: int, label: str,
                       history: list[float] | None = None):
    """
    LBFGS polish over multiple surfaces simultaneously.

    packs: tuple list of (P_in, P_bdry, N_bdry, name)
      - Each array is **fixed** (no randomness) to keep objective deterministic.
    The objective is the sum of full-batch totals across surfaces (equal weighting).
    """
    if steps <= 0:
        print(f"[LBFGS-many] Skipped (steps={steps}).")
        return model

    runtime.lam_bc = float(lam_bc)  # use final λ_bc

    # Filter model leaves (only inexact arrays are optimized)
    params_f, params_s = eqx.partition(model, eqx.is_inexact_array)

    # Make packs JAX-friendly (tuple of tuples of jnp arrays)
    packs_tup = tuple((jnp.asarray(Pi), jnp.asarray(Pb), jnp.asarray(Nb), str(name))
                      for (Pi, Pb, Nb, name) in packs)

    # Initial diagnostics (aggregate)
    def _agg_stats(mod):
        vals = []
        for (Pi, Pb, Nb, _nm) in packs_tup:
            t, (Lin, Lbc, lap, n) = eval_full(mod, Pi, Pb, Nb)
            vals.append((
                float(t), float(Lin), float(Lbc),
                float(jnp.sqrt(jnp.mean(lap**2)) ),
                float(jnp.sqrt(jnp.mean(n**2)) )))
        # mean over surfaces
        import numpy as _np
        arr = _np.array(vals)
        return {
            "f": arr[:,0].mean(), "lin": arr[:,1].mean(), "lbc": arr[:,2].mean(),
            "lap_rms": arr[:,3].mean(), "nbc_rms": arr[:,4].mean()
        }

    s0 = _agg_stats(model)
    print(f"[LBFGS-many:{label} INIT] f={s0['f']:.6e} lin={s0['lin']:.3e} lbc={s0['lbc']:.3e} "
          f"|lap|_rms={s0['lap_rms']:.3e} |n·∇u|_rms={s0['nbc_rms']:.3e}")

    include_zm = bool(getattr(runtime, "lbfgs_include_zero_mean", True))
    include_al = bool(getattr(runtime, "lbfgs_include_aug_lagrangian", True))
    l2_weight  = float(getattr(runtime, "lbfgs_l2", 1e-8))

    def obj(params_f_opt):
        m = eqx.combine(params_f_opt, params_s)
        total = 0.0
        for (Pi, Pb, Nb, _nm) in packs_tup:
            if include_zm or include_al:
                total = total + _full_objective_like_training(m, Pi, Pb, Nb)
            else:
                t, _ = eval_full(m, Pi, Pb, Nb)
                total = total + t
        total = total / float(len(packs_tup))
        l2 = sum([jnp.sum(w*w) for w in jax.tree_util.tree_leaves(params_f_opt)])
        val = total + l2_weight * l2
        return jnp.asarray(_finite_or_big(val), dtype=jnp.float64)

    solver = LBFGS(
        fun=obj,
        value_and_grad=False,
        has_aux=False,
        tol=tol,
        stepsize=5e-4,             # << smaller initial step
        linesearch="zoom",         # more conservative than Hager–Zhang in practice here
        maxls=80,                  # allow more backtracking
        history_size=30,
        verbose=False,
    )

    state = solver.init_state(params_f)
    params_curr = params_f

    for it in range(1, int(steps) + 1):
        params_curr, state = solver.update(params_curr, state)
        m_for_hist = eqx.combine(params_curr, params_s)
        # sum training-like objective over surfaces, mean for scale stability
        tot_train_like = 0.0
        for (Pi, Pb, Nb, _nm) in packs_tup:
            tot_train_like = tot_train_like + _full_objective_like_training(m_for_hist, Pi, Pb, Nb)
        tot_train_like = tot_train_like / float(len(packs_tup))
        if history is not None:
            history.append(float(tot_train_like))
        if hasattr(runtime, "_lbfgs_eval_full_hist_sink") and runtime._lbfgs_eval_full_hist_sink is not None:
            s_eval = _agg_stats(m_for_hist)   # uses eval_full on all packs
            runtime._lbfgs_eval_full_hist_sink.append(float(s_eval["f"]))
        if (it % print_every) == 0 or it == 1 or it == steps:
            m = eqx.combine(params_curr, params_s)
            si = _agg_stats(m)
            gn = getattr(state, "grad_norm", None)
            gtxt = (f"  grad_norm={float(gn):.3e}" if gn is not None else "")
            print(f"[LBFGS-many:{label} {it:4d}/{steps}] f={si['f']:.6e} lin={si['lin']:.3e} "
                  f"lbc={si['lbc']:.3e} |lap|_rms={si['lap_rms']:.3e} |n·∇u|_rms={si['nbc_rms']:.3e}{gtxt}")
        err = getattr(state, "error", None)
        if (err is not None) and (float(err) <= tol):
            break

    model_opt = eqx.combine(params_curr, params_s)
    sf = _agg_stats(model_opt)
    print(f"[LBFGS-many:{label} DONE] f={sf['f']:.6e} lin={sf['lin']:.3e} lbc={sf['lbc']:.3e} "
          f"|lap|_rms={sf['lap_rms']:.3e} |n·∇u|_rms={sf['nbc_rms']:.3e}")
    return model_opt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PINN for Laplace on a torus (single or multi-surface)")
    parser.add_argument("--config", "-c", type=str, default="input.toml",
                        help="Path to TOML configuration file")
    args = parser.parse_args()
    _ = main(config_path=args.config)