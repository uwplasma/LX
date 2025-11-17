#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_ml_v2.py

Data-driven reduced model for Harris-sheet tearing (v2, using mhd_tearing_solve.py):

  1) Uses solve_tearing_case from mhd_tearing_solve.py to generate a database
     of tearing evolutions (full MHD solves, all physics centralized there).
  2) Extracts diagnostics from each .npz solution:
       - tearing amplitude A(t) via tearing_amplitude(B_hat, Lx, Ly, Lz)
       - linear growth rate γ_fit (from log A(t) in linear phase)
       - saturated amplitude A_sat
       - physics-aware features from the equilibrium parameters.
  3) Trains:
       - Supervised surrogate MLP:
             physics-aware features -> (γ_fit, A_sat)
         using log-transformed, standardized targets.
       - Latent ODE "autoencoder" for A(t):
         shared latent dynamics, parameter-dependent encoder/decoder.

Key points:
  - All PDE / MHD physics, numerics and normalization are in mhd_tearing_solve.py.
  - This script ONLY:
        * calls solve_tearing_case(...) to generate .npz solutions,
        * post-processes them to build ML datasets,
        * trains the surrogate and latent ODE models,
        * produces plots and saves trained models.

Outputs:
  - surrogate_model.pkl  (NN params + normalization stats)
  - latent_ode_model.pkl (NN params + logA normalization)
  - Plots:
      * surrogate_loss_curves.png
      * surrogate_parity.png
      * latent_ode_loss_curve.png
      * latent_ode_recon.png
"""

from __future__ import annotations

import os
import math
import time
import pickle
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import diffrax as dfx
import optax
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------#
# Import physics/numerics from mhd_tearing_solve (single source of truth)
# -----------------------------------------------------------------------------#

from mhd_tearing_solve import (
    solve_tearing_case,
    tearing_amplitude,
    fkr_gamma,  # available if we want to compare later
)

# -----------------------------------------------------------------------------#
# Simple utilities
# -----------------------------------------------------------------------------#

LABEL_EPS = 1e-8  # for log-transform of γ, A_sat


def safe_log(x, eps: float = 1e-16):
    return jnp.log(jnp.clip(x, eps, None))


def fit_growth_rate(
    ts: jnp.ndarray,
    amp: jnp.ndarray,
    frac_window: Tuple[float, float] = (0.15, 0.5),
) -> float:
    """
    Fit γ from A(t) ≈ A0 exp(γ t) using least-squares on log A.

    Strategy:
      1) select a coarse window [f1, f2] in time,
      2) scan a few overlapping subwindows inside it,
      3) pick the subwindow with the best linear fit (largest R^2).
    """
    ts = jnp.asarray(ts)
    amp = jnp.asarray(amp)
    t0, t1 = ts[0], ts[-1]
    T = t1 - t0

    t_start = t0 + frac_window[0] * T
    t_end   = t0 + frac_window[1] * T
    base_mask = (ts >= t_start) & (ts <= t_end)
    ts_base = ts[base_mask]
    amp_base = amp[base_mask]

    # Fallbacks if something went wrong
    if ts_base.shape[0] < 6:
        ts_base = ts
        amp_base = amp

    y_base = safe_log(amp_base)

    # Scan 3 overlapping subwindows inside the base window
    n = ts_base.shape[0]
    n_sub = max(4, n // 2)
    best_gamma = 0.0
    best_r2 = -jnp.inf

    for offset in [0, (n - n_sub) // 2, n - n_sub]:
        offset = int(jnp.clip(offset, 0, n - n_sub))
        ts_sub = ts_base[offset : offset + n_sub]
        y_sub = y_base[offset : offset + n_sub]

        t_mean = jnp.mean(ts_sub)
        y_mean = jnp.mean(y_sub)
        cov_ty = jnp.mean((ts_sub - t_mean) * (y_sub - y_mean))
        var_t = jnp.mean((ts_sub - t_mean) ** 2) + 1e-16
        gamma = cov_ty / var_t

        # R^2 of this local fit
        y_pred = y_mean + gamma * (ts_sub - t_mean)
        ss_res = jnp.mean((y_sub - y_pred) ** 2)
        ss_tot = jnp.mean((y_sub - y_mean) ** 2) + 1e-16
        r2 = 1.0 - ss_res / ss_tot

        best_gamma = jnp.where(r2 > best_r2, gamma, best_gamma)
        best_r2 = jnp.maximum(best_r2, r2)

    return float(best_gamma)


def compute_saturated_amplitude(amp: jnp.ndarray, frac_tail: float = 0.2) -> float:
    """
    Simple proxy: average of last frac_tail fraction of A(t).
    """
    n = amp.size
    start = int((1.0 - frac_tail) * n)
    tail = amp[start:]
    return float(jnp.mean(tail))


# -----------------------------------------------------------------------------#
# Parameter containers and feature vector
# -----------------------------------------------------------------------------#


@dataclass
class GridConfig:
    Nx: int = 32
    Ny: int = 32
    Nz: int = 1
    Lx: float = 2.0 * math.pi
    Ly: float = 2.0 * math.pi
    Lz: float = 2.0 * math.pi


@dataclass
class PhysicalParams:
    nu: float
    eta: float
    B0: float
    a: float
    B_g: float
    eps_B: float


def build_feature_vector(config: GridConfig, phys: PhysicalParams) -> jnp.ndarray:
    """
    Physics-aware feature vector for surrogate / latent models.

    Features:
      x0 = log S,   S = B0 * a / eta
      x1 = log Pm,  Pm = nu / eta
      x2 = log(k_y a),  k_y = 2π / Ly  (fundamental tearing mode)
      x3 = eps_B

    Returns: shape (4,)
    """
    B0 = phys.B0
    a = phys.a
    nu = phys.nu
    eta = phys.eta
    eps_B = phys.eps_B
    Ly = config.Ly

    S = B0 * a / eta
    Pm = nu / eta
    ky = 2.0 * math.pi / Ly
    ky_a = ky * a

    feat = jnp.array(
        [
            safe_log(S + 1e-16),
            safe_log(Pm + 1e-16),
            safe_log(ky_a + 1e-16),
            eps_B,
        ],
        dtype=jnp.float64,
    )
    return feat


# -----------------------------------------------------------------------------#
# Single run using solve_tearing_case from mhd_tearing_solve.py
# -----------------------------------------------------------------------------#


def run_single_sim(
    grid: GridConfig,
    phys: PhysicalParams,
    t0: float,
    t1: float,
    n_frames: int,
    outdir: str,
    case_id: int,
    dt0_override: float | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one tearing simulation via solve_tearing_case from mhd_tearing_solve.py.

    Steps:
      1) Build an outfile path unique to this case.
      2) Call solve_tearing_case(..., outfile=...).
      3) Load the .npz file.
      4) Extract ts and B_hat(t), compute A(t) via tearing_amplitude.
      5) Fit γ_fit and A_sat.
      6) Build physics-aware feature vector from the parameters.

    Returns dict with:
      - ts:        (T,)
      - amp:       (T,)
      - gamma_fit: scalar
      - A_sat:     scalar
      - eq_params: (4,) feature vector
    """
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"tearing_ml_case_{case_id:04d}.npz")

    if verbose:
        print(f"\n[RUN] Case {case_id}: saving to {outfile}")

    # Use the central solver (all physics/numerics defined there)
    solve_tearing_case(
        Nx=grid.Nx,
        Ny=grid.Ny,
        Nz=grid.Nz,
        Lx=grid.Lx,
        Ly=grid.Ly,
        Lz=grid.Lz,
        nu=phys.nu,
        eta=phys.eta,
        B0=phys.B0,
        a=phys.a,
        B_g=phys.B_g,
        eps_B=phys.eps_B,
        t0=t0,
        t1=t1,
        n_frames=n_frames,
        dt0=dt0_override,
        outfile=outfile,
    )

    # Load from the .npz solution
    data = np.load(outfile)
    ts = jnp.array(data["ts"])  # (T,)
    B_hat_frames = jnp.array(data["B_hat"])  # (T, 3, Nx, Ny, Nz)
    Lx = float(data["Lx"])
    Ly = float(data["Ly"])
    Lz = float(data["Lz"])

    # Compute A(t) using tearing_amplitude from mhd_tearing_solve
    amp_list = []
    for k in range(B_hat_frames.shape[0]):
        amp_k = tearing_amplitude(B_hat_frames[k], Lx, Ly, Lz)
        amp_list.append(amp_k)
    amp = jnp.array(amp_list)

    gamma_fit = fit_growth_rate(ts, amp)
    A_sat = compute_saturated_amplitude(amp)

    eq_params = build_feature_vector(grid, phys)  # (4,)

    return dict(
        ts=ts,
        amp=amp,
        gamma_fit=gamma_fit,
        A_sat=A_sat,
        eq_params=eq_params,
    )


# -----------------------------------------------------------------------------#
# Build a training database via a parameter scan
# -----------------------------------------------------------------------------#


def build_dataset(
    grid: GridConfig,
    n_cases: int = 32,
    t0: float = 0.0,
    t1: float = 40.0,
    n_frames: int = 80,
    seed: int = 0,
    outdir: str = "tearing_ml_runs",
    verbose: bool = True,
) -> Dict[str, jnp.ndarray]:
    """
    Sample a small parameter space and build:
      - eq_params: (N, P)  physics-aware features
      - ts:        (T,)  (shared)
      - amp:       (N, T)
      - gamma_fit: (N,)
      - A_sat:     (N,)
    """
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []

    for i in range(n_cases):
        # Baseline geometry and equilibrium
        B0 = 1.0
        a = grid.Lx / 16.0      # fixed shear-layer width
        B_g = 0.2               # fixed guide field (cleaner scan)
        eps_B = 1.0e-2          # fixed perturbation amplitude

        # Wider log-uniform scan in resistivity:
        #   eta ~ [3e-5, 3e-3]  → S ~ O(10^2–10^4)
        log_eta = rng.uniform(-4.5, -2.5)
        eta = 10.0 ** log_eta

        # Keep Pm ≈ 1 to avoid additional parameter entanglement
        Pm_target = 1.0
        nu = Pm_target * eta

        phys = PhysicalParams(
            nu=nu,
            eta=eta,
            B0=B0,
            a=a,
            B_g=B_g,
            eps_B=eps_B,
        )

        if verbose:
            print(f"\n[SCAN] Case {i+1}/{n_cases}: {phys}")

        rec = run_single_sim(
            grid=grid,
            phys=phys,
            t0=t0,
            t1=t1,
            n_frames=n_frames,
            outdir=outdir,
            case_id=i,
            dt0_override=None,
            verbose=verbose,
        )
        records.append(rec)

    ts = records[0]["ts"]
    amp = jnp.stack([r["amp"] for r in records], axis=0)  # (N, T)
    gamma_fit = jnp.array([r["gamma_fit"] for r in records])  # (N,)
    A_sat = jnp.array([r["A_sat"] for r in records])  # (N,)
    eq_params = jnp.stack([r["eq_params"] for r in records], 0)  # (N, P)

    if verbose:
        print("\n[DATASET] Built dataset:")
        print(f"  eq_params: {eq_params.shape}")
        print(f"  amp:       {amp.shape}")
        print(f"  gamma_fit: {gamma_fit.shape}")
        print(f"  A_sat:     {A_sat.shape}")

    return dict(ts=ts, amp=amp, gamma_fit=gamma_fit, A_sat=A_sat, eq_params=eq_params)


# -----------------------------------------------------------------------------#
# Generic JAX MLP utilities (params as PyTrees)
# -----------------------------------------------------------------------------#


def init_mlp(
    rng_key,
    in_dim: int,
    out_dim: int,
    width: int = 32,
    depth: int = 2,
) -> Dict[str, Any]:
    """
    Simple fully-connected MLP with 'depth' hidden layers.

    Returns params as a pytree with only arrays:
      params = {"Ws": [W0,...,WL], "bs": [b0,...,bL]}
    """
    params = {"Ws": [], "bs": []}
    key = rng_key
    layer_dims = [in_dim] + [width] * depth + [out_dim]

    for i in range(len(layer_dims) - 1):
        key, subkey = jax.random.split(key)
        w_shape = (layer_dims[i], layer_dims[i + 1])
        b_shape = (layer_dims[i + 1],)

        w = jax.random.normal(subkey, w_shape) / jnp.sqrt(layer_dims[i])
        b = jnp.zeros(b_shape)
        params["Ws"].append(w)
        params["bs"].append(b)

    return params


def mlp_apply(
    params: Dict[str, Any],
    x: jnp.ndarray,
    activation=jax.nn.tanh,
) -> jnp.ndarray:
    """
    Apply MLP to input x.

    params: {"Ws": [W0,...], "bs": [b0,...]}
    x: (..., in_dim)
    """
    Ws = params["Ws"]
    bs = params["bs"]
    depth = len(Ws) - 1  # number of hidden layers

    h = x
    for i in range(depth):
        h = activation(jnp.dot(h, Ws[i]) + bs[i])
    y = jnp.dot(h, Ws[-1]) + bs[-1]
    return y


# -----------------------------------------------------------------------------#
# Surrogate model: eq_params → (γ_fit, A_sat)
# -----------------------------------------------------------------------------#


@dataclass
class SurrogateModel:
    params: Dict[str, Any]
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    x_mean: jnp.ndarray | None = None
    x_std: jnp.ndarray | None = None
    y_mean: jnp.ndarray | None = None
    y_std: jnp.ndarray | None = None


def init_surrogate_model(
    rng_key,
    in_dim: int,
    out_dim: int = 2,
) -> SurrogateModel:
    mlp_params = init_mlp(
        rng_key,
        in_dim=in_dim,
        out_dim=out_dim,
        width=48,
        depth=2,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3, weight_decay=5e-3),
    )
    opt_state = optimizer.init(mlp_params)
    return SurrogateModel(
        params=mlp_params,
        optimizer=optimizer,
        opt_state=opt_state,
    )


def surrogate_loss(params, batch):
    eq_params, targets = batch
    preds = mlp_apply(params, eq_params, activation=jax.nn.swish)
    return jnp.mean((preds - targets) ** 2)


def make_surrogate_train_step(optimizer):
    @jax.jit
    def step(params, opt_state, batch):
        grads = jax.grad(surrogate_loss)(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        loss_val = surrogate_loss(new_params, batch)
        return new_params, new_opt_state, loss_val

    return step


def train_surrogate(
    eq_train: jnp.ndarray,
    y_train_0: jnp.ndarray,
    y_train_1: jnp.ndarray,
    eq_val: jnp.ndarray | None = None,
    y_val_0: jnp.ndarray | None = None,
    y_val_1: jnp.ndarray | None = None,
    n_epochs: int = 200,
    batch_size: int = 16,
    seed: int = 0,
) -> Tuple[SurrogateModel, np.ndarray, np.ndarray]:
    """
    Train surrogate on already-normalized inputs/targets.

    Here y_train_0/y_train_1 are the two components of the *normalized* targets.
    """
    rng = jax.random.PRNGKey(seed)
    N_train, P = eq_train.shape
    targets_train = jnp.stack([y_train_0, y_train_1], axis=-1)

    model = init_surrogate_model(rng, in_dim=P, out_dim=2)
    optimizer = model.optimizer
    surrogate_train_step = make_surrogate_train_step(optimizer)

    params = model.params
    opt_state = model.opt_state

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_params = params
    best_val_loss = float("inf")
    patience = 40  # epochs
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        perm = np.random.permutation(N_train)
        eq_shuf = eq_train[perm]
        tar_shuf = targets_train[perm]

        n_batches = int(math.ceil(N_train / batch_size))
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, N_train)
            batch_eq = eq_shuf[start:end]
            batch_tar = tar_shuf[start:end]

            params, opt_state, loss_val = surrogate_train_step(
                params, opt_state, (batch_eq, batch_tar)
            )
            epoch_loss += float(loss_val) * (end - start)

        epoch_loss /= N_train
        train_losses.append(epoch_loss)

        # Validation loss
        if (
            eq_val is not None
            and y_val_0 is not None
            and y_val_1 is not None
            and eq_val.shape[0] > 0
        ):
            targets_val = jnp.stack([y_val_0, y_val_1], axis=-1)
            preds_val = mlp_apply(params, eq_val, activation=jax.nn.swish)
            val_loss = jnp.mean((preds_val - targets_val) ** 2)
            val_loss_f = float(val_loss)
            val_losses.append(val_loss_f)

            # Early stopping logic
            if val_loss_f < best_val_loss - 1e-4:
                best_val_loss = val_loss_f
                best_params = params
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 20 == 0 or epoch == 1 or epoch == n_epochs:
                print(
                    f"[Surrogate] Epoch {epoch:4d}, "
                    f"MSE_train={epoch_loss:.3e}, MSE_val={val_loss_f:.3e}"
                )

            if patience_counter >= patience:
                print(
                    f"[Surrogate] Early stopping at epoch {epoch}, "
                    f"best MSE_val={best_val_loss:.3e}"
                )
                break

        else:
            if epoch % 20 == 0 or epoch == 1 or epoch == n_epochs:
                print(f"[Surrogate] Epoch {epoch:4d}, MSE_train={epoch_loss:.3e}")

    # Use the best parameters found on the validation set (if any)
    if val_losses:
        model.params = best_params
    else:
        model.params = params
    model.opt_state = opt_state
    return model, np.array(train_losses), np.array(val_losses)



def surrogate_predict(
    model: SurrogateModel,
    eq_raw: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convenience helper for inference.

    eq_raw: (..., P) in *original* feature space (before normalization).

    Returns:
        gamma_pred, A_sat_pred with shapes matching eq_raw[..., 0]
    """
    x_mean = model.x_mean
    x_std = model.x_std
    y_mean = model.y_mean
    y_std = model.y_std
    assert x_mean is not None and x_std is not None
    assert y_mean is not None and y_std is not None

    eq_n = (eq_raw - x_mean) / x_std
    y_n = mlp_apply(model.params, eq_n, activation=jax.nn.swish)
    y_raw = y_n * y_std + y_mean  # (.., 2)
    gamma_log = y_raw[..., 0]
    Asat_log = y_raw[..., 1]
    gamma = jnp.exp(gamma_log) - LABEL_EPS
    Asat = jnp.exp(Asat_log) - LABEL_EPS
    return gamma, Asat


# -----------------------------------------------------------------------------#
# Latent ODE autoencoder for amplitude time series
# -----------------------------------------------------------------------------#


@dataclass
class LatentODEModel:
    encoder_params: Dict[str, Any]
    prior_params: Dict[str, Any]
    ode_params: Dict[str, Any]
    decoder_params: Dict[str, Any]
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    logA_mean: float = 0.0
    logA_std: float = 1.0


def init_latent_ode_model(
    rng_key,
    amp_len: int,
    eq_dim: int,
    latent_dim: int = 3,
) -> LatentODEModel:
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)

    # Encoder: (log_amp_norm(t), eq_params) -> z0_enc
    encoder_in_dim = amp_len + eq_dim
    encoder_params = init_mlp(
        k1,
        in_dim=encoder_in_dim,
        out_dim=latent_dim,
        width=64,
        depth=3,
    )

    # Prior: eq_params -> z0_prior
    prior_params = init_mlp(
        k2,
        in_dim=eq_dim,
        out_dim=latent_dim,
        width=64,
        depth=3,
    )

    # ODE field: z -> dz/dt  (shared dynamics; no explicit param dependence)
    ode_in_dim = latent_dim
    ode_params = init_mlp(
        k3,
        in_dim=ode_in_dim,
        out_dim=latent_dim,
        width=64,
        depth=3,
    )

    # Decoder: (z, eq_params) -> log_amp_norm_hat
    dec_in_dim = latent_dim + eq_dim
    decoder_params = init_mlp(
        k4,
        in_dim=dec_in_dim,
        out_dim=1,
        width=64,
        depth=3,
    )

    dummy_tree = dict(
        encoder_params=encoder_params,
        prior_params=prior_params,
        ode_params=ode_params,
        decoder_params=decoder_params,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=3e-4),
    )
    opt_state = optimizer.init(dummy_tree)

    return LatentODEModel(
        encoder_params=encoder_params,
        prior_params=prior_params,
        ode_params=ode_params,
        decoder_params=decoder_params,
        optimizer=optimizer,
        opt_state=opt_state,
    )


# --- Single-trajectory forward (for diagnostics / plotting) ------------------#


def latent_ode_rhs_single(t, z, ode_params):
    dz = mlp_apply(ode_params, z)
    return dz


def latent_ode_forward_single(
    model: LatentODEModel,
    ts: jnp.ndarray,
    amp: jnp.ndarray,
    eq_params: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    One trajectory:
      - encode z0_enc from (log_amp_norm, eq_params)
      - integrate z(t) with shared latent dynamics
      - decode to log_amp_hat(t) (un-normalized)

    Returns:
        log_amp_hat(t), z_traj(t)
          log_amp_hat : (T,)
          z_traj      : (T, latent_dim)
    """
    log_amp = safe_log(amp)
    log_amp_n = (log_amp - model.logA_mean) / model.logA_std

    # Encoder (uses entire normalized time series + params)
    enc_input = jnp.concatenate([log_amp_n, eq_params], axis=-1)
    z0_enc = mlp_apply(model.encoder_params, enc_input)

    # (We still build a prior inside the loss, so we don't need it here.)

    term = dfx.ODETerm(latent_ode_rhs_single)
    solver = dfx.Tsit5()
    t0 = ts[0]
    t1 = ts[-1]
    dt0 = ts[1] - ts[0] if ts.size > 1 else 0.1
    saveat = dfx.SaveAt(ts=ts)

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=z0_enc,
        args=model.ode_params,
        saveat=saveat,
        max_steps=10_000,
    )
    z_traj = sol.ys  # (T, latent_dim)

    def decode_step(z_t):
        dec_input = jnp.concatenate([z_t, eq_params], axis=-1)
        log_amp_hat_n = mlp_apply(model.decoder_params, dec_input)[0]
        return model.logA_mean + model.logA_std * log_amp_hat_n

    log_amp_hat = jax.vmap(decode_step)(z_traj)  # (T,)
    return log_amp_hat, z_traj


# --- Batched ODE for training ------------------------------------------------#


def latent_ode_model_to_pytree(model: LatentODEModel):
    return dict(
        encoder_params=model.encoder_params,
        prior_params=model.prior_params,
        ode_params=model.ode_params,
        decoder_params=model.decoder_params,
    )


def latent_ode_from_pytree(tree, base_model: LatentODEModel) -> LatentODEModel:
    return LatentODEModel(
        encoder_params=tree["encoder_params"],
        prior_params=tree["prior_params"],
        ode_params=tree["ode_params"],
        decoder_params=tree["decoder_params"],
        optimizer=base_model.optimizer,
        opt_state=base_model.opt_state,
        logA_mean=base_model.logA_mean,
        logA_std=base_model.logA_std,
    )


def latent_ode_rhs_batched(t, z, ode_params):
    """
    Batched RHS: z has shape (B, latent_dim)
    """
    dz = jax.vmap(mlp_apply, in_axes=(None, 0))(ode_params, z)  # (B, latent_dim)
    return dz


def latent_ode_loss_batched(
    params_tree,
    ts: jnp.ndarray,
    amp_batch: jnp.ndarray,
    eq_batch: jnp.ndarray,
    logA_mean: float,
    logA_std: float,
    w_recon: float = 1.0,
    w_prior: float = 1e-1,
) -> jnp.ndarray:
    """
    Batched latent ODE loss:
      - encoder takes full normalized log_amp(t) and eq_params
      - ODE is integrated for the entire batch in one diffrax call
      - decoder reconstructs normalized log_amp_hat(t) for all samples
    """
    encoder_params = params_tree["encoder_params"]
    prior_params = params_tree["prior_params"]
    ode_params = params_tree["ode_params"]
    decoder_params = params_tree["decoder_params"]

    B, T = amp_batch.shape
    _, P = eq_batch.shape

    log_amp = safe_log(amp_batch)  # (B, T)
    log_amp_n = (log_amp - logA_mean) / logA_std

    # Encoder input: concatenate along feature dimension → (B, T+P)
    enc_input = jnp.concatenate([log_amp_n, eq_batch], axis=1)
    z0_enc = jax.vmap(mlp_apply, in_axes=(None, 0))(
        encoder_params, enc_input
    )  # (B, latent_dim)

    # Prior: eq_params -> z0_prior
    z0_prior = jax.vmap(mlp_apply, in_axes=(None, 0))(prior_params, eq_batch)

    # Batched ODE integration
    term = dfx.ODETerm(latent_ode_rhs_batched)
    solver = dfx.Tsit5()
    t0 = ts[0]
    t1 = ts[-1]
    dt0 = ts[1] - ts[0] if ts.size > 1 else 0.1
    saveat = dfx.SaveAt(ts=ts)

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=z0_enc,  # (B, latent_dim)
        args=ode_params,
        saveat=saveat,
        max_steps=10_000,
    )
    z_traj = sol.ys  # (T, B, latent_dim)

    # Decode each time slice
    def decode_time_step(z_t):  # z_t: (B, latent_dim)
        dec_input = jnp.concatenate([z_t, eq_batch], axis=1)  # (B, latent+P)
        log_amp_hat_n_t = jax.vmap(mlp_apply, in_axes=(None, 0))(
            decoder_params, dec_input
        )  # (B,1)
        return log_amp_hat_n_t[:, 0]  # (B,)
    
    log_amp_hat_n_TB = jax.vmap(decode_time_step)(z_traj)  # (T, B)
    log_amp_hat_n = log_amp_hat_n_TB.T  # (B, T)

    # --- time-weighted reconstruction loss: emphasize nonlinear / late times
    T = log_amp_n.shape[1]
    w_t = jnp.linspace(0.5, 1.8, T)          # slightly larger weight near t = t1
    w_t = w_t / jnp.mean(w_t)                # normalize so <w_t> = 1
    T = log_amp_n.shape[1]

    recon = jnp.mean(w_t[None, :] * (log_amp_hat_n - log_amp_n) ** 2)
    prior_pen = jnp.mean((z0_prior - z0_enc) ** 2)
    return w_recon * recon + w_prior * prior_pen



def make_latent_ode_train_step(
    optimizer,
    ts: jnp.ndarray,
    logA_mean: float,
    logA_std: float,
    w_recon: float = 1.0,
    w_prior: float = 1e-1,
):
    @jax.jit
    def step(params_tree, opt_state, amp_batch, eq_batch):
        def loss_fn(pytree):
            return latent_ode_loss_batched(
                pytree,
                ts,
                amp_batch,
                eq_batch,
                logA_mean,
                logA_std,
                w_recon=w_recon,
                w_prior=w_prior,
            )

        loss_val, grads = jax.value_and_grad(loss_fn)(params_tree)
        updates, new_opt_state = optimizer.update(grads, opt_state, params_tree)
        new_params_tree = optax.apply_updates(params_tree, updates)
        return new_params_tree, new_opt_state, loss_val

    return step


def train_latent_ode(
    ts: jnp.ndarray,
    amp: jnp.ndarray,
    eq_params: jnp.ndarray,
    latent_dim: int = 2,
    n_epochs: int = 200,
    batch_size: int = 8,
    seed: int = 1,
) -> Tuple[LatentODEModel, np.ndarray]:
    rng = jax.random.PRNGKey(seed)
    N, T = amp.shape
    _, P = eq_params.shape

    # Global statistics of log A over the dataset
    log_amp_all = safe_log(amp)
    logA_mean = jnp.mean(log_amp_all)
    logA_std = jnp.std(log_amp_all) + 1e-8

    model = init_latent_ode_model(rng, amp_len=T, eq_dim=P, latent_dim=latent_dim)
    params_tree = latent_ode_model_to_pytree(model)
    optimizer = model.optimizer
    opt_state = model.opt_state

    latent_train_step = make_latent_ode_train_step(
        optimizer, ts, logA_mean, logA_std, w_recon=1.0, w_prior=1e-2
    )

    epoch_losses: List[float] = []

    for epoch in range(1, n_epochs + 1):
        perm = np.random.permutation(N)
        amp_shuf = amp[perm]
        eq_shuf = eq_params[perm]

        n_batches = int(math.ceil(N / batch_size))
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, N)
            batch_amp = amp_shuf[start:end]
            batch_eq = eq_shuf[start:end]

            params_tree, opt_state, loss_val = latent_train_step(
                params_tree, opt_state, batch_amp, batch_eq
            )
            epoch_loss += float(loss_val) * (end - start)

        epoch_loss /= N
        epoch_losses.append(epoch_loss)
        if epoch % 20 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"[Latent ODE] Epoch {epoch:4d}, loss={epoch_loss:.3e}")

    new_model = latent_ode_from_pytree(params_tree, model)
    new_model.opt_state = opt_state
    new_model.logA_mean = float(logA_mean)
    new_model.logA_std = float(logA_std)
    return new_model, np.array(epoch_losses)


# -----------------------------------------------------------------------------#
# Plotting utilities
# -----------------------------------------------------------------------------#


def plot_surrogate_curves(
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    outpath: str = "surrogate_loss_curves.png",
):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.semilogy(epochs, train_losses, label="Train")
    if val_losses.size > 0:
        plt.semilogy(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_surrogate_parity(
    gamma_train_true,
    gamma_train_pred,
    gamma_val_true,
    gamma_val_pred,
    Asat_train_true,
    Asat_train_pred,
    Asat_val_true,
    Asat_val_pred,
    outpath: str = "surrogate_parity.png",
):
    plt.figure(figsize=(10, 4))

    # γ_fit parity
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(
        gamma_train_true, gamma_train_pred, marker="o", label="Train", alpha=0.8
    )
    if gamma_val_true is not None and len(gamma_val_true) > 0:
        ax1.scatter(
            gamma_val_true, gamma_val_pred, marker="s", label="Val", alpha=0.8
        )
    all_gamma = np.concatenate(
        [
            gamma_train_true,
            gamma_val_true if gamma_val_true is not None else [],
        ]
    )
    if all_gamma.size > 0:
        gmin, gmax = all_gamma.min(), all_gamma.max()
        pad = 0.05 * (gmax - gmin + 1e-8)
        ax1.plot(
            [gmin - pad, gmax + pad],
            [gmin - pad, gmax + pad],
            "k--",
            lw=1,
        )
    ax1.set_xlabel(r"$\gamma_{\mathrm{true}}$")
    ax1.set_ylabel(r"$\gamma_{\mathrm{pred}}$")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # A_sat parity
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(
        Asat_train_true, Asat_train_pred, marker="o", label="Train", alpha=0.8
    )
    if Asat_val_true is not None and len(Asat_val_true) > 0:
        ax2.scatter(
            Asat_val_true, Asat_val_pred, marker="s", label="Val", alpha=0.8
        )
    all_Asat = np.concatenate(
        [
            Asat_train_true,
            Asat_val_true if Asat_val_true is not None else [],
        ]
    )
    if all_Asat.size > 0:
        amin, amax = all_Asat.min(), all_Asat.max()
        pad = 0.05 * (amax - amin + 1e-8)
        ax2.plot(
            [amin - pad, amax + pad],
            [amin - pad, amax + pad],
            "k--",
            lw=1,
        )
    ax2.set_xlabel(r"$A_{\mathrm{sat,true}}$")
    ax2.set_ylabel(r"$A_{\mathrm{sat,pred}}$")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_latent_ode_reconstructions(
    ts: jnp.ndarray,
    amp: jnp.ndarray,
    eq_params: jnp.ndarray,
    latent_model: LatentODEModel,
    indices: List[int],
    outpath: str = "latent_ode_recon.png",
):
    ts_np = np.array(ts)
    n_cases = len(indices)
    plt.figure(figsize=(5 * n_cases, 4))

    for idx_plot, idx in enumerate(indices):
        amp_i = amp[idx]
        eq_i = eq_params[idx]
        log_amp_hat_i, _ = latent_ode_forward_single(
            latent_model, ts, amp_i, eq_i
        )
        amp_hat_i = jnp.exp(log_amp_hat_i)

        ax = plt.subplot(1, n_cases, idx_plot + 1)
        ax.plot(ts_np, np.array(amp_i), label="True", lw=2)
        ax.plot(ts_np, np.array(amp_hat_i), "--", label="Reconstructed", lw=2)
        ax.set_xlabel(r"$t$")
        if idx_plot == 0:
            ax.set_ylabel(r"$A(t)$")
        ax.set_title(f"Case {idx}")
        ax.grid(True, alpha=0.3)
        if idx_plot == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------#
# Save/load helpers  (only save params & stats, not optimizers/closures)
# -----------------------------------------------------------------------------#


def save_surrogate_model(
    model: SurrogateModel,
    path: str = "surrogate_model.pkl",
):
    """Save surrogate params + normalization stats."""
    payload = dict(
        params=model.params,
        x_mean=np.array(model.x_mean),
        x_std=np.array(model.x_std),
        y_mean=np.array(model.y_mean),
        y_std=np.array(model.y_std),
    )
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_surrogate_model(
    path: str = "surrogate_model.pkl",
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-3,
) -> SurrogateModel:
    """Load surrogate params and rebuild the optimizer + opt_state."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    params = payload["params"]
    x_mean = jnp.array(payload["x_mean"])
    x_std = jnp.array(payload["x_std"])
    y_mean = jnp.array(payload["y_mean"])
    y_std = jnp.array(payload["y_std"])

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(params)
    return SurrogateModel(
        params=params,
        optimizer=optimizer,
        opt_state=opt_state,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )


def save_latent_ode_model(
    model: LatentODEModel,
    path: str = "latent_ode_model.pkl",
):
    """Save latent ODE parameter PyTree + logA stats."""
    params_tree = latent_ode_model_to_pytree(model)
    payload = dict(
        params_tree=params_tree,
        logA_mean=model.logA_mean,
        logA_std=model.logA_std,
    )
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_latent_ode_model(
    path: str = "latent_ode_model.pkl",
    learning_rate: float = 1e-3,
) -> LatentODEModel:
    """Load latent ODE params and rebuild optimizer + opt_state."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    params_tree = payload["params_tree"]
    logA_mean = payload["logA_mean"]
    logA_std = payload["logA_std"]

    encoder_params = params_tree["encoder_params"]
    prior_params = params_tree["prior_params"]
    ode_params = params_tree["ode_params"]
    decoder_params = params_tree["decoder_params"]

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=learning_rate),
    )

    opt_state = optimizer.init(params_tree)

    return LatentODEModel(
        encoder_params=encoder_params,
        prior_params=prior_params,
        ode_params=ode_params,
        decoder_params=decoder_params,
        optimizer=optimizer,
        opt_state=opt_state,
        logA_mean=logA_mean,
        logA_std=logA_std,
    )


# -----------------------------------------------------------------------------#
# Main driver: build dataset, train surrogate + latent ODE
# -----------------------------------------------------------------------------#


def main():
    # Root folder for all ML outputs
    output_dir = "output_ml"
    os.makedirs(output_dir, exist_ok=True)

    # Subfolder for raw MHD runs
    runs_dir = os.path.join(output_dir, "tearing_ml_runs")
    os.makedirs(runs_dir, exist_ok=True)

    # 1) Build a training database (all physics is in mhd_tearing_solve.py)
    grid = GridConfig(Nx=32, Ny=32, Nz=1, Lx=2 * math.pi, Ly=2 * math.pi, Lz=2 * math.pi)

    print("=== Generating tearing database (v2, using mhd_tearing_solve.py) ===")
    t0 = 0.0
    t1 = 40.0
    n_frames = 100
    n_cases = 48  # adjust depending on HPC resources

    t_start = time.time()
    data = build_dataset(
        grid,
        n_cases=n_cases,
        t0=t0,
        t1=t1,
        n_frames=n_frames,
        seed=0,
        outdir=runs_dir,
        verbose=True,
    )
    print(f"[DATASET] Generation took {time.time() - t_start:.2f} s")

    ts = data["ts"]
    amp = data["amp"]
    gamma_fit = data["gamma_fit"]
    A_sat = data["A_sat"]
    eq_params = data["eq_params"]

    # 2) Train/val split for surrogate (in feature space)
    N = eq_params.shape[0]
    rng_np = np.random.default_rng(0)
    idx_all = rng_np.permutation(N)
    n_train = max(2, int(0.75 * N))
    train_idx = idx_all[:n_train]
    val_idx = idx_all[n_train:]

    eq_train_raw = eq_params[train_idx]
    gamma_train_raw = gamma_fit[train_idx]
    A_train_raw = A_sat[train_idx]

    eq_val_raw = eq_params[val_idx] if val_idx.size > 0 else None
    gamma_val_raw = gamma_fit[val_idx] if val_idx.size > 0 else None
    A_val_raw = A_sat[val_idx] if val_idx.size > 0 else None

    # Standardize inputs (features) based on training set
    x_mean = jnp.mean(eq_train_raw, axis=0)
    x_std = jnp.std(eq_train_raw, axis=0) + 1e-8
    eq_train = (eq_train_raw - x_mean) / x_std
    if eq_val_raw is not None:
        eq_val = (eq_val_raw - x_mean) / x_std
    else:
        eq_val = None

    # Log-transform and standardize targets (γ, A_sat)
    y_train_raw = jnp.stack(
        [
            safe_log(gamma_train_raw + LABEL_EPS),
            safe_log(A_train_raw + LABEL_EPS),
        ],
        axis=-1,
    )  # (N_train, 2)
    y_mean = jnp.mean(y_train_raw, axis=0)
    y_std = jnp.std(y_train_raw, axis=0) + 1e-8
    y_train_n = (y_train_raw - y_mean) / y_std

    if gamma_val_raw is not None:
        y_val_raw = jnp.stack(
            [
                safe_log(gamma_val_raw + LABEL_EPS),
                safe_log(A_val_raw + LABEL_EPS),
            ],
            axis=-1,
        )
        y_val_n = (y_val_raw - y_mean) / y_std
    else:
        y_val_n = None

    # 3) Train surrogate for (γ_fit, A_sat)
    print("\n=== Training surrogate MLP (features -> log γ, log A_sat) ===")
    if y_val_n is not None:
        surrogate, train_losses, val_losses = train_surrogate(
            eq_train,
            y_train_n[:, 0],
            y_train_n[:, 1],
            eq_val=eq_val,
            y_val_0=y_val_n[:, 0],
            y_val_1=y_val_n[:, 1],
            n_epochs=200,
            batch_size=16,
            seed=0,
        )
    else:
        surrogate, train_losses, val_losses = train_surrogate(
            eq_train,
            y_train_n[:, 0],
            y_train_n[:, 1],
            eq_val=None,
            y_val_0=None,
            y_val_1=None,
            n_epochs=200,
            batch_size=16,
            seed=0,
        )

    # Attach normalization stats to surrogate model
    surrogate.x_mean = x_mean
    surrogate.x_std = x_std
    surrogate.y_mean = y_mean
    surrogate.y_std = y_std

    # Diagnostics on train/val sets in physical space
    # Training set
    gamma_pred_train, Asat_pred_train = surrogate_predict(surrogate, eq_train_raw)

    if eq_val_raw is not None:
        gamma_pred_val, Asat_pred_val = surrogate_predict(surrogate, eq_val_raw)
    else:
        gamma_pred_val = np.array([])
        Asat_pred_val = np.array([])

    print("\n[Surrogate] Training set diagnostics:")
    print(f"  γ_fit true (train) : {np.array(gamma_train_raw)}")
    print(f"  γ_fit pred (train) : {np.array(gamma_pred_train)}")
    print(f"  A_sat true (train) : {np.array(A_train_raw)}")
    print(f"  A_sat pred (train) : {np.array(Asat_pred_train)}")

    if eq_val_raw is not None:
        print("\n[Surrogate] Validation set diagnostics:")
        print(f"  γ_fit true (val)   : {np.array(gamma_val_raw)}")
        print(f"  γ_fit pred (val)   : {np.array(gamma_pred_val)}")
        print(f"  A_sat true (val)   : {np.array(A_val_raw)}")
        print(f"  A_sat pred (val)   : {np.array(Asat_pred_val)}")

    # 4) Surrogate plots
    plot_surrogate_curves(
        train_losses,
        val_losses,
        outpath=os.path.join(output_dir, "surrogate_loss_curves.png"),
    )
    plot_surrogate_parity(
        np.array(gamma_train_raw),
        np.array(gamma_pred_train),
        np.array(gamma_val_raw) if eq_val_raw is not None else None,
        np.array(gamma_pred_val) if eq_val_raw is not None else None,
        np.array(A_train_raw),
        np.array(Asat_pred_train),
        np.array(A_val_raw) if eq_val_raw is not None else None,
        np.array(Asat_pred_val) if eq_val_raw is not None else None,
        outpath=os.path.join(output_dir, "surrogate_parity.png"),
    )

    # 5) Train latent ODE on full dataset (using raw feature vectors)
    print("\n=== Training latent ODE autoencoder for A(t) ===")
    latent_model, latent_losses = train_latent_ode(
        ts,
        amp,
        eq_params,
        latent_dim=3,
        n_epochs=200,
        batch_size=8,
        seed=1,
    )

    # Latent ODE training curve
    plt.figure(figsize=(6, 4))
    epochs_latent = np.arange(1, len(latent_losses) + 1)
    plt.semilogy(epochs_latent, latent_losses, label="Latent ODE train loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalized log A)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_ode_loss_curve.png"), dpi=300)
    plt.close()

    # 6) Example reconstructions
    print("\n=== Example latent ODE reconstructions ===")
    n_examples = min(3, N)
    example_indices = idx_all[:n_examples].tolist()
    for idx in example_indices:
        amp_i = amp[idx]
        eq_i = eq_params[idx]
        log_amp_hat_i, _ = latent_ode_forward_single(latent_model, ts, amp_i, eq_i)
        amp_hat_i = jnp.exp(log_amp_hat_i)
        print(f"  Case {idx}:")
        print("    t[0:5]      =", np.array(ts[:5]))
        print("    A_true[0:5] =", np.array(amp_i[:5]))
        print("    A_hat[0:5]  =", np.array(amp_hat_i[:5]))

    plot_latent_ode_reconstructions(
        ts,
        amp,
        eq_params,
        latent_model,
        indices=example_indices,
        outpath=os.path.join(output_dir, "latent_ode_recon.png"),
    )

    # 7) Save models for later reuse
    save_surrogate_model(
        surrogate,
        path=os.path.join(output_dir, "surrogate_model.pkl"),
    )
    save_latent_ode_model(
        latent_model,
        path=os.path.join(output_dir, "latent_ode_model.pkl"),
    )
    print(
        "\n[IO] Saved models to:",
        os.path.join(output_dir, "surrogate_model.pkl"),
        "and",
        os.path.join(output_dir, "latent_ode_model.pkl"),
    )

    print("\nDone. You now have in", output_dir + ":")
    print("  - surrogate_model.pkl: physics features -> (γ_fit, A_sat)")
    print("  - latent_ode_model.pkl: latent ODE autoencoder for A(t)")
    print("  - plots: surrogate_loss_curves.png, surrogate_parity.png,")
    print("           latent_ode_loss_curve.png, latent_ode_recon.png")
    print("  - raw MHD runs: tearing_ml_runs/tearing_ml_case_*.npz (from mhd_tearing_solve.py)")


if __name__ == "__main__":
    main()
