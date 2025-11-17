#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_ml.py

Data-driven reduced model for Harris-sheet tearing:
  1) Uses mhd_tearing_solve to generate a database of tearing evolutions
  2) Extracts diagnostics:
       - tearing amplitude A(t) (Bx RMS around current sheet)
       - linear growth rate γ_fit (from log A(t) in linear phase)
       - saturated amplitude A_sat
  3) Trains:
       - Supervised surrogate MLP: eq_params -> (γ_fit, A_sat)
       - Latent ODE "autoencoder" for A(t), conditioned on eq_params

This version:
  - Uses fully JAX/Optax training for both models
  - Batches and JITs the latent ODE loss with a single batched diffrax solve
  - Adds a train/validation split for the surrogate
  - Produces publication-ready plots:
      * surrogate train/val loss curves
      * surrogate parity plots (γ_fit, A_sat)
      * latent ODE reconstructions A(t) vs A_hat(t)
  - Saves trained models to disk (pickle) for later reuse.

Dependencies:
    - jax, jaxlib
    - diffrax
    - optax
    - numpy
    - matplotlib

Make sure mhd_tearing_solve.py is in the same directory and importable.
"""

from __future__ import annotations

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
# Import core MHD tools from your solver
# -----------------------------------------------------------------------------#

from mhd_tearing_solve import (
    make_k_arrays,
    make_dealias_mask,
    project_div_free,
    grad_vec_from_hat,
    directional_derivative_vec,
    init_equilibrium,
    energy_from_hat,
    estimate_max_dt,
    make_mhd_rhs,
    tearing_amplitude,
    fkr_gamma,
)

# -----------------------------------------------------------------------------#
# Simple utilities
# -----------------------------------------------------------------------------#

def safe_log(x, eps=1e-16):
    return jnp.log(jnp.clip(x, eps, None))

def fit_growth_rate(ts: jnp.ndarray,
                    amp: jnp.ndarray,
                    frac_window: Tuple[float, float] = (0.2, 0.6)
                    ) -> float:
    """
    Fit γ from A(t) ≈ A0 exp(γ t) using least-squares on log A.

    frac_window: use [t0 + f1*(t1-t0), t0 + f2*(t1-t0)] as "linear phase"
    """
    ts = jnp.asarray(ts)
    amp = jnp.asarray(amp)

    t0 = ts[0]
    t1 = ts[-1]
    t_start = t0 + frac_window[0] * (t1 - t0)
    t_end   = t0 + frac_window[1] * (t1 - t0)

    # primary window
    mask = (ts >= t_start) & (ts <= t_end)
    ts_sel = ts[mask]
    amp_sel = amp[mask]

    # if too few points, fall back to a broader middle window
    if ts_sel.shape[0] < 4:
        mid_mask = (ts >= (t0 + 0.25 * (t1 - t0))) & (ts <= (t0 + 0.75 * (t1 - t0)))
        ts_sel = ts[mid_mask]
        amp_sel = amp[mid_mask]

    # final safety: if still too small, just use all points
    if ts_sel.shape[0] < 2:
        ts_sel = ts
        amp_sel = amp

    y = safe_log(amp_sel)
    t_mean = jnp.mean(ts_sel)
    y_mean = jnp.mean(y)
    cov_ty = jnp.mean((ts_sel - t_mean) * (y - y_mean))
    var_t  = jnp.mean((ts_sel - t_mean) ** 2) + 1e-16
    gamma = cov_ty / var_t
    return float(gamma)

def compute_saturated_amplitude(amp: jnp.ndarray, frac_tail: float = 0.2) -> float:
    """
    Simple proxy: average of last frac_tail fraction of A(t).
    """
    n = amp.size
    start = int((1.0 - frac_tail) * n)
    tail = amp[start:]
    return float(jnp.mean(tail))

# -----------------------------------------------------------------------------#
# Run a single tearing simulation and extract diagnostics for ML
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

def run_single_sim(config: GridConfig,
                   phys: PhysicalParams,
                   t0: float = 0.0,
                   t1: float = 40.0,
                   n_frames: int = 80,
                   dt0_override: float | None = None,
                   verbose: bool = False,
                   ) -> Dict[str, Any]:
    """
    Run one tearing simulation and return:
      ts, amplitude(t), gamma_fit, A_sat, eq_param_vector
    """
    Nx, Ny, Nz = config.Nx, config.Ny, config.Nz
    Lx, Ly, Lz = config.Lx, config.Ly, config.Lz
    nu, eta = phys.nu, phys.eta
    B0, a, B_g, eps_B = phys.B0, phys.a, phys.B_g, phys.eps_B

    if verbose:
        print("=== Single tearing sim ===")
        print(config)
        print(phys)

    kx, ky, kz, k2, NX, NY, NZ = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    mask_dealias = make_dealias_mask(Nx, Ny, Nz, NX, NY, NZ)

    v0_real, B0_real = init_equilibrium(
        Nx, Ny, Nz, Lx, Ly, Lz,
        B0=B0, a=a, B_g=B_g, eps_B=eps_B
    )
    v0_hat = jnp.fft.fftn(v0_real, axes=(1, 2, 3))
    B0_hat = jnp.fft.fftn(B0_real, axes=(1, 2, 3))
    v0_hat = v0_hat * mask_dealias
    B0_hat = B0_hat * mask_dealias
    v0_hat = project_div_free(v0_hat, kx, ky, kz, k2)
    B0_hat = project_div_free(B0_hat, kx, ky, kz, k2)

    E_kin0, E_mag0 = energy_from_hat(v0_hat, B0_hat, Lx, Ly, Lz)
    if verbose:
        print(f"[INIT] E_kin0={float(E_kin0):.3e}, E_mag0={float(E_mag0):.3e}")

    if dt0_override is None:
        dt_max = estimate_max_dt(v0_hat, B0_hat, Lx, Ly, Lz, nu, eta)
        dt0 = min(1e-3, 0.5 * dt_max)
    else:
        dt0 = dt0_override

    rhs = make_mhd_rhs(nu, eta, kx, ky, kz, k2, mask_dealias)
    term = dfx.ODETerm(rhs)
    solver = dfx.Dopri8()
    ts_save = jnp.linspace(t0, t1, n_frames)
    saveat = dfx.SaveAt(ts=ts_save)
    controller = dfx.PIDController(rtol=1e-5, atol=1e-7)

    if verbose:
        print(f"[RUN] t0={t0}, t1={t1}, dt0≈{dt0:.3e}, n_frames={n_frames}")

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=(v0_hat, B0_hat),
        args=None,
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=int((t1 - t0) / dt0) + 20_000,
    )
    ts = jnp.array(sol.ts)
    v_hat_frames, B_hat_frames = sol.ys

    # Compute tearing amplitude A(t)
    amp_list: List[float] = []
    for k in range(n_frames):
        amp_k = tearing_amplitude(B_hat_frames[k], Lx, Ly, Lz)
        amp_list.append(amp_k)
    amp = jnp.array(amp_list)

    gamma_fit = fit_growth_rate(ts, amp)
    A_sat = compute_saturated_amplitude(amp)

    # equilibrium feature vector for ML
    eq_params = jnp.array([
        B0,
        a,
        B_g,
        eps_B,
        nu,
        eta,
        Ly,
    ], dtype=jnp.float64)

    return dict(
        ts=ts,
        amp=amp,
        gamma_fit=gamma_fit,
        A_sat=A_sat,
        eq_params=eq_params,
    )

# -----------------------------------------------------------------------------#
# Build a training database via a small parameter scan
# -----------------------------------------------------------------------------#

def build_dataset(config: GridConfig,
                  n_cases: int = 16,
                  t0: float = 0.0,
                  t1: float = 40.0,
                  n_frames: int = 80,
                  seed: int = 0,
                  verbose: bool = True,
                  ) -> Dict[str, jnp.ndarray]:
    """
    Sample a small parameter space and build:
      - eq_params: (N, P)
      - ts:        (T,)  (shared)
      - amp:       (N, T)
      - gamma_fit: (N,)
      - A_sat:     (N,)
    """
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []

    for i in range(n_cases):
        # Sample physically reasonable parameters around a baseline
        B0 = 1.0
        a = config.Lx / 16.0 * rng.uniform(0.7, 1.3)
        B_g = 0.2 * rng.uniform(0.7, 1.3)
        eps_B = 0.01 * rng.uniform(0.5, 2.0)
        nu = 1e-3 * rng.uniform(0.3, 3.0)
        eta = 1e-3 * rng.uniform(0.3, 3.0)

        phys = PhysicalParams(nu=nu, eta=eta, B0=B0, a=a, B_g=B_g, eps_B=eps_B)

        if verbose:
            print(f"\n[SCAN] Case {i+1}/{n_cases}: {phys}")

        rec = run_single_sim(
            config=config,
            phys=phys,
            t0=t0,
            t1=t1,
            n_frames=n_frames,
            dt0_override=None,
            verbose=False,
        )
        records.append(rec)

    # Stack to JAX arrays
    ts = records[0]["ts"]
    amp = jnp.stack([r["amp"] for r in records], axis=0)         # (N, T)
    gamma_fit = jnp.array([r["gamma_fit"] for r in records])     # (N,)
    A_sat = jnp.array([r["A_sat"] for r in records])             # (N,)
    eq_params = jnp.stack([r["eq_params"] for r in records], 0)  # (N, P)

    if verbose:
        print("\n[DATASET] Built dataset:")
        print(f"  eq_params: {eq_params.shape}")
        print(f"  amp:       {amp.shape}")
        print(f"  gamma_fit: {gamma_fit.shape}")
        print(f"  A_sat:     {A_sat.shape}")

    return dict(
        ts=ts,
        amp=amp,
        gamma_fit=gamma_fit,
        A_sat=A_sat,
        eq_params=eq_params,
    )

# -----------------------------------------------------------------------------#
# Generic JAX MLP utilities (params as PyTrees)
# -----------------------------------------------------------------------------#

def init_mlp(rng_key,
             in_dim: int,
             out_dim: int,
             width: int = 64,
             depth: int = 2) -> Dict[str, Any]:
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


def mlp_apply(params: Dict[str, Any],
              x: jnp.ndarray,
              activation=jax.nn.tanh) -> jnp.ndarray:
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

def init_surrogate_model(rng_key, in_dim: int, out_dim: int = 2) -> SurrogateModel:
    mlp_params = init_mlp(rng_key, in_dim=in_dim, out_dim=out_dim,
                          width=64, depth=3)
    optimizer = optax.adam(learning_rate=5e-3)
    opt_state = optimizer.init(mlp_params)
    return SurrogateModel(params=mlp_params, optimizer=optimizer, opt_state=opt_state)

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

def train_surrogate(eq_train: jnp.ndarray,
                    gamma_train: jnp.ndarray,
                    A_train: jnp.ndarray,
                    eq_val: jnp.ndarray | None = None,
                    gamma_val: jnp.ndarray | None = None,
                    A_val: jnp.ndarray | None = None,
                    n_epochs: int = 200,
                    batch_size: int = 16,
                    seed: int = 0) -> Tuple[SurrogateModel, np.ndarray, np.ndarray]:

    rng = jax.random.PRNGKey(seed)
    N_train, P = eq_train.shape
    targets_train = jnp.stack([gamma_train, A_train], axis=-1)

    model = init_surrogate_model(rng, in_dim=P, out_dim=2)
    optimizer = model.optimizer
    surrogate_train_step = make_surrogate_train_step(optimizer)

    params = model.params
    opt_state = model.opt_state

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, n_epochs + 1):
        perm = np.random.permutation(N_train)
        eq_shuf = eq_train[perm]
        tar_shuf = targets_train[perm]

        n_batches = int(math.ceil(N_train / batch_size))
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min((b+1) * batch_size, N_train)
            batch_eq = eq_shuf[start:end]
            batch_tar = tar_shuf[start:end]

            params, opt_state, loss_val = surrogate_train_step(
                params, opt_state, (batch_eq, batch_tar)
            )
            epoch_loss += float(loss_val) * (end - start)

        epoch_loss /= N_train
        train_losses.append(epoch_loss)

        # Validation loss
        if eq_val is not None and gamma_val is not None and A_val is not None:
            targets_val = jnp.stack([gamma_val, A_val], axis=-1)
            preds_val = mlp_apply(params, eq_val, activation=jax.nn.swish)
            val_loss = jnp.mean((preds_val - targets_val) ** 2)
            val_losses.append(float(val_loss))
            if epoch % 20 == 0 or epoch == 1 or epoch == n_epochs:
                print(f"[Surrogate] Epoch {epoch:4d}, "
                      f"MSE_train={epoch_loss:.3e}, MSE_val={float(val_loss):.3e}")
        else:
            if epoch % 20 == 0 or epoch == 1 or epoch == n_epochs:
                print(f"[Surrogate] Epoch {epoch:4d}, MSE_train={epoch_loss:.3e}")

    model.params = params
    model.opt_state = opt_state
    return model, np.array(train_losses), np.array(val_losses)

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

def init_latent_ode_model(rng_key,
                          amp_len: int,
                          eq_dim: int,
                          latent_dim: int = 4) -> LatentODEModel:
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)

    # Encoder: (log_amp(t), eq_params) -> z0_enc
    encoder_in_dim = amp_len + eq_dim
    encoder_params = init_mlp(k1, in_dim=encoder_in_dim,
                              out_dim=latent_dim,
                              width=64, depth=2)

    # Prior: eq_params -> z0_prior
    prior_params = init_mlp(k2, in_dim=eq_dim,
                            out_dim=latent_dim,
                            width=64, depth=2)

    # ODE field: (z, eq_params) -> dz/dt
    ode_in_dim = latent_dim + eq_dim
    ode_params = init_mlp(k3, in_dim=ode_in_dim,
                          out_dim=latent_dim,
                          width=64, depth=2)

    # Decoder: (z, eq_params) -> log_amp_hat
    dec_in_dim = latent_dim + eq_dim
    decoder_params = init_mlp(k4, in_dim=dec_in_dim,
                              out_dim=1,
                              width=64, depth=2)

    dummy_tree = dict(
        encoder_params=encoder_params,
        prior_params=prior_params,
        ode_params=ode_params,
        decoder_params=decoder_params,
    )
    optimizer = optax.adam(learning_rate=1e-3)
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

def latent_ode_rhs_single(t, z, args):
    ode_params, eq_params = args
    inp = jnp.concatenate([z, eq_params], axis=-1)
    dz = mlp_apply(ode_params, inp)
    return dz

def latent_ode_forward_single(model: LatentODEModel,
                              ts: jnp.ndarray,
                              amp: jnp.ndarray,
                              eq_params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    One trajectory:
      - encode z0_enc from (log_amp, eq_params)
      - integrate z(t)
      - decode to log_amp_hat(t)
      Returns:
        log_amp_hat(t), z0_prior
    """
    log_amp = safe_log(amp)
    enc_input = jnp.concatenate([log_amp, eq_params], axis=-1)
    z0_enc = mlp_apply(model.encoder_params, enc_input)

    # Prior (param-only initial condition)
    z0_prior = mlp_apply(model.prior_params, eq_params)

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
        args=(model.ode_params, eq_params),
        saveat=saveat,
        max_steps=10_000,
    )
    z_traj = sol.ys  # (T, latent_dim)

    def decode_step(z_t):
        dec_input = jnp.concatenate([z_t, eq_params], axis=-1)
        return mlp_apply(model.decoder_params, dec_input)[0]

    log_amp_hat = jax.vmap(decode_step)(z_traj)  # (T,)
    return log_amp_hat, z0_prior

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
    )

def latent_ode_rhs_batched(t, z, args):
    """
    Batched RHS: z has shape (B, latent_dim)
    args = (ode_params, eq_batch) where eq_batch is (B, P)
    """
    ode_params, eq_batch = args
    inp = jnp.concatenate([z, eq_batch], axis=-1)  # (B, latent+P)
    # vmap over batch dimension
    dz = jax.vmap(mlp_apply, in_axes=(None, 0))(ode_params, inp)  # (B, latent_dim)
    return dz

def latent_ode_loss_batched(params_tree,
                            ts: jnp.ndarray,
                            amp_batch: jnp.ndarray,
                            eq_batch: jnp.ndarray,
                            w_recon: float = 1.0,
                            w_prior: float = 1e-2) -> jnp.ndarray:
    """
    Batched latent ODE loss:
      - encoder takes full log_amp(t) and eq_params as a single vector per sample
      - ODE is integrated for the entire batch in one diffrax call
      - decoder reconstructs log_amp_hat(t) for all samples
    """
    encoder_params = params_tree["encoder_params"]
    prior_params = params_tree["prior_params"]
    ode_params = params_tree["ode_params"]
    decoder_params = params_tree["decoder_params"]

    B, T = amp_batch.shape
    _, P = eq_batch.shape

    log_amp = safe_log(amp_batch)  # (B, T)
    # Encoder input: concatenate along feature dimension → (B, T+P)
    enc_input = jnp.concatenate([log_amp, eq_batch], axis=1)
    z0_enc = jax.vmap(mlp_apply, in_axes=(None, 0))(encoder_params, enc_input)  # (B, latent_dim)

    # Prior: eq_params -> z0_prior
    z0_prior = jax.vmap(mlp_apply, in_axes=(None, 0))(prior_params, eq_batch)   # (B, latent_dim)

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
        y0=z0_enc,                        # (B, latent_dim)
        args=(ode_params, eq_batch),      # eq_batch: (B, P)
        saveat=saveat,
        max_steps=10_000,
    )
    z_traj = sol.ys  # (T, B, latent_dim)

    # Decode each time slice
    def decode_time_step(z_t):  # z_t: (B, latent_dim)
        dec_input = jnp.concatenate([z_t, eq_batch], axis=1)  # (B, latent+P)
        log_amp_hat_t = jax.vmap(mlp_apply, in_axes=(None, 0))(decoder_params, dec_input)  # (B,1)
        return log_amp_hat_t[:, 0]  # (B,)

    log_amp_hat_TB = jax.vmap(decode_time_step)(z_traj)  # (T, B)
    log_amp_hat = log_amp_hat_TB.T  # (B, T)

    recon = jnp.mean((log_amp_hat - log_amp) ** 2)
    prior_pen = jnp.mean((z0_prior - z0_enc) ** 2)
    return w_recon * recon + w_prior * prior_pen

def make_latent_ode_train_step(optimizer,
                               ts: jnp.ndarray,
                               w_recon: float = 1.0,
                               w_prior: float = 1e-2):
    @jax.jit
    def step(params_tree, opt_state, amp_batch, eq_batch):
        def loss_fn(pytree):
            return latent_ode_loss_batched(pytree, ts, amp_batch, eq_batch,
                                           w_recon=w_recon, w_prior=w_prior)
        loss_val, grads = jax.value_and_grad(loss_fn)(params_tree)
        updates, new_opt_state = optimizer.update(grads, opt_state, params_tree)
        new_params_tree = optax.apply_updates(params_tree, updates)
        return new_params_tree, new_opt_state, loss_val
    return step

def train_latent_ode(ts: jnp.ndarray,
                     amp: jnp.ndarray,
                     eq_params: jnp.ndarray,
                     latent_dim: int = 4,
                     n_epochs: int = 200,
                     batch_size: int = 8,
                     seed: int = 1) -> Tuple[LatentODEModel, np.ndarray]:
    rng = jax.random.PRNGKey(seed)
    N, T = amp.shape
    _, P = eq_params.shape

    model = init_latent_ode_model(rng, amp_len=T, eq_dim=P, latent_dim=latent_dim)
    params_tree = latent_ode_model_to_pytree(model)
    optimizer = model.optimizer
    opt_state = model.opt_state

    latent_train_step = make_latent_ode_train_step(optimizer, ts)

    epoch_losses: List[float] = []

    for epoch in range(1, n_epochs + 1):
        perm = np.random.permutation(N)
        amp_shuf = amp[perm]
        eq_shuf = eq_params[perm]

        n_batches = int(math.ceil(N / batch_size))
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min((b+1) * batch_size, N)
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
    return new_model, np.array(epoch_losses)

# -----------------------------------------------------------------------------#
# Plotting utilities
# -----------------------------------------------------------------------------#

def plot_surrogate_curves(train_losses: np.ndarray,
                          val_losses: np.ndarray,
                          outpath: str = "surrogate_loss_curves.png"):
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

def plot_surrogate_parity(gamma_train_true, gamma_train_pred,
                          gamma_val_true, gamma_val_pred,
                          Asat_train_true, Asat_train_pred,
                          Asat_val_true, Asat_val_pred,
                          outpath: str = "surrogate_parity.png"):
    plt.figure(figsize=(10, 4))

    # γ_fit parity
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(gamma_train_true, gamma_train_pred, marker="o", label="Train", alpha=0.8)
    if gamma_val_true is not None and len(gamma_val_true) > 0:
        ax1.scatter(gamma_val_true, gamma_val_pred, marker="s", label="Val", alpha=0.8)
    all_gamma = np.concatenate([gamma_train_true,
                                gamma_val_true if gamma_val_true is not None else []])
    if all_gamma.size > 0:
        gmin, gmax = all_gamma.min(), all_gamma.max()
        pad = 0.05 * (gmax - gmin + 1e-8)
        ax1.plot([gmin - pad, gmax + pad], [gmin - pad, gmax + pad], "k--", lw=1)
    ax1.set_xlabel(r"$\gamma_{\mathrm{true}}$")
    ax1.set_ylabel(r"$\gamma_{\mathrm{pred}}$")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # A_sat parity
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(Asat_train_true, Asat_train_pred, marker="o", label="Train", alpha=0.8)
    if Asat_val_true is not None and len(Asat_val_true) > 0:
        ax2.scatter(Asat_val_true, Asat_val_pred, marker="s", label="Val", alpha=0.8)
    all_Asat = np.concatenate([Asat_train_true,
                               Asat_val_true if Asat_val_true is not None else []])
    if all_Asat.size > 0:
        amin, amax = all_Asat.min(), all_Asat.max()
        pad = 0.05 * (amax - amin + 1e-8)
        ax2.plot([amin - pad, amax + pad], [amin - pad, amax + pad], "k--", lw=1)
    ax2.set_xlabel(r"$A_{\mathrm{sat,true}}$")
    ax2.set_ylabel(r"$A_{\mathrm{sat,pred}}$")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_latent_ode_reconstructions(ts: jnp.ndarray,
                                    amp: jnp.ndarray,
                                    eq_params: jnp.ndarray,
                                    latent_model: LatentODEModel,
                                    indices: List[int],
                                    outpath: str = "latent_ode_recon.png"):
    ts_np = np.array(ts)
    n_cases = len(indices)
    plt.figure(figsize=(5 * n_cases, 4))

    for idx_plot, idx in enumerate(indices):
        amp_i = amp[idx]
        eq_i = eq_params[idx]
        log_amp_hat_i, _ = latent_ode_forward_single(latent_model, ts, amp_i, eq_i)
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
# Save/load helpers  (only save params, not optimizers/closures)
# -----------------------------------------------------------------------------#

def save_surrogate_model(model: SurrogateModel, path: str = "surrogate_model.pkl"):
    """Save only the surrogate params as a PyTree."""
    payload = dict(params=model.params)
    with open(path, "wb") as f:
        pickle.dump(payload, f)

def load_surrogate_model(path: str = "surrogate_model.pkl",
                         learning_rate: float = 5e-3) -> SurrogateModel:
    """Load surrogate params and rebuild the optimizer + opt_state."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    params = payload["params"]
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    return SurrogateModel(params=params, optimizer=optimizer, opt_state=opt_state)


def save_latent_ode_model(model: LatentODEModel, path: str = "latent_ode_model.pkl"):
    """Save only the latent ODE parameter PyTree."""
    params_tree = latent_ode_model_to_pytree(model)
    payload = dict(params_tree=params_tree)
    with open(path, "wb") as f:
        pickle.dump(payload, f)

def load_latent_ode_model(path: str = "latent_ode_model.pkl",
                          learning_rate: float = 1e-3) -> LatentODEModel:
    """Load latent ODE params and rebuild optimizer + opt_state."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    params_tree = payload["params_tree"]

    # Rebuild LatentODEModel from the params_tree and a fresh optimizer
    encoder_params     = params_tree["encoder_params"]
    prior_params       = params_tree["prior_params"]
    ode_params         = params_tree["ode_params"]
    decoder_params     = params_tree["decoder_params"]

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params_tree)

    return LatentODEModel(
        encoder_params=encoder_params,
        prior_params=prior_params,
        ode_params=ode_params,
        decoder_params=decoder_params,
        optimizer=optimizer,
        opt_state=opt_state,
    )

# -----------------------------------------------------------------------------#
# Main driver: build dataset, train surrogate + latent ODE
# -----------------------------------------------------------------------------#

def main():
    # 1) Build a small training database
    grid = GridConfig(Nx=32, Ny=32, Nz=1, Lx=2*math.pi, Ly=2*math.pi, Lz=2*math.pi)

    print("=== Generating tearing database ===")
    t0 = 0.0
    t1 = 40.0
    n_frames = 80
    n_cases = 16  # increase as needed / HPC allows

    t_start = time.time()
    data = build_dataset(grid,
                         n_cases=n_cases,
                         t0=t0,
                         t1=t1,
                         n_frames=n_frames,
                         seed=0,
                         verbose=True)
    print(f"[DATASET] Generation took {time.time() - t_start:.2f} s")

    ts        = data["ts"]
    amp       = data["amp"]
    gamma_fit = data["gamma_fit"]
    A_sat     = data["A_sat"]
    eq_params = data["eq_params"]

    # 2) Train/val split for surrogate
    N = eq_params.shape[0]
    rng_np = np.random.default_rng(0)
    idx_all = rng_np.permutation(N)
    n_train = max(1, int(0.75 * N))
    train_idx = idx_all[:n_train]
    val_idx   = idx_all[n_train:]

    eq_train = eq_params[train_idx]
    gamma_train = gamma_fit[train_idx]
    A_train = A_sat[train_idx]

    eq_val = eq_params[val_idx] if val_idx.size > 0 else None
    gamma_val = gamma_fit[val_idx] if val_idx.size > 0 else None
    A_val = A_sat[val_idx] if val_idx.size > 0 else None

    # 3) Train surrogate for (γ_fit, A_sat)
    print("\n=== Training surrogate MLP (eq_params -> γ_fit, A_sat) ===")
    surrogate, train_losses, val_losses = train_surrogate(
        eq_train, gamma_train, A_train,
        eq_val=eq_val, gamma_val=gamma_val, A_val=A_val,
        n_epochs=200, batch_size=8, seed=0
    )

    # Diagnostics on train/val sets
    preds_train = mlp_apply(surrogate.params, eq_train, activation=jax.nn.swish)
    gamma_pred_train = np.array(preds_train[:, 0])
    Asat_pred_train = np.array(preds_train[:, 1])

    if eq_val is not None:
        preds_val = mlp_apply(surrogate.params, eq_val, activation=jax.nn.swish)
        gamma_pred_val = np.array(preds_val[:, 0])
        Asat_pred_val = np.array(preds_val[:, 1])
    else:
        gamma_pred_val = np.array([])
        Asat_pred_val = np.array([])

    print("\n[Surrogate] Training set diagnostics:")
    print(f"  γ_fit true (train) : {np.array(gamma_train)}")
    print(f"  γ_fit pred (train) : {gamma_pred_train}")
    print(f"  A_sat true (train) : {np.array(A_train)}")
    print(f"  A_sat pred (train) : {Asat_pred_train}")

    if eq_val is not None:
        print("\n[Surrogate] Validation set diagnostics:")
        print(f"  γ_fit true (val)   : {np.array(gamma_val)}")
        print(f"  γ_fit pred (val)   : {gamma_pred_val}")
        print(f"  A_sat true (val)   : {np.array(A_val)}")
        print(f"  A_sat pred (val)   : {Asat_pred_val}")

    # 4) Surrogate plots
    plot_surrogate_curves(train_losses, val_losses,
                          outpath="surrogate_loss_curves.png")
    plot_surrogate_parity(
        np.array(gamma_train), gamma_pred_train,
        np.array(gamma_val) if eq_val is not None else None,
        gamma_pred_val if eq_val is not None else None,
        np.array(A_train), Asat_pred_train,
        np.array(A_val) if eq_val is not None else None,
        Asat_pred_val if eq_val is not None else None,
        outpath="surrogate_parity.png",
    )

    # 5) Train latent ODE on full dataset
    print("\n=== Training latent ODE autoencoder for A(t) ===")
    latent_model, latent_losses = train_latent_ode(
        ts, amp, eq_params,
        latent_dim=4,
        n_epochs=200,
        batch_size=4,
        seed=1,
    )

    # (Optional) latent ODE training curve
    plt.figure(figsize=(6, 4))
    epochs_latent = np.arange(1, len(latent_losses) + 1)
    plt.semilogy(epochs_latent, latent_losses, label="Latent ODE train loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log A)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("latent_ode_loss_curve.png", dpi=300)
    plt.close()

    # 6) Example reconstructions
    print("\n=== Example latent ODE reconstructions ===")
    # choose up to 3 cases (mix of train/val)
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
        ts, amp, eq_params, latent_model,
        indices=example_indices,
        outpath="latent_ode_recon.png",
    )

    # 7) Save models for later reuse
    save_surrogate_model(surrogate, path="surrogate_model.pkl")
    save_latent_ode_model(latent_model, path="latent_ode_model.pkl")
    print("\n[IO] Saved models to 'surrogate_model.pkl' and 'latent_ode_model.pkl'.")

    print("\nDone. You now have:")
    print("  - surrogate_model.pkl: eq_params -> (γ_fit, A_sat)")
    print("  - latent_ode_model.pkl: latent ODE autoencoder for A(t)")
    print("  - plots: surrogate_loss_curves.png, surrogate_parity.png,")
    print("           latent_ode_loss_curve.png, latent_ode_recon.png")

if __name__ == "__main__":
    main()
