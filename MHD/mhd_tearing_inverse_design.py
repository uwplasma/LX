#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_inverse_design.py

End-to-end differentiable inverse design for tearing-mediated reconnection.

We use the differentiable pseudo-spectral MHD tearing solver
(mhd_tearing_solve.py) as a *layer* inside a neural network:

    z  --(MLP g_theta)-->  (log10_eta, log10_nu)
                       ->  (eta, nu)
                       ->  MHD simulation
                       ->  reconnection metrics (f_kin, C_plasmoid)

and train the MLP parameters theta by *backpropagating through the MHD
simulation* so that the reconnection metrics match a desired target:

    y* = (f_kin*, C_plasmoid*)

This script runs the inverse design, saves:
  - baseline (mid-range) run,
  - mid-training run,
  - final designed run,
  - training history for post-processing,

and produces publication-ready diagnostic figures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import equinox as eqx
import optax

from mhd_tearing_solve import (
    _run_tearing_simulation_and_diagnostics,
    plasmoid_complexity_metric,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class InverseDesignConfig:
    # Grid and box
    Nx: int = 64
    Ny: int = 64
    Nz: int = 1
    Lx: float = 2.0 * math.pi
    Ly: float = 2.0 * math.pi
    Lz: float = 2.0 * math.pi

    # Fixed physical parameters
    B0: float = 1.0
    B_g: float = 0.2
    a: float = 0.25
    eps_B: float = 1e-3

    # Time integration
    t0: float = 0.0
    t1: float = 60.0
    n_frames: int = 150
    dt0: float = 5e-4

    # Equilibrium
    equilibrium_mode: str = "forcefree"   # "forcefree" or "original"

    # Target reconnection behaviour (values are overridden in main()
    # depending on equilibrium_mode)
    target_f_kin: float = 0.03
    target_complexity: float = 1e-5

    # Trade-off between matching f_kin and complexity
    lambda_complexity: float = 1.0

    # Bounds for eta and nu (log10-space) 
    log10_eta_min: float = -4.5
    log10_eta_max: float = -2.0
    log10_nu_min: float = -4.5
    log10_nu_max: float = -2.0

    # Neural network + training hyperparameters
    latent_dim: int = 1                  # dimension of latent design variable z
    hidden_width: int = 32
    hidden_depth: int = 2
    learning_rate: float = 1e-3
    n_train_steps: int = 25              # each step runs a full simulation
    print_every: int = 1

    # Latent design value to train at (scalar)
    z_train: float = 0.0

    # Random seed
    seed: int = 1234

# -----------------------------------------------------------------------------
# Small neural network: design MLP (manual stack of Linear layers)
# -----------------------------------------------------------------------------

class DesignMLP(eqx.Module):
    """MLP mapping latent design z -> (log10_eta, log10_nu)."""

    layers: List[eqx.nn.Linear]
    activation: Any = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden_width: int, hidden_depth: int,
                 key: jax.random.PRNGKey):
        # We build:
        #   Linear(in_dim -> hidden_width)
        #   (hidden_depth-1) × Linear(hidden_width -> hidden_width)
        #   Linear(hidden_width -> 2)
        keys = jax.random.split(key, hidden_depth + 1)

        layers: List[eqx.nn.Linear] = []
        # input -> hidden
        layers.append(eqx.nn.Linear(in_dim, hidden_width, key=keys[0]))
        # hidden -> hidden (hidden_depth-1 times)
        for i in range(hidden_depth - 1):
            layers.append(eqx.nn.Linear(hidden_width, hidden_width, key=keys[i + 1]))
        # last hidden -> 2 outputs (log10_eta, log10_nu)
        layers.append(eqx.nn.Linear(hidden_width, 2, key=keys[-1]))

        self.layers = layers
        self.activation = jax.nn.tanh

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.

        For the current config we take latent_dim = 1, so:
          - z is a scalar or shape (1,)
          - we treat it as a 1D vector of length in_dim
        No batch dimension is used; each Linear sees a vector of shape (in_features,).
        """
        x = jnp.asarray(z, dtype=jnp.float64)

        # Ensure x has shape (in_dim,) rather than scalar
        if x.ndim == 0:
            x = jnp.expand_dims(x, 0)   # (1,)

        # Pass through hidden layers
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))   # stays 1D

        # Final linear layer -> 2 outputs
        x = self.layers[-1](x)             # shape (2,)

        return x  # (log10_eta, log10_nu)


# -----------------------------------------------------------------------------
# MHD simulation wrapper: metrics for given (eta, nu)
# -----------------------------------------------------------------------------

def _simulate_metrics(eta: jnp.ndarray,
                      nu: jnp.ndarray,
                      cfg: InverseDesignConfig) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """
    Run the tearing simulation and return:

        f_kin, complexity, res

    where
      - f_kin: late-time kinetic-energy fraction
      - complexity: plasmoid complexity metric from A_z midplane at final time
      - res: full simulation result dict
    """
    res = _run_tearing_simulation_and_diagnostics(
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        Nz=cfg.Nz,
        Lx=cfg.Lx,
        Ly=cfg.Ly,
        Lz=cfg.Lz,
        nu=nu,
        eta=eta,
        B0=cfg.B0,
        a=cfg.a,
        B_g=cfg.B_g,
        eps_B=cfg.eps_B,
        t0=cfg.t0,
        t1=cfg.t1,
        n_frames=cfg.n_frames,
        dt0=cfg.dt0,
        equilibrium_mode=cfg.equilibrium_mode,
    )

    ts = res["ts"]
    E_kin = res["E_kin"]
    E_mag = res["E_mag"]
    Az_final_mid = res["Az_final_mid"]

    # Average kinetic-energy fraction over last 30% of the simulation
    T = ts.shape[0]
    i0 = int(0.7 * (T - 1))
    E_kin_tail = E_kin[i0:]
    E_mag_tail = E_mag[i0:]

    E_kin_mean = jnp.mean(E_kin_tail)
    E_mag_mean = jnp.mean(E_mag_tail)
    E_tot_mean = E_kin_mean + E_mag_mean + 1e-30
    f_kin = E_kin_mean / E_tot_mean

    complexity = plasmoid_complexity_metric(Az_final_mid)

    gamma_fit = res["gamma_fit"]
    return f_kin, complexity, gamma_fit, res


def _squash_to_interval(raw: jnp.ndarray, xmin: float, xmax: float) -> jnp.ndarray:
    """
    Map an unconstrained 'raw' value to [xmin, xmax] smoothly using tanh.

    raw ~ O(1)  -> near the middle of the interval
    large |raw| -> asymptotically approach bounds, but with nonzero gradients.
    """
    center = 0.5 * (xmin + xmax)
    half_width = 0.5 * (xmax - xmin)
    return center + half_width * jnp.tanh(raw)


# -----------------------------------------------------------------------------
# Loss function and training step
# -----------------------------------------------------------------------------

def make_loss_fn(cfg: InverseDesignConfig):
    """
    Build a loss function:

      L(theta) = (f_kin - f_kin*)^2 + lambda * (C_plasmoid - C*)^2

    where (eta, nu) = 10^{g_theta(z)} and the MHD simulation gives
    (f_kin, C_plasmoid).
    """

    target = jnp.array([cfg.target_f_kin, cfg.target_complexity], dtype=jnp.float64)
    z_train = jnp.array(cfg.z_train, dtype=jnp.float64)

    def loss_fn(model: DesignMLP, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        # Forward pass through MLP
        raw_log10_eta, raw_log10_nu = model(z_train)  # unconstrained outputs
        log10_eta = _squash_to_interval(raw_log10_eta, cfg.log10_eta_min, cfg.log10_eta_max)
        log10_nu  = _squash_to_interval(raw_log10_nu,  cfg.log10_nu_min,  cfg.log10_nu_max)

        # Convert to physical parameters
        eta = 10.0**log10_eta
        nu  = 10.0**log10_nu

        # Run MHD simulation and get metrics (differentiable!)
        f_kin, complexity, gamma_fit, res = _simulate_metrics(eta, nu, cfg)

        diff_f = f_kin - target[0]
        diff_c = complexity - target[1]

        loss = diff_f**2 + cfg.lambda_complexity * diff_c**2

        # Debug printing (AD-safe)
        jax.debug.print(
            "[LOSS] log10_eta={logeta:.3f}, log10_nu={lognu:.3f}, "
            "eta={eta:.3e}, nu={nu:.3e}, f_kin={f_kin:.4f}, "
            "complexity={comp:.3e}, L={loss:.3e}",
            logeta=log10_eta,
            lognu=log10_nu,
            eta=eta,
            nu=nu,
            f_kin=f_kin,
            comp=complexity,
            loss=loss,
        )

        aux = {
            "log10_eta": log10_eta,
            "log10_nu": log10_nu,
            "eta": eta,
            "nu": nu,
            "f_kin": f_kin,
            "complexity": complexity,
        }
        return loss, aux

    return loss_fn


def build_training_step(
    cfg: InverseDesignConfig,
    optimizer: optax.GradientTransformation,
    static_model: DesignMLP,
):
    """Build a jitted training step that updates only the trainable params."""
    loss_fn = make_loss_fn(cfg)

    # loss as a function of (params, key), with static_model closed over
    def loss_from_params(params, key):
        # Rebuild full model from params + static parts
        model = eqx.combine(params, static_model)
        loss, aux = loss_fn(model, key)
        return loss, aux

    # Take gradient w.r.t. `params` only, but keep aux
    grad_fn = jax.value_and_grad(loss_from_params, argnums=0, has_aux=True)

    @jax.jit
    def step(params, opt_state, key):
        # Compute loss and gradients w.r.t. params
        (loss_val, aux), grads = grad_fn(params, key)

        # Optax update on params only
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss_val, aux

    return step


# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------

def plot_training_history(history: Dict[str, List[float]]):
    steps = np.arange(len(history["loss"]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    axes[0, 0].semilogy(steps, history["loss"], marker="o")
    axes[0, 0].set_xlabel("training step")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].set_title("Inverse-design loss")

    axes[0, 1].plot(steps, history["log10_eta"], marker="o", label=r"$\log_{10}\eta$")
    axes[0, 1].plot(steps, history["log10_nu"], marker="s", label=r"$\log_{10}\nu$")
    axes[0, 1].set_xlabel("training step")
    axes[0, 1].set_ylabel("log10 parameters")
    axes[0, 1].set_title("Dissipation parameters")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, history["f_kin"], marker="o")
    axes[1, 0].set_xlabel("training step")
    axes[1, 0].set_ylabel(r"$f_{\mathrm{kin}}$")
    axes[1, 0].set_title("Kinetic energy fraction (tail-averaged)")

    axes[1, 1].plot(steps, history["complexity"], marker="o")
    axes[1, 1].set_xlabel("training step")
    axes[1, 1].set_ylabel(r"$C_{\mathrm{plasmoid}}$")
    axes[1, 1].set_title("Plasmoid complexity (midplane)")

    fig.suptitle("Differentiable inverse design training history", fontsize=14)
    fig.savefig("inverse_design_training_history.png", dpi=300)
    print("[PLOT] Saved inverse_design_training_history.png")
    plt.close(fig)


def plot_energy_evolution(res_init: Dict[str, Any],
                          res_final: Dict[str, Any]):
    """Kinetic and magnetic energy vs time, initial vs designed."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for res, lab, color in [
        (res_init, "baseline", "C0"),
        (res_final, "designed", "C3"),
    ]:
        ts = np.array(res["ts"])
        E_kin = np.array(res["E_kin"])
        E_mag = np.array(res["E_mag"])

        axes[0].plot(ts, E_kin, label=lab, color=color)
        axes[1].plot(ts, E_mag, label=lab, color=color)

    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$E_{\mathrm{kin}}$")
    axes[0].set_title("Kinetic energy vs time")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$E_{\mathrm{mag}}$")
    axes[1].set_title("Magnetic energy vs time")
    axes[1].legend(fontsize=8)

    fig.suptitle("Energy evolution: baseline vs inversely-designed run", fontsize=14)
    fig.savefig("inverse_design_energy_evolution.png", dpi=300)
    print("[PLOT] Saved inverse_design_energy_evolution.png")
    plt.close(fig)


def plot_energy_fraction(res_init: Dict[str, Any],
                         res_final: Dict[str, Any]):
    """Plot f_kin(t) = E_kin / (E_kin + E_mag) for baseline vs designed."""
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)

    for res, lab, color in [
        (res_init, "baseline", "C0"),
        (res_final, "designed", "C3"),
    ]:
        ts = np.array(res["ts"])
        E_kin = np.array(res["E_kin"])
        E_mag = np.array(res["E_mag"])
        f_kin_t = E_kin / (E_kin + E_mag + 1e-30)
        ax.plot(ts, f_kin_t, label=lab, color=color)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$E_{\mathrm{kin}} / (E_{\mathrm{kin}} + E_{\mathrm{mag}})$")
    ax.set_title("Kinetic-energy fraction vs time")
    ax.legend(fontsize=8)
    fig.savefig("inverse_design_energy_fraction.png", dpi=300)
    print("[PLOT] Saved inverse_design_energy_fraction.png")
    plt.close(fig)


def plot_reconnection_rate(res_init: Dict[str, Any],
                           res_final: Dict[str, Any]):
    """Plot reconnection-rate proxy E_rec(t) from A_z at the X-point."""
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)

    for res, lab, color in [
        (res_init, "baseline", "C0"),
        (res_final, "designed", "C3"),
    ]:
        ts = np.array(res["ts"])
        E_rec = np.array(res["E_rec_series"])
        ax.plot(ts, E_rec, label=lab, color=color)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$E_{\mathrm{rec}}(t)$ (proxy)")
    ax.set_title("Reconnection-rate proxy from $A_z$ at X-point")
    ax.legend(fontsize=8)
    fig.savefig("inverse_design_reconnection_rate.png", dpi=300)
    print("[PLOT] Saved inverse_design_reconnection_rate.png")
    plt.close(fig)


def plot_tearing_growth(res_init: Dict[str, Any],
                        res_final: Dict[str, Any]):
    """
    Plot tearing mode amplitude and exponential fit, baseline vs designed.

    Shows |B_x(kx=0,ky=1,kz=0)|(t) on a log scale, with lnA_fit(t) overlaid
    and the linear-fit window shaded.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    for ax, (res, lab) in zip(
        axes,
        [(res_init, "baseline"), (res_final, "designed")],
    ):
        ts = np.array(res["ts"])
        mode_amp = np.array(res["mode_amp_series"])
        lnA_fit = np.array(res["lnA_fit"])
        mask_lin = np.array(res["mask_lin"]).astype(bool)

        A_fit = np.exp(lnA_fit)

        ax.semilogy(ts, mode_amp, label=r"$|B_x(k_x{=}0,k_y{=}1,k_z{=}0)|$", lw=1.8)
        ax.semilogy(ts, A_fit, "--", label=r"fit: $\exp(\gamma t)$", lw=1.8)

        # Shade the linear fit window
        if np.any(mask_lin):
            t_min = ts[mask_lin].min()
            t_max = ts[mask_lin].max()
            ax.axvspan(t_min, t_max, alpha=0.15, color="grey",
                       label="fit window")

        gamma_fit = float(np.array(res["gamma_fit"]))
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"mode amplitude")
        ax.set_title(f"Tearing growth ({lab}), $\\gamma_{{\\rm fit}}\\approx{gamma_fit:.3e}$")
        ax.legend(fontsize=8)

    fig.suptitle("Tearing-mode amplitude and fitted exponential growth", fontsize=14)
    fig.savefig("inverse_design_tearing_growth.png", dpi=300)
    print("[PLOT] Saved inverse_design_tearing_growth.png")
    plt.close(fig)


def plot_Az_midplane_profiles(res_init: Dict[str, Any],
                              res_final: Dict[str, Any]):
    """
    Plot A_z(y) on the midplane at final time, baseline vs designed.

    Uses Az_final_mid, which is A_z(x_mid, y, z=0, t=t_final).
    """
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    for res, lab, color in [
        (res_init, "baseline", "C0"),
        (res_final, "designed", "C3"),
    ]:
        Az_mid = np.array(res["Az_final_mid"])
        Ly = float(res["Ly"])
        Ny = Az_mid.shape[0]
        y = np.linspace(0.0, Ly, Ny, endpoint=False)
        complexity = float(np.array(res["complexity_final"]))

        ax.plot(y, Az_mid, label=f"{lab} (C≈{complexity:.2e})", color=color)

    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$A_z(x_{\mathrm{mid}},y,z{=}0,t_{\mathrm{final}})$")
    ax.set_title(r"Midplane flux function $A_z$ at final time")
    ax.legend(fontsize=8)
    fig.savefig("inverse_design_Az_midplane_profiles.png", dpi=300)
    print("[PLOT] Saved inverse_design_Az_midplane_profiles.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main():
    cfg = InverseDesignConfig()

    cfg.target_f_kin = 0.03
    cfg.target_complexity = 1e-5
    cfg.lambda_complexity = 1.0

    cfg.lambda_complexity = 1e3  # complexity term scaled to match f_kin variations

    print("========================================================")
    print(" Differentiable inverse design for tearing reconnection ")
    print("========================================================")
    print(cfg)

    # 1. Initialize MLP and optimizer
    key = jax.random.PRNGKey(cfg.seed)
    key_model, key_train = jax.random.split(key)

    model = DesignMLP(
        in_dim=cfg.latent_dim,
        hidden_width=cfg.hidden_width,
        hidden_depth=cfg.hidden_depth,
        key=key_model,
    )

    # Split model into trainable array params and static structure
    params, static_model = eqx.partition(model, eqx.is_array)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),    # NEW: gradient clipping
        optax.adam(cfg.learning_rate),
    )
    opt_state = optimizer.init(params)
    
    # Build jitted training step that closes over optimizer + static_model
    training_step = build_training_step(cfg, optimizer, static_model)

    # 2. Baseline simulation at mid-range (eta0, nu0) for comparison
    log10_eta0 = 0.5 * (cfg.log10_eta_min + cfg.log10_eta_max)
    log10_nu0  = 0.5 * (cfg.log10_nu_min  + cfg.log10_nu_max)
    eta0 = 10.0**log10_eta0
    nu0  = 10.0**log10_nu0

    print("\n[BASELINE] Running baseline simulation at mid-range (eta0, nu0)...")
    f_kin0, comp0, gamma_fit0, res_init = _simulate_metrics(
        jnp.array(eta0, dtype=jnp.float64),
        jnp.array(nu0, dtype=jnp.float64),
        cfg,
    )
    print(
        f"[BASELINE] log10_eta0={log10_eta0:.3f}, log10_nu0={log10_nu0:.3f}, "
        f"eta0={eta0:.3e}, nu0={nu0:.3e}, "
        f"f_kin0={float(f_kin0):.4f}, complexity0={float(comp0):.3e}"
    )
    # Save a dedicated NPZ for the initial run so it can be post-processed
    outfile = f"mhd_tearing_inverse_design_solution_initial_{cfg.equilibrium_mode}.npz"
    np.savez(outfile, **res_init)
    print(f"[SAVE] Saved initial solution to {outfile}")

    # Quick gradient check in (log10_eta, log10_nu) space around (-3,-3)
    def simple_loss(log10_eta, log10_nu):
        eta = 10.0**log10_eta
        nu  = 10.0**log10_nu
        f_kin, C, gamma_fit, _ = _simulate_metrics(eta, nu, cfg)
        return (f_kin - cfg.target_f_kin)**2 + cfg.lambda_complexity * (C - cfg.target_complexity)**2

    grad_eta, grad_nu = jax.grad(simple_loss, argnums=(0, 1))(jnp.array(-3.0), jnp.array(-3.0))
    print(f"Gradient at (-3,-3): dL/dlog10_eta={grad_eta:.3e}, dL/dlog10_nu={grad_nu:.3e}")

    # 3. Training loop (each step runs one full MHD simulation)
    history: Dict[str, List[float]] = {
        "loss": [],
        "log10_eta": [],
        "log10_nu": [],
        "eta": [],
        "nu": [],
        "f_kin": [],
        "complexity": [],
    }

    last_aux = None
    
    # Track best parameters by loss (early-stopping style)
    best_loss = float("inf")
    best_params = params
    best_aux = None

    print("\n[TRAIN] Starting inverse-design training loop...")
    for step in range(cfg.n_train_steps):
        key_train, key_step = jax.random.split(key_train)

        params, opt_state, loss_val, aux = training_step(
            params, opt_state, key_step
        )

        loss_float = float(loss_val)
        log10_eta = float(aux["log10_eta"])
        log10_nu  = float(aux["log10_nu"])
        eta       = float(aux["eta"])
        nu        = float(aux["nu"])
        f_kin     = float(aux["f_kin"])
        comp      = float(aux["complexity"])

        history["loss"].append(loss_float)
        history["log10_eta"].append(log10_eta)
        history["log10_nu"].append(log10_nu)
        history["eta"].append(eta)
        history["nu"].append(nu)
        history["f_kin"].append(f_kin)
        history["complexity"].append(comp)
        
        if loss_float < best_loss:
            best_loss = loss_float
            best_params = params
            best_aux = aux

        if (step % cfg.print_every) == 0:
            print(
                f"[STEP {step:03d}] "
                f"L={loss_float:.3e}, "
                f"log10_eta={log10_eta:.3f}, log10_nu={log10_nu:.3f}, "
                f"eta={eta:.3e}, nu={nu:.3e}, "
                f"f_kin={f_kin:.4f}, complexity={comp:.3e}"
            )

        last_aux = aux

    # Save training history for publication-grade postprocessing
    hist_path = f"inverse_design_history_{cfg.equilibrium_mode}.npz"
    np.savez(hist_path, **history)
    print(f"[SAVE] Saved inverse-design training history to {hist_path}")

    # 4. Final designed parameters from best training step (early stopping)
    assert best_aux is not None, "Training loop did not run."
    params = best_params          # use the best parameters, not the last ones
    last_aux = best_aux

    eta_final = float(last_aux["eta"])
    nu_final  = float(last_aux["nu"])
    print(
        "\n[FINAL] Designed parameters (best checkpoint): "
        f"eta={eta_final:.3e}, nu={nu_final:.3e}, best_loss={best_loss:.3e}"
    )

    # 4b. Mid-training run for diagnostics
    mid_index = len(history["eta"]) // 2
    eta_mid = history["eta"][mid_index]
    nu_mid = history["nu"][mid_index]
    print(
        f"[MID] Running mid-training simulation at step {mid_index}: "
        f"eta_mid={eta_mid:.3e}, nu_mid={nu_mid:.3e}"
    )
    f_kin_mid, comp_mid, gamma_fit_mid, res_mid = _simulate_metrics(
        jnp.array(eta_mid, dtype=jnp.float64),
        jnp.array(nu_mid, dtype=jnp.float64),
        cfg,
    )
    print(
        f"[MID] f_kin_mid={float(f_kin_mid):.4f}, "
        f"complexity_mid={float(comp_mid):.3e}"
    )
    outfile_mid = f"mhd_tearing_inverse_design_solution_mid_{cfg.equilibrium_mode}.npz"
    np.savez(outfile_mid, **res_mid)
    print(f"[SAVE] Saved mid-training solution to {outfile_mid}")

    # 4c. Final dedicated simulation at (eta_final, nu_final)
    f_kin_final, comp_final, gamma_fit_final, res_final = _simulate_metrics(
        jnp.array(eta_final, dtype=jnp.float64),
        jnp.array(nu_final, dtype=jnp.float64),
        cfg,
    )
    print(
        f"[FINAL] f_kin_final={float(f_kin_final):.4f}, "
        f"complexity_final={float(comp_final):.3e}"
    )

    # Rebuild the final DesignMLP from the trained params + static structure
    final_model = eqx.combine(params, static_model)

    # Save the trained design network to disk
    model_path = f"design_mlp_final_{cfg.equilibrium_mode}.eqx"
    eqx.tree_serialise_leaves(model_path, final_model)
    print(f"[SAVE] Saved trained DesignMLP to {model_path}")

    # 5. Save a dedicated NPZ for the final run so it can be post-processed
    outfile_final = f"mhd_tearing_inverse_design_solution_final_{cfg.equilibrium_mode}.npz"
    np.savez(outfile_final, **res_final)
    print(f"[SAVE] Saved final designed solution to {outfile_final}")

    # 6. Make plots (training + baseline vs designed)
    print("\n[PLOT] Making training and physics diagnostics plots...")
    plot_training_history(history)
    plot_energy_evolution(res_init, res_final)
    plot_energy_fraction(res_init, res_final)
    plot_reconnection_rate(res_init, res_final)
    plot_tearing_growth(res_init, res_final)
    plot_Az_midplane_profiles(res_init, res_final)

    print("\n[DONE] Inverse design script finished.")


if __name__ == "__main__":
    main()
