#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_inverse_design.py

End-to-end differentiable inverse design for tearing-mediated reconnection.

We use the differentiable pseudo-spectral MHD tearing solver
(mhd_tearing_solve.py) as a *layer* inside a neural network:

    z  --(MLP g_theta)-->  (log_eta, log_nu)
                       ->  (eta, nu)
                       ->  MHD simulation
                       ->  reconnection metrics (f_kin, C_plasmoid)

and train the MLP parameters theta by *backpropagating through the MHD
simulation* so that the reconnection metrics match a desired target:

    y* = (f_kin*, C_plasmoid*)

This is a minimal demonstration of "differentiable physics for inverse
design" applied to MHD tearing and plasmoid reconnection.

Outputs:
  - History plots:
      * inverse_design_training_history.png
  - Post-optimization simulation comparison:
      * inverse_design_energy_evolution.png
  - NPZ checkpoint for the final ("designed") run:
      * mhd_tearing_inverse_design_solution_final.npz

The final NPZ can be post-processed with:

    python mhd_tearing_postprocess.py mhd_tearing_inverse_design_solution_final.npz
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import jax
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
    n_frames: int = 120
    dt0: float = 5e-4

    # Equilibrium
    equilibrium_mode: str = "forcefree"   # plasmoid-prone; "original" also possible

    # Target reconnection behaviour
    # These are dimensionless, data-driven and should be tuned
    target_f_kin: float = 0.03           # desired late-time kinetic energy fraction
    target_complexity: float = 1e-5      # desired plasmoid complexity

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
    learning_rate: float = 1e-2
    n_train_steps: int = 10              # keep small initially (each step runs a full sim)
    print_every: int = 1

    # Latent design value to train at (scalar)
    z_train: float = 0.0

    # Random seed
    seed: int = 1234


# -----------------------------------------------------------------------------
# Small neural network: design MLP
# -----------------------------------------------------------------------------

class DesignMLP(eqx.Module):
    """MLP mapping latent design z -> (log10_eta, log10_nu)."""

    layers: List[eqx.nn.Linear]
    activation: Any = eqx.static_field()

    def __init__(self, in_dim: int, hidden_width: int, hidden_depth: int,
                 key: jax.random.PRNGKey):
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
        x = z
        # Ensure shape (..., in_dim)
        if x.ndim == 0:
            x = x[None]  # (1,)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        # x has shape (batch, 2) for batched input; we want (2,) for scalar z
        return x.squeeze(0)


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

    return f_kin, complexity, res


# -----------------------------------------------------------------------------
# Loss function and training step
# -----------------------------------------------------------------------------

def _clip_log10(x_log10: jnp.ndarray, xmin: float, xmax: float) -> jnp.ndarray:
    """Clip log10 values into [xmin, xmax] in a differentiable-friendly way."""
    return jnp.clip(x_log10, xmin, xmax)


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
        log10_eta_nu = model(z_train)  # shape (2,)
        log10_eta = _clip_log10(log10_eta_nu[0], cfg.log10_eta_min, cfg.log10_eta_max)
        log10_nu  = _clip_log10(log10_eta_nu[1], cfg.log10_nu_min,  cfg.log10_nu_max)

        # Convert to physical parameters
        eta = 10.0**log10_eta
        nu  = 10.0**log10_nu

        # Run MHD simulation and get metrics (differentiable!)
        f_kin, complexity, res = _simulate_metrics(eta, nu, cfg)

        # Loss
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
            "res": res,
        }
        return loss, aux

    return loss_fn


# Filtered versions for Equinox (only differentiate w.r.t. trainable params)
def build_training_step(cfg: InverseDesignConfig):
    loss_fn = make_loss_fn(cfg)

    @eqx.filter_value_and_grad
    def loss_and_grad(model: DesignMLP, key: jax.random.PRNGKey):
        return loss_fn(model, key)

    def step(model: DesignMLP,
             opt_state: optax.OptState,
             key: jax.random.PRNGKey,
             optimizer: optax.GradientTransformation):
        (loss_val, aux), grads = loss_and_grad(model, key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val, aux

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
    axes[1, 0].set_title("Kinetic energy fraction")

    axes[1, 1].plot(steps, history["complexity"], marker="o")
    axes[1, 1].set_xlabel("training step")
    axes[1, 1].set_ylabel(r"$C_{\mathrm{plasmoid}}$")
    axes[1, 1].set_title("Plasmoid complexity")

    fig.suptitle("Differentiable inverse design training history", fontsize=14)
    fig.savefig("inverse_design_training_history.png", dpi=300)
    print("[PLOT] Saved inverse_design_training_history.png")
    plt.close(fig)


def plot_energy_evolution(res_init: Dict[str, Any],
                          res_final: Dict[str, Any]):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for res, lab, color in [
        (res_init, "initial", "C0"),
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

    fig.suptitle("Energy evolution: initial vs inversely-designed run", fontsize=14)
    fig.savefig("inverse_design_energy_evolution.png", dpi=300)
    print("[PLOT] Saved inverse_design_energy_evolution.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main():
    cfg = InverseDesignConfig()

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

    optimizer = optax.adam(cfg.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    training_step = build_training_step(cfg)

    # 2. Optional: run a baseline simulation with some reference eta0, nu0
    #    (e.g., the central point in log10-space), for comparison.
    log10_eta0 = 0.5 * (cfg.log10_eta_min + cfg.log10_eta_max)
    log10_nu0  = 0.5 * (cfg.log10_nu_min  + cfg.log10_nu_max)
    eta0 = 10.0**log10_eta0
    nu0  = 10.0**log10_nu0

    print("\n[BASELINE] Running baseline simulation at mid-range (eta0, nu0)...")
    f_kin0, comp0, res_init = _simulate_metrics(eta0, nu0, cfg)
    print(
        f"[BASELINE] log10_eta0={log10_eta0:.3f}, log10_nu0={log10_nu0:.3f}, "
        f"eta0={eta0:.3e}, nu0={nu0:.3e}, "
        f"f_kin0={float(f_kin0):.4f}, complexity0={float(comp0):.3e}"
    )

    # 3. Training loop (each step runs one full MHD simulation)
    history = {
        "loss": [],
        "log10_eta": [],
        "log10_nu": [],
        "eta": [],
        "nu": [],
        "f_kin": [],
        "complexity": [],
    }

    last_aux = None

    print("\n[TRAIN] Starting inverse-design training loop...")
    for step in range(cfg.n_train_steps):
        key_train, key_step = jax.random.split(key_train)

        model, opt_state, loss_val, aux = training_step(
            model, opt_state, key_step, optimizer
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

        if (step % cfg.print_every) == 0:
            print(
                f"[STEP {step:03d}] "
                f"L={loss_float:.3e}, "
                f"log10_eta={log10_eta:.3f}, log10_nu={log10_nu:.3f}, "
                f"eta={eta:.3e}, nu={nu:.3e}, "
                f"f_kin={f_kin:.4f}, complexity={comp:.3e}"
            )

        last_aux = aux

    # 4. Final designed simulation result
    assert last_aux is not None, "Training loop did not run."
    res_final = last_aux["res"]
    eta_final = float(last_aux["eta"])
    nu_final  = float(last_aux["nu"])
    print(
        "\n[FINAL] Designed parameters: "
        f"eta={eta_final:.3e}, nu={nu_final:.3e}"
    )

    # 5. Save a dedicated NPZ for the final run so it can be post-processed
    #    by mhd_tearing_postprocess.py
    outfile = "mhd_tearing_inverse_design_solution_final.npz"
    np.savez(outfile, **res_final)
    print(f"[SAVE] Saved final designed solution to {outfile}")

    # 6. Make plots
    print("\n[PLOT] Making training and energy plots...")
    plot_training_history(history)
    plot_energy_evolution(res_init, res_final)

    print("\n[DONE] Inverse design script finished.")


if __name__ == "__main__":
    main()
