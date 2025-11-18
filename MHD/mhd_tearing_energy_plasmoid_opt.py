#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_energy_plasmoid_opt.py

Differentiable reconnection design:
-----------------------------------
Use JAX autodiff + the Harris-sheet MHD tearing solver
(mhd_tearing_solve.py) to *optimize* dissipation parameters (eta, nu) so that:

  1) The *fraction* of kinetic energy at late times is large, and
  2) The midplane flux function A_z exhibits strong fine-scale structure
     (a smooth proxy for plasmoid richness).

We optimize over the vector of variables:

    theta = [log_eta, log_nu]

and define a scalar objective:

    score(theta) = alpha * f_kin(theta) + beta * C_plasmoid(theta)

    J(theta) = - score(theta)          (we minimize J)

where
  - f_kin is the averaged kinetic-energy fraction near saturation,
  - C_plasmoid is the mean-squared curvature of A_z on the midplane at
    final time (computed by plasmoid_complexity_metric).

This script:
  1) Runs the MHD solver for each (eta, nu) via _run_tearing_simulation_and_diagnostics.
  2) Uses the JAX-based plasmoid_complexity_metric and energy traces to
     build the objective.
  3) Performs gradient descent on log(eta), log(nu).
  4) Produces publication-ready plots:
       - optimization history,
       - energy evolution,
       - midplane A_z profile (initial vs optimized).

Usage:
  python mhd_tearing_energy_plasmoid_opt.py

Edit EnergyPlasmoidConfig below for resolution, weights, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, List

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mhd_tearing_solve import (
    _run_tearing_simulation_and_diagnostics,
    plasmoid_complexity_metric,
)


# --------------------------- Configuration dataclass -------------------------#

@dataclass
class EnergyPlasmoidConfig:
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
    a: float = 0.25          # fixed current-sheet half-width
    eps_B: float = 1e-3

    # Time integration
    t0: float = 0.0
    t1: float = 60.0
    n_frames: int = 120
    dt0: float = 5e-4

    # Objective weights (alpha for kinetic fraction, beta for complexity)
    alpha: float = 1.0
    beta: float = 0.5

    # Optimization hyperparameters
    n_opt_steps: int = 15
    lr_log_eta: float = 0.5
    lr_log_nu: float = 0.5

    # Initial guesses
    eta0: float = 1e-3
    nu0: float = 1e-3

    equilibrium_mode: str = "forcefree"


# --------------------------- Objective functional ---------------------------#

def _simulate_energy_plasmoid(theta: jnp.ndarray, cfg: EnergyPlasmoidConfig):
    """
    Run the tearing simulation for given theta=[log_eta, log_nu] and
    return (f_kin, complexity, res).

    f_kin: averaged kinetic-energy fraction near late times
    complexity: plasmoid complexity metric from A_z midplane at final time
    res: full simulation result dict (for plotting/debug)
    """
    log_eta, log_nu = theta
    eta = jnp.exp(log_eta)
    nu = jnp.exp(log_nu)

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

    # Average over late-time window (last 30% of the simulation)
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


def objective(theta: jnp.ndarray, cfg: EnergyPlasmoidConfig) -> jnp.ndarray:
    """
    Objective functional for energy/plasmoid design:

        score(theta) = alpha * f_kin + beta * complexity
        J(theta) = - score(theta)

    We *minimize* J(theta) using gradient descent.

    NOTE: Must be JAX-AD-safe:
      - no float(...) on tracers
      - use jax.debug.print for logging.
    """
    f_kin, complexity, _ = _simulate_energy_plasmoid(theta, cfg)

    score = cfg.alpha * f_kin + cfg.beta * complexity
    J = -score

    # AD-safe debug printing
    eta = jnp.exp(theta[0])
    nu = jnp.exp(theta[1])
    jax.debug.print(
        "[OBJ] eta={eta:.4e}, nu={nu:.4e}, f_kin={f_kin:.4f}, "
        "complexity={comp:.4e}, score={score:.4e}, J={J:.4e}",
        eta=eta,
        nu=nu,
        f_kin=f_kin,
        comp=complexity,
        score=score,
        J=J,
    )

    return J


# --------------------------- Optimization driver ----------------------------#

def run_optimization(cfg: EnergyPlasmoidConfig):
    """
    Gradient-descent optimization of (log_eta, log_nu).

    Returns:
      history: dict with arrays of eta, nu, f_kin, complexity, score, J
      res_init: simulation at initial (eta0,nu0)
      res_opt:  simulation at optimized (eta,nu)
    """
    theta0 = jnp.array([jnp.log(cfg.eta0), jnp.log(cfg.nu0)])

    value_and_grad = jax.value_and_grad(objective)

    history = {
        "eta": [],
        "nu": [],
        "f_kin": [],
        "complexity": [],
        "score": [],
        "J": [],
    }

    print("\n[INIT] Evaluating objective at initial (eta0, nu0)...")
    f_kin0, comp0, res_init = _simulate_energy_plasmoid(theta0, cfg)
    score0 = cfg.alpha * f_kin0 + cfg.beta * comp0
    J0 = -score0

    eta0_val = float(cfg.eta0)
    nu0_val = float(cfg.nu0)
    print(
        f"[INIT] eta0={eta0_val:.4e}, nu0={nu0_val:.4e}, "
        f"f_kin0={float(f_kin0):.4f}, "
        f"complexity0={float(comp0):.4e}, "
        f"score0={float(score0):.4e}, J0={float(J0):.4e}"
    )

    history["eta"].append(eta0_val)
    history["nu"].append(nu0_val)
    history["f_kin"].append(float(f_kin0))
    history["complexity"].append(float(comp0))
    history["score"].append(float(score0))
    history["J"].append(float(J0))

    theta = theta0
    res_opt = res_init

    print("\n[OPT] Starting gradient descent on (log_eta, log_nu)...")
    for k in range(cfg.n_opt_steps):
        J_val, grad_theta = value_and_grad(theta, cfg)
        g_eta, g_nu = grad_theta

        # Update
        theta = theta - jnp.array([cfg.lr_log_eta * g_eta,
                                   cfg.lr_log_nu * g_nu])

        # Diagnostics at new theta
        f_kin_k, comp_k, res_k = _simulate_energy_plasmoid(theta, cfg)
        score_k = cfg.alpha * f_kin_k + cfg.beta * comp_k

        eta_k = float(jnp.exp(theta[0]))
        nu_k = float(jnp.exp(theta[1]))

        history["eta"].append(eta_k)
        history["nu"].append(nu_k)
        history["f_kin"].append(float(f_kin_k))
        history["complexity"].append(float(comp_k))
        history["score"].append(float(score_k))
        history["J"].append(float(J_val))

        print(
            f"[OPT step {k:02d}] eta={eta_k:.4e}, nu={nu_k:.4e}, "
            f"f_kin={float(f_kin_k):.4f}, "
            f"complexity={float(comp_k):.4e}, "
            f"score={float(score_k):.4e}, J={float(J_val):.4e}, "
            f"|grad_eta|={float(jnp.abs(g_eta)):.3e}, "
            f"|grad_nu|={float(jnp.abs(g_nu)):.3e}"
        )

        res_opt = res_k

    return history, res_init, res_opt


# --------------------------- Plotting utilities -----------------------------#

def plot_optimization_history(history: Dict[str, List[float]], cfg: EnergyPlasmoidConfig):
    iters = np.arange(len(history["eta"]))

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)

    axes[0, 0].plot(iters, history["eta"], marker="o")
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].set_ylabel(r"$\eta$")
    axes[0, 0].set_title("Resistivity")

    axes[0, 1].plot(iters, history["nu"], marker="o")
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel(r"$\nu$")
    axes[0, 1].set_title("Viscosity")

    axes[0, 2].plot(iters, history["f_kin"], marker="o")
    axes[0, 2].set_xlabel("iteration")
    axes[0, 2].set_ylabel(r"$f_{\mathrm{kin}}$")
    axes[0, 2].set_title("Kinetic energy fraction")

    axes[1, 0].plot(iters, history["complexity"], marker="o")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel(r"$C_{\mathrm{plasmoid}}$")
    axes[1, 0].set_title("Plasmoid complexity")

    axes[1, 1].plot(iters, history["score"], marker="o")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].set_ylabel("score")
    axes[1, 1].set_title("Objective score")

    axes[1, 2].semilogy(iters, history["J"], marker="o")
    axes[1, 2].set_xlabel("iteration")
    axes[1, 2].set_ylabel(r"$J$")
    axes[1, 2].set_title("Loss")

    fig.suptitle("Energy/Plasmoid optimization history", fontsize=14)
    fig.savefig("energy_plasmoid_optimization_history.png", dpi=300)
    print("[PLOT] Saved energy_plasmoid_optimization_history.png")


def plot_energy_comparison(res_init: Dict[str, Any], res_opt: Dict[str, Any]):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for res, lab, color in [
        (res_init, "initial", "C0"),
        (res_opt, "optimized", "C3"),
    ]:
        ts = np.array(res["ts"])
        E_kin = np.array(res["E_kin"])
        E_mag = np.array(res["E_mag"])

        axes[0].plot(ts, E_kin, label=f"{lab}", color=color)
        axes[1].plot(ts, E_mag, label=f"{lab}", color=color)

    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$E_{\mathrm{kin}}$")
    axes[0].set_title("Kinetic energy")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$E_{\mathrm{mag}}$")
    axes[1].set_title("Magnetic energy")
    axes[1].legend(fontsize=8)

    fig.suptitle("Energy evolution: initial vs optimized", fontsize=14)
    fig.savefig("energy_plasmoid_energy_comparison.png", dpi=300)
    print("[PLOT] Saved energy_plasmoid_energy_comparison.png")


def plot_Az_midplane_comparison(res_init: Dict[str, Any], res_opt: Dict[str, Any]):
    """
    Compare midplane A_z profile at final time for initial vs optimized runs.
    """
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    Az_init = np.array(res_init["Az_final_mid"])
    Az_opt  = np.array(res_opt["Az_final_mid"])

    s = np.arange(Az_init.shape[0])

    ax.plot(s, Az_init, label="initial", alpha=0.8)
    ax.plot(s, Az_opt, label="optimized", alpha=0.8)
    ax.set_xlabel("midplane grid index")
    ax.set_ylabel(r"$A_z(x=x_\mathrm{sheet}, y)$")
    ax.set_title(r"Midplane $A_z$ at final time")
    ax.legend(fontsize=8)

    fig.savefig("energy_plasmoid_Az_midplane_comparison.png", dpi=300)
    print("[PLOT] Saved energy_plasmoid_Az_midplane_comparison.png")


# ----------------------------------- main -----------------------------------#

def main():
    cfg = EnergyPlasmoidConfig()

    print("========================================================")
    print(" Energy/Plasmoid optimization (differentiable MHD)")
    print("========================================================")
    print(cfg)

    history, res_init, res_opt = run_optimization(cfg)

    print("\n[POST] Making plots...")
    plot_optimization_history(history, cfg)
    plot_energy_comparison(res_init, res_opt)
    plot_Az_midplane_comparison(res_init, res_opt)

    print("\n[DONE] Energy/Plasmoid optimization finished.")


if __name__ == "__main__":
    main()
