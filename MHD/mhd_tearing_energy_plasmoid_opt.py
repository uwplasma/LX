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
    J(theta)     = - score(theta)          (we minimize J)

where
  - f_kin is the averaged kinetic-energy fraction near saturation,
  - C_plasmoid is the mean-squared curvature of A_z on the midplane at
    final time (computed by plasmoid_complexity_metric).

This script:
  1) Runs the MHD solver for each (eta, nu) via
       _run_tearing_simulation_and_diagnostics.
  2) Uses the JAX-based plasmoid_complexity_metric and energy traces to
     build the objective.
  3) Performs gradient descent on log(eta), log(nu).
  4) Produces publication-ready plots:
       - optimization history,
       - (eta,nu) "phase diagram" colored by complexity,
       - energy evolution (initial vs optimized, with late-time window),
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

# --- modest styling to make plots “paper-like” ---
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 11,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
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

    # Late-time averaging window (fraction of total time)
    tail_frac_start: float = 0.7   # average over [tail_frac_start * t1, t1]

    # Objective weights (alpha for kinetic fraction, beta for complexity)
    alpha: float = 1.0
    beta: float = 0.5

    # Optimization hyperparameters
    n_opt_steps: int = 20
    lr_log_eta: float = 0.5
    lr_log_nu: float = 0.5

    # Initial guesses
    eta0: float = 1e-3
    nu0: float = 1e-3

    # Plasmoid-like regime is usually better in the force-free equilibrium
    equilibrium_mode: str = "forcefree"
    
# ---- helper
    
def _prepare_npz_payload(res: Dict[str, Any],
                         extra_meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Convert the result dict from _run_tearing_simulation_and_diagnostics
    into something np.savez can digest (NumPy arrays + scalars).

    Any extra_meta entries are added on top.
    """
    payload: Dict[str, Any] = {}
    for key, val in res.items():
        if isinstance(val, (int, float, np.number, str)):
            payload[key] = val
            continue
        try:
            payload[key] = np.asarray(val)
        except Exception:
            pass

    if extra_meta is not None:
        payload.update(extra_meta)
    return payload

# --------------------------- Objective functional ---------------------------#

def _simulate_energy_plasmoid(theta: jnp.ndarray, cfg: EnergyPlasmoidConfig):
    """
    Run the tearing simulation for given theta=[log_eta, log_nu] and
    return (f_kin, complexity, t_tail_start, t_tail_end, res).

    f_kin:       averaged kinetic-energy fraction near late times
    complexity:  plasmoid complexity metric from A_z midplane at final time
    t_tail_*:    time window used for f_kin
    res:         full simulation result dict (for plotting/debug)
    """
    log_eta, log_nu = theta
    eta = jnp.exp(log_eta)
    nu = jnp.exp(log_nu)

    # Run the MHD simulation
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

    # --- Late-time window diagnostics ---
    T = ts.shape[0]          # integer (Python int when traced)
    i0 = int(cfg.tail_frac_start * (T - 1))
    i0 = max(0, min(i0, T - 1))
    t_tail_start = ts[i0]
    t_tail_end = ts[-1]

    E_kin_tail = E_kin[i0:]
    E_mag_tail = E_mag[i0:]

    E_kin_mean = jnp.mean(E_kin_tail)
    E_mag_mean = jnp.mean(E_mag_tail)
    E_tot_mean = E_kin_mean + E_mag_mean + 1e-30
    f_kin = E_kin_mean / E_tot_mean

    complexity = plasmoid_complexity_metric(Az_final_mid)

    # Attach a few diagnostics back into res for plotting
    res = dict(res)
    res["f_kin"] = f_kin
    res["complexity"] = complexity
    res["t_tail_start"] = t_tail_start
    res["t_tail_end"] = t_tail_end
    res["E_kin_tail_mean"] = E_kin_mean
    res["E_mag_tail_mean"] = E_mag_mean

    return f_kin, complexity, t_tail_start, t_tail_end, res


def objective(theta: jnp.ndarray, cfg: EnergyPlasmoidConfig) -> jnp.ndarray:
    """
    Objective functional for energy/plasmoid design:

        score(theta) = alpha * f_kin + beta * complexity
        J(theta)     = - score(theta)

    We *minimize* J(theta) using gradient descent.

    NOTE: Must be JAX-AD-safe:
      - no float(...) on tracers
      - use jax.debug.print for logging.
    """
    f_kin, complexity, t_tail_start, t_tail_end, _ = _simulate_energy_plasmoid(
        theta, cfg
    )

    score = cfg.alpha * f_kin + cfg.beta * complexity
    J = -score

    # AD-safe debug printing
    eta = jnp.exp(theta[0])
    nu = jnp.exp(theta[1])
    jax.debug.print(
        "[OBJ] eta={eta:.4e}, nu={nu:.4e}, "
        "f_kin={f_kin:.4f}, C_plas={comp:.4e}, "
        "score={score:.4e}, J={J:.4e}, "
        "t_tail=[{t0:.2f},{t1:.2f}]",
        eta=eta,
        nu=nu,
        f_kin=f_kin,
        comp=complexity,
        score=score,
        J=J,
        t0=t_tail_start,
        t1=t_tail_end,
    )

    return J


# --------------------------- Optimization driver ----------------------------#

def run_optimization(cfg: EnergyPlasmoidConfig):
    """
    Gradient-descent optimization of (log_eta, log_nu).

    Returns:
      history: dict with arrays of eta, nu, f_kin, complexity, score, J, etc.
      res_init: simulation at initial (eta0,nu0)
      res_opt:  simulation at optimized (eta,nu)
    """
    theta0 = jnp.array([jnp.log(cfg.eta0), jnp.log(cfg.nu0)])

    value_and_grad = jax.value_and_grad(objective)

    history: Dict[str, List[float]] = {
        "eta": [],
        "nu": [],
        "f_kin": [],
        "complexity": [],
        "score": [],
        "J": [],
        "t_tail_start": [],
        "t_tail_end": [],
        "grad_eta": [],
        "grad_nu": [],
    }

    # --------------------- Initial evaluation ---------------------#
    print("\n[INIT] Evaluating objective at initial (eta0, nu0)...")
    f_kin0, comp0, t_tail_start0, t_tail_end0, res_init = _simulate_energy_plasmoid(
        theta0, cfg
    )
    score0 = cfg.alpha * f_kin0 + cfg.beta * comp0
    J0 = -score0

    eta0_val = float(cfg.eta0)
    nu0_val = float(cfg.nu0)
    print(
        "[INIT] eta0={eta:.4e}, nu0={nu:.4e}, "
        "f_kin0={fk:.4f}, C_plas0={cp:.4e}, "
        "score0={sc:.4e}, J0={J:.4e}, "
        "t_tail=[{t0:.2f},{t1:.2f}]".format(
            eta=eta0_val,
            nu=nu0_val,
            fk=float(f_kin0),
            cp=float(comp0),
            sc=float(score0),
            J=float(J0),
            t0=float(t_tail_start0),
            t1=float(t_tail_end0),
        )
    )

    history["eta"].append(eta0_val)
    history["nu"].append(nu0_val)
    history["f_kin"].append(float(f_kin0))
    history["complexity"].append(float(comp0))
    history["score"].append(float(score0))
    history["J"].append(float(J0))
    history["t_tail_start"].append(float(t_tail_start0))
    history["t_tail_end"].append(float(t_tail_end0))
    history["grad_eta"].append(np.nan)
    history["grad_nu"].append(np.nan)

    theta = theta0
    res_opt = res_init

    # --------------------- Optimization loop ----------------------#
    print("\n[OPT] Starting gradient descent on (log_eta, log_nu)...")
    for k in range(cfg.n_opt_steps):
        J_val, grad_theta = value_and_grad(theta, cfg)
        g_eta, g_nu = grad_theta

        # Update in log-space
        theta = theta - jnp.array([cfg.lr_log_eta * g_eta,
                                   cfg.lr_log_nu * g_nu])

        # Diagnostics at new theta
        f_kin_k, comp_k, t_tail_start_k, t_tail_end_k, res_k = (
            _simulate_energy_plasmoid(theta, cfg)
        )
        score_k = cfg.alpha * f_kin_k + cfg.beta * comp_k

        eta_k = float(jnp.exp(theta[0]))
        nu_k = float(jnp.exp(theta[1]))
        grad_eta_k = float(jnp.abs(g_eta))
        grad_nu_k = float(jnp.abs(g_nu))

        history["eta"].append(eta_k)
        history["nu"].append(nu_k)
        history["f_kin"].append(float(f_kin_k))
        history["complexity"].append(float(comp_k))
        history["score"].append(float(score_k))
        history["J"].append(float(J_val))
        history["t_tail_start"].append(float(t_tail_start_k))
        history["t_tail_end"].append(float(t_tail_end_k))
        history["grad_eta"].append(grad_eta_k)
        history["grad_nu"].append(grad_nu_k)

        print(
            "[OPT step {k:02d}] eta={eta:.4e}, nu={nu:.4e}, "
            "f_kin={fk:.4f}, C_plas={cp:.4e}, "
            "score={sc:.4e}, J={J:.4e}, "
            "|grad_eta|={ge:.3e}, |grad_nu|={gn:.3e}, "
            "t_tail=[{t0:.2f},{t1:.2f}]".format(
                k=k,
                eta=eta_k,
                nu=nu_k,
                fk=float(f_kin_k),
                cp=float(comp_k),
                sc=float(score_k),
                J=float(J_val),
                ge=grad_eta_k,
                gn=grad_nu_k,
                t0=float(t_tail_start_k),
                t1=float(t_tail_end_k),
            )
        )

        res_opt = res_k

    return history, res_init, res_opt


# --------------------------- Plotting utilities -----------------------------#

def plot_optimization_history(history: Dict[str, List[float]],
                              cfg: EnergyPlasmoidConfig):
    iters = np.arange(len(history["eta"]))

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)

    axes[0, 0].plot(iters, history["eta"], marker="o")
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].set_ylabel(r"$\eta$")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Resistivity")

    axes[0, 1].plot(iters, history["nu"], marker="o")
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel(r"$\nu$")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Viscosity")

    axes[0, 2].plot(iters, history["f_kin"], marker="o")
    axes[0, 2].set_xlabel("iteration")
    axes[0, 2].set_ylabel(r"$f_{\mathrm{kin}}$")
    axes[0, 2].set_title("Kinetic-energy fraction")

    axes[1, 0].plot(iters, history["complexity"], marker="o")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel(r"$C_{\mathrm{plasmoid}}$")
    axes[1, 0].set_title("Plasmoid complexity")

    axes[1, 1].plot(iters, history["score"], marker="o")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].set_ylabel("score")
    axes[1, 1].set_title("Objective score (to maximize)")

    # J is negative (J = -score); plot |J| on log scale
    J_abs = np.abs(np.asarray(history["J"]))
    J_abs[J_abs == 0.0] = np.nan
    axes[1, 2].semilogy(iters, J_abs, marker="o")
    axes[1, 2].set_xlabel("iteration")
    axes[1, 2].set_ylabel(r"$|J|$")
    axes[1, 2].set_title(r"Loss $J=-\mathrm{score}$")

    fig.suptitle("Energy/Plasmoid optimization history", fontsize=14)
    fig.savefig("energy_plasmoid_optimization_history.png", dpi=300)
    print("[PLOT] Saved energy_plasmoid_optimization_history.png")


def plot_eta_nu_phase(history: Dict[str, List[float]],
                      cfg: EnergyPlasmoidConfig):
    """Scatter of (eta, nu) colored by complexity."""
    eta = np.asarray(history["eta"])
    nu = np.asarray(history["nu"])
    comp = np.asarray(history["complexity"])
    iters = np.arange(len(eta))

    fig, ax = plt.subplots(figsize=(5.0, 4.5), constrained_layout=True)

    sc = ax.scatter(eta, nu, c=comp, cmap="viridis", s=50, edgecolor="k")
    for i, (x, y) in enumerate(zip(eta, nu)):
        ax.text(x, y, str(i), fontsize=7, ha="center", va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\nu$")
    ax.set_title(r"Parameter path in $(\eta,\nu)$ space")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$C_{\mathrm{plasmoid}}$")

    fig.savefig("energy_plasmoid_eta_nu_phase.png", dpi=300)
    print("[PLOT] Saved energy_plasmoid_eta_nu_phase.png")


def plot_energy_comparison(res_init: Dict[str, Any],
                           res_opt: Dict[str, Any],
                           cfg: EnergyPlasmoidConfig):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for res, lab, color in [
        (res_init, "initial", "C0"),
        (res_opt, "optimized", "C3"),
    ]:
        ts = np.array(res["ts"])
        E_kin = np.array(res["E_kin"])
        E_mag = np.array(res["E_mag"])

        t_tail_start = float(res.get("t_tail_start", cfg.tail_frac_start * cfg.t1))
        t_tail_end = float(res.get("t_tail_end", cfg.t1))

        axes[0].plot(ts, E_kin, label=f"{lab}", color=color)
        axes[0].axvspan(t_tail_start, t_tail_end,
                        color=color, alpha=0.08, lw=0)

        axes[1].plot(ts, E_mag, label=f"{lab}", color=color)
        axes[1].axvspan(t_tail_start, t_tail_end,
                        color=color, alpha=0.08, lw=0)

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


def plot_Az_midplane_comparison(res_init: Dict[str, Any],
                                res_opt: Dict[str, Any]):
    """
    Compare midplane A_z profile at final time for initial vs optimized runs.
    """
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    Az_init = np.array(res_init["Az_final_mid"])
    Az_opt = np.array(res_opt["Az_final_mid"])

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
    plot_eta_nu_phase(history, cfg)
    plot_energy_comparison(res_init, res_opt, cfg)
    plot_Az_midplane_comparison(res_init, res_opt)
    
    # Save initial and optimized solutions as .npz for postprocessing
    print("\n[SAVE] Writing initial and optimized solutions for postprocessing...")

    # Common stem: plasmoid optimization, equilibrium mode, fixed a
    stem_base = f"mhd_tearing_solution_plasmoid_{cfg.equilibrium_mode}_a{cfg.a:.3f}"

    payload_init = _prepare_npz_payload(
        res_init,
        extra_meta={
            "opt_script": "mhd_tearing_energy_plasmoid_opt",
            "opt_kind": "plasmoid_init",
            "alpha": cfg.alpha,
            "beta": cfg.beta,
        },
    )
    fname_init = stem_base + "_init.npz"
    np.savez(fname_init, **payload_init)
    print(f"[SAVE] Initial solution saved to {fname_init}")

    payload_opt = _prepare_npz_payload(
        res_opt,
        extra_meta={
            "opt_script": "mhd_tearing_energy_plasmoid_opt",
            "opt_kind": "plasmoid_opt",
            "alpha": cfg.alpha,
            "beta": cfg.beta,
        },
    )
    fname_opt = stem_base + "_opt.npz"
    np.savez(fname_opt, **payload_opt)
    print(f"[SAVE] Optimized solution saved to {fname_opt}")


    print("\n[DONE] Energy/Plasmoid optimization finished.")


if __name__ == "__main__":
    main()
