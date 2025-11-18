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
  3) Performs gradient descent on log(eta), log(nu), tracking the
     *best-so-far* parameters.
  4) Produces publication-ready plots:
       - optimization history,
       - (eta,nu) "phase diagram" colored by complexity,
       - energy evolution (initial vs optimized, with late-time window),
       - midplane A_z profile (initial vs optimized),
       - final-time B_x(x,y) and A_z(x,y) fields (initial vs optimized).

Usage:
  python mhd_tearing_energy_plasmoid_opt.py

Edit EnergyPlasmoidConfig below for resolution, weights, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mhd_tearing_solve import (
    _run_tearing_simulation_and_diagnostics,
    plasmoid_complexity_metric,
    compute_k_arrays_np,
    compute_Az_hat_np,
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
    # Learning rates in *log*-space; a bit conservative for stability
    lr_log_eta: float = 0.3
    lr_log_nu: float = 0.3

    # Initial guesses
    eta0: float = 1e-3
    nu0: float = 1e-3

    # Plasmoid-like regime is usually better in the force-free equilibrium
    equilibrium_mode: str = "forcefree"


# ---- helper for saving to NPZ ------------------------------------------------

def _prepare_npz_payload(res: Dict[str, Any],
                         extra_meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Convert the result dict from _run_tearing_simulation_and_diagnostics
    into something np.savez can digest (NumPy arrays + scalars).

    Any extra_meta entries are added on top.
    """
    payload: Dict[str, Any] = {}
    for key, val in res.items():
        # Simple scalars / strings
        if isinstance(val, (int, float, np.number, str)):
            payload[key] = val
            continue
        # JAX / NumPy arrays, lists, etc.
        try:
            payload[key] = np.asarray(val)
        except Exception:
            # If it really cannot be converted, we just skip it.
            pass

    if extra_meta is not None:
        payload.update(extra_meta)
    return payload


# --------------------------- Objective functional ---------------------------#

def _simulate_energy_plasmoid(theta: jnp.ndarray,
                              cfg: EnergyPlasmoidConfig):
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

    # Run the MHD simulation (JAX-based, but called in Python space)
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

    # --- Late-time window diagnostics (in index space) ---
    T = ts.shape[0]          # number of frames
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

    # Attach a few diagnostics back into res for plotting / saving
    res = dict(res)
    res["f_kin"] = f_kin
    res["complexity"] = complexity
    res["t_tail_start"] = t_tail_start
    res["t_tail_end"] = t_tail_end
    res["E_kin_tail_mean"] = E_kin_mean
    res["E_mag_tail_mean"] = E_mag_mean

    return f_kin, complexity, t_tail_start, t_tail_end, res


def objective(theta: jnp.ndarray,
              cfg: EnergyPlasmoidConfig) -> jnp.ndarray:
    """
    Objective functional for energy/plasmoid design:

        score(theta) = alpha * f_kin + beta * complexity
        J(theta)     = - score(theta)

    We *minimize* J(theta) using gradient descent.

    NOTE: Must be JAX-AD-safe:
      - no float(...) on tracers
      - use jax.debug.print for logging.
    """
    f_kin, complexity, t_tail_start, t_tail_end, res = _simulate_energy_plasmoid(
        theta, cfg
    )

    score = cfg.alpha * f_kin + cfg.beta * complexity
    J = -score

    # AD-safe debug printing
    eta = jnp.exp(theta[0])
    nu = jnp.exp(theta[1])

    # Extra debug: growth rate comparison if available
    gamma_FKR = res.get("gamma_FKR", jnp.nan)
    gamma_fit = res.get("gamma_fit", jnp.nan)

    jax.debug.print(
        "[OBJ] eta={eta:.4e}, nu={nu:.4e}, "
        "f_kin={f_kin:.4f}, C_plas={comp:.4e}, "
        "score={score:.4e}, J={J:.4e}, "
        "gamma_FKR={gF:.3e}, gamma_fit={gf:.3e}, "
        "t_tail=[{t0:.2f},{t1:.2f}]",
        eta=eta,
        nu=nu,
        f_kin=f_kin,
        comp=complexity,
        score=score,
        J=J,
        gF=gamma_FKR,
        gf=gamma_fit,
        t0=t_tail_start,
        t1=t_tail_end,
    )

    return J


# --------------------------- Optimization driver ----------------------------#

def run_optimization(cfg: EnergyPlasmoidConfig):
    """
    Gradient-descent optimization of (log_eta, log_nu).

    Returns:
      history:   dict with arrays of eta, nu, f_kin, complexity, score, J, etc.
      res_init:  simulation at initial (eta0,nu0)
      res_best:  simulation at *best-scoring* (eta,nu)
      best_info: dict with best theta, iteration, score, etc.
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

    gamma_FKR0 = float(res_init.get("gamma_FKR", np.nan))
    gamma_fit0 = float(res_init.get("gamma_fit", np.nan))

    print(
        "[INIT] eta0={eta:.4e}, nu0={nu:.4e}, "
        "f_kin0={fk:.4f}, C_plas0={cp:.4e}, "
        "score0={sc:.4e}, J0={J:.4e}, "
        "gamma_FKR0={gF:.3e}, gamma_fit0={gfi:.3e}, "
        "t_tail=[{t0:.2f},{t1:.2f}]".format(
            eta=eta0_val,
            nu=nu0_val,
            fk=float(f_kin0),
            cp=float(comp0),
            sc=float(score0),
            J=float(J0),
            gF=gamma_FKR0,
            gfi=gamma_fit0,
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

    # Best-so-far bookkeeping
    best_score = float(score0)
    best_theta = np.array(theta0)
    best_iter = 0
    res_best = res_init
    best_f_kin = float(f_kin0)
    best_complexity = float(comp0)

    # --------------------- Optimization loop ----------------------#
    print("\n[OPT] Starting gradient descent on (log_eta, log_nu)...")
    for k in range(cfg.n_opt_steps):
        # Evaluate objective & gradient at current theta
        J_val, grad_theta = value_and_grad(theta, cfg)
        g_eta, g_nu = grad_theta

        # Simple gradient descent in log-space
        theta = theta - jnp.array(
            [cfg.lr_log_eta * g_eta, cfg.lr_log_nu * g_nu]
        )

        # Diagnostics at updated theta
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
        history["J"].append(float(J_val))  # J evaluated at pre-update theta
        history["t_tail_start"].append(float(t_tail_start_k))
        history["t_tail_end"].append(float(t_tail_end_k))
        history["grad_eta"].append(grad_eta_k)
        history["grad_nu"].append(grad_nu_k)

        print(
            "[OPT step {k:02d}] eta={eta:.4e}, nu={nu:.4e}, "
            "f_kin={fk:.4f}, C_plas={cp:.4e}, "
            "score={sc:.4e}, J(prev)={J:.4e}, "
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

        # Update best-so-far
        if float(score_k) > best_score:
            best_score = float(score_k)
            best_theta = np.array(theta)
            best_iter = k + 1  # because k=0 corresponds to 1st update
            res_best = res_k
            best_f_kin = float(f_kin_k)
            best_complexity = float(comp_k)
            print(
                "      [BEST] updated at iter {it:02d}: "
                "score={sc:.4e}, f_kin={fk:.4f}, C_plas={cp:.4e}, "
                "eta={eta:.4e}, nu={nu:.4e}".format(
                    it=best_iter,
                    sc=best_score,
                    fk=best_f_kin,
                    cp=best_complexity,
                    eta=float(jnp.exp(best_theta[0])),
                    nu=float(jnp.exp(best_theta[1])),
                )
            )

    best_info = dict(
        best_theta=best_theta,
        best_iter=best_iter,
        best_score=best_score,
        best_f_kin=best_f_kin,
        best_complexity=best_complexity,
    )

    return history, res_init, res_best, best_info


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
        E_mag = np.array(res["E_kin"]*0 + res["E_mag"])  # ensure copy

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
    Compare midplane A_z profile at final time for initial vs optimized runs
    (1D cut).
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


def _extract_final_Bx_Az_xy(res: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    From a simulation result dict, reconstruct final-time B_x(x,y) and
    A_z(x,y) on the z=0 midplane for visualization.
    """
    B_hat_T = np.asarray(res["B_hat"][-1])  # (3, Nx, Ny, Nz)
    Nx, Ny, Nz = B_hat_T.shape[1], B_hat_T.shape[2], B_hat_T.shape[3]
    Lx, Ly, Lz = float(res["Lx"]), float(res["Ly"]), float(res["Lz"])

    # Real-space B
    B_real = np.fft.ifftn(B_hat_T, axes=(1, 2, 3)).real
    Bx_xy = B_real[0, :, :, 0]  # z=0 plane

    # Flux function A_z from B_hat
    kx, ky, kz, NX, NY, NZ = compute_k_arrays_np(Nx, Ny, Nz, Lx, Ly, Lz)
    Az_hat = compute_Az_hat_np(B_hat_T, kx, ky)
    Az_real = np.fft.ifftn(Az_hat, axes=(0, 1, 2)).real
    Az_xy = Az_real[:, :, 0]  # z=0 plane

    return Bx_xy, Az_xy


def plot_final_field_comparison(res_init: Dict[str, Any],
                                res_opt: Dict[str, Any],
                                cfg: EnergyPlasmoidConfig):
    """
    2D fields at final time for initial vs optimized runs:

      - B_x(x,y,t_final) on z=0 plane
      - A_z(x,y,t_final) on z=0 plane
    """
    Bx_init, Az_init = _extract_final_Bx_Az_xy(res_init)
    Bx_opt, Az_opt = _extract_final_Bx_Az_xy(res_opt)

    x = np.linspace(0.0, cfg.Lx, cfg.Nx, endpoint=False)
    y = np.linspace(0.0, cfg.Ly, cfg.Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    im0 = axes[0, 0].pcolormesh(X, Y, Bx_init, shading="auto")
    axes[0, 0].set_title(r"$B_x(x,y,t_{\mathrm{final}})$ (initial)")
    axes[0, 0].set_xlabel(r"$x$")
    axes[0, 0].set_ylabel(r"$y$")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].pcolormesh(X, Y, Bx_opt, shading="auto")
    axes[0, 1].set_title(r"$B_x(x,y,t_{\mathrm{final}})$ (optimized)")
    axes[0, 1].set_xlabel(r"$x$")
    axes[0, 1].set_ylabel(r"$y$")
    fig.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].pcolormesh(X, Y, Az_init, shading="auto")
    axes[1, 0].set_title(r"$A_z(x,y,t_{\mathrm{final}})$ (initial)")
    axes[1, 0].set_xlabel(r"$x$")
    axes[1, 0].set_ylabel(r"$y$")
    fig.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].pcolormesh(X, Y, Az_opt, shading="auto")
    axes[1, 1].set_title(r"$A_z(x,y,t_{\mathrm{final}})$ (optimized)")
    axes[1, 1].set_xlabel(r"$x$")
    axes[1, 1].set_ylabel(r"$y$")
    fig.colorbar(im3, ax=axes[1, 1])

    fig.suptitle("Final-time fields: initial vs optimized", fontsize=14)
    fig.savefig("energy_plasmoid_final_fields.png", dpi=300)
    print("[PLOT] Saved energy_plasmoid_final_fields.png")


# ----------------------------------- main -----------------------------------#

def main():
    cfg = EnergyPlasmoidConfig()

    print("========================================================")
    print(" Energy/Plasmoid optimization (differentiable MHD)")
    print("========================================================")
    print(cfg)

    history, res_init, res_opt, best_info = run_optimization(cfg)

    print(
        "\n[SUMMARY] Best score={sc:.4e} at iter={it:02d} with "
        "f_kin={fk:.4f}, C_plas={cp:.4e}, "
        "eta={eta:.4e}, nu={nu:.4e}".format(
            sc=best_info["best_score"],
            it=best_info["best_iter"],
            fk=best_info["best_f_kin"],
            cp=best_info["best_complexity"],
            eta=float(np.exp(best_info["best_theta"][0])),
            nu=float(np.exp(best_info["best_theta"][1])),
        )
    )

    print("\n[POST] Making plots...")
    plot_optimization_history(history, cfg)
    plot_eta_nu_phase(history, cfg)
    plot_energy_comparison(res_init, res_opt, cfg)
    plot_Az_midplane_comparison(res_init, res_opt)
    plot_final_field_comparison(res_init, res_opt, cfg)

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
            "opt_kind": "plasmoid_opt_best",
            "alpha": cfg.alpha,
            "beta": cfg.beta,
        },
    )
    fname_opt = stem_base + "_opt_best.npz"
    np.savez(fname_opt, **payload_opt)
    print(f"[SAVE] Optimized (best) solution saved to {fname_opt}")

    print("\n[DONE] Energy/Plasmoid optimization finished.")


if __name__ == "__main__":
    main()
