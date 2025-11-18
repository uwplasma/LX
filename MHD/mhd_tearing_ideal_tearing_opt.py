#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_ideal_tearing_opt.py

Differentiable tearing benchmark:
---------------------------------
Use JAX autodiff + the incompressible pseudo-spectral MHD solver
(mhd_tearing_solve.py) to *optimize* the current-sheet half-width `a` at
fixed Lundquist number S_target so that the normalized tearing growth rate

    gamma_hat = gamma * a / vA

is as close as possible to order unity (ideal tearing regime).

We use:
    S = a B0 / eta  (with B0=1, rho=1)
and enforce the constraint by setting eta(a) = a B0 / S_target.

The objective functional is
    J(a) = (gamma_hat(a) - gamma_star)^2

with gamma_star ≈ 1.

This script:
  1) Defines an objective(log_a) that:
       - runs the MHD tearing simulation,
       - extracts gamma_fit from mode_amp_series,
       - builds gamma_hat(a),
       - returns J(a) with detailed debug information.
  2) Uses gradient descent on log(a) to minimize J.
  3) Produces publication-ready plots:
       - optimization history (gamma_hat, a, J vs iteration),
       - auxiliary diagnostics (eta/S and linear-fit window vs iteration),
       - gamma_hat vs a across iterations,
       - ln|B_x(kx=0,ky=1)| vs t for initial and optimal runs (with fits),
       - energy evolution for initial and optimal runs.

Usage:
  python mhd_tearing_ideal_tearing_opt.py

Adjust the CONFIG section at the bottom for numerical resolution, S_target,
number of optimization steps, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mhd_tearing_solve import (
    _run_tearing_simulation_and_diagnostics,
    estimate_growth_rate,
)


# Some modest styling for nicer, “paper-y” plots
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
class IdealTearingConfig:
    # Grid and box
    Nx: int = 64
    Ny: int = 64
    Nz: int = 1
    Lx: float = 2.0 * math.pi
    Ly: float = 2.0 * math.pi
    Lz: float = 2.0 * math.pi

    # Physical parameters
    B0: float = 1.0
    B_g: float = 0.2
    nu: float = 1e-3
    eps_B: float = 1e-3

    # Time integration
    t0: float = 0.0
    t1: float = 20.0
    n_frames: int = 80
    dt0: float = 5e-4

    # Ideal tearing target
    S_target: float = 1e4      # target Lundquist number based on sheet half-width a
    gamma_star: float = 1.0    # target normalized growth rate gamma_hat = gamma a / vA

    # Optimization hyperparameters
    n_opt_steps: int = 20
    lr_log_a: float = 0.5      # learning rate for log(a)

    # Initial guess for a
    a0: float = 0.25           # initial half-width (in units of Lx ~ 2π)

    equilibrium_mode: str = "original"


# --------------------------- Objective functional ---------------------------#

def _simulate_and_gamma_hat(log_a: jnp.ndarray, cfg: IdealTearingConfig):
    """
    Given log(a), run the tearing simulation and return:

      gamma_hat, gamma_fit, a, eta, S,
      t_lin_start, t_lin_end, n_lin, res

    where
      gamma_hat = gamma_fit * a / vA   (vA = B0)

    `res` is the full result dict from
    _run_tearing_simulation_and_diagnostics, with a few extra fields added.
    """
    a = jnp.exp(log_a)
    B0 = cfg.B0
    vA = B0

    # Enforce S = a B0 / eta = S_target  ->  eta = a B0 / S_target
    eta = a * B0 / cfg.S_target
    nu = cfg.nu

    # Run the MHD simulation with these parameters
    res = _run_tearing_simulation_and_diagnostics(
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        Nz=cfg.Nz,
        Lx=cfg.Lx,
        Ly=cfg.Ly,
        Lz=cfg.Lz,
        nu=nu,
        eta=eta,
        B0=B0,
        a=a,
        B_g=cfg.B_g,
        eps_B=cfg.eps_B,
        t0=cfg.t0,
        t1=cfg.t1,
        n_frames=cfg.n_frames,
        dt0=cfg.dt0,
        equilibrium_mode=cfg.equilibrium_mode,
    )

    ts = res["ts"]
    mode_amp = res["mode_amp_series"]

    gamma_fit, lnA_fit, mask_lin = estimate_growth_rate(ts, mode_amp, w0=mode_amp[0])
    gamma_hat = gamma_fit * a / vA
    S = a * B0 / eta  # should be ≈ S_target, but we recompute for debug

    # Linear-fit window diagnostics
    ts_lin = ts[mask_lin]
    # Guard against pathological cases with no linear points
    has_lin = jnp.any(mask_lin)
    t_lin_start = jnp.where(has_lin, ts_lin[0], jnp.asarray(cfg.t0))
    t_lin_end = jnp.where(has_lin, ts_lin[-1], jnp.asarray(cfg.t1))
    n_lin = jnp.sum(mask_lin.astype(jnp.int32))

    # Attach some of this back to the result for later plotting
    res = dict(res)
    res["gamma_fit"] = gamma_fit
    res["gamma_hat"] = gamma_hat
    res["mask_lin"] = mask_lin
    res["t_lin_start"] = t_lin_start
    res["t_lin_end"] = t_lin_end
    res["lnA_fit"] = lnA_fit

    return gamma_hat, gamma_fit, a, eta, S, t_lin_start, t_lin_end, n_lin, res


def objective(log_a: jnp.ndarray, cfg: IdealTearingConfig) -> jnp.ndarray:
    """
    Objective functional J(log_a) for ideal tearing:
        J = (gamma_hat(a) - gamma_star)^2
    where gamma_hat(a) = gamma_fit(a) * a / vA.
    """
    (
        gamma_hat,
        gamma_fit,
        a,
        eta,
        S,
        t_lin_start,
        t_lin_end,
        n_lin,
        _,
    ) = _simulate_and_gamma_hat(log_a, cfg)

    J = (gamma_hat - cfg.gamma_star) ** 2

    # AD-safe debug printing
    jax.debug.print(
        "[OBJ] a={a:.4e}, eta={eta:.4e}, S≈{S:.3e}, "
        "gamma={gamma:.4e}, gamma_hat={gh:.4e}, J={J:.4e}, "
        "t_lin=[{t0:.3f},{t1:.3f}], N_lin={n_lin}",
        a=a,
        eta=eta,
        S=S,
        gamma=gamma_fit,
        gh=gamma_hat,
        J=J,
        t0=t_lin_start,
        t1=t_lin_end,
        n_lin=n_lin,
    )

    return J

# --------- helper

def _prepare_npz_payload(res: Dict[str, Any],
                         extra_meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Convert the result dict from _run_tearing_simulation_and_diagnostics
    into something np.savez can digest (NumPy arrays + scalars).

    Any extra_meta entries are added on top.
    """
    payload: Dict[str, Any] = {}
    for key, val in res.items():
        # Keep simple scalars/strings as-is
        if isinstance(val, (int, float, np.number, str)):
            payload[key] = val
            continue
        # Try to turn everything else into a NumPy array
        try:
            payload[key] = np.asarray(val)
        except Exception:
            # If something really can't be converted (e.g. a callable), skip it
            pass

    if extra_meta is not None:
        payload.update(extra_meta)
    return payload


# --------------------------- Optimization driver ----------------------------#

def run_optimization(cfg: IdealTearingConfig):
    """
    Gradient-descent optimization of log(a) to reach ideal tearing.

    Returns:
      history: dict with arrays of log_a, a, gamma_hat, gamma_fit, J, etc.
      res_init: simulation result at initial a0
      res_opt:  simulation result at optimized a
    """
    log_a0 = jnp.log(cfg.a0)

    value_and_grad = jax.value_and_grad(objective)

    history: Dict[str, List[float]] = {
        "log_a": [],
        "a": [],
        "eta": [],
        "S": [],
        "gamma_hat": [],
        "gamma_fit": [],
        "J": [],
        "t_lin_start": [],
        "t_lin_end": [],
        "n_lin": [],
        "grad_log_a": [],
    }

    # ------------------------------------------------------------------#
    # Initial evaluation (for a nice baseline)
    # ------------------------------------------------------------------#
    print("\n[INIT] Evaluating objective at initial a0...")
    (
        gamma_hat0,
        gamma_fit0,
        a0_eff,
        eta0,
        S0,
        t_lin_start0,
        t_lin_end0,
        n_lin0,
        res_init,
    ) = _simulate_and_gamma_hat(log_a0, cfg)
    J0 = (gamma_hat0 - cfg.gamma_star) ** 2

    history["log_a"].append(float(log_a0))
    history["a"].append(float(a0_eff))
    history["eta"].append(float(eta0))
    history["S"].append(float(S0))
    history["gamma_hat"].append(float(gamma_hat0))
    history["gamma_fit"].append(float(gamma_fit0))
    history["J"].append(float(J0))
    history["t_lin_start"].append(float(t_lin_start0))
    history["t_lin_end"].append(float(t_lin_end0))
    history["n_lin"].append(float(n_lin0))
    history["grad_log_a"].append(np.nan)  # no gradient yet

    print(
        "[INIT] a0={a:.4e}, eta0={eta:.4e}, S0≈{S:.3e}, "
        "gamma0={g:.4e}, gamma_hat0={gh:.4e}, "
        "t_lin=[{t0:.3f},{t1:.3f}], N_lin={n_lin:d}, J0={J:.4e}".format(
            a=float(a0_eff),
            eta=float(eta0),
            S=float(S0),
            g=float(gamma_fit0),
            gh=float(gamma_hat0),
            t0=float(t_lin_start0),
            t1=float(t_lin_end0),
            n_lin=int(n_lin0),
            J=float(J0),
        )
    )

    # ------------------------------------------------------------------#
    # Optimization loop
    # ------------------------------------------------------------------#
    log_a = log_a0
    res_opt = res_init  # will be overwritten by the last evaluation

    print("\n[OPT] Starting gradient descent on log(a)...")
    for k in range(cfg.n_opt_steps):
        J_val, grad_log_a = value_and_grad(log_a, cfg)

        # Gradient step in log(a) (ensures a>0)
        log_a = log_a - cfg.lr_log_a * grad_log_a

        # For diagnostics, re-evaluate gamma_hat, etc. at the updated log_a
        (
            gamma_hat_k,
            gamma_fit_k,
            a_k,
            eta_k,
            S_k,
            t_lin_start_k,
            t_lin_end_k,
            n_lin_k,
            res_k,
        ) = _simulate_and_gamma_hat(log_a, cfg)

        history["log_a"].append(float(log_a))
        history["a"].append(float(a_k))
        history["eta"].append(float(eta_k))
        history["S"].append(float(S_k))
        history["gamma_hat"].append(float(gamma_hat_k))
        history["gamma_fit"].append(float(gamma_fit_k))
        history["J"].append(float(J_val))
        history["t_lin_start"].append(float(t_lin_start_k))
        history["t_lin_end"].append(float(t_lin_end_k))
        history["n_lin"].append(float(n_lin_k))
        history["grad_log_a"].append(float(jnp.abs(grad_log_a)))

        print(
            "[OPT step {k:02d}] log(a)={loga:+.4f}, a={a:.4e}, "
            "eta={eta:.4e}, S≈{S:.3e}, "
            "gamma={g:.4e}, gamma_hat={gh:.4e}, "
            "|grad_log_a|={grad:.3e}, "
            "t_lin=[{t0:.3f},{t1:.3f}], N_lin={n_lin:d}, J={J:.4e}".format(
                k=k,
                loga=float(log_a),
                a=float(a_k),
                eta=float(eta_k),
                S=float(S_k),
                g=float(gamma_fit_k),
                gh=float(gamma_hat_k),
                grad=float(jnp.abs(grad_log_a)),
                t0=float(t_lin_start_k),
                t1=float(t_lin_end_k),
                n_lin=int(n_lin_k),
                J=float(J_val),
            )
        )

        res_opt = res_k

    return history, res_init, res_opt


# --------------------------- Plotting utilities -----------------------------#

def plot_optimization_history(history: Dict[str, List[float]], cfg: IdealTearingConfig):
    iters = np.arange(len(history["a"]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

    # a vs iteration
    axes[0].plot(iters, history["a"], marker="o")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel(r"$a$")
    axes[0].set_title("Sheet half-width $a$")

    # gamma_hat vs iteration
    axes[1].plot(iters, history["gamma_hat"], marker="o")
    axes[1].axhline(cfg.gamma_star, color="k", linestyle="--", linewidth=1)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"$\hat{\gamma} = \gamma a / v_A$")
    axes[1].set_title("Normalized growth rate")

    # J vs iteration (log scale)
    J = np.abs(history["J"])
    axes[2].semilogy(iters, J, marker="o")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel(r"$J$")
    axes[2].set_title("Objective")

    fig.suptitle("Ideal tearing optimization history", fontsize=14)
    fig.savefig("ideal_tearing_optimization_history.png", dpi=300)
    print("[PLOT] Saved ideal_tearing_optimization_history.png")


def plot_aux_diagnostics(history: Dict[str, List[float]], cfg: IdealTearingConfig):
    """Extra plots to debug eta/S and the linear-fit window."""
    iters = np.arange(len(history["a"]))
    eta = np.array(history["eta"])
    S = np.array(history["S"])
    t0_lin = np.array(history["t_lin_start"])
    t1_lin = np.array(history["t_lin_end"])
    grad = np.array(history["grad_log_a"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

    # Left: eta (log) and S (linear) vs iteration
    ax0 = axes[0]
    ax0.semilogy(iters, eta, marker="o", label=r"$\eta$")
    ax0.set_xlabel("iteration")
    ax0.set_ylabel(r"$\eta$")
    ax0.set_title(r"$\eta$ and $S$ vs iteration")

    ax0b = ax0.twinx()
    ax0b.plot(iters, S, "k--", label=r"$S$")
    ax0b.axhline(cfg.S_target, color="gray", linestyle=":", linewidth=1)
    ax0b.set_ylabel(r"$S$")
    ax0b.tick_params(axis="y")

    # Right: linear fit window and grad vs iteration
    ax1 = axes[1]
    ax1.plot(iters, t0_lin, "o-", label=r"$t_{\mathrm{lin,start}}$")
    ax1.plot(iters, t1_lin, "o-", label=r"$t_{\mathrm{lin,end}}$")
    ax1.axhline(cfg.t1, color="k", linestyle=":", label=r"$t_1$")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel(r"$t$")
    ax1.set_title("Linear fit window")
    ax1.legend(fontsize=8, loc="upper left")

    ax1b = ax1.twinx()
    ax1b.semilogy(iters, np.where(np.isnan(grad), np.nan, grad), "C3--", label=r"$|\nabla_{\log a} J|$")
    ax1b.set_ylabel(r"$|\nabla_{\log a} J|$")
    ax1b.tick_params(axis="y", labelcolor="C3")

    fig.suptitle("Ideal tearing auxiliary diagnostics", fontsize=14)
    fig.savefig("ideal_tearing_optimization_aux_diagnostics.png", dpi=300)
    print("[PLOT] Saved ideal_tearing_optimization_aux_diagnostics.png")


def plot_gamma_vs_a(history: Dict[str, List[float]], cfg: IdealTearingConfig):
    """Direct view of gamma_hat vs a across all iterations."""
    a = np.array(history["a"])
    gamma_hat = np.array(history["gamma_hat"])

    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    ax.plot(a, gamma_hat, "o-")
    ax.axhline(cfg.gamma_star, color="k", linestyle="--", linewidth=1,
               label=rf"$\hat\gamma_*={cfg.gamma_star:.1f}$")
    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"$\hat{\gamma}$")
    ax.set_title(r"Normalized growth rate vs sheet width $a$")
    ax.legend(fontsize=9)
    fig.savefig("ideal_tearing_gamma_vs_a.png", dpi=300)
    print("[PLOT] Saved ideal_tearing_gamma_vs_a.png")


def plot_growth_rate_comparison(res_init: Dict[str, Any],
                                res_opt: Dict[str, Any]):
    """
    Plot ln|mode_amp| vs t for initial and optimized runs,
    with fitted lines and reported gamma.
    """
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    for res, lab, color in [
        (res_init, "initial", "C0"),
        (res_opt, "optimized", "C3"),
    ]:
        ts = np.array(res["ts"])
        mode_amp = np.array(res["mode_amp_series"])
        gamma_fit, lnA_fit, mask_lin = estimate_growth_rate(
            jnp.asarray(ts), jnp.asarray(mode_amp), w0=mode_amp[0]
        )
        gamma_val = float(gamma_fit)

        ax.plot(
            ts,
            np.log(mode_amp + 1e-30),
            label=rf"{lab} $\ln A$ ($\gamma \approx {gamma_val:.3e}$)",
            alpha=0.8,
            color=color,
        )
        ax.plot(
            ts,
            np.array(lnA_fit),
            linestyle="--",
            alpha=0.7,
            color=color,
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\ln |B_x(k_x=0,k_y=1,k_z=0)|$")
    ax.set_title("Tearing-mode growth rate: initial vs optimized")
    ax.legend(fontsize=8)
    fig.savefig("ideal_tearing_gamma_comparison.png", dpi=300)
    print("[PLOT] Saved ideal_tearing_gamma_comparison.png")


def plot_energy_comparison(res_init: Dict[str, Any],
                           res_opt: Dict[str, Any]):
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
    fig.savefig("ideal_tearing_energy_comparison.png", dpi=300)
    print("[PLOT] Saved ideal_tearing_energy_comparison.png")


# ----------------------------------- main -----------------------------------#

def main():
    cfg = IdealTearingConfig()

    print("========================================================")
    print(" Ideal tearing optimization (differentiable MHD)")
    print("========================================================")
    print(cfg)

    history, res_init, res_opt = run_optimization(cfg)

    print("\n[POST] Making plots...")
    plot_optimization_history(history, cfg)
    plot_aux_diagnostics(history, cfg)
    plot_gamma_vs_a(history, cfg)
    plot_growth_rate_comparison(res_init, res_opt)
    plot_energy_comparison(res_init, res_opt)
    
    # Save initial and optimized solutions as .npz for postprocessing
    print("\n[SAVE] Writing initial and optimized solutions for postprocessing...")

    # Common stem: ideal-tearing, equilibrium mode, target S
    stem_base = f"mhd_tearing_solution_ideal_{cfg.equilibrium_mode}_S{int(cfg.S_target)}"

    payload_init = _prepare_npz_payload(
        res_init,
        extra_meta={
            "opt_script": "mhd_tearing_ideal_tearing_opt",
            "opt_kind": "ideal_tearing_init",
            "S_target": cfg.S_target,
            "gamma_star": cfg.gamma_star,
        },
    )
    fname_init = stem_base + "_init.npz"
    np.savez(fname_init, **payload_init)
    print(f"[SAVE] Initial solution saved to {fname_init}")

    payload_opt = _prepare_npz_payload(
        res_opt,
        extra_meta={
            "opt_script": "mhd_tearing_ideal_tearing_opt",
            "opt_kind": "ideal_tearing_opt",
            "S_target": cfg.S_target,
            "gamma_star": cfg.gamma_star,
        },
    )
    fname_opt = stem_base + "_opt.npz"
    np.savez(fname_opt, **payload_opt)
    print(f"[SAVE] Optimized solution saved to {fname_opt}")


    print("\n[DONE] Ideal tearing optimization finished.")


if __name__ == "__main__":
    main()
