#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_ideal_tearing_opt.py

Differentiable “ideal tearing” benchmark
========================================

Goal
----
We use the incompressible pseudo-spectral MHD solver
`_run_tearing_simulation_and_diagnostics` (from mhd_tearing_solve.py),
together with JAX autodiff, to *optimize* the current-sheet half-width `a`
at fixed Lundquist number S_target so that the normalized tearing growth rate

    gamma_hat = gamma * a / v_A

is as close as possible to order unity, i.e. the *ideal tearing* regime
(Pucci & Velli).

We take:
    S = a B0 / eta     (with B0 = v_A = 1, rho = 1)
and enforce the constraint by setting

    eta(a) = a B0 / S_target.

The scalar objective is

    J(a) = (gamma_hat(a) - gamma_star)^2,

with gamma_star ≈ 1.

What this script does
---------------------
1) Defines an objective J(log_a) that:
     - runs one MHD tearing simulation,
     - extracts the growth rate gamma_fit from mode_amp_series via a linear
       fit of ln A(t),
     - builds gamma_hat(a) = gamma_fit * a / v_A,
     - returns J with detailed, AD-safe debug printing.
2) Performs gradient descent on log(a) (ensuring a > 0) to minimize J.
3) Generates a suite of *publication-ready* figures:
     - optimization history (a, gamma_hat, J vs iteration),
     - auxiliary diagnostics (eta, S, linear-fit window, |grad J|),
     - gamma_hat vs a across iterations,
     - ln|B_x(kx=0,ky=1)| vs t for initial vs optimized (with fits),
     - energy evolution (initial vs optimized),
     - real-space fields at t=0 and t=t_final for initial vs optimized:
         * B_x(x,y) on the midplane,
         * A_z(x,y) (flux function) on the midplane.
4) Saves initial and optimized simulations as .npz files with metadata,
   for later post-processing.

Usage
-----
    python mhd_tearing_ideal_tearing_opt.py

Adjust the `IdealTearingConfig` dataclass below for resolution, S_target,
number of optimization steps, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mhd_tearing_solve import (
    _run_tearing_simulation_and_diagnostics,
    estimate_growth_rate,
    compute_k_arrays_np,
    compute_Az_hat_np,
)

# -------------------------- Global plotting style ---------------------------#

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


# --------------------------- Configuration dataclass ------------------------#

@dataclass
class IdealTearingConfig:
    """
    Configuration for the ideal-tearing optimization experiment.

    All dimensional quantities are in the same units used by
    mhd_tearing_solve.py. By default we work on a 2π-periodic square box
    in x and y, with Nz = 1 (kx, ky spectral grid; kz=0).
    """
    # Grid and box
    Nx: int = 64
    Ny: int = 64
    Nz: int = 1
    Lx: float = 2.0 * math.pi
    Ly: float = 2.0 * math.pi
    Lz: float = 2.0 * math.pi

    # Physical parameters
    B0: float = 1.0      # upstream field; also v_A since rho=1
    B_g: float = 0.2     # guide field
    nu: float = 1e-3     # viscosity
    eps_B: float = 1e-3  # noise level used in initialization (if any)

    # Time integration
    t0: float = 0.0
    t1: float = 20.0
    n_frames: int = 80
    dt0: float = 5e-4

    # Ideal tearing target
    S_target: float = 1e4      # target Lundquist number based on sheet half-width a
    gamma_star: float = 1.0    # target normalized growth rate gamma_hat = gamma a / v_A

    # Optimization hyperparameters
    n_opt_steps: int = 20
    lr_log_a: float = 0.5      # learning rate for log(a)

    # Initial guess for a (current-sheet half width)
    a0: float = 0.25           # in units of the box length (Lx ~ 2π)

    # Equilibrium branch; "original" is Harris-sheet-like
    equilibrium_mode: str = "original"


# --------------------------- Objective functional ---------------------------#

def _simulate_and_gamma_hat(log_a: jnp.ndarray,
                            cfg: IdealTearingConfig):
    """
    Given log(a), run one tearing simulation and return:

      gamma_hat, gamma_fit, a, eta, S,
      t_lin_start, t_lin_end, n_lin, res

    where
        a            = exp(log_a)  (current-sheet half width),
        eta(a)       = a * B0 / S_target,
        S(a)         = a B0 / eta(a) (≈ S_target, used only for diagnostics),
        gamma_fit    = growth rate of tearing amplitude A(t),
        gamma_hat    = gamma_fit * a / v_A,  v_A = B0,

    and `res` is the full result dict from the MHD solver, augmented
    with a few extra keys used for plotting and debugging.

    The growth rate is extracted from the time series of the tearing-mode
    amplitude Bx(kx=0, ky=1, kz=0) via a linear fit to ln A(t) over a
    semi-automatically chosen linear window.
    """
    a = jnp.exp(log_a)
    B0 = cfg.B0
    vA = B0

    # Enforce S(a) = a B0 / eta(a) = S_target  ->  eta(a) = a B0 / S_target.
    eta = a * B0 / cfg.S_target
    nu = cfg.nu

    # Run the MHD tearing simulation with these parameters.
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

    ts = res["ts"]                       # shape (T,)
    mode_amp = res["mode_amp_series"]    # tearing amplitude A(t)

    # Fit ln A(t) in the linear phase.
    gamma_fit, lnA_fit, mask_lin = estimate_growth_rate(ts, mode_amp, w0=mode_amp[0])

    # Normalized growth rate: gamma_hat = gamma * a / v_A.
    gamma_hat = gamma_fit * a / vA
    S = a * B0 / eta  # should be ≈ S_target (debug only)

    # Linear-fit window diagnostics
    ts_lin = ts[mask_lin]
    has_lin = jnp.any(mask_lin)
    t_lin_start = jnp.where(has_lin, ts_lin[0], jnp.asarray(cfg.t0))
    t_lin_end = jnp.where(has_lin, ts_lin[-1], jnp.asarray(cfg.t1))
    n_lin = jnp.sum(mask_lin.astype(jnp.int32))

    # Attach relevant diagnostics back onto the result dict for plotting later.
    res = dict(res)
    res["gamma_fit"] = gamma_fit
    res["gamma_hat"] = gamma_hat
    res["mask_lin"] = mask_lin
    res["t_lin_start"] = t_lin_start
    res["t_lin_end"] = t_lin_end
    res["lnA_fit"] = lnA_fit
    res["a"] = a
    res["eta"] = eta
    res["S"] = S

    return gamma_hat, gamma_fit, a, eta, S, t_lin_start, t_lin_end, n_lin, res


def objective(log_a: jnp.ndarray, cfg: IdealTearingConfig) -> jnp.ndarray:
    """
    Objective functional J(log_a) for ideal tearing:

        J = (gamma_hat(a) - gamma_star)^2

    where gamma_hat(a) = gamma_fit(a) * a / v_A.

    This function is JAX-differentiable and uses jax.debug.print for
    detailed, AD-safe diagnostics.
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

    # AD-safe debug printing: this runs only in traced mode.
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


# ----------------------- Helper: NPZ payload conversion ---------------------#

def _prepare_npz_payload(res: Dict[str, Any],
                         extra_meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Convert the result dict from `_run_tearing_simulation_and_diagnostics`
    into something `np.savez` can digest (NumPy arrays + scalars).

    Any `extra_meta` entries are added on top.
    Non-convertible objects are silently skipped.
    """
    payload: Dict[str, Any] = {}
    for key, val in res.items():
        if isinstance(val, (int, float, np.number, str)):
            payload[key] = val
            continue
        try:
            payload[key] = np.asarray(val)
        except Exception:
            # e.g. callables; safe to ignore for saving.
            pass

    if extra_meta is not None:
        payload.update(extra_meta)
    return payload


# --------------------------- Optimization driver ----------------------------#

def run_optimization(cfg: IdealTearingConfig):
    """
    Perform gradient-descent optimization of log(a) to reach ideal tearing.

    Returns
    -------
    history : dict of lists
        Time series of a, eta, S, gamma_hat, gamma_fit, J, etc.
    res_init : dict
        MHD simulation result for the initial a0.
    res_opt : dict
        MHD simulation result for the *last* iterate (optimized a).
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
    # 1) Initial evaluation: baseline tearing run at a0.
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
    history["grad_log_a"].append(np.nan)  # undefined for the very first point

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
    # 2) Gradient-descent loop in log(a).
    # ------------------------------------------------------------------#
    log_a = log_a0
    res_opt = res_init  # will be overwritten each iteration

    print("\n[OPT] Starting gradient descent on log(a)...")
    for k in range(cfg.n_opt_steps):
        # Compute J and its gradient with respect to log(a).
        J_val, grad_log_a = value_and_grad(log_a, cfg)

        # Gradient step in log-space (a always positive).
        log_a = log_a - cfg.lr_log_a * grad_log_a

        # Re-evaluate tearing run at updated a.
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

        # Record in Python lists for plotting.
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


# --------------------------- Field reconstruction ---------------------------#

def _extract_Bx_Az_xy(res: Dict[str, Any], time_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct B_x(x,y) and A_z(x,y) on the z=0 midplane from one
    snapshot of the spectral fields stored in `res["B_hat"]`.

    Parameters
    ----------
    res : dict
        Result dictionary from the MHD solver.
    time_index : int
        Index in time (0 for t0, -1 for final time, etc.).

    Returns
    -------
    Bx_xy : ndarray, shape (Nx, Ny)
        Real-space B_x(x,y,z=0) at the given time index.
    Az_xy : ndarray, shape (Nx, Ny)
        Real-space A_z(x,y,z=0), reconstructed from B_x, B_y via
        the helper compute_Az_hat_np.
    """
    B_hat_T = np.asarray(res["B_hat"][time_index])  # shape (3, Nx, Ny, Nz)
    Nx, Ny, Nz = B_hat_T.shape[1:]
    Lx, Ly, Lz = float(res["Lx"]), float(res["Ly"]), float(res["Lz"])

    # Real-space magnetic field via inverse FFT.
    B_real = np.fft.ifftn(B_hat_T, axes=(1, 2, 3)).real
    Bx_xy = B_real[0, :, :, 0]  # midplane z=0

    # Flux function A_z such that B = ∇ × (A_z e_z) + ... (here we use helper).
    kx, ky, kz, NX, NY, NZ = compute_k_arrays_np(Nx, Ny, Nz, Lx, Ly, Lz)
    Az_hat = compute_Az_hat_np(B_hat_T, kx, ky)
    Az_real = np.fft.ifftn(Az_hat, axes=(0, 1, 2)).real
    Az_xy = Az_real[:, :, 0]   # z=0

    return Bx_xy, Az_xy


# --------------------------- Plotting utilities -----------------------------#

def plot_optimization_history(history: Dict[str, List[float]],
                              cfg: IdealTearingConfig):
    """
    Plot the global optimization history: a, gamma_hat, and J vs iteration.
    """
    iters = np.arange(len(history["a"]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

    # (1) a vs iteration
    axes[0].plot(iters, history["a"], marker="o")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel(r"$a$")
    axes[0].set_title("Sheet half-width $a$")

    # (2) gamma_hat vs iteration
    axes[1].plot(iters, history["gamma_hat"], marker="o")
    axes[1].axhline(cfg.gamma_star, color="k", linestyle="--", linewidth=1)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"$\hat{\gamma} = \gamma a / v_A$")
    axes[1].set_title("Normalized growth rate")

    # (3) J vs iteration (log scale)
    J = np.abs(history["J"])
    axes[2].semilogy(iters, J, marker="o")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel(r"$J$")
    axes[2].set_title("Objective")

    fig.suptitle("Ideal tearing optimization history", fontsize=14)
    fig.savefig("ideal_tearing_optimization_history.png", dpi=300)
    print("[PLOT] Saved ideal_tearing_optimization_history.png")


def plot_aux_diagnostics(history: Dict[str, List[float]],
                         cfg: IdealTearingConfig):
    """
    Plot auxiliary diagnostics:
      - eta and S vs iteration,
      - linear-fit time window and |grad log a J| vs iteration.
    """
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

    # Right: linear fit window and gradient norm
    ax1 = axes[1]
    ax1.plot(iters, t0_lin, "o-", label=r"$t_{\mathrm{lin,start}}$")
    ax1.plot(iters, t1_lin, "o-", label=r"$t_{\mathrm{lin,end}}$")
    ax1.axhline(cfg.t1, color="k", linestyle=":", label=r"$t_1$")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel(r"$t$")
    ax1.set_title("Linear fit window")
    ax1.legend(fontsize=8, loc="upper left")

    ax1b = ax1.twinx()
    ax1b.semilogy(
        iters,
        np.where(np.isnan(grad), np.nan, grad),
        "C3--",
        label=r"$|\nabla_{\log a} J|$",
    )
    ax1b.set_ylabel(r"$|\nabla_{\log a} J|$")
    ax1b.tick_params(axis="y", labelcolor="C3")

    fig.suptitle("Ideal tearing auxiliary diagnostics", fontsize=14)
    fig.savefig("ideal_tearing_optimization_aux_diagnostics.png", dpi=300)
    print("[PLOT] Saved ideal_tearing_optimization_aux_diagnostics.png")


def plot_gamma_vs_a(history: Dict[str, List[float]],
                    cfg: IdealTearingConfig):
    """
    Plot normalized growth rate gamma_hat vs sheet half-width a across
    all iterations. This gives a direct view of the ideal-tearing trend.
    """
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
    Compare ln|B_x(kx=0,ky=1)| vs t for initial and optimized runs,
    including the linear fits used to extract gamma.
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
            label=rf"{lab} data ($\gamma \approx {gamma_val:.3e}$)",
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
    ax.set_title("Tearing-mode growth: initial vs optimized")
    ax.legend(fontsize=8)
    fig.savefig("ideal_tearing_gamma_comparison.png", dpi=300)
    print("[PLOT] Saved ideal_tearing_gamma_comparison.png")


def plot_energy_comparison(res_init: Dict[str, Any],
                           res_opt: Dict[str, Any]):
    """
    Compare kinetic and magnetic energy evolution for initial vs optimized
    tearing runs.
    """
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


def plot_initial_final_fields(res_init: Dict[str, Any],
                              res_opt: Dict[str, Any],
                              cfg: IdealTearingConfig):
    """
    Real-space fields at t=0 and t=t_final on the midplane z=0, for both
    the initial and optimized sheet width.

    Two figures are generated:

      1) Bx(x,y) at (t0, t_final) for initial and optimized runs.
      2) Az(x,y) at (t0, t_final) for initial and optimized runs.
    """
    # Spatial grids (same for all runs).
    x = np.linspace(0.0, cfg.Lx, cfg.Nx, endpoint=False)
    y = np.linspace(0.0, cfg.Ly, cfg.Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # --- Bx fields ----------------------------------------------------#
    Bx_i_0, _ = _extract_Bx_Az_xy(res_init, time_index=0)
    Bx_i_f, _ = _extract_Bx_Az_xy(res_init, time_index=-1)
    Bx_o_0, _ = _extract_Bx_Az_xy(res_opt, time_index=0)
    Bx_o_f, _ = _extract_Bx_Az_xy(res_opt, time_index=-1)

    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    im = axes1[0, 0].pcolormesh(X, Y, Bx_i_0, shading="auto")
    axes1[0, 0].set_title(r"Initial $a$: $B_x(x,y,t_0)$")
    axes1[0, 0].set_xlabel(r"$x$")
    axes1[0, 0].set_ylabel(r"$y$")
    fig1.colorbar(im, ax=axes1[0, 0])

    im = axes1[0, 1].pcolormesh(X, Y, Bx_i_f, shading="auto")
    axes1[0, 1].set_title(r"Initial $a$: $B_x(x,y,t_{\mathrm{final}})$")
    axes1[0, 1].set_xlabel(r"$x$")
    axes1[0, 1].set_ylabel(r"$y$")
    fig1.colorbar(im, ax=axes1[0, 1])

    im = axes1[1, 0].pcolormesh(X, Y, Bx_o_0, shading="auto")
    axes1[1, 0].set_title(r"Optimized $a$: $B_x(x,y,t_0)$")
    axes1[1, 0].set_xlabel(r"$x$")
    axes1[1, 0].set_ylabel(r"$y$")
    fig1.colorbar(im, ax=axes1[1, 0])

    im = axes1[1, 1].pcolormesh(X, Y, Bx_o_f, shading="auto")
    axes1[1, 1].set_title(r"Optimized $a$: $B_x(x,y,t_{\mathrm{final}})$")
    axes1[1, 1].set_xlabel(r"$x$")
    axes1[1, 1].set_ylabel(r"$y$")
    fig1.colorbar(im, ax=axes1[1, 1])

    fig1.suptitle("Midplane $B_x(x,y)$: initial vs optimized, early vs late", fontsize=14)
    fig1.savefig("ideal_tearing_Bx_fields.png", dpi=300)
    print("[PLOT] Saved ideal_tearing_Bx_fields.png")

    # --- Az fields ----------------------------------------------------#
    _, Az_i_0 = _extract_Bx_Az_xy(res_init, time_index=0)
    _, Az_i_f = _extract_Bx_Az_xy(res_init, time_index=-1)
    _, Az_o_0 = _extract_Bx_Az_xy(res_opt, time_index=0)
    _, Az_o_f = _extract_Bx_Az_xy(res_opt, time_index=-1)

    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    im = axes2[0, 0].pcolormesh(X, Y, Az_i_0, shading="auto")
    axes2[0, 0].set_title(r"Initial $a$: $A_z(x,y,t_0)$")
    axes2[0, 0].set_xlabel(r"$x$")
    axes2[0, 0].set_ylabel(r"$y$")
    fig2.colorbar(im, ax=axes2[0, 0])

    im = axes2[0, 1].pcolormesh(X, Y, Az_i_f, shading="auto")
    axes2[0, 1].set_title(r"Initial $a$: $A_z(x,y,t_{\mathrm{final}})$")
    axes2[0, 1].set_xlabel(r"$x$")
    axes2[0, 1].set_ylabel(r"$y$")
    fig2.colorbar(im, ax=axes2[0, 1])

    im = axes2[1, 0].pcolormesh(X, Y, Az_o_0, shading="auto")
    axes2[1, 0].set_title(r"Optimized $a$: $A_z(x,y,t_0)$")
    axes2[1, 0].set_xlabel(r"$x$")
    axes2[1, 0].set_ylabel(r"$y$")
    fig2.colorbar(im, ax=axes2[1, 0])

    im = axes2[1, 1].pcolormesh(X, Y, Az_o_f, shading="auto")
    axes2[1, 1].set_title(r"Optimized $a$: $A_z(x,y,t_{\mathrm{final}})$")
    axes2[1, 1].set_xlabel(r"$x$")
    axes2[1, 1].set_ylabel(r"$y$")
    fig2.colorbar(im, ax=axes2[1, 1])

    fig2.suptitle("Midplane $A_z(x,y)$: initial vs optimized, early vs late", fontsize=14)
    fig2.savefig("ideal_tearing_Az_fields.png", dpi=300)
    print("[PLOT] Saved ideal_tearing_Az_fields.png")


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
    plot_initial_final_fields(res_init, res_opt, cfg)

    # Save initial and optimized solutions as .npz for postprocessing
    print("\n[SAVE] Writing initial and optimized solutions for postprocessing...")

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
