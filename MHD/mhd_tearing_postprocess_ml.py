#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_postprocess_ml.py

Compare data-driven reduced models for Harris-sheet tearing against
a full MHD simulation:

  - Load one MHD tearing solution (.npz) from mhd_tearing_solve.py
  - Recompute tearing amplitude A(t), fitted growth rate gamma_fit,
    and saturated amplitude A_sat from the simulation
  - Load:
      * surrogate_model.pkl  (eq_params -> [gamma_fit, A_sat])
      * latent_ode_model.pkl (latent ODE autoencoder for A(t))
  - Compare:
      * gamma_FKR (theory) vs gamma_fit (sim) vs gamma_surrogate
      * A_sat (sim) vs A_sat_surrogate
      * A(t) vs latent-ODE reconstruction vs simple exponentials
  - Produce publication-ready plots:
      * tearing_ml_growth_and_saturation.png
      * tearing_ml_amplitude_timeseries.png
      * tearing_ml_reconstruction_errors.png
      * tearing_ml_latent_phase.png
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "text.usetex": False,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 200,
})

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax as dfx

from mhd_tearing_solve import (
    make_k_arrays,
    make_dealias_mask,
    project_div_free,
    tearing_amplitude,
)

# Import ML tools and models from the training script
from mhd_tearing_ml import (
    safe_log,
    fit_growth_rate,
    compute_saturated_amplitude,
    mlp_apply,
    load_surrogate_model,
    load_latent_ode_model,
    latent_ode_forward_single,
)

# -----------------------------------------------------------------------------#
# Extra latent utilities: full latent trajectory (for phase plots)
# -----------------------------------------------------------------------------#

def latent_ode_rhs_single(t, z, args):
    """
    RHS used to reconstruct latent trajectories for plotting.

    args = (ode_params, eq_params)
    """
    ode_params, eq_params = args
    inp = jnp.concatenate([z, eq_params], axis=-1)
    dz = mlp_apply(ode_params, inp)
    return dz


def compute_latent_trajectory(latent_model, ts, amp, eq_params):
    """
    Integrate the latent ODE starting from the encoder-based z0_enc
    and return z(t) along with the reconstructed log_amp_hat(t).

    This mirrors latent_ode_forward_single but keeps the full z(t).
    """
    log_amp = safe_log(amp)
    enc_input = jnp.concatenate([log_amp, eq_params], axis=-1)
    z0_enc = mlp_apply(latent_model.encoder_params, enc_input)

    term   = dfx.ODETerm(latent_ode_rhs_single)
    solver = dfx.Tsit5()
    t0 = float(ts[0])
    t1 = float(ts[-1])
    dt0 = float(ts[1] - ts[0]) if ts.size > 1 else 0.1
    saveat = dfx.SaveAt(ts=ts)

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=z0_enc,
        args=(latent_model.ode_params, eq_params),
        saveat=saveat,
        max_steps=10_000,
    )
    z_traj = sol.ys  # (T, latent_dim)

    # Decode to log_amp_hat(t)
    def decode_step(z_t):
        dec_input = jnp.concatenate([z_t, eq_params], axis=-1)
        return mlp_apply(latent_model.decoder_params, dec_input)[0]

    log_amp_hat = jax.vmap(decode_step)(z_traj)  # (T,)
    return z_traj, log_amp_hat


# -----------------------------------------------------------------------------#
# Argument parsing
# -----------------------------------------------------------------------------#

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare ML tearing surrogates and latent ODE "
                    "against a full MHD tearing solution."
    )
    p.add_argument("infile", nargs="?", default="mhd_tearing_solution.npz",
                   help="Input .npz file produced by mhd_tearing_solve.py")
    p.add_argument("--surrogate", type=str,
                   default="surrogate_model.pkl",
                   help="Path to surrogate_model.pkl")
    p.add_argument("--latent", type=str,
                   default="latent_ode_model.pkl",
                   help="Path to latent_ode_model.pkl")
    p.add_argument("--prefix", type=str, default="",
                   help="Prefix for all output figures.")
    return p.parse_args()


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#

def main():
    args = parse_args()

    # ------------------------- Load MHD solution ---------------------------- #
    data = np.load(args.infile, allow_pickle=True)

    ts = np.array(data["ts"])
    v_hat_frames = np.array(data["v_hat"])
    B_hat_frames = np.array(data["B_hat"])

    Nx = int(data["Nx"]); Ny = int(data["Ny"]); Nz = int(data["Nz"])
    Lx = float(data["Lx"]); Ly = float(data["Ly"]); Lz = float(data["Lz"])
    nu = float(data["nu"]); eta = float(data["eta"])
    B0 = float(data["B0"]); a = float(data["a"])
    B_g = float(data["B_g"]); eps_B = float(data["eps_B"])
    gamma_FKR = float(data["gamma_FKR"])

    print("=== ML post-processing of MHD tearing solution ===")
    print(f"infile = {args.infile}")
    print(f"Nx,Ny,Nz = {Nx},{Ny},{Nz}")
    print(f"Lx,Ly,Lz = {Lx},{Ly},{Lz}")
    print(f"nu={nu}, eta={eta}, B0={B0}, a={a}, B_g={B_g}, eps_B={eps_B}")
    print(f"gamma_FKR = {gamma_FKR:.3e}")
    print("===================================================")

    # Spectral operators for tearing amplitude computation
    kx, ky, kz, k2, NX_arr, NY_arr, NZ_arr = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    mask_dealias = make_dealias_mask(Nx, Ny, Nz, NX_arr, NY_arr, NZ_arr)

    # -------------------- Compute A(t), gamma_fit, A_sat -------------------- #
    amp_list = []
    for i in range(len(ts)):
        B_hat_i = jnp.array(B_hat_frames[i]) * mask_dealias
        B_hat_i = project_div_free(B_hat_i, kx, ky, kz, k2)
        A_i = tearing_amplitude(B_hat_i, Lx, Ly, Lz)  # same as in mhd_tearing_ml.py
        amp_list.append(float(A_i))

    amp = np.array(amp_list)
    ts_j = jnp.asarray(ts)
    amp_j = jnp.asarray(amp)

    gamma_fit_sim = fit_growth_rate(ts_j, amp_j)
    Asat_sim = compute_saturated_amplitude(amp_j)

    print(f"[SIM] gamma_fit (sim) ≈ {gamma_fit_sim:.3e}")
    print(f"[SIM] A_sat (sim)     ≈ {Asat_sim:.3e}")

    # eq_params layout must match the training:
    # [B0, a, B_g, eps_B, nu, eta, Ly]
    eq_params = jnp.array([B0, a, B_g, eps_B, nu, eta, Ly], dtype=jnp.float64)

    # ---------------------- Load surrogate model ---------------------------- #
    surrogate = load_surrogate_model(args.surrogate, learning_rate=5e-3)
    print(f"[IO] Loaded surrogate model from '{args.surrogate}'")

    # Surrogate prediction for this equilibrium
    eq_batch = eq_params[None, :]  # shape (1, 7)
    preds = mlp_apply(surrogate.params, eq_batch, activation=jax.nn.swish)
    preds_np = np.array(preds[0])
    gamma_pred = float(preds_np[0])
    Asat_pred  = float(preds_np[1])

    print(f"[SURR] gamma_pred  ≈ {gamma_pred:.3e}")
    print(f"[SURR] A_sat_pred  ≈ {Asat_pred:.3e}")
    if gamma_FKR > 0.0:
        print(f"[RATIO] gamma_fit_sim / gamma_FKR   ≈ {gamma_fit_sim/gamma_FKR:.3f}")
        print(f"[RATIO] gamma_pred     / gamma_FKR   ≈ {gamma_pred/gamma_FKR:.3f}")

    # ---------------------- Load latent ODE model --------------------------- #
    latent_model = load_latent_ode_model(args.latent, learning_rate=1e-3)
    print(f"[IO] Loaded latent ODE model from '{args.latent}'")

    # ------------------------------------------------------------------
    # Reconstruct amplitude using latent ODE
    # We must match the *trained* encoder input dimension:
    #   encoder_in_dim = amp_len_trained + eq_dim
    # so infer amp_len_trained from the first encoder weight matrix
    # and resample A(t) to that length.
    # ------------------------------------------------------------------
    W0_enc = latent_model.encoder_params["Ws"][0]
    in_dim_enc = int(W0_enc.shape[0])
    eq_dim = int(eq_params.shape[0])   # should be 7
    amp_len_trained = in_dim_enc - eq_dim
    if amp_len_trained <= 0:
        raise ValueError(
            f"Invalid encoder input dimension: {in_dim_enc} with eq_dim={eq_dim}"
        )

    # Resample A(t) from the simulation onto the trained encoder grid
    ts_ml = np.linspace(ts[0], ts[-1], amp_len_trained)
    amp_ml = np.interp(ts_ml, ts, amp)

    ts_ml_j = jnp.asarray(ts_ml)
    amp_ml_j = jnp.asarray(amp_ml)

    # Reconstruct amplitude on this ML grid
    log_amp_hat_j, _ = latent_ode_forward_single(
        latent_model, ts_ml_j, amp_ml_j, eq_params
    )
    amp_hat_ml = np.array(jnp.exp(log_amp_hat_j))

    # Also compute full latent trajectory for phase plots on this grid
    z_traj_j, log_amp_hat_traj_j = compute_latent_trajectory(
        latent_model, ts_ml_j, amp_ml_j, eq_params
    )
    z_traj = np.array(z_traj_j)  # (T_ml, latent_dim)

    # -------------------------------------------------------------------------#
    # 1) Growth and saturation comparison: theory vs sim vs surrogate
    # -------------------------------------------------------------------------#

    prefix = args.prefix

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # --- Growth rates ---
    labels_gamma = [r"$\gamma_{\rm FKR}$",
                    r"$\gamma_{\rm fit,\,sim}$",
                    r"$\gamma_{\rm surrogate}$"]
    values_gamma = [gamma_FKR, gamma_fit_sim, gamma_pred]

    ax = axes[0]
    x_pos = np.arange(len(labels_gamma))
    bars = ax.bar(x_pos, values_gamma, color=["#4C72B0", "#55A868", "#C44E52"],
                  alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_gamma, rotation=20)
    ax.set_ylabel(r"Growth rate $\gamma$")
    ax.set_title("Tearing growth rate comparison")
    ax.grid(True, axis="y", alpha=0.3)

    for b, v in zip(bars, values_gamma):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.3e}", ha="center", va="bottom", fontsize=9)

    # --- Saturated amplitudes ---
    labels_Asat = [r"$A_{\rm sat,\,sim}$",
                   r"$A_{\rm sat,\,surrogate}$"]
    values_Asat = [Asat_sim, Asat_pred]

    ax2 = axes[1]
    x2 = np.arange(len(labels_Asat))
    bars2 = ax2.bar(x2, values_Asat, color=["#4C72B0", "#C44E52"],
                    alpha=0.9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels_Asat, rotation=15)
    ax2.set_ylabel(r"Saturated amplitude $A_{\rm sat}$")
    ax2.set_title("Saturated amplitude comparison")
    ax2.grid(True, axis="y", alpha=0.3)

    for b, v in zip(bars2, values_Asat):
        ax2.text(b.get_x() + b.get_width() / 2, v,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out1 = prefix + "tearing_ml_growth_and_saturation.png"
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out1}")

    # -------------------------------------------------------------------------#
    # 2) Amplitude time series: A(t) vs ML reconstructions and exponentials
    #     (all on the ML grid ts_ml)
    # -------------------------------------------------------------------------#

    # Simple exponential models on the ML grid, anchored at t0
    A0 = amp_ml[0]
    A_exp_fit  = A0 * np.exp(gamma_fit_sim * (ts_ml - ts_ml[0]))
    A_exp_pred = A0 * np.exp(gamma_pred    * (ts_ml - ts_ml[0]))
    if gamma_FKR > 0.0:
        A_exp_FKR = A0 * np.exp(gamma_FKR * (ts_ml - ts_ml[0]))
    else:
        A_exp_FKR = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Linear scale ---
    ax = axes[0]
    ax.plot(ts_ml, amp_ml, label="Sim (resampled)", lw=2)
    ax.plot(ts_ml, amp_hat_ml, "--", label="Latent ODE", lw=2)
    ax.plot(ts_ml, A_exp_fit, ":", label=r"Exp($\gamma_{\rm fit,sim}$)", lw=1.8)
    ax.plot(ts_ml, A_exp_pred, "-.", label=r"Exp($\gamma_{\rm surrogate}$)", lw=1.8)
    if A_exp_FKR is not None:
        ax.plot(ts_ml, A_exp_FKR, "--", color="gray",
                label=r"Exp($\gamma_{\rm FKR}$)", lw=1.5)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$A(t)$")
    ax.set_title("Tearing amplitude (linear scale)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # --- Log scale ---
    ax2 = axes[1]
    ax2.plot(ts_ml, np.log(amp_ml + 1e-16), label="Sim (resampled)", lw=2)
    ax2.plot(ts_ml, np.log(amp_hat_ml + 1e-16), "--", label="Latent ODE", lw=2)
    ax2.plot(ts_ml, np.log(A_exp_fit + 1e-16), ":", label=r"Exp($\gamma_{\rm fit,sim}$)", lw=1.8)
    ax2.plot(ts_ml, np.log(A_exp_pred + 1e-16), "-.", label=r"Exp($\gamma_{\rm surrogate}$)", lw=1.8)
    if A_exp_FKR is not None:
        ax2.plot(ts_ml, np.log(A_exp_FKR + 1e-16), "--", color="gray",
                 label=r"Exp($\gamma_{\rm FKR}$)", lw=1.5)

    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$\ln A(t)$")
    ax2.set_title("Tearing amplitude (log scale)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    out2 = prefix + "tearing_ml_amplitude_timeseries.png"
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out2}")

    # -------------------------------------------------------------------------#
    # 3) Reconstruction errors: latent ODE vs simulation (on ML grid)
    # -------------------------------------------------------------------------#

    rel_err = (amp_hat_ml - amp_ml) / (amp_ml + 1e-16)
    abs_err = amp_hat_ml - amp_ml

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(ts_ml, rel_err, lw=1.8)
    ax.axhline(0.0, color="k", lw=1, alpha=0.5)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"Relative error $(A_{\rm LODE}-A_{\rm sim})/A_{\rm sim}$")
    ax.set_title("Latent ODE reconstruction error (time trace)")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(rel_err, bins=20, density=True, alpha=0.8, edgecolor="k")
    ax2.set_xlabel(r"Relative error")
    ax2.set_ylabel("PDF")
    ax2.set_title("Distribution of reconstruction error")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out3 = prefix + "tearing_ml_reconstruction_errors.png"
    fig.savefig(out3, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out3}")

    # -------------------------------------------------------------------------#
    # 4) Latent phase portrait: first two latent coordinates vs time
    # -------------------------------------------------------------------------#

    latent_dim = z_traj.shape[1]
    fig = plt.figure(figsize=(11, 4))

    # Left: z1 vs z2 colored by time
    if latent_dim >= 2:
        ax1 = fig.add_subplot(1, 2, 1)
        sc = ax1.scatter(z_traj[:, 0], z_traj[:, 1],
                         c=ts_ml, cmap="viridis", s=20, edgecolors="none")
        ax1.set_xlabel(r"$z_1$")
        ax1.set_ylabel(r"$z_2$")
        ax1.set_title("Latent trajectory (first two dims)")
        ax1.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax1)
        cbar.set_label(r"$t$")
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.text(0.5, 0.5, "latent_dim < 2", ha="center", va="center")
        ax1.set_axis_off()

    # Right: latent coordinates vs time
    ax2 = fig.add_subplot(1, 2, 2)
    for i in range(min(latent_dim, 4)):
        ax2.plot(ts_ml, z_traj[:, i], label=fr"$z_{i+1}(t)$", lw=1.8)
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$z_i(t)$")
    ax2.set_title("Latent coordinates vs time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    out4 = prefix + "tearing_ml_latent_phase.png"
    fig.savefig(out4, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out4}")

    print("\n[DONE] ML postprocessing complete.")
    print("Generated figures:")
    print(f"  - {out1}")
    print(f"  - {out2}")
    print(f"  - {out3}")
    print(f"  - {out4}")


if __name__ == "__main__":
    main()
