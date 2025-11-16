#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-structure manufactured-solution test for the FCI operator on a torus.

PDE (the one the code actually discretizes):

    kappa_par * (b · ∇)^2 ψ  +  kappa_perp * ∇² ψ = f

with b = B / |B|. We use a synthetic "magnetic field" B = e_phi (pure
toroidal). The FCI machinery only needs the direction of B.

Manufactured solution:

    ψ_exact(x,y,z) = z^3

Then

    (b · ∇) ψ_exact = 0,
    ∇² ψ_exact = 6 z,

so we choose

    f(x,y,z) = 6 * kappa_perp * z

which makes ψ_exact an exact *continuous* solution of the PDE.

Domain:

    Analytic circular torus with:
        R0 = major radius,
        a  = minor radius.

We work on a uniform Cartesian grid that contains the torus, and define a
logical mask for the interior of the torus. On top of that we define:

    - an axis band (small region near the magnetic axis),
    - a boundary band (shell near the torus boundary).

On both bands we impose Dirichlet boundary conditions using ψ_exact.

We then:

  1. Build FCI connectivity with build_fci_connectivity_chunked.
  2. Build the JAX FCI operator with make_fci_operator_jax.
  3. Assemble and solve the lifted system A ψ = f using JAX CG.
  4. Measure relative errors in the interior (excluding the bands).
  5. Produce publication-ready plots:
       * Error histogram (relative error).
       * Midplane 1D profile + relative error.
       * Midplane slices of ψ_exact, ψ_num, relative error.
       * 3D boundary scatter plots: |∇φ| (here |B|), ψ_num, relative error.
  6. Run a convergence sweep over N and fit the observed order of
     convergence (log-log slope vs grid spacing h ~ 1/N).

This is designed to test the *full* FCI pipeline (connectivity, operator,
bands, CG) on a non-trivial torus geometry, and to show the expected
second-order convergence in grid spacing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from solve_flux_psi_fci import (
    build_fci_connectivity_chunked,
    make_fci_operator_jax,
    cg_jax,
)

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def pct(a, p):
    return float(np.percentile(np.asarray(a), p))

def pinfo(msg):
    print(f"[INFO] {msg}")


# ---------------------------------------------------------------------------
# Analytic torus geometry and grid
# ---------------------------------------------------------------------------

@dataclass
class TorusGeom:
    R0: float  # major radius
    a: float   # minor radius


def cartesian_grid_for_torus(N: int, geom: TorusGeom, pad_factor: float = 1.2):
    """
    Build a Cartesian grid that comfortably contains the torus.

    Returns: xs, ys, zs, dx, dy, dz, Xq (Ntot x 3)
    """
    R0, a = geom.R0, geom.a
    R_max = R0 + a
    x_max = pad_factor * R_max
    x_min = -x_max
    y_min, y_max = x_min, x_max
    z_min, z_max = -pad_factor * a, pad_factor * a

    xs = np.linspace(x_min, x_max, N)
    ys = np.linspace(y_min, y_max, N)
    zs = np.linspace(z_min, z_max, N)

    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)

    Xq = np.column_stack([XX.ravel(order="C"),
                          YY.ravel(order="C"),
                          ZZ.ravel(order="C")])
    return xs, ys, zs, dx, dy, dz, Xq


def torus_inside_mask(xs, ys, zs, geom: TorusGeom):
    """
    inside3[i,j,k] = True if point (x_i, y_j, z_k) lies inside the torus:

        ρ <= a  with  ρ = sqrt((R - R0)^2 + Z^2),  R = sqrt(x^2 + y^2).
    """
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)

    R = np.sqrt(XX**2 + YY**2)
    rho = np.sqrt((R - geom.R0)**2 + ZZ**2)
    inside3 = rho <= geom.a
    return inside3


def build_axis_and_boundary_bands(xs, ys, zs, geom: TorusGeom,
                                  h_band_factor: float = 2.0,
                                  axis_frac: float = 0.1):
    """
    Construct logical masks for:
        - inside nodes  (torus domain),
        - boundary band near ρ ≈ a,
        - axis band      near ρ ≈ 0.

    h_band_factor controls the thickness of the boundary band in units of
    min(dx,dy,dz). axis_frac controls what fraction of the minor radius
    is considered "axis band".
    """
    nx, ny, nz = len(xs), len(ys), len(zs)
    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]
    voxel = min(dx, dy, dz)

    inside3 = torus_inside_mask(xs, ys, zs, geom)
    inside_flat = inside3.ravel(order="C")

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)

    R = np.sqrt(XX**2 + YY**2)
    rho = np.sqrt((R - geom.R0)**2 + ZZ**2)

    # Axis band: rho <= axis_frac * a
    R_axis_band = axis_frac * geom.a
    axis_band3 = (rho <= R_axis_band) & inside3

    # Boundary band: |rho - a| <= h_band
    h_band = h_band_factor * voxel
    boundary_band3 = (np.abs(rho - geom.a) <= h_band) & inside3

    axis_band_flat = axis_band3.ravel(order="C")
    boundary_band_flat = boundary_band3.ravel(order="C")
    rho_flat = rho.ravel(order="C")
    return inside_flat, axis_band_flat, boundary_band_flat, rho_flat


# ---------------------------------------------------------------------------
# Manufactured solution and RHS
# ---------------------------------------------------------------------------

def psi_exact_fn_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Manufactured exact solution ψ_exact(x,y,z) = z^3."""
    return z**3


def analytic_rhs_f_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       kappa_perp: float):
    """
    Analytic forcing for

        kappa_par * ∂_∥^2 ψ + kappa_perp * ∇² ψ = f

    with ψ = z^3 and B = e_phi (so ∂_∥ ψ = 0):

        ∇² z^3 = 6 z  ⇒  f = 6 * kappa_perp * z
    """
    return 6.0 * kappa_perp * z


# ---------------------------------------------------------------------------
# Synthetic "∇φ" field for FCI (pure toroidal)
# ---------------------------------------------------------------------------

@jit
def grad_phi_toroidal(X: jnp.ndarray) -> jnp.ndarray:
    """Return a synthetic "∇φ" that is a pure toroidal unit vector B = e_phi.

    Only the direction of B matters for FCI.
    """
    x = X[..., 0]
    y = X[..., 1]
    z = X[..., 2]
    R = jnp.sqrt(x * x + y * y)

    ephi = jnp.stack([-y, x, 0.0 * z], axis=-1)
    R_safe = jnp.maximum(R, 1e-8)
    B = ephi / R_safe[..., None]
    B = jnp.where(R[..., None] > 1e-8, B, jnp.zeros_like(B))
    return B


# ---------------------------------------------------------------------------
# Single torus MMS solve
# ---------------------------------------------------------------------------

def solve_fci_torus_mms(
    N: int = 64,
    R0: float = 1.5,
    a: float = 0.5,
    kappa_par: float = 1.0,
    kappa_perp: float = 0.05,
    h_band_factor: float = 2.0,
    axis_frac: float = 0.1,
    nfp: int = 2,
    fci_nsteps: int = 16,
    fci_planes_per_field_period: int = 8,
    make_plots: bool = True,
    save_prefix: str = "fci_torus_mms",
):
    """Run a single manufactured-solution test on a torus."""
    geom = TorusGeom(R0=R0, a=a)

    print("\n" + "=" * 70)
    print(f"FCI torus manufactured test: N={N}, R0={R0}, a={a}, "
          f"kappa_perp={kappa_perp}")
    print("=" * 70)

    xs, ys, zs, dx, dy, dz, Xq = cartesian_grid_for_torus(N, geom)
    nx, ny, nz = len(xs), len(ys), len(zs)
    Ntot = nx * ny * nz
    pinfo(f"[GRID] nx,ny,nz = {nx},{ny},{nz} (Ntot={Ntot})")
    pinfo(f"[GRID] dx,dy,dz ≈ {dx:.4g}, {dy:.4g}, {dz:.4g}")

    inside_flat, axis_band_flat, boundary_band_flat, _ = \
        build_axis_and_boundary_bands(
            xs, ys, zs, geom, h_band_factor=h_band_factor, axis_frac=axis_frac
        )
    inside3 = inside_flat.reshape(nx, ny, nz)

    pinfo(f"[DOMAIN] Inside nodes: {inside_flat.sum()} / {Ntot}")
    pinfo(f"[BANDS] #boundary band nodes: {boundary_band_flat.sum()} / {Ntot}")
    pinfo(f"[BANDS] #axis band nodes    : {axis_band_flat.sum()} / {Ntot}")

    # Manufactured exact solution and analytic RHS
    xq, yq, zq = Xq[:, 0], Xq[:, 1], Xq[:, 2]
    psi_exact = psi_exact_fn_xyz(xq, yq, zq)
    f_rhs = analytic_rhs_f_xyz(xq, yq, zq, kappa_perp=kappa_perp)

    # Build FCI connectivity
    pinfo("[FCI] Building connectivity ...")
    fci = build_fci_connectivity_chunked(
        xs, ys, zs,
        inside_mask=inside_flat,
        grad_phi_fn=grad_phi_toroidal,
        nfp=nfp,
        dphi_per_step=None,
        nsteps=fci_nsteps,
        verbose=True,
        chunk_size=None,
        fci_planes_per_field_period=fci_planes_per_field_period,
    )
    pinfo(f"[FCI] valid connectivity nodes: {fci.valid.sum()} / {inside_flat.sum()}")

    # Core region (where FCI parallel operator is applied)
    core3 = np.zeros_like(inside3, dtype=bool)
    core3[1:-1, 1:-1, 1:-1] = True
    core3 &= inside3
    core_flat = core3.ravel(order="C")

    # Build combined FCI + 7-point Laplacian operator
    A_pde_jax, deep_inside = make_fci_operator_jax(
        nx, ny, nz,
        xs, ys, zs,
        inside_flat,
        fci,
        core_mask=core_flat,
        kappa_par=kappa_par,
        kappa_perp=kappa_perp,
    )

    # Dirichlet BCs on boundary + axis bands from ψ_exact
    fixed = (boundary_band_flat | axis_band_flat)
    val = np.zeros(Ntot, dtype=float)
    val[fixed] = psi_exact[fixed]

    free = deep_inside & (~fixed)
    if not np.any(free):
        raise RuntimeError("No free nodes in torus MMS test.")
    pinfo(f"[SOLVE] Free nodes: {free.sum()} / {Ntot}")

    # Lifting: ψ = ψ_fixed + ψ_free
    psi_fixed_full = np.zeros(Ntot, dtype=float)
    psi_fixed_full[fixed] = val[fixed]

    # RHS in the lifted system:
    #   A[ψ_free] = f_rhs - A[ψ_fixed]
    Apsi_fixed_full = np.array(A_pde_jax(jnp.asarray(psi_fixed_full)))
    b_full = f_rhs - Apsi_fixed_full
    b_free = b_full[free]
    free_mask = free.copy()

    def matvec_free_jax(u_free_j: jnp.ndarray) -> jnp.ndarray:
        u_full = jnp.zeros(Ntot, dtype=jnp.float64)
        u_full = u_full.at[free_mask].set(u_free_j)
        Au_full = A_pde_jax(u_full)
        return Au_full[free_mask]

    pinfo("[SOLVE] Solving A ψ = f (JAX CG) ...")
    b_free_j = jnp.asarray(b_free, dtype=jnp.float64)
    psi_free_j, res_norm = cg_jax(
        matvec_free_jax, b_free_j, tol=1e-10, maxiter=5000
    )
    pinfo(f"[SOLVE] CG finished with ||r||₂ ≈ {float(res_norm):.3e}")

    psi_num = np.array(psi_fixed_full)
    psi_num[free] = np.asarray(psi_free_j)

    # Interior region = inside but not in axis / boundary bands
    interior = inside_flat & (~boundary_band_flat) & (~axis_band_flat)
    psi_int_exact = psi_exact[interior]
    psi_int_num = psi_num[interior]

    denom_int = np.maximum(np.abs(psi_int_exact), 1e-12)
    rel_err_int = np.abs(psi_int_num - psi_int_exact) / denom_int

    rel_L2 = (np.linalg.norm(psi_int_num - psi_int_exact) /
              np.linalg.norm(psi_int_exact))
    Linf_rel = float(np.max(rel_err_int))

    print(f"[ERROR] relative L2 error (interior) = {rel_L2:.3e}")
    print(f"[ERROR] max relative error (interior) = {Linf_rel:.3e}")

    # ----------------------------------------------------------------------
    # Publication-style plots
    # ----------------------------------------------------------------------
    if make_plots:
        # reshape helpers
        psi_ex_3 = psi_exact.reshape(nx, ny, nz)
        psi_num_3 = psi_num.reshape(nx, ny, nz)
        inside3 = inside_flat.reshape(nx, ny, nz)

        # 1) Error histogram (relative error, log y-scale)
        fig_hist, axh = plt.subplots(figsize=(3.2, 2.6))
        axh.hist(rel_err_int, bins=80)
        axh.set_yscale("log")
        axh.set_xlabel(r"$|\psi_{\rm num}-\psi_{\rm exact}|/|\psi_{\rm exact}|$")
        axh.set_ylabel("count")
        axh.set_title(
            r"Error histogram (interior)" + "\n" +
            rf"median={np.median(rel_err_int):.2e}, 95%={pct(rel_err_int,95):.2e}",
            fontsize=9
        )
        fig_hist.tight_layout()
        fig_hist.savefig(f"{save_prefix}_error_hist.png", dpi=300)

        # 2) 1D midplane profile: y=0, z≈0
        j_mid = ny // 2
        k_mid = nz // 2

        x_line = xs
        psi_line_exact = psi_ex_3[:, j_mid, k_mid]
        psi_line_num = psi_num_3[:, j_mid, k_mid]

        fig_line, axs = plt.subplots(
            1, 2, figsize=(6.0, 2.6), constrained_layout=True
        )

        ax0 = axs[0]
        ax0.plot(x_line, psi_line_exact, "k-", lw=1.4, label=r"$\psi_{\rm exact}$")
        ax0.plot(x_line, psi_line_num, "r--", lw=1.2, label=r"$\psi_{\rm num}$")
        ax0.set_xlabel(r"$x$ (midplane: $y=0$, $z\approx0$)")
        ax0.set_ylabel(r"$\psi$")
        ax0.set_title("Midplane lineout", fontsize=10)
        ax0.legend(frameon=False, fontsize=8)

        ax1 = axs[1]
        denom_line = np.maximum(np.abs(psi_line_exact), 1e-12)
        rel_err_line = np.abs(psi_line_num - psi_line_exact) / denom_line
        ax1.plot(x_line, rel_err_line, "b-", lw=1.2)
        ax1.set_yscale("log")
        ax1.set_xlabel(r"$x$ (midplane: $y=0$, $z\approx0$)")
        ax1.set_ylabel(r"$|\psi_{\rm num}-\psi_{\rm exact}|/|\psi_{\rm exact}|$")
        ax1.set_title("Relative error (log scale)", fontsize=10)

        fig_line.savefig(f"{save_prefix}_midplane_profile.png", dpi=300)

        # 3) Midplane slices x-z at y≈0
        fig_slice, axes = plt.subplots(
            1, 3, figsize=(7.2, 2.4), constrained_layout=True
        )

        inside_mid = inside3[:, j_mid, :]
        psi_exact_mid = np.ma.masked_where(~inside_mid, psi_ex_3[:, j_mid, :])
        psi_num_mid = np.ma.masked_where(~inside_mid, psi_num_3[:, j_mid, :])

        denom_mid = np.maximum(np.abs(psi_exact_mid), 1e-12)
        rel_err_mid = np.ma.masked_where(
            ~inside_mid,
            np.abs(psi_num_mid - psi_exact_mid) / denom_mid
        )

        vmin_psi = psi_exact_mid.min()
        vmax_psi = psi_exact_mid.max()

        im0 = axes[0].pcolormesh(xs, zs, psi_exact_mid.T, shading="auto")
        im0.set_clim(vmin_psi, vmax_psi)
        axes[0].set_title(r"$\psi_{\rm exact}$ (midplane)", fontsize=9)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("z")
        axes[0].set_aspect("equal", "box")
        c0 = fig_slice.colorbar(im0, ax=axes[0], shrink=0.9)
        c0.set_label(r"$\psi_{\rm exact}$", fontsize=8)

        im1 = axes[1].pcolormesh(xs, zs, psi_num_mid.T, shading="auto")
        im1.set_clim(vmin_psi, vmax_psi)
        axes[1].set_title(r"$\psi_{\rm num}$ (midplane)", fontsize=9)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("z")
        axes[1].set_aspect("equal", "box")
        c1 = fig_slice.colorbar(im1, ax=axes[1], shrink=0.9)
        c1.set_label(r"$\psi_{\rm num}$", fontsize=8)

        im2 = axes[2].pcolormesh(xs, zs, rel_err_mid.T, shading="auto")
        axes[2].set_title("Relative error (midplane)", fontsize=9)
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("z")
        axes[2].set_aspect("equal", "box")
        c2 = fig_slice.colorbar(im2, ax=axes[2], shrink=0.9)
        c2.set_label(
            r"$|\psi_{\rm num}-\psi_{\rm exact}|/|\psi_{\rm exact}|$",
            fontsize=8
        )

        fig_slice.savefig(f"{save_prefix}_midplane_slices.png", dpi=300)

        # 4) 3D boundary plots: |∇φ|, ψ_num, relative error
        P = Xq[boundary_band_flat]
        P_j = jnp.asarray(P, dtype=jnp.float64)
        B_boundary = np.asarray(grad_phi_toroidal(P_j))
        Bnorm = np.linalg.norm(B_boundary, axis=1)

        psi_bnd_exact = psi_exact[boundary_band_flat]
        psi_bnd_num = psi_num[boundary_band_flat]
        denom_bnd = np.maximum(np.abs(psi_bnd_exact), 1e-12)
        rel_err_bnd = np.abs(psi_bnd_num - psi_bnd_exact) / denom_bnd

        fig3d = plt.figure(figsize=(8.0, 2.6))
        titles = [r"$|\nabla\phi|$ on boundary",
                  r"$\psi_{\rm num}$ on boundary",
                  r"Relative error on boundary"]
        data = [Bnorm, psi_bnd_num, rel_err_bnd]
        cb_labels = [r"$|\nabla\phi|$",
                     r"$\psi_{\rm num}$",
                     r"$|\psi_{\rm num}-\psi_{\rm exact}|/|\psi_{\rm exact}|$"]

        for k in range(3):
            ax = fig3d.add_subplot(1, 3, k+1, projection="3d")
            sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2],
                            c=data[k], s=2, linewidths=0)
            ax.set_title(titles[k], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1, 1, 0.6))
            cb = fig3d.colorbar(sc, ax=ax, shrink=0.8, pad=0.03)
            cb.set_label(cb_labels[k], fontsize=8)

        fig3d.tight_layout()
        fig3d.savefig(f"{save_prefix}_boundary_3d.png", dpi=300)

    return rel_L2, Linf_rel


# ---------------------------------------------------------------------------
# Convergence sweep
# ---------------------------------------------------------------------------

def convergence_sweep(
    N_list: List[int],
    R0: float = 1.5,
    a: float = 0.5,
    kappa_par: float = 1.0,
    kappa_perp: float = 0.05,
    h_band_factor: float = 2.0,
    axis_frac: float = 0.1,
    nfp: int = 2,
    fci_nsteps: int = 16,
    fci_planes_per_field_period: int = 8,
    save_prefix: str = "fci_torus_mms",
):
    """
    Sweep over grid resolutions N, run the MMS, and plot convergence
    versus grid spacing h ~ 1/N.
    """
    rows: List[Tuple[int, float, float]] = []

    for N in N_list:
        relL2, Linf_rel = solve_fci_torus_mms(
            N=N,
            R0=R0,
            a=a,
            kappa_par=kappa_par,
            kappa_perp=kappa_perp,
            h_band_factor=h_band_factor,
            axis_frac=axis_frac,
            nfp=nfp,
            fci_nsteps=fci_nsteps,
            fci_planes_per_field_period=fci_planes_per_field_period,
            make_plots=False,
            save_prefix=f"{save_prefix}_N{N}",
        )
        rows.append((N, relL2, Linf_rel))

    print("\n=== FCI torus MMS convergence summary ===")
    print("  N    relL2          max_rel")
    print("--------------------------------")
    for N, rL2, Lr in rows:
        print(f"{N:4d}  {rL2:.3e}  {Lr:.3e}")

    Ns = np.array([r[0] for r in rows], dtype=float)
    h = 1.0 / Ns
    err_L2 = np.array([r[1] for r in rows])
    err_max = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.loglog(h, err_L2, "o-", label=r"relative $L^2$")
    ax.loglog(h, err_max, "s--", label=r"max relative")

    # Fit slopes in log-log space to estimate order
    coeff_L2 = np.polyfit(np.log(h), np.log(err_L2), 1)
    coeff_max = np.polyfit(np.log(h), np.log(err_max), 1)
    p_L2 = coeff_L2[0]
    p_max = coeff_max[0]

    ax.loglog(h, np.exp(coeff_L2[1]) * h**p_L2, "k:", lw=1.0,
              label=fr"fit $p_{{L^2}}\approx {p_L2:.2f}$")
    ax.loglog(h, np.exp(coeff_max[1]) * h**p_max, "k-.", lw=1.0,
              label=fr"fit $p_{{\infty}}\approx {p_max:.2f}$")

    ax.set_xlabel(r"grid spacing $h\sim 1/N$")
    ax.set_ylabel("relative error")
    ax.set_title("FCI torus MMS convergence", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_convergence.png", dpi=300)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full-structure FCI torus manufactured-solution test."
    )
    parser.add_argument("--N", type=int, default=64,
                        help="Grid resolution per axis")
    parser.add_argument("--R0", type=float, default=1.5,
                        help="Torus major radius")
    parser.add_argument("--a", type=float, default=0.5,
                        help="Torus minor radius")
    parser.add_argument("--kappa-par", type=float, default=1.0,
                        help="Parallel diffusivity")
    parser.add_argument("--kappa-perp", type=float, default=0.05,
                        help="Perpendicular diffusivity")
    parser.add_argument("--h-band-factor", type=float, default=2.0,
                        help="Thickness of boundary band (in voxel units)")
    parser.add_argument("--axis-frac", type=float, default=0.1,
                        help="Axis band radius fraction of minor radius")
    parser.add_argument("--nfp", type=int, default=2,
                        help="Field periods for FCI")
    parser.add_argument("--fci-nsteps", type=int, default=16,
                        help="FCI steps per field period")
    parser.add_argument("--fci-planes-per-field-period", type=int, default=8,
                        help="FCI planes per field period")
    parser.add_argument("--no-sweep", dest="sweep", action="store_false",
                        help="Do not run convergence sweep")
    parser.add_argument("--Ns", type=str, default="24,36,48,60,72,84",
                        help="Comma-separated N list for convergence sweep")
    args = parser.parse_args()

    if args.sweep:
        N_list = [int(s) for s in args.Ns.split(",")]
        convergence_sweep(
            N_list=N_list,
            R0=args.R0,
            a=args.a,
            kappa_par=args.kappa_par,
            kappa_perp=args.kappa_perp,
            h_band_factor=args.h_band_factor,
            axis_frac=args.axis_frac,
            nfp=args.nfp,
            fci_nsteps=args.fci_nsteps,
            fci_planes_per_field_period=args.fci_planes_per_field_period,
            save_prefix="fci_torus_mms",
        )
    else:
        solve_fci_torus_mms(
            N=args.N,
            R0=args.R0,
            a=args.a,
            kappa_par=args.kappa_par,
            kappa_perp=args.kappa_perp,
            h_band_factor=args.h_band_factor,
            axis_frac=args.axis_frac,
            nfp=args.nfp,
            fci_nsteps=args.fci_nsteps,
            fci_planes_per_field_period=args.fci_planes_per_field_period,
            make_plots=True,
            save_prefix="fci_torus_mms",
        )
