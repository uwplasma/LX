#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manufactured-solution test for the anisotropic diffusion operator
used in solve_flux_psi_fci.py on a simple Cartesian box.

We test two cases for the PDE

    -div( D ∇ψ ) = 0  in Ω = [0,1]^3
    ψ = ψ_exact on ∂Ω,

with the *same* harmonic exact solution

    ψ_exact(x,y,z) = x + 0.3*y + 0.1*z,

for which div(D ∇ψ_exact) = 0 for any constant symmetric D.

Cases:
  1. Isotropic: D = I
  2. Anisotropic: D = diffusion_tensor_jax(G=(0,0,1), eps, delta=0) → diag(eps, eps, 1).

We solve using the same JAX operator and CG as in solve_flux_psi_fci.py
and compare numerical vs exact solution.
"""

import argparse

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Import the core pieces from your solver
from solve_flux_psi_fci import make_linear_operator_jax, diffusion_tensor_jax, cg_jax


def build_box_grid(N: int):
    """Build [0,1]^3 grid and return xs, ys, zs, dx, dy, dz, Xq (N^3,3)."""
    xs = np.linspace(0.0, 1.0, N)
    ys = np.linspace(0.0, 1.0, N)
    zs = np.linspace(0.0, 1.0, N)
    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]

    # These must match the flattening convention used in make_linear_operator_jax
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)
    Xq = np.column_stack([XX.ravel(order="C"),
                          YY.ravel(order="C"),
                          ZZ.ravel(order="C")])
    return xs, ys, zs, dx, dy, dz, Xq


def psi_exact_fn(x, y, z):
    """Manufactured harmonic solution."""
    return x + 0.3*y + 0.1*z


def build_boundary_mask(nx, ny, nz):
    """Boolean mask of boundary nodes on the full grid."""
    inside3 = np.ones((nx, ny, nz), dtype=bool)
    boundary3 = np.zeros_like(inside3, dtype=bool)

    boundary3[0, :, :] = True
    boundary3[-1, :, :] = True
    boundary3[:, 0, :] = True
    boundary3[:, -1, :] = True
    boundary3[:, :, 0] = True
    boundary3[:, :, -1] = True

    boundary_flat = boundary3.ravel(order="C")
    inside_flat = inside3.ravel(order="C")
    return inside_flat, boundary_flat


def solve_manufactured(
    Dfield: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    Xq: np.ndarray,
    tol: float = 1e-10,
    maxiter: int = 5000,
):
    """
    Solve -div(D∇ψ)=0 in the box with Dirichlet BC given by ψ_exact.

    Dfield: (N,3,3) numpy array, constant or spatially varying.
    """
    nx, ny, nz = len(xs), len(ys), len(zs)
    Ntot = nx * ny * nz

    # Full domain is "inside"
    inside_flat, boundary_flat = build_boundary_mask(nx, ny, nz)

    # Build JAX operator
    A_pde_jax, deep_inside = make_linear_operator_jax(
        nx, ny, nz,
        dx, dy, dz,
        inside_flat,
        Dfield,
    )

    # Manufactured exact solution on the grid
    psi_exact = psi_exact_fn(Xq[:, 0], Xq[:, 1], Xq[:, 2])

    # Dirichlet BC: ψ = ψ_exact on the boundary
    fixed = boundary_flat.copy()
    val = np.zeros(Ntot, dtype=float)
    val[fixed] = psi_exact[fixed]

    # Free nodes are the "deep inside" ones that are not fixed
    free = deep_inside & (~fixed)
    if not np.any(free):
        raise RuntimeError("No free nodes in manufactured test (grid too small?).")

    # Lifting: ψ = ψ_free + ψ_fixed
    psi_fixed_full = np.zeros(Ntot, dtype=float)
    psi_fixed_full[fixed] = val[fixed]

    # Compute A ψ_fixed
    F0_full = np.array(A_pde_jax(jnp.asarray(psi_fixed_full)))
    b_free = -F0_full[free]

    Nfree = int(free.sum())
    print(f"[TEST] free nodes: {Nfree} / {Ntot}")

    # JAX-CG on the reduced system
    free_mask = free  # just a clearer name

    def matvec_free_jax(u_free_j):
        u_full = jnp.zeros(Ntot, dtype=jnp.float64)
        u_full = u_full.at[free_mask].set(u_free_j)
        Au_full = A_pde_jax(u_full)
        return Au_full[free_mask]

    print("[TEST] Solving linear system (JAX CG) ...")
    b_free_j = jnp.asarray(b_free, dtype=jnp.float64)
    psi_free_j, res_norm = cg_jax(matvec_free_jax, b_free_j, tol=tol, maxiter=maxiter)
    psi_free = np.asarray(psi_free_j)
    print(f"[TEST] CG finished with final residual norm ||r||₂ ≈ {float(res_norm):.3e}")

    # Assemble full solution
    psi_num = np.array(psi_fixed_full)
    psi_num[free] = psi_free

    return psi_num, psi_exact, inside_flat, boundary_flat


def run_tests(N: int = 32, eps_aniso: float = 1e-3):
    # Build box grid
    xs, ys, zs, dx, dy, dz, Xq = build_box_grid(N)
    nx, ny, nz = len(xs), len(ys), len(zs)
    Ntot = nx * ny * nz

    # === Case 1: isotropic D = I ===
    print("\n=== Manufactured test: isotropic D = I ===")
    D_iso = np.zeros((Ntot, 3, 3), dtype=float)
    for i in range(3):
        D_iso[:, i, i] = 1.0

    psi_iso, psi_exact, inside_flat, boundary_flat = solve_manufactured(
        D_iso, xs, ys, zs, dx, dy, dz, Xq
    )

    # Error metrics (exclude boundary to focus on interior)
    interior = inside_flat & (~boundary_flat)
    err_iso = psi_iso - psi_exact
    L2_iso = np.linalg.norm(err_iso[interior]) / np.linalg.norm(psi_exact[interior])
    Linf_iso = np.max(np.abs(err_iso[interior]))
    print(f"[ISO] relative L2 error (interior) = {L2_iso:.3e}")
    print(f"[ISO] Linf error (interior)        = {Linf_iso:.3e}")

    # === Case 2: anisotropic D from diffusion_tensor_jax(G=(0,0,1)) ===
    print("\n=== Manufactured test: anisotropic D from diffusion_tensor_jax ===")
    G_const = np.tile(np.array([0.0, 0.0, 1.0]), (Ntot, 1))
    D_aniso_j = diffusion_tensor_jax(jnp.asarray(G_const), eps=eps_aniso, delta=0.0)
    D_aniso = np.asarray(D_aniso_j)

    psi_aniso, _, _, _ = solve_manufactured(
        D_aniso, xs, ys, zs, dx, dy, dz, Xq
    )

    err_aniso = psi_aniso - psi_exact
    L2_aniso = np.linalg.norm(err_aniso[interior]) / np.linalg.norm(psi_exact[interior])
    Linf_aniso = np.max(np.abs(err_aniso[interior]))
    print(f"[ANISO] relative L2 error (interior) = {L2_aniso:.3e}")
    print(f"[ANISO] Linf error (interior)        = {Linf_aniso:.3e}")

    # === Plot slices ===
    psi_exact_3 = psi_exact.reshape(nx, ny, nz, order="C")
    psi_iso_3   = psi_iso.reshape(nx, ny, nz, order="C")
    psi_aniso_3 = psi_aniso.reshape(nx, ny, nz, order="C")

    k_mid = nz // 2  # mid-plane in z

    # Isotropic figure
    fig_iso, axs_iso = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    im0 = axs_iso[0].imshow(psi_exact_3[:, :, k_mid].T, origin="lower",
                            extent=extent, aspect="equal")
    axs_iso[0].set_title("Exact ψ (z mid-plane)")
    axs_iso[0].set_xlabel("x"); axs_iso[0].set_ylabel("y")
    fig_iso.colorbar(im0, ax=axs_iso[0], shrink=0.8)

    im1 = axs_iso[1].imshow(psi_iso_3[:, :, k_mid].T, origin="lower",
                            extent=extent, aspect="equal")
    axs_iso[1].set_title("Numerical ψ (isotropic)")
    axs_iso[1].set_xlabel("x"); axs_iso[1].set_ylabel("y")
    fig_iso.colorbar(im1, ax=axs_iso[1], shrink=0.8)

    slice_exact = psi_exact_3[:, :, k_mid]
    slice_err_iso = psi_iso_3[:, :, k_mid] - slice_exact
    rel_err_iso = np.zeros_like(slice_err_iso)
    mask_nonzero = np.abs(slice_exact) > 1e-14
    rel_err_iso[mask_nonzero] = slice_err_iso[mask_nonzero] / slice_exact[mask_nonzero]

    im2 = axs_iso[2].imshow(rel_err_iso.T, origin="lower",
                            extent=extent, aspect="equal")
    axs_iso[2].set_title("Relative error (isotropic)")
    axs_iso[2].set_xlabel("x"); axs_iso[2].set_ylabel("y")
    fig_iso.colorbar(im2, ax=axs_iso[2], shrink=0.8)

    fig_iso.suptitle(f"Manufactured solution test (D=I, N={N})")

    # Anisotropic figure
    fig_an, axs_an = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0a = axs_an[0].imshow(psi_exact_3[:, :, k_mid].T, origin="lower",
                            extent=extent, aspect="equal")
    axs_an[0].set_title("Exact ψ (z mid-plane)")
    axs_an[0].set_xlabel("x"); axs_an[0].set_ylabel("y")
    fig_an.colorbar(im0a, ax=axs_an[0], shrink=0.8)

    im1a = axs_an[1].imshow(psi_aniso_3[:, :, k_mid].T, origin="lower",
                            extent=extent, aspect="equal")
    axs_an[1].set_title(f"Numerical ψ (anisotropic, eps={eps_aniso})")
    axs_an[1].set_xlabel("x"); axs_an[1].set_ylabel("y")
    fig_an.colorbar(im1a, ax=axs_an[1], shrink=0.8)

    slice_err_an = psi_aniso_3[:, :, k_mid] - slice_exact
    rel_err_an = np.zeros_like(slice_err_an)
    rel_err_an[mask_nonzero] = slice_err_an[mask_nonzero] / slice_exact[mask_nonzero]

    im2a = axs_an[2].imshow(rel_err_an.T, origin="lower",
                            extent=extent, aspect="equal")
    axs_an[2].set_title("Relative error (anisotropic)")
    axs_an[2].set_xlabel("x"); axs_an[2].set_ylabel("y")
    fig_an.colorbar(im2a, ax=axs_an[2], shrink=0.8)

    fig_an.suptitle(f"Manufactured solution test (anisotropic, eps={eps_aniso}, N={N})")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manufactured-solution box test for the anisotropic diffusion operator."
    )
    parser.add_argument("--N", type=int, default=32,
                        help="Grid resolution per axis (default: 32)")
    parser.add_argument("--eps", type=float, default=1e-3,
                        help="Anisotropy parameter eps for diffusion_tensor_jax (default: 1e-3)")
    args = parser.parse_args()

    run_tests(N=args.N, eps_aniso=args.eps)
