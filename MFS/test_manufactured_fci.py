#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manufactured-solution test for the FCI operator.

We solve  A[ψ] = 0  with A built by make_fci_operator_jax using connectivity
from build_fci_connectivity_chunked, and compare against the exact solution

    ψ_exact(x,y,z) = z

on a simple box domain.  The "magnetic field" used by FCI is a pure toroidal
unit vector B = e_phi, i.e.

    B(x,y,z) = (-sinφ, cosφ, 0),  φ = atan2(y,x)

so field lines are circles at constant R and Z.

For this choice:

  * B·∇ψ_exact = 0, so the parallel operator should see ψ constant along
    field lines, and thus the discrete parallel second derivative vanishes.

  * ∇²ψ_exact = 0, so ψ is harmonic in the box; the perpendicular 7-point
    Laplacian is also exactly zero for a linear function in z.

Hence the continuous and discrete PDE are both satisfied, and we expect
errors at roundoff level if the FCI machinery is implemented correctly.
"""

import argparse
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from solve_flux_psi_fci import (
    build_fci_connectivity_chunked,
    make_fci_operator_jax,
    cg_jax,
)

# ----------------------------------------------------------------------
# Manufactured exact solution
# ----------------------------------------------------------------------

def psi_exact_fn(x, y, z):
    return z   # simple linear function in z


# ----------------------------------------------------------------------
# Grid / geometry helpers
# ----------------------------------------------------------------------

def build_cartesian_grid(bounds, N):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    xs = np.linspace(xmin, xmax, N)
    ys = np.linspace(ymin, ymax, N)
    zs = np.linspace(zmin, zmax, N)

    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)

    Xq = np.column_stack([
        XX.ravel(order="C"),
        YY.ravel(order="C"),
        ZZ.ravel(order="C"),
    ])
    return xs, ys, zs, dx, dy, dz, Xq


def inside_box(xs, ys, zs):
    nx, ny, nz = len(xs), len(ys), len(zs)
    return np.ones((nx, ny, nz), dtype=bool)


def compute_discrete_boundary(inside3):
    """
    Same idea as in the tensor test: a node is "boundary" if
      - it lies on the outer box faces, OR
      - it has a 6-connected neighbour outside the domain.
    """
    nx, ny, nz = inside3.shape
    boundary3 = np.zeros_like(inside3, dtype=bool)

    # interior cells that see an outside neighbour
    core = inside3[1:-1, 1:-1, 1:-1]
    neigh_out = (
        ~inside3[0:-2, 1:-1, 1:-1] |
        ~inside3[2:  , 1:-1, 1:-1] |
        ~inside3[1:-1, 0:-2, 1:-1] |
        ~inside3[1:-1, 2:  , 1:-1] |
        ~inside3[1:-1, 1:-1, 0:-2] |
        ~inside3[1:-1, 1:-1, 2:  ]
    )
    boundary3[1:-1, 1:-1, 1:-1] = core & neigh_out

    # outer box faces
    boundary3[0,   :,   :] |= inside3[0,   :,   :]
    boundary3[nx-1, :,   :] |= inside3[nx-1, :,   :]
    boundary3[:,   0,   :] |= inside3[:,   0,   :]
    boundary3[:,   ny-1, :] |= inside3[:,   ny-1, :]
    boundary3[:,   :,   0] |= inside3[:,   :,   0]
    boundary3[:,   :, nz-1] |= inside3[:,   :, nz-1]

    inside_flat   = inside3.ravel(order="C")
    boundary_flat = boundary3.ravel(order="C")
    return inside_flat, boundary_flat


# ----------------------------------------------------------------------
# Analytic "B field" used by FCI (really just the direction field)
# ----------------------------------------------------------------------

def grad_phi_toroidal(X: jnp.ndarray) -> jnp.ndarray:
    """
    Return a pure toroidal unit vector B = e_phi at each point X.

    X: (..., 3)
    """
    x = X[..., 0]
    y = X[..., 1]
    z = X[..., 2]

    r = jnp.sqrt(x * x + y * y)
    # avoid r=0 singularity: set B=0 there (parallel operator will be disabled)
    ephi = jnp.stack([-y, x, 0.0 * z], axis=-1)
    ephi_norm = jnp.maximum(r, 1e-8)
    B = ephi / ephi_norm[..., None]
    B = jnp.where(r[..., None] > 1e-8, B, jnp.zeros_like(B))
    return B


# ----------------------------------------------------------------------
# Core manufactured solve using the FCI operator
# ----------------------------------------------------------------------

def solve_fci_manufactured(
    N=40,
    kappa_par=1.0,
    kappa_perp=1e-1,
    nfp=1,
    fci_nsteps=32,
):
    """
    Build FCI connectivity + operator on a box and solve A[ψ]=0 with ψ=z
    on the discrete boundary.  Returns (rel_L2, Linf) interior errors.
    """
    # Box domain slightly away from origin to keep r>0 for most nodes
    bounds = ((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5))
    xs, ys, zs, dx, dy, dz, Xq = build_cartesian_grid(bounds, N)
    nx, ny, nz = len(xs), len(ys), len(zs)
    Ntot = nx * ny * nz

    inside3 = inside_box(xs, ys, zs)
    inside_flat, boundary_flat = compute_discrete_boundary(inside3)

    # Build FCI connectivity (this internally uses JAX field-line tracing etc.)
    fci = build_fci_connectivity_chunked(
        xs, ys, zs,
        inside_mask=inside_flat,
        grad_phi_fn=grad_phi_toroidal,
        nfp=nfp,
        dphi_per_step=None,
        nsteps=fci_nsteps,
        verbose=True,
        chunk_size=None,
    )
    print(f"[FCI TEST] valid connectivity nodes: {fci.valid.sum()} / {inside_flat.sum()}")

    # Define core region: inside, at least one cell away from the box faces
    inside3 = inside_flat.reshape(nx, ny, nz)
    core3 = np.zeros_like(inside3, dtype=bool)
    core3[1:-1, 1:-1, 1:-1] = True
    core3 &= inside3
    core_flat = core3.ravel(order="C")

    # Build FCI operator
    A_pde_jax, deep_inside = make_fci_operator_jax(
        nx, ny, nz,
        xs, ys, zs,
        inside_flat,
        fci,
        core_mask=core_flat,
        kappa_par=kappa_par,
        kappa_perp=kappa_perp,
    )

    # Manufactured exact solution on all nodes
    psi_exact = psi_exact_fn(Xq[:, 0], Xq[:, 1], Xq[:, 2])

    # Dirichlet BC on boundary nodes
    fixed = boundary_flat.copy()
    val = np.zeros(Ntot, dtype=float)
    val[fixed] = psi_exact[fixed]

    free = deep_inside & (~fixed)
    if not np.any(free):
        raise RuntimeError("No free nodes in FCI manufactured test.")

    print(f"[FCI TEST] free nodes: {free.sum()} / {Ntot}")

    # Lifting: ψ = ψ_free + ψ_fixed
    psi_fixed_full = np.zeros(Ntot, dtype=float)
    psi_fixed_full[fixed] = val[fixed]

    F0_full = np.array(A_pde_jax(jnp.asarray(psi_fixed_full)))
    b_free = -F0_full[free]

    free_mask = free.copy()

    def matvec_free_jax(u_free_j):
        u_full = jnp.zeros(Ntot, dtype=jnp.float64)
        u_full = u_full.at[free_mask].set(u_free_j)
        Au_full = A_pde_jax(u_full)
        return Au_full[free_mask]

    print("[FCI TEST] Solving (JAX CG) ...")
    b_free_j = jnp.asarray(b_free, dtype=jnp.float64)
    psi_free_j, res_norm = cg_jax(matvec_free_jax, b_free_j, tol=1e-10, maxiter=5000)
    psi_free = np.asarray(psi_free_j)
    print(f"[FCI TEST] CG finished with ||r||₂ ≈ {float(res_norm):.3e}")

    psi_num = np.array(psi_fixed_full)
    psi_num[free] = psi_free

    # Error metrics in the interior (inside & not boundary)
    interior = inside_flat & (~boundary_flat)
    err = psi_num - psi_exact
    rel_L2 = np.linalg.norm(err[interior]) / np.linalg.norm(psi_exact[interior])
    Linf = np.max(np.abs(err[interior]))
    print(f"[FCI TEST] relative L2 error (interior) = {rel_L2:.3e}")
    print(f"[FCI TEST] Linf error (interior)        = {Linf:.3e}")
    return rel_L2, Linf


def convergence_sweep_fci(N_list=(16, 24, 32, 40), kappa_par=1.0, kappa_perp=1e-6):
    rows = []
    for N in N_list:
        print("\n" + "=" * 50)
        print(f"FCI manufactured test at N = {N}")
        print("=" * 50)
        relL2, Linf = solve_fci_manufactured(N=N, kappa_par=kappa_par, kappa_perp=kappa_perp)
        rows.append((N, relL2, Linf))

    print("\n=== FCI convergence summary (relative L2, Linf) ===")
    print(" N    relL2         Linf")
    print("--------------------------------")
    for N, relL2, Linf in rows:
        print(f"{N:3d}  {relL2:.3e}  {Linf:.3e}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manufactured-solution test for FCI operator.")
    parser.add_argument("--N", type=int, default=40, help="Grid resolution per axis")
    parser.add_argument("--no-sweep", dest="sweep", action="store_false", help="Run a convergence sweep over N.")
    parser.add_argument("--Ns", type=str, default="16,24,32,40",
                        help="Comma-separated list of N values for the FCI sweep.")
    parser.add_argument("--kappa-par", type=float, default=1.0,
                        help="Parallel conductivity coefficient (default: 1.0)")
    parser.add_argument("--kappa-perp", type=float, default=1e-1,
                        help="Perpendicular conductivity coefficient")
    args = parser.parse_args()

    if args.sweep:
        N_list = [int(s) for s in args.Ns.split(",")]
        convergence_sweep_fci(N_list=N_list,
                              kappa_par=args.kappa_par,
                              kappa_perp=args.kappa_perp)
    else:
        solve_fci_manufactured(N=args.N,
                               kappa_par=args.kappa_par,
                               kappa_perp=args.kappa_perp)
