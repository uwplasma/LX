#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discrete manufactured-solution test for the FCI flux solver.

Here we test the *discrete* FCI operator A_FCI from solve_flux_psi_fci.py.

We:
  * build a circular torus domain in Cartesian coordinates,
  * define a smooth manufactured solution ψ_exact(x,y,z),
  * compute the discrete RHS f_disc = A_FCI ψ_exact (using the same operator
    that is used in the solver),
  * impose Dirichlet data ψ = ψ_exact on:
      - a thin outer boundary band near the torus surface, and
      - a small inner "axis band" near ρ = 0,
  * solve A_FCI ψ_num = f_disc with those Dirichlet data,
  * compare ψ_num to ψ_exact on the free interior nodes.

This test directly checks:
  - FCI connectivity (field line tracing, interpolation),
  - parallel stencil construction,
  - perpendicular Laplacian,
  - lifting / boundary treatment,
  - JAX CG solver.

Unlike a continuum MMS, we do *not* use Hessians or the continuous operator:
we explicitly test the discrete operator A_FCI against itself, which is the
right way to validate a complicated FCI stencil.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, gmres, bicgstab

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Import the FCI machinery from your solver
from solve_flux_psi_fci import (
    build_fci_connectivity_chunked,
    make_fci_operator_jax,
    cg_jax,
    pinfo,
)


# -----------------------------------------------------------------------------
# Geometry and manufactured solution
# -----------------------------------------------------------------------------

@dataclass
class TorusParams:
    R0: float = 1.5   # major radius
    a: float = 0.5    # minor radius
    m: int = 1        # toroidal mode in φ


def torus_signed_distance(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray,
                          params: TorusParams) -> jnp.ndarray:
    """
    Signed distance to a circular torus of major radius R0 and minor radius a.

      (sqrt(x^2 + y^2) - R0)^2 + z^2 = a^2

    Negative inside, positive outside.
    """
    R = jnp.sqrt(x * x + y * y)
    rho = jnp.sqrt((R - params.R0) ** 2 + z * z)
    return rho - params.a


def rho_from_xyz(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray,
                 params: TorusParams) -> jnp.ndarray:
    """Distance ρ from the torus magnetic axis."""
    R = jnp.sqrt(x * x + y * y)
    return jnp.sqrt((R - params.R0) ** 2 + z * z)


def psi_exact_point(xyz: jnp.ndarray, params: TorusParams) -> jnp.ndarray:
    """
    Smooth manufactured ψ_exact that varies in both radius and toroidal angle.

    We use:

        ψ_exact(ρ, φ) = (ρ/a)^2 * (1 - (ρ/a)^2) * cos(m φ),

    where ρ is the distance from the torus magnetic axis, and φ = atan2(y,x).
    This vanishes smoothly at ρ=0 and ρ=a, and has nontrivial φ variation
    to exercise the parallel FCI stencil.
    """
    x, y, z = xyz
    R = jnp.sqrt(x * x + y * y)
    phi = jnp.arctan2(y, x)
    rho = rho_from_xyz(x, y, z, params)
    s = rho / params.a
    radial = s * s * (1.0 - s * s)
    return radial * jnp.cos(params.m * phi)
    # return 1.0


# Convenience JAX vectorized ψ_exact
params_global = TorusParams()
psi_exact_point_jax = lambda xyz: psi_exact_point(xyz, params_global)
psi_exact_vmap = jax.jit(jax.vmap(psi_exact_point_jax, in_axes=(0,)))


def B_field_xyz(xyz: jnp.ndarray) -> jnp.ndarray:
    """
    Magnetic field B(x) used by the FCI operator.

    We pick a simple toroidal field with unit magnitude:

        B = e_phi = (-sin φ, cos φ, 0),

    so |B| = 1, and field lines are circles at fixed R and z.
    """
    x, y, z = xyz
    phi = jnp.arctan2(y, x)
    bx = -jnp.sin(phi)
    by =  jnp.cos(phi)
    bz =  0.0 * z
    return jnp.stack([bx, by, bz])


B_field_vmap = jax.jit(jax.vmap(B_field_xyz, in_axes=(0,)))


# -----------------------------------------------------------------------------
# Grid and masks
# -----------------------------------------------------------------------------

def build_torus_grid_and_masks(
    N: int,
    params: TorusParams,
    band_width_factor: float = 1.5,
    axis_width_factor: float = 0.20,
):
    """
    Build a Cartesian grid and logical masks for a circular torus:

        - inside_flat: torus interior
        - band_flat: thin boundary band near ρ ≈ a
        - axis_flat: small inner band near ρ ≈ 0

    All masks are length Ntot and flatten in C-order with (nx,ny,nz).
    """
    R0, a = params.R0, params.a

    # A box comfortably containing the torus
    R_max = R0 + 1.1 * a
    x_min, x_max = -R_max, R_max
    y_min, y_max = -R_max, R_max
    z_min, z_max = -1.1 * a, 1.1 * a

    nx = ny = N
    nz = max(16, N // 2)   # fewer points in z-direction

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    zs = np.linspace(z_min, z_max, nz)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]
    voxel = min(dx, dy, dz)

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)
    Xq = np.column_stack([XX.ravel(order="C"),
                          YY.ravel(order="C"),
                          ZZ.ravel(order="C")])
    Ntot = Xq.shape[0]

    Xj = jnp.asarray(Xq, dtype=jnp.float64)
    sd = jax.vmap(torus_signed_distance, in_axes=(0, 0, 0, None))(
        Xj[:, 0], Xj[:, 1], Xj[:, 2], params
    )
    signed_dist = np.asarray(sd, dtype=float)

    rho = jax.vmap(rho_from_xyz, in_axes=(0, 0, 0, None))(
        Xj[:, 0], Xj[:, 1], Xj[:, 2], params
    )
    rho_all = np.asarray(rho, dtype=float)

    inside_flat = signed_dist < 0.0

    # Outer boundary band: |distance| <= band_width
    band_width = band_width_factor * voxel
    band_flat = np.abs(signed_dist) <= band_width

    # Inner axis band: ρ ≤ axis_radius
    axis_radius = axis_width_factor * params.a
    axis_flat = (rho_all <= axis_radius) & inside_flat

    # Disambiguate any overlaps
    overlap = band_flat & axis_flat
    axis_flat[overlap] = True
    band_flat[overlap] = False

    pinfo(
        f"[GRID] N={N}: nx={nx}, ny={ny}, nz={nz}, Ntot={Ntot}, "
        f"dx≈{dx:.3e}, dy≈{dy:.3e}, dz≈{dz:.3e}"
    )
    print(f"[GRID] inside nodes        : {inside_flat.sum()} / {Ntot}")
    print(f"[GRID] boundary band nodes : {band_flat.sum()} / {Ntot}")
    print(f"[GRID] axis band nodes     : {axis_flat.sum()} / {Ntot}")
    print(f"[GRID] band width ≈ {band_width:.3e}, axis radius ≈ {axis_radius:.3e}")

    return xs, ys, zs, Xq, inside_flat, band_flat, axis_flat


# -----------------------------------------------------------------------------
# Single-resolution discrete MMS
# -----------------------------------------------------------------------------

@dataclass
class DiscreteMMSResult:
    N: int
    L2_rel: float
    Linf_rel: float
    cg_res: float


def run_discrete_mms_single(
    N: int,
    params: TorusParams,
    eps: float = 0.05,
    band_factor: float = 1.5,
    axis_factor: float = 0.2,
    nfp: int = 2,
    # IMPORTANT: refine FCI parallel step with N so the operator is “well-resolved”
    fci_planes_per_field_period_scale: float = 0.5,
    fci_nsteps: int = 16,
    cg_tol: float = 1e-10,
    cg_maxit: int = 4000,
    verbose: bool = True,
) -> DiscreteMMSResult:
    """
    Run one discrete MMS test at resolution N.

    f_disc is *exactly* A_FCI ψ_exact, using the same discrete operator we
    then use to solve the linear system.
    """
    if verbose:
        print("\n" + "=" * 70)
        print(f"FCI discrete MMS (torus): N={N}, R0={params.R0}, a={params.a}, eps={eps}")
        print("=" * 70)

    xs, ys, zs, Xq, inside_flat, band_flat, axis_flat = \
        build_torus_grid_and_masks(N, params, band_factor, axis_factor)

    nx, ny, nz = len(xs), len(ys), len(zs)
    Ntot = Xq.shape[0]

    # Manufactured ψ_exact on *all* nodes
    Xj = jnp.asarray(Xq, dtype=jnp.float64)
    psi_exact_all = np.asarray(psi_exact_vmap(Xj), dtype=float)

    # Candidate free nodes: core interior (as you already define)
    inside3 = inside_flat.reshape(nx, ny, nz)
    band3   = band_flat.reshape(nx, ny, nz)
    axis3   = axis_flat.reshape(nx, ny, nz)

    core3 = np.zeros_like(inside3, dtype=bool)
    core3[1:-1, 1:-1, 1:-1] = True
    core3 &= inside3 & (~band3) & (~axis3)
    core_flat = core3.ravel(order="C")

    # For MMS: free = core nodes; everything else is fixed and set to ψ_exact
    free = core_flat.copy()
    fixed = ~free

    psi_fixed_full = np.zeros(Ntot, dtype=float)
    psi_fixed_full[fixed] = psi_exact_all[fixed]
    
    # Build FCI operator
    @jax.jit
    def grad_phi_fn(X: jnp.ndarray) -> jnp.ndarray:
        # reuse B-field as "∇φ" proxy; FCI only cares about direction
        return B_field_vmap(X)

    if verbose:
        pinfo("Building FCI connectivity ...")

    # scale FCI planes with N so Δφ shrinks as we refine Cartesian grid
    fci_planes = max(16, int(fci_planes_per_field_period_scale * N))

    fci = build_fci_connectivity_chunked(
        xs, ys, zs,
        inside_flat,
        grad_phi_fn=grad_phi_fn,
        nfp=nfp,
        dphi_per_step=None,          # use built-in Δφ = 2π / (nfp * fci_planes)
        nsteps=fci_nsteps,
        verbose=verbose,
        chunk_size=None,
        fci_planes_per_field_period=fci_planes,
    )

    if verbose:
        pinfo(f"[FCI] valid connectivity nodes: {fci.valid.sum()} / {inside_flat.sum()}")

    # “Core” region: interior plus not on bands
    inside3 = inside_flat.reshape(nx, ny, nz)
    band3 = band_flat.reshape(nx, ny, nz)
    axis3 = axis_flat.reshape(nx, ny, nz)

    core3 = np.zeros_like(inside3, dtype=bool)
    core3[1:-1, 1:-1, 1:-1] = True
    core3 &= inside3 & (~band3) & (~axis3)
    core_flat = core3.ravel(order="C")
    if verbose:
        pinfo(f"[CORE] core nodes: {core_flat.sum()} / {inside_flat.sum()}")

    # Build FCI operator
    kappa_par = 1.0
    kappa_perp = eps ** 2

    A_pde_jax, deep_inside = make_fci_operator_jax(
        nx, ny, nz,
        xs, ys, zs,
        inside_flat,
        fci,
        core_mask=core_flat,
        kappa_par=kappa_par,
        kappa_perp=kappa_perp,
    )

    # Full matvec helper in numpy land (for clarity)
    def matvec_full(u_np: np.ndarray) -> np.ndarray:
        return np.asarray(A_pde_jax(jnp.asarray(u_np)))

    # Discrete RHS f_disc = A_FCI ψ_exact
    if verbose:
        pinfo("Computing discrete RHS f_disc = A_FCI ψ_exact ...")
    t0 = time.time()
    f_disc = matvec_full(psi_exact_all)
    if verbose:
        pinfo(f"Finished f_disc in {time.time() - t0:.2f} s.")
        
    # Build free-node system with lifting
    free = deep_inside & (~fixed)
    if not np.any(free):
        raise RuntimeError("No free nodes to solve for; check masks.")

    # RHS: f_disc - A ψ_fixed
    Apsi_fixed = matvec_full(psi_fixed_full)
    rhs_full = f_disc - Apsi_fixed
    b_free = rhs_full[free]
    b_free_j = jnp.asarray(b_free)

    Au_fixed_full_j = A_pde_jax(jnp.asarray(psi_fixed_full))
    Au_fixed_free_j = Au_fixed_full_j[free]

    # Free-node matvec in JAX
    def matvec_free_jax(u_free_j: jnp.ndarray) -> jnp.ndarray:
        u_full = jnp.asarray(psi_fixed_full)
        u_full = u_full.at[free].set(u_free_j)
        Au_full = A_pde_jax(u_full)
        return Au_full[free] - Au_fixed_free_j
    
    # --- DEBUG: build dense A matrix for a very small N ---
    Ntot_debug = Xq.shape[0]
    if N <= 10:  # only do this for small runs
        A_dense = np.zeros((Ntot_debug, Ntot_debug))
        for j in range(Ntot_debug):
            e = np.zeros(Ntot_debug)
            e[j] = 1.0
            A_dense[:, j] = matvec_full(e)

        free_idx  = np.where(free)[0]
        fixed_idx = np.where(fixed)[0]

        A_ff = A_dense[np.ix_(free_idx, free_idx)]
        A_fF = A_dense[np.ix_(free_idx, fixed_idx)]

        psi_exact_f = psi_exact_all[free_idx]
        psi_exact_F = psi_exact_all[fixed_idx]

        # f_disc_dense = A ψ_exact
        f_disc_dense = A_dense @ psi_exact_all
        rhs_f = f_disc_dense[free_idx] - A_fF @ psi_exact_F

        # Direct solve
        u_f = np.linalg.solve(A_ff, rhs_f)
        
        # After building A_dense and before printing DENSE DEBUG:
        f_disc_from_dense = A_dense @ psi_exact_all
        lin_check = np.linalg.norm(f_disc_from_dense - f_disc)
        print("[DENSE DEBUG] linearity check ||A_dense psi - f_disc||_2 =",
            lin_check)

        print("[DENSE DEBUG] ||u_f - psi_exact_f||_2 =",
            np.linalg.norm(u_f - psi_exact_f))

        psi_exact_f_norm = np.linalg.norm(psi_exact_f)
        print("[DENSE DEBUG] ||psi_exact_f||_2 =", psi_exact_f_norm)
        print("[DENSE DEBUG] relative error =", 
            np.linalg.norm(u_f - psi_exact_f) / (psi_exact_f_norm or 1.0))
        
        free_idx = np.where(free)[0]
        A_ff = A_dense[np.ix_(free_idx, free_idx)]

        # Quick-and-dirty: ratio of largest to smallest singular values
        s = np.linalg.svd(A_ff, compute_uv=False)
        condA = s[0] / s[-1]
        print("cond(A_ff) ~", condA)

    # Solve with JAX CG
    if verbose:
        pinfo("Solving A_FCI ψ_num = f_disc on free nodes (JAX CG) ...")
    t0 = time.time()
    psi_free_j, res_norm = cg_jax(matvec_free_jax, b_free_j, tol=cg_tol, maxiter=cg_maxit)
    psi_free = np.asarray(psi_free_j)
    if verbose:
        pinfo(f"CG finished in {time.time() - t0:.2f} s with ||r|| ≈ {res_norm:.3e}")
    
    # # Baseline: Au for the fixed field only
    # Au_fixed_full = matvec_full(psi_fixed_full)
    # Au_fixed_free = Au_fixed_full[free]   # shape (Nfree,)
    # Nfree = int(np.count_nonzero(free))
    # def matvec_free_np(u_free_np: np.ndarray) -> np.ndarray:
    #     # Build full field from free unknowns + fixed Dirichlet values
    #     u_full = psi_fixed_full.copy()
    #     u_full[free] = u_free_np
    #     Au_full = matvec_full(u_full)
    #     # Subtract baseline (A applied to the fixed-only field)
    #     # so that the operator is purely A_ff on the free unknowns.
    #     return Au_full[free] - Au_fixed_free
    # A_free = LinearOperator(
    #     shape=(Nfree, Nfree),
    #     matvec=matvec_free_np,
    #     dtype=float,)
    # u_test = np.random.randn(Nfree)
    # v_test = A_free @ u_test
    # print("Finite? ", np.all(np.isfinite(v_test)), " max |v|=", np.max(np.abs(v_test)))
    # if verbose: pinfo("Solving A_FCI ψ_num = f_disc on free nodes (SciPy GMRES) ...")
    # t0 = time.time()
    # psi_free, info = gmres(A_free, b_free, rtol=1e-8, restart=250, maxiter=3000)
    # t_solve = time.time() - t0
    # if verbose: pinfo(f"GMRES finished in {t_solve:.2f} s with info={info}")
    # # Compute residual norm explicitly
    # r = b_free - A_free @ psi_free
    # res_norm = np.linalg.norm(r)

    # Assemble full ψ_num
    psi_num = psi_fixed_full.copy()
    psi_num[free] = psi_free

    # Error on true interior (inside & not in either band)
    mask_err = inside_flat & (~band_flat) & (~axis_flat)
    diff = psi_num[mask_err] - psi_exact_all[mask_err]

    L2 = np.linalg.norm(diff)
    L2_ref = np.linalg.norm(psi_exact_all[mask_err]) or 1.0
    Linf = np.max(np.abs(diff))
    Linf_ref = np.max(np.abs(psi_exact_all[mask_err])) or 1.0

    L2_rel = L2 / L2_ref
    Linf_rel = Linf / Linf_ref

    print(f"[MMS] N={N:3d}  L2_rel={L2_rel:.3e}  Linf_rel={Linf_rel:.3e}  "
          f"CG_res={res_norm:.3e}")

    print("[DEBUG] N=", N)
    print("  free      =", np.count_nonzero(free))
    print("  deep_inside=", np.count_nonzero(deep_inside))
    print("  fixed     =", np.count_nonzero(fixed))
    print("  inside    =", np.count_nonzero(inside_flat))

    return DiscreteMMSResult(N=N, L2_rel=L2_rel, Linf_rel=Linf_rel, cg_res=float(res_norm))


# -----------------------------------------------------------------------------
# Driver: run several N and make a summary plot
# -----------------------------------------------------------------------------

def main():
    params = TorusParams(R0=1.5, a=0.5, m=1)
    eps = 1
    
    # A few resolutions
    N_list = [10, 24, 36, 48, 64]

    results: List[DiscreteMMSResult] = []
    for N in N_list:
        res = run_discrete_mms_single(
            N,
            params,
            eps=eps,
            band_factor=2.5,
            axis_factor=0.1,
            nfp=2,
            fci_planes_per_field_period_scale=0.05,  # Δφ ~ π/N
            fci_nsteps=16,
            cg_tol=1e-12,
            cg_maxit=4000,
            verbose=True,
        )
        results.append(res)

    print("\n=== Discrete FCI MMS summary ===")
    for r in results:
        print(
            f"N={r.N:3d}  L2_rel={r.L2_rel:.3e}  "
            f"Linf_rel={r.Linf_rel:.3e}  CG_res={r.cg_res:.3e}"
        )

    # Simple log plot of errors vs N (should be flat and tiny)
    Ns = np.array([r.N for r in results], dtype=float)
    L2s = np.array([r.L2_rel for r in results])
    Linfs = np.array([r.Linf_rel for r in results])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(Ns, L2s, "o-", label=r"$L^2$ error")
    ax.loglog(Ns, Linfs, "s--", label=r"$L^\infty$ error")
    ax.set_xlabel("N (grid points in x,y)")
    ax.set_ylabel("relative error")
    ax.set_title("Discrete MMS for FCI operator")
    ax.legend()
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig("fci_discrete_mms_convergence.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
