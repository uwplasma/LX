#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discrete manufactured-solution tests for the cylindrical FCI flux solver.

We test the *discrete* cylindrical operators from solve_flux_psi_fci.py:

  1) Full FCI operator in cylindrical coords:
       A_FCI_cyl[ψ] = -κ_par ∂_s^2 ψ - κ_perp ∇^2 ψ_cyl,

     where ∂_s^2 is represented via FCI connectivity along field lines,
     and ∇² is the *physical* cylindrical Laplacian

         ∇²ψ = (1/R) ∂_R (R ∂_R ψ) + (1/R²) ∂²_φ ψ + ∂²_Z ψ.

  2) A pure cylindrical Laplacian (no FCI):
       A_LAP_cyl[ψ] = -κ_perp ∇² ψ_cyl   (κ_par = 0),

     using the same discretization of ∇², but without field-line coupling.

We use a *discrete MMS* approach:

  • Build an analytic circular torus domain in cylindrical coordinates (R,φ,Z).
  • Define a smooth manufactured solution ψ_exact(R,φ,Z) that depends on
    both ρ (distance from axis) and φ.
  • For each operator A (FCI or Laplace):
      – Compute f_disc = A[ψ_exact] with the *same discrete operator* used
        in the solver.
      – Impose Dirichlet data ψ = ψ_exact on:
           • a thin outer boundary band near ρ ≈ a, and
           • a small inner "axis band" near ρ ≈ 0.
      – Solve A[ψ_num] = f_disc with those Dirichlet data (via lifting)
        on the free interior nodes.
      – Compare ψ_num to ψ_exact on the true interior region:
           inside & not in boundary band & not in axis band.

This directly validates the *discrete* operator (including the cylindrical
metric factors, periodic φ treatment, band handling, and FCI mapping).

We also:

  • Run a resolution study over (nR, nφ, nZ),
  • Plot L² and L^∞ errors vs resolution for both operators,
  • For the finest resolution, produce diagnostic plots suitable for
    publication:
      – R–Z contour slices of ψ_exact, ψ_num, and their difference,
      – 1D profiles vs R at fixed (φ,Z),
      – Histograms of the error.

NOTE: This script does *not* use the MFS multipole φ; instead it drives the
cylindrical operator with a synthetic toroidal field B = e_φ. That way we
test the FCI machinery itself (field-line tracing + interpolation +
metric) in isolation.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Import cylindrical FCI machinery from your solver
from solve_flux_psi_fci_cyl import (
    build_fci_connectivity_cylindrical,
    make_fci_operator_cylindrical,
    cg_jax,
    pinfo,
)


# =============================================================================
# Geometry and manufactured solution (cylindrical torus)
# =============================================================================

@dataclass
class TorusParams:
    R0: float = 1.5   # major radius
    a: float = 0.5    # minor radius
    m: int = 1        # toroidal mode in φ


def rho_from_RZ(R: jnp.ndarray, Z: jnp.ndarray, params: TorusParams) -> jnp.ndarray:
    """
    Distance ρ from the torus magnetic axis, expressed in cylindrical geometry:

        ρ = sqrt((R - R0)^2 + Z^2).
    """
    return jnp.sqrt((R - params.R0) ** 2 + Z * Z)


def psi_exact_cyl(R: jnp.ndarray, phi: jnp.ndarray, Z: jnp.ndarray,
                  params: TorusParams) -> jnp.ndarray:
    """
    Smooth manufactured ψ_exact(R,φ,Z) that varies in both radius and toroidal angle.

    We use:

        ψ_exact(ρ, φ) = (ρ/a)^2 * (1 - (ρ/a)^2) * cos(m φ),

    where ρ = sqrt((R - R0)^2 + Z^2).

    Properties:
      • ψ_exact → 0 smoothly at ρ = 0 (magnetic axis) and ρ = a (outer boundary),
      • non-trivial φ dependence to exercise the FCI parallel stencil.
    """
    rho = rho_from_RZ(R, Z, params)
    s = rho / params.a
    radial = s * s * (1.0 - s * s)
    return radial * jnp.cos(params.m * phi)


# Convenience JAX vectorized ψ_exact over (R,φ,Z) points
params_global = TorusParams()
def psi_exact_vec(R: jnp.ndarray, phi: jnp.ndarray, Z: jnp.ndarray) -> jnp.ndarray:
    return psi_exact_cyl(R, phi, Z, params_global)

psi_exact_vmap = jax.jit(
    jax.vmap(
        psi_exact_vec,
        in_axes=(0, 0, 0),
    )
)


# =============================================================================
# Synthetic toroidal B field (for FCI)
# =============================================================================

def B_field_xyz(xyz: jnp.ndarray) -> jnp.ndarray:
    """
    Magnetic field B(x) used by the FCI operator.

    We pick a simple *toroidal* field with unit magnitude:

        B = e_φ = (-sin φ, cos φ, 0),

    so |B| = 1 and field lines are circles at fixed R and Z.
    """
    x, y, z = xyz
    phi = jnp.arctan2(y, x)
    bx = -jnp.sin(phi)
    by =  jnp.cos(phi)
    bz =  0.0 * z
    return jnp.stack([bx, by, bz])


B_field_vmap = jax.jit(jax.vmap(B_field_xyz, in_axes=(0,)))


@jax.jit
def grad_phi_fn_from_B(X: jnp.ndarray) -> jnp.ndarray:
    """
    "grad φ" proxy for FCI: we just feed the B-field as the direction of ∇φ.

    The FCI operator only cares about the direction of the field, not the
    magnitude. This lets us test the cylindrical FCI machinery independently
    of the MFS multipole φ.
    """
    return B_field_vmap(X)


# =============================================================================
# Cylindrical grid and masks
# =============================================================================

def build_cyl_grid_and_masks(
    nR: int,
    nphi: int,
    nZ: int,
    params: TorusParams,
    band_width_factor: float = 1.5,
    axis_width_factor: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a cylindrical grid (R, φ, Z) and logical masks for a circular torus:

        - inside_flat: torus interior ρ < a,
        - band_flat  : thin boundary band near ρ ≈ a,
        - axis_flat  : small inner band near ρ ≈ 0.

    All masks are length Ntot and flatten in C order with (nR, nphi, nZ).

    band_width_factor and axis_width_factor are in units of:

        band_width  ≈ band_width_factor * min(dR, R_mid*dφ, dZ)
        axis_radius ≈ axis_width_factor * a.
    """
    R0, a = params.R0, params.a

    # Radial extent around the torus
    R_min = R0 - 1.1 * a
    R_max = R0 + 1.1 * a
    Z_min = -1.1 * a
    Z_max =  1.1 * a

    Rs   = np.linspace(R_min, R_max, nR)
    phis = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    Zs   = np.linspace(Z_min, Z_max, nZ)

    dR   = Rs[1]   - Rs[0]   if nR   > 1 else 1.0
    dphi = phis[1] - phis[0] if nphi > 1 else 1.0
    dZ   = Zs[1]   - Zs[0]   if nZ   > 1 else 1.0

    # Characteristic physical "voxel" size
    R_mid   = 0.5 * (R_min + R_max)
    voxel   = min(dR, R_mid * dphi, dZ)
    band_width  = band_width_factor * voxel
    axis_radius = axis_width_factor * a

    RR, PHI, ZZ = np.meshgrid(Rs, phis, Zs, indexing="ij")  # (nR,nphi,nZ)
    R_flat   = RR.ravel(order="C")
    phi_flat = PHI.ravel(order="C")
    Z_flat   = ZZ.ravel(order="C")

    # Compute ρ and distance to boundary in cylindrical form
    Rj   = jnp.asarray(R_flat,   dtype=jnp.float64)
    Zj   = jnp.asarray(Z_flat,   dtype=jnp.float64)
    rhoj = rho_from_RZ(Rj, Zj, params)
    rho  = np.asarray(rhoj, dtype=float)

    # Signed distance: negative inside, positive outside
    signed_dist = rho - params.a

    inside_flat = signed_dist < 0.0

    # Outer boundary band: |distance| <= band_width
    band_flat = np.abs(signed_dist) <= band_width

    # Inner axis band: ρ ≤ axis_radius, but only inside the torus
    axis_flat = (rho <= axis_radius) & inside_flat

    # Disambiguate any overlaps
    overlap = band_flat & axis_flat
    axis_flat[overlap] = True
    band_flat[overlap] = False

    Ntot = R_flat.size
    pinfo(
        f"[CYL-GRID] nR={nR}, nphi={nphi}, nZ={nZ}, Ntot={Ntot}, "
        f"dR≈{dR:.3e}, dφ≈{dphi:.3e}, dZ≈{dZ:.3e}"
    )
    print(f"[CYL-GRID] inside nodes        : {inside_flat.sum()} / {Ntot}")
    print(f"[CYL-GRID] boundary band nodes : {band_flat.sum()} / {Ntot}")
    print(f"[CYL-GRID] axis band nodes     : {axis_flat.sum()} / {Ntot}")
    print(f"[CYL-GRID] band width ≈ {band_width:.3e}, axis radius ≈ {axis_radius:.3e}")

    return Rs, phis, Zs, inside_flat, band_flat, axis_flat


# =============================================================================
# Pure cylindrical Laplacian operator (no FCI)
# =============================================================================

def make_cyl_laplacian_operator(
    nR: int,
    nphi: int,
    nZ: int,
    Rs: np.ndarray,
    phis: np.ndarray,
    Zs: np.ndarray,
    inside_flat: np.ndarray,
    kappa_perp: float,
):
    """
    Build a *pure* cylindrical Laplacian operator:

        A_LAP[ψ] = -κ_perp ∇² ψ_cyl,

    where

        ∇²ψ = (1/R) ∂_R (R ∂_R ψ) + (1/R²) ∂²_φ ψ + ∂²_Z ψ,

    using second-order central differences (interior in R,Z; periodic in φ).

    Returns:
      A_lap_jax   : function u_flat -> A_LAP[u_flat],
      deep_inside : boolean mask (numpy) of nodes where A_LAP is meaningful.
    """
    inside3 = inside_flat.reshape(nR, nphi, nZ)

    Rs_j   = jnp.asarray(Rs)
    phis_j = jnp.asarray(phis)
    Zs_j   = jnp.asarray(Zs)

    dR   = float(Rs[1]   - Rs[0])   if nR   > 1 else 1.0
    dphi = float(phis[1] - phis[0]) if nphi > 1 else 1.0
    dZ   = float(Zs[1]   - Zs[0])   if nZ   > 1 else 1.0

    # R array for metric term
    R3 = jnp.broadcast_to(Rs_j[:, None, None], (nR, nphi, nZ))

    @jax.jit
    def A_lap_jax(u_flat: jnp.ndarray) -> jnp.ndarray:
        u3 = u_flat.reshape((nR, nphi, nZ))
        lap3 = jnp.zeros_like(u3)

        # Interior in R,Z, all φ for each piece
        # R-derivatives
        uR_plus  = u3[2:,   :, 1:-1]
        uR_0     = u3[1:-1, :, 1:-1]
        uR_minus = u3[:-2,  :, 1:-1]
        R_mid    = R3[1:-1, :, 1:-1]

        d2u_dR2 = (uR_plus - 2.0 * uR_0 + uR_minus) / (dR * dR)
        du_dR   = (uR_plus - uR_minus) / (2.0 * dR)
        lap_R   = d2u_dR2 + du_dR / jnp.maximum(R_mid, 1e-8)

        # φ-derivatives (periodic)
        u_phi = u3[1:-1, :, 1:-1]  # (nR-2, nphi, nZ-2)
        u_phi_plus  = jnp.roll(u_phi, -1, axis=1)
        u_phi_minus = jnp.roll(u_phi, +1, axis=1)
        d2u_dphi2   = (u_phi_plus - 2.0 * u_phi + u_phi_minus) / (dphi * dphi)

        R_mid_phi = R3[1:-1, :, 1:-1]
        lap_phi   = d2u_dphi2 / jnp.maximum(R_mid_phi * R_mid_phi, 1e-12)

        # Z-derivatives
        uZ_plus  = u3[1:-1, :, 2:]
        uZ_0     = u3[1:-1, :, 1:-1]
        uZ_minus = u3[1:-1, :, :-2]
        d2u_dZ2  = (uZ_plus - 2.0 * uZ_0 + uZ_minus) / (dZ * dZ)

        total_lap = lap_R + lap_phi + d2u_dZ2
        lap3 = lap3.at[1:-1, :, 1:-1].set(total_lap)

        # Mask outside interior
        lap3 = jnp.where(inside3, lap3, 0.0)
        return -kappa_perp * lap3.ravel(order="C")

    deep_inside = np.asarray(inside3.ravel(order="C"))
    return A_lap_jax, deep_inside


# =============================================================================
# Single-resolution discrete MMS (FCI cylindrical)
# =============================================================================

@dataclass
class CylMMSResult:
    N_R: int
    N_phi: int
    N_Z: int
    L2_rel: float
    Linf_rel: float
    cg_res: float
    label: str


def run_discrete_mms_cyl_fci_single(
    nR: int,
    nphi: int,
    nZ: int,
    params: TorusParams,
    eps: float = 0.05,
    band_factor: float = 1.5,
    axis_factor: float = 0.2,
    nfp: int = 2,
    fci_planes_per_field_period_scale: float = 0.5,
    fci_nsteps: int = 16,
    cg_tol: float = 1e-11,
    cg_maxit: int = 4000,
    verbose: bool = True,
) -> CylMMSResult:
    """
    Run one discrete MMS test for the *cylindrical FCI operator*.

    The discrete operator is built from:

      build_fci_connectivity_cylindrical + make_fci_operator_cylindrical

    with κ_par=1, κ_perp=eps.

    The manufactured solution is ψ_exact_cyl(R,φ,Z) as defined above.
    """
    if verbose:
        print("\n" + "=" * 72)
        print(f"CYL-FCI discrete MMS: nR={nR}, nphi={nphi}, nZ={nZ}, "
              f"R0={params.R0}, a={params.a}, eps={eps}")
        print("=" * 72)

    Rs, phis, Zs, inside_flat, band_flat, axis_flat = \
        build_cyl_grid_and_masks(nR, nphi, nZ, params,
                                 band_width_factor=band_factor,
                                 axis_width_factor=axis_factor)

    nR, nphi, nZ = len(Rs), len(phis), len(Zs)
    Ntot = nR * nphi * nZ

    # Build coordinate arrays and ψ_exact on all nodes
    RR, PHI, ZZ = np.meshgrid(Rs, phis, Zs, indexing="ij")
    R_flat   = RR.ravel(order="C")
    phi_flat = PHI.ravel(order="C")
    Z_flat   = ZZ.ravel(order="C")

    Rj   = jnp.asarray(R_flat,   dtype=jnp.float64)
    phij = jnp.asarray(phi_flat, dtype=jnp.float64)
    Zj   = jnp.asarray(Z_flat,   dtype=jnp.float64)

    psi_exact_all = np.asarray(psi_exact_vmap(Rj, phij, Zj), dtype=float)

    # Free vs fixed nodes: for MMS, fix everything outside a "core" region.
    inside3 = inside_flat.reshape(nR, nphi, nZ)
    band3   = band_flat.reshape(nR, nphi, nZ)
    axis3   = axis_flat.reshape(nR, nphi, nZ)

    core3 = np.zeros_like(inside3, dtype=bool)
    # interior in R and Z, all φ are "core candidates"
    core3[1:-1, :, 1:-1] = True
    core3 &= inside3 & (~band3) & (~axis3)
    core_flat = core3.ravel(order="C")

    # For MMS: free = core nodes; everything else is fixed and set to ψ_exact
    free = core_flat.copy()
    fixed = ~free

    psi_fixed_full = np.zeros(Ntot, dtype=float)
    psi_fixed_full[fixed] = psi_exact_all[fixed]

    # Build FCI connectivity
    if verbose:
        pinfo("Building cylindrical FCI connectivity ...")

    # Scale FCI planes with nphi so Δφ shrinks with resolution
    fci_planes = max(16, int(fci_planes_per_field_period_scale * nphi))

    fci = build_fci_connectivity_cylindrical(
        Rs, phis, Zs,
        inside_flat,
        grad_phi_fn=grad_phi_fn_from_B,
        nfp=nfp,
        dphi_per_step=None,           # use Δφ = 2π / (nfp * fci_planes)
        nsteps=fci_nsteps,
        verbose=verbose,
        chunk_size=None,
        fci_planes_per_field_period=fci_planes,
    )

    if verbose:
        pinfo(f"[FCI-cyl] valid connectivity nodes: "
              f"{fci.valid.sum()} / {inside_flat.sum()}")

    # Build cylindrical FCI operator
    kappa_par  = 1.0
    kappa_perp = eps

    A_pde_jax, deep_inside = make_fci_operator_cylindrical(
        nR, nphi, nZ,
        Rs, phis, Zs,
        inside_flat,
        fci,
        core_mask=core_flat,
        kappa_par=kappa_par,
        kappa_perp=kappa_perp,
    )

    def matvec_full(u_np: np.ndarray) -> np.ndarray:
        return np.asarray(A_pde_jax(jnp.asarray(u_np)))

    # Discrete RHS f_disc = A_FCI ψ_exact
    if verbose:
        pinfo("Computing discrete RHS f_disc = A_FCI_cyl ψ_exact ...")
    t0 = time.time()
    f_disc = matvec_full(psi_exact_all)
    if verbose:
        pinfo(f"Finished f_disc in {time.time() - t0:.2f} s.")

    # Build free-node system with lifting
    free = deep_inside & (~fixed)
    if not np.any(free):
        raise RuntimeError("No free nodes to solve for in CYL-FCI; check masks.")

    Apsi_fixed = matvec_full(psi_fixed_full)
    rhs_full   = f_disc - Apsi_fixed
    b_free     = rhs_full[free]
    b_free_j   = jnp.asarray(b_free)

    # Baseline Au_fixed on free nodes for clean A_ff
    Au_fixed_full_j = A_pde_jax(jnp.asarray(psi_fixed_full))
    Au_fixed_free_j = Au_fixed_full_j[free]

    def matvec_free_jax(u_free_j: jnp.ndarray) -> jnp.ndarray:
        u_full = jnp.asarray(psi_fixed_full)
        u_full = u_full.at[free].set(u_free_j)
        Au_full = A_pde_jax(u_full)
        return Au_full[free] - Au_fixed_free_j

    # DEBUG: dense A for tiny grids
    Ntot_debug = Ntot
    if nR * nphi * nZ <= 10**3 and nR <= 10 and nphi <= 16 and nZ <= 10:
        pinfo("[DENSE-FCI] Building dense A matrix for tiny grid ...")
        A_dense = np.zeros((Ntot_debug, Ntot_debug))
        for j_col in range(Ntot_debug):
            e = np.zeros(Ntot_debug)
            e[j_col] = 1.0
            A_dense[:, j_col] = matvec_full(e)

        free_idx  = np.where(free)[0]
        fixed_idx = np.where(fixed)[0]
        A_ff = A_dense[np.ix_(free_idx, free_idx)]
        A_fF = A_dense[np.ix_(free_idx, fixed_idx)]

        psi_exact_f = psi_exact_all[free_idx]
        psi_exact_F = psi_exact_all[fixed_idx]
        f_disc_dense = A_dense @ psi_exact_all
        rhs_f_dense  = f_disc_dense[free_idx] - A_fF @ psi_exact_F

        # Direct solve
        u_f = np.linalg.solve(A_ff, rhs_f_dense)
        lin_check = np.linalg.norm(f_disc_dense - f_disc)
        print("[DENSE-FCI] ||A_dense ψ_exact - f_disc||_2 =", lin_check)
        rel_err_dense = np.linalg.norm(u_f - psi_exact_f) / (np.linalg.norm(psi_exact_f) or 1.0)
        print("[DENSE-FCI] relative error (dense) =", rel_err_dense)

    if verbose:
        pinfo("Solving A_FCI_cyl ψ_num = f_disc on free nodes (JAX CG) ...")
    t0 = time.time()
    psi_free_j, res_norm, k_final = cg_jax(
        matvec_free_jax,
        b_free_j,
        tol=cg_tol,
        maxiter=cg_maxit,
    )
    psi_free = np.asarray(psi_free_j)
    if verbose:
        pinfo(f"CG finished in {time.time() - t0:.2f} s "
              f"with rel-res ≈ {res_norm:.3e}, iters={int(k_final)}")

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

    L2_rel   = L2 / L2_ref
    Linf_rel = Linf / Linf_ref

    print(f"[CYL-FCI-MMS] nR={nR:3d} nφ={nphi:3d} nZ={nZ:3d}  "
          f"L2_rel={L2_rel:.3e}  Linf_rel={Linf_rel:.3e}  "
          f"CG_res={res_norm:.3e}")

    # Extra debug counts
    print("[CYL-FCI-DEBUG]")
    print("  free        =", np.count_nonzero(free))
    print("  deep_inside =", np.count_nonzero(deep_inside))
    print("  fixed       =", np.count_nonzero(fixed))
    print("  inside      =", np.count_nonzero(inside_flat))

    return CylMMSResult(
        N_R=nR, N_phi=nphi, N_Z=nZ,
        L2_rel=L2_rel,
        Linf_rel=Linf_rel,
        cg_res=float(res_norm),
        label="FCI-cyl",
    )


# =============================================================================
# Single-resolution discrete MMS (pure cylindrical Laplacian)
# =============================================================================

def run_discrete_mms_cyl_lap_single(
    nR: int,
    nphi: int,
    nZ: int,
    params: TorusParams,
    band_factor: float = 1.5,
    axis_factor: float = 0.2,
    kappa_perp: float = 1.0,
    cg_tol: float = 1e-11,
    cg_maxit: int = 4000,
    verbose: bool = True,
) -> CylMMSResult:
    """
    Run one discrete MMS test for the *pure cylindrical Laplacian* (no FCI):

        A_LAP[ψ] = -κ_perp ∇² ψ_cyl.

    This tests:
      • cylindrical metric factors (R,φ,Z),
      • periodic φ implementation,
      • band / axis lifting in cylindrical coordinates.
    """
    if verbose:
        print("\n" + "-" * 72)
        print(f"CYL-LAP discrete MMS: nR={nR}, nphi={nphi}, nZ={nZ}, "
              f"R0={params.R0}, a={params.a}, κ_perp={kappa_perp}")
        print("-" * 72)

    Rs, phis, Zs, inside_flat, band_flat, axis_flat = \
        build_cyl_grid_and_masks(nR, nphi, nZ, params,
                                 band_width_factor=band_factor,
                                 axis_width_factor=axis_factor)

    nR, nphi, nZ = len(Rs), len(phis), len(Zs)
    Ntot = nR * nphi * nZ

    RR, PHI, ZZ = np.meshgrid(Rs, phis, Zs, indexing="ij")
    R_flat   = RR.ravel(order="C")
    phi_flat = PHI.ravel(order="C")
    Z_flat   = ZZ.ravel(order="C")

    Rj   = jnp.asarray(R_flat,   dtype=jnp.float64)
    phij = jnp.asarray(phi_flat, dtype=jnp.float64)
    Zj   = jnp.asarray(Z_flat,   dtype=jnp.float64)

    psi_exact_all = np.asarray(psi_exact_vmap(Rj, phij, Zj), dtype=float)

    # Masks
    inside3 = inside_flat.reshape(nR, nphi, nZ)
    band3   = band_flat.reshape(nR, nphi, nZ)
    axis3   = axis_flat.reshape(nR, nphi, nZ)

    core3 = np.zeros_like(inside3, dtype=bool)
    core3[1:-1, :, 1:-1] = True
    core3 &= inside3 & (~band3) & (~axis3)
    core_flat = core3.ravel(order="C")

    # For MMS: again fix everything except core
    free = core_flat.copy()
    fixed = ~free

    psi_fixed_full = np.zeros(Ntot, dtype=float)
    psi_fixed_full[fixed] = psi_exact_all[fixed]

    # Build cylindrical Laplacian operator
    A_lap_jax, deep_inside = make_cyl_laplacian_operator(
        nR, nphi, nZ,
        Rs, phis, Zs,
        inside_flat,
        kappa_perp=kappa_perp,
    )

    def matvec_full(u_np: np.ndarray) -> np.ndarray:
        return np.asarray(A_lap_jax(jnp.asarray(u_np)))

    if verbose:
        pinfo("Computing discrete RHS f_disc = A_LAP_cyl ψ_exact ...")
    t0 = time.time()
    f_disc = matvec_full(psi_exact_all)
    if verbose:
        pinfo(f"Finished f_disc in {time.time() - t0:.2f} s.")

    free = deep_inside & (~fixed)
    if not np.any(free):
        raise RuntimeError("No free nodes to solve for in CYL-LAP; check masks.")

    Apsi_fixed = matvec_full(psi_fixed_full)
    rhs_full   = f_disc - Apsi_fixed
    b_free     = rhs_full[free]
    b_free_j   = jnp.asarray(b_free)

    Au_fixed_full_j = A_lap_jax(jnp.asarray(psi_fixed_full))
    Au_fixed_free_j = Au_fixed_full_j[free]

    def matvec_free_jax(u_free_j: jnp.ndarray) -> jnp.ndarray:
        u_full = jnp.asarray(psi_fixed_full)
        u_full = u_full.at[free].set(u_free_j)
        Au_full = A_lap_jax(u_full)
        return Au_full[free] - Au_fixed_free_j

    if verbose:
        pinfo("Solving A_LAP_cyl ψ_num = f_disc on free nodes (JAX CG) ...")
    t0 = time.time()
    psi_free_j, res_norm, k_final = cg_jax(
        matvec_free_jax,
        b_free_j,
        tol=cg_tol,
        maxiter=cg_maxit,
    )
    psi_free = np.asarray(psi_free_j)
    if verbose:
        pinfo(f"CG finished in {time.time() - t0:.2f} s "
              f"with rel-res ≈ {res_norm:.3e}, iters={int(k_final)}")

    psi_num = psi_fixed_full.copy()
    psi_num[free] = psi_free

    mask_err = inside_flat & (~band_flat) & (~axis_flat)
    diff = psi_num[mask_err] - psi_exact_all[mask_err]

    L2 = np.linalg.norm(diff)
    L2_ref = np.linalg.norm(psi_exact_all[mask_err]) or 1.0
    Linf = np.max(np.abs(diff))
    Linf_ref = np.max(np.abs(psi_exact_all[mask_err])) or 1.0

    L2_rel   = L2 / L2_ref
    Linf_rel = Linf / Linf_ref

    print(f"[CYL-LAP-MMS] nR={nR:3d} nφ={nphi:3d} nZ={nZ:3d}  "
          f"L2_rel={L2_rel:.3e}  Linf_rel={Linf_rel:.3e}  "
          f"CG_res={res_norm:.3e}")

    print("[CYL-LAP-DEBUG]")
    print("  free        =", np.count_nonzero(free))
    print("  deep_inside =", np.count_nonzero(deep_inside))
    print("  fixed       =", np.count_nonzero(fixed))
    print("  inside      =", np.count_nonzero(inside_flat))

    return CylMMSResult(
        N_R=nR, N_phi=nphi, N_Z=nZ,
        L2_rel=L2_rel,
        Linf_rel=Linf_rel,
        cg_res=float(res_norm),
        label="LAP-cyl",
    )


# =============================================================================
# Diagnostics / plotting for finest resolution
# =============================================================================

def plot_cyl_slice_comparison(
    nR: int,
    nphi: int,
    nZ: int,
    Rs: np.ndarray,
    phis: np.ndarray,
    Zs: np.ndarray,
    psi_exact_all: np.ndarray,
    psi_num_all: np.ndarray,
    inside_flat: np.ndarray,
    band_flat: np.ndarray,
    axis_flat: np.ndarray,
    params: TorusParams,
    phi_index: int = 0,
    title_prefix: str = "CYL-FCI",
    save_prefix: str = "cyl_fci",
):
    """
    Make R–Z slice plots at a fixed φ index for:

      • ψ_exact(R,Z),
      • ψ_num(R,Z),
      • difference ψ_num - ψ_exact,

    on the interior region, with boundary/axis bands overplotted.
    """
    RR, PHI, ZZ = np.meshgrid(Rs, phis, Zs, indexing="ij")
    psi_exact_3 = psi_exact_all.reshape((nR, nphi, nZ), order="C")
    psi_num_3   = psi_num_all.reshape((nR, nphi, nZ), order="C")

    inside3 = inside_flat.reshape((nR, nphi, nZ))
    band3   = band_flat.reshape((nR, nphi, nZ))
    axis3   = axis_flat.reshape((nR, nphi, nZ))

    # slice
    psi_ex_slice = np.where(inside3[:, phi_index, :],
                            psi_exact_3[:, phi_index, :],
                            np.nan)
    psi_num_slice = np.where(inside3[:, phi_index, :],
                             psi_num_3[:, phi_index, :],
                             np.nan)
    diff_slice = psi_num_slice - psi_ex_slice

    R_slice = RR[:, phi_index, :]
    Z_slice = ZZ[:, phi_index, :]

    extent = [Rs[0], Rs[-1], Zs[0], Zs[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    im0 = axes[0].imshow(
        psi_ex_slice.T, origin="lower", extent=extent, aspect="equal"
    )
    axes[0].set_title(r"$\psi_{\rm exact}$")
    axes[0].set_xlabel("R")
    axes[0].set_ylabel("Z")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(
        psi_num_slice.T, origin="lower", extent=extent, aspect="equal"
    )
    axes[1].set_title(r"$\psi_{\rm num}$")
    axes[1].set_xlabel("R")
    axes[1].set_ylabel("Z")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(
        diff_slice.T, origin="lower", extent=extent, aspect="equal"
    )
    axes[2].set_title(r"$\psi_{\rm num}-\psi_{\rm exact}$")
    axes[2].set_xlabel("R")
    axes[2].set_ylabel("Z")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle(f"{title_prefix}: R–Z slice at φ index j={phi_index}")
    fig.savefig(f"{save_prefix}_RZ_slice_phi{phi_index:03d}.png", dpi=200)
    plt.show()

    # Histograms of errors
    mask_err = inside_flat & (~band_flat) & (~axis_flat)
    diff_all = psi_num_all[mask_err] - psi_exact_all[mask_err]
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax2[0].hist(diff_all, bins=80)
    ax2[0].set_xlabel(r"$\psi_{\rm num}-\psi_{\rm exact}$")
    ax2[0].set_ylabel("count")
    ax2[0].set_title("Error histogram")

    ax2[1].hist(np.abs(diff_all), bins=80)
    ax2[1].set_xlabel(r"$|\psi_{\rm num}-\psi_{\rm exact}|$")
    ax2[1].set_ylabel("count")
    ax2[1].set_yscale("log")
    ax2[1].set_title("Absolute error (log scale)")

    fig2.suptitle(f"{title_prefix}: Error statistics")
    fig2.savefig(f"{save_prefix}_error_histograms.png", dpi=200)
    plt.show()


# =============================================================================
# Driver
# =============================================================================

def main():
    params = TorusParams(R0=1.5, a=0.5, m=1)

    # Choose a set of resolutions in R,Z and scale φ accordingly.
    # nphi ~ 2*nR gives roughly isotropic resolution in the toroidal direction.
    N_list = [12, 20, 28, 36]

    eps = 0.05   # perpendicular anisotropy for FCI operator
    kappa_perp_lap = 1.0

    fci_results: List[CylMMSResult] = []
    lap_results: List[CylMMSResult] = []

    for N in N_list:
        nR  = N
        nZ  = N
        nphi = 2 * N

        # FCI cylindrical MMS
        res_fci = run_discrete_mms_cyl_fci_single(
            nR, nphi, nZ,
            params,
            eps=eps,
            band_factor=2.0,
            axis_factor=0.15,
            nfp=2,
            fci_planes_per_field_period_scale=0.3,
            fci_nsteps=16,
            cg_tol=1e-11,
            cg_maxit=4000,
            verbose=True,
        )
        fci_results.append(res_fci)

        # Pure cylindrical Laplacian MMS (no FCI)
        res_lap = run_discrete_mms_cyl_lap_single(
            nR, nphi, nZ,
            params,
            band_factor=2.0,
            axis_factor=0.15,
            kappa_perp=kappa_perp_lap,
            cg_tol=1e-11,
            cg_maxit=4000,
            verbose=True,
        )
        lap_results.append(res_lap)

    print("\n=== Discrete CYL-FCI MMS summary ===")
    for r in fci_results:
        print(
            f"[FCI] nR={r.N_R:3d} nφ={r.N_phi:3d} nZ={r.N_Z:3d}  "
            f"L2_rel={r.L2_rel:.3e}  Linf_rel={r.Linf_rel:.3e}  "
            f"CG_res={r.cg_res:.3e}"
        )

    print("\n=== Discrete CYL-LAP MMS summary ===")
    for r in lap_results:
        print(
            f"[LAP] nR={r.N_R:3d} nφ={r.N_phi:3d} nZ={r.N_Z:3d}  "
            f"L2_rel={r.L2_rel:.3e}  Linf_rel={r.Linf_rel:.3e}  "
            f"CG_res={r.cg_res:.3e}"
        )

    # Simple log-log plots vs "N" ~ nR to show convergence (or flat discrete MMS)
    Ns = np.array([r.N_R for r in fci_results], dtype=float)
    L2s_fci   = np.array([r.L2_rel for r in fci_results])
    Linfs_fci = np.array([r.Linf_rel for r in fci_results])

    L2s_lap   = np.array([r.L2_rel for r in lap_results])
    Linfs_lap = np.array([r.Linf_rel for r in lap_results])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(Ns, L2s_fci, "o-",  label=r"$L^2$ (FCI-cyl)")
    ax.loglog(Ns, Linfs_fci, "s--", label=r"$L^\infty$ (FCI-cyl)")

    ax.loglog(Ns, L2s_lap, "o-",  alpha=0.6, label=r"$L^2$ (LAP-cyl)")
    ax.loglog(Ns, Linfs_lap, "s--", alpha=0.6, label=r"$L^\infty$ (LAP-cyl)")

    ax.set_xlabel(r"$N_R$ (grid points in $R$)")
    ax.set_ylabel("relative error")
    ax.set_title("Discrete MMS for cylindrical operators")
    ax.legend()
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig("cyl_fci_lap_discrete_mms_convergence.png", dpi=200)
    plt.show()

    # For the finest resolution, make nice R–Z slice plots for FCI solution
    N_best = N_list[-1]
    nR  = N_best
    nZ  = N_best
    nphi = 2 * N_best

    # Re-run a FCI case, but keep fields for plotting
    Rs, phis, Zs, inside_flat, band_flat, axis_flat = \
        build_cyl_grid_and_masks(nR, nphi, nZ, params,
                                 band_width_factor=2.0,
                                 axis_width_factor=0.15)
    RR, PHI, ZZ = np.meshgrid(Rs, phis, Zs, indexing="ij")
    R_flat   = RR.ravel(order="C")
    phi_flat = PHI.ravel(order="C")
    Z_flat   = ZZ.ravel(order="C")

    Rj   = jnp.asarray(R_flat,   dtype=jnp.float64)
    phij = jnp.asarray(phi_flat, dtype=jnp.float64)
    Zj   = jnp.asarray(Z_flat,   dtype=jnp.float64)
    psi_exact_all = np.asarray(psi_exact_vmap(Rj, phij, Zj), dtype=float)

    # Build operator and solve once more (FCI)
    fci_planes = max(16, int(0.3 * nphi))
    fci = build_fci_connectivity_cylindrical(
        Rs, phis, Zs,
        inside_flat,
        grad_phi_fn=grad_phi_fn_from_B,
        nfp=2,
        dphi_per_step=None,
        nsteps=16,
        verbose=True,
        chunk_size=None,
        fci_planes_per_field_period=fci_planes,
    )

    inside3 = inside_flat.reshape(nR, nphi, nZ)
    band3   = band_flat.reshape(nR, nphi, nZ)
    axis3   = axis_flat.reshape(nR, nphi, nZ)
    core3   = np.zeros_like(inside3, dtype=bool)
    core3[1:-1, :, 1:-1] = True
    core3 &= inside3 & (~band3) & (~axis3)
    core_flat = core3.ravel(order="C")

    A_pde_jax, deep_inside = make_fci_operator_cylindrical(
        nR, nphi, nZ,
        Rs, phis, Zs,
        inside_flat,
        fci,
        core_mask=core_flat,
        kappa_par=1.0,
        kappa_perp=eps,
    )

    def matvec_full(u_np: np.ndarray) -> np.ndarray:
        return np.asarray(A_pde_jax(jnp.asarray(u_np)))

    Ntot = nR * nphi * nZ
    free = deep_inside & core_flat
    fixed = ~free

    psi_fixed_full = np.zeros(Ntot, dtype=float)
    psi_fixed_full[fixed] = psi_exact_all[fixed]

    f_disc = matvec_full(psi_exact_all)
    Apsi_fixed = matvec_full(psi_fixed_full)
    rhs_full   = f_disc - Apsi_fixed
    b_free     = rhs_full[free]
    b_free_j   = jnp.asarray(b_free)

    Au_fixed_full_j = A_pde_jax(jnp.asarray(psi_fixed_full))
    Au_fixed_free_j = Au_fixed_full_j[free]

    def matvec_free_jax(u_free_j: jnp.ndarray) -> jnp.ndarray:
        u_full = jnp.asarray(psi_fixed_full)
        u_full = u_full.at[free].set(u_free_j)
        Au_full = A_pde_jax(u_full)
        return Au_full[free] - Au_fixed_free_j

    pinfo("[PLOT-FCI] Solving at finest resolution for slice plots ...")
    psi_free_j, res_norm, k_final = cg_jax(
        matvec_free_jax,
        b_free_j,
        tol=1e-11,
        maxiter=4000,
    )
    psi_free = np.asarray(psi_free_j)
    psi_num_all = psi_fixed_full.copy()
    psi_num_all[free] = psi_free

    # R–Z slice and error plots
    plot_cyl_slice_comparison(
        nR, nphi, nZ,
        Rs, phis, Zs,
        psi_exact_all,
        psi_num_all,
        inside_flat,
        band_flat,
        axis_flat,
        params,
        phi_index=0,
        title_prefix="CYL-FCI MMS",
        save_prefix="cyl_fci_mms",
    )


if __name__ == "__main__":
    main()
