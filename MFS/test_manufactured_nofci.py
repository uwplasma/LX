#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manufactured-solution tests for -div(D ∇ψ) = 0 on:
  - Box [0,1]^3
  - Cylinder (finite-length, centered on z-axis)
  - Torus (donut)

We reuse the JAX tensor operator from solve_flux_psi_fci.py:
    make_linear_operator_jax, diffusion_tensor_jax, cg_jax

Exact manufactured solution:
    ψ_exact(x,y,z) = x + 0.3*y + 0.1*z

For any constant symmetric tensor D, this satisfies -div(D ∇ψ_exact)=0.

For each geometry we test:
  1) Isotropic D = I
  2) Anisotropic D from diffusion_tensor_jax(G=(0,0,1), eps, delta=0)

Dirichlet BC: ψ = ψ_exact on the *discrete boundary* of the domain
(inside nodes that have at least one neighbor outside).

We compute relative L2 and Linf errors in the interior and plot mid-plane slices.
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from solve_flux_psi_fci import make_linear_operator_jax, diffusion_tensor_jax, cg_jax


# ---------------------- Manufactured solution ---------------------- #

def psi_exact_fn(x, y, z):
    """Simple linear harmonic solution."""
    return x + 0.3 * y + 0.1 * z


# ---------------------- Grid + flattening helpers ------------------ #

def build_cartesian_grid(bounds, N):
    """
    Build uniform Cartesian grid and Xq with the same flattening convention
    used in solve_flux_psi_fci.py.

    bounds = ((xmin,xmax), (ymin,ymax), (zmin,zmax))
    N      = number of points per axis (int)
    """
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
        ZZ.ravel(order="C")
    ])
    return xs, ys, zs, dx, dy, dz, Xq


def compute_discrete_boundary(inside3):
    """
    Given inside3 (nx,ny,nz) boolean, return (inside_flat, boundary_flat)
    where boundary_flat is True for inside nodes that have at least one
    6-connected neighbor outside *or* lie on the outer box faces.

    This makes the unit box [0,1]^3 case well-posed (faces are Dirichlet),
    while still detecting curved boundaries for cylinder/torus.
    """
    nx, ny, nz = inside3.shape
    boundary3 = np.zeros_like(inside3, dtype=bool)

    # interior cells that see an outside neighbour
    core = inside3[1:-1, 1:-1, 1:-1]
    neigh_out = (
        ~inside3[0:-2, 1:-1, 1:-1] |  # left
        ~inside3[2:  , 1:-1, 1:-1] |  # right
        ~inside3[1:-1, 0:-2, 1:-1] |  # back
        ~inside3[1:-1, 2:  , 1:-1] |  # front
        ~inside3[1:-1, 1:-1, 0:-2] |  # bottom
        ~inside3[1:-1, 1:-1, 2:  ]    # top
    )
    boundary3[1:-1, 1:-1, 1:-1] = core & neigh_out

    # additionally: any inside cell on the outer box faces is boundary
    # (needed when inside3 is "full", e.g. unit box)
    boundary3[0,   :,   :] |= inside3[0,   :,   :]
    boundary3[nx-1, :,   :] |= inside3[nx-1, :,   :]
    boundary3[:,   0,   :] |= inside3[:,   0,   :]
    boundary3[:,   ny-1, :] |= inside3[:,   ny-1, :]
    boundary3[:,   :,   0] |= inside3[:,   :,   0]
    boundary3[:,   :, nz-1] |= inside3[:,   :, nz-1]

    inside_flat   = inside3.ravel(order="C")
    boundary_flat = boundary3.ravel(order="C")
    return inside_flat, boundary_flat

# ---------------------- Geometries: inside masks ------------------- #

def inside_box(xs, ys, zs):
    """
    Simple box [0,1]^3 embedded in its own grid: everything inside True.
    """
    nx, ny, nz = len(xs), len(ys), len(zs)
    inside3 = np.ones((nx, ny, nz), dtype=bool)
    return inside3


def inside_cylinder(xs, ys, zs, R0=0.6, Lz=1.0):
    """
    Finite cylinder aligned with z-axis, centered at z=0:
        x^2 + y^2 <= R0^2,   |z| <= Lz/2
    """
    nx, ny, nz = len(xs), len(ys), len(zs)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)

    R2 = XX**2 + YY**2
    inside3 = (R2 <= R0**2) & (np.abs(ZZ) <= Lz / 2.0)
    return inside3


def inside_torus(xs, ys, zs, R0=1.5, a=0.5):
    """
    Standard axisymmetric torus centered on z-axis:
        (sqrt(x^2 + y^2) - R0)^2 + z^2 <= a^2
    """
    nx, ny, nz = len(xs), len(ys), len(zs)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)

    R = np.sqrt(XX**2 + YY**2)
    dist_minor2 = (R - R0)**2 + ZZ**2
    inside3 = dist_minor2 <= a**2
    return inside3


# ---------------------- Core manufactured solve -------------------- #

def solve_manufactured_with_operator(
    xs,
    ys,
    zs,
    dx,
    dy,
    dz,
    Xq,
    inside3,
    Dfield,
    tol=1e-10,
    maxiter=5000,
    label="",
):
    """
    Solve -div(D ∇ψ) = 0 with Dirichlet BC ψ=ψ_exact on the discrete boundary
    of the given geometry.

    xs,ys,zs,dx,dy,dz,Xq: grid
    inside3: (nx,ny,nz) boolean indicating domain (True inside)
    Dfield:  (N,3,3) numpy array with D at all grid nodes (constant or varying)

    Returns:
      psi_num, psi_exact, inside_flat, boundary_flat
    """
    nx, ny, nz = len(xs), len(ys), len(zs)
    Ntot = nx * ny * nz

    inside_flat, boundary_flat = compute_discrete_boundary(inside3)

    # Build operator
    A_pde_jax, deep_inside = make_linear_operator_jax(
        nx, ny, nz,
        dx, dy, dz,
        inside_flat,
        Dfield,
    )

    # Manufactured exact solution
    psi_exact = psi_exact_fn(Xq[:, 0], Xq[:, 1], Xq[:, 2])

    # Dirichlet BC on boundary nodes
    fixed = boundary_flat.copy()
    val = np.zeros(Ntot, dtype=float)
    val[fixed] = psi_exact[fixed]

    # "deep_inside" from operator is just inside_flat in this construction,
    # but we keep it for consistency with your main solver.
    free = deep_inside & (~fixed)
    if not np.any(free):
        raise RuntimeError(f"[{label}] No free nodes (geometry too small or degenerate).")

    print(f"[{label}] free nodes: {free.sum()} / {Ntot}")

    # Lifting: ψ = ψ_free + ψ_fixed
    psi_fixed_full = np.zeros(Ntot, dtype=float)
    psi_fixed_full[fixed] = val[fixed]

    F0_full = np.array(A_pde_jax(jnp.asarray(psi_fixed_full)))
    b_free = -F0_full[free]

    # JAX CG on reduced system
    free_mask = free

    def matvec_free_jax(u_free_j):
        u_full = jnp.zeros(Ntot, dtype=jnp.float64)
        u_full = u_full.at[free_mask].set(u_free_j)
        Au_full = A_pde_jax(u_full)
        return Au_full[free_mask]

    b_free_j = jnp.asarray(b_free, dtype=jnp.float64)
    print(f"[{label}] Solving (JAX CG) ...")
    psi_free_j, res_norm = cg_jax(matvec_free_jax, b_free_j, tol=tol, maxiter=maxiter)
    psi_free = np.asarray(psi_free_j)
    print(f"[{label}] CG finished with ||r||₂ ≈ {float(res_norm):.3e}")

    psi_num = np.array(psi_fixed_full)
    psi_num[free] = psi_free

    return psi_num, psi_exact, inside_flat, boundary_flat


def compute_errors(psi_num, psi_exact, inside_flat, boundary_flat, label=""):
    """Compute L2 and Linf errors in the interior (inside & not boundary)."""
    interior = inside_flat & (~boundary_flat)
    err = psi_num - psi_exact
    rel_L2 = np.linalg.norm(err[interior]) / np.linalg.norm(psi_exact[interior])
    Linf = np.max(np.abs(err[interior]))
    print(f"[{label}] relative L2 error (interior) = {rel_L2:.3e}")
    print(f"[{label}] Linf error (interior)        = {Linf:.3e}")
    return rel_L2, Linf


def plot_midplane_slices(xs, ys, zs, psi_exact, psi_num, inside_flat, geometry_label, D_label):
    """Plot mid-plane (z≈0) slices: exact, numeric, relative error."""
    nx, ny, nz = len(xs), len(ys), len(zs)
    psi_exact_3 = psi_exact.reshape(nx, ny, nz, order="C")
    psi_num_3   = psi_num.reshape(nx, ny, nz, order="C")
    inside3     = inside_flat.reshape(nx, ny, nz, order="C")

    # choose k closest to z=0
    k_mid = np.argmin(np.abs(zs - 0.0))

    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    # mask outside domain for plotting
    slice_exact = psi_exact_3[:, :, k_mid]
    slice_num   = psi_num_3[:, :, k_mid]
    slice_mask  = inside3[:, :, k_mid]

    slice_exact_plot = np.where(slice_mask, slice_exact, np.nan)
    slice_num_plot   = np.where(slice_mask, slice_num, np.nan)

    im0 = axs[0].imshow(slice_exact_plot.T, origin="lower",
                        extent=extent, aspect="equal")
    axs[0].set_title(f"Exact ψ (z≈0)")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("y")
    fig.colorbar(im0, ax=axs[0], shrink=0.8)

    im1 = axs[1].imshow(slice_num_plot.T, origin="lower",
                        extent=extent, aspect="equal")
    axs[1].set_title(f"Numerical ψ ({D_label})")
    axs[1].set_xlabel("x"); axs[1].set_ylabel("y")
    fig.colorbar(im1, ax=axs[1], shrink=0.8)

    # relative error on the slice
    rel_err = np.zeros_like(slice_num)
    mask_nonzero = (np.abs(slice_exact) > 1e-14) & slice_mask
    rel_err[mask_nonzero] = (slice_num[mask_nonzero] - slice_exact[mask_nonzero]) / slice_exact[mask_nonzero]
    rel_err_plot = np.where(slice_mask, rel_err, np.nan)

    im2 = axs[2].imshow(rel_err_plot.T, origin="lower",
                        extent=extent, aspect="equal")
    axs[2].set_title("Relative error")
    axs[2].set_xlabel("x"); axs[2].set_ylabel("y")
    fig.colorbar(im2, ax=axs[2], shrink=0.8)

    fig.suptitle(f"{geometry_label}, {D_label}")
    
    plt.savefig(f"mfs_{geometry_label.replace(' ','_')}_{D_label.replace(' ','_')}_midplane.png")
    return fig


# ---------------------- Driver: run three geometries ---------------- #

def run_all_geometries(N=40, eps_aniso=1e-3, plot=True):
    """
    Run box, cylinder and torus tests at a single resolution N.

    Returns a dict of errors:
      {
        'box': {'iso': (relL2, Linf), 'aniso': (relL2, Linf)},
        'cyl': {'iso': ...},
        'tor': {'iso': ...},
      }
    """
    results = {
        'box': {},
        'cyl': {},
        'tor': {},
    }

    # --- Geometry 1: Box [0,1]^3 ---
    print("\n================ BOX GEOMETRY ================")
    bounds_box = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    xs, ys, zs, dx, dy, dz, Xq = build_cartesian_grid(bounds_box, N)
    inside3_box = inside_box(xs, ys, zs)

    # D = I
    Ntot = N**3
    D_iso = np.zeros((Ntot, 3, 3), dtype=float)
    for i in range(3):
        D_iso[:, i, i] = 1.0

    psi_num, psi_exact, inside_flat, boundary_flat = solve_manufactured_with_operator(
        xs, ys, zs, dx, dy, dz, Xq,
        inside3_box, D_iso,
        label="BOX / D=I"
    )
    relL2, Linf = compute_errors(psi_num, psi_exact, inside_flat, boundary_flat, label="BOX / D=I")
    results['box']['iso'] = (relL2, Linf)
    if plot:
        plot_midplane_slices(xs, ys, zs, psi_exact, psi_num, inside_flat,
                             "Box [0,1]^3", "D = I")

    # D = anisotropic from diffusion_tensor_jax
    print("\n[BOX] Anisotropic D (diffusion_tensor_jax, G=(0,0,1))")
    G_const = np.tile(np.array([0.0, 0.0, 1.0]), (Ntot, 1))
    D_aniso_j = diffusion_tensor_jax(jnp.asarray(G_const), eps=eps_aniso, delta=0.0)
    D_aniso = np.asarray(D_aniso_j)

    psi_num_a, _, _, _ = solve_manufactured_with_operator(
        xs, ys, zs, dx, dy, dz, Xq,
        inside3_box, D_aniso,
        label="BOX / D_aniso"
    )
    relL2, Linf = compute_errors(psi_num_a, psi_exact, inside_flat, boundary_flat, label="BOX / D_aniso")
    results['box']['aniso'] = (relL2, Linf)
    if plot:
        plot_midplane_slices(xs, ys, zs, psi_exact, psi_num_a, inside_flat,
                             "Box [0,1]^3", f"D_aniso (eps={eps_aniso:g})")

    # --- Geometry 2: Cylinder ---
    print("\n================ CYLINDER GEOMETRY ================")
    R0 = 0.6
    Lz = 1.0
    bounds_cyl = ((-0.9, 0.9), (-0.9, 0.9), (-0.7, 0.7))
    xs, ys, zs, dx, dy, dz, Xq = build_cartesian_grid(bounds_cyl, N)
    inside3_cyl = inside_cylinder(xs, ys, zs, R0=R0, Lz=Lz)

    Ntot = xs.size * ys.size * zs.size
    D_iso = np.zeros((Ntot, 3, 3), dtype=float)
    for i in range(3):
        D_iso[:, i, i] = 1.0

    psi_num, psi_exact, inside_flat, boundary_flat = solve_manufactured_with_operator(
        xs, ys, zs, dx, dy, dz, Xq,
        inside3_cyl, D_iso,
        label="CYL / D=I"
    )
    relL2, Linf = compute_errors(psi_num, psi_exact, inside_flat, boundary_flat, label="CYL / D=I")
    results['cyl']['iso'] = (relL2, Linf)
    if plot:
        plot_midplane_slices(xs, ys, zs, psi_exact, psi_num, inside_flat,
                             "Cylinder (R0=0.6, Lz=1)", "D = I")

    print("\n[CYL] Anisotropic D (diffusion_tensor_jax, G=(0,0,1))")
    G_const = np.tile(np.array([0.0, 0.0, 1.0]), (Ntot, 1))
    D_aniso_j = diffusion_tensor_jax(jnp.asarray(G_const), eps=eps_aniso, delta=0.0)
    D_aniso = np.asarray(D_aniso_j)

    psi_num_a, _, _, _ = solve_manufactured_with_operator(
        xs, ys, zs, dx, dy, dz, Xq,
        inside3_cyl, D_aniso,
        label="CYL / D_aniso"
    )
    relL2, Linf = compute_errors(psi_num_a, psi_exact, inside_flat, boundary_flat, label="CYL / D_aniso")
    results['cyl']['aniso'] = (relL2, Linf)
    if plot:
        plot_midplane_slices(xs, ys, zs, psi_exact, psi_num_a, inside_flat,
                             "Cylinder (R0=0.6, Lz=1)",
                             f"D_aniso (eps={eps_aniso:g})")

    # --- Geometry 3: Torus ---
    print("\n================ TORUS GEOMETRY ================")
    R0_t = 1.5
    a_t = 0.5
    bounds_tor = ((-2.2, 2.2), (-2.2, 2.2), (-0.9, 0.9))
    xs, ys, zs, dx, dy, dz, Xq = build_cartesian_grid(bounds_tor, N)
    inside3_tor = inside_torus(xs, ys, zs, R0=R0_t, a=a_t)

    Ntot = xs.size * ys.size * zs.size
    D_iso = np.zeros((Ntot, 3, 3), dtype=float)
    for i in range(3):
        D_iso[:, i, i] = 1.0

    psi_num, psi_exact, inside_flat, boundary_flat = solve_manufactured_with_operator(
        xs, ys, zs, dx, dy, dz, Xq,
        inside3_tor, D_iso,
        label="TOR / D=I"
    )
    relL2, Linf = compute_errors(psi_num, psi_exact, inside_flat, boundary_flat, label="TOR / D=I")
    results['tor']['iso'] = (relL2, Linf)
    if plot:
        plot_midplane_slices(xs, ys, zs, psi_exact, psi_num, inside_flat,
                             f"Torus (R0={R0_t}, a={a_t})", "D = I")

    print("\n[TOR] Anisotropic D (diffusion_tensor_jax, G=(0,0,1))")
    G_const = np.tile(np.array([0.0, 0.0, 1.0]), (Ntot, 1))
    D_aniso_j = diffusion_tensor_jax(jnp.asarray(G_const), eps=eps_aniso, delta=0.0)
    D_aniso = np.asarray(D_aniso_j)

    psi_num_a, _, _, _ = solve_manufactured_with_operator(
        xs, ys, zs, dx, dy, dz, Xq,
        inside3_tor, D_aniso,
        label="TOR / D_aniso"
    )
    relL2, Linf = compute_errors(psi_num_a, psi_exact, inside_flat, boundary_flat, label="TOR / D_aniso")
    results['tor']['aniso'] = (relL2, Linf)
    if plot:
        plot_midplane_slices(xs, ys, zs, psi_exact, psi_num_a, inside_flat,
                             f"Torus (R0={R0_t}, a={a_t})",
                             f"D_aniso (eps={eps_aniso:g})")

    if plot:
        plt.show()

    return results

def convergence_sweep(N_list=(16, 24, 32, 40, 48), eps_aniso=1e-3, plot=True):
    """
    Sweep over resolutions N and print a convergence table for each geometry.
    """
    rows = []

    for N in N_list:
        print("\n" + "#" * 30)
        print(f"#  N = {N}")
        print("#" * 30)
        res = run_all_geometries(N=N, eps_aniso=eps_aniso, plot=False)

        row = {
            'N': N,
            'box_iso':   res['box']['iso'][0],
            'box_aniso': res['box']['aniso'][0],
            'cyl_iso':   res['cyl']['iso'][0],
            'cyl_aniso': res['cyl']['aniso'][0],
            'tor_iso':   res['tor']['iso'][0],
            'tor_aniso': res['tor']['aniso'][0],
        }
        rows.append(row)

    # Pretty-print relative L2 convergence
    print("\n=== Convergence summary (relative L2 errors) ===")
    header = (" N   "
              "  box_iso    box_aniso   "
              "  cyl_iso    cyl_aniso   "
              "  tor_iso    tor_aniso")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['N']:3d}  "
              f"{r['box_iso']:.3e}  {r['box_aniso']:.3e}  "
              f"{r['cyl_iso']:.3e}  {r['cyl_aniso']:.3e}  "
              f"{r['tor_iso']:.3e}  {r['tor_aniso']:.3e}")
    
    if plot:
        fig, ax = plt.subplots(figsize=(8,6))
        for geom in ['box', 'cyl', 'tor']:
            for dtype, style in zip(['iso', 'aniso'], ['o-', 's--']):
                Ns = [r['N'] for r in rows]
                errs = [r[f'{geom}_{dtype}'] for r in rows]
                ax.loglog(Ns, errs, style, label=f"{geom} / {dtype}")
        ax.set_xlabel("N (grid points per axis)")
        ax.set_ylabel("Relative L2 error")
        ax.set_title("Convergence of Manufactured Solution Tests")
        ax.grid(True, which="both", ls="--")
        ax.legend()
        plt.savefig("mfs_convergence.png")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manufactured-solution tests for box, cylinder, and torus geometries."
    )
    parser.add_argument("--N", type=int, default=40,
                        help="Grid resolution per axis (default: 40)")
    parser.add_argument("--eps", type=float, default=1e-3,
                        help="Anisotropy parameter eps for diffusion_tensor_jax (default: 1e-3)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plots")
    parser.add_argument('--no-sweep', dest='sweep', action='store_false', help="Run a single N instead of a convergence sweep.")
    parser.add_argument("--Ns", type=str, default="16,24,32,40,48",
                        help="Comma-separated list of N values for the sweep (used with --sweep).")
    args = parser.parse_args()

    if args.sweep:
        N_list = [int(s) for s in args.Ns.split(",")]
        convergence_sweep(N_list=N_list, eps_aniso=args.eps)
    else:
        run_all_geometries(N=args.N, eps_aniso=args.eps, plot=not args.no_plot)
