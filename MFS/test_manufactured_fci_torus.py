#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Second manufactured-solution test for the FCI operator on a toroidal domain.

This test mimics the structure of the full FCI solver:

  * Analytic torus boundary (major radius R0, minor radius a) in 3D.
  * Domain interior defined via inside_mask_from_surface(P, N, Xq).
  * Boundary band (ψ ≈ 1) near the torus surface.
  * Axis band (ψ ≈ 0) near the magnetic axis circle.
  * Strongly anisotropic diffusion with an FCI parallel operator built from
    a toroidal "magnetic field" B = e_phi.
  * Perpendicular diffusion via a 7-point Cartesian Laplacian.
  * Manufactured exact solution ψ_exact(r_minor) = (r_minor / a)^2, where
        r_minor = sqrt((sqrt(x^2 + y^2) - R0)^2 + z^2),
    so ψ_exact = 0 on the axis and ψ_exact = 1 on the torus surface.

We construct a *discrete* manufactured solution by:
  1. Building the full operator A_pde_jax with FCI connectivity.
  2. Computing f_full = A_pde_jax(ψ_exact) on the grid.
  3. Solving A ψ = f with Dirichlet bands (ψ fixed to ψ_exact on axis+boundary).
  4. Comparing ψ_num with ψ_exact in the free interior.

If the FCI machinery is correct, ψ_num ≈ ψ_exact to roundoff, even in the
strongly anisotropic regime.
"""

import argparse
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from solve_flux_psi_fci import (
    build_fci_connectivity_chunked,
    make_fci_operator_jax,
    cg_jax,
    inside_mask_from_surface,
)


# ----------------------------------------------------------------------
# Analytic torus geometry
# ----------------------------------------------------------------------

def torus_surface_points_normals(R0=1.5, a=0.5,
                                 ntheta=128, nphi=256):
    """
    Build a point cloud (P, N) on an axisymmetric torus:

        X(θ,φ) = [(R0 + a cosθ) cosφ,
                  (R0 + a cosθ) sinφ,
                  a sinθ]

    Normals are taken as the vector from the axis circle at radius R0 to
    the torus surface point, and then normalized.
    """
    theta = np.linspace(0.0, 2.0*np.pi, ntheta, endpoint=False)
    phi   = np.linspace(0.0, 2.0*np.pi, nphi,   endpoint=False)

    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")

    R = R0 + a * np.cos(Theta)
    X = R * np.cos(Phi)
    Y = R * np.sin(Phi)
    Z = a * np.sin(Theta)

    # Axis circle at radius R0 in the midplane (z=0)
    Xc = R0 * np.cos(Phi)
    Yc = R0 * np.sin(Phi)
    Zc = np.zeros_like(Z)

    # "Radial" vector from axis circle to torus point
    Nx = X - Xc
    Ny = Y - Yc
    Nz = Z - Zc

    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    N = np.stack([Nx, Ny, Nz], axis=-1).reshape(-1, 3)
    N_norm = np.linalg.norm(N, axis=1, keepdims=True)
    N /= np.maximum(N_norm, 1e-14)

    return P, N


# ----------------------------------------------------------------------
# Manufactured exact solution on torus-like domain
# ----------------------------------------------------------------------

def psi_exact_torus(X, R0=1.5, a=0.5):
    """
    ψ_exact = (r_minor / a)^2
    where r_minor is the distance to the magnetic axis circle in R-Z:

      r_minor = sqrt((sqrt(x^2 + y^2)-R0)^2 + z^2)

    On the axis: r_minor = 0 => ψ_exact=0.
    On the torus surface: r_minor = a => ψ_exact=1.
    """
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    R = np.sqrt(x*x + y*y)
    r_minor = np.sqrt((R - R0)**2 + z*z)
    return (r_minor / a)**2


# ----------------------------------------------------------------------
# Analytic "B field" used by FCI: pure toroidal B = e_phi
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
    ephi = jnp.stack([-y, x, 0.0 * z], axis=-1)
    ephi_norm = jnp.maximum(r, 1e-8)
    B = ephi / ephi_norm[..., None]
    B = jnp.where(r[..., None] > 1e-8, B, jnp.zeros_like(B))
    return B


# ----------------------------------------------------------------------
# Grid and helper masks
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


# ----------------------------------------------------------------------
# Main manufactured test
# ----------------------------------------------------------------------

def solve_fci_manufactured_torus(
    N=96,
    R0=1.5,
    a=0.5,
    eps=1e-2,
    band_h=1.5,
    nfp=1,
    fci_nsteps=32,
    plot=True,
):
    """
    Full-solver-style manufactured test on a torus:

      - Builds torus boundary, inside mask, axis & boundary bands.
      - Builds FCI operator with strong anisotropy.
      - Constructs discrete manufactured RHS f = A ψ_exact.
      - Solves A ψ = f with Dirichlet bands matching ψ_exact.
      - Returns relative L2 and Linf errors in the free interior.
    """
    print("=" * 70)
    print(f"FCI torus manufactured test: N={N}, R0={R0}, a={a}, eps={eps}")
    print("=" * 70)

    # --- 1. Build torus surface and analytic axis ---------------------
    P_surf, N_surf = torus_surface_points_normals(R0=R0, a=a,
                                                  ntheta=128, nphi=256)

    # Axis: circle in midplane at radius R0
    n_axis_pts = 512
    phis_axis = np.linspace(0.0, 2.0*np.pi, n_axis_pts, endpoint=False)
    R_axis = R0 * np.ones_like(phis_axis)
    Z_axis = np.zeros_like(phis_axis)
    axis_pts = np.stack([
        R_axis * np.cos(phis_axis),
        R_axis * np.sin(phis_axis),
        Z_axis,
    ], axis=1)

    # --- 2. Build Cartesian grid bounding the torus -------------------

    # Torus spans roughly:
    #   R ∈ [R0 - a, R0 + a],  z ∈ [-a, a]
    # x,y ∈ [-(R0 + a), +(R0 + a)]
    Rmax = R0 + a
    bounds = (
        (-(Rmax + 0.1*a),  Rmax + 0.1*a),
        (-(Rmax + 0.1*a),  Rmax + 0.1*a),
        (-a - 0.1*a,       a + 0.1*a),
    )
    xs, ys, zs, dx, dy, dz, Xq = build_cartesian_grid(bounds, N)
    nx, ny, nz = len(xs), len(ys), len(zs)
    Ntot = nx * ny * nz
    voxel = min(dx, dy, dz)

    print(f"[GRID] nx,ny,nz = {nx},{ny},{nz} (Ntot={Ntot})")
    print(f"[GRID] dx,dy,dz ≈ {dx:.3g}, {dy:.3g}, {dz:.3g}")

    # --- 3. Inside mask from torus surface ----------------------------

    inside_flat, nn_idx, signed_dist = inside_mask_from_surface(
        P_surf, N_surf, Xq
    )
    inside3 = inside_flat.reshape(nx, ny, nz)

    if not np.any(inside_flat):
        raise RuntimeError("Inside mask is empty; check torus vs grid bounds.")

    print(f"[DOMAIN] Inside nodes: {inside_flat.sum()} / {Ntot}")

    # --- 4. Build boundary and axis bands -----------------------------

    # Boundary band: distance to torus surface |signed_dist| <= h_band
    h_band_vox = float(band_h) * voxel
    band = (inside_flat & (np.abs(signed_dist) <= h_band_vox))
    band3 = band.reshape(nx, ny, nz)

    # Axis band: near the analytic axis points
    inside_idx = np.where(inside_flat)[0]
    X_inside = Xq[inside_idx]

    nbrs_axis = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(axis_pts)
    d_axis, _ = nbrs_axis.kneighbors(X_inside)
    d_axis = d_axis[:, 0]

    # Choose the closest few percent of inside nodes as axis band
    frac_axis = 0.02
    n_axis_nodes = max(20, int(frac_axis * inside_idx.size))
    order = np.argsort(d_axis)
    chosen_inside = inside_idx[order[:n_axis_nodes]]

    axis_band = np.zeros_like(inside_flat, dtype=bool)
    axis_band[chosen_inside] = True
    axis_band3 = axis_band.reshape(nx, ny, nz)

    axis_band_radius_eff = float(d_axis[order[:n_axis_nodes]].max())

    # Avoid overlap: if a node is both in boundary band and axis band,
    # keep it as axis band and remove from boundary band.
    overlap = band & axis_band
    axis_band[overlap] = True
    band[overlap] = False
    band3 = band.reshape(nx, ny, nz)

    print(f"[BANDS] #boundary band nodes: {band.sum()} / {Ntot}")
    print(f"[BANDS] #axis band nodes    : {axis_band.sum()} / {Ntot}")
    print(f"[BANDS] boundary band width ≈ {h_band_vox:.3e}")
    print(f"[BANDS] axis band radius    ≈ {axis_band_radius_eff:.3e}")

    # --- 5. Build FCI connectivity -----------------------------------

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
    print(f"[FCI] valid connectivity nodes: {fci.valid.sum()} / {inside_flat.sum()}")

    # --- 6. Core region for parallel operator -------------------------

    core3 = np.zeros_like(inside3, dtype=bool)
    core3[1:-1, 1:-1, 1:-1] = True
    core3 &= inside3
    core3 &= ~band3
    core3 &= ~axis_band3
    core_flat = core3.ravel(order="C")

    if core_flat.sum() < 10:
        print(f"[WARN] Core region has only {core_flat.sum()} nodes.")

    # --- 7. Build FCI operator (anisotropic) --------------------------

    kappa_par = 1.0
    kappa_perp = eps**2

    A_pde_jax, deep_inside = make_fci_operator_jax(
        nx, ny, nz,
        xs, ys, zs,
        inside_flat,
        fci,
        core_mask=core_flat,
        kappa_par=kappa_par,
        kappa_perp=kappa_perp,
    )

    # --- 8. Manufactured exact solution -------------------------------

    psi_exact = psi_exact_torus(Xq, R0=R0, a=a)  # shape (Ntot,)
    psi_exact_j = jnp.asarray(psi_exact, dtype=jnp.float64)

    # Discrete manufactured RHS: f = A ψ_exact
    f_full_j = A_pde_jax(psi_exact_j)
    f_full = np.asarray(f_full_j)

    # --- 9. Dirichlet bands and lifting -------------------------------

    fixed = (band | axis_band)
    val = np.zeros(Ntot, dtype=float)
    val[fixed] = psi_exact[fixed]

    free = deep_inside & (~fixed)
    if not np.any(free):
        raise RuntimeError("No free nodes in manufactured torus test.")

    print(f"[SOLVE] Free nodes: {free.sum()} / {Ntot}")

    psi_fixed_full = np.zeros(Ntot, dtype=float)
    psi_fixed_full[fixed] = val[fixed]
    psi_fixed_j = jnp.asarray(psi_fixed_full, dtype=jnp.float64)

    Afixed_j = A_pde_jax(psi_fixed_j)
    Afixed_full = np.asarray(Afixed_j)

    # We want A ψ = f_full with ψ = ψ_free + ψ_fixed.
    # => A ψ_free = f_full - A ψ_fixed.
    b_full = f_full - Afixed_full
    b_free = b_full[free]

    free_mask = free.copy()

    def matvec_free_jax(u_free_j):
        u_full = jnp.zeros(Ntot, dtype=jnp.float64)
        u_full = u_full.at[free_mask].set(u_free_j)
        Au_full = A_pde_jax(u_full)
        return Au_full[free_mask]

    # --- 10. Solve for ψ_free using JAX CG ----------------------------

    print("[SOLVE] Solving A ψ = f (JAX CG) ...")
    b_free_j = jnp.asarray(b_free, dtype=jnp.float64)
    psi_free_j, res_norm = cg_jax(
        matvec_free_jax, b_free_j,
        tol=1e-10, maxiter=5000
    )
    psi_free = np.asarray(psi_free_j)
    print(f"[SOLVE] CG finished with ||r||₂ ≈ {float(res_norm):.3e}")

    psi_num = np.array(psi_fixed_full)
    psi_num[free] = psi_free

    # --- 11. Error metrics --------------------------------------------

    interior = inside_flat & (~fixed)
    err = psi_num - psi_exact

    rel_L2 = np.linalg.norm(err[interior]) / np.linalg.norm(psi_exact[interior])
    Linf = np.max(np.abs(err[interior]))
    print(f"[ERROR] relative L2 error (interior) = {rel_L2:.3e}")
    print(f"[ERROR] Linf error (interior)        = {Linf:.3e}")

    # --- 12. Publication-quality plots --------------------------------

    if plot:
        psi_num3   = psi_num.reshape(nx, ny, nz)
        psi_exact3 = psi_exact.reshape(nx, ny, nz)
        err3       = err.reshape(nx, ny, nz)
        inside3    = inside_flat.reshape(nx, ny, nz)

        # Midplane slice y ≈ 0 (j = ny//2)
        j_mid = ny // 2
        x_slice = xs
        z_slice = zs
        X_slice, Z_slice = np.meshgrid(x_slice, z_slice, indexing="ij")

        mask_slice = inside3[:, j_mid, :]

        # Shared extent for imshow
        extent = [xs[0], xs[-1], zs[0], zs[-1]]

        plt.rcParams.update({
            "figure.dpi": 150,
            "font.size": 10,
        })

        # --- Figure 1: ψ_exact, ψ_num, error on x–z plane --------------
        fig1, axes = plt.subplots(1, 3, figsize=(11, 3.5),
                                  constrained_layout=True)

        # ψ_exact slice
        im0 = axes[0].imshow(
            np.where(mask_slice, psi_exact3[:, j_mid, :], np.nan).T,
            origin="lower", extent=extent, aspect="equal"
        )
        axes[0].set_title(r"$\psi_{\rm exact}$ (midplane)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("z")
        cbar0 = fig1.colorbar(im0, ax=axes[0], shrink=0.85)
        cbar0.set_label(r"$\psi$")

        # ψ_num slice
        im1 = axes[1].imshow(
            np.where(mask_slice, psi_num3[:, j_mid, :], np.nan).T,
            origin="lower", extent=extent, aspect="equal"
        )
        axes[1].set_title(r"$\psi_{\rm num}$ (midplane)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("z")
        cbar1 = fig1.colorbar(im1, ax=axes[1], shrink=0.85)
        cbar1.set_label(r"$\psi$")

        # Error slice
        im2 = axes[2].imshow(
            np.where(mask_slice, err3[:, j_mid, :], np.nan).T,
            origin="lower", extent=extent, aspect="equal"
        )
        axes[2].set_title(r"$\psi_{\rm num}-\psi_{\rm exact}$ (midplane)")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("z")
        vmax_err = np.nanmax(np.abs(err3[:, j_mid, :]))
        if vmax_err == 0.0:
            vmax_err = 1e-16
        im2.set_clim(-vmax_err, vmax_err)
        cbar2 = fig1.colorbar(im2, ax=axes[2], shrink=0.85)
        cbar2.set_label(r"error")

        fig1.suptitle("FCI torus manufactured solution: midplane slices",
                      y=1.02)

        # Overlay axis and boundary intersection with midplane
        tol_y = 0.5 * dy
        mask_axis_mid = np.abs(axis_pts[:, 1]) < tol_y
        X_axis_mid = axis_pts[mask_axis_mid, 0]
        Z_axis_mid = axis_pts[mask_axis_mid, 2]

        mask_P_mid = np.abs(P_surf[:, 1]) < tol_y
        X_bnd_mid = P_surf[mask_P_mid, 0]
        Z_bnd_mid = P_surf[mask_P_mid, 2]

        for ax in axes:
            ax.scatter(X_bnd_mid, Z_bnd_mid, s=5, c="k", marker=".", alpha=0.8,
                       label="boundary")
            ax.scatter(X_axis_mid, Z_axis_mid, s=15, c="white", marker="o",
                       edgecolors="k", linewidths=0.5, label="axis")

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig1.legend(handles, labels, loc="upper right", frameon=True)

        fig1.savefig("fci_torus_manufactured_midplane.png", dpi=300)

        # --- Figure 2: 1D profile along x-axis (y=0,z=0) ----------------
        # Choose the grid line closest to y=0, z=0
        j_y = np.argmin(np.abs(ys - 0.0))
        k_z = np.argmin(np.abs(zs - 0.0))

        mask_line = inside3[:, j_y, k_z]
        x_line = xs
        psi_exact_line = psi_exact3[:, j_y, k_z]
        psi_num_line   = psi_num3[:, j_y, k_z]
        err_line       = err3[:, j_y, k_z]

        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3.5),
                                 constrained_layout=True)

        ax2[0].plot(x_line[mask_line], psi_exact_line[mask_line], "k-", lw=1.5,
                    label=r"$\psi_{\rm exact}$")
        ax2[0].plot(x_line[mask_line], psi_num_line[mask_line], "r--", lw=1.2,
                    label=r"$\psi_{\rm num}$")
        ax2[0].set_xlabel("x  (y=0,z≈0)")
        ax2[0].set_ylabel(r"$\psi$")
        ax2[0].set_title("1D profile through midplane axis")
        ax2[0].legend(frameon=True)

        ax2[1].plot(x_line[mask_line], np.abs(err_line[mask_line]), "b-",
                    lw=1.2)
        ax2[1].set_yscale("log")
        ax2[1].set_xlabel("x  (y=0,z≈0)")
        ax2[1].set_ylabel(r"$|\psi_{\rm num}-\psi_{\rm exact}|$")
        ax2[1].set_title("Absolute error (log scale)")

        fig2.suptitle("FCI torus manufactured solution: 1D midplane profile",
                      y=1.02)
        fig2.savefig("fci_torus_manufactured_profile.png", dpi=300)

        # --- Figure 3: Error histogram in the interior ------------------
        fig3, ax3 = plt.subplots(1, 1, figsize=(4.5, 3.5),
                                 constrained_layout=True)
        ax3.hist(np.abs(err[interior]), bins=80)
        ax3.set_yscale("log")
        ax3.set_xlabel(r"$|\psi_{\rm num}-\psi_{\rm exact}|$")
        ax3.set_ylabel("count")
        ax3.set_title("Error histogram (free interior nodes)")
        fig3.savefig("fci_torus_manufactured_error_hist.png", dpi=300)

        plt.show()

    return rel_L2, Linf


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full-solver-style manufactured FCI test on a torus."
    )
    parser.add_argument("--N", type=int, default=64,
                        help="Grid resolution per axis (default: 96)")
    parser.add_argument("--R0", type=float, default=1.5,
                        help="Torus major radius R0 (default: 1.5)")
    parser.add_argument("--a", type=float, default=0.5,
                        help="Torus minor radius a (default: 0.5)")
    parser.add_argument("--eps", type=float, default=1e-3,
                        help="Perpendicular diffusivity scale; kappa_perp=eps^2 (default: 1e-2)")
    parser.add_argument("--band-h", type=float, default=3.0,
                        help="Boundary band half-thickness in units of voxel (default: 1.5)")
    parser.add_argument("--nfp", type=int, default=1,
                        help="Number of field periods for FCI (default: 1 for B=e_phi)")
    parser.add_argument("--fci-nsteps", type=int, default=8,
                        help="Number of RK2 steps per Δφ for FCI (default: 32)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plotting and only print errors.")
    args = parser.parse_args()

    relL2, Linf = solve_fci_manufactured_torus(
        N=args.N,
        R0=args.R0,
        a=args.a,
        eps=args.eps,
        band_h=args.band_h,
        nfp=args.nfp,
        fci_nsteps=args.fci_nsteps,
        plot=(not args.no_plot),
    )

    print("\n=== FCI torus manufactured test summary ===")
    print(f"relative L2 error (interior) = {relL2:.3e}")
    print(f"Linf error (interior)        = {Linf:.3e}")
