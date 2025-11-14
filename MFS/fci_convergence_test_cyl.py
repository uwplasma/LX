#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Laplace convergence test for the FCI diffusion operator with D = I.

We solve

    div(D ∇ψ) = 0  with  D = I

inside a cylindrical shell:

    R_axis < r < R_out,   z ∈ [-Lz/2, Lz/2],

with Dirichlet boundary conditions

    ψ = 0  at  r = R_axis,
    ψ = 1  at  r = R_out,

and on the top/bottom planes we impose the *same* radial profile

    ψ(r) = log(r/R_axis) / log(R_out/R_axis).

Thus the exact solution is

    ψ_exact(r,z) = log(r/R_axis) / log(R_out/R_axis),

independent of z and harmonic in 3D.

We embed the shell in a Cartesian box and use the same
make_linear_operator(D=I) machinery as in the FCI solver.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, cg

from solve_flux_psi_fci import make_linear_operator  # diffusion_tensor unused here


def run_convergence_test(
    N_list=(32, 48, 64, 96),
    R_axis=0.2,
    R_out=1.0,
    Lz=1.0,
    cg_tol=1e-8,
    cg_maxit=2000,
):
    print("# 3D Laplace convergence test in cylindrical shell")
    print("# R_axis = {:.3f}, R_out = {:.3f}, Lz = {:.3f}".format(R_axis, R_out, Lz))
    print("")
    print("{:>6s} {:>10s} {:>10s} {:>10s} {:>10s} {:>8s} {:>8s}".format(
        "N", "h_min", "L2_err", "Linf_err", "t_build", "t_solve", "CG_it")
    )

    Ns = []
    hs = []
    L2_errs = []
    Linf_errs = []
    build_times = []
    solve_times = []
    cg_its = []

    last_snapshot = None  # to keep data for 3D plot

    for N in N_list:
        nx = ny = nz = int(N)

        t_build_start = time.perf_counter()

        # Cartesian box that contains the cylinder
        xs = np.linspace(-R_out, R_out, nx)
        ys = np.linspace(-R_out, R_out, ny)
        zs = np.linspace(-0.5 * Lz, 0.5 * Lz, nz)

        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        dz = zs[1] - zs[0]
        voxel = min(dx, dy, dz)

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        R = np.sqrt(X**2 + Y**2)

        # Domain: *shell* R_axis < r < R_out
        inside_mask = (R > R_axis) & (R < R_out)

        # One-cell thick radial bands for inner / outer Dirichlet BCs
        h_band_vox = 0.5 * voxel

        axis_band = inside_mask & (np.abs(R - R_axis) <= h_band_vox)
        boundary_band = inside_mask & (np.abs(R - R_out) <= h_band_vox)

        # Top / bottom Dirichlet planes (use one-cell thick slabs)
        z_top = 0.5 * Lz
        z_bot = -0.5 * Lz
        z_band = 0.5 * dz

        top_band = inside_mask & (np.abs(Z - z_top) <= z_band)
        bot_band = inside_mask & (np.abs(Z - z_bot) <= z_band)

        # Make sure radial bands don't overlap (if very coarse)
        overlap = axis_band & boundary_band
        boundary_band[overlap] = False

        Ntot = nx * ny * nz

        # Flatten everything in C-order
        X_flat = X.ravel(order="C")
        Y_flat = Y.ravel(order="C")
        Z_flat = Z.ravel(order="C")
        R_flat = R.ravel(order="C")

        inside_flat = inside_mask.ravel(order="C")
        axis_flat = axis_band.ravel(order="C")
        boundary_flat = boundary_band.ravel(order="C")
        top_flat = top_band.ravel(order="C")
        bot_flat = bot_band.ravel(order="C")

        # Effective radii from the radial Dirichlet bands
        if np.any(axis_flat):
            R_axis_eff = float(np.mean(R_flat[axis_flat]))
        else:
            R_axis_eff = R_axis

        if np.any(boundary_flat):
            R_out_eff = float(np.mean(R_flat[boundary_flat]))
        else:
            R_out_eff = R_out

        # Build isotropic diffusion tensor D = I (everywhere inside)
        D = np.zeros((Ntot, 3, 3), dtype=float)
        D[:, 0, 0] = 1.0
        D[:, 1, 1] = 1.0
        D[:, 2, 2] = 1.0

        # Build PDE operator
        A_pde, deep_inside = make_linear_operator(
            nx, ny, nz, dx, dy, dz,
            inside_flat,
            D,
        )

        # Dirichlet lifting: ψ_full = ψ_free + ψ_fixed
        fixed = np.zeros(Ntot, dtype=bool)
        val = np.zeros(Ntot, dtype=float)

        # Helper: analytic ψ(r) everywhere inside shell
        psi_analytic_all = np.zeros(Ntot, dtype=float)
        shell_flat = inside_flat & (R_flat > R_axis_eff) & (R_flat < R_out_eff)
        psi_analytic_all[shell_flat] = np.log(R_flat[shell_flat] / R_axis_eff) / np.log(
            R_out_eff / R_axis_eff
        )
        psi_analytic_all = np.clip(psi_analytic_all, 0.0, 1.0)

        # Impose Dirichlet BCs on:
        #  - inner radial band
        #  - outer radial band
        #  - top plane
        #  - bottom plane
        for band_flat in (axis_flat, boundary_flat, top_flat, bot_flat):
            fixed[band_flat] = True
            val[band_flat] = psi_analytic_all[band_flat]

        # Free unknowns: deep interior & not fixed
        free = deep_inside & (~fixed)
        if not np.any(free):
            raise RuntimeError(f"No free nodes for N={N} (bands too fat or domain too small)")

        psi_fixed_full = np.zeros(Ntot, dtype=float)
        psi_fixed_full[fixed] = val[fixed]

        # RHS from Dirichlet lifting
        F0_full = A_pde @ psi_fixed_full
        b_free = -F0_full[free]

        Nfree = int(free.sum())

        def matvec_free(u_free: np.ndarray) -> np.ndarray:
            u_full = np.zeros(Ntot, dtype=float)
            u_full[free] = u_free
            Au_full = A_pde @ u_full
            return Au_full[free]

        A_eff = LinearOperator(
            (Nfree, Nfree),
            matvec=matvec_free,
            rmatvec=matvec_free,
            dtype=float,
        )

        t_build_end = time.perf_counter()
        t_build = t_build_end - t_build_start

        # Manufactured residual check: how well does A_pde annihilate ψ_exact?
        psi_manu = np.zeros(Ntot, dtype=float)
        psi_manu[shell_flat] = psi_analytic_all[shell_flat]
        res_full = A_pde @ psi_manu

        # consider interior nodes away from all boundaries for this check
        safety = 2.0 * voxel
        mask_resid = (
            deep_inside &
            (~fixed) &
            (R_flat > R_axis_eff + safety) &
            (R_flat < R_out_eff - safety) &
            (Z_flat > z_bot + safety) &
            (Z_flat < z_top - safety)
        )
        if np.any(mask_resid):
            res_interior = res_full[mask_resid]
            manuf_L2 = np.linalg.norm(res_interior) / np.sqrt(res_interior.size)
        else:
            manuf_L2 = np.nan
        print(f"    N={N:4d} manufactured residual L2 = {manuf_L2:.3e}")

        # ------------------------------------------------------------------
        # Solve with CG
        # ------------------------------------------------------------------
        t_solve_start = time.perf_counter()

        cg_iter = 0

        def _cg_callback(_xk):
            nonlocal cg_iter
            cg_iter += 1

        psi_free, info = cg(
            A_eff, b_free,
            rtol=cg_tol,
            maxiter=cg_maxit,
            callback=_cg_callback,
        )
        t_solve_end = time.perf_counter()
        t_solve = t_solve_end - t_solve_start

        if info != 0:
            print(f"[WARN] N={N} CG returned info={info} (0 means full convergence).")

        psi = np.array(psi_fixed_full)
        psi[free] = psi_free

        # Residual check on free unknowns for the computed solution
        res_free = matvec_free(psi_free) - b_free
        res_rel = np.linalg.norm(res_free) / max(np.linalg.norm(b_free), 1e-30)

        # ------------------------------------------------------------------
        # Compare to analytic solution in the shell interior
        # ------------------------------------------------------------------
        h_min = voxel
        safety = 2.0 * h_min

        mask_int = (
            deep_inside &
            (~fixed) &
            (R_flat > R_axis_eff + safety) &
            (R_flat < R_out_eff - safety) &
            (Z_flat > z_bot + safety) &
            (Z_flat < z_top - safety)
        )
        if np.any(mask_int):
            err = psi[mask_int] - psi_analytic_all[mask_int]
            L2_err = np.linalg.norm(err) / np.sqrt(err.size)
            Linf_err = np.max(np.abs(err))
        else:
            L2_err = np.nan
            Linf_err = np.nan

        print(
            "{:6d} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:8d}  (res_rel={:.2e})".format(
                N, h_min, L2_err, Linf_err, t_build, t_solve, cg_iter, res_rel
            )
        )

        Ns.append(N)
        hs.append(h_min)
        L2_errs.append(L2_err)
        Linf_errs.append(Linf_err)
        build_times.append(t_build)
        solve_times.append(t_solve)
        cg_its.append(cg_iter)

        # Save snapshot for 3D visualization for the *last* N
        last_snapshot = {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "X": X,
            "Y": Y,
            "Z": Z,
            "inside": inside_mask,
            "psi_num": psi.reshape(nx, ny, nz, order="C"),
            "psi_exact": psi_analytic_all.reshape(nx, ny, nz, order="C"),
            "axis_band": axis_band,
            "boundary_band": boundary_band,
        }

    # Pack results
    Ns = np.asarray(Ns, dtype=float)
    hs = np.asarray(hs, dtype=float)
    L2_errs = np.asarray(L2_errs, dtype=float)
    Linf_errs = np.asarray(Linf_errs, dtype=float)
    build_times = np.asarray(build_times, dtype=float)
    solve_times = np.asarray(solve_times, dtype=float)
    cg_its = np.asarray(cg_its, dtype=float)

    results = {
        "N": Ns,
        "h_min": hs,
        "L2_err": L2_errs,
        "Linf_err": Linf_errs,
        "t_build": build_times,
        "t_solve": solve_times,
        "cg_iters": cg_its,
    }

    make_publication_plots(results)
    if last_snapshot is not None:
        make_3d_diagnostic_plot(last_snapshot)

    return results


def make_publication_plots(res):
    """Error vs h and timing vs N_cells, publication style."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "figure.dpi": 120,
    })

    h = res["h_min"]
    L2 = res["L2_err"]
    Linf = res["Linf_err"]
    t_solve = res["t_solve"]
    N = res["N"]

    # --- Error vs h_min (log-log) ---
    fig1, ax1 = plt.subplots(figsize=(5.0, 4.0))

    mask_L2 = np.isfinite(L2) & (L2 > 0.0)
    mask_Linf = np.isfinite(Linf) & (Linf > 0.0)

    ax1.loglog(h[mask_L2], L2[mask_L2], "o-", label=r"$L_2$ error")
    ax1.loglog(h[mask_Linf], Linf[mask_Linf], "s--", label=r"$L_\infty$ error")

    if mask_L2.sum() >= 2:
        p = np.polyfit(np.log(h[mask_L2]), np.log(L2[mask_L2]), 1)
        order_L2 = p[0]
        txt = r"observed order $p \approx {:.2f}$".format(order_L2)
        ax1.text(0.05, 0.05, txt, transform=ax1.transAxes,
                 ha="left", va="bottom")

    ax1.set_xlabel(r"$h_{\min}$")
    ax1.set_ylabel(r"Error")
    ax1.set_title(r"3D Laplace: cylindrical shell convergence")
    ax1.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax1.legend(loc="best")

    fig1.tight_layout()
    fig1.savefig("fci_3d_laplace_convergence_errors.png", dpi=300)

    # --- Timing vs number of cells ---
    fig2, ax2 = plt.subplots(figsize=(5.0, 4.0))

    N_cells = N**3
    ax2.loglog(N_cells, t_solve, "o-", label=r"solve time")

    mask_t = t_solve > 0.0
    if mask_t.sum() >= 2:
        q = np.polyfit(np.log(N_cells[mask_t]), np.log(t_solve[mask_t]), 1)
        alpha = q[0]
        txt = r"$t_\mathrm{solve} \propto N_\mathrm{cells}^{%.2f}$" % alpha
        ax2.text(0.05, 0.05, txt, transform=ax2.transAxes,
                 ha="left", va="bottom")

    ax2.set_xlabel(r"Number of cells $N_\mathrm{cells} = N^3$")
    ax2.set_ylabel(r"CPU time [s]")
    ax2.set_title(r"CG solve time vs grid size (3D Laplace)")
    ax2.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax2.legend(loc="best")

    fig2.tight_layout()
    fig2.savefig("fci_3d_laplace_convergence_timings.png", dpi=300)


def make_3d_diagnostic_plot(snap):
    """
    Visual 3D diagnostics for the highest-resolution run.

    - Midplane z≈0 slice: ψ_num, ψ_exact, and difference.
    - Sparse 3D scatter colored by ψ_num inside the shell.
    """
    X = snap["X"]
    Y = snap["Y"]
    Z = snap["Z"]
    psi_num = snap["psi_num"]
    psi_exact = snap["psi_exact"]
    inside = snap["inside"]
    axis_band = snap["axis_band"]
    boundary_band = snap["boundary_band"]

    nx, ny, nz = psi_num.shape
    k_mid = nz // 2

    # --- 2D slices at midplane ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    # Numerical
    im0 = axes[0].pcolormesh(
        X[:, :, k_mid], Y[:, :, k_mid],
        psi_num[:, :, k_mid],
        shading="auto"
    )
    axes[0].set_aspect("equal")
    axes[0].set_title(r"$\psi_{\mathrm{num}}(x,y,z\approx 0)$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    # Analytic
    im1 = axes[1].pcolormesh(
        X[:, :, k_mid], Y[:, :, k_mid],
        psi_exact[:, :, k_mid],
        shading="auto"
    )
    axes[1].set_aspect("equal")
    axes[1].set_title(r"$\psi_{\mathrm{exact}}(x,y,z\approx 0)$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    # Difference
    diff = psi_num[:, :, k_mid] - psi_exact[:, :, k_mid]
    im2 = axes[2].pcolormesh(
        X[:, :, k_mid], Y[:, :, k_mid],
        diff,
        shading="auto"
    )
    axes[2].set_aspect("equal")
    axes[2].set_title(r"$\psi_{\mathrm{num}} - \psi_{\mathrm{exact}}$ at midplane")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2])

    fig.savefig("fci_3d_laplace_midplane_slices.png", dpi=300)

    # --- Sparse 3D scatter of ψ_num inside the shell ---
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig3 = plt.figure(figsize=(6, 5))
    ax3 = fig3.add_subplot(111, projection="3d")

    mask_shell = inside & (~axis_band) & (~boundary_band)
    # Subsample for plotting
    idx = np.where(mask_shell.ravel(order="C"))[0]
    if idx.size > 5000:
        idx = np.random.choice(idx, size=5000, replace=False)

    x_s = X.ravel(order="C")[idx]
    y_s = Y.ravel(order="C")[idx]
    z_s = Z.ravel(order="C")[idx]
    psi_s = psi_num.ravel(order="C")[idx]

    sc = ax3.scatter(x_s, y_s, z_s, c=psi_s, s=3, alpha=0.6)
    fig3.colorbar(sc, ax=ax3, label=r"$\psi_{\mathrm{num}}$")

    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.set_title("3D view of ψ inside cylindrical shell (sparse points)")
    ax3.view_init(elev=20, azim=45)

    fig3.tight_layout()
    fig3.savefig("fci_3d_laplace_3d_shell.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_convergence_test()
