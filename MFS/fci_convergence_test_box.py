#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Box convergence test for the FCI anisotropic diffusion solver
using a simple analytic harmonic solution on a full Cartesian box.

We solve (with D = I everywhere):

    -div( D ∇u ) = 0  in Ω = [0,1]^3

with Dirichlet boundary conditions taken from the analytic solution

    u_exact(x,y,z) = x^2 - y^2

which satisfies Δu_exact = 0 in continuous space.

This lets us test whether the *discrete* operator approximates the
Laplacian with the expected order of accuracy when the geometry is
a simple box (no masking complications).

We also:
  * compute the manufactured residual: A_pde[u_exact],
  * visualize a mid-plane slice of u_exact, u_num, and relative error,
  * sanity-check diffusion_tensor(G, eps, delta) for G = (0,0,1)
    against the identity tensor D = I.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, cg

from solve_flux_psi_fci import diffusion_tensor, make_linear_operator


def analytic_solution(X, Y, Z):
    """
    Analytic harmonic solution u(x,y,z) = x^2 - y^2.
    """
    return X**2 - Y**2


def run_box_convergence_test(
    N_list=(16, 24, 32, 40, 48),
    cg_tol=1e-8,
    cg_maxit=2000,
):
    print("# 3D box convergence test with D = I")
    print("# Domain: [0,1]^3, u_exact = x^2 - y^2, Dirichlet on all faces")
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
    manuf_res_L2 = []

    # For plotting slices later, we keep the finest-grid fields
    slice_data = None

    for N in N_list:
        nx = ny = nz = int(N)

        t_build_start = time.perf_counter()

        # Uniform box [0,1]^3
        xs = np.linspace(0.0, 1.0, nx)
        ys = np.linspace(0.0, 1.0, ny)
        zs = np.linspace(0.0, 1.0, nz)

        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        dz = zs[1] - zs[0]
        voxel = min(dx, dy, dz)

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        # Full box: everything inside
        inside_mask = np.ones_like(X, dtype=bool)
        inside_flat = inside_mask.ravel(order="C")

        Ntot = nx * ny * nz

        # Coordinates flattened
        X_flat = X.ravel(order="C")
        Y_flat = Y.ravel(order="C")
        Z_flat = Z.ravel(order="C")

        # Analytic solution on full grid
        u_exact_full = analytic_solution(X_flat, Y_flat, Z_flat)

        # D = I everywhere
        D = np.zeros((Ntot, 3, 3), dtype=float)
        D[:, 0, 0] = 1.0
        D[:, 1, 1] = 1.0
        D[:, 2, 2] = 1.0

        # Build matrix-free PDE operator A_pde and deep-interior mask
        A_pde, deep_inside = make_linear_operator(
            nx, ny, nz, dx, dy, dz,
            inside_flat,
            D,
        )

        # Dirichlet boundary on *all* six faces from u_exact
        boundary3 = np.zeros((nx, ny, nz), dtype=bool)
        boundary3[0, :, :] = True
        boundary3[-1, :, :] = True
        boundary3[:, 0, :] = True
        boundary3[:, -1, :] = True
        boundary3[:, :, 0] = True
        boundary3[:, :, -1] = True
        boundary_flat = boundary3.ravel(order="C")

        # sanity: deep_inside should be exactly "not boundary"
        # but we won't rely on that; we'll always use deep_inside
        fixed = boundary_flat.copy()
        val = np.zeros(Ntot, dtype=float)
        val[fixed] = u_exact_full[fixed]

        # Free unknowns: deep interior and not fixed
        free = deep_inside & (~fixed)
        if not np.any(free):
            raise RuntimeError(f"No free nodes for N={N}")

        u_fixed_full = np.zeros(Ntot, dtype=float)
        u_fixed_full[fixed] = val[fixed]

        # Laplace equation: A_pde[u] = 0, u|boundary = u_exact
        # So for u = u_fixed + u_free, we have:
        # A_pde[u_free] = -A_pde[u_fixed]
        F0_full = A_pde @ u_fixed_full
        b_free = -F0_full[free]

        Nfree = int(free.sum())

        def matvec_free(u_free):
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

        # Manufactured residual: A_pde[u_exact]
        A_u_exact = A_pde @ u_exact_full
        manuf_res = A_u_exact[deep_inside]  # only deep interior
        manuf_res_L2_val = np.linalg.norm(manuf_res) / np.sqrt(manuf_res.size)
        print(f"    N={N:3d} manufactured residual L2 = {manuf_res_L2_val:.3e}")

        # Solve with CG
        t_solve_start = time.perf_counter()

        cg_iter = 0

        def _cg_callback(_xk):
            nonlocal cg_iter
            cg_iter += 1

        u_free, info = cg(
            A_eff, b_free,
            rtol=cg_tol,
            maxiter=cg_maxit,
            callback=_cg_callback,
        )
        t_solve_end = time.perf_counter()
        t_solve = t_solve_end - t_solve_start

        if info != 0:
            print(f"[WARN] N={N} CG returned info={info} (0 means full convergence).")

        u_num_full = u_fixed_full.copy()
        u_num_full[free] = u_free

        # Residual check on free unknowns (should match CG's internal)
        res_free = matvec_free(u_free) - b_free
        res_rel = np.linalg.norm(res_free) / max(np.linalg.norm(b_free), 1e-30)

        # Error on deep interior nodes
        err = u_num_full[deep_inside] - u_exact_full[deep_inside]
        L2_err = np.linalg.norm(err) / np.sqrt(err.size)
        Linf_err = np.max(np.abs(err))

        print("{:6d} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:8d}  (res_rel={:.2e})".format(
            N, voxel, L2_err, Linf_err, t_build, t_solve, cg_iter, res_rel)
        )

        Ns.append(N)
        hs.append(voxel)
        L2_errs.append(L2_err)
        Linf_errs.append(Linf_err)
        build_times.append(t_build)
        solve_times.append(t_solve)
        cg_its.append(cg_iter)
        manuf_res_L2.append(manuf_res_L2_val)

        # Keep finest grid fields for slice plots
        if N == N_list[-1]:
            u_exact_3d = u_exact_full.reshape(nx, ny, nz, order="C")
            u_num_3d = u_num_full.reshape(nx, ny, nz, order="C")
            slice_data = (xs, ys, zs, u_exact_3d, u_num_3d)

    # Pack results
    results = {
        "N": np.asarray(Ns),
        "h_min": np.asarray(hs),
        "L2_err": np.asarray(L2_errs),
        "Linf_err": np.asarray(Linf_errs),
        "t_build": np.asarray(build_times),
        "t_solve": np.asarray(solve_times),
        "cg_iters": np.asarray(cg_its),
        "manuf_L2": np.asarray(manuf_res_L2),
        "slice_data": slice_data,
    }

    make_plots(results)
    sanity_check_diffusion_tensor()

    return results


def make_plots(res):
    """
    - Error vs h_min (log-log)
    - Manufactured residual vs h_min
    - Timing vs N_cells
    - Slice plots of analytic vs numeric vs relative error on finest grid
    """
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
    manuf = res["manuf_L2"]
    t_solve = res["t_solve"]
    N = res["N"]

    # --- Error vs h_min (log-log) ---
    fig1, ax1 = plt.subplots(figsize=(5.0, 4.0))

    mask_L2 = np.isfinite(L2) & (L2 > 0.0)
    mask_Linf = np.isfinite(Linf) & (Linf > 0.0)

    ax1.loglog(h[mask_L2], L2[mask_L2], "o-", label=r"$L_2$ error")
    ax1.loglog(h[mask_Linf], Linf[mask_Linf], "s--", label=r"$L_\infty$ error")

    # Fit observed order for L2
    if mask_L2.sum() >= 2:
        p = np.polyfit(np.log(h[mask_L2]), np.log(L2[mask_L2]), 1)
        order_L2 = p[0]
        txt = r"order $\approx {:.2f}$".format(order_L2)
        ax1.text(0.05, 0.05, txt, transform=ax1.transAxes,
                 ha="left", va="bottom")
    ax1.set_xlabel(r"$h$")
    ax1.set_ylabel(r"Error")
    ax1.set_title(r"Box test: error vs grid spacing")
    ax1.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax1.legend(loc="best")

    fig1.tight_layout()
    fig1.savefig("box_convergence_errors.png", dpi=300)

    # --- Manufactured residual vs h ---
    fig2, ax2 = plt.subplots(figsize=(5.0, 4.0))

    mask_manuf = manuf > 0.0
    ax2.loglog(h[mask_manuf], manuf[mask_manuf], "o-", label=r"$\|A u_\text{exact}\|_2$")

    if mask_manuf.sum() >= 2:
        q = np.polyfit(np.log(h[mask_manuf]), np.log(manuf[mask_manuf]), 1)
        order_m = q[0]
        txt = r"residual order $\approx {:.2f}$".format(order_m)
        ax2.text(0.05, 0.05, txt, transform=ax2.transAxes,
                 ha="left", va="bottom")

    ax2.set_xlabel(r"$h$")
    ax2.set_ylabel(r"Manufactured residual $L_2$")
    ax2.set_title(r"Box test: manufactured residual vs $h$")
    ax2.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax2.legend(loc="best")

    fig2.tight_layout()
    fig2.savefig("box_convergence_manufactured.png", dpi=300)

    # --- Timing vs number of cells ---
    fig3, ax3 = plt.subplots(figsize=(5.0, 4.0))

    N_cells = N**3
    ax3.loglog(N_cells, t_solve, "o-", label=r"solve time")

    mask_t = t_solve > 0.0
    if mask_t.sum() >= 2:
        r = np.polyfit(np.log(N_cells[mask_t]), np.log(t_solve[mask_t]), 1)
        alpha = r[0]
        txt = r"$t_\mathrm{solve} \propto N_\mathrm{cells}^{%.2f}$" % alpha
        ax3.text(0.05, 0.05, txt, transform=ax3.transAxes,
                 ha="left", va="bottom")

    ax3.set_xlabel(r"Number of cells $N_\mathrm{cells} = N^3$")
    ax3.set_ylabel(r"CPU time [s]")
    ax3.set_title(r"Box test: CG solve time vs grid size")
    ax3.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax3.legend(loc="best")

    fig3.tight_layout()
    fig3.savefig("box_convergence_timings.png", dpi=300)

    # --- Slice plots on finest grid ---
    if res["slice_data"] is not None:
        xs, ys, zs, u_exact_3d, u_num_3d = res["slice_data"]
        nx, ny, nz = u_exact_3d.shape

        k_mid = nz // 2
        Xs, Ys = np.meshgrid(xs, ys, indexing="ij")

        ue_slice = u_exact_3d[:, :, k_mid]
        un_slice = u_num_3d[:, :, k_mid]
        rel_err_slice = np.zeros_like(ue_slice)
        denom = np.maximum(np.abs(ue_slice), 1e-14)
        rel_err_slice = (un_slice - ue_slice) / denom

        vmin = min(ue_slice.min(), un_slice.min())
        vmax = max(ue_slice.max(), un_slice.max())

        fig4, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

        im0 = axes[0].imshow(ue_slice.T, origin="lower",
                             extent=[xs[0], xs[-1], ys[0], ys[-1]],
                             aspect="equal", vmin=vmin, vmax=vmax)
        axes[0].set_title("Analytic solution (z mid-plane)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig4.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(un_slice.T, origin="lower",
                             extent=[xs[0], xs[-1], ys[0], ys[-1]],
                             aspect="equal", vmin=vmin, vmax=vmax)
        axes[1].set_title("Numerical solution")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig4.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(rel_err_slice.T, origin="lower",
                             extent=[xs[0], xs[-1], ys[0], ys[-1]],
                             aspect="equal")
        axes[2].set_title("Relative error")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        fig4.colorbar(im2, ax=axes[2])

        fig4.suptitle("Box test: slice at z = {:.3f}".format(zs[k_mid]), y=1.03)
        fig4.savefig("box_slice_comparison.png", dpi=300)


def sanity_check_diffusion_tensor(eps=1e-4, delta=1e-3):
    """
    Simple sanity check on diffusion_tensor for G = (0,0,1):
    compare diag(D_aniso) vs diag(I).
    """
    G = np.zeros((1, 3), dtype=float)
    G[0, 2] = 1.0  # field along z

    D_aniso = diffusion_tensor(G, eps=eps, delta=delta)[0]
    D_iso = np.eye(3)

    diag_aniso = np.diag(D_aniso)
    diag_iso = np.diag(D_iso)

    print("\n[Sanity check] diffusion_tensor for G = (0,0,1):")
    print("  diag(D_aniso) =", diag_aniso)
    print("  diag(D_iso)   =", diag_iso)

    labels = ["xx", "yy", "zz"]
    x = np.arange(3)

    width = 0.35
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    ax.bar(x - width/2, diag_iso, width, label="D = I")
    ax.bar(x + width/2, diag_aniso, width, label="D_aniso(G)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Diagonal entry")
    ax.set_title("diffusion_tensor vs identity (G = (0,0,1))")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig("diffusion_tensor_sanity.png", dpi=300)
    
    plt.show()


if __name__ == "__main__":
    run_box_convergence_test()
