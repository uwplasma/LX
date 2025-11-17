#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###
###
# Example usage:
#   clear; clear; python optimize_surface_qs.py \
#       --xyz_csv ~/local/LX/inputs/wout_precise_QA.csv \
#       --normals_csv ~/local/LX/inputs/wout_precise_QA_normals.csv \
#       --mfs_script ~/local/LX/main.py \
#       --psi_script ~/local/LX/solve_flux_psi_fci_cyl.py \
#       --optimizer least_squares \
#       --verbose
###
"""
Automated shape optimisation for quasisymmetry using the MFS and FCI solvers.

This script ties together three separate components:

1. **Method of Fundamental Solutions (MFS)**: solves the Neumann problem
   ∇²φ=0 inside a prescribed boundary.  The MFS solver is provided by
   the script ``main.py`` in this repository.  Given a set of boundary
   points and normals on the surface, it writes a checkpoint file
   (``*_solution.npz``) containing the potential φ, its gradient
   ∇φ, the source locations, and other metadata.

2. **Flux–coordinate independent (FCI) ψ solver**: constructs a flux-like
   coordinate ψ by solving an anisotropic diffusion equation along the
   magnetic field lines of ∇φ.  This solver is provided by
   ``solve_flux_psi_fci_cyl.py``.  It reads the MFS checkpoint and
   writes a snapshot file containing ψ on a cylindrical or Cartesian grid.

3. **Quasisymmetry triple-product diagnostic**: evaluates the
   normalized triple-product error

       f̂_T(ψ) = ⟨R⟩² ⟨|f_T|⟩ / ⟨B⟩⁴,

   where f_T = (∇ψ × ∇B) · ∇(B · ∇B), B = |∇φ| and ⟨·⟩ denotes a
   flux-surface average.  The diagnostic is implemented via the
   same machinery as ``qs_diagnostics.py`` (triple-product objective),
   and does **not** require Boozer coordinates.  A lower value of f̂_T
   indicates a magnetic field closer to quasisymmetry.

This optimiser perturbs the boundary points along their normals to
minimise the global triple-product error.  The optimisation supports:

  * A **SciPy least-squares driver** (default): optimise a single scalar
    parameter α such that

        P_new = P_orig + α n̂,

    and minimise a least-squares objective based on the per-surface
    triple-product errors.

  * A **SciPy minimise driver**: same parameterisation but using
    general-purpose minimisation (e.g. L-BFGS-B).

  * A simple **random-search optimiser**: at each iteration a random
    normal displacement is applied to each boundary point and accepted
    only if it improves the metric.

For each candidate shape the workflow is:

  * Save the boundary points and (optionally) normals to temporary CSV
    files.
  * Run the MFS solver as a subprocess on these files to obtain a
    potential solution (with internal verbose output disabled).
  * Run the FCI ψ solver on the resulting MFS checkpoint to build
    flux surfaces (with plotting and figure saving disabled).
  * Evaluate the triple-product metric on these surfaces.

The best candidate so far is retained.  After the run, the script writes
the best boundary shape to disk and prints the final QS metric.

Note
----
Running this optimisation loop is computationally expensive.  Each
function evaluation requires solving a large linear system and tracing
field lines.  Use **small numbers of parameters** (here: one scalar
α) and modest tolerances while experimenting, and consider higher-level
parameterisations only once the workflow is robust.
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.optimize import least_squares, minimize

from qs_diagnostics import (
    load_psi_snapshot,
    load_mfs_grad_phi,
    compute_grad_psi_cylindrical,
    compute_grad_psi_cartesian,
    build_axis_interp,
    sample_points_on_psi_levels,
    build_triple_product_fn,
)

###############################################################################
# Utility functions: geometry IO
###############################################################################

def load_surface_csv(xyz_csv: str, normals_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load boundary point coordinates and normals from CSV files.

    The CSV files must have header lines and three columns each for
    x,y,z and nx,ny,nz respectively.

    Parameters
    ----------
    xyz_csv : str
        Path to the CSV file containing the boundary points.
    normals_csv : str
        Path to the CSV file containing surface normals at those points.

    Returns
    -------
    P : ndarray of shape (N,3)
        Boundary points.
    N : ndarray of shape (N,3)
        Corresponding outward normals (unit vectors).
    """
    print(f"[DEBUG] Loading surface points from: {xyz_csv}")
    print(f"[DEBUG] Loading surface normals from: {normals_csv}")
    P = np.loadtxt(xyz_csv, delimiter=",", skiprows=1)
    N = np.loadtxt(normals_csv, delimiter=",", skiprows=1)
    assert P.shape[1] == 3 and N.shape[1] == 3, \
        "Input CSVs must have three columns each."
    # Normalise normals to unit vectors
    norms = np.linalg.norm(N, axis=1, keepdims=True)
    N = N / np.maximum(norms, 1e-15)
    print(f"[DEBUG] Loaded {P.shape[0]} boundary points.")
    return P, N


def save_surface_csv(P: np.ndarray, N: np.ndarray, xyz_file: str, normals_file: str) -> None:
    """Save boundary points and normals to CSV files.

    Parameters
    ----------
    P : ndarray of shape (N,3)
        Boundary points.
    N : ndarray of shape (N,3)
        Normals at each point.
    xyz_file : str
        Output filename for point coordinates.
    normals_file : str
        Output filename for normals.
    """
    print(f"[DEBUG] Saving {P.shape[0]} boundary points to: {xyz_file}")
    print(f"[DEBUG] Saving normals to: {normals_file}")
    header_xyz = "x,y,z"
    header_normals = "nx,ny,nz"
    np.savetxt(xyz_file, P, delimiter=",", header=header_xyz, comments="")
    np.savetxt(normals_file, N, delimiter=",", header=header_normals, comments="")

###############################################################################
# External solvers (MFS and FCI ψ)
###############################################################################

def run_mfs_solver(
    mfs_script: str,
    xyz_csv: str,
    normals_csv: str,
    sf_min: float = 1.5,
    sf_max: float = 4.5,
    lbfgs_maxiter: int = 5,
    k_nn: int = 64,
    mv_weight: float = 0.5,
    interior_eps_factor: float = 5e-3,
    use_mv: bool = True,
    verbose: bool = False,
) -> str:
    """Run the MFS solver as a subprocess on the given geometry.

    The solver writes an ``*_solution.npz`` file into the ``outputs``
    subdirectory relative to the script location.  This function
    returns the path to that file.

    Notes
    -----
    * Internal MFS verbose logging is disabled via ``--no-verbose``.
    * Any plotting in the MFS script should be behind flags; this
      function does not request plots.

    Parameters
    ----------
    mfs_script : str
        Path to the ``main.py`` script for the MFS solver.
    xyz_csv : str
        Path to the boundary points CSV file.
    normals_csv : str
        Path to the normals CSV file.
    sf_min, sf_max, lbfgs_maxiter, k_nn, mv_weight, interior_eps_factor :
        Configuration parameters forwarded to the MFS solver.  See
        ``main.py`` for details.
    use_mv : bool, optional
        Whether to include multivalued basis functions.  Default True.
    verbose : bool, optional
        If True, prints the MFS solver stdout/stderr to the terminal.

    Returns
    -------
    mfs_out_file : str
        Path to the .npz file containing the MFS solution.
    """
    print(f"[DEBUG] Running MFS solver on geometry: {xyz_csv}")
    # Build command line
    cmd = [
        "python", str(mfs_script),
        xyz_csv, normals_csv,
        "--sf_min", str(sf_min),
        "--sf_max", str(sf_max),
        "--lbfgs-maxiter", str(lbfgs_maxiter),
        "--k-nn", str(k_nn),
        "--mv-weight", str(mv_weight),
        "--interior-eps-factor", str(interior_eps_factor),
        "--no-verbose",  # disable internal verbose logging
    ]
    if not use_mv:
        cmd.append("--no-use-mv")  # MFS script flag to disable MV regularization

    # Determine output name
    basename = Path(xyz_csv).stem
    outputs_dir = Path(mfs_script).resolve().parent.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    mfs_out_file = str(outputs_dir / f"{basename}_solution.npz")
    cmd.extend(["--mfs-out", mfs_out_file])

    print(f"[DEBUG] MFS command: {' '.join(cmd)}")
    print(f"[DEBUG] Expected MFS output: {mfs_out_file}")
    # Run MFS solver
    if verbose:
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if not os.path.exists(mfs_out_file):
        raise FileNotFoundError(f"MFS output file not found: {mfs_out_file}")
    print("[DEBUG] MFS solver finished successfully.")
    return mfs_out_file


def run_psi_solver(
    psi_script: str,
    mfs_npz: str,
    grid_N: int = 64,
    N_phi: int = 128,
    eps: float = 1e-5,
    band_h: float = 2.0,
    cg_tol: float = 1e-8,
    cg_maxit: int = 4000,
    nfp: int = 2,
    delta: float = 5e-3,
    use_fci: bool = True,
    fci_nsteps: int = 16,
    fci_planes_per_field_period: int = 16,
    verbose: bool = False,
) -> str:
    """Run the FCI flux solver as a subprocess to compute ψ on a grid.

    Internal plotting and figure saving are disabled via ``--no-plot``
    and ``--no-save-figures``.

    Parameters
    ----------
    psi_script : str
        Path to the ``solve_flux_psi_fci_cyl.py`` script.
    mfs_npz : str
        Path to the MFS solution checkpoint.
    grid_N, N_phi, eps, band_h, cg_tol, cg_maxit, nfp, delta,
    use_fci, fci_nsteps, fci_planes_per_field_period :
        Configuration parameters forwarded to the ψ solver.  See
        ``solve_flux_psi_fci_cyl.py`` for details.
    verbose : bool
        If True, prints the solver output to stdout.

    Returns
    -------
    psi_npz : str
        Path to the .npz file containing the computed ψ snapshot.
    """
    print(f"[DEBUG] Running ψ solver on MFS file: {mfs_npz}")
    # Build command line
    cmd = [
        "python", str(psi_script),
        mfs_npz,
        "--N", str(grid_N),
        "--N_phi", str(N_phi),
        "--eps", str(eps),
        "--band-h", str(band_h),
        "--cg-tol", str(cg_tol),
        "--cg-maxit", str(cg_maxit),
        "--nfp", str(nfp),
        "--delta", str(delta),
        "--fci-nsteps", str(fci_nsteps),
        "--fci-planes-per-field-period", str(fci_planes_per_field_period),
        "--no-plot",
        "--no-save-figures",
    ]
    if not use_fci:
        cmd.append("--no-fci")

    # Determine expected output name used by solve_fci
    base = Path(mfs_npz).stem  # remove suffix
    out_name = f"{base}_psi_fci_cyl_N{grid_N}_Nphi{N_phi}.npz"
    print(f"[DEBUG] ψ solver command: {' '.join(cmd)}")
    print(f"[DEBUG] Expected ψ output: {out_name}")

    # Run ψ solver
    if verbose:
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    psi_npz = out_name
    if not os.path.exists(psi_npz):
        # If not found, try outputs directory
        alt = Path(psi_script).resolve().parent.parent / "outputs" / out_name
        psi_npz = str(alt)
        print(f"[DEBUG] ψ output not found in CWD, trying: {psi_npz}")
    if not os.path.exists(psi_npz):
        raise FileNotFoundError(f"ψ output file not found at {psi_npz}")
    print("[DEBUG] ψ solver finished successfully.")
    return psi_npz

###############################################################################
# Triple-product QS metric (programmatic interface)
###############################################################################

def compute_qs_triple_metric(
    psi_npz: str,
    mfs_npz: str,
    n_surfaces: int = 8,
    band_frac: float = 0.01,
    max_points_per_level: int = 8000,
):
    """
    Programmatic interface to the triple-product QS diagnostic.

    This function mirrors the logic in ``qs_diagnostics.py`` but
    avoids any plotting or CLI interaction.

    Parameters
    ----------
    psi_npz : str
        Path to psi_fci snapshot (.npz) produced by the FCI solver.
    mfs_npz : str
        Path to MFS solution checkpoint (.npz) to rebuild grad_phi.
    n_surfaces : int, optional
        Number of ψ surfaces to sample between ψ_min and ψ_max
        (excluding a small margin near axis and boundary).
    band_frac : float, optional
        Relative ψ-band half-width around each level:
            |ψ - ψ_level| < band_frac * (ψ_max - ψ_min).
    max_points_per_level : int, optional
        Maximum number of points per ψ level used in QS diagnostic.

    Returns
    -------
    qs_global : float
        Global triple-product QS metric (mean of \hat f_T over surfaces).
    qs_errors : np.ndarray
        Array of per-surface \hat f_T(ψ_i), shape (nsurf,).
    psi_levels_used : np.ndarray
        Array of ψ_i values (same length as qs_errors).
    """
    print(f"[DEBUG] Computing QS triple-product metric for:")
    print(f"        psi_npz = {psi_npz}")
    print(f"        mfs_npz = {mfs_npz}")

    # 1) Load ψ snapshot and MFS grad_phi
    psi_data = load_psi_snapshot(psi_npz)
    psi3 = psi_data["psi3"]
    grid = psi_data["grid"]
    inside = psi_data["inside"]
    axis_points = psi_data["axis_points"]

    grad_phi, P_surf, N_surf = load_mfs_grad_phi(mfs_npz)

    # 2) Precompute ∇ψ on the grid (Cartesian components)
    print("[DEBUG] Precomputing ∇ψ on the ψ grid ...")
    if grid["type"] == "cylindrical":
        Rs = grid["Rs"]
        phis = grid["phis"]
        Zs = grid["Zs"]
        psi_x, psi_y, psi_z = compute_grad_psi_cylindrical(psi3, Rs, phis, Zs)
    else:
        xs = grid["xs"]
        ys = grid["ys"]
        zs = grid["zs"]
        psi_x, psi_y, psi_z = compute_grad_psi_cartesian(psi3, xs, ys, zs)

    gx_flat = psi_x.ravel(order="C")
    gy_flat = psi_y.ravel(order="C")
    gz_flat = psi_z.ravel(order="C")

    # 3) Build axis interpolants (for θ,φ coordinates)
    R_axis_interp, Z_axis_interp = build_axis_interp(axis_points)

    # 4) Choose ψ-levels for surfaces (exclude bands near ψ≈min and ψ≈max)
    psi_flat = psi3.ravel(order="C")
    mask_inside = inside.astype(bool)
    psi_inside = psi_flat[mask_inside]
    psi_min = float(np.min(psi_inside))
    psi_max = float(np.max(psi_inside))

    eps_psi = 0.05 * (psi_max - psi_min)
    psi_levels = np.linspace(psi_min + eps_psi, psi_max - eps_psi, n_surfaces)
    print(f"[DEBUG] ψ range used for QS: [{psi_min:.3e}, {psi_max:.3e}]")
    print(f"[DEBUG] Sampling {len(psi_levels)} surfaces in this range.")

    # 5) Sample points in thin ψ-bands, with ∇ψ evaluated at those nodes
    level_data = sample_points_on_psi_levels(
        psi3,
        grid,
        inside,
        psi_levels=psi_levels,
        band_frac=band_frac,
        max_points_per_level=max_points_per_level,
        gradpsi_flat=(gx_flat, gy_flat, gz_flat),
    )

    if len(level_data) == 0:
        raise RuntimeError(
            "No ψ-levels with valid points in compute_qs_triple_metric; "
            "check band_frac and ψ-range."
        )

    print(f"[DEBUG] Number of ψ surfaces with valid samples: {len(level_data)}")

    # 6) Build triple-product evaluator
    triple_prod_fn = build_triple_product_fn(grad_phi)

    qs_values = []

    for isurf, surf in enumerate(level_data):
        psi0 = surf["psi_level"]
        X = surf["X"]               # (N,3)
        gradpsi = surf["grad_psi"]  # (N,3)

        print(f"[DEBUG] Evaluating triple-product on surface {isurf} with ψ≈{psi0:.3e}, "
              f"N_points={X.shape[0]}")

        # Evaluate triple product and |B|
        fT_j, Bmag_j, R_j = triple_prod_fn(X, gradpsi)
        fT = np.asarray(fT_j)
        Bmag = np.asarray(Bmag_j)
        R_vals = np.asarray(R_j)

        # Filter out any pathological points
        mask_good = (
            np.isfinite(fT) &
            np.isfinite(Bmag) &
            (Bmag > 1e-14)
        )
        n_good = mask_good.sum()
        print(f"[DEBUG]   Good points on this surface: {n_good}")
        if n_good < 200:
            print("[WARN]    Not enough good points; skipping this surface.")
            continue

        fT = fT[mask_good]
        Bmag = Bmag[mask_good]
        R_vals = R_vals[mask_good]

        # Normalized triple-product QS error per surface:
        #   \hat f_T(ψ) = <R>^2 <|f_T|> / <B>^4
        mean_abs_fT = float(np.mean(np.abs(fT)))
        mean_R = float(np.mean(R_vals))
        mean_B = float(np.mean(Bmag))
        if mean_B <= 0.0:
            print("[WARN]    Mean |B| <= 0 on this surface; skipping.")
            continue

        qs_surf = (mean_R**2 * mean_abs_fT) / (mean_B**4)
        qs_values.append((psi0, qs_surf))
        print(f"[DEBUG]   Surface ψ≈{psi0:.3e}: hat(f_T) = {qs_surf:.3e}")

    if len(qs_values) == 0:
        raise RuntimeError(
            "No surfaces produced a valid QS triple-product metric "
            "in compute_qs_triple_metric."
        )

    qs_values = np.array(qs_values)  # (nsurf, 2)
    psi_levels_used = qs_values[:, 0]
    qs_errors = qs_values[:, 1]

    # 7) Global QS metric (simple average over surfaces)
    qs_global = float(np.mean(qs_errors))
    print(f"[INFO] Global triple-product QS metric (mean hat f_T) = {qs_global:.5e}")

    return qs_global, qs_errors, psi_levels_used

###############################################################################
# Optimisation helpers
###############################################################################

def evaluate_qs_for_geometry(
    P: np.ndarray,
    N: np.ndarray,
    mfs_script: str,
    psi_script: str,
    sf_min: float,
    sf_max: float,
    k_nn: int,
    n_surfaces: int,
    band_frac: float,
    max_points_per_level: int,
    label: str = "candidate",
    solver_verbose: bool = False,
) -> float:
    """Helper: given a geometry (P, N), run MFS + ψ + QS and return metric."""
    print(f"[DEBUG] Evaluating QS for geometry ({label}) with {P.shape[0]} points.")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_xyz = os.path.join(tmpdir, f"{label}_points.csv")
        tmp_norm = os.path.join(tmpdir, f"{label}_normals.csv")
        save_surface_csv(P, N, tmp_xyz, tmp_norm)

        mfs_out = run_mfs_solver(
            mfs_script, tmp_xyz, tmp_norm,
            sf_min=sf_min, sf_max=sf_max, k_nn=k_nn,
            verbose=solver_verbose,
        )
        psi_out = run_psi_solver(
            psi_script, mfs_out,
            verbose=solver_verbose,
        )
        metric, _, _ = compute_qs_triple_metric(
            psi_out, mfs_out,
            n_surfaces=n_surfaces, band_frac=band_frac,
            max_points_per_level=max_points_per_level,
        )
    print(f"[INFO] QS metric for {label} = {metric:.3e}")
    return metric

###############################################################################
# Random-search optimiser (original algorithm)
###############################################################################

def optimise_boundary_random(
    xyz_csv: str,
    normals_csv: str,
    mfs_script: str,
    psi_script: str,
    n_iterations: int = 3,
    perturbation_scale: float = 0.005,
    sf_min: float = 1.5,
    sf_max: float = 4.0,
    k_nn: int = 64,
    n_surfaces: int = 6,
    band_frac: float = 0.01,
    max_points_per_level: int = 4000,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """Optimise a boundary surface for quasisymmetry via random perturbations.

    At each iteration a random displacement along the normals is applied
    to every boundary point.  If the resulting shape yields a lower global
    triple-product error, it is accepted; otherwise it is discarded.  The
    perturbation amplitude is gradually reduced to focus on fine
    adjustments near the optimum.

    This is the original/simple optimiser; it is included as an option
    primarily for debugging and comparison.

    Returns
    -------
    best_P : ndarray
        The boundary points of the best shape found.
    best_metric : float
        The global triple-product metric for the best shape.
    """
    P_orig, N_orig = load_surface_csv(xyz_csv, normals_csv)

    # Evaluate baseline configuration
    metric0 = evaluate_qs_for_geometry(
        P_orig, N_orig,
        mfs_script, psi_script,
        sf_min=sf_min, sf_max=sf_max, k_nn=k_nn,
        n_surfaces=n_surfaces, band_frac=band_frac,
        max_points_per_level=max_points_per_level,
        label="baseline",
        solver_verbose=verbose,
    )

    best_P = P_orig.copy()
    best_metric = metric0
    if verbose:
        print(f"[INFO] Initial QS metric = {best_metric:.3e}")

    current_scale = perturbation_scale
    for it in range(1, n_iterations + 1):
        print(f"[INFO] Random-search iteration {it}/{n_iterations}")
        # Generate random displacement along normals
        random_factors = np.random.randn(P_orig.shape[0])
        disp = current_scale * random_factors[:, None] * N_orig
        P_new = best_P + disp

        try:
            metric_new = evaluate_qs_for_geometry(
                P_new, N_orig,
                mfs_script, psi_script,
                sf_min=sf_min, sf_max=sf_max, k_nn=k_nn,
                n_surfaces=n_surfaces, band_frac=band_frac,
                max_points_per_level=max_points_per_level,
                label=f"random_it{it}",
                solver_verbose=False,
            )
        except RuntimeError as e:
            print(f"[WARN] QS evaluation failed at iteration {it}: {e}")
            metric_new = np.inf

        if verbose:
            print(f"[INFO] Iteration {it}: candidate metric = {metric_new:.3e} (best {best_metric:.3e})")
        # Accept if improved
        if metric_new < best_metric:
            best_metric = metric_new
            best_P = P_new
            current_scale *= 0.7  # reduce step size
            if verbose:
                print(f"[INFO]  Accepted new shape.  Updated QS metric = {best_metric:.3e} "
                      f"and scale={current_scale:.3e}")
        else:
            # Otherwise discard and reduce scale slightly
            current_scale *= 0.9
            if verbose:
                print(f"[INFO]  Rejected candidate. New perturbation scale={current_scale:.3e}")
    return best_P, best_metric

###############################################################################
# SciPy optimisers (least_squares and minimize) with 1D normal displacement
###############################################################################

def optimise_boundary_scipy(
    xyz_csv: str,
    normals_csv: str,
    mfs_script: str,
    psi_script: str,
    optimizer: str = "least_squares",
    alpha0: float = 0.0,
    alpha_bounds: Tuple[float, float] = (-0.01, 0.01),
    sf_min: float = 1.5,
    sf_max: float = 4.0,
    k_nn: int = 64,
    n_surfaces: int = 6,
    band_frac: float = 0.01,
    max_points_per_level: int = 4000,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """Optimise the boundary using SciPy least_squares or minimize.

    Parameterisation
    ----------------
    The shape is parameterised by a single scalar α:

        P(α) = P_orig + α * n̂,

    where n̂ are the original (unit) normals. Positive α expands the
    surface along the normals; negative α contracts it.

    * ``least_squares``: minimise the vector of per-surface QS errors
      (hat(f_T)(ψ_i)) in a least-squares sense.
    * ``minimize``: minimise the global QS metric directly.

    This is intentionally minimal (1D parameter) to keep the number of
    expensive MFS+ψ+QS evaluations manageable.  It serves as a template
    for richer parameterisations (e.g. multiple Fourier modes) later.

    Returns
    -------
    best_P : ndarray
        Boundary points corresponding to the optimal α.
    best_metric : float
        Global QS metric at the optimum.
    """
    P_orig, N_orig = load_surface_csv(xyz_csv, normals_csv)

    # Precompute the baseline QS metric (α = 0)
    metric0 = evaluate_qs_for_geometry(
        P_orig, N_orig,
        mfs_script, psi_script,
        sf_min=sf_min, sf_max=sf_max, k_nn=k_nn,
        n_surfaces=n_surfaces, band_frac=band_frac,
        max_points_per_level=max_points_per_level,
        label="alpha_0",
        solver_verbose=verbose,
    )
    if verbose:
        print(f"[INFO] Baseline metric (α=0) = {metric0:.3e}")

    def make_geometry(alpha: float) -> np.ndarray:
        """Return boundary points for given scalar displacement α."""
        return P_orig + alpha * N_orig

    def residuals(alpha_vec: np.ndarray) -> np.ndarray:
        """Residual vector for least-squares: per-surface QS errors."""
        alpha = float(alpha_vec[0])
        if verbose:
            print(f"[DEBUG] [LS] Evaluating residuals at α={alpha:.6e}")
        try:
            P_new = make_geometry(alpha)
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_xyz = os.path.join(tmpdir, "ls_points.csv")
                tmp_norm = os.path.join(tmpdir, "ls_normals.csv")
                save_surface_csv(P_new, N_orig, tmp_xyz, tmp_norm)
                mfs_out = run_mfs_solver(
                    mfs_script, tmp_xyz, tmp_norm,
                    sf_min=sf_min, sf_max=sf_max, k_nn=k_nn,
                    verbose=False,
                )
                psi_out = run_psi_solver(
                    psi_script, mfs_out,
                    verbose=False,
                )
                _, qs_errors, _ = compute_qs_triple_metric(
                    psi_out, mfs_out,
                    n_surfaces=n_surfaces, band_frac=band_frac,
                    max_points_per_level=max_points_per_level,
                )
            if verbose:
                print(f"[DEBUG] [LS] α={alpha:.6e}, "
                      f"mean(hat(f_T))={np.mean(qs_errors):.3e}")
            return qs_errors  # LS objective is sum of squares of these
        except Exception as e:
            print(f"[WARN] [LS] Failure at α={alpha:.6e}: {e}")
            # Return large residuals to penalise failure
            return np.array([1e6])

    def objective(alpha_vec: np.ndarray) -> float:
        """Scalar objective for minimize: global QS metric."""
        alpha = float(alpha_vec[0])
        if verbose:
            print(f"[DEBUG] [MIN] Evaluating objective at α={alpha:.6e}")
        try:
            P_new = make_geometry(alpha)
            metric = evaluate_qs_for_geometry(
                P_new, N_orig,
                mfs_script, psi_script,
                sf_min=sf_min, sf_max=sf_max, k_nn=k_nn,
                n_surfaces=n_surfaces, band_frac=band_frac,
                max_points_per_level=max_points_per_level,
                label=f"alpha_{alpha:.3e}",
                solver_verbose=False,
            )
            return metric
        except Exception as e:
            print(f"[WARN] [MIN] Failure at α={alpha:.6e}: {e}")
            return 1e9

    alpha0_vec = np.array([alpha0], dtype=float)
    bounds = (np.array([alpha_bounds[0]]), np.array([alpha_bounds[1]]))

    if optimizer == "least_squares":
        if verbose:
            print("[INFO] Using SciPy least_squares optimiser.")
        res = least_squares(
            residuals,
            alpha0_vec,
            bounds=bounds,
            verbose=2 if verbose else 0,
        )
        alpha_opt = float(res.x[0])
        if verbose:
            print(f"[INFO] least_squares finished with α*={alpha_opt:.6e}")
    elif optimizer == "minimize":
        if verbose:
            print("[INFO] Using SciPy minimize optimiser.")
        res = minimize(
            objective,
            alpha0_vec,
            method="L-BFGS-B",
            bounds=[alpha_bounds],
            options={"disp": verbose},
        )
        alpha_opt = float(res.x[0])
        if verbose:
            print(f"[INFO] minimize finished with α*={alpha_opt:.6e}, "
                  f"f(α*)={res.fun:.3e}")
    else:
        raise ValueError(f"Unknown optimizer='{optimizer}'. "
                         "Use 'least_squares', 'minimize', or 'random'.")

    # Evaluate final geometry at α_opt
    best_P = make_geometry(alpha_opt)
    best_metric = evaluate_qs_for_geometry(
        best_P, N_orig,
        mfs_script, psi_script,
        sf_min=sf_min, sf_max=sf_max, k_nn=k_nn,
        n_surfaces=n_surfaces, band_frac=band_frac,
        max_points_per_level=max_points_per_level,
        label=f"alpha_opt_{alpha_opt:.3e}",
        solver_verbose=verbose,
    )
    return best_P, best_metric

###############################################################################
# CLI entry point
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Optimise a stellarator boundary for quasisymmetry using the triple-product metric."
    )
    parser.add_argument(
        "--xyz_csv",
        required=True,
        help="Path to CSV file containing the boundary point coordinates",
    )
    parser.add_argument(
        "--normals_csv",
        required=True,
        help="Path to CSV file containing the boundary normals",
    )
    parser.add_argument(
        "--mfs_script",
        required=True,
        help="Path to main.py (MFS solver)",
    )
    parser.add_argument(
        "--psi_script",
        required=True,
        help="Path to solve_flux_psi_fci_cyl.py (FCI ψ solver)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["least_squares", "minimize", "random"],
        default="least_squares",
        help="Optimiser to use: 'least_squares' (default), 'minimize', or 'random'.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of optimisation iterations for 'random' optimiser (default 3)",
    )
    parser.add_argument(
        "--perturb-scale",
        type=float,
        default=0.005,
        help="Initial perturbation amplitude for 'random' optimiser (default 0.005 units)",
    )
    parser.add_argument(
        "--alpha0",
        type=float,
        default=0.0,
        help="Initial α for SciPy optimisers (default 0.0)",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=-0.01,
        help="Lower bound for α in SciPy optimisers (default -0.01)",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=0.01,
        help="Upper bound for α in SciPy optimisers (default 0.01)",
    )
    parser.add_argument(
        "--sf_min",
        type=float,
        default=1.5,
        help="Minimum source factor passed to MFS solver",
    )
    parser.add_argument(
        "--sf_max",
        type=float,
        default=4.0,
        help="Maximum source factor passed to MFS solver",
    )
    parser.add_argument(
        "--k_nn",
        type=int,
        default=64,
        help="Number of nearest neighbours for MFS area weighting",
    )
    parser.add_argument(
        "--n_surfaces",
        type=int,
        default=6,
        help="Number of ψ surfaces used for the triple-product diagnostic",
    )
    parser.add_argument(
        "--band_frac",
        type=float,
        default=0.01,
        help="Relative half-width of sampling band around ψ levels",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=4000,
        help="Maximum points sampled per surface for QS metric",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress and debug information",
    )
    args = parser.parse_args()

    if args.verbose:
        print("[INFO] Starting boundary optimisation for QS.")
        print(f"[INFO] Optimizer: {args.optimizer}")

    if args.optimizer == "random":
        best_P, best_metric = optimise_boundary_random(
            args.xyz_csv,
            args.normals_csv,
            args.mfs_script,
            args.psi_script,
            n_iterations=args.iterations,
            perturbation_scale=args.perturb_scale,
            sf_min=args.sf_min,
            sf_max=args.sf_max,
            k_nn=args.k_nn,
            n_surfaces=args.n_surfaces,
            band_frac=args.band_frac,
            max_points_per_level=args.max_points,
            verbose=args.verbose,
        )
    else:
        best_P, best_metric = optimise_boundary_scipy(
            args.xyz_csv,
            args.normals_csv,
            args.mfs_script,
            args.psi_script,
            optimizer=args.optimizer,
            alpha0=args.alpha0,
            alpha_bounds=(args.alpha_min, args.alpha_max),
            sf_min=args.sf_min,
            sf_max=args.sf_max,
            k_nn=args.k_nn,
            n_surfaces=args.n_surfaces,
            band_frac=args.band_frac,
            max_points_per_level=args.max_points,
            verbose=args.verbose,
        )

    # Save the optimised boundary
    P_orig, N_orig = load_surface_csv(args.xyz_csv, args.normals_csv)
    out_xyz = Path(args.xyz_csv).with_name(Path(args.xyz_csv).stem + "_qsoptim.csv")
    out_norm = Path(args.normals_csv).with_name(Path(args.normals_csv).stem + "_qsoptim_normals.csv")
    save_surface_csv(best_P, N_orig, str(out_xyz), str(out_norm))
    print(f"[INFO] Optimisation complete. Best QS metric = {best_metric:.3e}")
    print(f"[INFO] Optimised boundary saved to {out_xyz} and {out_norm}")


if __name__ == "__main__":
    main()
