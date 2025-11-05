#!/usr/bin/env python3
"""
Export a pyQSC boundary surface (x,y,z) to a single file for downstream use.

Outputs one of:
  - .npz  -> keys: X, Y, Z  (grid, preferred)
  - .npy  -> dict with keys: X, Y, Z  (grid)
  - .csv  -> flattened rows: x,y,z     (points)

Example:
  python qsc_export_boundary.py --r 0.10 --ntheta 64 --nphi 256 \
      --out surf_grid_w7x.npz --rc 1 0.09 --zs 0 -0.09 --nfp 2 \
      --etabar 0.95 --I2 0.9 --order r2 --B2c -0.7 --p2 -600000

Then in input.toml:
  [surfaces]
  mode = "files"
  files = [
    { name = "W7X_like", path = "surf_grid_w7x.npz", format = "grid",
      periodic_theta = true, periodic_phi = true }
  ]
"""

import argparse
# import numpy as np
# from qsc import Qsc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3D proj)
import sys

# def build_qsc_from_args(args) -> Qsc:
#     # Minimal constructor covering common knobs; feel free to extend
#     return Qsc(
#         rc=args.rc, zs=args.zs, nfp=args.nfp,
#         etabar=args.etabar, I2=args.I2,
#         order=args.order, B2c=args.B2c, p2=args.p2
#     )
from scipy.io import netcdf_file
import numpy as np
from jax import vmap
import jax.numpy as jnp

def compute_boundary(filename: str, s: float, ntheta: int, nphi: int):
    """
    Compute the boundary surface and analytic unit normals from a VMEC wout file.

    Returns (X, Y, Z, NX, NY, NZ) where NX, NY and NZ are the components of
    outward‐pointing unit normals on the (ntheta×nphi) grid.
    """

    f = netcdf_file(filename,'r',mmap=False)
    ns = f.variables['ns'][()]
    xn = jnp.array(f.variables['xn'][()])
    xm = jnp.array(f.variables['xm'][()])
    rmnc = jnp.array(f.variables['rmnc'][()])
    zmns = jnp.array(f.variables['zmns'][()])
    lasym = f.variables['lasym__logical__'][()]
    if lasym==1:
        rmns = jnp.array(f.variables['rmns'][()])
        zmnc = jnp.array(f.variables['zmnc'][()])
    else:
        rmns = 0*rmnc
        zmnc = 0*rmnc
        
    print(f"[INFO] Loaded VMEC wout file '{filename}' with ns={ns}, ntheta={ntheta}, nphi={nphi}, ")
    print(f"       Fourier modes: {len(xn)}")
    print(f"       Using radial location s={s} (0<=s<=1)")
    print(f"       with xm shape {xm.shape}, xn shape {xn.shape}")

    theta1D = jnp.linspace(0,2*jnp.pi,num=ntheta)
    phi1D = jnp.linspace(0,2*jnp.pi,num=nphi)
    # phi2D, theta2D = jnp.meshgrid(phi1D,theta1D)
    
    Z = jnp.zeros((ntheta,nphi))
    X = jnp.zeros((ntheta,nphi))
    Y = jnp.zeros((ntheta,nphi))
    
    s_full_grid = jnp.linspace(0, 1, ns)
    rmnc_interp = vmap(lambda row: jnp.interp(s, s_full_grid, row, left='extrapolate'), in_axes=1)(rmnc)
    rmns_interp = vmap(lambda row: jnp.interp(s, s_full_grid, row, left='extrapolate'), in_axes=1)(rmns)
    zmns_interp = vmap(lambda row: jnp.interp(s, s_full_grid, row, left='extrapolate'), in_axes=1)(zmns)
    zmnc_interp = vmap(lambda row: jnp.interp(s, s_full_grid, row, left='extrapolate'), in_axes=1)(zmnc)
    
    # Build (θ,φ) grids and precompute angle arrays
    theta2d, phi2d = jnp.meshgrid(theta1D, phi1D, indexing="ij")
    angles = xm[:, None, None] * theta2d[None, :, :] - xn[:, None, None] * phi2d[None, :, :]
    sin_angles = jnp.sin(angles)
    cos_angles = jnp.cos(angles)

    # Fourier sums for R and Z
    r_coordinate = (
        jnp.einsum("m,mjk->jk", rmnc_interp, cos_angles)
        + jnp.einsum("m,mjk->jk", rmns_interp, sin_angles)
    )
    z_coordinate = (
        jnp.einsum("m,mjk->jk", zmns_interp, sin_angles)
        + jnp.einsum("m,mjk->jk", zmnc_interp, cos_angles)
    )
    X = r_coordinate * jnp.cos(phi2d)
    Y = r_coordinate * jnp.sin(phi2d)
    Z = z_coordinate

    # Analytic partial derivatives (see explanation above)
    dR_dtheta = (
        jnp.einsum("m,mjk,m->jk", rmnc_interp, -sin_angles, xm)
        + jnp.einsum("m,mjk,m->jk", rmns_interp,  cos_angles, xm)
    )
    dZ_dtheta = (
        jnp.einsum("m,mjk,m->jk", zmns_interp,  cos_angles, xm)
        + jnp.einsum("m,mjk,m->jk", zmnc_interp, -sin_angles, xm)
    )
    dX_dtheta = dR_dtheta * jnp.cos(phi2d)
    dY_dtheta = dR_dtheta * jnp.sin(phi2d)

    dR_dphi = (
        jnp.einsum("m,mjk,m->jk", rmnc_interp,  sin_angles, xn)
        + jnp.einsum("m,mjk,m->jk", rmns_interp, -cos_angles, xn)
    )
    dZ_dphi = (
        jnp.einsum("m,mjk,m->jk", zmns_interp, -cos_angles, xn)
        + jnp.einsum("m,mjk,m->jk", zmnc_interp,  sin_angles, xn)
    )
    dX_dphi = dR_dphi * jnp.cos(phi2d) - r_coordinate * jnp.sin(phi2d)
    dY_dphi = dR_dphi * jnp.sin(phi2d) + r_coordinate * jnp.cos(phi2d)

    g_theta = jnp.stack([dX_dtheta, dY_dtheta, dZ_dtheta], axis=-1)
    g_phi   = jnp.stack([dX_dphi,   dY_dphi,   dZ_dphi  ], axis=-1)
    normal  = jnp.cross(g_theta, g_phi, axis=-1)
    mag     = jnp.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12
    unit_normal = normal / mag

    # Outward orientation by centroid test
    coords = jnp.stack([X, Y, Z], axis=-1)
    centroid = jnp.mean(coords.reshape(-1, 3), axis=0)
    dots = jnp.sum((coords - centroid) * unit_normal, axis=-1)
    mean_dot = jnp.mean(dots)
    unit_normal = jnp.where(mean_dot < 0, -unit_normal, unit_normal)

    NX, NY, NZ = unit_normal[..., 0], unit_normal[..., 1], unit_normal[..., 2]
    return jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z), jnp.asarray(NX), jnp.asarray(NY), jnp.asarray(NZ)

# ----------------------------- I/O helpers -----------------------------

def save_single_file(out_path: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                  NX: np.ndarray | None = None, NY: np.ndarray | None = None,
                  NZ: np.ndarray | None = None) -> None:
    """
    Save surface coordinates (and optionally normals) to a single file.

    For .npz files, the keys NX, NY and NZ are stored when normals are supplied.
    For .npy files, a dictionary with keys X,Y,Z (and NX,NY,NZ if present) is saved.
    For .csv files, the surface points are saved in the original file and the
    normals are written to a companion file named `<stem>_normals.csv`.
    """
    ext = out_path.lower().split(".")[-1]
    have_normals = (NX is not None) and (NY is not None) and (NZ is not None)
    if ext == "npz":
        if have_normals:
            np.savez(out_path, X=X, Y=Y, Z=Z, NX=NX, NY=NY, NZ=NZ)
        else:
            np.savez(out_path, X=X, Y=Y, Z=Z)
    elif ext == "npy":
        data_dict = {"X": X, "Y": Y, "Z": Z}
        if have_normals:
            data_dict["NX"] = NX
            data_dict["NY"] = NY
            data_dict["NZ"] = NZ
        np.save(out_path, data_dict, allow_pickle=True)
    elif ext == "csv":
        P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        np.savetxt(out_path, P, delimiter=",", header="x,y,z", comments="")
        if have_normals:
            Pn = np.stack([NX, NY, NZ], axis=-1).reshape(-1, 3)
            norm_path = out_path.rsplit(".", 1)[0] + "_normals.csv"
            np.savetxt(norm_path, Pn, delimiter=",", header="nx,ny,nz", comments="")
            print(f"[INFO] Saved companion normals file to {norm_path}")
    else:
        raise ValueError(f"Unsupported output extension: .{ext}")

def load_saved_file(path: str):
    """
    Load saved surface back from disk.
    Returns (kind, X, Y, Z) where:
      - kind = "grid" for .npz/.npy with 2D X,Y,Z
      - kind = "points" for .csv where X,Y,Z are 1D flattened arrays
    """
    ext = path.lower().split(".")[-1]
    if ext == "npz":
        nz = np.load(path)
        X, Y, Z = nz["X"], nz["Y"], nz["Z"]
        return "grid", np.asarray(X), np.asarray(Y), np.asarray(Z)
    elif ext == "npy":
        obj = np.load(path, allow_pickle=True).item()
        X, Y, Z = obj["X"], obj["Y"], obj["Z"]
        return "grid", np.asarray(X), np.asarray(Y), np.asarray(Z)
    elif ext == "csv":
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr[None, :]
        X, Y, Z = arr[:, 0], arr[:, 1], arr[:, 2]
        return "points", X, Y, Z
    else:
        raise ValueError(f"Unsupported extension for loading: .{ext}")

# ------------------------ metrics & visualization ------------------------

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def compare_and_plot(out_path: str, X_ref: np.ndarray, Y_ref: np.ndarray, Z_ref: np.ndarray):
    """
    Reload the saved file, compute errors vs. the reference arrays (from pyQSC),
    and plot side-by-side. Handles both grid and point outputs.
    """
    kind, X2, Y2, Z2 = load_saved_file(out_path)
    print(f"[VERIFY] Reloaded '{out_path}' as kind={kind} with shapes:")
    print(f"         X2={X2.shape} Y2={Y2.shape} Z2={Z2.shape}")

    if kind == "grid":
        # Require same shape for direct comparison:
        if X_ref.shape != X2.shape or Y_ref.shape != Y2.shape or Z_ref.shape != Z2.shape:
            print("[WARN] Saved grid shape differs from reference. "
                  "Reshape/compare flattened values instead.", file=sys.stderr)
        # Compute metrics (flattened)
        xr = X_ref.reshape(-1); yr = Y_ref.reshape(-1); zr = Z_ref.reshape(-1)
        x2 = X2.reshape(-1);    y2 = Y2.reshape(-1);    z2 = Z2.reshape(-1)
        max_abs = float(np.max(np.abs(np.stack([xr - x2, yr - y2, zr - z2], axis=0))))
        rmse_all = float(np.sqrt(np.mean((xr - x2)**2 + (yr - y2)**2 + (zr - z2)**2)))
        print(f"[ERROR] max |Δ| = {max_abs:.3e}, RMSE (xyz stacked) = {rmse_all:.3e}")

        # Plot surfaces
        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        ax1.plot_surface(X_ref, Y_ref, Z_ref, rstride=1, cstride=1, linewidth=0, alpha=0.9, shade=False)
        ax1.set_title("Original (pyQSC)")
        ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

        ax2.plot_surface(X2, Y2, Z2, rstride=1, cstride=1, linewidth=0, alpha=0.9, shade=False)
        ax2.set_title("Reloaded from file")
        ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

        # keep similar aspect
        for ax in (ax1, ax2):
            _equal_aspect_3d(ax)

        plt.show()

        # Optional: color error heatmap on the ref grid
        err = np.sqrt((X_ref - X2)**2 + (Y_ref - Y2)**2 + (Z_ref - Z2)**2)
        fig2 = plt.figure(figsize=(6, 5), constrained_layout=True)
        axe = fig2.add_subplot(1, 1, 1, projection='3d')
        # Show error as facecolors; normalize for visibility
        e_norm = (err - err.min()) / (err.ptp() + 1e-15)
        # Use a grayscale-ish colormap via facecolors:
        facecolors = plt.cm.viridis(e_norm)
        axe.plot_surface(X_ref, Y_ref, Z_ref, facecolors=facecolors,
                         rstride=1, cstride=1, linewidth=0, alpha=1.0, shade=False)
        m = plt.cm.ScalarMappable(cmap='viridis')
        m.set_array(err)
        cbar = plt.colorbar(m, shrink=0.7)
        cbar.set_label("Pointwise |Δr|")
        axe.set_title("Pointwise error |Δr| on original grid")
        for ax in (axe,):
            _equal_aspect_3d(ax)
        plt.show()

    else:  # kind == "points" (CSV)
        # For CSV, we only saved flattened points, so compare to a flattened version:
        Pr = np.stack([X_ref.reshape(-1), Y_ref.reshape(-1), Z_ref.reshape(-1)], axis=-1)
        P2 = np.stack([X2.reshape(-1),    Y2.reshape(-1),    Z2.reshape(-1)], axis=-1)

        # If counts differ, compare against a min subset
        n = min(len(Pr), len(P2))
        Pr = Pr[:n]; P2 = P2[:n]

        max_abs = float(np.max(np.abs(Pr - P2)))
        rmse_all = _rmse(Pr, P2)
        print(f"[ERROR] (points) max |Δ| = {max_abs:.3e}, RMSE = {rmse_all:.3e} (compared on first {n} pts)")

        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        ax1.scatter(Pr[:, 0], Pr[:, 1], Pr[:, 2], s=2)
        ax1.set_title("Original (flattened grid pts)")
        ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

        ax2.scatter(P2[:, 0], P2[:, 1], P2[:, 2], s=2)
        ax2.set_title("Reloaded from CSV")
        ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

        for ax in (ax1, ax2):
            _equal_aspect_3d(ax)
        plt.show()

def _equal_aspect_3d(ax):
    """
    Make 3D axes have equal aspect ratio.
    """
    xs, ys, zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xmid, ymid, zmid = np.mean(xs), np.mean(ys), np.mean(zs)
    max_range = max(xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]) / 2.0
    ax.set_xlim3d([xmid - max_range, xmid + max_range])
    ax.set_ylim3d([ymid - max_range, ymid + max_range])
    ax.set_zlim3d([zmid - max_range, zmid + max_range])

def main():
    ap = argparse.ArgumentParser(description="Export VMEC boundary (x,y,z) to a single file.")
    # Geometry / near-axis config (replicates your example defaults)
    ap.add_argument("--s", type=float, default=1.0, help="Radial location between 0 and 1 (s=1.0 -> boundary)")
    ap.add_argument("--filename", type=str, default="wout.nc", help="VMEC output wout file path")

    # Surface resolution
    ap.add_argument("--ntheta", type=int, default=32, help="Grid points in poloidal angle")
    ap.add_argument("--nphi", type=int, default=64, help="Grid points in toroidal angle")

    # Output
    ap.add_argument("--out", type=str, default=".csv", help="Output file: .npz | .npy | .csv")

    args = ap.parse_args()

    output_file = args.filename[:-3]+args.out
    X, Y, Z, NX, NY, NZ = compute_boundary(args.filename, args.s, args.ntheta, args.nphi)
    save_single_file(output_file, X, Y, Z, NX, NY, NZ)

    # Optional: quick info
    print(f"Saved boundary to {output_file} with shape X,Y,Z = {X.shape}")
    
    # Verify: reload and compare
    compare_and_plot(output_file, X, Y, Z)

if __name__ == "__main__":
    main()
