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
import numpy as np
from qsc import Qsc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3D proj)
import sys

def build_qsc_from_args(args) -> Qsc:
    # Minimal constructor covering common knobs; feel free to extend
    return Qsc(
        rc=args.rc, zs=args.zs, nfp=args.nfp,
        etabar=args.etabar, I2=args.I2,
        order=args.order, B2c=args.B2c, p2=args.p2
    )

def compute_boundary(stel: Qsc, r: float, ntheta: int, nphi: int):
    """
    Returns X, Y, Z on a (ntheta x nphi) grid using pyQSC's get_boundary.
    """
    # get_boundary returns x_2D_plot, y_2D_plot, z_2D_plot, R_2Dnew
    X, Y, Z, _R = stel.get_boundary(r=r, ntheta=ntheta, nphi=nphi)
    # Ensure dtype and contiguous arrays
    return np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64), np.asarray(Z, dtype=np.float64)

# ----------------------------- I/O helpers -----------------------------

def save_single_file(out_path: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """
    Save X,Y,Z to a single file. Format inferred from extension:
      .npz -> np.savez(out_path, X=X, Y=Y, Z=Z)
      .npy -> np.save(out_path, {"X":X, "Y":Y, "Z":Z})
      .csv -> rows "x,y,z" flattened in C-order
    """
    ext = out_path.lower().split(".")[-1]
    if ext == "npz":
        np.savez(out_path, X=X, Y=Y, Z=Z)
    elif ext == "npy":
        np.save(out_path, {"X": X, "Y": Y, "Z": Z}, allow_pickle=True)
    elif ext == "csv":
        P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        np.savetxt(out_path, P, delimiter=",", header="x,y,z", comments="")
    else:
        raise ValueError(f"Unsupported output extension: .{ext}. Use .npz, .npy, or .csv")

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
    ap = argparse.ArgumentParser(description="Export pyQSC boundary (x,y,z) to a single file.")
    # Geometry / near-axis config (replicates your example defaults)
    ap.add_argument("--rc", type=float, nargs="+", default=[1.0, 0.045], help="rc Fourier coefficients (e.g. --rc 1 0.09)")
    ap.add_argument("--zs", type=float, nargs="+", default=[0.0, -0.045], help="zs Fourier coefficients (e.g. --zs 0 -0.09)")
    ap.add_argument("--nfp", type=int, default=3, help="Number of field periods")
    ap.add_argument("--etabar", type=float, default=0.9, help="Etabar parameter")
    ap.add_argument("--I2", type=float, default=0.0, help="Toroidal current parameter")
    ap.add_argument("--order", type=str, default="r1", help="Near-axis order, e.g. r1, r2, r3")
    ap.add_argument("--B2c", type=float, default=0.0)
    ap.add_argument("--p2", type=float, default=0.0)

    # Surface resolution
    ap.add_argument("--r", type=float, default=0.10, help="Near-axis radius for the surface")
    ap.add_argument("--ntheta", type=int, default=32, help="Grid points in poloidal angle")
    ap.add_argument("--nphi", type=int, default=64, help="Grid points in toroidal angle")

    # Output
    ap.add_argument("--out", type=str, default="pyqsc_surface.csv", help="Output path: .npz | .npy | .csv")

    args = ap.parse_args()

    stel = build_qsc_from_args(args)
    X, Y, Z = compute_boundary(stel, r=args.r, ntheta=args.ntheta, nphi=args.nphi)
    save_single_file(args.out, X, Y, Z)

    # Optional: quick info
    print(f"Saved boundary to {args.out} with shape X,Y,Z = {X.shape}")
    
    # Verify: reload and compare
    compare_and_plot(args.out, X, Y, Z)

if __name__ == "__main__":
    main()
