#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert an STL surface into two CSVs usable by the MFS Laplace solver:
  1) <outstem>.csv          with columns: x,y,z
  2) <outstem>_normals.csv  with columns: nx,ny,nz

Sampling options:
  - Uniform by triangle area (default), or
  - "Even-ish" Poisson-like via trimesh.sample.sample_surface_even (--even)

Normals:
  - Outward unit face normals at the sampled locations.
  - Orientation is verified using centroid; flipped if needed.

Preview:
  - 3D scatter of points, with a quiver of a subsample of normals.

Usage:
  python stl_to_boundary.py model.stl --n 2048 --even --outstem wout_precise_QA
"""

import argparse
import numpy as np
import trimesh as tm
import sys
import os
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

def write_points_csv(path, P):
    header = "x,y,z"
    np.savetxt(path, P, delimiter=",", header=header, comments="", fmt="%.18e")

def write_normals_csv(path, N):
    header = "nx,ny,nz"
    np.savetxt(path, N, delimiter=",", header=header, comments="", fmt="%.18e")

def ensure_outward_normals(points, normals):
    """
    Heuristic: compute centroid c; if average (p - c)·n < 0, flip normals.
    Works well for closed shells where 'outward' roughly points away from centroid.
    """
    c = points.mean(axis=0)
    s = np.mean(np.einsum('ij,ij->i', points - c, normals))
    if s < 0.0:
        normals = -normals
        flipped = True
    else:
        flipped = False
    return normals, flipped

def plot_preview(P, N, quiver_frac=0.03, title="Sampled boundary + normals"):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:,0], P[:,1], P[:,2], s=5, alpha=0.7)

    # Subsample normals for clarity
    m = P.shape[0]
    step = max(1, int(1.0 / max(quiver_frac, 1e-6)))
    idx = np.arange(0, m, step)
    L = np.linalg.norm(P.max(axis=0) - P.min(axis=0))
    qlen = 0.05 * L

    ax.quiver(P[idx,0], P[idx,1], P[idx,2],
              N[idx,0], N[idx,1], N[idx,2],
              length=qlen, normalize=True, color="k", linewidth=0.6)

    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    # make axes roughly equal
    mins = P.min(axis=0); maxs = P.max(axis=0)
    ctr = (mins + maxs) / 2.0
    rad = 0.5 * np.max(maxs - mins)
    ax.set_xlim(ctr[0]-rad, ctr[0]+rad)
    ax.set_ylim(ctr[1]-rad, ctr[1]+rad)
    ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl", help="Input STL file (ASCII or binary)")
    ap.add_argument("--n", type=int, default=2048, help="Number of points to sample (default: 2048)")
    ap.add_argument("--even", action="store_true",
                    help="Use sample_surface_even for more even spacing (slower)")
    ap.add_argument("--outstem", default=None,
                    help="Output stem for CSVs (default: derived from STL filename)")
    ap.add_argument("--preview", action="store_true", help="Show a 3D preview plot at the end")
    ap.add_argument("--no-fix-normals", action="store_true",
                    help="Disable trimesh normal fixing (not recommended)")
    args = ap.parse_args()

    in_path = args.stl
    if not os.path.isfile(in_path):
        print(f"ERROR: file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    if args.outstem is None:
        stem = os.path.splitext(os.path.basename(in_path))[0]
    else:
        stem = args.outstem

    # Load mesh
    try:
        mesh = tm.load(in_path, force='mesh')
        if not isinstance(mesh, tm.Trimesh):
            # Could be a Scene; try to merge
            if isinstance(mesh, tm.Scene) and len(mesh.geometry):
                mesh = tm.util.concatenate(list(mesh.geometry.values()))
            else:
                raise ValueError("Loaded object is not a Trimesh or usable Scene.")
    except Exception as e:
        print("ERROR: failed to read STL with trimesh:", e, file=sys.stderr)
        sys.exit(2)

    # Normalize / fix normals if desired
    if not args.no_fix_normals:
        # Re-zero to improve numerical stability; fix winding/normals if watertight or almost
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh.rezero()
            try:
                mesh.fix_normals()
            except Exception:
                pass

    if mesh.faces.shape[0] == 0 or mesh.vertices.shape[0] == 0:
        print("ERROR: mesh seems empty.", file=sys.stderr)
        sys.exit(3)

    if not mesh.is_watertight:
        print("[WARN] Mesh is not watertight; outward direction may be ambiguous. Proceeding anyway.")

    # Sampling
    n = max(1, int(args.n))
    if args.even:
        # Even-ish sampling (Poisson-like)
        points = tm.sample.sample_surface_even(mesh, n)
        # sample_surface_even returns just points; map to nearest faces for normals
        # Use nearest-face query:
        _, face_idx = mesh.nearest.on_surface(points)
    else:
        # Area-weighted uniform sampling
        points, face_idx = tm.sample.sample_surface(mesh, n)

    # Face normals at sampled points (unit)
    fn = mesh.face_normals
    normals = fn[face_idx]
    # Enforce unit length (just in case)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-30)

    # Heuristic outward check; flip if needed
    normals, flipped = ensure_outward_normals(points, normals)
    if flipped:
        print("[INFO] Flipped normals to enforce outward orientation (centroid test).")

    # Write CSVs
    pts_csv = f"{stem}.csv"
    nrm_csv = f"{stem}_normals.csv"
    write_points_csv(pts_csv, points)
    write_normals_csv(nrm_csv, normals)
    print(f"[SAVE] Wrote points → {pts_csv}  (columns: x,y,z)")
    print(f"[SAVE] Wrote normals → {nrm_csv}  (columns: nx,ny,nz)")
    print(f"[INFO] Npoints={points.shape[0]}")

    # Quick stats (matching your solver’s prints vibe)
    mins = points.min(axis=0); maxs = points.max(axis=0)
    print("[LOAD-like] point extents (min..max) per axis:")
    for k, nm in enumerate("xyz"):
        print(f"  {nm}: {mins[k]:.6g} .. {maxs[k]:.6g}")
    nlen = np.linalg.norm(normals, axis=1)
    print(f"[LOAD-like] normal lengths: min={nlen.min():.3g}, max={nlen.max():.3g}, mean={nlen.mean():.3g}")

    if args.preview:
        plot_preview(points, normals, title=f"{stem}: {points.shape[0]} samples")

if __name__ == "__main__":
    main()
