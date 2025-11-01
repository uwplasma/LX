#!/usr/bin/env python3
"""
Fetch a set/range of stellarator surfaces from your pyQSC API and save them
into ONE .npz file that main.py can train on (multi-surface).

Usage:
  python fetch_qsc_surfaces.py --api https://stellarator.physics.wisc.edu/backend/api \
      --ids 3,7,11,15 --out qsc_ids_3_7_11_15.npz

Or select by an ID range:
  python fetch_qsc_surfaces.py --api https://stellarator.physics.wisc.edu/backend/api --id-range 1:50 --stride 2 --out qsc_1to50_s2.npz

Notes:
- Uses /api/plot/<id> which returns x,y,z grids. We compute normals on the grid.
- Stores each surface with a consistent (P_bdry, N_bdry) flattened layout (row-major),
  plus (n_theta, n_phi) so plotting as imshow is trivial.
"""

import argparse, json, os, sys
import numpy as np
import requests

def _compute_normals_from_grid(X, Y, Z):
    """
    X,Y,Z: (nθ, nφ) arrays describing the surface r(θ,φ).
    Returns Nhat_grid (nθ, nφ, 3), outward normals (heuristic orientation).
    """
    # central differences in θ and φ
    # roll wraps around to preserve periodicity
    dX_dth = 0.5 * (np.roll(X, -1, axis=0) - np.roll(X, +1, axis=0))
    dY_dth = 0.5 * (np.roll(Y, -1, axis=0) - np.roll(Y, +1, axis=0))
    dZ_dth = 0.5 * (np.roll(Z, -1, axis=0) - np.roll(Z, +1, axis=0))

    dX_dph = 0.5 * (np.roll(X, -1, axis=1) - np.roll(X, +1, axis=1))
    dY_dph = 0.5 * (np.roll(Y, -1, axis=1) - np.roll(Y, +1, axis=1))
    dZ_dph = 0.5 * (np.roll(Z, -1, axis=1) - np.roll(Z, +1, axis=1))

    e_th = np.stack([dX_dth, dY_dth, dZ_dth], axis=-1)  # (nθ,nφ,3)
    e_ph = np.stack([dX_dph, dY_dph, dZ_dph], axis=-1)

    # normal via cross product; choose orientation by average radial dot
    N = np.cross(e_th, e_ph)  # (nθ,nφ,3)
    norm = np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
    Nhat = N / norm

    # heuristic to make normals outward: dot with position vector
    R = np.stack([X, Y, Z], axis=-1)
    s = np.sign(np.mean((Nhat * R).sum(-1)))
    if s < 0:
        Nhat = -Nhat
    return Nhat

def _grid_to_flat(X, Y, Z, Nhat):
    """Flatten to (Nb,3) arrays in row-major order (θ major, φ minor)."""
    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    N = Nhat.reshape(-1, 3)
    return P, N

def _fetch_surface(api_base, sid):
    """GET /api/plot/<sid>, return X,Y,Z as (nθ,nφ) floats."""
    url = f"{api_base.rstrip('/')}/plot/{sid}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    # interactive_data contains a Plotly surface spec
    js = json.loads(data["interactive_data"])
    s = js["data"][0]
    X = np.array(s["x"], dtype=float)
    Y = np.array(s["y"], dtype=float)
    Z = np.array(s["z"], dtype=float)
    # Sanity
    assert X.shape == Y.shape == Z.shape and X.ndim == 2, "Non-rectangular surface grid."
    return X, Y, Z

def parse_ids(ids_str, id_range, stride):
    ids = []
    if ids_str:
        ids.extend([int(x.strip()) for x in ids_str.split(",") if x.strip()])
    if id_range:
        a, b = id_range.split(":")
        a, b = int(a), int(b)
        ids.extend(list(range(a, b + 1, max(stride, 1))))
    # uniquify, keep order
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="https://stellarator.physics.wisc.edu/backend/api", help="Base API URL, e.g. https://stellarator.physics.wisc.edu/backend/api")
    ap.add_argument("--ids", default="", help="Comma-separated IDs, e.g. 3,7,11")
    ap.add_argument("--id-range", default="1:15", help="Inclusive range a:b, e.g. 1:50")
    ap.add_argument("--stride", type=int, default=1, help="Stride for --id-range (default 10)")
    ap.add_argument("--out", default="qsc_surfaces_from_api.npz", help="Output .npz path")
    ap.add_argument(
        "--debug-plot",
        dest="debug_plot",
        action=argparse.BooleanOptionalAction,  # gives --debug-plot / --no-debug-plot
        default=True,                            # default ON
        help="Preview surfaces before saving (default: on). Use --no-debug-plot to disable."
    )
    args = ap.parse_args()

    ids = parse_ids(args.ids, args.id_range, args.stride)
    if not ids:
        print("No IDs selected. Use --ids or --id-range.")
        sys.exit(1)

    bundle = {"ids": np.array(ids, dtype=int)}
    meta = []

    for sid in ids:
        print(f"[fetch] id={sid} ...", flush=True)
        X, Y, Z = _fetch_surface(args.api, sid)
        print(f"[fetch] id={sid}: X.shape={X.shape}, Y.shape={Y.shape}, Z.shape={Z.shape}, "
            f"X.ndim={X.ndim}, Y.ndim={Y.ndim}, Z.ndim={Z.ndim}")
        print(f"[fetch] id={sid}: X[min,max]=({X.min():+.3e},{X.max():+.3e}), "
            f"Y[min,max]=({Y.min():+.3e},{Y.max():+.3e}), Z[min,max]=({Z.min():+.3e},{Z.max():+.3e})")
        Nhat = _compute_normals_from_grid(X, Y, Z)
        P, N  = _grid_to_flat(X, Y, Z, Nhat)
        print(f"[pack ] id={sid}: P.shape={P.shape}, N.shape={N.shape}, "
            f"mean|N|={np.linalg.norm(N,axis=1).mean():.3f}")
        # store per-surface payload
        bundle[f"P_bdry_{sid}"] = P
        bundle[f"N_bdry_{sid}"] = N
        bundle[f"shape_{sid}"]  = np.array(X.shape, dtype=int)  # (nθ, nφ)
        meta.append((sid, X.shape[0], X.shape[1]))
        
    # after computing X,Y,Z and Nhat:
    if args.debug_plot:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            # small triangulation identical to your loader:
            nθ, nφ = X.shape
            V = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
            def idx(i,j): return (i % nθ) * nφ + (j % nφ)
            faces = []
            for i in range(nθ):
                for j in range(nφ):
                    i2, j2 = i+1, j
                    i3, j3 = i, j+1
                    i4, j4 = i+1, j+1
                    faces.append([idx(i,j), idx(i2,j2), idx(i3,j3)])
                    faces.append([idx(i2,j2), idx(i4,j4), idx(i3,j3)])
            F = np.asarray(faces, dtype=int)

            fig = plt.figure(figsize=(6,5))
            ax = fig.add_subplot(111, projection="3d")
            tris = Poly3DCollection(V[F], alpha=0.6)
            ax.add_collection3d(tris)
            ax.auto_scale_xyz(V[:,0], V[:,1], V[:,2])
            ax.set_title(f"pyQSC id={sid} (from API X/Y/Z)")
            plt.show()
        except Exception as e:
            print(f"[debug-plot] Failed: {e}")
            
    if args.debug_plot:
        try:
            Pr = P.reshape(nθ, nφ, 3)
            Vr = Pr.reshape(-1, 3)
            fig = plt.figure(figsize=(6,5))
            ax = fig.add_subplot(111, projection="3d")
            tris = Poly3DCollection(Vr[F], alpha=0.6)
            ax.add_collection3d(tris)
            ax.auto_scale_xyz(Vr[:,0], Vr[:,1], Vr[:,2])
            ax.set_title(f"pyQSC id={sid} (flattened P reshaped → grid)")
            plt.show()
        except Exception as e:
            print(f"[debug-plot] (reshape) Failed: {e}")
            
    if args.debug_plot:
        try:
            # Preview the LAST surface we just packed (same code-path as main)
            nT, nP = meta[-1][1], meta[-1][2]
            P = bundle[f"P_bdry_{meta[-1][0]}"]      # (Nb,3) row-major (θ-major, φ-minor)
            Pgrid = P.reshape(nT, nP, 3, order="C")  # ← SAME reshape as main
            X, Y, Z = Pgrid[...,0], Pgrid[...,1], Pgrid[...,2]
            fig = plt.figure(figsize=(6.5, 4.8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, alpha=0.9)
            # ax.set_box_aspect([X.ptp(), Y.ptp(), Z.ptp()])
            ax.set_title(f"Preview id={meta[-1][0]} (nθ={nT}, nφ={nP})")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[WARN] debug plot failed: {e}")

    np.savez(args.out, **bundle)
    print(f"[done] Saved {len(ids)} surfaces to {args.out}")
    print("       meta:", ", ".join([f"id={sid}(nθ={nT},nφ={nP})" for sid, nT, nP in meta]))

if __name__ == "__main__":
    main()
