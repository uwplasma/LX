#!/usr/bin/env python3
"""
Robust boundary exporter for two input formats:

1) SLAM .npz  -> {theta_grid, phi_grid, R_grid, Z_grid}
   - Builds X,Y,Z and computes outward unit normals NX,NY,NZ
   - Optional periodic resampling to (ntheta x nphi)

2) SFLM .npy  -> array of shape (6, N): [x,y,z,nx,ny,nz]
   - Validates/renormalizes provided normals (optional)
   - No resampling (prints a warning if requested)

Outputs one of:
  --ext npz : np.savez with keys X,Y,Z,(NX,NY,NZ)
  --ext npy : np.save(dict(...), allow_pickle=True)
  --ext csv : points CSV + optional *_normals.csv

The output file name is derived from the input:
  <input_stem>[_coarse]?[_normals]?.<ext>

Debug prints and assertions included to ease troubleshooting.
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np

# ----------------------------- Debug utils ----------------------------- #

def dbg_header(msg: str):
    print("\n" + "="*72)
    print(msg)
    print("="*72)

def dbg_kv(key: str, val):
    print(f"[DBG] {key}: {val}")

def dbg_arr(name: str, a: np.ndarray, samples: int = 3):
    a = np.asarray(a)
    print(f"[DBG] {name}: shape={a.shape}, dtype={a.dtype}, min={np.nanmin(a):.6g}, max={np.nanmax(a):.6g}")
    flat = a.reshape(-1)
    s = min(samples, flat.size)
    if s > 0:
        print(f"      {name}[:{s}] = {flat[:s]}")

# ------------------------------ I/O utils ------------------------------ #

def save_single_file(out_path: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                     NX: np.ndarray | None = None, NY: np.ndarray | None = None,
                     NZ: np.ndarray | None = None) -> None:
    ext = out_path.lower().split(".")[-1]
    have_normals = (NX is not None) and (NY is not None) and (NZ is not None)
    print(f"[INFO] Saving -> {out_path} (normals={have_normals})")
    if ext == "npz":
        if have_normals:
            np.savez(out_path, X=X, Y=Y, Z=Z, NX=NX, NY=NY, NZ=NZ)
        else:
            np.savez(out_path, X=X, Y=Y, Z=Z)
    elif ext == "npy":
        obj = {"X": X, "Y": Y, "Z": Z}
        if have_normals:
            obj.update({"NX": NX, "NY": NY, "NZ": NZ})
        np.save(out_path, obj, allow_pickle=True)
    elif ext == "csv":
        P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        np.savetxt(out_path, P, delimiter=",", header="x,y,z", comments="")
        if have_normals:
            Pn = np.stack([NX, NY, NZ], axis=-1).reshape(-1, 3)
            stem = out_path.rsplit(".", 1)[0]
            # avoid "..._normals_normals.csv"
            if stem.endswith("_normals"):
                norm_path = stem + "_vectors.csv"      # companion normals file
            else:
                norm_path = stem + "_normals.csv"
            np.savetxt(norm_path, Pn, delimiter=",", header="nx,ny,nz", comments="")
            print(f"[INFO] Saved companion normals file -> {norm_path}")
    else:
        raise ValueError(f"Unsupported output extension: .{ext}")

def derive_out_path(inp: str, ext: str, coarse: bool, normals: bool) -> str:
    base = os.path.basename(inp)
    stem = os.path.splitext(base)[0]
    suffix = []
    if coarse:
        suffix.append("coarse")
    if normals:
        suffix.append("normals")
    sfx = ("_" + "_".join(suffix)) if suffix else ""
    return f"{stem}{sfx}.{ext}"

# -------------------------- SLAM (npz) loader -------------------------- #

def load_slam_npz(path: str):
    dbg_header("LOAD SLAM NPZ")
    nz = np.load(path)
    keys = list(nz.keys())
    dbg_kv("npz keys", keys)

    required = ["theta_grid", "phi_grid", "R_grid", "Z_grid"]
    for k in required:
        if k not in nz:
            raise ValueError(f"[ERR] Missing key '{k}' in {path}. Found keys: {keys}")

    theta = np.asarray(nz["theta_grid"])
    phi   = np.asarray(nz["phi_grid"])
    Rg    = np.asarray(nz["R_grid"])
    Zg    = np.asarray(nz["Z_grid"])

    dbg_arr("theta_grid", theta)
    dbg_arr("phi_grid",   phi)
    dbg_arr("R_grid",     Rg)
    dbg_arr("Z_grid",     Zg)

    if Rg.shape != Zg.shape:
        raise AssertionError(f"[ERR] R and Z shapes must match. R{Rg.shape} != Z{Zg.shape}")

    # Handle 1D theta/phi with 2D R/Z (separable grid)
    if theta.ndim == 1 and phi.ndim == 1 and Rg.ndim == 2:
        n0, n1 = Rg.shape
        if len(phi) == n0 and len(theta) == n1:
            axis_phi, axis_theta = 0, 1
        elif len(theta) == n0 and len(phi) == n1:
            axis_phi, axis_theta = 1, 0
        else:
            raise AssertionError(
                f"[ERR] Could not match lengths: len(theta)={len(theta)}, len(phi)={len(phi)}, R shape={Rg.shape}"
            )

        # Broadcast angle grids to match R/Z
        if axis_phi == 0 and axis_theta == 1:
            phi2d   = np.broadcast_to(phi[:, None],   Rg.shape)
            theta2d = np.broadcast_to(theta[None, :], Rg.shape)
        else:
            theta2d = np.broadcast_to(theta[:, None], Rg.shape)
            phi2d   = np.broadcast_to(phi[None, :],   Rg.shape)

        dbg_kv("axis_theta", axis_theta)
        dbg_kv("axis_phi", axis_phi)
        dbg_arr("theta2d", theta2d, 2)
        dbg_arr("phi2d",   phi2d,   2)
        return ("slam", (theta, phi, theta2d, phi2d, Rg, Zg, axis_theta, axis_phi))

    raise AssertionError(
        "[ERR] This loader expects 1D theta/phi and 2D R/Z. "
        "Ping me if you truly have 2D theta/phi (non-separable grids)."
    )

# -------- periodic central difference along a chosen axis (SLAM) ------- #

def dd_along_axis_periodic(A: np.ndarray, coord_1d: np.ndarray, axis: int) -> np.ndarray:
    A = np.asarray(A)
    coord = np.asarray(coord_1d).astype(float)
    # Unwrap angles if needed
    coord = np.unwrap(coord)

    n = coord.size
    if A.shape[axis] != n:
        raise AssertionError(f"[ERR] coord length {n} != A.shape[{axis}]={A.shape[axis]}")

    Ap = np.roll(A, -1, axis=axis)
    Am = np.roll(A,  1, axis=axis)

    coord_p = np.roll(coord, -1)
    coord_m = np.roll(coord,  1)
    denom = coord_p - coord_m
    denom = np.where(denom == 0.0, 1e-15, denom)

    shp = [1]*A.ndim; shp[axis] = n
    denom_b = denom.reshape(shp)
    return (Ap - Am) / denom_b

def slam_xyz_normals(theta1d, phi1d, theta2d, phi2d, Rg, Zg, axis_theta, axis_phi):
    dbg_header("SLAM -> XYZ + normals")
    # Cartesian embedding
    cphi = np.cos(phi2d); sphi = np.sin(phi2d)
    X = Rg * cphi
    Y = Rg * sphi
    Z = Zg
    dbg_arr("X", X); dbg_arr("Y", Y); dbg_arr("Z", Z)

    # Parametric derivatives
    R_t = dd_along_axis_periodic(Rg, theta1d, axis_theta)
    Z_t = dd_along_axis_periodic(Zg, theta1d, axis_theta)
    R_p = dd_along_axis_periodic(Rg, phi1d,   axis_phi)
    Z_p = dd_along_axis_periodic(Zg, phi1d,   axis_phi)

    S_t = np.stack([R_t * cphi, R_t * sphi, Z_t], axis=-1)
    S_p = np.stack([R_p * cphi - Rg * sphi, R_p * sphi + Rg * cphi, Z_p], axis=-1)
    N = np.cross(S_t, S_p, axis=-1)
    Nmag = np.sqrt(np.sum(N*N, axis=-1, keepdims=True)) + 1e-15
    N_hat = N / Nmag

    # Orient outward via centroid test
    P = np.stack([X, Y, Z], axis=-1)
    centroid = np.mean(P.reshape(-1, 3), axis=0)
    dots = np.sum((P - centroid) * N_hat, axis=-1)
    if np.mean(dots) < 0.0:
        N_hat = -N_hat

    NX, NY, NZ = N_hat[..., 0], N_hat[..., 1], N_hat[..., 2]
    dbg_arr("NX", NX); dbg_arr("NY", NY); dbg_arr("NZ", NZ)
    return X, Y, Z, NX, NY, NZ

# -------------------------- SLAM resampling (opt) ---------------------- #

def _unwrap_periodic(x, period=2*np.pi):
    x = np.asarray(x, float)
    x0 = (x - x[0]) % period
    return x[0] + x0

def _periodic_interp1d(values, x, x_new, period=2*np.pi, axis=-1):
    values = np.asarray(values)
    x = _unwrap_periodic(np.asarray(x, float), period=period)
    x_new = _unwrap_periodic(np.asarray(x_new, float), period=period)

    v = np.take(values, indices=range(values.shape[axis]), axis=axis)
    v_tile = np.concatenate([v, v, v], axis=axis)
    x_tile = np.concatenate([x - period, x, x + period])

    v_move = np.moveaxis(v_tile, axis, -1)  # (..., N*3)
    out_shape = list(v_move.shape); out_shape[-1] = x_new.shape[0]
    v_flat = v_move.reshape(-1, v_move.shape[-1])
    out_flat = np.empty((v_flat.shape[0], x_new.shape[0]), dtype=values.dtype)
    for i in range(v_flat.shape[0]):
        out_flat[i, :] = np.interp(x_new, x_tile, v_flat[i, :])
    out = out_flat.reshape(out_shape)
    out = np.moveaxis(out, -1, axis)
    return out

def resample_slam_grid(theta1d, phi1d, Rg, Zg, axis_theta, axis_phi, ntheta_out, nphi_out):
    dbg_header("SLAM RESAMPLE")
    dbg_kv("input Rg.shape", Rg.shape)
    theta_out = np.linspace(0.0, 2*np.pi, int(ntheta_out), endpoint=False)
    phi_out   = np.linspace(0.0, 2*np.pi, int(nphi_out),   endpoint=False)
    dbg_arr("theta_out", theta_out); dbg_arr("phi_out", phi_out)

    R1 = _periodic_interp1d(Rg, theta1d, theta_out, axis=axis_theta)
    Z1 = _periodic_interp1d(Zg, theta1d, theta_out, axis=axis_theta)
    R2 = _periodic_interp1d(R1, phi1d,   phi_out,   axis=axis_phi)
    Z2 = _periodic_interp1d(Z1, phi1d,   phi_out,   axis=axis_phi)

    shape_out = list(Rg.shape)
    shape_out[axis_phi]   = nphi_out
    shape_out[axis_theta] = ntheta_out
    R2 = R2.reshape(shape_out)
    Z2 = Z2.reshape(shape_out)

    # build 2D angle grids
    if axis_phi == 0 and axis_theta == 1:
        phi2d   = np.broadcast_to(phi_out[:, None],   R2.shape)
        theta2d = np.broadcast_to(theta_out[None, :], R2.shape)
    else:
        theta2d = np.broadcast_to(theta_out[:, None], R2.shape)
        phi2d   = np.broadcast_to(phi_out[None, :],   R2.shape)

    dbg_kv("output R.shape", R2.shape)
    return theta_out, phi_out, theta2d, phi2d, R2, Z2

# --------------------------- SFLM (npy) loader ------------------------- #

def load_sflm_npy(path: str):
    dbg_header("LOAD SFLM NPY")
    arr = np.load(path, allow_pickle=True)
    dbg_arr("loaded_npy", arr, 6)
    if not isinstance(arr, np.ndarray):
        raise AssertionError("[ERR] Unexpected npy content (not an ndarray).")
    if arr.ndim == 2 and arr.shape[0] == 6:
        x, y, z, nx, ny, nz = arr
    elif arr.ndim == 1 and arr.shape[0] == 6:
        # Could be object array with 6 elements
        x, y, z, nx, ny, nz = list(arr)
    else:
        raise AssertionError(f"[ERR] Expected shape (6, N). Got {arr.shape}")

    # Validate shapes
    N = x.size
    for name, v in [("y", y), ("z", z), ("nx", nx), ("ny", ny), ("nz", nz)]:
        assert v.size == N, f"[ERR] {name} has size {v.size} != {N}"

    # Normalize normals (in case they are not perfectly unit)
    nmag = np.sqrt(nx*nx + ny*ny + nz*nz) + 1e-15
    nx_n = nx / nmag; ny_n = ny / nmag; nz_n = nz / nmag

    # Debug prints
    for nm, v in [("x", x), ("y", y), ("z", z), ("nx", nx_n), ("ny", ny_n), ("nz", nz_n)]:
        dbg_arr(nm, v, samples=5)

    # Return as 1D arrays; saving will reshape/flatten as needed
    return ("sflm", (x, y, z, nx_n, ny_n, nz_n))

# ------------------------------- Driver -------------------------------- #

def detect_and_load(path: str):
    ext = os.path.splitext(path)[1].lower()
    dbg_kv("input path", path)
    dbg_kv("input ext", ext)

    # First try NPZ (SLAM)
    if ext == ".npz":
        try:
            return load_slam_npz(path)
        except Exception as e:
            print(f"[WARN] NPZ handler failed: {e}")
            raise

    # Then try NPY (SFLM)
    if ext == ".npy":
        try:
            return load_sflm_npy(path)
        except Exception as e:
            print(f"[WARN] NPY handler failed: {e}")
            raise

    # Unknown extension: try both
    try:
        return load_slam_npz(path)
    except Exception as e_npz:
        print(f"[WARN] NPZ parse failed: {e_npz}")
    try:
        return load_sflm_npy(path)
    except Exception as e_npy:
        print(f"[WARN] NPY parse failed: {e_npy}")
        raise AssertionError(f"[ERR] Could not parse {path} as SLAM (.npz) or SFLM (.npy).")

def main():
    ap = argparse.ArgumentParser(description="Export boundary (SLAM .npz or SFLM .npy) to Cartesian with normals.")
    ap.add_argument("--in", dest="inp", type=str, required=True, help="Input file (.npz for SLAM, .npy for SFLM)")
    ap.add_argument("--ext", type=str, default="csv", choices=["npz","npy","csv"], help="Output format")
    ap.add_argument("--ntheta", type=int, default=None, help="(SLAM only) output theta count")
    ap.add_argument("--nphi",   type=int, default=None, help="(SLAM only) output phi count")
    ap.add_argument("--no-normals", action="store_true", help="(SLAM only) skip normal computation")
    ap.add_argument("--preview", action="store_true", default=True, help="Quick 3D preview (requires matplotlib)")
    ap.add_argument("--npoints", type=int, default=1500, help="(SFLM only) downsample to ~npoints by striding before save/preview")
    args = ap.parse_args()

    kind, payload = detect_and_load(args.inp)

    if kind == "slam":
        (theta1d, phi1d, theta2d, phi2d, Rg, Zg, axis_theta, axis_phi) = payload
        coarse = False
        # Optional resampling
        if (args.ntheta is not None) or (args.nphi is not None):
            ntheta_out = args.ntheta or theta1d.size
            nphi_out   = args.nphi   or phi1d.size
            theta1d, phi1d, theta2d, phi2d, Rg, Zg = resample_slam_grid(
                theta1d, phi1d, Rg, Zg, axis_theta, axis_phi, ntheta_out, nphi_out
            )
            coarse = True

        X, Y, Z, NX, NY, NZ = slam_xyz_normals(theta1d, phi1d, theta2d, phi2d, Rg, Zg, axis_theta, axis_phi)
        if args.no_normals:
            NX = NY = NZ = None

        out_path = derive_out_path(args.inp, args.ext, coarse=coarse, normals=False)
        save_single_file(out_path, X, Y, Z, NX, NY, NZ)

        if args.preview:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                fig = plt.figure(figsize=(7,6))
                ax = fig.add_subplot(1,1,1, projection='3d')
                ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, alpha=1.0, shade=False)
                ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title("SLAM surface + normals")

                # --- overlay normals (subsample to avoid clutter) ---
                try:
                    # pick a stride so roughly ~500 arrows max
                    ntheta, nphi = X.shape
                    stride_theta = max(1, ntheta // 20)
                    stride_phi   = max(1, nphi   // 25)
                    Xs  = X[::stride_theta, ::stride_phi]
                    Ys  = Y[::stride_theta, ::stride_phi]
                    Zs  = Z[::stride_theta, ::stride_phi]
                    Nxs = (NX if NX is not None else np.zeros_like(X))[::stride_theta, ::stride_phi]
                    Nys = (NY if NY is not None else np.zeros_like(Y))[::stride_theta, ::stride_phi]
                    Nzs = (NZ if NZ is not None else np.zeros_like(Z))[::stride_theta, ::stride_phi]
                    # scale arrows by a small factor of typical grid size
                    scale = 10.0
                    ax.quiver(Xs, Ys, Zs, Nxs, Nys, Nzs, length=np.nanmax([np.ptp(X),np.ptp(Y),np.ptp(Z)]) / scale, normalize=True)
                except Exception as _e:
                    print(f"[WARN] normals quiver failed: {_e}")

                # equal aspect:
                xs,ys,zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
                xmid, ymid, zmid = np.mean(xs), np.mean(ys), np.mean(zs)
                half = max(xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]) / 2.0
                ax.set_xlim3d([xmid-half, xmid+half]); ax.set_ylim3d([ymid-half, ymid+half]); ax.set_zlim3d([zmid-half, zmid+half])
                plt.tight_layout(); plt.show()
            except Exception as e:
                print(f"[WARN] preview failed: {e}")

    elif kind == "sflm":
        (x, y, z, nx, ny, nz) = payload
        coarse = False  # needed for derive_out_path
        # Flattened point cloud; normals provided
        X = x; Y = y; Z = z; NX = nx; NY = ny; NZ = nz

        # --- SFLM downsampling by stride if requested ---
        if args.npoints is not None:
            N = X.size
            if args.npoints <= 0:
                print("[WARN] --npoints must be > 0; ignoring.")
            elif args.npoints < N:
                step = max(1, N // args.npoints)
                sel = slice(0, N, step)
                X, Y, Z = X[sel], Y[sel], Z[sel]
                NX, NY, NZ = NX[sel], NY[sel], NZ[sel]
                # If we overshot, trim to exactly npoints (optional)
                if X.size > args.npoints:
                    X, Y, Z = X[:args.npoints], Y[:args.npoints], Z[:args.npoints]
                    NX, NY, NZ = NX[:args.npoints], NY[:args.npoints], NZ[:args.npoints]
                print(f"[INFO] SFLM downsample: kept {X.size} / {N} points (stride≈{step})")
            else:
                print("[INFO] --npoints >= N; keeping all points.")

        out_path = derive_out_path(args.inp, args.ext, coarse=coarse, normals=False)
        save_single_file(out_path, X, Y, Z, NX, NY, NZ)

        if args.preview:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                fig = plt.figure(figsize=(7,6))
                ax = fig.add_subplot(1,1,1, projection='3d')
                ax.scatter(X, Y, Z, s=1.5)
                ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title("SFLM point cloud + normals")

                # --- overlay normals (subsample to avoid clutter) ---
                try:
                    N = X.size
                    step = max(1, N // 1000)  # show up to ~1000 arrows
                    Xs, Ys, Zs = X[::step], Y[::step], Z[::step]
                    Nxs, Nys, Nzs = NX[::step], NY[::step], NZ[::step]
                    scale = 10.0
                    ax.quiver(Xs, Ys, Zs, Nxs, Nys, Nzs, length=np.nanmax([np.ptp(X),np.ptp(Y),np.ptp(Z)]) / scale, normalize=True)
                except Exception as _e:
                    print(f"[WARN] normals quiver failed: {_e}")

                xs,ys,zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
                xmid, ymid, zmid = np.mean(xs), np.mean(ys), np.mean(zs)
                half = max(xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]) / 2.0
                ax.set_xlim3d([xmid-half, xmid+half]); ax.set_ylim3d([ymid-half, ymid+half]); ax.set_zlim3d([zmid-half, zmid+half])
                plt.tight_layout(); plt.show()

            except Exception as e:
                print(f"[WARN] preview failed: {e}")
    else:
        raise AssertionError(f"[ERR] Unknown kind: {kind}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[EXC] Uncaught exception:", repr(e))
        import traceback; traceback.print_exc()
        print("\n[HINT] Please copy-paste everything above so I can debug quickly.")
