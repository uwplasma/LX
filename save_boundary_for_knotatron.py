#!/usr/bin/env python3
"""
Generate (X,Y,Z) surface points and outward unit normals (NX,NY,NZ)
for a self-intersecting, winding tube around a (p,q) torus knot.

- Pure NumPy implementation with a parallel-transport (Bishop) frame.
- Suitable for trefoil (2,3) and any coprime (p,q) torus knot.
- User can specify tube minor radius, base torus radii, and grid sizes.

Examples
--------
# Trefoil (2,3) with tube radius 0.1:
python knot_torus_surface.py --p 2 --q 3 --R 1.0 --r0 0.35 --tube_r 0.10 \
    --nu 400 --ntheta 96 --out trefoil_23.npz

# Another torus knot:
python knot_torus_surface.py --p 3 --q 5 --tube_r 0.08 --out p3q5.csv
"""

from __future__ import annotations
import argparse
import numpy as np
import sys

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


# ---------------------------- Geometry core ---------------------------- #

def torus_knot_centerline(u: np.ndarray, p: int, q: int, R: float, r0: float) -> np.ndarray:
    """
    Parametric (p,q) torus-knot centerline on a standard torus.

    Parameters
    ----------
    u   : array of shape (n_u,)
    p,q : coprime integers; the knot wraps p times around toroidal and q poloidal
    R   : major radius of base torus (distance from origin to tube centerline)
    r0  : minor amplitude of the base torus (radius of the underlying circle)

    Returns
    -------
    C : array (n_u, 3) with centerline points
    """
    # Standard embedding of (p,q) torus knot:
    # angle_tor = p*u, angle_pol = q*u
    # r = R + r0*cos(q*u), z = r0*sin(q*u), x = r*cos(p*u), y = r*sin(p*u)
    pu = p * u
    qu = q * u
    r = R + r0 * np.cos(qu)
    x = r * np.cos(pu)
    y = r * np.sin(pu)
    z = r0 * np.sin(qu)
    return np.stack([x, y, z], axis=-1)


def _safe_norm(v: np.ndarray, axis=-1, keepdims=False, eps=1e-15) -> np.ndarray:
    return np.sqrt(np.sum(v*v, axis=axis, keepdims=keepdims)) + eps


def cumulative_arclength(C: np.ndarray, closed: bool = True) -> np.ndarray:
    """
    C: (N,3). Returns s of shape (N+1,) if closed (wraps to first), else (N,)
    For closed=True, s[-1] is total length and corresponds to C[0] again.
    """
    if closed:
        d = np.linalg.norm(np.roll(C, -1, axis=0) - C, axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        return s  # length N+1
    else:
        d = np.linalg.norm(C[1:] - C[:-1], axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        return s  # length N

def resample_polyline_periodic(C: np.ndarray, n_out: int) -> np.ndarray:
    """
    Equal-arclength resample of a closed polyline C (N,3) -> (n_out,3).
    Uses linear interpolation on x,y,z vs arclength with periodic wrap.
    """
    N = C.shape[0]
    s = cumulative_arclength(C, closed=True)          # (N+1,)
    L = s[-1]
    # Make periodic by appending C[0] at end for each coord:
    C_per = np.vstack([C, C[0]])
    # Target equal spacing in [0, L)
    s_target = np.linspace(0.0, L, n_out, endpoint=False)
    # Interp each coord independently (np.interp expects increasing x):
    X = np.interp(s_target, s, C_per[:, 0])
    Y = np.interp(s_target, s, C_per[:, 1])
    Z = np.interp(s_target, s, C_per[:, 2])
    return np.stack([X, Y, Z], axis=-1)


def parallel_transport_frame(C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a parallel-transport (Bishop) frame (T, N, B) along polyline C.

    This avoids the unnecessary twist of a Frenet frame at low curvature.

    Parameters
    ----------
    C : (n_u, 3) centerline points, assumed closed (first ~ last)

    Returns
    -------
    T, N, B : each is (n_u, 3), orthonormal at every u
    """
    n = C.shape[0]
    Cp = np.roll(C, -1, axis=0)
    Cm = np.roll(C,  1, axis=0)
    T = Cp - Cm
    T /= _safe_norm(T, axis=1, keepdims=True)

    z_axis = np.array([0.0, 0.0, 1.0])
    ref = z_axis if np.abs(np.dot(T[0], z_axis)) <= 0.9 else np.array([1.0, 0.0, 0.0])

    N = np.zeros_like(C)
    B = np.zeros_like(C)

    N0 = ref - np.dot(ref, T[0]) * T[0]
    if np.linalg.norm(N0) < 1e-12:
        ref = np.array([0.0, 1.0, 0.0])
        N0 = ref - np.dot(ref, T[0]) * T[0]
        if np.linalg.norm(N0) < 1e-12:
            N0 = np.cross(T[0], [1.0, 0.0, 0.0])
    N[0] = N0 / _safe_norm(N0)
    B[0] = np.cross(T[0], N[0])

    for i in range(1, n):
        v = np.cross(T[i-1], T[i])
        s = _safe_norm(v)
        c = np.clip(np.dot(T[i-1], T[i]), -1.0, 1.0)

        if s < 1e-12:
            N[i] = N[i-1]
            B[i] = B[i-1]
        else:
            v /= s
            def rod(w):
                return w*c + np.cross(v, w)*s + v*np.dot(v, w)*(1.0 - c)
            N[i] = rod(N[i-1])
            B[i] = np.cross(T[i], N[i])

        N[i] -= np.dot(N[i], T[i]) * T[i]
        N[i] /= _safe_norm(N[i])
        B[i] = np.cross(T[i], N[i])
        B[i] /= _safe_norm(B[i])

    return T, N, B


def tube_surface_from_centerline(
    C: np.ndarray, N: np.ndarray, B: np.ndarray, tube_r: float, n_theta: int
):
    """
    Wrap a radius=tube_r tube around centerline C using frame (N,B).

    Returns
    -------
    X,Y,Z,NX,NY,NZ : each shape (n_u, n_theta)
        Surface points and outward unit normals on a regular grid.
    """
    n_u = C.shape[0]
    thetas = np.linspace(0.0, 2.0*np.pi, n_theta, endpoint=False)
    ct = np.cos(thetas)[None, :, None]
    st = np.sin(thetas)[None, :, None]
    Rdir = N[:, None, :] * ct + B[:, None, :] * st
    P = C[:, None, :] + tube_r * Rdir
    normals = Rdir / _safe_norm(Rdir, axis=2, keepdims=True)
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    NX, NY, NZ = normals[..., 0], normals[..., 1], normals[..., 2]
    return X, Y, Z, NX, NY, NZ


def build_torus_knot_surface(
    p: int = 2,
    q: int = 3,
    R: float = 1.0,
    r0: float = 0.35,
    tube_r: float = 0.10,
    nu: int = 400,
    ntheta: int = 96,
    oversample: int = None,
):
    """
    High-level convenience: make a (p,q) torus-knot tube surface.

    Parameters
    ----------
    p, q   : integers (coprime for a single closed knot)
    R      : base torus major radius
    r0     : base torus minor amplitude (controls knot "non-axisymmetry")
    tube_r : tube minor radius (surface thickness around centerline)
    nu     : number of samples along the knot
    ntheta : number of samples around each cross-section
    oversample: dense samples used before resampling (default: 10*nu).

    Returns
    -------
    X, Y, Z, NX, NY, NZ : arrays of shape (nu, ntheta)
    """
    if oversample is None:
        oversample = max(10 * nu, 8 * q * p)  # ensure plenty of samples

    u_dense = np.linspace(0.0, 2.0*np.pi, oversample, endpoint=False)
    C_dense = torus_knot_centerline(u_dense, p=p, q=q, R=R, r0=r0)

    # Equal-arclength resample to nu points (periodic/closed)
    C = resample_polyline_periodic(C_dense, n_out=nu)

    # Frame + tube
    T, N, B = parallel_transport_frame(C)
    return tube_surface_from_centerline(C, N, B, tube_r=tube_r, n_theta=ntheta)


# ------------------------------- I/O ---------------------------------- #

def save_single_file(out_path: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                     NX: np.ndarray | None = None, NY: np.ndarray | None = None,
                     NZ: np.ndarray | None = None) -> None:
    ext = out_path.lower().split(".")[-1]
    have_normals = (NX is not None) and (NY is not None) and (NZ is not None)
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
            Np = np.stack([NX, NY, NZ], axis=-1).reshape(-1, 3)
            norm_path = out_path.rsplit(".", 1)[0] + "_normals.csv"
            np.savetxt(norm_path, Np, delimiter=",", header="nx,ny,nz", comments="")
            print(f"[INFO] Saved companion normals file to {norm_path}")
    else:
        raise ValueError(f"Unsupported output extension: .{ext}")


def _equal_aspect_3d(ax):
    xs, ys, zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xmid, ymid, zmid = np.mean(xs), np.mean(ys), np.mean(zs)
    half = max(xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]) / 2.0
    ax.set_xlim3d([xmid - half, xmid + half])
    ax.set_ylim3d([ymid - half, ymid + half])
    ax.set_zlim3d([zmid - half, zmid + half])


def quick_plot(X, Y, Z, title="Torus-knot tube (surface)"):
    if not _HAVE_MPL:
        print("[WARN] matplotlib not available; skipping preview.", file=sys.stderr)
        return
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, alpha=1.0, shade=False)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    _equal_aspect_3d(ax)
    plt.tight_layout()
    plt.show()


# ------------------------------- CLI ---------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Generate a (p,q) torus-knot tube surface with normals.")
    ap.add_argument("--p", type=int, default=2, help="Torus-knot toroidal wraps (e.g., 2 for trefoil (2,3)).")
    ap.add_argument("--q", type=int, default=3, help="Torus-knot poloidal wraps (e.g., 3 for trefoil (2,3)).")
    ap.add_argument("--R", type=float, default=1.0, help="Base torus major radius.")
    ap.add_argument("--r0", type=float, default=0.45, help="Base torus minor amplitude for the knot path.")
    ap.add_argument("--tube_r", type=float, default=0.35, help="Tube (surface) minor radius.")
    ap.add_argument("--nu", type=int, default=128, help="Samples along knot (u-direction).")
    ap.add_argument("--ntheta", type=int, default=16, help="Samples around tube cross-section (theta-direction).")
    ap.add_argument("--out", type=str, default="knot_tube.csv", help="Output: .npz | .npy | .csv")
    ap.add_argument("--preview", action="store_true", default=True, help="Show a quick 3D preview (default: True).")
    args = ap.parse_args()

    X, Y, Z, NX, NY, NZ = build_torus_knot_surface(
        p=args.p, q=args.q, R=args.R, r0=args.r0,
        tube_r=args.tube_r, nu=args.nu, ntheta=args.ntheta,
        oversample=None
    )
    save_single_file(args.out, X, Y, Z, NX, NY, NZ)
    print(f"[OK] Saved surface + normals to {args.out} | shapes: X{X.shape}, NX{NX.shape}")

    if args.preview:
        quick_plot(X, Y, Z, title=f"(p,q)=({args.p},{args.q}), R={args.R}, r0={args.r0}, tube_r={args.tube_r}")


if __name__ == "__main__":
    main()
