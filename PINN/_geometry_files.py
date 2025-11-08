# _geometry_files.py
from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional
import os
import numpy as np
import jax.numpy as jnp

from _multisurface import SurfaceItem

# ------------------------- Optional dependency (meshes) -------------------------
try:
    import trimesh  # pip install trimesh
except Exception:
    trimesh = None

# ------------------------------ Small utilities --------------------------------
def _attach_normals_source(obj, src: str):
    try:
        setattr(obj, "normals_source", src)
    except Exception:
        pass
    return obj

def _to_jnp(x: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(x, dtype=jnp.float64)

def _norm_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(A, axis=1, keepdims=True) + eps
    return A / n

class _SurfacePack:
    def __init__(self, name: str, P_bdry: jnp.ndarray, N_bdry: jnp.ndarray,
                 inside_mask_fn: Callable[[jnp.ndarray], jnp.ndarray]):
        self.name = name
        self.P_bdry = P_bdry
        self.N_bdry = N_bdry
        self.inside_mask_fn = inside_mask_fn  # (M,3)->bool[M]

def _bbox_inside_fn(P_bdry_np: np.ndarray, pad: float = 0.02):
    """Lightweight interior test: axis-aligned bbox with a small pad."""
    mins = P_bdry_np.min(axis=0) - pad
    maxs = P_bdry_np.max(axis=0) + pad
    def fn(P: jnp.ndarray) -> jnp.ndarray:
        return (
            (P[:,0] >= mins[0]) & (P[:,0] <= maxs[0]) &
            (P[:,1] >= mins[1]) & (P[:,1] <= maxs[1]) &
            (P[:,2] >= mins[2]) & (P[:,2] <= maxs[2])
        )
    return fn

def build_surfaces_from_npz(npz_path: str) -> List[_SurfacePack]:
    data = np.load(npz_path)
    ids = list(map(int, data["ids"]))
    out = []
    for sid in ids:
        P = jnp.asarray(data[f"P_bdry_{sid}"])  # (Nb,3)
        N = jnp.asarray(data[f"N_bdry_{sid}"])  # (Nb,3)
        shp = tuple(map(int, np.asarray(data.get(f"shape_{sid}", [0, 0]))))  # (nθ,nφ) if present
        inside = _bbox_inside_fn(np.asarray(P))
        pack = _SurfacePack(
            name=f"pyQSC#{sid}",
            P_bdry=P,
            N_bdry=N,
            inside_mask_fn=inside,
        )
        # Attach shape onto the pack so we can transfer it later:
        pack.shape_thetaphi = shp if all(shp) else None
        out.append(pack)
    return out

def build_surfaces_from_files_or_npz(files_list):
    outs = []
    for item in files_list:
        if str(item).lower().endswith(".npz"):
            packs = build_surfaces_from_npz(item)
            for p in packs:
                outs.append(SurfaceItem(
                    name=p.name,
                    P_bdry=p.P_bdry,
                    N_bdry=p.N_bdry,
                    inside_mask_fn=p.inside_mask_fn,
                    shape_thetaphi=getattr(p, "shape_thetaphi", None),  # NEW
                ))
        else:
            outs.extend(build_surfaces_from_files([item]))
    return outs


# =============================== LOADERS =======================================

# ---- Mesh (STL/PLY/OBJ) ----
def _load_mesh(path: str):
    if trimesh is None:
        raise RuntimeError("`trimesh` is required for mesh files. Run: pip install trimesh")
    m = trimesh.load_mesh(path, process=True)
    if not isinstance(m, trimesh.Trimesh):
        if hasattr(m, "geometry") and len(m.geometry) > 0:
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        else:
            raise RuntimeError(f"Unsupported mesh container at {path}")
    # ensure normals
    m.rezero()
    m.remove_unreferenced_vertices()
    m.fix_normals()
    return m

def _inside_mask_from_mesh(mesh) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def inside_mask(P_box: jnp.ndarray) -> jnp.ndarray:
        P_np = np.asarray(P_box, dtype=np.float64)
        try:
            contained = mesh.contains(P_np)  # watertight needed
        except Exception:
            # Fallback: classify by signed distance sign if available
            if hasattr(mesh, "nearest"):
                d, _, _ = trimesh.proximity.closest_point(mesh, P_np)
                contained = d < 1e-9
            else:
                raise
        return jnp.asarray(contained, dtype=bool)
    return inside_mask

def _surface_from_mesh(path: str, name: str,
                       sample: Optional[int], every: Optional[int]) -> SurfaceItem:
    m = _load_mesh(path)
    V0 = np.asarray(m.vertices, dtype=np.float64)
    F  = np.asarray(m.faces, dtype=np.int32)

    # --- robust area-weighted face normals → vertex normals ---
    fn = np.cross(V0[F[:,1]] - V0[F[:,0]], V0[F[:,2]] - V0[F[:,0]])
    fn_len = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-18
    fn_hat = fn / fn_len

    # accumulate area-weighted normals at vertices
    Vn = np.zeros_like(V0)
    np.add.at(Vn, F[:,0], fn)
    np.add.at(Vn, F[:,1], fn)
    np.add.at(Vn, F[:,2], fn)

    # optional Laplacian smoothing of vertex normals (1–2 its)
    for _ in range(2):
        Vn_sm = Vn.copy()
        # simple umbrella operator via adjacency
        A = m.vertex_adjacency_graph.tocsr()
        deg = np.array(A.sum(axis=1)).ravel()
        neigh_sum = A @ Vn_sm
        Vn = 0.5*Vn + 0.5*(neigh_sum / (deg[:,None] + 1e-18))

    Vn = Vn / (np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-18)

    # --- proceed with your sampling/subsampling logic ---
    V = V0
    if every and every > 1:
        V = V[::int(every)]
    if sample and sample > 0 and sample < len(V):
        rng = np.random.default_rng(0)
        idx_pick = rng.choice(len(V), size=sample, replace=False)
        V = V[idx_pick]
        # map normals via NN in original vertex set
        try:
            from scipy.spatial import cKDTree
            idx = cKDTree(V0).query(V, k=1)[1]
        except Exception:
            d2 = ((V[:, None, :] - V0[None, :, :])**2).sum(-1)
            idx = np.argmin(d2, axis=1)
        N = Vn[idx]
    else:
        # same size → aligned
        N = Vn

    N = _norm_rows(N)
    inside_mask_fn = _inside_mask_from_mesh(m)
    return SurfaceItem(name=name, P_bdry=_to_jnp(V), N_bdry=_to_jnp(N), inside_mask_fn=inside_mask_fn)


# ---- Grid (npz/npy with X,Y,Z arrays) ----
def _load_grid_npz_or_npy(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        nz = np.load(path)
        X, Y, Z = nz["X"], nz["Y"], nz["Z"]
    elif ext == ".npy":
        obj = np.load(path, allow_pickle=True).item()
        X, Y, Z = obj["X"], obj["Y"], obj["Z"]
    else:
        raise ValueError("Grid surfaces must be .npz or dict-like .npy with keys X,Y,Z")
    return X.astype(np.float64), Y.astype(np.float64), Z.astype(np.float64)

def _triangulate_periodic_grid(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                               periodic_theta=True, periodic_phi=True) -> Tuple[np.ndarray, np.ndarray]:
    nθ, nφ = X.shape
    V = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    def idx(i, j):
        if periodic_theta: i %= nθ
        if periodic_phi:   j %= nφ
        return i * nφ + j

    faces = []
    i_max = nθ if periodic_theta else nθ - 1
    j_max = nφ if periodic_phi   else nφ - 1
    for i in range(i_max):
        for j in range(j_max):
            i1, j1 = i, j
            i2, j2 = i+1, j
            i3, j3 = i, j+1
            i4, j4 = i+1, j+1
            faces.append([idx(i1,j1), idx(i2,j2), idx(i3,j3)])
            faces.append([idx(i2,j2), idx(i4,j4), idx(i3,j3)])
    F = np.asarray(faces, dtype=np.int32)
    return V, F

def _solid_angle_tri(a: np.ndarray, b: np.ndarray, c: np.ndarray, p: np.ndarray) -> float:
    ra = a - p; rb = b - p; rc = c - p
    la = np.linalg.norm(ra); lb = np.linalg.norm(rb); lc = np.linalg.norm(rc)
    num = np.dot(ra, np.cross(rb, rc))
    den = la*lb*lc + np.dot(ra, rb)*lc + np.dot(rb, rc)*la + np.dot(rc, ra)*lb
    return 2.0 * np.arctan2(num, den + 1e-18)

def _winding_number_mask(V: np.ndarray, F: np.ndarray, Q: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    A = V[F[:,0]]; B = V[F[:,1]]; C = V[F[:,2]]
    # vectorized over faces per point (loop over points only)
    wn = np.zeros(Q.shape[0], dtype=np.float64)
    for i, p in enumerate(Q):
        ang = _solid_angle_tri(A, B, C, p)  # (Nt,) via numpy broadcasting inside
        wn[i] = abs(ang.sum())
    return wn > (4.0*np.pi - tol)

def _grid_normals(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    # central differences on param grid (θ,φ), then cross to get normal; normalize
    # handle periodic wrap consistent with triangulation flags (assume periodic both)
    dθ = np.roll(np.stack([X,Y,Z], -1), -1, axis=0) - np.roll(np.stack([X,Y,Z], -1), 1, axis=0)
    dφ = np.roll(np.stack([X,Y,Z], -1), -1, axis=1) - np.roll(np.stack([X,Y,Z], -1), 1, axis=1)
    N = np.cross(dθ, dφ)
    N = N.reshape(-1, 3)
    return _norm_rows(N)

def _surface_from_grid(path: str, name: str,
                       periodic_theta: bool, periodic_phi: bool) -> SurfaceItem:
    X, Y, Z = _load_grid_npz_or_npy(path)
    nθ, nφ = X.shape
    # Try to load analytic normals if the file contains NX,NY,NZ.
    N = None
    ext = os.path.splitext(path)[1].lower()
    if ext in (".npz", ".npy"):
        try:
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                keys = data.files
                if "NX" in keys and "NY" in keys and "NZ" in keys:
                    NX = data["NX"]
                    NY = data["NY"]
                    NZ = data["NZ"]
                    N = np.stack([NX, NY, NZ], axis=-1).reshape(-1, 3)
            elif isinstance(data, dict):
                if "NX" in data and "NY" in data and "NZ" in data:
                    NX, NY, NZ = data["NX"], data["NY"], data["NZ"]
                    N = np.stack([NX, NY, NZ], axis=-1).reshape(-1, 3)
        except Exception:
            pass
    # Fall back to finite‐difference normals if no analytic normals are present.
    if N is None:
        try:
            from _normals_debug import grid_normals_4th_order
            N = grid_normals_4th_order(X, Y, Z)   # use 4th-order instead of 2nd
        except Exception:
            N = _grid_normals(X, Y, Z)            # safe fallback
    # Triangulate and enforce outward orientation of normals
    V, F = _triangulate_periodic_grid(X, Y, Z, periodic_theta, periodic_phi)
    from _normals_debug import _outward_auto_flip
    N, _ = _outward_auto_flip(V, N)
    N = _norm_rows(N)

    # Inside/outside test using winding numbers on triangulated grid
    def inside_mask_fn(P_box: jnp.ndarray) -> jnp.ndarray:
        Q = np.asarray(P_box, dtype=np.float64)
        msk = _winding_number_mask(V, F, Q, tol=1e-3)
        return jnp.asarray(msk, dtype=bool)

    # Pack and return
    item = SurfaceItem(
        name=name,
        P_bdry=_to_jnp(V),
        N_bdry=_to_jnp(N),
        inside_mask_fn=inside_mask_fn,
        shape_thetaphi=(int(nθ), int(nφ)),
    )
    _attach_normals_source(item, "grid-analytic" if ext in (".npz", ".npy") and N is not None else "grid-fd")
    return item

# ---- Points (csv/txt/npy/npz) ----
def _load_points_any(path: str) -> np.ndarray:
    """
    Load Nx3 points from .npy/.npz/.csv/.txt.
    - .npz: accepts 'P' (Nx3) or ('X','Y','Z') arrays (any shape, will be flattened).
    - .npy: expects (N,3) numeric array.
    - .csv/.txt: accepts either raw 3 numeric columns OR a header with x,y,z (any case).
    """
    import os
    import numpy as np

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[-1] != 3:
            raise ValueError(f"{path}: expected (N,3) array")
        return arr.astype(np.float64)

    if ext == ".npz":
        nz = np.load(path)
        if "P" in nz:
            P = nz["P"].reshape(-1, 3).astype(np.float64)
            return P
        # allow X,Y,Z with any shape
        for keys in (("X","Y","Z"), ("x","y","z")):
            if all(k in nz for k in keys):
                X, Y, Z = nz[keys[0]], nz[keys[1]], nz[keys[2]]
                P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float64)
                return P
        raise ValueError(f"{path}: .npz must have 'P' or ('X','Y','Z').")

    if ext in (".csv", ".txt"):
        delim = "," if ext == ".csv" else None

        # 1) Try raw numeric table (no header)
        try:
            arr = np.loadtxt(path, delimiter=delim, ndmin=2)
            if arr.shape[1] < 3:
                raise ValueError(f"{path}: need at least 3 columns for x,y,z")
            return arr[:, :3].astype(np.float64)
        except ValueError:
            pass

        # 2) Try numeric with a single header row
        try:
            arr = np.loadtxt(path, delimiter=delim, ndmin=2, skiprows=1)
            if arr.shape[1] >= 3:
                return arr[:, :3].astype(np.float64)
        except ValueError:
            pass

        # 3) Structured read with named columns (header like 'x,y,z' or 'X,Y,Z', etc.)
        data = np.genfromtxt(path, delimiter=delim, names=True, dtype=None, encoding="utf-8", autostrip=True)
        if data.size == 0 or data.dtype.names is None:
            raise ValueError(f"{path}: could not parse CSV/TXT")

        names = [n.lower() for n in data.dtype.names]
        def _col(name_opts):
            for opt in name_opts:
                if opt in names:
                    return opt
            return None

        xk = _col(("x", "x_coord", "xcoordinate"))
        yk = _col(("y", "y_coord", "ycoordinate"))
        zk = _col(("z", "z_coord", "zcoordinate"))
        if not (xk and yk and zk):
            raise ValueError(f"{path}: header must include columns named x,y,z (case-insensitive). Found {data.dtype.names}")

        X = np.asarray(data[xk], dtype=np.float64)
        Y = np.asarray(data[yk], dtype=np.float64)
        Z = np.asarray(data[zk], dtype=np.float64)
        return np.stack([X, Y, Z], axis=-1)

    raise ValueError(f"Unsupported file type: {ext}")

def _load_normals_optional(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    import os, numpy as np
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        N = np.load(path).astype(np.float64)
    elif ext == ".npz":
        nz = np.load(path)
        key = "N" if "N" in nz else list(nz.keys())[0]
        N = nz[key].astype(np.float64)
    elif ext in (".csv", ".txt"):
        delim = "," if ext == ".csv" else None
        # raw numeric
        try:
            arr = np.loadtxt(path, delimiter=delim, ndmin=2)
        except ValueError:
            # header names
            data = np.genfromtxt(path, delimiter=delim, names=True, dtype=None, encoding="utf-8", autostrip=True)
            names = [n.lower() for n in data.dtype.names or []]
            def pick(*opts):
                for o in opts:
                    if o in names:
                        return np.asarray(data[o], dtype=np.float64)
                return None
            nx = pick("nx", "n_x"); ny = pick("ny", "n_y"); nz = pick("nz", "n_z")
            if nx is None or ny is None or nz is None:
                raise ValueError(f"{path}: normals CSV/TXT must have nx,ny,nz columns")
            N = np.stack([nx, ny, nz], axis=-1)
        else:
            if arr.shape[1] < 3:
                raise ValueError(f"{path}: normals need 3 columns")
            N = arr[:, :3].astype(np.float64)
    else:
        raise ValueError(f"Unsupported normals file: {ext}")

    if N.shape[-1] != 3:
        raise ValueError("Normals must be (N,3).")
    return N

def _estimate_normals_pca(P: np.ndarray, k: int = 16) -> np.ndarray:
    Np = P.shape[0]
    # brute-force kNN (fine for typical boundary sizes; swap to faiss if huge)
    d2 = ((P[None, :, :] - P[:, None, :])**2).sum(-1)
    k = min(k, max(2, Np-1))
    idx = np.argpartition(d2, kth=range(1, k+1), axis=1)[:, 1:k+1]  # (N,k)
    N = np.zeros_like(P, dtype=np.float64)
    for i in range(Np):
        Q = P[idx[i]]
        mu = Q.mean(0)
        C = (Q - mu).T @ (Q - mu) / max(k-1, 1)
        w, V = np.linalg.eigh(C)
        n = V[:, 0]  # smallest eigenvector
        N[i] = n
    # outward-ish orientation by centroid
    c = P.mean(0)
    sign = np.sign(((P - c) * N).sum(-1, keepdims=True) + 1e-12)
    return _norm_rows(N * sign)

def _inside_mask_from_points(P: np.ndarray, N: np.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Heuristic inside test: nearest boundary point normal test.
    Marks inside if (q - p_nn)·n_nn < 0 assuming normals are outward.
    Works OK for many closed shells; not as robust as a watertight mesh.
    """
    P0 = P.copy()
    N0 = _norm_rows(N.copy())

    def inside_mask(points: jnp.ndarray) -> jnp.ndarray:
        Q = np.asarray(points, dtype=np.float64)
        M, Nb = Q.shape[0], P0.shape[0]
        # chunk to limit memory
        chunk = max(1, 40000 // max(Nb, 1))
        lab = np.zeros(M, dtype=bool)
        s = 0
        while s < M:
            e = min(M, s + chunk)
            d2 = ((Q[s:e, None, :] - P0[None, :, :])**2).sum(-1)
            j = np.argmin(d2, axis=1)
            vec = Q[s:e, :] - P0[j, :]
            sign = (vec * N0[j, :]).sum(-1)
            lab[s:e] = (sign < 0.0)
            s = e
        return jnp.asarray(lab)
    return inside_mask

def _surface_from_points(path: str, name: str,
                         normals_path: str | None,
                         normals_mode: str = "auto") -> SurfaceItem:
    
    # --- local spacing helper (8-NN median distance) ---
    def _local_spacing(P: np.ndarray, k_nn: int = 8) -> np.ndarray:
        D2 = ((P[None, :, :] - P[:, None, :])**2).sum(-1)
        idx = np.argpartition(D2, kth=range(1, min(k_nn+1, P.shape[0])), axis=1)[:, 1:k_nn+1]
        d = np.sqrt(np.take_along_axis(D2, idx, axis=1))
        return np.median(d, axis=1)  # (N,)
    
    P = _load_points_any(path)
    N_file = _load_normals_optional(normals_path) if normals_path else None

    # Normalize mode string
    normals_mode = str(normals_mode or "auto").lower()
    if normals_mode not in ("auto", "provided", "estimate"):
        normals_mode = "auto"

    # Decide source according to mode
    if normals_mode == "provided" and (N_file is not None and N_file.shape[0] == P.shape[0]):
        src = "file"
        N = N_file
    elif normals_mode == "estimate":
        src = "pca"   # we’ll run the estimator below
        N = None
    else:
        # auto: prefer provided if usable, else estimate
        if N_file is not None and N_file.shape[0] == P.shape[0]:
            src = "file"
            N = N_file
        else:
            src = "pca"
            N = None

    # if src == "pca":
    #     # Multiscale PCA + orientation (more stable across noise/scales)
    #     from _normals_debug import pca_normals_multiscale, orient_normals_spanning_tree
    #     N = pca_normals_multiscale(P, k_list=(12, 24, 36, 48))
    #     N = orient_normals_spanning_tree(P, N)
    if src == "pca":
        from _normals_debug import (
            pca_normals_multiscale,
            pca_confidence,
            mls_quad_normals,
            orient_normals_spanning_tree,
        )
        Nb = P.shape[0]

        # 1) local spacing and adaptive neighbors
        h = _local_spacing(P, k_nn=8)                  # (N,)
        h_med = float(np.median(h))
        # Adaptive neighbor counts, clipped to a safe range
        k_pca = int(np.clip(0.03 * Nb, 16, 64))        # for confidence eval
        k_mls = int(np.clip(0.06 * Nb, 24, 96))

        # 2) PCA backbone (multiscale for stability)
        N_pca = pca_normals_multiscale(P, k_list=(12, 24, 36, 48))

        # 3) Confidence → decide where MLS is needed
        _, conf = pca_confidence(P, k=k_pca)           # conf ∈ [0,1] (lower = worse PCA plane)
        # tighter threshold near high curvature / noisy patches
        tau = 0.18

        # 4) MLS normals with an adaptive Gaussian kernel width
        #    sigma ~ 1.2 * local spacing (capped for robustness)
        sigma = 1.2 * np.clip(h, 0.5*h_med, 2.0*h_med) # (N,)
        # Two smoothing iters is a sweet spot; too many over-smooths features.
        N_mls = mls_quad_normals(P, k=k_mls, sigma=sigma, iters=2)

        # 5) Blend: MLS where PCA confidence is poor
        use_mls = (conf < tau)
        N = np.where(use_mls[:, None], N_mls, N_pca)

        # 6) Global orientation propagation on the blended field
        N = orient_normals_spanning_tree(P, N)

    from _normals_debug import _outward_auto_flip
    # Try a toroidal outward cue: radial from z-axis
    Rxy = np.linalg.norm(P[:, :2], axis=1, keepdims=True) + 1e-12
    Rhat = np.concatenate([P[:, :2] / Rxy, np.zeros_like(Rxy)], axis=1)  # (N,3)
    s_radial = float((Rhat * N).sum(axis=1).mean())
    if s_radial < 0.0:
        N = -N
    # Safety: run your centroid-based flip as well
    N, _ = _outward_auto_flip(P, N)
    
    # --- one-pass bilateral smoothing on the normal field ---
    try:
        from _normals_debug import bilateral_normal_smooth
        N = bilateral_normal_smooth(P, N, k=16, sigma_spatial=1.0*h_med, sigma_angle=np.deg2rad(20.0))
    except Exception:
        pass

    inside_mask_fn = _inside_mask_from_points(P, N)  # heuristic inside test

    pack = SurfaceItem(name=name, P_bdry=_to_jnp(P), N_bdry=_to_jnp(N), inside_mask_fn=inside_mask_fn)
    pack.normals_source = src
    pack.shape_thetaphi = None

    # Compare: estimated (MLS/PCA) vs provided (VMEC), regardless of which source is active.
    if normals_path:
        try:
            from _normals_debug import (
                _angle_degrees,
                pca_normals_multiscale, pca_confidence, mls_quad_normals, orient_normals_spanning_tree,
            )
            N_file_cmp = _load_normals_optional(normals_path)
            if N_file_cmp is not None and N_file_cmp.shape == P.shape:
                # Build an estimated normals field for comparison (doesn't affect the active N)
                Np = pca_normals_multiscale(P, k_list=(12, 24, 36, 48))
                _, conf = pca_confidence(P, k=24)
                Nm = mls_quad_normals(P, k=36, sigma=None, iters=1)
                N_est = np.where((conf < 0.25)[:, None], Nm, Np)
                N_est = orient_normals_spanning_tree(P, N_est)
                # Make both outward-ish so angle is sign-insensitive
                from _normals_debug import _outward_auto_flip
                N_est, _ = _outward_auto_flip(P, N_est)
                N_file_cmp, _ = _outward_auto_flip(P, N_file_cmp)
                ang = _angle_degrees(N_file_cmp, N_est)
                rms = float(np.sqrt(np.mean(ang**2)))
                med = float(np.median(ang))
                p95 = float(np.percentile(ang, 95))
                print(f"[NORMALS:{name}] points-estimated vs provided (deg): rms={rms:.3f} med={med:.3f} p95={p95:.3f}")
        except Exception as e:
            print(f"[NORMALS:{name}] comparison failed: {e!r}")

    # Log which source ended up active for this surface
    try:
        print(f"[NORMALS:{name}] ACTIVE normals source = {pack.normals_source.upper()}")
    except Exception:
        pass

    return pack

# ============================ PUBLIC DISPATCH ==================================
def build_surfaces_from_files(files_list: List[Dict]) -> List[SurfaceItem]:
    """
    Accepts entries with either:
      - format="mesh"   OR kind="mesh"     -> path: *.stl/*.ply/*.obj
           optional: sample (int), every (int)
      - format="grid"   OR kind="grid"     -> path: .npz/.npy with X,Y,Z
           optional: periodic_theta (bool, default true), periodic_phi (bool, default true)
      - format="points" OR kind="xyz"      -> path: .csv/.txt/.npy/.npz of Nx3 points
           optional: normals_path (file with Nx3 normals); else PCA normals
    Returns: List[SurfaceItem(name, P_bdry, N_bdry, inside_mask_fn)]
    """
    out: List[SurfaceItem] = []
    for i, f in enumerate(files_list):
        name = f.get("name", f"surf_{i}")
        fmt = str(f.get("format", f.get("kind", "mesh"))).lower()
        path = str(f["path"])

        if fmt == "mesh":
            sample = f.get("sample", None)
            every  = f.get("every", None)
            out.append(_surface_from_mesh(path, name,
                                          int(sample) if sample else None,
                                          int(every) if every else None))

        elif fmt == "grid":
            periodic_theta = bool(f.get("periodic_theta", True))
            periodic_phi   = bool(f.get("periodic_phi", True))
            out.append(_surface_from_grid(path, name, periodic_theta, periodic_phi))

        elif fmt in ("points", "xyz"):
            normals_path = f.get("normals_path", None) or None
            normals_mode = f.get("normals_mode", "auto")  # "auto" | "provided" | "estimate"
            out.append(_surface_from_points(path, name, normals_path, normals_mode))

        else:
            raise RuntimeError(f"Unknown file format/kind={fmt!r} for entry #{i} ({name})")
    return out
