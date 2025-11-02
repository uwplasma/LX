# _normals_debug.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import numpy as np
import jax.numpy as jnp

# ----------------------------- helpers ---------------------------------

def _unit_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(A, axis=1, keepdims=True) + eps
    return A / n

def _angle_degrees(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # angle between row vectors (in degrees), insensitive to sign (min with flipped)
    ad = _unit_rows(a, eps); bd = _unit_rows(b, eps)
    cosang = np.clip((ad * bd).sum(-1), -1.0, 1.0)
    cosang_flipped = np.clip((ad * (-bd)).sum(-1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    ang_flip = np.degrees(np.arccos(cosang_flipped))
    return np.minimum(ang, ang_flip)

def _outward_auto_flip(P: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, float]:
    # Flip to maximize outwardness vs centroid direction (like you do in main.py)
    c = P.mean(0, keepdims=True)
    R = P - c
    Rhat = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    s = (Rhat * N).sum(-1).mean()
    if s < 0:
        return -N, -s
    return N, s

# -------------------- grid-based finite difference normals --------------------

def grid_normals_2nd_order(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    # periodic central differences (2nd-order) in (θ,φ)
    G = np.stack([X, Y, Z], axis=-1)
    dθ = np.roll(G, -1, axis=0) - np.roll(G, 1, axis=0)
    dφ = np.roll(G, -1, axis=1) - np.roll(G, 1, axis=1)
    N = np.cross(dθ, dφ)
    return _unit_rows(N.reshape(-1, 3))

def grid_normals_4th_order(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    # 4th-order periodic stencil to reduce noise/dispersion
    # f' ≈ ( -f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2} ) / 12
    G = np.stack([X, Y, Z], axis=-1)
    def d_axis(A, ax):
        return (-np.roll(A, -2, ax) + 8*np.roll(A, -1, ax) - 8*np.roll(A, 1, ax) + np.roll(A, 2, ax)) / 12.0
    dθ = d_axis(G, 0)
    dφ = d_axis(G, 1)
    N = np.cross(dθ, dφ)
    return _unit_rows(N.reshape(-1, 3))

# --------------------- point-cloud normals (PCA / robust) ---------------------

def pca_normals(P: np.ndarray, k: int = 24) -> np.ndarray:
    # classic Hoppe-style kNN PCA (unoriented)
    N = np.zeros_like(P)
    # brute force kNN (small to mid Nb). For big Nb, swap to KD-tree.
    D2 = ((P[None, :, :] - P[:, None, :])**2).sum(-1)
    idx = np.argpartition(D2, kth=range(1, min(k+1, P.shape[0])), axis=1)[:, 1:k+1]
    for i in range(P.shape[0]):
        Q = P[idx[i]]
        mu = Q.mean(0)
        C = (Q - mu).T @ (Q - mu) / max(k-1, 1)
        w, V = np.linalg.eigh(C)
        N[i] = V[:, 0]
    return _unit_rows(N)

def pca_normals_multiscale(P: np.ndarray, k_list=(12, 24, 36, 48)) -> np.ndarray:
    # choose k minimizing condition number instability (heuristic)
    cand = [pca_normals(P, k) for k in k_list]
    # simple smoothness score: neighbor normal variance (lower is better)
    D2 = ((P[None, :, :] - P[:, None, :])**2).sum(-1)
    idx = np.argpartition(D2, kth=range(1, 17), axis=1)[:, 1:17]
    scores = []
    for N in cand:
        v = []
        for i in range(P.shape[0]):
            v.append(np.var((N[idx[i]] * N[i]).sum(-1)))
        scores.append(float(np.mean(v)))
    best = int(np.argmin(np.asarray(scores)))
    return cand[best]

def orient_normals_spanning_tree(P: np.ndarray, N: np.ndarray) -> np.ndarray:
    # orient using a proximity graph spanning-tree (Kazhdan-style propagation)
    D2 = ((P[None, :, :] - P[:, None, :])**2).sum(-1)
    # build a sparse graph by connecting 8 NN
    k = min(8, max(1, P.shape[0]-1))
    nbrs = np.argpartition(D2, kth=range(1, k+1), axis=1)[:, 1:k+1]
    visited = np.zeros(P.shape[0], dtype=bool)
    N_or = N.copy()
    stack = [0]
    visited[0] = True
    while stack:
        i = stack.pop()
        for j in nbrs[i]:
            if not visited[j]:
                # flip j to align with i
                if (N_or[i] @ N_or[j]) < 0:
                    N_or[j] = -N_or[j]
                visited[j] = True
                stack.append(j)
    return N_or

# --------------------- MLS quadratic fit normals (high accuracy) --------------

def _local_pca_frame(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centroid c, basis (3x3) with columns [t1, t2, n] from PCA."""
    c = Q.mean(0)
    X = Q - c
    C = (X.T @ X) / max(len(Q)-1, 1)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)  # ascending -> last is largest
    V = V[:, order]        # columns are eigenvectors
    n = V[:, 0]            # smallest eigenvalue -> normal
    # make a right-handed basis: [t1, t2, n]
    t1 = V[:, 2]
    t2 = np.cross(n, t1)
    B = np.stack([t1, t2, n], axis=1)
    return c, B, w[0]

def _k_nn_indices(P: np.ndarray, k: int) -> np.ndarray:
    D2 = ((P[None, :, :] - P[:, None, :])**2).sum(-1)
    k = min(k, max(2, P.shape[0]-1))
    return np.argpartition(D2, kth=range(1, k+1), axis=1)[:, 1:k+1]

def mls_quad_normals(P: np.ndarray, k: int = 36, sigma=None, iters: int = 1) -> np.ndarray:
    """
    Quadratic MLS fit of a local surface z = a x^2 + b xy + c y^2 + d x + e y around each point.
    Returns unit normals per point. Supports sigma as:
      - None: auto (1.2 * median NN distance)
      - float: global scalar bandwidth
      - array shape (N,): per-point bandwidths
    """
    P = np.asarray(P, dtype=np.float64)
    Np = P.shape[0]

    # kNN (brute force; swap to KD-tree if needed)
    D2 = ((P[None, :, :] - P[:, None, :])**2).sum(-1)                  # (N,N)
    # neighbors (exclude self)
    k_use = int(np.clip(k, 6, max(6, Np-1)))
    nbr_idx = np.argpartition(D2, kth=range(1, k_use+1), axis=1)[:, 1:k_use+1]  # (N,k)
    # median NN distance as fallback scale
    nn_d = np.sqrt(np.take_along_axis(D2, nbr_idx[:, :8], axis=1))      # (N,8)
    h_med = float(np.median(nn_d))

    # Prepare output
    N_out = np.zeros_like(P)

    # Normalize sigma into a helper that returns a scalar per i
    def sigma_i(i: int) -> float:
        if sigma is None:
            return 1.2 * h_med
        s = sigma
        if np.ndim(s) == 0:
            return float(s)
        # array-like per-point
        return float(s[i])

    # Fit loop (can be batched if desired)
    for i in range(Np):
        J = nbr_idx[i]                                 # (k,)
        Q = P[J]                                       # (k,3)
        p0 = P[i]
        X = Q - p0                                     # (k,3)

        # Build local tangent frame via PCA (robust orientation for x,y)
        C = (X.T @ X) / max(k_use-1, 1)
        w_eig, V = np.linalg.eigh(C)
        # normal is smallest eigvec
        n0 = V[:, 0]
        # ensure right-handed (tangent basis u,v)
        u = V[:, 1] / (np.linalg.norm(V[:, 1]) + 1e-18)
        v = V[:, 2] / (np.linalg.norm(V[:, 2]) + 1e-18)
        # re-orthonormalize n0
        n0 = np.cross(u, v)
        n0 = n0 / (np.linalg.norm(n0) + 1e-18)

        # project neighbors to local (u,v,n0) frame
        x = X @ u                                      # (k,)
        y = X @ v                                      # (k,)
        z = X @ n0                                     # (k,)

        # weights with per-point sigma
        s = sigma_i(i)
        r2 = x*x + y*y
        w = np.exp(-r2 / (2.0 * s * s))                # (k,)

        # quadratic MLS: z ≈ a x^2 + b x y + c y^2 + d x + e y
        # Build design matrix
        A = np.stack([x*x, x*y, y*y, x, y], axis=1)    # (k,5)
        W = w[:, None]                                 # (k,1)
        Aw = A * W                                     # (k,5)
        zw = z * w                                     # (k,)

        # normal equations (5x5)
        M = Aw.T @ A                                   # (5,5)
        bvec = Aw.T @ z                                # (5,)

        # regularize lightly for stability
        lam = 1e-10 * np.trace(M)
        M_reg = M + lam * np.eye(5)
        coeff = np.linalg.solve(M_reg, bvec)           # [a,b,c,d,e]

        # Normal of the quadratic surface at the origin:
        # f(x,y) = a x^2 + b x y + c y^2 + d x + e y => ∇f(0,0) = [d, e]
        # Surface normal n ≈ normalize( n0 - d * u - e * v )
        d_lin, e_lin = float(coeff[3]), float(coeff[4])
        n = n0 - d_lin * u - e_lin * v
        n = n / (np.linalg.norm(n) + 1e-18)

        # optional smoothing iterations: reproject and refit quickly (one-step)
        n_acc = n.copy()
        for _ in range(max(0, iters-1)):
            # small refinement of tangent frame about n_acc
            u2 = u - (u @ n_acc) * n_acc
            u2 /= (np.linalg.norm(u2) + 1e-18)
            v2 = np.cross(n_acc, u2)
            v2 /= (np.linalg.norm(v2) + 1e-18)
            x2 = X @ u2; y2 = X @ v2; z2 = X @ n_acc
            r2 = x2*x2 + y2*y2
            w2 = np.exp(-r2 / (2.0 * s * s))
            A2 = np.stack([x2*x2, x2*y2, y2*y2, x2, y2], axis=1)
            Aw2 = A2 * w2[:, None]
            zw2 = z2 * w2
            M2 = Aw2.T @ A2
            b2 = Aw2.T @ z2
            lam2 = 1e-10 * np.trace(M2)
            coeff2 = np.linalg.solve(M2 + lam2 * np.eye(5), b2)
            d2, e2 = float(coeff2[3]), float(coeff2[4])
            n_acc = n_acc - d2 * u2 - e2 * v2
            n_acc = n_acc / (np.linalg.norm(n_acc) + 1e-18)

        N_out[i] = n_acc

    return _unit_rows(N_out)

def pca_confidence(P: np.ndarray, k: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (normals, confidence) where confidence = 1 - (λ0/λ1) in [0,1),
    λ0 <= λ1 <= λ2 are local PCA eigenvalues. Larger is better (clear normal).
    """
    idx = _k_nn_indices(P, k)
    N = np.zeros_like(P)
    conf = np.zeros((P.shape[0],), dtype=float)
    for i in range(P.shape[0]):
        Q = P[idx[i]]
        c = Q.mean(0)
        X = Q - c
        C = (X.T @ X) / max(len(Q)-1, 1)
        w, V = np.linalg.eigh(C)
        n = V[:, 0]
        N[i] = n / (np.linalg.norm(n)+1e-12)
        # λ0 small vs λ1: flatness & separation
        conf[i] = 1.0 - float(w[0] / (w[1] + 1e-18))
        conf[i] = max(0.0, min(1.0, conf[i]))
    return _unit_rows(N), conf

# ----------------------------- comparison API --------------------------------

def compare_normals_report(
    name: str,
    P_bdry: np.ndarray,
    N_provided: Optional[np.ndarray],
    *,
    grid_shape: Optional[Tuple[int, int]] = None,
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    Z: Optional[np.ndarray] = None,
    use_fourth_order: bool = True,
    do_pca_check: bool = True,
    print_hist: bool = True,
) -> Dict[str, float]:
    """
    Returns dict of RMS/median angle (deg) between provided normals (if any) and:
      - FD grid normals (2nd or 4th order) when grid provided
      - PCA normals for point clouds
    Also enforces outwardness with centroid test and reports mean outwardness.
    """
    out = {}

    # Outwardness check (and flip)
    if N_provided is not None:
        Np, s = _outward_auto_flip(P_bdry, N_provided)
        out["outwardness_mean_dot"] = float(s)
        N_provided = Np

    # Grid path
    if grid_shape is not None and X is not None:
        if use_fourth_order:
            N_fd = grid_normals_4th_order(X, Y, Z)
            out["fd_scheme"] = 4
        else:
            N_fd = grid_normals_2nd_order(X, Y, Z)
            out["fd_scheme"] = 2
        if N_provided is not None:
            ang = _angle_degrees(np.asarray(N_provided), np.asarray(N_fd))
            out["grid_vs_provided_rms_deg"]    = float(np.sqrt(np.mean(ang**2)))
            out["grid_vs_provided_median_deg"] = float(np.median(ang))
            if print_hist:
                hist, edges = np.histogram(ang, bins=[0,1,2,5,10,20,45,90])
                print(f"[NORMALS:{name}] Grid v Provided angle hist (deg): bins{list(edges)} counts{hist.tolist()}")
        # Tag outwardness of FD normals as well
        N_fd, s_fd = _outward_auto_flip(P_bdry, N_fd)
        out["fd_outwardness_mean_dot"] = float(s_fd)

    # Point-cloud PCA path
    if do_pca_check:
        N_pca = pca_normals(np.asarray(P_bdry), k=24)
        N_pca = orient_normals_spanning_tree(np.asarray(P_bdry), N_pca)
        if N_provided is not None:
            ang = _angle_degrees(np.asarray(N_provided), N_pca)
            out["pca_vs_provided_rms_deg"]    = float(np.sqrt(np.mean(ang**2)))
            out["pca_vs_provided_median_deg"] = float(np.median(ang))
            if print_hist:
                hist, edges = np.histogram(ang, bins=[0,1,2,5,10,20,45,90])
                print(f"[NORMALS:{name}] PCA v Provided angle hist (deg): bins{list(edges)} counts{hist.tolist()}")
        N_pca, s_p = _outward_auto_flip(P_bdry, N_pca)
        out["pca_outwardness_mean_dot"] = float(s_p)

    # Quick confirmation prints
    src = "PROVIDED" if (N_provided is not None) else "ESTIMATED"
    print(f"[NORMALS:{name}] Using normals source = {src}")
    for k, v in out.items():
        print(f"[NORMALS:{name}] {k} = {v:.6g}")
    return out
