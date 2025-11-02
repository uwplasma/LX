"""
Plotting helpers for 3D surface and vector overlays.
These functions are independent of the model; they operate on provided arrays.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def _downsample_grid_for_quiver(X, Y, Z, V, step_theta=6, step_phi=8):
    """
    Downsample θ,φ grid for quiver clarity.
      X,Y,Z: (nθ,nφ)
      V:     (nθ,nφ,3) vectors to plot
    Returns flattened (x,y,z,u,v,w) for ax.quiver.
    """
    Xs = X[::step_theta, ::step_phi]
    Ys = Y[::step_theta, ::step_phi]
    Zs = Z[::step_theta, ::step_phi]
    Vs = V[::step_theta, ::step_phi, :]

    x = Xs.ravel(); y = Ys.ravel(); z = Zs.ravel()
    u = Vs[...,0].ravel(); v = Vs[...,1].ravel(); w = Vs[...,2].ravel()
    return x, y, z, u, v, w


def fix_matplotlib_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_surface_with_vectors_ax(ax, X, Y, Z, Gmag, Nhat, Gvec=None,
                                 title="∇u on torus surface",
                                 cmap="viridis", quiver_len=0.15,
                                 step_theta=6, step_phi=8,
                                 vmin=None, vmax=None, plot_normals=True,
                                 grad_color="red", normals_color="black",
                                 surf_offset=0.0):
    """
    Plot a colored torus surface and vector overlays.
    'surf_offset' shifts ONLY the surface inward along outward normal so
    quivers appear outside. Quivers are anchored at the true boundary.

    Returns a ScalarMappable for colorbars.
    """
    from matplotlib import colormaps as _cmaps
    import numpy as _np
    _cmap = _cmaps.get_cmap(cmap)

    gmin = _np.min(Gmag) if vmin is None else vmin
    gmax = _np.max(Gmag) if vmax is None else vmax
    normed = (Gmag - gmin) / (gmax - gmin + 1e-12)
    facecolors = _cmap(normed)

    if surf_offset != 0.0:
        Xs = X - surf_offset * Nhat[..., 0]
        Ys = Y - surf_offset * Nhat[..., 1]
        Zs = Z - surf_offset * Nhat[..., 2]
    else:
        Xs, Ys, Zs = X, Y, Z

    ax.plot_surface(Xs, Ys, Zs, facecolors=facecolors, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)

    if plot_normals:
        xn, yn, zn, un, vn, wn = _downsample_grid_for_quiver(X, Y, Z, Nhat, step_theta, step_phi)
        ax.quiver(xn, yn, zn, un, vn, wn,
                  length=quiver_len, normalize=True, linewidth=1.0,
                  color=normals_color, alpha=0.7)

    if Gvec is not None:
        xg, yg, zg, ug, vg, wg = _downsample_grid_for_quiver(X, Y, Z, Gvec, step_theta, step_phi)
        ax.quiver(xg, yg, zg, ug, vg, wg,
                  length=quiver_len, normalize=True, linewidth=1.2,
                  color=grad_color, alpha=0.95)

    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1,1,1))

    m = plt.cm.ScalarMappable(cmap=_cmap)
    m.set_array(Gmag)
    m.set_clim(gmin, gmax)
    return m

def draw_box_edges(ax, xmin, xmax, ymin, ymax, zmin, zmax, lw=1.0, alpha=0.6):
    """Draw a rectangular box as 12 edges."""
    # 8 corners
    C = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin],
        [xmin, ymax, zmin], [xmax, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax],
        [xmin, ymax, zmax], [xmax, ymax, zmax],
    ])
    # edges as pairs of indices
    E = [(0,1),(0,2),(1,3),(2,3),
         (4,5),(4,6),(5,7),(6,7),
         (0,4),(1,5),(2,6),(3,7)]
    for i,j in E:
        ax.plot([C[i,0],C[j,0]], [C[i,1],C[j,1]], [C[i,2],C[j,2]],
                linewidth=lw, alpha=alpha, color="k")

def _unit(x, eps=1e-12):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)

def _local_pca_normals(P: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Estimate per-point normals via local PCA (smallest singular vector of neighbors).
    Returns normals with roughly consistent outward orientation (centroid heuristic).
    Robust to small N, k >= N, and degenerate neighborhoods.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P must have shape (N,3)")
    Np = P.shape[0]
    if Np == 0:
        return np.zeros((0, 3), dtype=float)

    # Choose effective neighborhood size
    # Need at least 3 neighbors for a stable plane fit; clamp to [3, Np-1]
    if Np <= 3:
        # Fallback: all normals = 0 (or arbitrary unit z); we pick z and orient later
        centroid = P.mean(axis=0, keepdims=True)
        v = P - centroid
        n = np.tile(np.array([0.0, 0.0, 1.0]), (Np, 1))
        # orient rough-outward
        signs = np.sum(n * v, axis=1) < 0
        n[signs] *= -1.0
        return n

    k_eff = int(max(3, min(int(k), Np - 1)))

    # Distances (squared) and kNN (brute force; replace with KDTree if needed)
    dists2 = np.sum((P[:, None, :] - P[None, :, :])**2, axis=-1)
    np.fill_diagonal(dists2, np.inf)              # exclude self
    # We want the k_eff smallest indices along axis=1
    # argpartition kth must be < Np; we need the (k_eff-1) index as the pivot
    idx = np.argpartition(dists2, kth=k_eff-1, axis=1)[:, :k_eff]  # (Np,k_eff)

    centroid = P.mean(axis=0, keepdims=True)
    estN = np.zeros_like(P)

    for i in range(Np):
        nb = P[idx[i]]            # (k_eff,3)
        mu = nb.mean(axis=0)
        X  = nb - mu

        # Handle pathological case: all neighbors identical → X rank 0
        # SVD is stable; we still take the last right-singular vector.
        try:
            _, S, VT = np.linalg.svd(X, full_matrices=False)
            n = VT[-1]
        except np.linalg.LinAlgError:
            # fallback: use vector from centroid to point
            n = P[i] - centroid[0]
            n_norm = np.linalg.norm(n)
            if n_norm == 0:
                n = np.array([0.0, 0.0, 1.0])
            else:
                n = n / n_norm

        # Orient roughly outward (away from centroid)
        v = P[i] - centroid[0]
        if np.dot(n, v) < 0:
            n = -n

        # Normalize (protect against zero)
        n_norm = np.linalg.norm(n)
        if n_norm == 0.0:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / n_norm

        estN[i] = n

    return estN

# === DEBUG STATS helper ======================================================
def _stat_line(name, x, *, fmt=".3f", p95=True):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f"[NORMAL-CHECK] {name}: no finite data")
        return
    if p95:
        print(f"[NORMAL-CHECK] {name}: mean={x.mean():{fmt}}, p95={np.percentile(x,95):{fmt}}, max={x.max():{fmt}}")
    else:
        print(f"[NORMAL-CHECK] {name}: mean={x.mean():{fmt}}, std={x.std():{fmt}}, min={x.min():{fmt}}, max={x.max():{fmt}}")

def debug_plot_normals(
    P_bdry, N_bdry, *,
    k_pca: int = 20,
    quiver_max: int = 1500,
    quiver_len: float = 0.06,
    cmap: str = "viridis",
    title_prefix: str = "Boundary normals",
    align_provided_to_pca: bool = False,   # NEW: force provided normals to point like PCA normals
    show_plots: bool = True
):
    """
    P_bdry: (Nb,3) boundary points (numpy or jax array)
    N_bdry: (Nb,3) provided normals
    k_pca:  neighborhood size for PCA normal estimation
    align_provided_to_pca: if True, flips N_bdry so dot(N, N_ref) >= 0
    """
    P = np.asarray(P_bdry, dtype=float)
    N = _unit(np.asarray(N_bdry, dtype=float))
    Nb_pts = P.shape[0]

    # Estimate reference normals via local PCA
    N_ref = _local_pca_normals(P, k=k_pca)

    # Optionally align provided normals with PCA direction (remove in/out flips)
    if align_provided_to_pca:
        signs = np.sign(np.sum(N * N_ref, axis=-1, keepdims=True))
        signs[~np.isfinite(signs)] = 1.0
        signs[signs == 0.0] = 1.0
        N = N * signs

    # Metrics
    dots = np.sum(N * N_ref, axis=-1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)     # ignore sign after optional align; measures angular mismatch only
    angles_deg = np.degrees(np.arccos(dots))   # 0° identical (or opposite), 90° tangent

    N_tan = N - (np.sum(N * N_ref, axis=-1, keepdims=True)) * N_ref
    tan_leak = np.linalg.norm(N_tan, axis=-1)

    N_len = np.linalg.norm(N, axis=-1)         # <-- correct array; not Nb_pts

    # === 3D overlay plot ===
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    axR = fig.add_subplot(1, 2, 2, projection="3d")

    sc = axL.scatter(P[:,0], P[:,1], P[:,2], c=angles_deg, s=2, cmap=cmap)
    cb = fig.colorbar(sc, ax=axL, shrink=0.8)
    cb.set_label("angle(N, N_ref) [deg]")
    axL.set_title(f"{title_prefix}: angle to PCA normal")

    step = max(1, Nb_pts // min(Nb_pts, quiver_max))
    QP = P[::step]; QN = N[::step]
    axL.quiver(QP[:,0], QP[:,1], QP[:,2], QN[:,0], QN[:,1], QN[:,2],
               length=quiver_len, normalize=True, linewidth=0.6)

    sc2 = axR.scatter(P[:,0], P[:,1], P[:,2], c=tan_leak, s=2, cmap=cmap)
    cb2 = fig.colorbar(sc2, ax=axR, shrink=0.8)
    cb2.set_label(r"$\|N - (N\cdot N_{\rm ref}) N_{\rm ref}\|$  (tangent leakage)")
    axR.set_title(f"{title_prefix}: tangent leakage")
    QNref = N_ref[::step]
    axR.quiver(QP[:,0], QP[:,1], QP[:,2], QNref[:,0], QNref[:,1], QNref[:,2],
               length=quiver_len, normalize=True, linewidth=0.6)

    for ax in (axL, axR):
        ax.set_box_aspect([np.ptp(P[:,0]), np.ptp(P[:,1]), np.ptp(P[:,2])])
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    if show_plots: plt.show()

    # === Histograms / summary ===
    fig2, axs = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    axs[0].hist(angles_deg, bins=50)
    axs[0].set_title("angle(N, N_ref) [deg]")
    axs[0].set_xlabel("deg"); axs[0].set_ylabel("count")

    axs[1].hist(tan_leak, bins=50)
    axs[1].set_title("tangent leakage")
    axs[1].set_xlabel("||N - (N·Nref)Nref||")

    Nf = N_len[np.isfinite(N_len)]
    if Nf.size == 0:
        Nf = np.array([0.0])
    rng = float(Nf.max() - Nf.min())
    if rng == 0.0:
        bins = "auto"; range_kw = None
    else:
        bins = 50
        pad = 0.02 * rng + 1e-8
        range_kw = (float(Nf.min() - pad), float(Nf.max() + pad))

    axs[2].hist(Nf, bins=bins, range=range_kw)
    axs[2].set_title(r"Histogram of $\|{\bf n}\|$")
    axs[2].set_xlabel("norm")
    if show_plots: plt.show()

    # Print quick stats
    _stat_line("angle(deg)", angles_deg, fmt=".3f")
    _stat_line("leakage", tan_leak, fmt=".3e")
    _stat_line("|N_raw|", N_len, fmt=".5f", p95=False)

    # Return possibly aligned normals for downstream use
    return N
