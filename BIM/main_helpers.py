import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------- Paths and utils ------------------------- #
script_dir = Path(__file__).resolve().parent

def get_candidates(file_name, subdir="inputs", verbose=True):
    """
    Convenience helper: resolve ../inputs/<file_name>.csv and normals.
    """
    try:
        candidate_xyz = (script_dir / ".." / subdir / (file_name + ".csv")).resolve()
        candidate_normals = (script_dir / ".." / subdir / (file_name + "_normals.csv")).resolve()
        if candidate_xyz.exists():
            if verbose: print(f"Resolved xyz path -> {candidate_xyz}")
            candidate_xyz = str(candidate_xyz)
        else:
            if verbose: print(f"[WARN] Expected xyz at {candidate_xyz}; using literal path.")
        if candidate_normals.exists():
            if verbose: print(f"Resolved normals path -> {candidate_normals}")
            candidate_normals = str(candidate_normals)
        else:
            if verbose: print(f"[WARN] Expected normals at {candidate_normals}; using literal path.")
    except Exception as e:
        if verbose: print(f"[WARN] Failed to resolve ../{subdir} path: {e}")
    return candidate_xyz, candidate_normals


def pct(a, p):
    """Percentile helper using NumPy (for diagnostics only)."""
    return float(np.percentile(np.asarray(a), p))

def vec_stats(label, v):
    """
    Simple L2/Linf/mean stats for 1D arrays.

    Parameters
    ----------
    label : str
        Label printed before the statistics.
    v : array_like
        1D array (NumPy or JAX) whose norms are reported.
    """
    v_np = np.asarray(v)
    print(f"[STATS] {label}: "
          f"L2={np.linalg.norm(v_np):.3e}, "
          f"Linf={np.max(np.abs(v_np)):.3e}, "
          f"mean={np.mean(v_np):.3e}")

# ----------------------------------------------------------------------
# Diagnostics: inner-shell field, Laplacian of φ_s, boundary B
# ----------------------------------------------------------------------

def diagnostics_on_inner_shell(P, N, W, B_fn, verbose=True,
                               h_min=None, eps_factor=0.3, label="inner shell"):
    """
    Sample B slightly inside the boundary along normals, and report
    magnitude and normal component, plus flux through the shell.
    """
    X = jnp.asarray(P)
    Nw = jnp.asarray(N)
    Ww = jnp.asarray(W)

    if h_min is None:
        # crude spacing estimate if not provided
        h_min = jnp.min(jnp.sqrt(jnp.sum((X[1:] - X[:-1])**2, axis=1)))

    eps = eps_factor * h_min
    X_eval = X - eps * Nw

    B_on_shell = B_fn(X_eval)
    n_dot_B = jnp.sum(Nw * B_on_shell, axis=1)
    Bmag = jnp.linalg.norm(B_on_shell, axis=1)

    if verbose:
        vec_stats(f"B|{label} magnitude", Bmag)
        vec_stats(f"n·B|{label}", n_dot_B)

    flux = float(jnp.dot(Ww, n_dot_B))
    area = float(jnp.sum(Ww))
    if verbose: print(f"[CHK] Flux through {label}: Φ ≈ {flux:.6e}, avg n·B ≈ {flux/area:.3e}")

    return np.asarray(X_eval), np.asarray(B_on_shell), np.asarray(n_dot_B), np.asarray(Bmag)


def numerical_laplacian_phi_s(phi_s_fn, X_inner, h_fd):
    """
    Approximate Laplacian ∇²φ_s at inner-shell points via a 7-point stencil.
    """
    X = jnp.asarray(X_inner)
    h = h_fd

    ex = jnp.array([1.0, 0.0, 0.0])
    ey = jnp.array([0.0, 1.0, 0.0])
    ez = jnp.array([0.0, 0.0, 1.0])

    def phi_at_offset(X, direction):
        return phi_s_fn(X + h * direction[None, :]), phi_s_fn(X - h * direction[None, :])

    phi0 = phi_s_fn(X)

    phi_px_x, phi_mx_x = phi_at_offset(X, ex)
    phi_px_y, phi_mx_y = phi_at_offset(X, ey)
    phi_px_z, phi_mx_z = phi_at_offset(X, ez)

    lap = (phi_px_x + phi_mx_x - 2.0 * phi0
           + phi_px_y + phi_mx_y - 2.0 * phi0
           + phi_px_z + phi_mx_z - 2.0 * phi0) / (h * h)

    return np.asarray(lap)


# ----------------------------------------------------------------------
# Boundary coordinates relative to axis (ρ, φ)
# ----------------------------------------------------------------------

def compute_axis_coordinates(P, a_hat, E, center):
    """
    Compute distance to axis ρ and toroidal angle φ for each boundary point.

    Parameters
    ----------
    P : (N,3) array-like
        Boundary points in world coordinates.
    scinfo : ScaleInfo
        Contains center used for normalization.
    a_hat : (3,) array-like
        Unit vector along the PCA-based axis.
    E : (3,3) array-like
        PCA eigenvectors as columns. e1,e2 span the poloidal plane.

    Returns
    -------
    rho : (N,) ndarray
        Distances from the axis line.
    phi : (N,) ndarray
        Toroidal angles in [0, 2π), defined using projection onto (e1,e2).
    """
    Pj = jnp.asarray(P)
    a = a_hat / jnp.maximum(jnp.linalg.norm(a_hat), 1e-30)

    e1 = E[:, 0]
    e2 = E[:, 1]

    r_vec = Pj - center[None, :]
    r_par = jnp.sum(r_vec * a[None, :], axis=1, keepdims=True) * a[None, :]
    r_perp = r_vec - r_par

    rho = jnp.linalg.norm(r_perp, axis=1)

    x1 = jnp.sum(r_perp * e1[None, :], axis=1)
    x2 = jnp.sum(r_perp * e2[None, :], axis=1)
    phi = jnp.arctan2(x2, x1)
    # Map to [0, 2π)
    two_pi = 2.0 * jnp.pi
    phi = jnp.where(phi < 0.0, phi + two_pi, phi)

    return np.asarray(rho), np.asarray(phi)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def fix_matplotlib_3d(ax):
    """Equal aspect ratio in 3D plot."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1]-x_limits[0]); x_mid = 0.5*(x_limits[0]+x_limits[1])
    y_range = abs(y_limits[1]-y_limits[0]); y_mid = 0.5*(y_limits[0]+y_limits[1])
    z_range = abs(z_limits[1]-z_limits[0]); z_mid = 0.5*(z_limits[0]+z_limits[1])
    R = 0.5 * max(x_range, y_range, z_range)
    ax.set_xlim3d([x_mid-R, x_mid+R])
    ax.set_ylim3d([y_mid-R, y_mid+R])
    ax.set_zlim3d([z_mid-R, z_mid+R])

def make_3d_boundary_plots(P, Bmag, n_dot_B_norm, outfile):
    """
    3D diagnostic plots:
      - Surface colored by |B|
      - Surface colored by n·B/|B|
    """
    P_np = np.asarray(P)
    Bmag_np = np.asarray(Bmag)
    ndot_np = np.asarray(n_dot_B_norm)

    fig = plt.figure(figsize=(14, 6))

    # |B| on Γ
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    vmin = pct(Bmag_np, 1)
    vmax = pct(Bmag_np, 99)
    sc1 = ax1.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                      c=Bmag_np, s=6, cmap='viridis',
                      vmin=vmin, vmax=vmax)
    cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.7)
    cb1.set_label(r"$|B|$ on $\Gamma$")
    ax1.set_title("Boundary colored by $|B|$")
    fix_matplotlib_3d(ax1)

    # n·B/|B| on Γ
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    maxabs = np.max(np.abs(ndot_np))
    vmin2, vmax2 = -maxabs, maxabs
    sc2 = ax2.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2],
                      c=ndot_np, s=6, cmap='magma',
                      vmin=vmin2, vmax=vmax2)
    cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.7)
    cb2.set_label(r"$\mathbf{n}\!\cdot\!\mathbf{B}/|B|$ on $\Gamma$")
    ax2.set_title(r"Boundary colored by $\mathbf{n}\!\cdot\!\mathbf{B}/|B|$")
    fix_matplotlib_3d(ax2)

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    # plt.close(fig)


def make_1d_residual_plots(q_bdry, lap_inner, outfile):
    """
    1D residual plots:
      - q_bdry = n·B/|B| on Γ vs index
      - ∇²φ_s on inner shell vs index
    """
    q_np = np.asarray(q_bdry)
    lap_np = np.asarray(lap_inner)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(q_np, ".", ms=2)
    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[0].set_xlabel("Boundary point index")
    axes[0].set_ylabel(r"$\mathbf{n}\!\cdot\!\mathbf{B}/|B|$")
    axes[0].set_title(r"Boundary residual $\mathbf{n}\!\cdot\!\mathbf{B}/|B|$")

    axes[1].plot(lap_np, ".", ms=2)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("Inner-shell point index")
    axes[1].set_ylabel(r"$\nabla^2 \phi_s$ (FD)")
    axes[1].set_title(r"Laplacian residual $\nabla^2 \phi_s$ on inner shell")

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    # plt.close(fig)


def make_boundary_decomposition_vs_phi(phi, Bmag_tot, Bmag_mv, Bmag_s,
                                       n_dot_B, rho, outfile):
    """
    Plot |B|, |B_mv|, |B_s| and n·B vs toroidal angle φ.
    """
    phi_np = np.asarray(phi)
    Btot = np.asarray(Bmag_tot)
    Bmv = np.asarray(Bmag_mv)
    Bs  = np.asarray(Bmag_s)
    ndot = np.asarray(n_dot_B)
    rho_np = np.asarray(rho)

    # Sort by φ for clearer trends
    idx = np.argsort(phi_np)
    phi_s = phi_np[idx]
    Btot_s = Btot[idx]
    Bmv_s = Bmv[idx]
    Bs_s = Bs[idx]
    ndot_s = ndot[idx]
    rho_s = rho_np[idx]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(phi_s, Btot_s, ".", ms=2, label=r"$|B|$")
    axes[0].plot(phi_s, Bmv_s, ".", ms=2, label=r"$|B_{\mathrm{mv}}|$")
    axes[0].plot(phi_s, Bs_s, ".", ms=2, label=r"$|B_{\mathrm{MFS}}|$")
    axes[0].set_ylabel(r"$|B|$")
    axes[0].set_title(r"Boundary field decomposition vs toroidal angle $\varphi$")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(phi_s, ndot_s, ".", ms=2)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_ylabel(r"$\mathbf{n}\!\cdot\!\mathbf{B}$")

    axes[2].plot(phi_s, rho_s, ".", ms=2, color="tab:green")
    axes[2].set_xlabel(r"Toroidal angle $\varphi$ [rad]")
    axes[2].set_ylabel(r"$\rho$ (distance to axis)")

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    # plt.close(fig)


def make_boundary_decomposition_vs_rho(rho, Bmag_tot, Bmag_mv, Bmag_s,
                                       n_dot_B, outfile):
    """
    Plot |B|, |B_mv|, |B_s| and n·B vs distance to axis ρ.
    """
    rho_np = np.asarray(rho)
    Btot = np.asarray(Bmag_tot)
    Bmv = np.asarray(Bmag_mv)
    Bs  = np.asarray(Bmag_s)
    ndot = np.asarray(n_dot_B)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(rho_np, Btot, ".", ms=2, label=r"$|B|$")
    axes[0].plot(rho_np, Bmv, ".", ms=2, label=r"$|B_{\mathrm{mv}}|$")
    axes[0].plot(rho_np, Bs,  ".", ms=2, label=r"$|B_{\mathrm{MFS}}|$")
    axes[0].set_ylabel(r"$|B|$")
    axes[0].set_title(r"Boundary field decomposition vs distance to axis $\rho$")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(rho_np, ndot, ".", ms=2)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel(r"$\rho$ (distance to axis)")
    axes[1].set_ylabel(r"$\mathbf{n}\!\cdot\!\mathbf{B}$")

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    # plt.close(fig)


def make_boundary_geometry_plots(W, h, rho, outfile):
    """
    Plot geometric / quadrature diagnostics vs boundary index:
      - distance to axis ρ
      - area weights W
      - spacing h
    """
    W_np = np.asarray(W)
    h_np = np.asarray(h)
    rho_np = np.asarray(rho)
    idx = np.arange(len(W_np))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(idx, rho_np, ".", ms=2)
    axes[0].set_ylabel(r"$\rho$")

    axes[1].plot(idx, W_np, ".", ms=2)
    axes[1].set_ylabel(r"$W$ (area weight)")

    axes[2].plot(idx, h_np, ".", ms=2)
    axes[2].set_ylabel(r"$h$ (spacing)")
    axes[2].set_xlabel("Boundary point index")

    axes[0].set_title("Boundary geometry / quadrature diagnostics")

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    # plt.close(fig)
