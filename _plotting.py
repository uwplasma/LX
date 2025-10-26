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
