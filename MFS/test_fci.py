# test_fci_solver.py
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from solve_flux_psi_fci import (
    diffusion_tensor_jax,
    trilinear_weights,
    build_fci_connectivity_chunked,
    make_fci_operator_jax,
    FCIConnectivity,
    trace_to_delta_phi_batched,
    make_linear_operator_jax,
    make_fieldline_phi_rhs_jax,
)

# -------------------------------------------------------------------
# 1) diffusion_tensor_jax tests
# -------------------------------------------------------------------

def test_diffusion_tensor_isotropic_limit():
    """
    If gradphi is constant and aligned along x, D should be:
      D ≈ [[1,0,0],[0,eps,0],[0,0,eps]] (up to delta floor)
    """
    N = 5
    gradphi = np.zeros((N, 3))
    gradphi[:, 0] = 1.0  # gx=1, gy=gz=0

    eps = 1e-3
    delta = 0.0
    D = np.asarray(diffusion_tensor_jax(jnp.asarray(gradphi), eps=eps, delta=delta))

    # Check shape
    assert D.shape == (N, 3, 3)

    # Same matrix for all points
    D0 = D[0]
    # Parallel projector along x
    assert np.allclose(D0[0, 0], 1.0, atol=1e-12)
    assert np.allclose(D0[0, 1], 0.0, atol=1e-12)
    assert np.allclose(D0[0, 2], 0.0, atol=1e-12)

    # Perpendicular directions y,z ~ eps
    assert np.allclose(D0[1, 1], eps, atol=1e-12)
    assert np.allclose(D0[2, 2], eps, atol=1e-12)
    # Off-diagonals should be ~0
    offdiag = D0 - np.diag(np.diag(D0))
    assert np.allclose(offdiag, 0.0, atol=1e-12)


def test_diffusion_tensor_zero_gradphi_falls_back_to_I():
    """
    When |gradphi| is tiny, diffusion_tensor_jax should fall back to isotropic I.
    """
    N = 10
    gradphi = np.zeros((N, 3))  # all zeros

    eps = 1e-3
    delta = 0.0
    D = np.asarray(diffusion_tensor_jax(jnp.asarray(gradphi), eps=eps, delta=delta))

    I = np.eye(3)
    for n in range(N):
        assert np.allclose(D[n], I, atol=1e-12)


def test_diffusion_tensor_properties():
    rng = np.random.default_rng(42)
    gradphi = rng.standard_normal((10, 3))

    D = np.asarray(diffusion_tensor_jax(jnp.asarray(gradphi), eps=1e-3, delta=1e-2))

    # symmetric
    np.testing.assert_allclose(D, np.swapaxes(D, -1, -2), rtol=0, atol=1e-12)

    # SPD-ish: xᵀ D x >= 0
    x = rng.standard_normal((10, 3))
    Dx = np.einsum("nij,nj->ni", D, x)
    energy = np.einsum("ni,ni->n", x, Dx)
    assert np.all(energy >= -1e-12)

    # when gradφ ~ 0, result should be ~isotropic (I * (1+delta))
    gradphi_zero = np.zeros((5, 3))
    D_zero = np.asarray(diffusion_tensor_jax(jnp.asarray(gradphi_zero), eps=1e-3, delta=1e-2))
    I_scaled = (1.0 + 1e-2) * np.eye(3)
    for k in range(5):
        np.testing.assert_allclose(D_zero[k], I_scaled, rtol=1e-12, atol=1e-12)


# -------------------------------------------------------------------
# 2) trilinear_weights tests
# -------------------------------------------------------------------

def test_trilinear_weights_basic_properties():
    xs = np.linspace(0.0, 1.0, 5)
    ys = np.linspace(-1.0, 1.0, 5)
    zs = np.linspace(2.0, 3.0, 5)

    # Point strictly inside grid
    point = (0.3, 0.1, 2.4)
    idx, w = trilinear_weights(xs, ys, zs, point)

    # 8 corners
    assert idx.shape == (8,)
    assert w.shape == (8,)

    # weights sum to 1
    assert np.isclose(w.sum(), 1.0, atol=1e-12)

    # indices within range
    nx, ny, nz = len(xs), len(ys), len(zs)
    assert np.all((idx >= 0) & (idx < nx * ny * nz))

    # Linear reproduction test: if field is f(x,y,z) = x + 2y + 3z,
    # trilinear reconstruction at "point" should be exact on a rectilinear grid.
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)
    f_grid = (XX + 2 * YY + 3 * ZZ).ravel(order="C")

    f_interp = np.dot(w, f_grid[idx])
    f_true = point[0] + 2 * point[1] + 3 * point[2]
    assert np.isclose(f_interp, f_true, atol=1e-12)


# -------------------------------------------------------------------
# 3) Toy grad_phi fields
# -------------------------------------------------------------------

def grad_phi_pure_toroidal(X):
    """
    Toy grad_phi that produces a purely toroidal magnetic field in cylindrical coords.
    B = e_phi, so field lines keep (R,Z) constant and only change φ.
    This version is JAX-compatible.
    """
    X = jnp.asarray(X)
    x = X[..., 0]
    y = X[..., 1]
    phi = jnp.arctan2(y, x)

    Bx = -jnp.sin(phi)
    By =  jnp.cos(phi)
    Bz =  jnp.zeros_like(phi)
    return jnp.stack([Bx, By, Bz], axis=-1)


# -------------------------------------------------------------------
# 4) FCI connectivity tests (using chunked builder)
# -------------------------------------------------------------------

def test_build_fci_connectivity_pure_toroidal():
    """
    In a pure toroidal field (B = e_phi), field lines stay at constant (R,Z).
    FCI connectivity should be valid for many nodes that are 'inside', and
    the mapping endpoints lie at the same (R,Z).
    """
    xs = np.linspace(2.0, 3.0, 6)
    ys = np.linspace(0.5, 1.5, 6)
    zs = np.linspace(-0.5, 0.5, 6)

    nx, ny, nz = len(xs), len(ys), len(zs)
    N = nx * ny * nz

    inside_mask = np.ones(N, dtype=bool)
    nfp = 2

    fci = build_fci_connectivity_chunked(
        xs, ys, zs,
        inside_mask,
        grad_phi_fn=grad_phi_pure_toroidal,
        nfp=nfp,
        dphi_per_step=None,
        nsteps=16,
        verbose=False,
        chunk_size=None,
    )

    n_valid = int(fci.valid.sum())
    assert n_valid > 0.8 * inside_mask.sum()
    assert np.all(fci.L_plus[fci.valid] > 0.0)
    assert np.all(fci.L_minus[fci.valid] > 0.0)


def make_straight_Bz_connectivity(xs, ys, zs, inside_mask, L_par):
    nx, ny, nz = len(xs), len(ys), len(zs)
    N = nx * ny * nz
    idx_plus  = np.zeros((N, 8), int)
    idx_minus = np.zeros((N, 8), int)
    w_plus    = np.zeros((N, 8))
    w_minus   = np.zeros((N, 8))
    L_plus    = np.zeros(N)
    L_minus   = np.zeros(N)
    valid     = np.zeros(N, bool)

    inside3 = inside_mask.reshape(nx, ny, nz)
    for i in range(nx):
        for j in range(ny):
            for k in range(1, nz - 1):
                p = i * (ny * nz) + j * nz + k
                if not inside3[i, j, k]:
                    continue
                p_plus  = i * (ny * nz) + j * nz + (k + 1)
                p_minus = i * (ny * nz) + j * nz + (k - 1)
                if not (inside_mask[p_plus] and inside_mask[p_minus]):
                    continue
                # trivial "interpolation": weight 1 on the neighbor itself
                idx_plus[p, 0]  = p_plus
                w_plus[p, 0]    = 1.0
                idx_minus[p, 0] = p_minus
                w_minus[p, 0]   = 1.0
                L_plus[p]       = L_par
                L_minus[p]      = L_par
                valid[p]        = True

    return FCIConnectivity(
        idx_plus=idx_plus, w_plus=w_plus, L_plus=L_plus,
        idx_minus=idx_minus, w_minus=w_minus, L_minus=L_minus,
        valid=valid
    )


def _build_cartesian_coords(xs, ys, zs):
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)
    return XX, YY, ZZ


def test_fci_operator_linear_solution_straight_Bz():
    xs = np.linspace(0.0, 1.0, 8)
    ys = np.linspace(0.0, 1.0, 8)
    zs = np.linspace(0.0, 1.0, 8)
    nx, ny, nz = len(xs), len(ys), len(zs)
    N = nx * ny * nz

    inside_mask = np.ones(N, bool)
    L_par = zs[1] - zs[0]
    fci = make_straight_Bz_connectivity(xs, ys, zs, inside_mask, L_par)

    kappa_par  = 1.0
    kappa_perp = 1e-3
    A_pde_jax, deep_inside = make_fci_operator_jax(
        nx, ny, nz,
        xs, ys, zs,
        inside_mask,
        fci,
        kappa_par=kappa_par,
        kappa_perp=kappa_perp,
    )

    XX, YY, ZZ = _build_cartesian_coords(xs, ys, zs)
    psi = (1.2 * XX + 0.7 * YY - 0.3 * ZZ).ravel(order="C")

    Apsi = np.asarray(A_pde_jax(jnp.asarray(psi)))

    interior = np.zeros((nx, ny, nz), bool)
    interior[2:-2, 2:-2, 2:-2] = True
    interior_flat = interior.ravel(order="C")

    assert np.max(np.abs(Apsi[interior_flat])) < 1e-8


def test_fci_operator_linear_solution_pure_toroidal():
    """
    With B = e_phi and ψ = ax + by + cz, we expect:
        Δ_parallel ψ ≈ 0
        Δ_perp ψ = 0
    => A_pde[ψ] ≈ 0 in the interior.
    """
    xs = np.linspace(2.0, 3.0, 6)
    ys = np.linspace(0.5, 1.5, 6)
    zs = np.linspace(-0.5, 0.5, 6)

    nx, ny, nz = len(xs), len(ys), len(zs)
    N = nx * ny * nz
    inside_mask = np.ones(N, dtype=bool)

    nfp = 2
    fci = build_fci_connectivity_chunked(
        xs, ys, zs,
        inside_mask,
        grad_phi_fn=grad_phi_pure_toroidal,
        nfp=nfp,
        dphi_per_step=None,
        nsteps=16,
        verbose=False,
        chunk_size=None,
    )

    kappa_par = 1.0
    kappa_perp = 1e-3
    A_pde_jax, deep_inside = make_fci_operator_jax(
        nx, ny, nz,
        xs, ys, zs,
        inside_mask,
        fci,
        kappa_par=kappa_par,
        kappa_perp=kappa_perp,
    )

    XX, YY, ZZ = _build_cartesian_coords(xs, ys, zs)
    psi = (1.2 * XX + 0.7 * YY - 0.3 * ZZ).ravel(order="C")

    Apsi = np.asarray(A_pde_jax(jnp.asarray(psi)))

    interior = np.zeros((nx, ny, nz), dtype=bool)
    interior[2:-2, 2:-2, 2:-2] = True
    interior_flat = interior.ravel(order="C")

    # relaxed tolerance due to coarse grid + FCI mapping
    assert np.max(np.abs(Apsi[interior_flat])) < 5.4e-1


def test_build_fci_connectivity_respects_inside_mask():
    """
    Nodes flagged as outside in inside_mask should *never* be marked valid.
    """
    xs = np.linspace(2.0, 3.0, 6)
    ys = np.linspace(0.5, 1.5, 6)
    zs = np.linspace(-0.5, 0.5, 6)
    nx, ny, nz = len(xs), len(ys), len(zs)
    N = nx * ny * nz

    inside_mask = np.ones((nx, ny, nz), bool)
    inside_mask[2:4, 2:4, 2:4] = False
    inside_flat = inside_mask.ravel(order="C")

    nfp = 2
    fci = build_fci_connectivity_chunked(
        xs, ys, zs,
        inside_flat,
        grad_phi_fn=grad_phi_pure_toroidal,
        nfp=nfp,
        dphi_per_step=None,
        nsteps=16,
        verbose=False,
        chunk_size=None,
    )

    assert not np.any(fci.valid[~inside_flat])
    assert np.all(fci.L_plus[fci.valid] > 0.0)
    assert np.all(fci.L_minus[fci.valid] > 0.0)


def test_build_fci_connectivity_pure_toroidal_geometry_consistency():
    xs = np.linspace(2.0, 3.0, 6)
    ys = np.linspace(0.5, 1.5, 6)
    zs = np.linspace(-0.5, 0.5, 6)
    nx, ny, nz = len(xs), len(ys), len(zs)
    N = nx * ny * nz

    inside_mask = np.ones(N, bool)
    nfp = 2

    fci = build_fci_connectivity_chunked(
        xs, ys, zs,
        inside_mask,
        grad_phi_fn=grad_phi_pure_toroidal,
        nfp=nfp,
        dphi_per_step=None,
        nsteps=16,
        verbose=False,
        chunk_size=None,
    )

    valid_indices = np.where(fci.valid)[0]
    assert valid_indices.size > 0

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)
    Xflat = np.column_stack([XX.ravel(order="C"),
                             YY.ravel(order="C"),
                             ZZ.ravel(order="C")])

    for p in valid_indices[::max(1, len(valid_indices)//5)]:
        x0, y0, z0 = Xflat[p]
        R0 = np.hypot(x0, y0)
        Z0 = z0

        xpf = np.dot(fci.w_plus[p],  Xflat[fci.idx_plus[p], 0])
        ypf = np.dot(fci.w_plus[p],  Xflat[fci.idx_plus[p], 1])
        zpf = np.dot(fci.w_plus[p],  Xflat[fci.idx_plus[p], 2])
        Rf = np.hypot(xpf, ypf)
        Zf = zpf

        xmb = np.dot(fci.w_minus[p], Xflat[fci.idx_minus[p], 0])
        ymb = np.dot(fci.w_minus[p], Xflat[fci.idx_minus[p], 1])
        zmb = np.dot(fci.w_minus[p], Xflat[fci.idx_minus[p], 2])
        Rb = np.hypot(xmb, ymb)
        Zb = zmb

        assert np.isclose(Rf, R0, atol=1e-2)
        assert np.isclose(Zf, Z0, atol=1e-2)
        assert np.isclose(Rb, R0, atol=1e-2)
        assert np.isclose(Zb, Z0, atol=1e-2)


# -------------------------------------------------------------------
# 5) Tracing tests (only batched + local scalar ref)
# -------------------------------------------------------------------

def test_trace_to_delta_phi_pure_toroidal_single():
    """
    In the toy field grad_phi_pure_toroidal (B = e_phi),
    field lines should remain at constant (R,Z) while φ increases linearly.
    """
    R0   = 2.0
    Z0   = 0.3
    phi0 = 0.7
    dphi = 0.5
    nsteps = 32

    R1_arr, Z1_arr, phi1_arr, L_arr = trace_to_delta_phi_batched(
        grad_phi_pure_toroidal,
        jnp.asarray([R0]),
        jnp.asarray([Z0]),
        jnp.asarray([phi0]),
        dphi_target=jnp.asarray([dphi]),
        nsteps=nsteps,
    )

    R1   = float(R1_arr[0])
    Z1   = float(Z1_arr[0])
    phi1 = float(phi1_arr[0])
    L    = float(L_arr[0])

    assert np.isclose(R1, R0, atol=1e-8)
    assert np.isclose(Z1, Z0, atol=1e-8)

    dphi_eff = np.angle(np.exp(1j * (phi1 - phi0)))
    assert np.isclose(dphi_eff, dphi, atol=1e-6)
    assert L > 0.0


def test_trace_to_delta_phi_batched_matches_scalar():
    """
    Batched trace_to_delta_phi_batched should match N independent
    scalar integrations using the same RK2 scheme.
    """
    # simple grad_phi with constant B (tilted) so Bphi ≠ 0
    def grad_phi_fn(X):
        B = jnp.array([0.1, 1.0, 0.0])
        return jnp.broadcast_to(B, X.shape)

    rhs = make_fieldline_phi_rhs_jax(grad_phi_fn)

    @jax.jit
    def trace_single(R0, Z0, phi0, dphi_target, nsteps=16):
        dphi = dphi_target / nsteps
        R, Z, phi = R0, Z0, phi0
        for _ in range(nsteps):
            RZ = jnp.stack([R, Z])
            k1 = rhs(phi, RZ, None)
            R_pred = R + dphi * k1[0]
            Z_pred = Z + dphi * k1[1]
            phi_pred = phi + dphi

            k2 = rhs(phi_pred, jnp.stack([R_pred, Z_pred]), None)
            R = R + 0.5 * dphi * (k1[0] + k2[0])
            Z = Z + 0.5 * dphi * (k1[1] + k2[1])
            phi = phi + dphi
        return R, Z, phi

    N = 32
    R0 = jnp.linspace(1.0, 1.5, N)
    Z0 = jnp.zeros_like(R0)
    phi0 = jnp.zeros_like(R0)
    dphi_target = jnp.full_like(R0, 0.4)

    Rb, Zb, phib, Lb = trace_to_delta_phi_batched(
        grad_phi_fn, R0, Z0, phi0, dphi_target, nsteps=16
    )

    R_ref, Z_ref, phi_ref = jax.vmap(trace_single)(
        R0, Z0, phi0, dphi_target
    )

    assert jnp.max(jnp.abs(Rb - R_ref)) < 1e-5
    assert jnp.max(jnp.abs(Zb - Z_ref)) < 1e-5
    assert jnp.max(jnp.abs(phib - phi_ref)) < 1e-5


# -------------------------------------------------------------------
# 6) JAX anisotropic operator sanity check
# -------------------------------------------------------------------

def test_linear_operator_jax_linear_solution():
    """
    For constant D and a linear ψ, -div(D ∇ψ) should be ~0 in the interior.
    """
    nx, ny, nz = 4, 5, 3
    N = nx * ny * nz

    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-0.5, 0.5, ny)
    zs = np.linspace(-0.2, 0.2, nz)
    dx, dy, dz = xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]

    inside = np.ones(N, dtype=bool)

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    X = np.stack([XX, YY, ZZ], axis=-1).reshape(-1, 3)
    G = np.zeros_like(X)
    G[:, 2] = 1.0  # gradφ = (0,0,1)
    Dfield = np.asarray(diffusion_tensor_jax(jnp.asarray(G), eps=1e-2, delta=1e-3))

    A_jax, deep = make_linear_operator_jax(
        nx, ny, nz,
        dx, dy, dz,
        inside,
        Dfield,
    )

    XXc, YYc, ZZc = XX, YY, ZZ
    u = (1.3 * XXc + 0.7 * YYc - 0.5 * ZZc).ravel(order="C")

    Au = np.asarray(A_jax(jnp.asarray(u)))

    interior = np.zeros((nx, ny, nz), bool)
    interior[1:-1, 1:-1, 1:-1] = True
    interior_flat = interior.ravel(order="C")

    assert np.max(np.abs(Au[interior_flat])) < 1e-6
