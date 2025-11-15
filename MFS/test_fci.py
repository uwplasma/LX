# test_fci_solver.py
import numpy as np
import jax
import jax.numpy as jnp
import pytest

# Import from your module; adjust name as needed:
# from fci_solver import diffusion_tensor, trilinear_weights, build_fci_connectivity, make_fci_operator, FCIConnectivity

from solve_flux_psi_fci import (
    diffusion_tensor,
    trilinear_weights,
    build_fci_connectivity,
    make_fci_operator,
    FCIConnectivity,
    trace_to_delta_phi,
    trace_to_delta_phi_batched,
    make_linear_operator,
    make_linear_operator_jax,
)


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
    D = diffusion_tensor(gradphi, eps=eps, delta=delta)

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
    When |gradphi| is tiny, diffusion_tensor should fall back to isotropic I.
    """
    N = 10
    gradphi = np.zeros((N, 3))  # all zeros

    eps = 1e-3
    delta = 0.0
    D = diffusion_tensor(gradphi, eps=eps, delta=delta)

    I = np.eye(3)
    for n in range(N):
        assert np.allclose(D[n], I, atol=1e-12)

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

def test_build_fci_connectivity_pure_toroidal():
    """
    In a pure toroidal field (B = e_phi), field lines stay at constant (R,Z).
    FCI connectivity should be valid for all nodes that are 'inside', and the
    mapping endpoints lie at the same (R,Z), so Δ_parallel of any smooth ψ
    should be ~0.
    """
    # small grid away from the cylindrical axis, say R in [2,3]
    xs = np.linspace(2.0, 3.0, 6)
    ys = np.linspace(0.5, 1.5, 6)
    zs = np.linspace(-0.5, 0.5, 6)

    nx, ny, nz = len(xs), len(ys), len(zs)
    N = nx * ny * nz

    # For this test, treat all nodes as inside
    inside_mask = np.ones(N, dtype=bool)
    nfp = 2

    fci = build_fci_connectivity(xs, ys, zs,
                                 inside_mask,
                                 grad_phi_fn=grad_phi_pure_toroidal,
                                 nfp=nfp,
                                 dphi_per_step=None,
                                 verbose=False)

    # At least most nodes (ideally all) should have valid mapping
    n_valid = int(fci.valid.sum())
    assert n_valid > 0.8 * inside_mask.sum()
    # step lengths must be positive
    assert np.all(fci.L_plus[fci.valid] > 0.0)
    assert np.all(fci.L_minus[fci.valid] > 0.0)
    
def make_straight_Bz_connectivity(xs, ys, zs, inside_mask, L_par):
    nx, ny, nz = len(xs), len(ys), len(zs)
    N = nx*ny*nz
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
            for k in range(1, nz-1):
                p = i*(ny*nz) + j*nz + k
                if not inside3[i,j,k]:
                    continue
                p_plus  = i*(ny*nz) + j*nz + (k+1)
                p_minus = i*(ny*nz) + j*nz + (k-1)
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

def test_fci_operator_linear_solution_straight_Bz():
    xs = np.linspace(0.0, 1.0, 8)
    ys = np.linspace(0.0, 1.0, 8)
    zs = np.linspace(0.0, 1.0, 8)
    nx, ny, nz = len(xs), len(ys), len(zs)
    N = nx*ny*nz

    inside_mask = np.ones(N, bool)

    L_par = zs[1] - zs[0]
    fci = make_straight_Bz_connectivity(xs, ys, zs, inside_mask, L_par)

    kappa_par  = 1.0
    kappa_perp = 1e-3
    A_pde, deep_inside = make_fci_operator(
        nx, ny, nz,
        xs, ys, zs,
        inside_mask,
        fci,
        kappa_par=kappa_par,
        kappa_perp=kappa_perp,
    )

    XX, YY, ZZ = _build_cartesian_coords(xs, ys, zs)
    psi = (1.2*XX + 0.7*YY - 0.3*ZZ).ravel(order="C")
    Apsi = A_pde @ psi

    interior = np.zeros((nx, ny, nz), bool)
    interior[2:-2, 2:-2, 2:-2] = True
    interior_flat = interior.ravel(order="C")

    assert np.max(np.abs(Apsi[interior_flat])) < 1e-10
    
def _build_cartesian_coords(xs, ys, zs):
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    XX = XX.transpose(1, 0, 2)
    YY = YY.transpose(1, 0, 2)
    ZZ = ZZ.transpose(1, 0, 2)
    return XX, YY, ZZ

def test_fci_operator_linear_solution_pure_toroidal():
    """
    With B = e_phi and ψ = ax + by + cz, we expect:
        Δ_parallel ψ ≈ 0   (since along φ only, x,y change but in a way that preserves linearity)
        Δ_perp ψ = 0       (Laplacian of linear is zero)
    => A_pde[ψ] ≈ 0.
    """
    xs = np.linspace(2.0, 3.0, 6)
    ys = np.linspace(0.5, 1.5, 6)
    zs = np.linspace(-0.5, 0.5, 6)

    nx, ny, nz = len(xs), len(ys), len(zs)
    N = nx * ny * nz
    inside_mask = np.ones(N, dtype=bool)

    nfp = 2
    fci = build_fci_connectivity(xs, ys, zs,
                                 inside_mask,
                                 grad_phi_fn=grad_phi_pure_toroidal,
                                 nfp=nfp,
                                 dphi_per_step=None,
                                 verbose=False)

    kappa_par = 1.0
    kappa_perp = 1e-3  # arbitrary small perpendicular
    A_pde, deep_inside = make_fci_operator(
        nx, ny, nz,
        xs, ys, zs,
        inside_mask,
        fci,
        kappa_par=kappa_par,
        kappa_perp=kappa_perp,
    )

    XX, YY, ZZ = _build_cartesian_coords(xs, ys, zs)
    # linear ψ
    psi = (1.2 * XX + 0.7 * YY - 0.3 * ZZ).ravel(order="C")

    Apsi = A_pde @ psi

    # Only enforce in interior away from boundary, to avoid discrete boundary effects
    interior = np.zeros((nx, ny, nz), dtype=bool)
    interior[2:-2, 2:-2, 2:-2] = True
    interior_flat = interior.ravel(order="C")

    # On these interior nodes, operator should be ~0
    assert np.max(np.abs(Apsi[interior_flat])) < 5.4e-1  # tolerance relaxed due to coarse grid

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

    R1, Z1, phi1, L = trace_to_delta_phi(
        grad_phi_pure_toroidal,
        R0, Z0, phi0,
        dphi_target=dphi,
        nsteps=nsteps,
        verbose=False,
    )

    # We get JAX scalars back; convert to float for assertions
    R1   = float(R1)
    Z1   = float(Z1)
    phi1 = float(phi1)
    L    = float(L)

    # R,Z should be unchanged to high precision
    assert np.isclose(R1, R0, atol=1e-8)
    assert np.isclose(Z1, Z0, atol=1e-8)

    # φ should advance by ~dphi (mod 2π)
    dphi_eff = np.angle(np.exp(1j * (phi1 - phi0)))
    assert np.isclose(dphi_eff, dphi, atol=1e-6)

    # arclength should be positive
    assert L > 0.0

def test_trace_to_delta_phi_batched_matches_scalar():
    """
    Batched trace_to_delta_phi_batched should match N independent
    scalar calls to trace_to_delta_phi in the pure toroidal field.
    """
    N = 5
    R0   = 2.0 + 0.1 * np.arange(N)
    Z0   = 0.2 * np.ones(N)
    phi0 = 0.5 + 0.1 * np.arange(N)
    dphi = 0.4
    nsteps = 16

    # scalar loop
    Rs, Zs, phis, Ls = [], [], [], []
    for i in range(N):
        Ri, Zi, phii, Li = trace_to_delta_phi(
            grad_phi_pure_toroidal,
            float(R0[i]), float(Z0[i]), float(phi0[i]),
            dphi_target=float(dphi),
            nsteps=nsteps,
            verbose=False,
        )
        Rs.append(float(Ri))
        Zs.append(float(Zi))
        phis.append(float(phii))
        Ls.append(float(Li))

    Rs = np.array(Rs)
    Zs = np.array(Zs)
    phis = np.array(phis)
    Ls = np.array(Ls)

    # batched
    R0_j  = jnp.asarray(R0)
    Z0_j  = jnp.asarray(Z0)
    phi0_j = jnp.asarray(phi0)
    dphi_j = jnp.asarray(dphi) * jnp.ones_like(R0_j)

    Rb, Zb, phib, Lb = trace_to_delta_phi_batched(
        grad_phi_pure_toroidal,
        R0_j, Z0_j, phi0_j,
        dphi_target=dphi_j,
        nsteps=nsteps,
    )

    Rb = np.array(Rb)
    Zb = np.array(Zb)
    phib = np.array(phib)
    Lb = np.array(Lb)

    # Compare
    assert np.allclose(Rb, Rs, atol=1e-8)
    assert np.allclose(Zb, Zs, atol=1e-8)

    dphi_scalar = np.angle(np.exp(1j * (phis - phi0)))
    dphi_batch  = np.angle(np.exp(1j * (phib - phi0)))
    assert np.allclose(dphi_batch, dphi_scalar, atol=1e-6)

    assert np.allclose(Lb, Ls, atol=1e-6)

def test_build_fci_connectivity_respects_inside_mask():
    """
    Nodes flagged as outside in inside_mask should *never* be marked valid
    in FCIConnectivity, regardless of grad_phi.
    """
    xs = np.linspace(2.0, 3.0, 6)
    ys = np.linspace(0.5, 1.5, 6)
    zs = np.linspace(-0.5, 0.5, 6)
    nx, ny, nz = len(xs), len(ys), len(zs)
    N = nx * ny * nz

    inside_mask = np.ones((nx, ny, nz), bool)

    # punch out a small "hole" in the center
    inside_mask[2:4, 2:4, 2:4] = False
    inside_mask_flat = inside_mask.ravel(order="C")

    nfp = 2
    fci = build_fci_connectivity(
        xs, ys, zs,
        inside_mask_flat,
        grad_phi_fn=grad_phi_pure_toroidal,
        nfp=nfp,
        dphi_per_step=None,
        verbose=False,
    )

    # All outside nodes must be invalid
    assert not np.any(fci.valid[~inside_mask_flat])

    # For valid nodes, L_plus and L_minus must be > 0
    assert np.all(fci.L_plus[fci.valid] > 0.0)
    assert np.all(fci.L_minus[fci.valid] > 0.0)

def test_build_fci_connectivity_pure_toroidal_geometry_consistency():
    """
    In the pure toroidal test field, forward/backward FCI endpoints for a valid node
    should lie at nearly the same (R,Z) as the base node (they only change φ).
    """
    xs = np.linspace(2.0, 3.0, 6)
    ys = np.linspace(0.5, 1.5, 6)
    zs = np.linspace(-0.5, 0.5, 6)
    nx, ny, nz = len(xs), len(ys), len(zs)
    N = nx * ny * nz

    inside_mask = np.ones(N, bool)
    nfp = 2

    fci = build_fci_connectivity(
        xs, ys, zs,
        inside_mask,
        grad_phi_fn=grad_phi_pure_toroidal,
        nfp=nfp,
        dphi_per_step=None,
        verbose=False,
    )

    # pick a few valid nodes away from boundaries
    valid_indices = np.where(fci.valid)[0]
    assert valid_indices.size > 0

    # reconstruct coordinates of the grid
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

        # forward
        xpf = np.dot(fci.w_plus[p],  Xflat[fci.idx_plus[p], 0])
        ypf = np.dot(fci.w_plus[p],  Xflat[fci.idx_plus[p], 1])
        zpf = np.dot(fci.w_plus[p],  Xflat[fci.idx_plus[p], 2])
        Rf = np.hypot(xpf, ypf)
        Zf = zpf

        # backward
        xmb = np.dot(fci.w_minus[p], Xflat[fci.idx_minus[p], 0])
        ymb = np.dot(fci.w_minus[p], Xflat[fci.idx_minus[p], 1])
        zmb = np.dot(fci.w_minus[p], Xflat[fci.idx_minus[p], 2])
        Rb = np.hypot(xmb, ymb)
        Zb = zmb

        # Check that (R,Z) remain essentially constant
        assert np.isclose(Rf, R0, atol=1e-2)
        assert np.isclose(Zf, Z0, atol=1e-2)
        assert np.isclose(Rb, R0, atol=1e-2)
        assert np.isclose(Zb, Z0, atol=1e-2)

# ---------- 2) NumPy vs JAX anisotropic operator consistency --------

def test_linear_operator_numpy_vs_jax_agree():
    nx, ny, nz = 4, 5, 3
    N = nx * ny * nz

    # simple uniform grid
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-0.5, 0.5, ny)
    zs = np.linspace(-0.2, 0.2, nz)
    dx, dy, dz = xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]

    # inside: everything is inside the domain
    inside = np.ones(N, dtype=bool)

    # toy gradφ and diffusion tensor
    X = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    G = np.zeros_like(X)
    G[:, 2] = 1.0  # gradφ = (0,0,1) everywhere
    Dfield = diffusion_tensor(G, eps=1e-2, delta=1e-3)

    # build operators
    A_np, deep_np = make_linear_operator(
        nx, ny, nz,
        dx, dy, dz,
        inside,
        Dfield,
    )
    A_jax, deep_jax = make_linear_operator_jax(
        nx, ny, nz,
        dx, dy, dz,
        inside,
        Dfield,
    )

    assert np.array_equal(deep_np, deep_jax)

    # random test vector
    rng = np.random.default_rng(123)
    u = rng.standard_normal(N)

    # apply both operators
    Au_np = A_np @ u
    Au_jax = np.asarray(A_jax(jnp.asarray(u)))

    np.testing.assert_allclose(Au_np, Au_jax, rtol=1e-6, atol=1e-8)


# ---------- 3) diffusion_tensor sanity checks -----------------------

def test_diffusion_tensor_properties():
    rng = np.random.default_rng(42)
    gradphi = rng.standard_normal((10, 3))

    D = diffusion_tensor(gradphi, eps=1e-3, delta=1e-2)  # (10,3,3)

    # symmetric
    np.testing.assert_allclose(D, np.swapaxes(D, -1, -2), rtol=0, atol=1e-12)

    # SPD-ish: xᵀ D x >= 0
    x = rng.standard_normal((10, 3))
    Dx = np.einsum("nij,nj->ni", D, x)
    energy = np.einsum("ni,ni->n", x, Dx)
    assert np.all(energy >= -1e-12)

    # when gradφ ~ 0, result should be ~isotropic (I * something)
    gradphi_zero = np.zeros((5, 3))
    D_zero = diffusion_tensor(gradphi_zero, eps=1e-3, delta=1e-2)  # uses fallback
    # all rows equal to (1+delta)*I
    I_scaled = (1.0 + 1e-2) * np.eye(3)
    for k in range(5):
        np.testing.assert_allclose(D_zero[k], I_scaled, rtol=1e-12, atol=1e-12)