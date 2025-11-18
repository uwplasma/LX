# test_bie_vacuum_energy_min_vmec_like.py
#
# Battery of unit + regression tests for
#   bim.py
#
# Run with:
#   pytest -q test_bie_vacuum_energy_min_vmec_like.py
#
# The tests use small synthetic geometries (sphere, circular torus)
# and stress:
#   - multivalued bases
#   - geometry normalization and normals
#   - quadrature weights and K' symmetry
#   - Neumann solve (½I+K')σ = g
#   - single-layer evaluators (φ_s, ∇φ_s)
#   - boundary regularization / jump relations
#   - energy + flux matrices
#   - full “analytic” torus comparison against the known field
#       B_exact = ∇ϕ_a = (-y/(x^2+y^2), x/(x^2+y^2), 0)
#
# Many tests print diagnostic stats; run with -s to see them.

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import bim_original as bie


# ---------------------------------------------------------------------
# Helpers: synthetic geometries
# ---------------------------------------------------------------------


def make_unit_sphere(n_theta=18, n_phi=36, R=1.0):
    """
    Regular-ish grid on a sphere of radius R.
    Returns: P (N,3), N (N,3) (outward normals).
    """
    thetas = np.linspace(1e-3, np.pi - 1e-3, n_theta)
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    pts = []
    nrm = []
    for th in thetas:
        for ph in phis:
            x = R * np.sin(th) * np.cos(ph)
            y = R * np.sin(th) * np.sin(ph)
            z = R * np.cos(th)
            pts.append((x, y, z))
            # Outward radial
            nrm.append((x / R, y / R, z / R))
    P = np.array(pts, dtype=float)
    N = np.array(nrm, dtype=float)
    return jnp.asarray(P), jnp.asarray(N)


def make_circular_torus(R0=1.5, a=0.4, n_theta=22, n_phi=40):
    """
    Circular torus around the z-axis:
        x = (R0 + a cos θ) cos φ
        y = (R0 + a cos θ) sin φ
        z = a sin θ
    Outward normal is the unit vector from tube centerline.
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    pts = []
    nrm = []
    for th in thetas:
        for ph in phis:
            x = (R0 + a * np.cos(th)) * np.cos(ph)
            y = (R0 + a * np.cos(th)) * np.sin(ph)
            z = a * np.sin(th)

            # outward from centerline
            nx = np.cos(th) * np.cos(ph)
            ny = np.cos(th) * np.sin(ph)
            nz = np.sin(th)
            pts.append((x, y, z))
            nrm.append((nx, ny, nz))

    P = np.array(pts, dtype=float)
    N = np.array(nrm, dtype=float)
    return jnp.asarray(P), jnp.asarray(N)


def analytic_axis_azimuth_grad(X):
    """
    Analytic B = ∇ϕ_a for axis along z:
        ϕ_a = atan2(y, x),
        ∇ϕ_a = (-y/(x^2 + y^2), x/(x^2 + y^2), 0).
    X: (N,3) ndarray.
    """
    X = np.asarray(X)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    r2 = x * x + y * y
    # torus never includes the axis, but just in case:
    r2 = np.maximum(r2, 1e-12)
    Bx = -y / r2
    By = x / r2
    Bz = np.zeros_like(Bx)
    return np.stack((Bx, By, Bz), axis=1)


# ---------------------------------------------------------------------
# Unit tests for low-level utilities
# ---------------------------------------------------------------------


def test_grad_azimuth_orthogonal_to_axis_and_scaling():
    rng = np.random.default_rng(0)
    a_hat = jnp.array([0.1, 0.3, 0.95])
    a_hat = a_hat / jnp.linalg.norm(a_hat)

    # Random points away from the axis
    X = rng.normal(size=(50, 3))
    X += np.array([2.0, 0.5, -0.2])

    Xj = jnp.asarray(X)
    G = bie.grad_azimuth_about_axis(Xj, a_hat)

    # a_hat · ∇ϕ_a ≈ 0
    dot = jnp.sum(G * a_hat[None, :], axis=1)
    assert float(jnp.max(jnp.abs(dot))) < 1e-10

    # scaling ~ 1/r_perp
    a = a_hat
    r_par = jnp.sum(Xj * a[None, :], axis=1, keepdims=True) * a[None, :]
    r_perp = Xj - r_par
    r = jnp.linalg.norm(r_perp, axis=1)
    inv_r = 1.0 / r
    mag = jnp.linalg.norm(G, axis=1)
    corr = np.corrcoef(np.asarray(inv_r), np.asarray(mag))[0, 1]
    print("[TEST] grad_azimuth: corr(1/r, |∇ϕ|) =", corr)
    assert corr > 0.99


def test_multivalued_bases_grad_t_constant():
    P, N = make_unit_sphere()
    Pn, scinfo = bie.normalize_geometry(P, verbose=False)
    a_hat = jnp.array([0.0, 0.0, 1.0])
    grad_t, grad_p = bie.multivalued_bases_about_axis(Pn, N, a_hat, verbose=False)

    Gt = grad_t(Pn)
    # All rows should be ~ identical and ~ a_hat/|a_hat|
    Gt0 = np.asarray(Gt[0])
    diffs = np.asarray(Gt) - Gt0[None, :]
    maxdiff = np.max(np.abs(diffs))
    print("[TEST] grad_t max row-to-row diff =", maxdiff)
    assert maxdiff < 1e-12
    assert np.allclose(np.linalg.norm(Gt0), 1.0, atol=1e-12)


def test_maybe_flip_normals_on_sphere():
    P, N_out = make_unit_sphere()
    N_in = -N_out
    N_fixed = bie.maybe_flip_normals(P, N_in)
    # We expect them to be flipped back to outward
    diff = np.asarray(N_fixed - N_out)  # <-- was + N_out
    maxdiff = np.max(np.abs(diff))
    print("[TEST] maybe_flip_normals max diff to outward =", maxdiff)
    assert maxdiff < 1e-12


def test_area_weights_positive_and_reasonable():
    P, N = make_unit_sphere(R=1.0)
    Pn, scinfo = bie.normalize_geometry(P, verbose=False)
    Wn, h_n = bie.estimate_area_weights_knn(Pn, k=16)

    scale = float(scinfo.scale)
    W = Wn / (scale**2)  # world area
    assert np.all(np.asarray(W) > 0.0)

    total_area_est = float(jnp.sum(W))
    true_area = 4.0 * np.pi * 1.0**2
    rel_err = abs(total_area_est - true_area) / true_area
    print("[TEST] sphere area: est =", total_area_est, " true =", true_area,
          " rel_err =", rel_err)
    assert rel_err < 0.25  # rough quadrature is ok


def test_Kprime_weighted_symmetry():
    P, N = make_unit_sphere(R=1.0)
    Pn, scinfo = bie.normalize_geometry(P, verbose=False)
    Wn, h_n = bie.estimate_area_weights_knn(Pn, k=16)
    scale = float(scinfo.scale)
    W = Wn / (scale**2)
    h = h_n / scale

    Kp = bie.build_Kprime(P, N, W, h)
    # Expect approximate symmetry in the operator sense: W_i K_ij ≈ W_j K_ji
    Wi = W[:, None]
    Wj = W[None, :]
    sym_diff = Wi * Kp - Wj * Kp.T
    num = float(jnp.max(jnp.abs(sym_diff)))
    den = float(jnp.max(jnp.abs(Wi * Kp)) + 1e-14)
    rel = num / den
    print("[TEST] weighted symmetry rel error =", rel)
    assert rel < 5e-2


def test_solve_density_zero_rhs_gives_zero_sigma():
    P, N = make_unit_sphere(R=1.0)
    Pn, scinfo = bie.normalize_geometry(P, verbose=False)
    Wn, h_n = bie.estimate_area_weights_knn(Pn, k=16)
    scale = float(scinfo.scale)
    W = Wn / (scale**2)
    h = h_n / scale

    g = jnp.zeros(P.shape[0], dtype=jnp.float64)
    sigma = bie.solve_density_sigma(P, N, W, h, g, reg=1e-8)
    norm_sigma = float(jnp.linalg.norm(sigma))
    print("[TEST] ||sigma||_2 for g=0 =", norm_sigma)
    assert norm_sigma < 1e-10


def test_compute_Bs_tan_constant_sigma_nearly_zero():
    P, N = make_unit_sphere(R=1.0)
    Pn, scinfo = bie.normalize_geometry(P, verbose=False)
    Wn, h_n = bie.estimate_area_weights_knn(Pn, k=16)
    scale = float(scinfo.scale)
    W = Wn / (scale**2)
    h = h_n / scale

    sigma_const = jnp.ones(P.shape[0], dtype=jnp.float64)
    Bs_tan = bie.compute_Bs_tan_on_boundary(P, N, W, sigma_const, h)
    norms = jnp.linalg.norm(Bs_tan, axis=1)
    max_norm = float(jnp.max(norms))
    print("[TEST] |B_s^tan| max for constant σ =", max_norm)
    assert max_norm < 5e-3


# ---------------------------------------------------------------------
# Tests for single-layer evaluators and Laplacian
# ---------------------------------------------------------------------


def test_single_layer_sphere_far_field_matches_point_charge():
    R = 1.0
    P, N = make_unit_sphere(R=R)
    Npts = P.shape[0]
    # Equal weights = exact area / N
    area = 4.0 * np.pi * R**2
    W = jnp.ones(Npts) * (area / Npts)
    sigma = jnp.ones(Npts)
    h_min = float((area / Npts) ** 0.5 / np.pi**0.5)

    phi_fn, grad_fn = bie.make_single_layer_evaluators(P, W, sigma, h_min)

    # Analytic: constant density σ=1 on sphere of radius R:
    # total "charge" Q = ∫σ dS = 4πR^2.
    # φ(r) = Q / (4π r) = R^2 / r
    # ∇φ = -Q * x / (4π r^3) = -R^2 x / r^3
    X_eval = np.array([[0.0, 0.0, 2.0],
                       [2.0, 0.0, 0.0],
                       [0.0, 2.0, 0.0]], dtype=float)
    r = np.linalg.norm(X_eval, axis=1)
    phi_exact = R**2 / r
    grad_exact = -R**2 * X_eval / (r[:, None] ** 3)

    phi_num = np.asarray(phi_fn(jnp.asarray(X_eval)))
    grad_num = np.asarray(grad_fn(jnp.asarray(X_eval)))

    rel_phi = np.linalg.norm(phi_num - phi_exact) / np.linalg.norm(phi_exact)
    rel_grad = np.linalg.norm(grad_num - grad_exact) / np.linalg.norm(grad_exact)
    print("[TEST] single-layer sphere rel_phi =", rel_phi,
          " rel_grad =", rel_grad)
    assert rel_phi < 0.1
    assert rel_grad < 0.25


def test_numerical_laplacian_phi_s_is_small_inside():
    R = 1.0
    P, N = make_unit_sphere(R=R)
    Npts = P.shape[0]
    area = 4.0 * np.pi * R**2
    W = jnp.ones(Npts) * (area / Npts)
    sigma = jnp.ones(Npts)
    h_min = float((area / Npts) ** 0.5 / np.pi**0.5)
    phi_fn, grad_fn = bie.make_single_layer_evaluators(P, W, sigma, h_min)

    # Points strictly inside the sphere
    X_inner = 0.5 * np.asarray(P)
    h_fd = 0.1 * R
    lap = bie.numerical_laplacian_phi_s(phi_fn, jnp.asarray(X_inner), h_fd=h_fd)
    lap = np.asarray(lap)
    rel = np.linalg.norm(lap) / (np.sqrt(lap.size) + 1e-14)
    print("[TEST] Laplacian φ_s inside sphere RMS ~", rel)
    assert rel < 5e-2


# ---------------------------------------------------------------------
# Energy / flux functionals
# ---------------------------------------------------------------------


def test_energy_matrix_positive_definite_on_torus():
    P, N = make_circular_torus()
    Pn, scinfo = bie.normalize_geometry(P, verbose=False)

    # simple area weights
    Wn, h_n = bie.estimate_area_weights_knn(Pn, k=16)
    scale = float(scinfo.scale)
    W = Wn / (scale**2)
    h = h_n / scale

    kind, a_hat, E, svals = bie.detect_geometry_and_axis(Pn, verbose=False)
    grad_t, grad_p = bie.multivalued_bases_about_axis(Pn, N, a_hat, verbose=False)

    (Dt, Dp), (sigma1, sigma2), (B_t_bdry, B_p_bdry) = bie.solve_basis_fields(
        P, N, W, h, Pn, scinfo, grad_t, grad_p, reg=1e-8
    )

    (phi1_fn, B1_fn, B1_mv_fn), (phi2_fn, B2_fn, B2_mv_fn) = bie.build_basis_evaluators(
        P, W, float(jnp.min(h)), scinfo, grad_t, grad_p, sigma1, sigma2
    )

    M, c, X_cs, B1_cs, B2_cs = bie.build_energy_flux_matrices_on_cross_section(
        P, a_hat, E, scinfo, B1_fn, B2_fn,
        n_r=16, n_theta=32, margin_frac=0.15
    )

    M_np = np.asarray(M)
    eig = np.linalg.eigvalsh(M_np)
    print("[TEST] energy matrix eigenvalues:", eig)
    assert np.all(eig > 0.0)


def test_energy_minimization_satisfies_flux_constraint():
    # Simple random positive-definite 2x2 M and random c
    rng = np.random.default_rng(1)
    A = rng.normal(size=(2, 2))
    M = jnp.asarray(A.T @ A + np.eye(2))  # SPD
    c = jnp.asarray(rng.normal(size=2))
    phiedge = 1.234

    a_star = bie.solve_energy_minimizing_coeffs(M, c, phiedge)
    # Check constraint
    flux = float(c @ a_star)
    rel = abs(flux - phiedge) / abs(phiedge)
    print("[TEST] flux constraint rel error =", rel)
    assert rel < 1e-12

    # Check local minimality along constraint-preserving directions
    def energy(a):
        return 0.5 * float(a @ (M @ a))

    c_np = np.asarray(c)
    a_np = np.asarray(a_star)
    for k in range(5):
        delta = rng.normal(size=2)
        # Project delta to be orthogonal to c => preserves flux to 1st order
        delta -= c_np * (delta @ c_np) / (c_np @ c_np)
        for eps in [1e-3, -1e-3]:
            E0 = energy(a_np)
            E1 = energy(a_np + eps * delta)
            print(f"[TEST] energy variation eps={eps:+.1e}: E1-E0=", E1 - E0)
            assert E1 >= E0 - 1e-10


# ---------------------------------------------------------------------
# Boundary-aware B and BC enforcement
# ---------------------------------------------------------------------


def test_build_B_on_boundary_enforces_n_dot_B_small():
    P, N = make_unit_sphere(R=1.0)
    Pn, scinfo = bie.normalize_geometry(P, verbose=False)
    Wn, h_n = bie.estimate_area_weights_knn(Pn, k=16)
    scale = float(scinfo.scale)
    W = Wn / (scale**2)
    h = h_n / scale
    a_hat = jnp.array([0.0, 0.0, 1.0])
    grad_t, grad_p = bie.multivalued_bases_about_axis(Pn, N, a_hat, verbose=False)

    # Use a_t=1, a_p=0 basis
    Gt = grad_t(Pn)
    B_t = scale * Gt
    g = jnp.sum(N * B_t, axis=1)
    sigma = bie.solve_density_sigma(P, N, W, h, g, reg=1e-8)

    a = jnp.array([1.0, 0.0])
    B_bdry = bie.build_B_on_boundary_with_jump(
        P, N, W, h, scinfo, a, grad_t, grad_p, sigma, g, clip_factor=0.2
    )
    Bmag = jnp.linalg.norm(B_bdry, axis=1)
    n_hat = N / jnp.maximum(jnp.linalg.norm(N, axis=1, keepdims=True), 1e-30)
    n_dot_B = jnp.sum(n_hat * B_bdry, axis=1)
    q = n_dot_B / jnp.maximum(Bmag, 1e-12)

    rms = float(jnp.linalg.norm(q) / jnp.sqrt(q.size))
    linf = float(jnp.max(jnp.abs(q)))
    print("[TEST] n·B/|B| on Γ: RMS =", rms, " Linf =", linf)
    assert rms < 5e-3
    assert linf < 5e-2


# ---------------------------------------------------------------------
# Analytic “circular torus” regression test
# ---------------------------------------------------------------------

def test_circular_torus_matches_analytic_axis_azimuth_field():
    """
    On a circular torus around the z-axis, the harmonic field

        B_exact = ∇ϕ_a = (-y/(x^2+y^2), x/(x^2+y^2), 0)

    is tangent to the torus surface (n·B = 0). In our framework, this
    corresponds to the multivalued basis with a_hat = e_z and a = (0,1)
    (poloidal "twist"). The Neumann solve should give σ ≈ 0 and
    B ≈ B_exact everywhere inside the torus.
    """
    P, N = make_circular_torus(R0=1.5, a=0.4, n_theta=24, n_phi=48)
    Pn, scinfo = bie.normalize_geometry(P, verbose=False)
    Wn, h_n = bie.estimate_area_weights_knn(Pn, k=16)
    scale = float(scinfo.scale)
    W = Wn / (scale**2)
    h = h_n / scale
    h_min = float(jnp.min(h))

    a_hat = jnp.array([0.0, 0.0, 1.0])
    grad_t, grad_p = bie.multivalued_bases_about_axis(Pn, N, a_hat, verbose=False)

    # Use purely "poloidal" basis a = (0,1)
    Gp = grad_p(Pn)
    B_p_bdry = scale * Gp
    g_p = jnp.sum(N * B_p_bdry, axis=1)

    # n·B on Γ should already be ~0 analytically
    g_p_rms = float(jnp.linalg.norm(g_p) / jnp.sqrt(g_p.size))
    print("[TEST] torus g_p = n·B_p RMS =", g_p_rms)
    assert g_p_rms < 5e-3

    sigma2 = bie.solve_density_sigma(P, N, W, h, g_p, reg=1e-8)
    norm_sigma = float(jnp.linalg.norm(sigma2))
    print("[TEST] torus ||σ_p||_2 for g_p ≈ 0 =", norm_sigma)
    assert norm_sigma < 1e-1

    a = jnp.array([0.0, 1.0])
    phi_s_fn, B_fn, B_mv_fn = bie.make_total_field_evaluators_for_fixed_a(
        P, W, sigma2, scinfo, a, grad_t, grad_p, h_min, clip_factor=0.2
    )

    # Interior sample points: smaller minor radius
    R0 = 1.5
    a_minor = 0.3  # < 0.4
    thetas = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    phis = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    pts = []
    for th in thetas:
        for ph in phis:
            x = (R0 + a_minor * np.cos(th)) * np.cos(ph)
            y = (R0 + a_minor * np.cos(th)) * np.sin(ph)
            z = a_minor * np.sin(th)
            pts.append((x, y, z))
    X_int = np.array(pts, dtype=float)

    B_num = np.asarray(B_fn(jnp.asarray(X_int)))
    B_exact = analytic_axis_azimuth_grad(X_int)

    rel_err = np.linalg.norm(B_num - B_exact) / np.linalg.norm(B_exact)
    print("[TEST] torus analytic comparison rel_err =", rel_err)
    assert rel_err < 5e-2


# ---------------------------------------------------------------------
# Very lightweight end-to-end regression on a sphere with φ_edge = 0
# ---------------------------------------------------------------------

def test_zero_flux_sphere_gives_near_zero_field():
    """
    For a nearly spherical surface and phiedge = 0, the minimum-energy
    solution should have a ≈ 0 and σ ≈ 0, so B ≈ 0 throughout.
    We bypass the CSV I/O and re-implement the main steps directly.
    """
    P, N = make_unit_sphere(R=1.0)
    Pn, scinfo = bie.normalize_geometry(P, verbose=False)
    N = bie.maybe_flip_normals(P, N)
    Wn, h_n = bie.estimate_area_weights_knn(Pn, k=16)
    scale = float(scinfo.scale)
    W = Wn / (scale**2)
    h = h_n / scale
    h_min = float(jnp.min(h))

    kind, a_hat, E, svals = bie.detect_geometry_and_axis(Pn, verbose=False)
    grad_t, grad_p = bie.multivalued_bases_about_axis(Pn, N, a_hat, verbose=False)

    (Dt, Dp), (sigma1, sigma2), (B_t_bdry, B_p_bdry) = bie.solve_basis_fields(
        P, N, W, h, Pn, scinfo, grad_t, grad_p, reg=1e-8
    )
    (phi1_fn, B1_fn, B1_mv_fn), (phi2_fn, B2_fn, B2_mv_fn) = bie.build_basis_evaluators(
        P, W, h_min, scinfo, grad_t, grad_p, sigma1, sigma2
    )

    M, c, X_cs, B1_cs, B2_cs = bie.build_energy_flux_matrices_on_cross_section(
        P, a_hat, E, scinfo, B1_fn, B2_fn,
        n_r=12, n_theta=24, margin_frac=0.2
    )

    phiedge = 0.0
    a_star = bie.solve_energy_minimizing_coeffs(M, c, phiedge)
    print("[TEST] sphere zero-flux a* =", np.asarray(a_star))
    assert np.linalg.norm(np.asarray(a_star)) < 1e-6

    sigma_star = a_star[0] * sigma1 + a_star[1] * sigma2

    phi_s_star_fn, B_star_fn, B_mv_star_fn = bie.make_total_field_evaluators_for_fixed_a(
        P, W, sigma_star, scinfo, a_star, grad_t, grad_p, h_min, clip_factor=0.2
    )

    # Evaluate B at random interior points
    rng = np.random.default_rng(3)
    X_int = rng.normal(size=(100, 3))
    X_int *= 0.3  # definitely inside sphere of radius 1
    B_int = np.asarray(B_star_fn(jnp.asarray(X_int)))
    Bmag = np.linalg.norm(B_int, axis=1)
    rms = np.linalg.norm(Bmag) / np.sqrt(Bmag.size)
    print("[TEST] sphere zero-flux interior |B| RMS =", rms)
    assert rms < 5e-3

def make_simple_torus(R0=1.5, a=0.5, ntheta=16, nphi=32):
    """
    Simple parametric circular torus:
        x = (R0 + a cosθ) cosφ
        y = (R0 + a cosθ) sinφ
        z = a sinθ

    Outward normals:
        n ∝ (cosθ cosφ, cosθ sinφ, sinθ)
    """
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    th, ph = np.meshgrid(theta, phi, indexing="ij")

    x = (R0 + a * np.cos(th)) * np.cos(ph)
    y = (R0 + a * np.cos(th)) * np.sin(ph)
    z = a * np.sin(th)

    nx = np.cos(th) * np.cos(ph)
    ny = np.cos(th) * np.sin(ph)
    nz = np.sin(th)

    P = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    N = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)

    # Normalize normals just in case
    N /= np.linalg.norm(N, axis=1, keepdims=True)

    return jnp.asarray(P, dtype=jnp.float64), jnp.asarray(N, dtype=jnp.float64)

def test_single_layer_sphere_constant_sigma_has_small_gradient_inside():
    R = 1.0
    P, N = make_unit_sphere(R=R)
    Npts = P.shape[0]

    # Exact area weights
    area = 4.0 * np.pi * R**2
    W = jnp.ones(Npts) * (area / Npts)
    sigma = jnp.ones(Npts)

    # h_min consistent with previous tests
    h_min = float(((area / Npts) / np.pi) ** 0.5)

    phi_fn, grad_fn = bie.make_single_layer_evaluators(P, W, sigma, h_min)

    # Strictly inside the sphere
    X_inner = 0.5 * np.asarray(P)
    grad_inner = np.asarray(grad_fn(jnp.asarray(X_inner)))
    rms_inner = np.linalg.norm(grad_inner) / (np.sqrt(grad_inner.size) + 1e-14)

    # Points outside the sphere
    X_outer = 1.5 * np.asarray(P)
    grad_outer = np.asarray(grad_fn(jnp.asarray(X_outer)))
    rms_outer = np.linalg.norm(grad_outer) / (np.sqrt(grad_outer.size) + 1e-14)

    print("[TEST] constant-σ sphere |∇φ_s| RMS inside ≈", rms_inner)
    print("[TEST] constant-σ sphere |∇φ_s| RMS outside ≈", rms_outer)

    # Sanity checks:
    #  - interior gradient should be smaller than the exterior one
    #  - interior gradient should not be O(1)
    assert rms_inner < rms_outer
    assert rms_inner < 0.5


def test_jump_relation_matches_fd_normal_derivative_on_sphere():
    R = 1.0
    P, N = make_unit_sphere(R=R)
    Npts = P.shape[0]

    area = 4.0 * np.pi * R**2
    W = jnp.ones(Npts) * (area / Npts)

    # Smooth non-constant σ: use σ = z/R
    P_np = np.asarray(P)
    sigma = jnp.asarray(P_np[:, 2] / R)

    # h_min as in other sphere tests
    h_min = float(((area / Npts) / np.pi) ** 0.5)
    h_arr = jnp.ones(Npts) * h_min

    phi_fn, _ = bie.make_single_layer_evaluators(P, W, sigma, h_min)

    # Finite-difference ∂nφ at the boundary points
    # Slightly smaller step than before to reduce FD error
    h_fd = 0.05 * R
    N_np = np.asarray(N)
    X_plus = P_np + h_fd * N_np
    X_minus = P_np - h_fd * N_np

    phi_plus = np.asarray(phi_fn(jnp.asarray(X_plus)))
    phi_minus = np.asarray(phi_fn(jnp.asarray(X_minus)))
    dn_phi_fd = (phi_plus - phi_minus) / (2.0 * h_fd)

    # Jump formula: ∂n φ_s = -1/2 σ - K'σ
    Kprime = bie.build_Kprime(jnp.asarray(P), jnp.asarray(N), W, h_arr)
    dn_phi_jump = -0.5 * np.asarray(sigma) - np.asarray(Kprime @ sigma)

    # Compare direction and norm, not tiny residual
    num = float(np.dot(dn_phi_fd, dn_phi_jump))
    den = (np.linalg.norm(dn_phi_fd) * np.linalg.norm(dn_phi_jump) + 1e-14)
    corr = num / den  # cosine of angle between the two vectors

    norm_ratio = np.linalg.norm(dn_phi_fd) / (np.linalg.norm(dn_phi_jump) + 1e-14)

    print("[TEST] jump vs FD ∂nφ on sphere: corr ≈", corr,
          " norm_ratio ≈", norm_ratio)

    # We want them aligned and of similar magnitude, but not necessarily
    # matching to a few percent because of discretization + regularization.
    assert corr > 0.6       # reasonably aligned
    assert 0.5 < norm_ratio < 2.0  # within a factor of 2 in RMS norm

def test_compute_Bs_tan_zero_for_constant_sigma():
    R = 1.0
    P, N = make_unit_sphere(R=R)
    Npts = P.shape[0]

    area = 4.0 * np.pi * R**2
    W = jnp.ones(Npts) * (area / Npts)
    sigma = jnp.ones(Npts)

    h_min = float(((area / Npts) / np.pi) ** 0.5)
    h_arr = jnp.ones(Npts) * h_min

    Bs_tan = bie.compute_Bs_tan_on_boundary(P, N, W, sigma, h_arr)
    Bs_tan_np = np.asarray(Bs_tan)

    rms = np.linalg.norm(Bs_tan_np) / (np.sqrt(Bs_tan_np.size) + 1e-14)
    print("[TEST] Bs_tan for constant σ RMS ≈", rms)

    # For constant σ, weight (σ_j - σ_i) is exactly zero, so Bs_tan should be ~0
    assert rms < 1e-13

def test_detect_geometry_and_axis_on_sphere():
    R = 1.0
    P, N = make_unit_sphere(R=R)
    Pn, scinfo = bie.normalize_geometry(jnp.asarray(P), verbose=False)

    kind, a_hat, E, svals = bie.detect_geometry_and_axis(Pn, verbose=False)

    a_hat_np = np.asarray(a_hat)
    E_np = np.asarray(E)
    S_np = np.asarray(svals)

    print("[TEST] PCA sphere singular values =", S_np)
    print("[TEST] PCA sphere kind =", kind, " ||a_hat|| =", np.linalg.norm(a_hat_np))

    # Should at least give a unit vector and treat this as 'torus' by fallback
    assert kind == "torus"
    assert np.allclose(np.linalg.norm(a_hat_np), 1.0, atol=1e-12)

    # E should be approximately orthonormal
    ET_E = E_np.T @ E_np
    assert np.allclose(ET_E, np.eye(3), atol=1e-10)

def test_detect_geometry_and_axis_on_prolate_mirror():
    # Prolate ellipsoid elongated in z: expect 'mirror' and axis ~ ẑ
    R = 1.0
    stretch = 3.0

    ntheta, nphi = 32, 32
    theta = np.linspace(0.0, np.pi, ntheta, endpoint=True)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    th, ph = np.meshgrid(theta, phi, indexing="ij")

    x = R * np.sin(th) * np.cos(ph)
    y = R * np.sin(th) * np.sin(ph)
    z = stretch * R * np.cos(th)

    P = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    Pn, scinfo = bie.normalize_geometry(jnp.asarray(P), verbose=False)

    kind, a_hat, E, svals = bie.detect_geometry_and_axis(Pn, verbose=False)
    a_hat_np = np.asarray(a_hat)
    z_hat = np.array([0.0, 0.0, 1.0])

    cosang = abs(a_hat_np @ z_hat) / (np.linalg.norm(a_hat_np) + 1e-14)
    print("[TEST] prolate mirror kind =", kind, " cos(angle(a_hat,ẑ)) =", cosang)

    assert kind == "mirror"
    assert cosang > 0.9

def test_solve_energy_minimizing_coeffs_matches_closed_form():
    M = jnp.array([[2.0, 0.5],
                   [0.5, 1.0]])
    c = jnp.array([1.0, 2.0])
    phiedge = 3.0

    a_star = bie.solve_energy_minimizing_coeffs(M, c, phiedge)
    M_np = np.asarray(M)
    c_np = np.asarray(c)
    Minv = np.linalg.inv(M_np)
    a_expected = (phiedge / (c_np @ Minv @ c_np)) * (Minv @ c_np)

    print("[TEST] a_star =", np.asarray(a_star),
          " a_expected =", a_expected)

    assert np.allclose(np.asarray(a_star), a_expected, rtol=1e-12, atol=1e-12)

def test_superposition_of_basis_fields_is_respected():
    # Use a sphere as simple geometry for superposition test
    R = 1.0
    P, N = make_unit_sphere(R=R)
    Pn, scinfo = bie.normalize_geometry(jnp.asarray(P), verbose=False)

    # Quadrature in normalized coords, then rescale
    Wn, h_n = bie.estimate_area_weights_knn(Pn, k=16)
    scale = float(scinfo.scale)
    W = Wn / (scale**2)
    h = h_n / scale
    h_min = float(jnp.min(h))

    kind, a_hat, E, svals = bie.detect_geometry_and_axis(Pn, verbose=False)
    grad_t_fn, grad_p_fn = bie.multivalued_bases_about_axis(Pn, N, a_hat, verbose=False)

    (Dt, Dp), (sigma1, sigma2), _ = bie.solve_basis_fields(
        jnp.asarray(P), jnp.asarray(N), W, h, Pn, scinfo,
        grad_t_fn, grad_p_fn, reg=1e-6
    )

    (phi1_fn, B1_fn, B1_mv_fn), (phi2_fn, B2_fn, B2_mv_fn) = bie.build_basis_evaluators(
        jnp.asarray(P), W, h_min, scinfo, grad_t_fn, grad_p_fn, sigma1, sigma2
    )

    a = jnp.array([0.3, 1.1])
    sigma_star = a[0] * sigma1 + a[1] * sigma2

    phi_star_fn, B_star_fn, B_mv_star_fn = bie.make_total_field_evaluators_for_fixed_a(
        jnp.asarray(P), W, sigma_star, scinfo, a,
        grad_t_fn, grad_p_fn, h_min, clip_factor=0.2
    )

    # Evaluate in the interior
    X_eval = 0.7 * np.asarray(P)
    B_star = np.asarray(B_star_fn(jnp.asarray(X_eval)))
    B_lin = (float(a[0]) * np.asarray(B1_fn(jnp.asarray(X_eval))) +
             float(a[1]) * np.asarray(B2_fn(jnp.asarray(X_eval))))

    rel = np.linalg.norm(B_star - B_lin) / (np.linalg.norm(B_lin) + 1e-14)
    print("[TEST] superposition B(a) vs a₁B¹+a₂B² rel ≈", rel)

    assert rel < 1e-10

def test_Kprime_is_approximately_self_adjoint_under_weights():
    R = 1.0
    P, N = make_unit_sphere(R=R)
    Npts = P.shape[0]

    area = 4.0 * np.pi * R**2
    W = jnp.ones(Npts) * (area / Npts)

    # Use uniform h for regularization
    h_min = float(((area / Npts) / np.pi) ** 0.5)
    h_arr = jnp.ones(Npts) * h_min

    Kprime = bie.build_Kprime(jnp.asarray(P), jnp.asarray(N), W, h_arr)
    W_np = np.asarray(W)
    K_np = np.asarray(Kprime)

    # Test approximate self-adjointness of the bilinear form:
    #   A = diag(W) K'  should be close to symmetric.
    A = np.diag(W_np) @ K_np
    asym = A - A.T

    num = np.linalg.norm(asym)
    den = np.linalg.norm(A) + 1e-14
    rel = num / den
    print("[TEST] weighted K' asymmetry rel ≈", rel)

    assert rel < 0.05

def test_energy_matrix_positive_semidefinite_on_sphere_cross_section():
    R = 1.0
    P, N = make_unit_sphere(R=R)
    Pn, scinfo = bie.normalize_geometry(jnp.asarray(P), verbose=False)

    Wn, h_n = bie.estimate_area_weights_knn(Pn, k=16)
    scale = float(scinfo.scale)
    W = Wn / (scale**2)
    h = h_n / scale
    h_min = float(jnp.min(h))

    kind, a_hat, E, svals = bie.detect_geometry_and_axis(Pn, verbose=False)
    grad_t_fn, grad_p_fn = bie.multivalued_bases_about_axis(Pn, N, a_hat, verbose=False)

    (Dt, Dp), (sigma1, sigma2), _ = bie.solve_basis_fields(
        jnp.asarray(P), jnp.asarray(N), W, h, Pn, scinfo,
        grad_t_fn, grad_p_fn, reg=1e-6
    )

    (phi1_fn, B1_fn, B1_mv_fn), (phi2_fn, B2_fn, B2_mv_fn) = bie.build_basis_evaluators(
        jnp.asarray(P), W, h_min, scinfo, grad_t_fn, grad_p_fn, sigma1, sigma2
    )

    M, c, X_cs, B1_cs, B2_cs = bie.build_energy_flux_matrices_on_cross_section(
        jnp.asarray(P), a_hat, E, scinfo,
        B1_fn, B2_fn,
        n_r=16, n_theta=32, margin_frac=0.15
    )

    M_np = np.asarray(M)
    vals = np.linalg.eigvalsh(M_np)
    print("[TEST] eigenvalues of M =", vals)

    # Energy matrix should be positive semi-definite
    assert np.all(vals >= -1e-10)
    assert np.max(vals) > 0.0

def test_boundary_condition_residual_small_for_torus_basis_field():
    P, N = make_simple_torus(R0=1.5, a=0.5, ntheta=16, nphi=32)
    Pn, scinfo = bie.normalize_geometry(P, verbose=False)

    # kNN quadrature in normalized coords, then rescale
    Wn, h_n = bie.estimate_area_weights_knn(Pn, k=16)
    scale = float(scinfo.scale)
    W = Wn / (scale**2)
    h = h_n / scale
    h_min = float(jnp.min(h))

    kind, a_hat, E, svals = bie.detect_geometry_and_axis(Pn, verbose=False)
    grad_t_fn, grad_p_fn = bie.multivalued_bases_about_axis(Pn, N, a_hat, verbose=False)

    (Dt, Dp), (sigma1, sigma2), (B_t_bdry, B_p_bdry) = bie.solve_basis_fields(
        P, N, W, h, Pn, scinfo, grad_t_fn, grad_p_fn, reg=1e-6
    )

    # Build B on boundary for a = (1,0)
    a = jnp.array([1.0, 0.0])
    B_bdry = bie.build_B_on_boundary_with_jump(
        P, N, W, h,
        scinfo=scinfo,
        a=a,
        grad_t_fn=grad_t_fn,
        grad_p_fn=grad_p_fn,
        sigma=sigma1,
        g=Dt,
        clip_factor=0.2,
    )

    B_bdry_np = np.asarray(B_bdry)
    N_np = np.asarray(N)
    W_np = np.asarray(W)

    Bmag = np.linalg.norm(B_bdry_np, axis=1)
    n_dot_B = np.sum(N_np * B_bdry_np, axis=1)

    q = n_dot_B / (Bmag + 1e-12)
    flux = float(W_np @ n_dot_B)

    print("[TEST] torus basis-1: L2(q) =", np.linalg.norm(q),
          " Linf(q) =", np.max(np.abs(q)),
          " flux ≈", flux)

    # We expect the Neumann BC n·B≈0 to hold reasonably well
    assert np.linalg.norm(q) < 5e-2
    assert np.max(np.abs(q)) < 1e-1
    assert abs(flux) < 1e-2


def test_manufactured_harmonic_polynomial_inside_sphere():
    """
    Manufactured-solution test on a sphere:

      φ_exact(x,y,z) = x^2 - y^2,   ∇^2 φ_exact = 0

    We:
      1) Use the exact normal derivative ∂nφ_exact on the sphere as Neumann data.
      2) Solve for a single-layer density σ via
             (1/2 I - K') σ = f,
         where f = ∂nφ_exact and K' is bim.build_Kprime.
      3) Build φ_s from σ with make_single_layer_evaluators.
      4) Compare φ_s to φ_exact at interior points (r = R/2), up to an additive
         constant (Neumann gauge). We check the relative RMS error.

    With the current point-cloud Nyström + clipped kernel we expect O(10^-1)
    relative accuracy at this resolution, not machine precision.
    """

    R = 1.0
    P, N = make_unit_sphere(R=R)  # (Npts,3) points and normals on sphere
    Npts = P.shape[0]

    # Exact surface area and uniform patch weights for the sphere
    area = 4.0 * np.pi * R**2
    W = jnp.ones(Npts) * (area / Npts)

    # Use a uniform h_min consistent with the patch area: π h^2 ≈ area/N
    h_min = float(((area / Npts) / np.pi) ** 0.5)
    h_arr = jnp.ones(Npts) * h_min

    # Exact harmonic potential and normal derivative on the boundary
    P_np = np.asarray(P)
    x, y, z = P_np[:, 0], P_np[:, 1], P_np[:, 2]

    phi_exact_bdry = x**2 - y**2
    grad_exact = np.stack([2.0 * x, -2.0 * y, np.zeros_like(z)], axis=1)

    N_np = np.asarray(N)
    f = np.einsum("ij,ij->i", grad_exact, N_np)  # f = ∂n φ_exact

    # Clip factor used in the main code (can be tuned if desired)
    clip_factor_test = 0.2

    # Build K' with the same regularization as in the main code
    Kprime = bie.build_Kprime(
        jnp.asarray(P),
        jnp.asarray(N),
        W,
        h_arr,
        clip_factor=clip_factor_test,
    )

    # Solve (1/2 I - K') σ = f  (Neumann interior single-layer relation
    # for the actual φ_s implementation in bim.py)
    reg = 1e-8
    I = jnp.eye(Npts)
    A = 0.5 * I - Kprime + reg * I
    sigma = jnp.linalg.solve(A, jnp.asarray(f))

    # Build single-layer potential from σ
    phi_fn, _ = bie.make_single_layer_evaluators(
        jnp.asarray(P), W, sigma, h_min, clip_factor=clip_factor_test
    )

    # Sample interior points at radius R/2 along the same directions
    X_inner = 0.5 * P_np
    X_inner_j = jnp.asarray(X_inner)

    phi_num = np.asarray(phi_fn(X_inner_j))

    x_in, y_in, z_in = X_inner[:, 0], X_inner[:, 1], X_inner[:, 2]
    phi_exact_inner = x_in**2 - y_in**2

    # Neumann problem is defined up to an additive constant: align means
    delta = phi_num - phi_exact_inner
    const_shift = delta.mean()
    err = phi_num - (phi_exact_inner + const_shift)

    rel_rms = np.linalg.norm(err) / (np.linalg.norm(phi_exact_inner) + 1e-14)

    print("[TEST] manufactured φ = x^2 - y^2 on sphere:")
    print("       rel RMS error inside ≈", rel_rms)

    # With the brutal Nyström + clipped kernel we target O(10^-1) here.
    assert rel_rms < 0.2
