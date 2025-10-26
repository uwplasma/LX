import jax.numpy as jnp
from jax import random

# =============================================================================
# =============================== GEOMETRY ====================================
# =============================================================================

def a_of_phi(phi: jnp.ndarray) -> jnp.ndarray:
    """Minor radius a(φ) = a0 + a1 cos(N_harm φ)."""
    return a0 + a1 * jnp.cos(N_harm * phi)

def cylindrical_phi(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(y, x)

def inside_torus_mask(x, y, z) -> jnp.ndarray:
    """Inside test for variable-radius torus."""
    r   = jnp.sqrt(x*x + y*y)
    phi = cylindrical_phi(x, y)
    rho = jnp.sqrt((r - R0)**2 + z*z)        # distance to circular axis at angle φ
    return rho <= a_of_phi(phi)

def build_surface_torus(n_theta: int, n_phi: int):
    """
    Parameterization:
      r(θ,φ) = [(R0 + a(φ) cosθ) cosφ, (R0 + a(φ) cosθ) sinφ, a(φ) sinθ]
    Returns X,Y,Z with shape [nθ, nφ].
    """
    theta = jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=True)
    phi   = jnp.linspace(0, 2*jnp.pi, n_phi,   endpoint=True)
    Θ, Φ  = jnp.meshgrid(theta, phi, indexing='ij')    # [nθ,nφ]
    aφ    = a_of_phi(Φ)
    Rring = R0 + aφ * jnp.cos(Θ)
    X = Rring * jnp.cos(Φ)
    Y = Rring * jnp.sin(Φ)
    Z = aφ * jnp.sin(Θ)
    return X, Y, Z

def normals_from_param_grid(X, Y, Z):
    """
    Estimate outward normals n̂ on a periodic (θ,φ) grid via:
      n ∝ ∂r/∂θ × ∂r/∂φ, then normalize.
    """
    def circ_diff(A, axis):
        return (jnp.roll(A, -1, axis=axis) - jnp.roll(A, 1, axis=axis)) * 0.5

    dX_dθ, dY_dθ, dZ_dθ = circ_diff(X,0), circ_diff(Y,0), circ_diff(Z,0)
    dX_dφ, dY_dφ, dZ_dφ = circ_diff(X,1), circ_diff(Y,1), circ_diff(Z,1)

    tθ = jnp.stack([dX_dθ, dY_dθ, dZ_dθ], axis=-1)  # [nθ,nφ,3]
    tφ = jnp.stack([dX_dφ, dY_dφ, dZ_dφ], axis=-1)

    n  = jnp.cross(tθ, tφ, axis=-1)
    n_norm = jnp.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    n_hat  = n / n_norm
    return n_hat

def surface_points_and_normals(nθ, nφ):
    X, Y, Z = build_surface_torus(nθ, nφ)
    Nhat    = normals_from_param_grid(X, Y, Z)
    P       = jnp.stack([X, Y, Z], axis=-1)
    return (P.reshape(-1,3), Nhat.reshape(-1,3), X, Y, Z)  # flattened + original grids

def sample_interior(key, n_points: int, oversample_factor: int = 8):
    """
    Single-shot JAX sampler: oversample once, keep first n_points valid points.
    Falls back to a second pass with larger oversampling if needed.
    """
    def _one_shot(key, n_points, factor):
        a_max = a0 + jnp.abs(a1)
        Lxy   = R0 + a_max
        M     = factor * n_points
        kx, ky, kz = random.split(key, 3)
        X = random.uniform(kx, (M,), minval=-Lxy,  maxval=Lxy)
        Y = random.uniform(ky, (M,), minval=-Lxy,  maxval=Lxy)
        Z = random.uniform(kz, (M,), minval=-a_max, maxval=a_max)
        mask = inside_torus_mask(X, Y, Z)                 # [M] boolean
        # Fixed-size index vector (pads with zeros):
        idx  = jnp.nonzero(mask, size=n_points, fill_value=0)[0]  # [n_points]
        # How many valid did we actually get?
        got  = jnp.sum(mask)
        pts  = jnp.stack([X[idx], Y[idx], Z[idx]], axis=-1)       # [n_points,3]
        return pts, got

    pts, got = _one_shot(key, n_points, oversample_factor)
    # If not enough points, try once more with a bigger factor (Python guard, no JIT).
    if int(got) < n_points:
        pts, _ = _one_shot(key, n_points, oversample_factor * 2)
    return pts