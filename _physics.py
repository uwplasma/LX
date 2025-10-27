import jax.numpy as jnp
import equinox as eqx
import jax
from _state import runtime

def u_multivalued(xyz: jnp.ndarray) -> jnp.ndarray:
    x, y = xyz[..., 0], xyz[..., 1]
    return runtime.kappa * jnp.arctan2(y, x) / runtime.R0

def grad_u_mv(xyz: jnp.ndarray) -> jnp.ndarray:
    """∇(kappa * atan2(y,x)/R0) = kappa/R0 * (-y/(x^2+y^2), x/(x^2+y^2), 0)"""
    x, y = xyz[..., 0], xyz[..., 1]
    r2 = x*x + y*y
    inv = jnp.where(r2 > 1e-18, 1.0 / r2, 0.0)
    gx, gy = -y * inv, x * inv
    gz = jnp.zeros_like(x)
    return (runtime.kappa / runtime.R0) * jnp.stack([gx, gy, gz], axis=-1)

@eqx.filter_jit
def lap_u_mv_zero(_params, _xyz):
    # Laplacian of atan2(y,x) is 0 away from r=0 (singular on axis only).
    return jnp.array(0.0)

def u_total(params, xyz: jnp.ndarray) -> jnp.ndarray:
    return u_multivalued(xyz) + params(xyz)

# Gradient of the NN-only part
@eqx.filter_jit
def grad_u_nn_scalar(params, xyz: jnp.ndarray) -> jnp.ndarray:
    return jax.grad(lambda q: params(q))(xyz)

# Laplacian via three Hessian–vector products (no full Hessian materialization)
# e_i are standard basis vectors in R^3
@eqx.filter_jit
def lap_u_nn_scalar(params, xyz: jnp.ndarray) -> jnp.ndarray:
    """Compute Laplacian via three JVPs with dtype-safe unit vectors."""
    def g(q):  # grad of NN-only potential
        return jax.grad(lambda q_: params(q_))(q)

    e = jnp.eye(3, dtype=xyz.dtype)  # unit basis with the SAME dtype as xyz
    _, He1 = jax.jvp(g, (xyz,), (e[0],))
    _, He2 = jax.jvp(g, (xyz,), (e[1],))
    _, He3 = jax.jvp(g, (xyz,), (e[2],))
    return He1[0] + He2[1] + He3[2]

# Batched wrappers
# u_nn_batch = jax.vmap(lambda p, q: p(q), in_axes=(None, 0))
u_total_batch   = jax.vmap(u_total, in_axes=(None, 0))
grad_u_nn_batch = jax.vmap(grad_u_nn_scalar, in_axes=(None, 0))
lap_u_nn_batch  = jax.vmap(lap_u_nn_scalar,  in_axes=(None, 0))

@eqx.filter_jit
def grad_u_total_batch(params, xyz_batch: jnp.ndarray) -> jnp.ndarray:
    return grad_u_mv(xyz_batch) + grad_u_nn_batch(params, xyz_batch)

@eqx.filter_jit
def lap_u_total_batch(params, xyz_batch: jnp.ndarray) -> jnp.ndarray:
    # Lap(u_mv)=0 away from r=0  ⇒ just Lap(u_nn)
    return lap_u_nn_batch(params, xyz_batch)

def eval_on_boundary(params, P_bdry, N_bdry, Xg, Yg, Zg):
    """
    Returns:
      Gvec: (nθ, nφ, 3)  gradient vectors at boundary grid
      Gmag: (nθ, nφ)     |∇u| at boundary grid
    """
    Gvec_flat = grad_u_total_batch(params, P_bdry)
    Gmag_flat = jnp.linalg.norm(Gvec_flat, axis=-1)
    Gvec = Gvec_flat.reshape(Xg.shape + (3,))
    Gmag = Gmag_flat.reshape(Xg.shape)
    Nhat = N_bdry.reshape(Xg.shape + (3,))
    return Gvec, Gmag, Nhat