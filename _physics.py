import jax.numpy as jnp
import equinox as eqx
import jax
from _state import runtime

EYE3_F32 = jnp.eye(3, dtype=jnp.float32)
EYE3_F64 = jnp.eye(3, dtype=jnp.float64)
def _eye3_like(x):
    return EYE3_F64 if x.dtype == jnp.float64 else EYE3_F32

@jax.jit
def u_multivalued(xyz: jnp.ndarray) -> jnp.ndarray:
    x, y = xyz[..., 0], xyz[..., 1]
    return runtime.kappa * jnp.arctan2(y, x) / runtime.R0

@jax.jit
def grad_u_mv(xyz: jnp.ndarray) -> jnp.ndarray:
    """∇(kappa * atan2(y,x)/R0) = kappa/R0 * (-y/(x^2+y^2), x/(x^2+y^2), 0)"""
    x, y = xyz[..., 0], xyz[..., 1]
    r2 = x * x + y * y
    inv = jnp.where(r2 > 1e-18, 1.0 / r2, 0.0)
    gx, gy = -y * inv, x * inv
    gz = jnp.zeros_like(x)
    return (runtime.kappa / runtime.R0) * jnp.stack([gx, gy, gz], axis=-1)

@eqx.filter_jit
def u_total(params, xyz: jnp.ndarray) -> jnp.ndarray:
    return u_multivalued(xyz) + params(xyz)

def _call_params(p, q):
    return p(q)

# Gradient of the NN-only part
@eqx.filter_jit
def grad_u_nn_scalar(params, xyz: jnp.ndarray) -> jnp.ndarray:
    return jax.grad(lambda q: _call_params(params, q))(xyz)

# Laplacian via three Hessian–vector products (no full Hessian materialization)
# e_i are standard basis vectors in R^3
@eqx.filter_jit
def lap_u_nn_scalar(params, xyz: jnp.ndarray) -> jnp.ndarray:
    """
    Laplacian of the NN-only potential using a single reverse pass + forward JVPs.

    We linearize g(x) = ∇u_nn(x) at the query point:
        g_x, jvp_lin = jax.linearize(g, xyz)
    Then columns of the Hessian H are jvp_lin(e_i). The Laplacian is trace(H).
    This avoids re-evaluating the reverse pass three times.
    """
    def g(q):
        return jax.grad(lambda q_: params(q_))(q)

    # One reverse pass to build the linearized JVP
    _, jvp_lin = jax.linearize(g, xyz)

    # Apply to basis to get columns of H; trace = sum diag = sum_i (H e_i)_i
    cols = jax.vmap(jvp_lin)(_eye3_like(xyz))  # (3,3)
    return jnp.sum(jnp.diag(cols))

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

@eqx.filter_jit
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
