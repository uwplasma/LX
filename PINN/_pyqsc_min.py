# _physics.py (or a new _forward_targets.py if you prefer)
import jax
import jax.numpy as jnp

# Example signature; implement inside using your pyQSC data & surf_id
@jax.jit
def grad_u_forward_batch(xyz_batch: jnp.ndarray, surf_id: int) -> jnp.ndarray:
    """
    Return target gradient field from pyQSC for these xyz points on the given surface.
    Shape: (N,3). Must be pure JAX and VMAP-able.
    """
    # TODO: your code here — either evaluate pyQSC directly (if JAXable)
    # or retrieve from precomputed arrays aligned with xyz_batch.
    raise NotImplementedError
