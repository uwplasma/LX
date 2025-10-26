import jax
import jax.numpy as jnp
from jax import random, grad
import equinox as eqx

from _physics import u_total, grad_u_total_batch, lap_u_total_batch

def _identity(x):
    return x

class PotentialMLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key, hidden_sizes=MLP_HIDDEN_SIZES, act=MLP_ACT):
        depth = len(hidden_sizes) if hidden_sizes else 1
        width_val = (hidden_sizes[0] if hidden_sizes else 64)

        self.mlp = eqx.nn.MLP(
            in_size=3, out_size=1,
            width_size=width_val,
            depth=depth,
            key=key,
            activation=act,
            final_activation=_identity,
        )

    def __call__(self, xyz: jnp.ndarray) -> jnp.ndarray:
        out = self.mlp(xyz)
        return out.squeeze(-1)



def loss_fn_batch(params: PotentialMLP,
            pts_interior: jnp.ndarray, pts_bdry: jnp.ndarray,
            normals_bdry: jnp.ndarray,
            key: jax.Array):
    """Sample a random interior and boundary mini-batch and compute MSE losses."""
    ni = pts_interior.shape[0]
    nb = pts_bdry.shape[0]

    key_i, key_b = random.split(key)
    idx_i = random.choice(key_i, ni, (min(BATCH_IN, ni),), replace=False)
    idx_b = random.choice(key_b, nb, (min(BATCH_BDRY, nb),), replace=False)

    Pi = pts_interior[idx_i]
    Pb = pts_bdry[idx_b]
    Nb_hat = normals_bdry[idx_b]

    lap = lap_u_total_batch(params, Pi)                # [Bi]
    g_b = grad_u_total_batch(params, Pb)               # [Bb,3]
    n_dot = jnp.sum(Nb_hat * g_b, axis=-1)             # [Bb]

    loss_in = jnp.mean(lap * lap)
    loss_bc = jnp.mean(n_dot * n_dot)
    total   = loss_in + lam_bc * loss_bc
    return total, (loss_in, loss_bc, lap, n_dot)


# Filtered versions keep non-JAX parts (if any) out of JIT
loss_value_and_grad = eqx.filter_value_and_grad(loss_fn_batch, has_aux=True)

@eqx.filter_jit
def train_step(params, opt_state, optimizer,
               pts_interior, pts_bdry, normals_bdry, key):
    (loss_val, aux), grads = loss_value_and_grad(
        params, pts_interior, pts_bdry, normals_bdry, key
    )
    params_f, params_s = eqx.partition(params, eqx.is_inexact_array)
    grads_f,  _        = eqx.partition(grads,  eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads_f, opt_state, params_f)
    params_f = optax.apply_updates(params_f, updates)
    params = eqx.combine(params_f, params_s)
    return params, opt_state, loss_val, aux

@eqx.filter_jit
def eval_full(params: PotentialMLP,
              pts_interior: jnp.ndarray, pts_bdry: jnp.ndarray,
              normals_bdry: jnp.ndarray):
    """Evaluate losses and residuals on the full sets (deterministic)."""
    lap = lap_u_total_batch(params, pts_interior)      # [Ni]
    g_b = grad_u_total_batch(params, pts_bdry)         # [Nb,3]
    n_dot = jnp.sum(normals_bdry * g_b, axis=-1)       # [Nb]

    loss_in = jnp.mean(lap * lap)
    loss_bc = jnp.mean(n_dot * n_dot)
    total   = loss_in + lam_bc * loss_bc
    return total, (loss_in, loss_bc, lap, n_dot)

def save_model(model, path: Path = CHECKPOINT_PATH):
    """Serialize model parameters to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(path), model)
    print(f"[CKPT] Saved model to: {path}")

def load_model_if_exists(template: eqx.Module, path: Path = CHECKPOINT_PATH) -> eqx.Module:
    """If checkpoint exists, load into the provided template model; else return template."""
    if path.exists():
        try:
            loaded = eqx.tree_deserialise_leaves(str(path), template)
            print(f"[CKPT] Loaded model from: {path}")
            return loaded
        except Exception as e:
            print(f"[CKPT] Failed to load checkpoint ({path}). Using fresh model. Reason: {e}")
    else:
        print(f"[CKPT] No checkpoint found at {path}. Starting fresh.")
    return template