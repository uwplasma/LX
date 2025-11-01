# _train_state.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Tuple
import jax
import equinox as eqx
import jax.numpy as jnp

# What we persist
# - opt_state:       Optax optimizer state (pytree)
# - ema_f:           EMA of trainable params (or None)
# - slow_f:          Lookahead slow weights (or None)
# - step:            Global Adam step count (int inferred from opt_state if missing)
# - al_lambda:       Augmented Lagrange multiplier λ_AL
# - key_train:       PRNGKey to continue exact sampling sequence

def leaves_fingerprint(params_f) -> float:
    # Simple, stable scalar summary of trainable leaves
    return float(sum([jnp.sum(x * x) for x in jax.tree_util.tree_leaves(params_f)]))

class TrainState(eqx.Module):
    opt_state: Any
    ema_f: Any
    slow_f: Any
    step: int
    al_lambda: float
    key_train: Any
    params_fp: Optional[float] = None

def _default_none():
    return None


def save_train_state(path: str | Path,
                     opt_state,
                     ema_f,
                     slow_f,
                     step: int,
                     al_lambda: float,
                     key_train,
                     params_fp: float | None = None) -> None:   # <-- NEW ARG (optional)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = TrainState(opt_state=opt_state,
                         ema_f=ema_f,
                         slow_f=slow_f,
                         step=int(step),
                         al_lambda=float(al_lambda),
                         key_train=key_train,
                         params_fp=params_fp)                     # <-- SAVE IT
    eqx.tree_serialise_leaves(str(path), payload)
    print(f"[STATE] Saved training state to: {path}")

def _empty_train_state_template() -> TrainState:
    # Minimal placeholders of correct dtypes/shapes
    return TrainState(
        opt_state=None,
        ema_f=None,
        slow_f=None,
        step=0,
        al_lambda=0.0,
        key_train=jax.random.PRNGKey(0),
        params_fp=None,
    )

def load_train_state(path: str | Path) -> Optional[TrainState]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        template = _empty_train_state_template()
        payload = eqx.tree_deserialise_leaves(str(path), template)
        print(f"[STATE] Loaded training state from: {path}")
        return payload
    except Exception as e:
        print(f"[STATE] Failed to load training state ({path}). Starting fresh. Reason: {e}")
        return None
