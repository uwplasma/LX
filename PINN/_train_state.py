# _train_state.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import equinox as eqx
import jax.numpy as jnp

class TrainState(eqx.Module):
    # Big trees (leaves)
    opt_state: Any
    ema_f: Any
    slow_f: Any

    # Small leaves (scalars)
    step: jnp.ndarray
    al_lambda: jnp.ndarray

    # Static (not serialized)
    params_fp: Optional[float] = eqx.field(static=True, default=None)
    version: int = eqx.field(static=True, default=2)
    key_train: Any = eqx.field(static=True, default=None)

def _template(dtype_int, dtype_float, like: Optional[TrainState] = None) -> TrainState:
    """Template matching dtypes and (if given) the **structure** of `like`."""
    if like is not None:
        # Keep the big-tree structures from `like`
        return TrainState(
            opt_state=like.opt_state,
            ema_f=like.ema_f,
            slow_f=like.slow_f,
            step=jnp.asarray(0, dtype=dtype_int),
            al_lambda=jnp.asarray(0.0, dtype=dtype_float),
            params_fp=None,
            version=2,
            key_train=None,  # static
        )
    # Fallback minimal template (rarely used; only works if file also used Nones)
    return TrainState(
        opt_state=None,
        ema_f=None,
        slow_f=None,
        step=jnp.asarray(0, dtype=dtype_int),
        al_lambda=jnp.asarray(0.0, dtype=dtype_float),
        params_fp=None,
        version=2,
        key_train=None,
    )

def save_train_state(path: str | Path,
                     opt_state,
                     ema_f,
                     slow_f,
                     step: int,
                     al_lambda: float,
                     params_fp: float | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = TrainState(
        opt_state=opt_state,
        ema_f=ema_f,
        slow_f=slow_f,
        step=jnp.asarray(int(step), dtype=jnp.int64),
        al_lambda=jnp.asarray(float(al_lambda), dtype=jnp.float64),
        params_fp=params_fp,
        version=2,
        key_train=None,
    )
    print(f"[STATE][SAVE] step={int(payload.step)} (dtype={payload.step.dtype}), "
          f"al_lambda={float(payload.al_lambda):.6g} (dtype={payload.al_lambda.dtype}), "
          f"params_fp={params_fp} [static], version=2 [static], path={path}")
    eqx.tree_serialise_leaves(str(path), payload)
    print(f"[STATE] Saved training state to: {path}")

def load_train_state(path: str | Path, like: Optional[TrainState] = None) -> Optional[TrainState]:
    path = Path(path)
    if not path.exists():
        return None

    last_err = None
    for d_int, d_float, tag in [(jnp.int64, jnp.float64, "64bit"),
                                (jnp.int32, jnp.float32, "32bit-FALLBACK")]:
        try:
            tmpl = _template(d_int, d_float, like=like)
            payload = eqx.tree_deserialise_leaves(str(path), tmpl)
            print(f"[STATE][LOAD] OK ({tag}) from: {path}  "
                  f"(step={int(payload.step)}, al_lambda={float(payload.al_lambda):.6g}, "
                  f"version={getattr(payload, 'version', None)})")
            if getattr(payload, "version", 0) != 2:
                print(f"[STATE][LOAD][WARN] version={getattr(payload,'version',None)} != 2 (continuing).")
            return payload
        except Exception as e:
            last_err = e
            continue

    print(f"[STATE] Failed to load training state ({path}). Starting fresh. Reason: {last_err!r}")
    print("[STATE][HINT] If this persists, delete the .trainstate.eqx file so we can regenerate it.")
    return None
