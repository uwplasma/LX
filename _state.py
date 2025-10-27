# _state.py
from dataclasses import dataclass

@dataclass
class Runtime:
    # geometry
    R0: float = 1.0
    a0: float = 0.35
    a1: float = 0.20
    N_harm: int = 3
    # physics
    kappa: float = 0.0
    # batching/loss
    BATCH_IN: int = 2048
    BATCH_BDRY: int = 2048
    lam_bc: float = 5.0

runtime = Runtime()