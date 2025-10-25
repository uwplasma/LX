from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from helpers import (
    plot_surface_package,
    make_spitzer_like_fig8_ctrl,
    make_param_dict, pack_params, theta_to_surface
)

# ------------------------------ USER INPUTS ----------------------------------
S_SAMPLES = 240
A_SAMPLES = 180

# Axis: either your numeric array or the generator (using the generator here)
N_CTRL_AXIS = 14
AXIS_CTRL = make_spitzer_like_fig8_ctrl(
    N_CTRL=N_CTRL_AXIS,
    L_plateau=1.25,
    R_bend=0.30,
    k_plateau=4.0,
    z_gap=0.12,
    twist_turns=0,
    enforce_stellarator_sym=True,
)

# Periodic s-control count (sampling grid for the Fourier series)
N_CTRL_S = 16

# α-mode list
M_LIST = (2, 3)

# ---------- Fourier numbers (what the optimizer controls) ----------
# Example: a0(s) = 0.07 + 0.015 cos(2π s) + 0.01 cos(4π s)
a0_cos = np.array([0.07, 0.015, 0.01])
a0_sin = np.array([])

# α0(s) = 0
alpha0_cos = np.array([0.0])
alpha0_sin = np.array([])

# ec_m(s), es_m(s) specs as Fourier-in-s numbers
ec_specs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
    2: (np.array([0.03, 0.01]), np.array([])),      # ec_2(s) = 0.03 + 0.01 cos(2π s)
    3: (np.array([0.02]),       np.array([0.01])),  # ec_3(s) = 0.02 + 0.01 sin(2π s)
}
es_specs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
    2: (np.array([0.00]),       np.array([])),      # es_2(s) = 0
    3: (np.array([0.00]),       np.array([])),      # es_3(s) = 0
}

def main():
    # 1) Build a single parameter dict of *numbers*
    P = make_param_dict(
        axis_ctrl=AXIS_CTRL,
        N_CTRL_S=N_CTRL_S,
        m_list=M_LIST,
        a0_cos=a0_cos, a0_sin=a0_sin,
        alpha0_cos=alpha0_cos, alpha0_sin=alpha0_sin,
        ec_specs=ec_specs,
        es_specs=es_specs,
    )

    # 2) Pack → single vector theta (for your optimizer), and meta for unpacking
    theta, meta = pack_params(P)
    print(f"[pack] theta.shape = {theta.shape}, total DOFs = {theta.size}")

    # 3) Build geometry from theta (this is what you'd do inside an objective)
    data = theta_to_surface(theta, meta, s_samples=S_SAMPLES, alpha_samples=A_SAMPLES)

    # 4) Visualize (unchanged)
    plot_surface_package(
        data,
        n_frame_arrows=24,
        n_cross_sections=8,
        figsize=(15, 10),
    )

if __name__ == "__main__":
    main()
