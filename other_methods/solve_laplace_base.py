"""
Baseline Laplace solver in tube coordinates (r, s, alpha) with Neumann BC.

Coordinates & mapping
---------------------
We use a normalized radial coordinate r ∈ [0,1], and set the physical tube as
    X(r,s,α) = r * a(s,α) * ν(s,α) + r0(s),
where r0(s) is the axis, ν(s,α) = cosα e1(s) + sinα e2(s) is the local
poloidal unit direction from the Bishop frame, and a(s,α) is the cross-section
radius map (from your Fourier DOFs).

Metric
------
Covariant basis: a_r = ∂X/∂r, a_s = ∂X/∂s, a_α = ∂X/∂α
G_ij = a_i · a_j,  √g = sqrt(det(G)), G^{ij} = (G_ij)^(-1)
Laplacian in curvilinear coords:
    ∇²φ = 1/√g ∂_i( √g G^{ij} ∂_j φ ),  i,j ∈ {r,s,α}

Boundary & circulations
-----------------------
We allow “circulation” via
    φ = φ̃ + G * s + I * α,
with φ̃ periodic in s,α. We solve
    ∇² φ̃ = - ∇² (G s + I α)
with Neumann (zero normal flux) on r=1 for φ̃ and periodicity in s,α.
At r=0 we use symmetry (∂φ̃/∂r = 0).
We remove the nullspace by enforcing mean(φ̃)=0 each iteration.

NOTE: For a coarse baseline we treat s∈[0,1) and α∈[0,2π) periodic. The “linear
parts” G s and I α are included as a forcing via RHS = -L(Gs+Iα). This is
consistent and yields nontrivial curl-free B fields.

Inputs
------
- theta, meta: the flattened parameter vector and its layout (from helpers.py)
- Nr, Ns, Na: grid sizes in r, s, α
- G, I: circulation amplitudes (floats)

Outputs
-------
- dict with phi (Nr,Ns,Na), B (Nr,Ns,Na,3), X (Nr,Ns,Na,3), and metric bits.

This is CPU NumPy and intentionally simple. It’s the right scaffold to JAX-ify
later (replace np with jnp, use jit/vmap, and plug into your adjoint).
"""

import numpy as np
from helpers_laplace import (solve_laplace_baseline, apply_laplacian, build_rhs_for_linear_part,
                             check_energy_symmetry, norm_w, check_neumann_bc, vol_mean_w, proj0_w,
                             plot_Bnorm_boundary_heatmap)
from helpers import plot_surface_package

theta = np.array([0.0, 0.0, 0.0, -0.12213994186083812, -0.06003682836054503, 0.4332772200865396, -0.5478979693508474, -0.09421107554844763, 0.24946508492382857, -0.9519926949447101, -0.1, -2.5131046182001966e-15, -1.356087420538574, -0.09421107554844732, -0.24946508492383432, -1.78184544802859, -0.06003682836054198, -0.43327722008653424, -1.903985389889412, 9.599029651964989e-16, 9.599010462526928e-15, -1.781845448028573, 0.060036828360545164, 0.43327722008653985, -1.3560874205385631, 0.09421107554844768, 0.24946508492382755, -0.9519926949446987, 0.1, -4.513615360402094e-15, -0.5478979693508406, 0.0942110755484474, -0.24946508492383287, -0.12213994186083645, 0.060036828360544685, -0.433277220086539, 0.1, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0], dtype=float)

meta = {
 'N_CTRL_AXIS': 12, 'N_CTRL_S': 12, 'm_list': (2, 3),
 'layout': [('axis_ctrl', None, 0, 36, (12, 3)),
            ('a0_cos', None, 36, 39, (3,)),
            ('a0_sin', None, 39, 39, (0,)),
            ('alpha0_cos', None, 39, 40, (1,)),
            ('alpha0_sin', None, 40, 40, (0,)),
            ('ec_specs', (2, 'cos'), 40, 42, (2,)),
            ('ec_specs', (2, 'sin'), 42, 42, (0,)),
            ('ec_specs', (3, 'cos'), 42, 43, (1,)),
            ('ec_specs', (3, 'sin'), 43, 44, (1,)),
            ('es_specs', (2, 'cos'), 44, 45, (1,)),
            ('es_specs', (2, 'sin'), 45, 45, (0,)),
            ('es_specs', (3, 'cos'), 45, 46, (1,)),
            ('es_specs', (3, 'sin'), 46, 46, (0,))]
}

# Choose a modest grid for a baseline
Nr, Ns, Na = 24, 240, 180
G, I = 0.1, 0.0   # start with no imposed circulation; try nonzero later

out = solve_laplace_baseline(theta, meta, Nr=Nr, Ns=Ns, Na=Na, Gc=G, Ic=I)

geom = out["geom"]
wvol = geom["wvol"]

# 1) L(constant) ≈ 0
c = np.ones((Nr,Ns,Na))
Lc = apply_laplacian(c, geom)
print("[test] ||L 1||_wvol =", norm_w(Lc, wvol))

# 2) Forcing consistency for Ic = 1
# α test — consistent with RHS construction
phi_lin_A = np.broadcast_to(geom["alpha"][None,None,:],
                            (Nr, Ns, Na)).copy()

Lalpha_raw = apply_laplacian(phi_lin_A, geom)
Lalpha     = proj0_w(Lalpha_raw, wvol)  # match RHS subspace

rhs2 = build_rhs_for_linear_part(geom, Gc=0.0, Ic=1.0)
print("[test] ||rhs + L(alpha)||_wvol =", norm_w(rhs2 + Lalpha, wvol))
print("[test] L(alpha) weighted mean:", vol_mean_w(Lalpha_raw, wvol))
print("[test] rhs2 weighted mean:",     vol_mean_w(rhs2,      wvol))

# 3) Symmetry again
check_energy_symmetry(geom, trials=3, seed=1)

# Optional: check boundary flux again after BC tau rows
phi_tilde = out["phi"] - (G * geom["s"][None,:,None] + I * geom["alpha"][None,None,:])
check_neumann_bc(phi_tilde, geom, sample_frac=0.5)

phi = out["phi"]   # (Nr,Ns,Na)
B   = out["B"]     # (Nr,Ns,Na,3)
X   = out["X"]     # (Nr,Ns,Na,3)

print(f"phi.shape = {phi.shape}, B.shape = {B.shape}, X.shape = {X.shape}")
print(f"phi min/max = {phi.min():.6f} / {phi.max():.6f}")
print(f"B min/max = {B.min():.6f} / {B.max():.6f}")
print(f"X min/max = {X.min():.6f} / {X.max():.6f}")

plot_Bnorm_boundary_heatmap(out["B"], out["geom"])

# boundary surface points (S,A,3) from your solver output
X_surf = out["geom"]["X"][-1]                 # r=1 boundary surface
B_surf = out["B"][-1]                         # (Ns,Na,3)
Bnorm_surf = np.linalg.norm(B_surf, axis=-1)  # (Ns,Na)

# Build a dict that matches what plot_surface_package expects:
surf_data = {
    "s"    : out["geom"]["s"],
    "alpha": out["geom"]["alpha"],
    "r0"   : out["geom"].get("r0", np.zeros((len(out["geom"]["s"]),3))),   # if stored
    "e1"   : out["geom"].get("e1", np.zeros((len(out["geom"]["s"]),3))),   # if stored
    "e2"   : out["geom"].get("e2", np.zeros((len(out["geom"]["s"]),3))),   # if stored
    "a"    : out["geom"].get("a", np.ones((len(out["geom"]["s"]),len(out["geom"]["alpha"])))), # if stored
    "X"    : X_surf,    # NOTE: this must be (S,A,3)
}

# 3D surface colored by |B| at the boundary
plot_surface_package(surf_data, surface_scalar=Bnorm_surf, scalar_label='|B|(r=1)')