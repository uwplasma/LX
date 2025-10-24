
from __future__ import annotations
"""
Vacuum Laplace Solver in a Toroidal Domain (JAX, Nyström BEM, differentiable)

Pedagogic overview
------------------
We want a curl-free, divergence-free vacuum magnetic field inside a torus:
    B = ∇Φ,    ∇·B = 0,    ∇×B = 0   in Ω,
with the boundary Γ being a magnetic flux surface: n·B = 0 on Γ
(i.e. homogeneous Neumann: n·∇Φ = 0 on Γ).

Because Ω (a solid torus) is not simply connected, a nontrivial B requires
Φ to be multi-valued: Φ must jump by a constant across a "cut" spanning one
hole. That jump is a *period* of B (its circulation around the corresponding
non-contractible loop). We model this by adding a double-layer potential on a
cut disk; its (uniform) density λ sets the potential jump and hence the
circulation. Physically, choosing λ means choosing how much harmonic, curl-free
field threads the hole.

Representation (layer potentials)
---------------------------------
We write
    Φ(x) = S_Γ[σ](x) + D_Scut[λ](x),

  • S_Γ[σ] is a single-layer potential on the boundary Γ:
        S_Γ[σ](x) = ∫_Γ G(x,y) σ(y) dS_y,
    with Laplace kernel  G(x,y) = 1 / (4π |x - y|).

  • D_Scut[λ] is a double-layer potential on the cut surface Scut:
        D_Scut[λ](x) = ∫_Scut ∂G/∂n_y (x,y) λ dS_y.
    Taking λ uniform gives one circulation degree of freedom.

We enforce  n·∇Φ = 0 on Γ, which becomes a linear system for the unknowns
u = [σ; λ]. This is a dense Nyström method (moderate N); swap the matvec for
an FMM when scaling up. Calderón preconditioning (not shown here) further
improves conditioning and is algebraic to add.

Design variables & geometry
---------------------------
We parameterize Γ by a "tube" built around a periodic centerline r0(s) with a
Bishop (parallel-transport) frame (e1,e2), and a cross-section radius function
a(s, α) with a few Fourier modes in α. All of these are smooth JAX functions of
a design vector p = {centerline DOFs, twist DOFs, cross-section DOFs, scales}.

Differentiation
---------------
We solve A(p) u = b(p) and get gradients via *implicit differentiation*:
  A u = b  ⇒  dJ/dp = J_p - λ_adj^T [A_p u - b_p],
where λ_adj solves A^T λ_adj = ∂J/∂u. We implement this with custom_vjp and
only differentiate the *assembly* (geometry → kernels). You can later swap in an
FMM for the matvec without changing the adjoint.

Notes, choices, tweaks
----------------------
• The “cut” provides one harmonic DOF. Add a second cut (or a few basis
  functions on the cut) to control toroidal + poloidal circulations separately.

• For interior surfaces (ψ-level sets) and field post-processing, we recommend
  computing them *a posteriori* by tracing surfaces and, if needed, fitting a
  Zernike basis per cross-section. For straight segments across radius, the
  tube-based radial coordinate (Chebyshev/Zernike in r) often stays better
  conditioned than global cylindrical coordinates.

• To scale up:
  – Replace lstsq with GMRES (or QR) wrapped in the same adjoint.
  – Add Calderón preconditioning (second-kind IE).
  – Swap dense matvec to FMM; adjoint is unchanged.

This file prints an example objective that “flattens |B|” along a straight-ish
segment of the axis, plus small regularizers. It also prints gradient norms of
the various DOF blocks to confirm differentiability.

Vacuum Laplace Solver in a Toroidal Domain
==========================================

(JAX, differentiable Nyström BEM with two independent circulations)

Pedagogic Overview
------------------
We seek a vacuum magnetic field inside a solid torus Ω with boundary Γ:

    B = ∇Φ,            ∇·B = 0,          ∇×B = 0     in Ω
    n · B = 0                          on Γ  (Neumann)

Because Ω is not simply connected, a nontrivial B requires Φ to be *multi-valued*:
Φ must jump by a constant across surfaces that span the torus holes (“cuts”).
Each jump sets a *period* (circulation) of B around the corresponding
non-contractible loop:

    ∮_γ B · dl = ΔΦ   (γ loops around a handle dual to the cut)

We model this via layer potentials:
    Φ(x) = S_Γ[σ](x) + D_{S_tor}[λ_tor](x) + D_{S_pol}[λ_pol](x),

  • S_Γ[σ](x)  = ∫_Γ        G(x,y) σ(y) dS_y        (single layer on Γ)
  • D_S[λ](x)  = ∫_S  ∂G/∂n_y (x,y) λ dS_y          (double layer on a cut S)

Here we take λ_tor and λ_pol as *uniform scalars* on their cuts, giving two
independent circulations (toroidal & poloidal). We enforce the Neumann boundary
condition n·∇Φ = 0 on Γ, which yields a linear system in the unknowns
u = [σ; λ_tor; λ_pol].

Discretization & Differentiation
--------------------------------
We use a dense Nyström method over Γ and the cuts:
- Γ is a tube built around a periodic centerline r0(s), using a Bishop frame
  (e1,e2) and cross-section radius a(s, α) with a few modes.
- The toroidal cut is a disk in the (e1,e2) plane at s=0.
- The poloidal cut is a “ribbon” surface X(ρ,s) = r0(s) + ρ r_dir(s) spanning
  the hole, with r_dir(s) = cos α_p e1(s) + sin α_p e2(s).

We solve A(p) u = b(p) and expose u(p) via a `custom_vjp` (implicit diff):
  A u = b  ⇒  dJ/dp = ∂J/∂p - λ_adjᵀ (A_p u - b_p),  with  Aᵀ λ_adj = ∂J/∂u.

This allows differentiating through the *geometry-to-system* assembly without
backpropagating through the linear solve iterations (we use a stable lstsq).
You can later swap the dense matvec for an FMM and/or add Calderón
preconditioning; the adjoint remains unchanged.

Performance Notes
-----------------
• All hot paths are JIT-compiled; heavy loops are vectorized via `vmap`.
• We keep kernels simple but numerically safe (small eps in distances).
• Boolean masking under JIT is avoided (static slicing is used).
• Double-layer gradient contributions from cuts are computed via autodiff in x.
  (For production, you can replace these with closed forms for more speed.)

Scaling
-------
• Add Calderón preconditioning (second-kind IE) to improve conditioning.
• Swap `lstsq` → GMRES or QR; the custom VJP stays the same.
• Replace dense matvecs by an FMM; no need to differentiate through the FMM.

This file prints:
- An example objective: “flatten |B|” along a fixed (static) axis segment.
- Gradient norms for each DOF block to confirm differentiability.

"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import jit, vmap, custom_vjp

# Enable float64 for better kernel accuracy
jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# USER TUNABLES (all inputs up front)
# -----------------------------------------------------------------------------

# Boundary Γ resolution
S_SAMPLES = 80         # axis samples s ∈ [0,1)
A_SAMPLES = 80         # poloidal angle samples α ∈ [0,2π)

# Toroidal cut (disk) resolution
CUT_NR = 24            # radial samples
CUT_NA = 64            # angular samples
CUT_RADIUS_FRACTION = 0.9  # r_cut = fraction * min(a0(s))

# Poloidal cut (ribbon) resolution
RIB_S_SAMPLES = 80     # s-samples on the ribbon (≤ S_SAMPLES)
RIB_NR = 24            # radial samples across the ribbon thickness
POLOIDAL_ALPHA0 = 0.0  # ribbon direction angle α_p in the cross-section

# Centerline control
N_CTRL = 12            # periodic cubic B-spline control points for r0(s)

# Cross-section Fourier modes (in α); e.g. (2,3) → ellipticity & triangularity
M_LIST = (2, 3)

# Segment used in the demo objective (static slice to avoid JIT boolean mask)
SEGMENT_FRACTION = 0.25   # first 25% of samples approximates s∈[0, 0.25)

# Regularization weights
REG_CENTERLINE = 1e-3
REG_XSEC      = 1e-3
REG_U         = 1e-6

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def softplus_pos(x: jnp.ndarray, eps: float = 1e-3) -> jnp.ndarray:
    """Positive parameterization with floor eps for robust geometry."""
    return jax.nn.softplus(x) + eps

def normalize(v: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Normalize vectors along the last axis with safety eps."""
    n = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.clip(n, eps, 1e12)

# -----------------------------------------------------------------------------
# Periodic cubic B-spline centerline
# -----------------------------------------------------------------------------

def periodic_bspline_basis(u: jnp.ndarray, k: int, n_ctrl: int) -> jnp.ndarray:
    """
    Periodic cardinal B-spline basis of degree k with n_ctrl control points.

    Args:
      u: shape (S,), parameter ∈ [0,1)
      k: degree (cubic = 3)
      n_ctrl: number of periodic control points

    Returns:
      B: shape (S, n_ctrl), basis functions evaluated at u.
    """
    def basis0(uu):
        idx = jnp.floor(uu * n_ctrl).astype(int) % n_ctrl
        return jax.nn.one_hot(idx, n_ctrl)

    B = vmap(basis0)(u)  # (S, n_ctrl)

    def step(B_prev, r):
        denom = r / n_ctrl
        ucell = (u * n_ctrl) % 1.0
        cl = ucell / jnp.maximum(denom, 1e-12)
        cr = (1.0 - ucell) / jnp.maximum(denom, 1e-12)
        return cl[:, None] * B_prev + cr[:, None] * jnp.roll(B_prev, -1, axis=-1)

    for r in range(1, k + 1):
        B = step(B, r)
    return B

def centerline_from_ctrl(ctrl: jnp.ndarray, s: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build centerline and tangent from periodic B-spline control points.

    Args:
      ctrl: (n_ctrl, 3) control points
      s:    (S,) samples in [0,1)

    Returns:
      r0(s): (S,3)
      t_hat(s): (S,3), unit tangent
    """
    n_ctrl = ctrl.shape[0]
    B = periodic_bspline_basis(s, k=3, n_ctrl=n_ctrl)
    r0 = B @ ctrl

    def r0_scalar(si):
        Bi = periodic_bspline_basis(si[None], k=3, n_ctrl=n_ctrl)[0]
        return Bi @ ctrl  # (3,)

    # d r0 / ds via autodiff of the scalarized interpolation
    dr0_ds = vmap(jax.jacfwd(r0_scalar))(s[:, None]).squeeze()
    t_hat = normalize(dr0_ds)
    return r0, t_hat

# -----------------------------------------------------------------------------
# Bishop (parallel-transport) frame + optional twist
# -----------------------------------------------------------------------------

def bishop_frame(t_hat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a minimal-twist frame (e1,e2) along a closed curve with tangents t_hat.

    Args:
      t_hat: (S,3), unit tangents

    Returns:
      e1,e2: (S,3), orthonormal vectors orthogonal to t_hat
    """
    S = t_hat.shape[0]
    z = jnp.array([0.0, 0.0, 1.0])
    e1_0 = normalize(jnp.cross(z, t_hat[0]))
    e1_0 = jnp.where(jnp.linalg.norm(e1_0) < 1e-8, jnp.array([1.0, 0.0, 0.0]), e1_0)
    e2_0 = jnp.cross(t_hat[0], e1_0)

    def step(carry, i):
        e1_prev, _ = carry
        t_next = t_hat[(i + 1) % S]
        # Transport by projection to the normal plane
        e1_tmp = e1_prev - jnp.dot(e1_prev, t_next) * t_next
        e1_new = normalize(e1_tmp)
        e2_new = jnp.cross(t_next, e1_new)
        return (e1_new, e2_new), (e1_new, e2_new)

    (_, _), (e1s, e2s) = jax.lax.scan(step, (e1_0, e2_0), jnp.arange(S - 1))
    e1s = jnp.vstack([e1_0, e1s])
    e2s = jnp.vstack([e2_0, e2s])
    return e1s, e2s

def apply_twist(e1s: jnp.ndarray, e2s: jnp.ndarray, theta_s: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Rotate (e1,e2) by twist angle θ(s).

    Args:
      e1s,e2s: (S,3)
      theta_s: (S,)

    Returns:
      e1,e2: rotated frames (S,3)
    """
    c, s = jnp.cos(theta_s)[:, None], jnp.sin(theta_s)[:, None]
    e1 = c * e1s + s * e2s
    e2 = -s * e1s + c * e2s
    return e1, e2

# -----------------------------------------------------------------------------
# Cross-section a(s, α)
# -----------------------------------------------------------------------------

def cross_section_a(params: Dict[str, jnp.ndarray],
                    s: jnp.ndarray,
                    alpha: jnp.ndarray) -> jnp.ndarray:
    """
    a(s, α) = softplus(a0(s)) · [1 + Σ_m (ec_m(s) cos m(α-α0(s)) + es_m(s) sin m(α-α0(s)))]

    Args:
      params: dict with keys "a0_s"(S,), "alpha0_s"(S,), "ec_s"(M,S), "es_s"(M,S), "m_list"(M,)
      s:      (S,) samples in [0,1)  (not used directly; arrays are already sampled on S)
      alpha:  (A,) samples in [0,2π)

    Returns:
      a: (S,A) cross-section radius
    """
    a0     = softplus_pos(params["a0_s"])
    alpha0 = params["alpha0_s"]
    mlist  = params["m_list"].astype(jnp.float64)
    ec     = params["ec_s"]
    es     = params["es_s"]

    S = a0.shape[0]
    A = alpha.shape[0]
    alpha_mat = alpha[None, :] - alpha0[:, None]
    contrib = jnp.zeros((S, A))
    for j, m in enumerate(mlist):
        contrib = contrib + ec[j, :, None] * jnp.cos(m * alpha_mat) \
                           + es[j, :, None] * jnp.sin(m * alpha_mat)
    return a0[:, None] * (1.0 + contrib)

# -----------------------------------------------------------------------------
# Surface Γ(s,α): points, normals, area weights
# -----------------------------------------------------------------------------

def tube_surface_points(r0: jnp.ndarray,
                        e1: jnp.ndarray,
                        e2: jnp.ndarray,
                        a: jnp.ndarray,
                        alpha: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build the tube surface Γ from centerline & frame.

    Returns:
      X:    (S,A,3) points
      n:    (S,A,3) unit normals (outward)
      dS:   (S,A)   area element per sample (|X_s × X_α|)
    """
    S, A = a.shape
    ca, sa = jnp.cos(alpha)[None, :], jnp.sin(alpha)[None, :]
    X = r0[:, None, :] + a[:, :, None] * (ca[:, :, None] * e1[:, None, :] +
                                          sa[:, :, None] * e2[:, None, :])

    # Tangents: s-direction by central diff (periodic), α-direction analytic
    Xp = jnp.roll(X, -1, axis=0)
    Xm = jnp.roll(X, +1, axis=0)
    Xs = (Xp - Xm) * 0.5
    Xalpha = a[:, :, None] * (-sa[:, :, None] * e1[:, None, :] +
                               ca[:, :, None] * e2[:, None, :])

    n = jnp.cross(Xs, Xalpha)
    n_hat = normalize(n)
    dS = jnp.linalg.norm(n, axis=-1)
    return X, n_hat, dS

# -----------------------------------------------------------------------------
# Laplace kernels
# -----------------------------------------------------------------------------

def dG_dn_y(x: jnp.ndarray, y: jnp.ndarray, n_y: jnp.ndarray) -> jnp.ndarray:
    """∂G/∂n_y (x,y) for Laplace G = 1/(4π|x-y|)."""
    rvec = x - y
    r = jnp.linalg.norm(rvec, axis=-1)
    return -jnp.sum(rvec * n_y, axis=-1) / (4.0 * jnp.pi * jnp.clip(r, 1e-12, 1e12) ** 3)

def gradG_x(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """∇_x G(x,y) for Laplace G = 1/(4π|x-y|)."""
    rvec = x - y
    r3 = jnp.clip(jnp.linalg.norm(rvec, axis=-1) ** 3, 1e-12, 1e12)
    return -rvec / (4.0 * jnp.pi * r3)[..., None]

# -----------------------------------------------------------------------------
# Poloidal ribbon (cut) geometry
# -----------------------------------------------------------------------------

def poloidal_ribbon_points(r0: jnp.ndarray,
                           t_hat: jnp.ndarray,
                           e1: jnp.ndarray,
                           e2: jnp.ndarray,
                           a0_s: jnp.ndarray,
                           alpha_p: float,
                           Nr: int,
                           Ns: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build a ribbon cut surface Scut_pol that spans the poloidal hole:

      X(ρ, s) = r0(s) + ρ · r_dir(s),     ρ ∈ [0, r_cut],  s sampled on [0,1)
      r_dir(s) = cos(α_p) e1(s) + sin(α_p) e2(s)

    Returns:
      Xcut_pol:   (Nr*Ns, 3) points
      ncut_pol:   (Nr*Ns, 3) unit normals (orientation follows Xs × Xρ)
      wcut_pol:   (Nr*Ns,)   quadrature weights  |Xs × Xρ| dρ
    """
    S_full = r0.shape[0]
    r_cut = CUT_RADIUS_FRACTION * jnp.min(a0_s)
    rho = jnp.linspace(0.0, r_cut, Nr)

    # s-indices (periodic subset of S_full)
    s_idx = (jnp.linspace(0, S_full, Ns, endpoint=False).astype(int)) % S_full

    rdir_full = jnp.cos(alpha_p) * e1 + jnp.sin(alpha_p) * e2
    rdir_s = rdir_full[s_idx]                     # (Ns,3)
    r0_s = r0[s_idx]                              # (Ns,3)

    RR, SS = jnp.meshgrid(rho, jnp.arange(Ns), indexing='ij')   # (Nr,Ns)
    rdir = rdir_s[SS]                                            # (Nr,Ns,3)
    base  = r0_s[SS]                                             # (Nr,Ns,3)
    X = base + RR[..., None] * rdir                              # (Nr,Ns,3)

    # Tangents: Xs via central diff on s-index; Xρ = rdir
    Xp = jnp.roll(X, -1, axis=1)
    Xm = jnp.roll(X, +1, axis=1)
    Xs = (Xp - Xm) * 0.5                                         # (Nr,Ns,3)
    Xrho = rdir                                                  # (Nr,Ns,3)

    n = jnp.cross(Xs, Xrho)
    n_hat = normalize(n)

    dr = rho[1] - rho[0] if Nr > 1 else r_cut
    w = jnp.linalg.norm(n, axis=-1) * dr                         # (Nr,Ns)

    return X.reshape((-1, 3)), n_hat.reshape((-1, 3)), w.reshape((-1,))

# -----------------------------------------------------------------------------
# Quadrature builders (Γ and both cuts)
# -----------------------------------------------------------------------------

def make_boundary_quadrature(p: Dict) -> Tuple:
    """
    Assemble geometry and quadrature nodes/weights for:
      - boundary Γ,
      - toroidal cut (disk at s=0),
      - poloidal cut (ribbon).

    Returns a tuple with everything the solver/evaluator needs.
    """
    # s, α grids on Γ
    S = S_SAMPLES
    A = A_SAMPLES
    s = jnp.linspace(0.0, 1.0, S, endpoint=False)
    alpha = jnp.linspace(0.0, 2 * jnp.pi, A, endpoint=False)

    # centerline + frame
    r0, t_hat = centerline_from_ctrl(p["ctrl"], s)
    e1, e2 = bishop_frame(t_hat)
    e1, e2 = apply_twist(e1, e2, p["twist_s"])

    # cross-section
    a = cross_section_a(p["xsec"], s, alpha)
    a0_s = softplus_pos(p["xsec"]["a0_s"])

    # boundary samples Γ
    X, n_hat, dS = tube_surface_points(r0, e1, e2, a, alpha)
    Xg = X.reshape((-1, 3))
    ng = n_hat.reshape((-1, 3))
    wg = dS.reshape((-1,))

    # Toroidal cut: disk in (e1,e2) plane at s=0
    r_cut = CUT_RADIUS_FRACTION * jnp.min(a0_s)
    rho = jnp.linspace(0.0, r_cut, CUT_NR)
    ang = jnp.linspace(0.0, 2 * jnp.pi, CUT_NA, endpoint=False)
    cr, sr = jnp.cos(ang), jnp.sin(ang)
    RR, AA = jnp.meshgrid(rho, ang, indexing='ij')  # (Nr,Na)
    Xcut_tor_grid = r0[0][None, None, :] + RR[:, :, None] * (
        cr[None, :, None] * e1[0][None, None, :] +
        sr[None, :, None] * e2[0][None, None, :]
    )
    Xcut_tor = Xcut_tor_grid.reshape((-1, 3))
    ncut_tor = jnp.tile(t_hat[0], (Xcut_tor.shape[0], 1))  # disk normal ~ +t_hat(0)
    dr_t = rho[1] - rho[0] if CUT_NR > 1 else r_cut
    dth = 2 * jnp.pi / CUT_NA
    W_tor = (RR * dr_t * dth).reshape((-1,))               # polar: ρ dρ dθ

    # Poloidal cut: ribbon through the hole
    Xcut_pol, ncut_pol, wcut_pol = poloidal_ribbon_points(
        r0=r0, t_hat=t_hat, e1=e1, e2=e2,
        a0_s=a0_s, alpha_p=POLOIDAL_ALPHA0,
        Nr=RIB_NR, Ns=RIB_S_SAMPLES
    )

    return (
        s, alpha, r0, t_hat, e1, e2, a,
        Xg, ng, wg,
        Xcut_tor, ncut_tor, W_tor,
        Xcut_pol, ncut_pol, wcut_pol
    )

# -----------------------------------------------------------------------------
# System assembly: enforce n·∇Φ = 0 on Γ
# -----------------------------------------------------------------------------

@jit
def assemble_system(Xg: jnp.ndarray, ng: jnp.ndarray, wg: jnp.ndarray,
                    Xcut_tor: jnp.ndarray, ncut_tor: jnp.ndarray, wcut_tor: jnp.ndarray,
                    Xcut_pol: jnp.ndarray, ncut_pol: jnp.ndarray, wcut_pol: jnp.ndarray
                    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build Neumann system:
       [A_nn | A_nc_tor | A_nc_pol] [σ; λ_tor; λ_pol] = 0

    Where:
      A_nn(i,j)  = n_i · ∇_x G(x_i, y_j) * w_j
      A_nc_*(i)  = n_i · ∇_x ∫_Scut ∂G/∂n_y (x_i, y) dS_y   (uniform λ_* scalar)
    """
    N = Xg.shape[0]
    x = Xg
    n = ng
    y = Xg

    # n · ∇_x S_Γ[σ]
    def entry_S(i, j):
        g = jnp.dot(n[i], gradG_x(x[i], y[j]))
        return g * wg[j]

    A_nn = vmap(lambda i: vmap(lambda j: entry_S(i, j))(jnp.arange(N)))(jnp.arange(N))

    # helper for one cut column
    def cut_column(Xc, nc, wc):
        def col_i(i):
            def phi_D_scalar(x_in):
                vals = vmap(lambda yy, nny, ww: dG_dn_y(x_in, yy, nny) * ww)(Xc, nc, wc)
                return jnp.sum(vals)
            g = jax.grad(phi_D_scalar)(x[i])         # ∇_x φ_D
            return jnp.dot(n[i], g)                  # n · ∇_x φ_D at x_i
        return vmap(col_i)(jnp.arange(N))            # (N,)

    A_nc_tor = cut_column(Xcut_tor, ncut_tor, wcut_tor)
    A_nc_pol = cut_column(Xcut_pol, ncut_pol, wcut_pol)

    A = jnp.concatenate([A_nn, A_nc_tor[:, None], A_nc_pol[:, None]], axis=1)  # (N, N+2)
    b = jnp.zeros((N,))
    return A, b

# -----------------------------------------------------------------------------
# Linear solve with implicit differentiation (custom_vjp)
# -----------------------------------------------------------------------------

def _solve_lstsq(A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Stable least-squares solve; differentiable and JIT-friendly."""
    u, *_ = jnp.linalg.lstsq(A, b, rcond=None)
    return u

@custom_vjp
def solve_u(p: Dict) -> Tuple[jnp.ndarray, Tuple]:
    """
    Assemble and solve for u = [σ; λ_tor; λ_pol], returning (u, aux).
    The `aux` carries geometry and system pieces for the backward pass.
    """
    (
        s, alpha, r0, t_hat, e1, e2, a,
        Xg, ng, wg,
        Xcut_tor, ncut_tor, wcut_tor,
        Xcut_pol, ncut_pol, wcut_pol
    ) = make_boundary_quadrature(p)

    A, b = assemble_system(
        Xg, ng, wg,
        Xcut_tor, ncut_tor, wcut_tor,
        Xcut_pol, ncut_pol, wcut_pol
    )
    u = _solve_lstsq(A, b)

    aux = (
        s, alpha, r0, t_hat, e1, e2, a,
        Xg, ng, wg,
        Xcut_tor, ncut_tor, wcut_tor,
        Xcut_pol, ncut_pol, wcut_pol,
        A, b, u
    )
    return u, aux

def solve_u_fwd(p: Dict):
    out = solve_u(p)
    return out, (out, p)

def solve_u_bwd(res, g):
    (u, aux), p = res
    gu, _ = g
    (
        s, alpha, r0, t_hat, e1, e2, a,
        Xg, ng, wg,
        Xcut_tor, ncut_tor, wcut_tor,
        Xcut_pol, ncut_pol, wcut_pol,
        A, b, u_primal
    ) = aux

    # Adjoint in unknown space: Aᵀ λ_adj = ∂J/∂u
    lam_adj = jnp.linalg.lstsq(A.T, gu, rcond=None)[0]

    # Pullback through (A,b) using VJP on the assembly function
    def A_b_of_p(p_in: Dict):
        (
            s2, a2, r02, t2, e12, e22, asec2,
            Xg2, ng2, wg2,
            Xct2, nct2, wct2,
            Xcp2, ncp2, wcp2
        ) = make_boundary_quadrature(p_in)
        A2, b2 = assemble_system(
            Xg2, ng2, wg2,
            Xct2, nct2, wct2,
            Xcp2, ncp2, wcp2
        )
        return A2, b2

    vjp_fun = jax.vjp(A_b_of_p, p)[1]
    # For residual R(A,b,u)=A u - b = 0, the contribution is -λ_adjᵀ (A_p u - b_p)
    cot_A = -(lam_adj[:, None] @ u_primal[None, :])
    cot_b = lam_adj
    (gp,) = vjp_fun((cot_A, cot_b))
    return (gp,)

solve_u.defvjp(solve_u_fwd, solve_u_bwd)

# -----------------------------------------------------------------------------
# Field evaluation: B = ∇Φ at targets Xt
# -----------------------------------------------------------------------------

@jit
def eval_B_at_targets(u: jnp.ndarray,
                      Xg: jnp.ndarray, ng: jnp.ndarray, wg: jnp.ndarray,
                      Xcut_tor: jnp.ndarray, ncut_tor: jnp.ndarray, wcut_tor: jnp.ndarray,
                      Xcut_pol: jnp.ndarray, ncut_pol: jnp.ndarray, wcut_pol: jnp.ndarray,
                      Xt: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate B = ∇Φ at a set of targets Xt using the solved densities.
    """
    N = Xg.shape[0]
    sigma   = u[:N]
    lam_tor = u[N]
    lam_pol = u[N + 1]

    # Single-layer gradient
    def grad_phi_S(x):
        # Sum_j ∇_x G(x, y_j) (w_j σ_j)
        g = vmap(lambda y, w, s: gradG_x(x, y) * (w * s),
                 in_axes=(0, 0, 0))(Xg, wg, sigma)
        return jnp.sum(g, axis=0)

    # Double-layer gradient from a cut (autodiff in x)
    def grad_phi_D_from_cut(x, Xc, nc, wc, lam):
        def phi_D_scalar(x_in):
            vals = vmap(lambda yy, nny, ww: dG_dn_y(x_in, yy, nny) * ww)(Xc, nc, wc)
            return lam * jnp.sum(vals)
        return jax.grad(phi_D_scalar)(x)

    def total_grad(x):
        gS = grad_phi_S(x)
        gDt = grad_phi_D_from_cut(x, Xcut_tor, ncut_tor, wcut_tor, lam_tor)
        gDp = grad_phi_D_from_cut(x, Xcut_pol, ncut_pol, wcut_pol, lam_pol)
        return gS + gDt + gDp

    return vmap(total_grad)(Xt)

# -----------------------------------------------------------------------------
# Design vector (initialization)
# -----------------------------------------------------------------------------

def make_design() -> Dict:
    """
    Build an initial design p with:
      - Circular centerline in XY of radius 1.0
      - Zero twist
      - Cross-section a0(s) mildly wavy, other modes zero
    """
    theta = jnp.linspace(0, 2 * jnp.pi, N_CTRL, endpoint=False)
    R0 = 1.0
    ctrl = jnp.stack([R0 * jnp.cos(theta),
                      R0 * jnp.sin(theta),
                      jnp.zeros_like(theta)], axis=1)  # (N_CTRL,3)

    twist_s = jnp.zeros((S_SAMPLES,))

    a0_s = 0.25 + 0.05 * jnp.sin(2 * jnp.pi * jnp.linspace(0, 1, S_SAMPLES))
    alpha0_s = jnp.zeros((S_SAMPLES,))
    M = len(M_LIST)
    ec_s = jnp.zeros((M, S_SAMPLES))
    es_s = jnp.zeros((M, S_SAMPLES))

    xsec = dict(a0_s=a0_s, alpha0_s=alpha0_s, m_list=jnp.array(M_LIST, dtype=jnp.float64),
                ec_s=ec_s, es_s=es_s)

    p = dict(ctrl=ctrl, twist_s=twist_s, xsec=xsec, scales=jnp.array([1.0]))
    return p

# -----------------------------------------------------------------------------
# Example objective: flatten |B| along a fixed s-segment of the axis
# -----------------------------------------------------------------------------

def objective_flatten_B(p: Dict) -> jnp.ndarray:
    """
    Example scalar objective suitable for gradient-based optimization.

    L = mean_{s in segment} (|B(s)| - mean|B|)^2
        + REG_CENTERLINE * ||Δ^2 ctrl||^2
        + REG_XSEC      * ||Δ^2 a0_s||^2
        + REG_U         * ||u||^2

    Notes:
      • The segment is a static slice of the axis samples to remain JIT-safe.
      • You can replace this with QS proxies, mirror shaping, etc.
    """
    (
        s, alpha, r0, t_hat, e1, e2, a,
        Xg, ng, wg,
        Xcut_tor, ncut_tor, wcut_tor,
        Xcut_pol, ncut_pol, wcut_pol
    ) = make_boundary_quadrature(p)

    u, _ = solve_u(p)

    # Evaluate on the first fraction of axis points (static slice)
    n1 = max(1, int(S_SAMPLES * SEGMENT_FRACTION))
    Xt = r0[:n1]

    B = eval_B_at_targets(u, Xg, ng, wg,
                          Xcut_tor, ncut_tor, wcut_tor,
                          Xcut_pol, ncut_pol, wcut_pol,
                          Xt)
    Bmag = jnp.linalg.norm(B, axis=1)
    Bbar = jnp.mean(Bmag)
    L_B = jnp.mean((Bmag - Bbar) ** 2)

    # Regularizers
    ctrl = p["ctrl"]
    ctrl2 = jnp.roll(ctrl, -1, axis=0) - 2 * ctrl + jnp.roll(ctrl, 1, axis=0)
    R_center = REG_CENTERLINE * jnp.mean(jnp.sum(ctrl2 ** 2, axis=1))

    a0 = softplus_pos(p["xsec"]["a0_s"])
    R_xsec = REG_XSEC * jnp.mean((jnp.roll(a0, -1) - 2 * a0 + jnp.roll(a0, 1)) ** 2)

    R_u = REG_U * jnp.mean(u ** 2)

    return L_B + R_center + R_xsec + R_u

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = make_design()

    obj = jit(objective_flatten_B)
    grad_obj = jit(jax.grad(objective_flatten_B))

    val = obj(p)
    g = grad_obj(p)

    print("Objective:", float(val))
    print("||∂L/∂ctrl||        :", float(jnp.linalg.norm(g["ctrl"])))
    print("||∂L/∂twist_s||     :", float(jnp.linalg.norm(g["twist_s"])))
    print("||∂L/∂a0_s||        :", float(jnp.linalg.norm(g["xsec"]["a0_s"])))
    print("||∂L/∂ec_s||        :", float(jnp.linalg.norm(g["xsec"]["ec_s"])))
    print("||∂L/∂es_s||        :", float(jnp.linalg.norm(g["xsec"]["es_s"])))
    print("||∂L/∂alpha0_s||    :", float(jnp.linalg.norm(g["xsec"]["alpha0_s"])))

if __name__ == "__main__":
    main()
