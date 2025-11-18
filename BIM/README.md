We compute a vacuum magnetic field B = ∇φ inside a closed 3D surface Γ
given by a point cloud (x,y,z) and outward normals n, enforcing

    n · B = 0  on Γ   (perfect conductor, field lines tangent to Γ),

and selecting among all such harmonic fields the one that

  (i)  has a prescribed *toroidal flux* phiedge, defined as
       Φ_tor = ∫_S B·a_hat dS through a poloidal cross-section S
       orthogonal to the PCA-based axis a_hat; and

  (ii) minimizes the *magnetic energy*

          E = ½ ∫_Ω |B|² dV,

       approximated by revolving that cross-section around the
       straight axis line:

          dV ≈ (2π ρ) dS,  where ρ is the distance to the axis.

Representation: MFS + multivalued bases
---------------------------------------

We represent φ as

    φ(x) = φ_mv(x) + φ_s(x),

where

  - φ_mv is a multi-valued harmonic potential whose gradient
    B_mv = ∇φ_mv spans the topological (toroidal / poloidal) space.
    We use two axis-aware multivalued basis fields:

        B_mv(x; a) = a_t b_t(x) + a_p b_p(x),  a = (a_t, a_p)^T.

  - φ_s is a *single-valued* harmonic potential represented by a sum
    of fundamental solutions (Method of Fundamental Solutions, MFS):

        φ_s(x) = ∑_j c_j G(x, y_j),
        G(x,y) = 1/(4π|x-y|),

    where the source points {y_j} lie on a *fictitious outer surface*
    Γ^d constructed by offsetting Γ along its outward normals:

        y_j = x_j + d_j n_j,   d_j ≈ source_offset_factor * h_j,

    with h_j a local spacing estimate.

The field is

    B(x) = ∇φ(x) = B_mv(x) + B_s(x),    B_s = ∇φ_s.

No singular integrals appear in the MFS formulation because all sources
are strictly outside the physical domain.

Boundary condition and basis solves
-----------------------------------

On the boundary Γ:

    n · B = 0  ⇒  n·B_mv + n·B_s = 0.

Using the MFS representation of B_s, this becomes

    ∑_j c_j [n_i · ∇_x G(x_i, y_j)] = - g_i,
    g_i = n_i · B_mv(x_i),

for boundary points x_i ∈ Γ with unit outward normals n_i.

We define the dense MFS Neumann matrix

    A_ij = n_i · ∇_x G(x_i, y_j),

and solve

    A c = -g.

To parameterize the topological space of vacuum fields, we choose two
multivalued basis fields:

  - grad_t(x) : constant harmonic field along the PCA axis (toroidal).
  - grad_p(x) : azimuthal harmonic field looping poloidally around the axis.

For each basis choice a^(1)=(1,0), a^(2)=(0,1), we solve:

  n·B_mv^{(k)}(x_i) = g^{(k)}_i,
  A c^{(k)} = -g^{(k)},   k = 1,2,

to obtain two full vacuum fields:

  B^{(k)}(x) = B_mv^{(k)}(x) + B_s^{(k)}(x),   k=1,2.

Any linear combination

    B(a) = a_1 B^{(1)} + a_2 B^{(2)}

is a divergence-free, harmonic field with n·B = 0 on Γ.

VMEC-like energy and toroidal flux
----------------------------------

We then construct a *poloidal cross-section* S:

  - Axis direction a_hat from a PCA of normalized coordinates (kind="torus").
  - Center at the geometric center of the point cloud.
  - Plane orthogonal to a_hat.
  - Polar grid (r,θ) inside the torus footprint.

On this cross-section, we define:

  - Area quadrature weights: dS = r dr dθ.
  - Toroidal flux:

        Φ_tor(a) = ∫_S B(a)·a_hat dS
                  ≈ ∑_q w_q (B(a,x_q)·a_hat).

  - Energy matrix M via an approximate volume integral:

        E(a) = ½ ∫_Ω |B(a)|² dV
              ≈ ½ ∑_q (2π ρ_q w_q) |B(a,x_q)|²
              = ½ aᵀ M a,

    where ρ_q is the distance from x_q to the axis line, and
    M_{ij} = ∑_q 2π ρ_q w_q [B^{(i)}(x_q)·B^{(j)}(x_q)].

Given a user-specified phiedge = Φ_tor target, we solve the quadratic
program

    minimize  E(a) = ½ aᵀ M a
    subject to   cᵀ a = phiedge,

with c_i = Φ_tor(B^{(i)}). The solution is analytic:

    a* = (phiedge / (cᵀ M⁻¹ c)) M⁻¹ c.

Diagnostics and plotting
------------------------

The script prints:

  - Geometry and PCA diagnostics
  - kNN-based quadrature / spacing stats
  - MFS source offset diagnostics
  - Condition numbers and norms for MFS systems
  - Boundary residuals n·B and n·B/|B| on Γ
  - Flux checks on boundary and on an inner shell
  - Laplacian residual ∇²φ_s on an inner shell (FD approximation)
  - Decomposition of B into multivalued and MFS parts on Γ
  - Correlations of |B| with distance to axis ρ and toroidal angle φ

and produces publication-ready plots (all saved under ../outputs):

  1. 3D: boundary colored by |B| and n·B/|B|.
  2. 1D: n·B/|B| on Γ; ∇²φ_s on inner shell.
  3. 1D: |B|, |B_mv|, |B_s| vs toroidal angle φ.
  4. 1D: |B|, |B_mv|, |B_s| vs distance to axis ρ.
  5. 1D: geometric/quad diagnostics (ρ, W, h) vs boundary index.

JAX and differentiability
-------------------------

All core pieces (MFS assembly, solves, field evaluation, energy and
flux integrals) are written in JAX using jnp, jit, and vmap. This makes
the vacuum field and energy functional differentiable with respect to
geometry (point cloud P, normals N, etc.) for use in shape optimization
and design.