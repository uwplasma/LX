# LX: Differentiable Laplace Solver for Toroidal Magnetic Fields

**University of Wisconsin–Madison — UWPlasma Research Group**

---

## Overview

**LX** is a differentiable, JAX-based solver for the Laplace equation inside toroidal domains (such as stellarator or mirror geometries). It computes **vacuum magnetic fields**  

$$
\nabla^2 \Phi = 0, \quad \mathbf{B} = \nabla \Phi,
$$  

subject to **magnetic flux surface** boundary conditions  

$$
\mathbf{n} \cdot \mathbf{B} = 0 \quad \text{on } \Gamma,
$$  

where the surface $$\Gamma$$ represents a magnetic surface of constant flux, analogous to the flux surfaces in magnetohydrodynamic (MHD) equilibrium codes like [DESC](https://desc-docs.readthedocs.io/en/stable/).

LX supports **automatic differentiation**, allowing exact gradients of objectives (e.g. magnetic field uniformity, quasi-symmetry proxies) with respect to all geometry degrees of freedom.  
This makes LX suitable for **geometry optimization**, **shape design**, and **inverse problems** in plasma confinement.

---

## Physics Background

The magnetic field in a current-free region (no plasma currents, no coils) satisfies:

$$
\nabla \cdot \mathbf{B} = 0, \quad \nabla \times \mathbf{B} = 0.
$$

Thus, the field can be expressed as the gradient of a scalar potential:

$$
\mathbf{B} = \nabla \Phi,
$$

and the potential satisfies **Laplace’s equation**:

$$
\nabla^2 \Phi = 0.
$$

Inside a **toroidal region**, $$\Phi$$ is *multi-valued* — it can jump by a constant around each non-contractible loop (toroidal and poloidal directions). These discontinuities correspond to **harmonic circulations** of $$\mathbf{B}$$, represented by uniform densities on “cut” surfaces that span the holes of the torus.

In LX, we model this as:

$$
\Phi(\mathbf{x}) = \int_{\Gamma} G(\mathbf{x},\mathbf{y}) \, \sigma(\mathbf{y}) \, dS_{\mathbf{y}} +
\int_{S_\text{tor}} \frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y}) \, \lambda_\text{tor} \, dS_{\mathbf{y}} +
\int_{S_\text{pol}} \frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y}) \, \lambda_\text{pol} \, dS_{\mathbf{y}},
$$

where:

- $$G(\mathbf{x},\mathbf{y}) = \frac{1}{4\pi|\mathbf{x}-\mathbf{y}|}$$ is the Laplace Green’s function,

- $$\sigma(\mathbf{y})$$ is the single-layer source density on the boundary $$\Gamma$$,

- $$\lambda_\text{tor}, \lambda_\text{pol}$$ are uniform strengths of the double-layer potentials over toroidal and poloidal cuts.

Enforcing the **Neumann condition** $$\mathbf{n}\cdot\nabla \Phi = 0$$ on $$\Gamma$$ leads to a dense boundary integral system:

$$
A(p)\,u = b(p),
$$

with $$u = [\sigma; \lambda_\text{tor}; \lambda_\text{pol}]$$ and geometry parameters $$p$$.

---

## Geometry Representation

The surface $$\Gamma$$ is parameterized as a **tubular surface** built around a 3D **centerline** $$\mathbf{r}_0(s)$$:

$$
\mathbf{r}(s,\alpha) = \mathbf{r}_0(s) + a(s,\alpha)\, [\cos(\alpha)\mathbf{e}_1(s) + \sin(\alpha)\mathbf{e}_2(s)],
$$

where:

- $$s \in [0,1)$$ is the arc-length parameter along the axis (periodic),

- $$\alpha \in [0,2\pi)$$ is the poloidal angle,

- $$\mathbf{e}_1, \mathbf{e}_2$$ form a **Bishop (parallel-transport) frame** [1],

- $$a(s,\alpha)$$

is a cross-section radius function:

  $$
  a(s,\alpha) = a_0(s)\left[ 1 + \sum_m \left( e_c^{(m)}(s)\cos[m(\alpha-\alpha_0(s))] + e_s^{(m)}(s)\sin[m(\alpha-\alpha_0(s))] \right) \right].
  $$

All quantities ($$a_0$$, $$\alpha_0$$, $$e_c^{(m)}$$, $$e_s^{(m)}$$) are smooth B-splines in $$s$$ and differentiable in JAX.

The design vector is:

$$
p = \{ \text{centerline control points}, \text{twist}, a_0(s), e_c^{(m)}(s), e_s^{(m)}(s), \text{scales} \}.
$$

---

## Numerical Method

LX uses a **Nyström discretization** of the boundary integral equation:
- Discretize $$\Gamma$$ at $$S\times A$$ quadrature points.
- Evaluate kernels via vectorized JAX `vmap` operations.
- Assemble dense matrices $$A(p)$$ and right-hand side $$b(p)$$.
- Solve $$A u = b$$ via least-squares (`jax.numpy.linalg.lstsq`).

Once $$u$$ is known, the field is evaluated as:

$$
\mathbf{B}(\mathbf{x}) = \int_\Gamma \nabla_x G(\mathbf{x},\mathbf{y})\,\sigma(\mathbf{y})\,dS_y +
\lambda_\text{tor}\!\!\int_{S_\text{tor}}\!\!\nabla_x \frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})\,dS_y +
\lambda_\text{pol}\!\!\int_{S_\text{pol}}\!\!\nabla_x \frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})\,dS_y.
$$

### Performance & Differentiability

- **Autodiff**: Every step (geometry construction, kernel evaluation, matrix assembly) is written in pure JAX for exact gradients.
- **JIT**: Functions are `@jit` compiled for speed.
- **Vectorization**: Kernels are computed with nested `vmap` for optimal GPU/TPU parallelism.
- **Adjoint Differentiation**: The solver uses `custom_vjp` to differentiate the implicit linear solve.

Future versions may swap the dense matvec with an **FMM (Fast Multipole Method)** [2] while preserving the same adjoint.

---

## Example Usage

Run the script directly:
```bash
python laplace_solver.py
```

<!-- ## Typical output:

Objective: 3.21e-06
||∂L/∂ctrl||        : 1.45e-03
||∂L/∂twist_s||     : 2.31e-04
||∂L/∂a0_s||        : 6.52e-05
||∂L/∂ec_s||        : 4.87e-05
||∂L/∂es_s||        : 5.12e-05
||∂L/∂alpha0_s||    : 9.10e-05 -->


The objective flattens $$|\mathbf{B}|$$ along a straight segment of the axis, with small regularizers enforcing smoothness and field energy control.

What to Change
Parameter	Meaning	Typical Range
S_SAMPLES, A_SAMPLES	Surface resolution	40–160
N_CTRL	B-spline control points for axis	8–20
M_LIST	Cross-section Fourier modes	(2,), (2,3,)
CUT_NR, CUT_NA	Cut resolution (circulation accuracy)	16–64
SEGMENT_FRACTION	Portion of the axis for the objective	0.1–0.5
REG_*	Regularization weights	1e-3–1e-6

You can modify:

Centerline → change make_design() to start from a straight line or figure-eight.

Cross-section → add Fourier modes for elongation or triangularity.

Cuts → adjust POLOIDAL_ALPHA0 for ribbon orientation.

Objective → define your own function of B, e.g. for quasi-symmetry or mirror shaping.

Improving the Code

Calderón Preconditioning
Convert the first-kind integral equation to a second-kind form [3] to improve conditioning.

Fast Multipole Acceleration
Replace dense integrals with a JAX-compatible FMM [4].

Higher-Order Quadrature
Use product Gauss–Legendre quadrature in $$(s,\alpha)$$ instead of uniform grids for better accuracy.

Additional Harmonic Modes
Replace scalar $$\lambda_\text{tor}$$ and $$\lambda_\text{pol}$$ with a small basis on each cut to represent richer harmonic fields.

Zernike Basis for Inner Surfaces
To represent interior flux surfaces smoothly, fit $$r(\rho,\alpha)$$ using Zernike polynomials as in [DESC, Ref. 5].

Optimization
Use jaxopt or optax for differentiable optimization with Adam or L-BFGS-B.

Gotchas & Numerical Tips

The Neumann problem for Laplace’s equation is singular up to an additive constant. Fix it by adding one extra constraint, e.g., $$\int_\Gamma \sigma,dS = 0$$.

When S_SAMPLES or A_SAMPLES is small, the Bishop frame may accumulate twist; reorthonormalize if needed.

Avoid self-intersecting geometries: check a0(s) < R(s) for tubular validity.

If the solver fails to converge, lower the number of modes or regularize the system matrix with small Tikhonov damping.

References

[1] Bishop, R. L. (1975). There is more than one way to frame a curve. Amer. Math. Monthly, 82, 246–251.
[2] Greengard, L., & Rokhlin, V. (1987). A fast algorithm for particle simulations. J. Comput. Phys., 73, 325–348.
[3] Kress, R. (1999). Linear Integral Equations (2nd ed.). Springer.
[4] Gumerov, N. A., & Duraiswami, R. (2004). Fast multipole methods for the Helmholtz equation in three dimensions. Elsevier.
[5] Dudt, D. W., & Landreman, M. (2020). DESC: A stellarator equilibrium solver using direct variational methods. Phys. Plasmas, 27, 102513.
[6] Landreman, M. et al. (2023). Magnetic fields with precise quasisymmetry. J. Plasma Phys., 89, 905890616.
[7] Garren, D. A., & Boozer, A. H. (1991). Existence of quasihelically symmetric stellarators. Phys. Fluids B, 3, 2822–2834.

Ways Forward

LX bridges applied mathematics, stellarator optimization, and computational plasma physics:

Integrate with coil design tools (e.g., SIMSOPT
) for differentiable coil optimization.

Extend to Poisson equations for plasma pressure-driven equilibria.

Couple with GR-based metrics for magnetogenesis or curved spacetime plasma simulations.

Enable multi-objective optimization for mirror shaping, magnetic well control, and quasisymmetry tuning.

Maintainers: UWPlasma Research Group,
Department of Physics, University of Wisconsin–Madison
Contact: uwplasma.github.io