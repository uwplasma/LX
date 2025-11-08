#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import jit, jacrev, jacfwd, value_and_grad, vmap
from flax import linen as nn
import optax
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# ------------------------ Precision toggle ------------------------
USE_X64 = False   # set True if you really need 64-bit; False is faster
from jax import config
config.update("jax_enable_x64", bool(USE_X64))

# ------------------------ Geometry & constants ------------------------
mu0 = 4e-7*jnp.pi
I   = 1.0e7
C   = mu0*I/(2*jnp.pi)

R0, a0, a1, Nmode = 1.00, 0.25, 0.10, 4

def a_of_phi(phi):
    return a0 + a1*jnp.cos(Nmode*phi)

def levelset_F(R, phi, Z):
    rho = jnp.sqrt((R - R0)**2 + Z**2)
    return rho - a_of_phi(phi)

# Cylindrical-normalized normal ~ (∂F/∂R, (1/R)∂F/∂φ, ∂F/∂Z)
def n_hat(R, phi, Z):
    rho = jnp.sqrt((R - R0)**2 + Z**2) + 1e-12
    dFR = (R - R0)/rho
    dFphi_overR = (a1*Nmode*jnp.sin(Nmode*phi)) / jnp.maximum(R, 1e-9)
    dFZ = Z/rho
    g = jnp.stack([dFR, dFphi_overR, dFZ], axis=-1)
    g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
    return g

# ------------------------ Sampling ------------------------
@partial(jit, static_argnames=("N",))
def sample_interior(key, N):
    # Uniform in (φ,θ), radius r ~ U(0, a(φ)) (not exact volume; OK for PINN)
    k1,k2,k3 = jax.random.split(key, 3)
    phi = jax.random.uniform(k1, (N,), minval=0.0, maxval=2*jnp.pi)
    th  = jax.random.uniform(k2, (N,), minval=0.0, maxval=2*jnp.pi)
    r   = jax.random.uniform(k3, (N,)) * a_of_phi(phi)
    R   = R0 + r*jnp.cos(th)
    Z   = r*jnp.sin(th)
    return R, phi, Z

@partial(jit, static_argnames=("N",))
def sample_surface(key, N):
    k1,k2 = jax.random.split(key, 2)
    phi = jax.random.uniform(k1, (N,), minval=0.0, maxval=2*jnp.pi)
    th  = jax.random.uniform(k2, (N,), minval=0.0, maxval=2*jnp.pi)
    r   = a_of_phi(phi)
    R   = R0 + r*jnp.cos(th)
    Z   = r*jnp.sin(th)
    return R, phi, Z

# ------------------------ Model (batched) ------------------------
class MLP(nn.Module):
    width: int = 64
    depth: int = 4
    @nn.compact
    def __call__(self, X):  # X: (...,4) with [R, cosφ, sinφ, Z]
        h = X
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width)(h))
        return nn.Dense(1)(h)[..., 0]

model = MLP(64, 4)

def u_apply(params, R, phi, Z):
    X = jnp.stack([R, jnp.cos(phi), jnp.sin(phi), Z], axis=-1)  # (N,4)
    return model.apply(params, X)  # (N,)

# ------------------------ Differential operators (batch, fast) ------------------------
# Use chain rule for φ-derivatives via (c,s)=(cosφ, sinφ).
# ∂u/∂φ = [∂u/∂c, ∂u/∂s]·[-sinφ, cosφ]
# ∂²u/∂φ² = vᵀ H_cs v + grad_cs·d v/dφ, with v=[-s, c] and d v/dφ=[-c, -s]

@jit
def _grads_hess_X(params, R, phi, Z):
    # Build inputs once
    X = jnp.stack([R, jnp.cos(phi), jnp.sin(phi), Z], axis=-1)  # (N,4)
    # Per-row scalar model: Xi has shape (4,)
    def f_row(Xi):
        # model.apply accepts (...,4); for (4,) we force a length-1 batch
        return model.apply(params, Xi[None, :])[0]
    # Per-row gradient (shape: (4,)) and Hessian (shape: (4,4))
    g_row   = jacrev(f_row)
    H_row   = jacfwd(jacrev(f_row))
    g = vmap(g_row)(X)      # (N,4)
    H = vmap(H_row)(X)      # (N,4,4)
    return X, g, H

@jit
def laplacian_cyl(params, R, phi, Z):
    X, g, H = _grads_hess_X(params, R, phi, Z)
    Rpos = jnp.maximum(X[:,0], 1e-9)
    c, s = X[:,1], X[:,2]

    # grads
    u_R, u_c, u_s, u_Z = g[:,0], g[:,1], g[:,2], g[:,3]
    # second diag terms
    u_RR = H[:,0,0]
    u_ZZ = H[:,3,3]

    # φ-terms via (c,s) block
    v = jnp.stack([-s, c], axis=-1)             # (N,2)
    Hcs = H[:,1:3,1:3]                           # (N,2,2)
    # v^T H v
    vHv = jnp.einsum('ni,nij,nj->n', v, Hcs, v)
    u_P  = u_c*(-s) + u_s*(c)
    u_PP = vHv - (u_c*c + u_s*s)

    return u_RR + (u_R/Rpos) + (u_PP/(Rpos**2)) + u_ZZ

@jit
def bc_residual(params, R, phi, Z):
    X, g, _ = _grads_hess_X(params, R, phi, Z)
    Rpos = jnp.maximum(X[:,0], 1e-9)
    c, s = X[:,1], X[:,2]
    u_R, u_c, u_s, u_Z = g[:,0], g[:,1], g[:,2], g[:,3]
    u_P = u_c*(-s) + u_s*(c)
    nR, nP, nZ = jnp.moveaxis(n_hat(R, phi, Z), -1, 0)
    # n·∇(φ_sec+u) = nR*u_R + nP*(u_P/R + C/R) + nZ*u_Z
    return nR*u_R + nP*(u_P/Rpos + C/Rpos) + nZ*u_Z

# ------------------------ Loss & training ------------------------
@partial(jit, static_argnames=("Nin","Nbc"))
def loss_fn(params, key, Nin=2048, Nbc=2048, lam_bc=2e3):
    kin, kbc = jax.random.split(key)
    Rin, Pin, Zin = sample_interior(kin, Nin)
    Rb,  Pb,  Zb  = sample_surface(kbc, Nbc)
    lap = laplacian_cyl(params, Rin, Pin, Zin)
    bc  = bc_residual(params, Rb, Pb, Zb)
    return jnp.mean(lap**2) + lam_bc*jnp.mean(bc**2)

def make_step(tx):
    @partial(jit, static_argnames=("Nin","Nbc"))
    def step(params, opt_state, key, Nin, Nbc, lam_bc):
        l, grads = value_and_grad(loss_fn)(params, key, Nin, Nbc, lam_bc)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l
    return step

def train(rng=0, steps=2000, lr=1e-3, Nin=2048, Nbc=2048, lam_bc=2e3,
          width=64, depth=4, log_every=250):
    global model
    model = MLP(width=width, depth=depth)

    key = jax.random.PRNGKey(rng)
    # initialize with a small dummy batch so shapes are fixed & batched
    params = model.init(key, jnp.ones((2,4)))
    tx = optax.adamax(lr)
    opt_state = tx.init(params)
    step = make_step(tx)

    # warmup JIT
    key, kw = jax.random.split(key)
    params, opt_state, _ = step(params, opt_state, kw, 64, 64, lam_bc)

    for t in range(1, steps+1):
        key, k = jax.random.split(key)
        params, opt_state, l = step(params, opt_state, k, Nin, Nbc, lam_bc)
        if (t % log_every) == 0:
            print(f"[it {t:4d}] loss={float(l):.3e}")
    return params

# ------------------------ Public API: φ and B ------------------------
@jit
def phi_total(params, R, phi, Z):
    return C*phi + u_apply(params, R, phi, Z)

@jit
def B_components(params, R, phi, Z):
    X, g, _ = _grads_hess_X(params, R, phi, Z)
    Rpos = jnp.maximum(X[:,0], 1e-9)
    c, s = X[:,1], X[:,2]
    u_R, u_c, u_s, u_Z = g[:,0], g[:,1], g[:,2], g[:,3]
    B_R = u_R
    B_P = (u_c*(-s) + u_s*(c))/Rpos + C/Rpos  # add secular analytically
    B_Z = u_Z
    return B_R, B_P, B_Z

# ------------------------ Plots & checks ------------------------
def plot_surface_fields(params, Nsurf=20000, seed=123):
    key = jax.random.PRNGKey(seed)
    Rs, Ps, Zs = sample_surface(key, Nsurf)
    BR, BP, BZ = B_components(params, Rs, Ps, Zs)
    Bmag = jnp.sqrt(BR**2 + BP**2 + BZ**2)

    X = Rs*jnp.cos(Ps); Y = Rs*jnp.sin(Ps)
    fig = plt.figure(figsize=(7.2,5.6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(np.array(X), np.array(Y), np.array(Zs),
                    c=np.array(Bmag), s=4)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('|∇φ| on non-axisymmetric torus (PINN)')
    cb = fig.colorbar(sc, shrink=0.8); cb.set_label('|∇φ|')

    # ⟨|∇φ|⟩ vs R
    Rvals = np.array(Rs); Bvals = np.array(Bmag)
    nb = 60
    bins = np.linspace(float(R0-(a0+a1)), float(R0+(a0+a1)), nb+1)
    mids = 0.5*(bins[:-1]+bins[1:])
    avg = np.full_like(mids, np.nan, dtype=float)
    for i in range(nb):
        m = (Rvals>=bins[i]) & (Rvals<bins[i+1])
        if m.any(): avg[i] = Bvals[m].mean()
    plt.figure(figsize=(6.0,4.2))
    plt.plot(mids, avg, 'o', ms=3, label='Surface ⟨|∇φ|⟩')
    plt.plot(mids, float(C)/np.maximum(mids, 1e-9), '-', label='C/R reference')
    plt.xlabel('R'); plt.ylabel('⟨|∇φ|⟩ on surface'); plt.legend()
    plt.tight_layout(); plt.show()

def report_bc(params, Ns=8192, seed=321):
    key = jax.random.PRNGKey(seed)
    Rs, Ps, Zs = sample_surface(key, Ns)
    bc = bc_residual(params, Rs, Ps, Zs)
    rms = float(jnp.sqrt(jnp.mean(bc**2)))
    return rms

# ------------------------ Example run ------------------------
if __name__ == "__main__":
    params = train(rng=0, steps=2000, lr=1e-3,
                   Nin=2048, Nbc=2048, lam_bc=2e3,
                   width=64, depth=4, log_every=250)
    print(f"[check] BC RMS on surface ≈ {report_bc(params):.3e}")
    plot_surface_fields(params)
