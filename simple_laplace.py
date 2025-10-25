#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mode-by-mode Laplace solver inside a non-axisymmetric torus (fixed BC band).

Key fix vs prior version:
- The boundary-penalty band is placed just INSIDE the interior domain:
  band = { a_min - th ≤ ρ < a_min }, so every BC row touches interior DOFs.
- BC assembled by φ-quadrature to avoid analytic cancellations.

We write φ = φ_sec(φ) + u, with φ_sec = (μ0 I / 2π) φ ⇒ ∇φ_sec = (C/R) e_φ, C=μ0 I/2π.
Expand u(R,φ,Z) = Σ_{m=-M..M} u_m(R,Z) e^{imφ}.
For each m: (1/R)∂R(R∂R u_m) + ∂Z² u_m − (m²/R²) u_m = 0 in ρ < a_min (a_min=a0−a1).

Oblique Neumann on a thin inside band near ρ≈a_min:
  n·∇(φ_sec + u) = 0  ⇒  n_R ∂R u + n_Z ∂Z u + (n_φ/R) ∂_φ u + (C/R) n_φ = 0
We enforce it in least-squares via (L + λ B^H B) u = -λ B^H b.

This is a teaching script; modest grids/modes recommended.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator

# -------------------- User knobs --------------------
mu0 = 4e-7*np.pi
I   = 1.0e7                      # makes C/R visible
R0, a0, a1, Nmode = 1.00, 0.25, 0.10, 4   # a(φ)=a0+a1 cos(N φ)
Mmax = max(2*Nmode, 8)                         # keep modes m=-M..M (must include ±N!)
lam_bc = 1e5                     # λ ~ 1e5..1e6 good on these grids
band_cells = 2                   # band half-thickness (1–3)
report = True

# Grid
NR, Nphi, NZ = 64, 96, 64
R_pad = 1.30*(a0+a1)

R   = np.linspace(R0 - R_pad, R0 + R_pad, NR)
Z   = np.linspace(-R_pad, +R_pad, NZ)
phi = np.linspace(0.0, 2*np.pi, Nphi, endpoint=False)

dR, dZ, dphi = R[1]-R[0], Z[1]-Z[0], phi[1]-phi[0]
RR, ZZ = np.meshgrid(R, Z, indexing="ij")
RR_safe = np.maximum(RR, 1e-9)
rho = np.sqrt((RR - R0)**2 + ZZ**2)

# Interior: unknowns exist only for ρ < a_min (inside for ALL φ)
a_min = max(a0 - a1, 1e-6)
inside = rho < a_min

# **FIX**: put the penalty band just INSIDE the interior boundary
th = band_cells * max(dR, dZ)
band = (rho >= (a_min - th)) & (rho < a_min)

# Secular coefficient
C = mu0 * I / (2*np.pi)

if report:
    print(f"[info] μ0={mu0:.3e} H/m, I={I:.2e} A,  C=μ0I/2π={C:.3e}")
    print(f"[torus] R0={R0:.2f}, a0={a0:.3f}, a1={a1:.3f}, N={Nmode}, a_min={a_min:.3f}")
    print(f"[grid] NR×Nφ×NZ={NR}×{Nphi}×{NZ}; dR={dR:.3e}, dφ={dphi:.3e}, dZ={dZ:.3e}")
    print(f"[modes] Mmax={Mmax} ⇒ #m = {2*Mmax+1}")
    print(f"[mask] inside pts={inside.sum()} / {inside.size} (~{100*inside.mean():.1f}%)")
    print(f"[band] (inside) band cells={band.sum()}, th={th:.3e}, λ={lam_bc:.1e}")

# -------------------- Indexing maps --------------------
idx2d = -np.ones_like(RR, dtype=int)
ik_inside = np.argwhere(inside)
for p,(i,k) in enumerate(ik_inside):
    idx2d[i,k] = p
Nunk2d = len(ik_inside)

m_list  = np.arange(-Mmax, Mmax+1, dtype=int)
Nm      = len(m_list)
m_to_off= {m:t for t,m in enumerate(m_list)}

def gidx(i,k,m):
    p = idx2d[i,k]
    return -1 if p < 0 else p + Nunk2d * m_to_off[m]

Nunk = Nunk2d * Nm
if report:
    print(f"[system] unknowns per-mode N2D={Nunk2d}, total N={Nunk} complex")

# -------------------- Interior operator (block-diag over modes) --------------
# L_m u = (1/R)∂R(R∂R u) + ∂Z² u − (m²/R²) u
A = lil_matrix((Nunk, Nunk), dtype=np.complex128)
for m in m_list:
    off = Nunk2d * m_to_off[m]
    for p,(i,k) in enumerate(ik_inside):
        Ri = R[i]
        diag = 0.0
        # R faces
        if i-1 >= 0 and inside[i-1,k]:
            Rm = 0.5*(R[i]+R[i-1]); w = Rm/(Ri*dR*dR)
            A[off+p, off+idx2d[i-1,k]] += +w; diag -= w
        if i+1 < NR and inside[i+1,k]:
            Rp = 0.5*(R[i]+R[i+1]); w = Rp/(Ri*dR*dR)
            A[off+p, off+idx2d[i+1,k]] += +w; diag -= w
        # Z faces
        if k-1 >= 0 and inside[i,k-1]:
            wz = 1.0/(dZ*dZ)
            A[off+p, off+idx2d[i,k-1]] += +wz; diag -= wz
        if k+1 < NZ and inside[i,k+1]:
            wz = 1.0/(dZ*dZ)
            A[off+p, off+idx2d[i,k+1]] += +wz; diag -= wz
        # φ term
        diag += (m*m)/(Ri*Ri)
        A[off+p, off+p] += -diag

# -------------------- Boundary coupling (B): analytic Fourier in φ --------------------
# nφ(φ) = -a1*N*sin(Nφ)  ⇒  nφ_hat[+N] = a1*N/(2i),  nφ_hat[-N] = -a1*N/(2i), others 0.
nphi_hat = { +Nmode:  (a1*Nmode)/(2j),   # 1/(2i) = -i/2 ; SciPy uses 1j for i
             -Nmode: -(a1*Nmode)/(2j) }

rows, cols, vals = [], [], []
b_vals = []
ik_band = np.argwhere(band)
Nb2d = len(ik_band)
row_id = 0

for (i,k) in ik_band:
    Ri   = R[i]
    rhoi = max(rho[i,k], 1e-12)
    nR0  = (R[i]-R0)/rhoi
    nZ0  = Z[k]/rhoi

    # conditioning weight (same as before)
    w_row = np.sqrt(max(Ri,1e-9) * dphi * th)

    for m in m_list:
        # --- φ-independent pieces (n_R ∂R u + n_Z ∂Z u) project only to m=0
        if m == 0:
            if i+1 < NR and inside[i+1,k]:
                g = gidx(i+1,k,m)
                if g >= 0: rows += [row_id]; cols += [g]; vals += [w_row * nR0/(2*dR)]
            if i-1 >= 0 and inside[i-1,k]:
                g = gidx(i-1,k,m)
                if g >= 0: rows += [row_id]; cols += [g]; vals += [-w_row * nR0/(2*dR)]
            if k+1 < NZ and inside[i,k+1]:
                g = gidx(i,k+1,m)
                if g >= 0: rows += [row_id]; cols += [g]; vals += [w_row * nZ0/(2*dZ)]
            if k-1 >= 0 and inside[i,k-1]:
                g = gidx(i,k-1,m)
                if g >= 0: rows += [row_id]; cols += [g]; vals += [-w_row * nZ0/(2*dZ)]

        # --- φ-dependent part: (nφ/R) ∂φ u  ⇒ convolution in m with nφ_hat
        # Only ±N are nonzero: so m receives contributions from q = m±N
        for s in (+Nmode, -Nmode):
            if (m - s) in m_to_off:
                q   = m - s
                coef = (1j * q) * (nphi_hat[s] / max(Ri,1e-9))   # (i q) * nφ_hat[s] / R
                g = gidx(i,k,q)
                if g >= 0:
                    rows += [row_id]; cols += [g]; vals += [w_row * coef]

        # --- RHS: (C/R) nφ ⇒ only m=±N
        rhs_val = 0.0+0.0j
        if m in nphi_hat:
            rhs_val = w_row * (C/max(Ri,1e-9)) * nphi_hat[m]

        b_vals.append(rhs_val)
        row_id += 1

Nb = row_id
print(f"[BC] (analytic φ-coupling in band) rows Nb={Nb} (= {Nb2d} band pts × {Nm} modes)")

rows_np = np.asarray(rows, dtype=int)
cols_np = np.asarray(cols, dtype=int)
vals_np = np.asarray(vals, dtype=np.complex128)
if not (len(rows_np) == len(cols_np) == len(vals_np)):
    raise RuntimeError(f"COO mismatch: len(rows)={len(rows_np)}, len(cols)={len(cols_np)}, len(vals)={len(vals_np)}")

from scipy.sparse import coo_matrix
B = coo_matrix((vals_np, (rows_np, cols_np)), shape=(Nb, Nunk), dtype=np.complex128).tocsr()
assert cols_np.min() >= 0 and cols_np.max() < Nunk, "Column index out of range"
b = np.array(b_vals, dtype=np.complex128)


# -------------------- Solve (L + λ B^H B) u = - λ B^H b --------------------
A = A.tocsr() + lam_bc * (B.conj().T @ B)
rhs = - lam_bc * (B.conj().T @ b)

Bh_b = (B.conj().T @ b)
print(f"[debug] ||B^H b||_2 = {np.linalg.norm(Bh_b):.3e}")

# Neumann gauge pin
pin_g = gidx(ik_inside[0][0], ik_inside[0][1], 0)
A[pin_g, pin_g] += 1e-12

if report: print("[solve] spsolve complex system ...")
u_vec = spsolve(A, rhs)

# Penalty residual in solver space
r = B @ u_vec + b
r_rms = np.sqrt((r.conj() @ r).real / max(1, r.size))
b_rms = np.sqrt((b.conj() @ b).real / max(1, b.size))
print(f"[penalty] RMS ||Bu+b|| = {r_rms:.3e}  (vs RHS RMS ||b|| = {b_rms:.3e})")

# -------------------- Unpack per-mode fields --------------------
u_modes = {m: np.full_like(RR, np.nan, dtype=np.complex128) for m in m_list}
for m in m_list:
    off = Nunk2d * m_to_off[m]
    um = np.zeros_like(RR, dtype=np.complex128)
    for p,(i,k) in enumerate(ik_inside):
        um[i,k] = u_vec[off+p]
    u_modes[m] = um

# -------------------- Reconstruct 3D field components --------------------
def dR_centered_2d(A2):
    Ap = np.roll(A2, -1, axis=0); Am = np.roll(A2, +1, axis=0)
    Ap[~inside] = A2[~inside]; Am[~inside] = A2[~inside]
    return (Ap - Am)/(2*dR)

def dZ_centered_2d(A2):
    Ap = np.roll(A2, -1, axis=1); Am = np.roll(A2, +1, axis=1)
    Ap[~inside] = A2[~inside]; Am[~inside] = A2[~inside]
    return (Ap - Am)/(2*dZ)

RR3, PP3, ZZ3 = np.meshgrid(R, phi, Z, indexing="ij")
RR3_safe = np.maximum(RR3, 1e-9)

u3   = np.zeros((NR, Nphi, NZ), dtype=np.complex128)
uR3  = np.zeros_like(u3)
uZ3  = np.zeros_like(u3)
uPH3 = np.zeros_like(u3)

for m in m_list:
    phase = np.exp(1j * m * PP3)
    um = u_modes[m]
    um_R = dR_centered_2d(um)
    um_Z = dZ_centered_2d(um)
    u3   += um[:,None,:] * phase
    uR3  += um_R[:,None,:] * phase
    uZ3  += um_Z[:,None,:] * phase
    uPH3 += (1j*m) * um[:,None,:] * phase

# Total field
B_R  = np.real(uR3)
B_Z  = np.real(uZ3)
B_PH = np.real(uPH3 / RR3_safe) + (C / RR3_safe)
grad_mag = np.sqrt(B_R**2 + B_Z**2 + B_PH**2)

# -------------------- Diagnostics --------------------
band3 = np.broadcast_to(band[:,None,:], grad_mag.shape)
nR0_2D = (RR - R0)/np.maximum(rho,1e-12)
nZ0_2D = (ZZ)/np.maximum(rho,1e-12)
a_phi_3D = a0 + a1*np.cos(Nmode*PP3)
nphi_3D = -(a1 * Nmode * np.sin(Nmode*PP3)) * (a_min / np.maximum(a_phi_3D, 1e-9))

res3 = (nR0_2D[:,None,:]*uR3 + nZ0_2D[:,None,:]*uZ3
        + (nphi_3D/RR3_safe)*uPH3 + (C/RR3_safe)*nphi_3D)
bc_L2 = np.sqrt(np.nanmean(np.abs(res3[band3])**2))
rhs_band = (C/np.maximum(RR3_safe,1e-12)) * nphi_3D
rhs_rms = np.sqrt(np.nanmean(np.abs(rhs_band[band3])**2))
rel_bc = bc_L2 / (rhs_rms if rhs_rms>0 else 1.0)

inside3 = np.broadcast_to(inside[:,None,:], grad_mag.shape)
mean_grad_u = np.nanmean(np.sqrt((np.abs(uR3)**2 + np.abs(uZ3)**2 + np.abs(uPH3/RR3_safe)**2)[inside3]))

print(f"[check] BC L2 on band ≈ {bc_L2:.3e} (relative {rel_bc:.3f} of forcing RMS {rhs_rms:.3e})")
print(f"[check] ⟨|∇u|⟩ inside ≈ {mean_grad_u:.3e}")

# -------------------- Surface sampling & plots --------------------
def surface_param(phi_s, theta_s, shrink=0.5*max(dR,dZ)):
    a_s = a0 + a1*np.cos(Nmode*phi_s) - shrink
    a_s = np.maximum(a_s, 1e-6)
    R_s = R0 + a_s*np.cos(theta_s)
    Z_s = a_s*np.sin(theta_s)
    return R_s, Z_s

Nsurf_phi, Nsurf_th = 240, 120
phi_s = np.linspace(0, 2*np.pi, Nsurf_phi, endpoint=False)
th_s  = np.linspace(0, 2*np.pi, Nsurf_th,  endpoint=False)
PHI_S, THETA_S = np.meshgrid(phi_s, th_s, indexing='ij')
R_S, Z_S = surface_param(PHI_S, THETA_S)
X_S, Y_S = R_S*np.cos(PHI_S), R_S*np.sin(PHI_S)

interp_BR  = RegularGridInterpolator((R, phi, Z), B_R,  bounds_error=False, fill_value=np.nan)
interp_BPH = RegularGridInterpolator((R, phi, Z), B_PH, bounds_error=False, fill_value=np.nan)
interp_BZ  = RegularGridInterpolator((R, phi, Z), B_Z,  bounds_error=False, fill_value=np.nan)

pts = np.stack([R_S.ravel(), PHI_S.ravel(), Z_S.ravel()], axis=1)
BR_S  = interp_BR(pts).reshape(R_S.shape)
BPH_S = interp_BPH(pts).reshape(R_S.shape)
BZ_S  = interp_BZ(pts).reshape(R_S.shape)
gradS_mag = np.sqrt(BR_S**2 + BPH_S**2 + BZ_S**2)

valid = np.isfinite(gradS_mag)
print(f"[plot] valid surface samples: {int(valid.sum())} / {valid.size}")

# Plot 1: |∇φ| on surface
fig = plt.figure(figsize=(7.6,5.6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_S[valid], Y_S[valid], Z_S[valid], c=gradS_mag[valid], s=8)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('|∇φ| on non-axisymmetric torus surface (mode-by-mode)')
cb = fig.colorbar(sc, shrink=0.8); cb.set_label('|∇φ|')

# Plot 2: ⟨|∇φ|⟩ vs R
R_bins = np.linspace(R0 - (a0+a1), R0 + (a0+a1), 50)
R_mid  = 0.5*(R_bins[:-1] + R_bins[1:])
vals = gradS_mag[valid].ravel()
Rvals = R_S[valid].ravel()
bin_avg = np.full_like(R_mid, np.nan, dtype=float)
for i in range(len(R_mid)):
    msk = (Rvals >= R_bins[i]) & (Rvals < R_bins[i+1])
    if np.any(msk): bin_avg[i] = np.nanmean(vals[msk])

plt.figure(figsize=(6.4,4.8))
plt.plot(R_mid, bin_avg, 'o', label='Surface ⟨|∇φ|⟩')
plt.plot(R_mid, C/np.maximum(R_mid, 1e-6), '-', label='C/R reference')
plt.xlabel('R'); plt.ylabel('⟨|∇φ|⟩ on surface'); plt.legend()
plt.title('~1/R (secular) + non-axisymmetric corrections (m ↔ all q via φ quadrature)')
plt.tight_layout(); plt.show()
