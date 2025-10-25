import numpy as np
from typing import Dict, Tuple, Any
from helpers import theta_to_surface
import matplotlib.pyplot as plt

def clenshaw_curtis_weights(N: int) -> np.ndarray:
    """
    Clenshaw–Curtis quadrature weights for Chebyshev–Lobatto nodes on [-1,1].
    Returns w (N,), exact for polynomials up to degree N-1.
    """
    if N == 1:
        return np.array([2.0])
    k = np.arange(0, N)
    w = np.zeros(N, dtype=float)
    w[0]  = 1.0
    w[-1] = 1.0
    for j in range(1, N-1):
        s = 0.0
        # closed-form series for CC weights
        for m in range(1, (N-1)//2 + 1):
            s += (2.0 / (4*m*m - 1.0)) * np.cos(2*m*np.pi*j/(N-1))
        if (N-1) % 2 == 0:
            m = (N-1)//2
            s += (1.0 / (4*m*m - 1.0)) * np.cos(m*np.pi*j/m)
        w[j] = 2.0*(1.0 - 2.0*s)/(N-1)
    return w


# ---------- Weighted algebra (weight = sqrtg) ----------
def vol_mean_w(u: np.ndarray, w: np.ndarray) -> float:
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    num = float((u * w).sum())
    den = float(w.sum() + 1e-300)
    return num / den

def proj0_w(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    return u - vol_mean_w(u, w)

def dot_w(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    val = float((u * v * w).sum())
    # guard tiny negatives from roundoff
    if val < 0.0 and abs(val) < 1e-30:
        val = 0.0
    return val

def norm_w(u: np.ndarray, w: np.ndarray) -> float:
    return np.sqrt(max(dot_w(u, u, w), 0.0) + 1e-300)


def print_operator_residuals(phi_tilde: np.ndarray, rhs: np.ndarray, geom: Dict[str,np.ndarray], tag: str="post"):
    res = apply_laplacian(phi_tilde, geom) - rhs
    w = geom["wvol"]
    n2  = norm_w(res, w)
    n_rhs = norm_w(rhs, w)
    print(f"[{tag}] ||L φ̃ - rhs||_w = {n2:.3e}   (||rhs||_w = {n_rhs:.3e}, rel = {n2/(n_rhs+1e-300):.3e})")


def check_neumann_bc(phi: np.ndarray, geom: Dict[str,np.ndarray], sample_frac: float=0.25):
    """
    Report the enforced BC n·(√g G^{·j} ∂_j φ) = 0 at r=1 (and r=0) by measuring the
    *radial flux* F_r = √g G^{rj} ∂_j φ on the boundary.
    """
    Dx_r = geom["Dx_r_bc"]
    sqrtg, Ginv = geom["sqrtg"], geom["Ginv"]
    Nr, Ns, Na = phi.shape
    M = Ns*Na

    # derivatives
    phi_rs  = phi.reshape(Nr, M)
    dphi_dr = (Dx_r @ phi_rs).reshape(Nr, Ns, Na)
    dphi_ds = d_periodic_fft_dealiased(phi, axis=1, L=1.0)
    dphi_dA = d_periodic_fft_dealiased(phi, axis=2, L=2.0*np.pi)

    # radial flux F_r = √g ( G^{r0} d_r + G^{r1} d_s + G^{r2} d_A )
    Fr = ( sqrtg * ( Ginv[...,0,0]*dphi_dr
                   + Ginv[...,0,1]*dphi_ds
                   + Ginv[...,0,2]*dphi_dA ) )

    # stats on r=1 (and r=0)
    for ridx, name in [(Nr-1,"r=1"), (0,"r=0")]:
        F = Fr[ridx]
        step_s = max(1, int(1.0/(sample_frac) * Ns//32))
        step_a = max(1, int(1.0/(sample_frac) * Na//32))
        samp = F[::step_s, ::step_a].ravel()
        print(f"[bc] {name} radial flux (mean/absmax) = {np.mean(samp):.3e} / {np.max(np.abs(samp)):.3e}")

def check_energy_symmetry(geom: Dict[str,np.ndarray], trials: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    Nr,Ns,Na = geom["X"].shape[:3]
    w = geom["wvol"]
    def P(u): return proj0_w(u, w)
    errs = []
    for t in range(trials):
        u = P(rng.standard_normal((Nr,Ns,Na)))
        v = P(rng.standard_normal((Nr,Ns,Na)))
        Lu = apply_laplacian(u, geom)
        Lv = apply_laplacian(v, geom)
        a = dot_w(u, Lv, w)
        b = dot_w(v, Lu, w)
        errs.append(abs(a-b)/(abs(a)+abs(b)+1e-300))
    print(f"[sym] <u,Lv>_w vs <v,Lu>_w rel errors: {errs}  (median={np.median(errs):.3e})")

# ----------------- FFT: periodic derivatives + 2/3 dealiasing -----------------

def _fft_wavenumbers(N: int, L: float) -> np.ndarray:
    """Angular wavenumbers (rad/unit) for length-L periodic domain sampled at N points."""
    return 2.0*np.pi * np.fft.fftfreq(N, d=L/N)

def _twothirds_mask(N: int) -> np.ndarray:
    """Boolean mask that keeps the lowest 2/3 of modes (centered), zeros the rest."""
    k = np.fft.fftfreq(N) * N  # integer modes in [-N/2, ..., N/2-1]
    cutoff = (2.0/3.0) * (N/2.0)
    return np.abs(k) <= cutoff

def d_periodic_fft_dealiased(f: np.ndarray, axis: int, L: float) -> np.ndarray:
    """
    Pseudo-spectral derivative ∂f/∂x with 2/3-rule de-aliasing
    along a periodic axis of period L.
    """
    N = f.shape[axis]
    k = _fft_wavenumbers(N, L)                      # (N,)
    F = np.fft.fft(f, axis=axis)
    # 2/3 de-aliasing mask -> broadcast to f's shape
    mask = _twothirds_mask(N).astype(float)         # (N,)
    bshape = [1]*f.ndim
    bshape[axis] = N
    F = F * mask.reshape(bshape)
    # derivative in Fourier space
    Fdiff = (1j * k.reshape(bshape)) * F
    return np.fft.ifft(Fdiff, axis=axis).real

def filter_periodic_dealias(f: np.ndarray, axis: int) -> np.ndarray:
    """
    Project to lowest 2/3 modes on a periodic axis (2/3-rule).
    """
    N = f.shape[axis]
    F = np.fft.fft(f, axis=axis)
    mask = _twothirds_mask(N).astype(float)         # (N,)
    bshape = [1]*f.ndim
    bshape[axis] = N
    F = F * mask.reshape(bshape)
    return np.fft.ifft(F, axis=axis).real

# ------------------------- Chebyshev collocation in r -------------------------

def cheb_grid_and_D(Nr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chebyshev-Lobatto collocation points x∈[-1,1] and first-derivative matrix D (Nr x Nr).
    """
    if Nr < 2:
        raise ValueError("Nr >= 2 required for Chebyshev grid.")
    k = np.arange(Nr)
    x = np.cos(np.pi * k / (Nr - 1))      # x_0=1, x_{Nr-1}=-1
    c = np.ones(Nr); c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1.0)**k)
    X = np.tile(x, (Nr,1))
    dX = X - X.T + np.eye(Nr)
    D = (c[:,None] / c[None,:]) / dX
    D = D - np.diag(np.sum(D, axis=1))
    return x, D  # x in [-1,1]

def map_unit_interval(xc: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Map Chebyshev x∈[-1,1] to r∈[0,1]. Return r, scaling factors dr/dx and dx/dr at interior.
    For differentiation: ∂/∂r = (dx/dr) ∂/∂x, with dx/dr = 2.
    """
    r = 0.5*(1.0 - xc)     # x=1 -> r=0,  x=-1 -> r=1
    drdx = -0.5 * np.ones_like(xc)
    dxdr = -2.0 * np.ones_like(xc)
    return r, drdx, dxdr

def d_periodic_fft(f: np.ndarray, axis: int, L: float) -> np.ndarray:
    """
    Spectral derivative ∂f/∂x along a periodic axis with period L.
    (No de-aliasing; used only if you intentionally want all modes.)
    """
    N = f.shape[axis]
    k = _fft_wavenumbers(N, L)
    F = np.fft.fft(f, axis=axis)
    bshape = [1]*f.ndim
    bshape[axis] = N
    Fdiff = (1j * k.reshape(bshape)) * F
    return np.fft.ifft(Fdiff, axis=axis).real

def div_periodic_fft(F: np.ndarray, axis: int, L: float) -> np.ndarray:
    """
    Spectral divergence component: ∂F/∂x along a periodic axis with length L.
    Just an alias of d_periodic_fft for readability.
    """
    return d_periodic_fft(F, axis=axis, L=L)

def diff_periodic(f: np.ndarray, axis: int, h: float) -> np.ndarray:
    """Centered 2nd-order derivative along a periodic axis."""
    return (np.roll(f, -1, axis=axis) - np.roll(f, +1, axis=axis)) / (2*h)

def assemble_tube_grid(theta: np.ndarray, meta: Dict[str,Any], Nr: int, Ns: int, Na: int) -> Dict[str, np.ndarray]:
    """
    Build tube grid: r∈[0,1] (Chebyshev), s periodic, α periodic; compute mapping X and geometric bases.
    Uses spectral derivatives in s, α (with 2/3 de-aliasing) and Chebyshev in r.
    """
    data_surf = theta_to_surface(theta, meta, s_samples=Ns, alpha_samples=Na)
    s = data_surf["s"]                           # (Ns,) in [0,1)
    alpha = data_surf["alpha"]                   # (Na,) in [0,2π)
    r0 = data_surf["r0"]                         # (Ns,3)
    e1 = data_surf["e1"]                         # (Ns,3)
    e2 = data_surf["e2"]                         # (Ns,3)
    a_surf = data_surf["a"]                      # (Ns,Na)
    a_surf = np.maximum(a_surf, 1e-4)

    # Chebyshev radial grid in r∈[0,1]
    xc, Dx = cheb_grid_and_D(Nr)                 # x∈[-1,1]
    r, drdx, dxdr = map_unit_interval(xc)        # r∈[0,1]
    # For first derivative wrt r: ∂/∂r = (dx/dr) ∂/∂x with dx/dr = -2
    Dx_r = (dxdr.reshape(-1,1)) * Dx             # (Nr,Nr)
    
    # Neumann BC via tau-style row zeroing in radial derivative
    Dx_r_bc = Dx_r.copy()
    Dx_r_bc[0, :]  = 0.0    # ∂/∂r at r=0 enforced to 0 in operator
    Dx_r_bc[-1, :] = 0.0    # ∂/∂r at r=1 enforced to 0 in operator

    ds = s[1]-s[0]             # Ls=1
    dA = alpha[1]-alpha[0]     # Lalpha=2π

    # ν(s,α) and X(r,s,α)
    ca = np.cos(alpha)[None,:]; sa = np.sin(alpha)[None,:]
    nu = ca[:,:,None]*e1[:,None,:] + sa[:,:,None]*e2[:,None,:]                   # (Ns,Na,3)
    # Dealias ν, a on periodic axes (tiny smoothing)
    nu  = filter_periodic_dealias(filter_periodic_dealias(nu, axis=0), axis=1)
    a_surf = filter_periodic_dealias(filter_periodic_dealias(a_surf, axis=0), axis=1)

    # Expand in r
    X = (r[:,None,None,None] * a_surf[None,:,:,None] * nu[None,:,:,:]) + r0[None,:,None,:]  # (Nr,Ns,Na,3)

    # Basis vectors
    # a_r = ∂X/∂r = a(s,α) * ν  (replicate along r explicitly)
    a_r_single = a_surf[None, :, :, None] * nu[None, :, :, :]           # (1,Ns,Na,3)
    a_r = np.broadcast_to(a_r_single, (Nr, Ns, Na, 3)).copy()           # (Nr,Ns,Na,3)


    # derivatives in s, α (spectral, dealiased)
    da_ds = d_periodic_fft_dealiased(a_surf, axis=0, L=1.0)                        # (Ns,Na)
    de1_ds = d_periodic_fft_dealiased(e1, axis=0, L=1.0)                           # (Ns,3)
    de2_ds = d_periodic_fft_dealiased(e2, axis=0, L=1.0)                           # (Ns,3)
    dnu_ds = ca[:,:,None]*de1_ds[:,None,:] + sa[:,:,None]*de2_ds[:,None,:]         # (Ns,Na,3)

    # a_s = ∂X/∂s = r*∂a/∂s*ν + r*a*∂ν/∂s + r0'(s)
    dr0_ds = d_periodic_fft_dealiased(r0, axis=0, L=1.0)                           # (Ns,3)
    a_s = (r[:,None,None,None]*da_ds[None,:,:,None]*nu[None,:,:,:] +
           r[:,None,None,None]*a_surf[None,:,:,None]*dnu_ds[None,:,:,:] +
           dr0_ds[None,:,None,:])

    # a_α = ∂X/∂α = r*∂a/∂α*ν + r*a*∂ν/∂α;  ∂ν/∂α = -sinα e1 + cosα e2
    da_dA = d_periodic_fft_dealiased(a_surf, axis=1, L=2.0*np.pi)                  # (Ns,Na)
    dnu_dA = (-sa[:,:,None]*e1[:,None,:] + ca[:,:,None]*e2[:,None,:])              # (Ns,Na,3)
    a_A = (r[:,None,None,None]*da_dA[None,:,:,None]*nu[None,:,:,:] +
           r[:,None,None,None]*a_surf[None,:,:,None]*dnu_dA[None,:,:,:])

    # Metric G_ij
    G = np.zeros((Nr,Ns,Na,3,3), dtype=float)
    bases = [a_r, a_s, a_A]
    for i in range(3):
        for j in range(3):
            G[..., i, j] = np.einsum('...k,...k->...', bases[i], bases[j])

    # det, inverse (with floors to avoid blowup)
    detG = (G[...,0,0]*(G[...,1,1]*G[...,2,2]-G[...,1,2]*G[...,2,1])
           -G[...,0,1]*(G[...,1,0]*G[...,2,2]-G[...,1,2]*G[...,2,0])
           +G[...,0,2]*(G[...,1,0]*G[...,2,1]-G[...,1,1]*G[...,2,0]))
    detG = np.maximum(detG, 1e-18)
    sqrtg = np.sqrt(detG)

    Ginv = np.zeros_like(G)
    c00 =  (G[...,1,1]*G[...,2,2]-G[...,1,2]*G[...,2,1])
    c01 = -(G[...,1,0]*G[...,2,2]-G[...,1,2]*G[...,2,0])
    c02 =  (G[...,1,0]*G[...,2,1]-G[...,1,1]*G[...,2,0])
    c10 = -(G[...,0,1]*G[...,2,2]-G[...,0,2]*G[...,2,1])
    c11 =  (G[...,0,0]*G[...,2,2]-G[...,0,2]*G[...,2,0])
    c12 = -(G[...,0,0]*G[...,2,1]-G[...,0,1]*G[...,2,0])
    c20 =  (G[...,0,1]*G[...,1,2]-G[...,0,2]*G[...,1,1])
    c21 = -(G[...,0,0]*G[...,1,2]-G[...,0,2]*G[...,1,0])
    c22 =  (G[...,0,0]*G[...,1,1]-G[...,0,1]*G[...,1,0])
    Ginv[...,0,0] = c00 / detG; Ginv[...,0,1] = c10 / detG; Ginv[...,0,2] = c20 / detG
    Ginv[...,1,0] = c01 / detG; Ginv[...,1,1] = c11 / detG; Ginv[...,1,2] = c21 / detG
    Ginv[...,2,0] = c02 / detG; Ginv[...,2,1] = c12 / detG; Ginv[...,2,2] = c22 / detG

    # Light regularization
    SQRTG_FLOOR = 1e-10
    GINV_CAP    = 1e6
    sqrtg = np.maximum(sqrtg, SQRTG_FLOOR)
    np.clip(Ginv, -GINV_CAP, GINV_CAP, out=Ginv)

    # ---- radial quadrature weights (Clenshaw–Curtis on x in [-1,1]) ----
    wr_x = clenshaw_curtis_weights(Nr)          # integrates over x∈[-1,1]
    wr   = 0.5 * wr_x                            # map x→r = (1-x)/2 ⇒ dr = -dx/2, so ∫_0^1 f(r)dr = 0.5∫_{-1}^1 f(x)dx
    wvol = sqrtg * wr[:, None, None]             # full volume measure ≈ sqrt(g) dr ds dα (ds,dα are uniform grids)

    return dict(
        s=s, alpha=alpha, r=r, xc=xc, Dx_r=Dx_r, Dx_r_bc=Dx_r_bc, ds=ds, dA=dA,
        X=X, a_r=a_r, a_s=a_s, a_A=a_A, G=G, Ginv=Ginv, sqrtg=sqrtg,
        wvol=wvol, wr=wr, r0=r0, e1=e1, e2=e2, a=a_surf,
    )

def apply_laplacian(phi: np.ndarray, geom: Dict[str,np.ndarray]) -> np.ndarray:
    Dx_r = geom["Dx_r_bc"]                 # << use BC-safe radial derivative (see §2)
    ds, dA = geom["ds"], geom["dA"]
    sqrtg, Ginv = geom["sqrtg"], geom["Ginv"]
    
    # --- shape guard (must match geometry grid) ---
    Nr_g, Ns_g, Na_g = geom["X"].shape[:3]
    if phi.shape != (Nr_g, Ns_g, Na_g):
        raise ValueError(f"[apply_lap] phi shape {phi.shape} != geometry {(Nr_g, Ns_g, Na_g)}")

    # Shape bookkeeping (and assertions)
    Nr, Ns, Na = map(int, phi.shape)
    M = int(Ns * Na)

    # ∂φ/∂r via Chebyshev (matmul in (Nr, M))
    phi_rs  = phi.reshape(Nr, M)
    dphi_dr = (Dx_r @ phi_rs).reshape(Nr, Ns, Na)

    # periodic derivatives (FFT + 2/3 de-alias)
    dfds = d_periodic_fft_dealiased(phi, axis=1, L=1.0)
    dfdA = d_periodic_fft_dealiased(phi, axis=2, L=2.0*np.pi)

    # fluxes
    F_r = sqrtg * (Ginv[...,0,0]*dphi_dr + Ginv[...,0,1]*dfds + Ginv[...,0,2]*dfdA)
    F_s = sqrtg * (Ginv[...,1,0]*dphi_dr + Ginv[...,1,1]*dfds + Ginv[...,1,2]*dfdA)
    F_A = sqrtg * (Ginv[...,2,0]*dphi_dr + Ginv[...,2,1]*dfds + Ginv[...,2,2]*dfdA)

    # dealias fluxes in periodic dirs (products → aliasing)
    F_s = filter_periodic_dealias(filter_periodic_dealias(F_s, axis=1), axis=2)
    F_A = filter_periodic_dealias(filter_periodic_dealias(F_A, axis=1), axis=2)
    
    # --- linear part for inhomogeneous Neumann (total φ) ---
    s_grid = geom["s"][None, :, None]                # (1, Ns, 1)
    A_grid = geom["alpha"][None, None, :]            # (1, 1, Na)
    Gc = geom.get("Gc", 0.0)
    Ic = geom.get("Ic", 0.0)
    phi_lin = Gc * s_grid + Ic * A_grid              # (1, Ns, Na)
    Nr, Ns, Na = map(int, phi.shape)

    # φ_lin is independent of r ⇒ ∂φ_lin/∂r = 0
    dlin_dr = np.zeros((Nr, Ns, Na), dtype=phi.dtype)

    # ∂φ_lin/∂s, ∂φ_lin/∂α computed on (1,Ns,Na), then broadcast across r
    dlin_ds_1 = d_periodic_fft_dealiased(phi_lin, axis=1, L=1.0)          # (1, Ns, Na)
    dlin_dA_1 = d_periodic_fft_dealiased(phi_lin, axis=2, L=2.0*np.pi)    # (1, Ns, Na)
    dlin_ds   = np.broadcast_to(dlin_ds_1, (Nr, Ns, Na)).copy()
    dlin_dA   = np.broadcast_to(dlin_dA_1, (Nr, Ns, Na)).copy()

    Fr_lin = sqrtg * ( Ginv[...,0,0]*dlin_dr
                    + Ginv[...,0,1]*dlin_ds
                    + Ginv[...,0,2]*dlin_dA )

    # enforce total zero-flux: F_r(φ̃) = -F_r(φ_lin) on both walls
    F_r[0,  :, :] = -Fr_lin[0,  :, :]
    F_r[-1, :, :] = -Fr_lin[-1, :, :]

    # divergence
    Fr_rs = F_r.reshape(Nr, M)              # << M is Ns*Na now (no more (Nr,Na))
    div_r = (Dx_r @ Fr_rs).reshape(Nr, Ns, Na)

    div_s = d_periodic_fft_dealiased(F_s, axis=1, L=1.0)
    div_A = d_periodic_fft_dealiased(F_A, axis=2, L=2.0*np.pi)

    lap = (div_r + div_s + div_A) / sqrtg
    
    if not np.isfinite(lap).all():
        bad = np.count_nonzero(~np.isfinite(lap))
        print(f"[apply_lap] WARNING: non-finite entries in lap: {bad}")
    
    return np.nan_to_num(lap, nan=0.0, posinf=0.0, neginf=0.0)

def build_rhs_for_linear_part(geom: Dict[str,np.ndarray], Gc: float, Ic: float) -> np.ndarray:
    Nr,Ns,Na = geom["X"].shape[:3]
    s = geom["s"][None,:,None]
    A = geom["alpha"][None,None,:]
    phi_lin = Gc * s + Ic * A
    rhs = -apply_laplacian(np.tile(phi_lin, (Nr,1,1)), geom)

    # ---- weighted mean removal (Neumann solvability) ----
    w = geom["wvol"]
    rhs = rhs - vol_mean_w(rhs, w)
    return rhs

def solve_neumann_poisson(rhs: np.ndarray, geom: Dict[str,np.ndarray],
                          maxiter: int = 500, tol: float = 1e-8, verbose: bool = True) -> np.ndarray:
    """
    CG on L with weighted inner product <u,v>_w = sum(u v sqrtg),
    restricted to weighted zero-mean subspace (Neumann).
    """
    w = geom["wvol"]

    def Aop(u):
        v = apply_laplacian(u, geom)
        # stay in the subspace: project to weighted zero-mean
        return proj0_w(v, w)

    # Ensure RHS is in the subspace
    rhs = proj0_w(rhs, w)

    x = np.zeros_like(rhs)
    r = rhs - Aop(x)
    p = r.copy()

    rr_old = max(dot_w(r, r, w), 0.0)
    if verbose:
        sg = geom["sqrtg"]; Ginv = geom["Ginv"]
        print("[CG] Geometry diagnostics:")
        print(f"      sqrtg min/max = {sg.min():.3e} / {sg.max():.3e}")
        print(f"      |Ginv| max    = {np.max(np.abs(Ginv)):.3e}")
        print(f"[CG] Initial ||r||_w = {np.sqrt(rr_old):.3e}")

    for it in range(1, maxiter+1):
        Ap = Aop(p)
        Ap = np.nan_to_num(Ap, nan=0.0, posinf=0.0, neginf=0.0)

        denom = dot_w(p, Ap, w)
        if (not np.isfinite(denom)) or abs(denom) < 1e-300:
            if verbose: print(f"[CG] denom non-finite/small at iter {it}, breaking.")
            break

        alpha = rr_old / denom
        if not np.isfinite(alpha):
            if verbose: print(f"[CG] alpha non-finite at iter {it}, breaking.")
            break

        x = x + alpha * p
        r = r - alpha * Ap
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

        rr_new = dot_w(r, r, w)
        if verbose and (it == 1 or it % 10 == 0):
            gain = rr_new / (rr_old + 1e-300)
            print(f"[CG] iter {it:4d}  ||r||_w = {np.sqrt(max(rr_new,0.0)):.3e}  α = {alpha:.3e}  gain={gain:.3e}")

        if rr_new < tol**2:
            if verbose:
                print(f"[CG] Converged in {it} iters, ||r||_w = {np.sqrt(max(rr_new,0.0)):.3e}")
            break

        beta = rr_new / (rr_old + 1e-300)
        p = r + beta * p
        rr_old = rr_new

    # Return weighted-zero-mean solution (it already is, but be safe)
    x = proj0_w(x, w)
    return x

def gradient_cartesian(phi: np.ndarray, geom: Dict[str,np.ndarray]) -> np.ndarray:
    """
    Compute B = ∇φ in Cartesian. Consistent with the spectral Laplacian:
      - Chebyshev in r via Dx_r
      - FFT + 2/3 de-aliasing in s and α
    Uses a^i = G^{ij} a_j (contravariant basis) and ∇φ = (∂φ/∂u^i) a^i,  u=(r,s,α).
    """
    Dx_r = geom["Dx_r_bc"]                 # (Nr,Ns,Na uses (Nr,Nr) matmul per (s,α) slab)
    Ginv  = geom["Ginv"]
    a_r, a_s, a_A = geom["a_r"], geom["a_s"], geom["a_A"]

    Nr, Ns, Na = phi.shape
    M = Ns * Na

    # ∂φ/∂r via Chebyshev
    phi_rs  = phi.reshape(Nr, M)
    dphi_dr = (Dx_r @ phi_rs).reshape(Nr, Ns, Na)

    # ∂φ/∂s, ∂φ/∂α via FFT (dealiased)
    dphi_ds = d_periodic_fft_dealiased(phi, axis=1, L=1.0)
    dphi_dA = d_periodic_fft_dealiased(phi, axis=2, L=2.0*np.pi)

    # Contravariant bases a^i = G^{ij} a_j
    a_contra_r = (Ginv[...,0,0][...,None]*a_r +
                  Ginv[...,0,1][...,None]*a_s +
                  Ginv[...,0,2][...,None]*a_A)
    a_contra_s = (Ginv[...,1,0][...,None]*a_r +
                  Ginv[...,1,1][...,None]*a_s +
                  Ginv[...,1,2][...,None]*a_A)
    a_contra_A = (Ginv[...,2,0][...,None]*a_r +
                  Ginv[...,2,1][...,None]*a_s +
                  Ginv[...,2,2][...,None]*a_A)

    # Assemble gradient
    B = (dphi_dr[...,None]*a_contra_r +
         dphi_ds[...,None]*a_contra_s +
         dphi_dA[...,None]*a_contra_A)

    # sanitize
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    return B

# --------------------- Top-level convenience ---------------------

def _print_geom_info(geom: Dict[str,np.ndarray], tag: str = "geom"):
    sqrtg = geom["sqrtg"]
    Ginv  = geom["Ginv"]
    print(f"[{tag}] sqrtg min/max = {sqrtg.min():.3e} / {sqrtg.max():.3e}")
    print(f"[{tag}] |Ginv| max    = {np.max(np.abs(Ginv)):.3e}")
    if np.any(sqrtg <= 1e-10):
        print(f"[{tag}] WARNING: sqrtg hit floor in {np.count_nonzero(sqrtg<=1e-10)} voxels.")
    if np.max(np.abs(Ginv)) >= 1e6:
        print(f"[{tag}] WARNING: Ginv hit cap (1e6). Consider gentler cross-sections / fewer modes.")
    # quick sanity on bases
    a_r = geom["a_r"]; a_s = geom["a_s"]; a_A = geom["a_A"]
    print(f"[{tag}] |a_r| med = {np.median(np.linalg.norm(a_r.reshape(-1,3), axis=1)):.3e}")
    print(f"[{tag}] |a_s| med = {np.median(np.linalg.norm(a_s.reshape(-1,3), axis=1)):.3e}")
    print(f"[{tag}] |a_A| med = {np.median(np.linalg.norm(a_A.reshape(-1,3), axis=1)):.3e}")
    wvol = geom["wvol"]
    print(f"[{tag}] volume weight sum = {wvol.sum():.6e}  (should be O(volume))")

def solve_laplace_baseline(theta: np.ndarray, meta: Dict[str,Any],
                           Nr: int, Ns: int, Na: int,
                           Gc: float = 0.0, Ic: float = 0.0,
                           maxiter: int = 400, tol: float = 1e-6, verbose: bool = True) -> Dict[str,np.ndarray]:
    geom = assemble_tube_grid(theta, meta, Nr=Nr, Ns=Ns, Na=Na)
    geom["Gc"] = float(Gc)
    geom["Ic"] = float(Ic)
    
    if verbose: _print_geom_info(geom, tag="assemble")

    rhs = build_rhs_for_linear_part(geom, Gc=Gc, Ic=Ic)
    if verbose:
        print(f"[rhs] mean={np.mean(rhs):.3e}  min/max={rhs.min():.3e}/{rhs.max():.3e}")
    if verbose: check_energy_symmetry(geom, trials=2)

    phi_tilde = solve_neumann_poisson(rhs, geom, maxiter=maxiter, tol=tol, verbose=verbose)
    if verbose:
        print_operator_residuals(phi_tilde, rhs, geom, tag="solve")
        # (Re)build full phi, then test Neumann BC on r=1
    s = geom["s"][None,:,None]
    A = geom["alpha"][None,None,:]
    phi = phi_tilde + Gc*s + Ic*A

    if verbose:
        check_neumann_bc(phi, geom, sample_frac=0.5)

    # Field
    B = gradient_cartesian(phi, geom)
    if verbose:
        bnorm = np.linalg.norm(B.reshape(-1,3), axis=1)
        print(f"[post] φ mean={np.mean(phi):.3e}  min/max={phi.min():.3e}/{phi.max():.3e}")
        print(f"[post] |B| min/max = {bnorm.min():.3e}/{bnorm.max():.3e}")

    return dict(phi=phi, B=B, X=geom["X"], geom=geom)

def plot_Bnorm_boundary_heatmap(B, geom, cmap='inferno'):
    """
    Heatmap of |B| on the boundary r=1. X-axis: s in [0,1).
    Y-axis: alpha in [0, 2π).
    """
    s = geom["s"]                    # (Ns,)
    alpha = geom["alpha"]            # (Na,)
    ridx = -1                        # r=1 boundary index
    Bnorm = np.linalg.norm(B[ridx], axis=-1)   # (Ns, Na)

    # extent: s from [s0, s_end], alpha from [0, 2π]
    ds = s[1] - s[0]
    extent = [s[0], s[-1] + ds, 0.0, 2.0*np.pi]

    plt.figure(figsize=(8, 4))
    im = plt.imshow(
        Bnorm.T, origin='lower', aspect='auto', cmap=cmap, extent=extent
    )
    plt.colorbar(im, label='|B| at r=1')
    plt.xlabel('s')
    plt.ylabel('α (rad)')
    plt.title('|B|(r=1) heatmap')
    plt.tight_layout()
    plt.show()