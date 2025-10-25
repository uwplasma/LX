import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (import side effect for 3D)

# ======================= Optimizer front-end (pack/unpack) =======================
import numpy as np
from typing import Dict, Tuple, Any

# We assume you already have in this module:
# - controls_from_fourier(...)
# - build_surface(...)

def make_param_dict(
    *,
    axis_ctrl: np.ndarray,                 # (N_CTRL_AXIS, 3)
    N_CTRL_S: int,
    m_list: Tuple[int, ...],
    a0_cos: np.ndarray, a0_sin: np.ndarray,
    alpha0_cos: np.ndarray, alpha0_sin: np.ndarray,
    ec_specs: Dict[int, Tuple[np.ndarray, np.ndarray]],   # m -> (cos_coeffs, sin_coeffs)
    es_specs: Dict[int, Tuple[np.ndarray, np.ndarray]],   # m -> (cos_coeffs, sin_coeffs)
) -> Dict[str, Any]:
    """
    Create a single dict that holds all geometry DOFs as *numbers*.

    This dict is what you hand to pack_params(...). Later you can unpack from the flat
    vector and build the surface using theta_to_surface(...).
    """
    # Defensive copies to keep everything float
    axis_ctrl = np.asarray(axis_ctrl, dtype=float)
    a0_cos    = np.asarray(a0_cos, dtype=float)
    a0_sin    = np.asarray(a0_sin, dtype=float)
    alpha0_cos = np.asarray(alpha0_cos, dtype=float)
    alpha0_sin = np.asarray(alpha0_sin, dtype=float)

    # Normalize ec/es specs into float arrays
    ec_specs = {
        int(m): (np.asarray(cos, dtype=float), np.asarray(sin, dtype=float))
        for m, (cos, sin) in ec_specs.items()
    }
    es_specs = {
        int(m): (np.asarray(cos, dtype=float), np.asarray(sin, dtype=float))
        for m, (cos, sin) in es_specs.items()
    }

    return dict(
        axis_ctrl=axis_ctrl,
        N_CTRL_S=int(N_CTRL_S),
        m_list=tuple(int(m) for m in m_list),
        a0_cos=a0_cos, a0_sin=a0_sin,
        alpha0_cos=alpha0_cos, alpha0_sin=alpha0_sin,
        ec_specs=ec_specs,
        es_specs=es_specs,
    )


def pack_params(P: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pack the parameter dict P into a single 1-D vector theta and a meta layout.

    Returns
    -------
    theta : (K,) float array
    meta  : dict describing how to unpack (names, shapes, slices, m_list, sizes)
    """
    axis_ctrl = np.asarray(P["axis_ctrl"], dtype=float)
    N_CTRL_AXIS = axis_ctrl.shape[0]
    N_CTRL_S = int(P["N_CTRL_S"])
    m_list = tuple(int(m) for m in P["m_list"])

    a0_cos = np.asarray(P["a0_cos"], dtype=float)
    a0_sin = np.asarray(P["a0_sin"], dtype=float)
    alpha0_cos = np.asarray(P["alpha0_cos"], dtype=float)
    alpha0_sin = np.asarray(P["alpha0_sin"], dtype=float)
    ec_specs = P["ec_specs"]
    es_specs = P["es_specs"]

    # Build a linear buffer
    chunks = []
    layout = []  # list of (name, key, start, stop, shape_meta)

    # 1) axis_ctrl
    start = 0
    vec = axis_ctrl.ravel()
    stop = start + vec.size
    chunks.append(vec)
    layout.append(("axis_ctrl", None, start, stop, axis_ctrl.shape))
    cursor = stop

    # 2) a0, alpha0
    for nm, arr in [("a0_cos", a0_cos), ("a0_sin", a0_sin),
                    ("alpha0_cos", alpha0_cos), ("alpha0_sin", alpha0_sin)]:
        vec = arr.ravel()
        start, stop = cursor, cursor + vec.size
        chunks.append(vec)
        layout.append((nm, None, start, stop, arr.shape))
        cursor = stop

    # 3) ec_specs and es_specs for each m in *sorted* m_list to ensure determinism
    for nm, specs in [("ec_specs", ec_specs), ("es_specs", es_specs)]:
        for m in sorted(m_list):
            cos_arr, sin_arr = specs.get(m, (np.zeros(0), np.zeros(0)))
            for sub, arr in [("cos", cos_arr), ("sin", sin_arr)]:
                vec = arr.ravel()
                start, stop = cursor, cursor + vec.size
                chunks.append(vec)
                layout.append((nm, (m, sub), start, stop, arr.shape))
                cursor = stop

    theta = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=float)

    meta = dict(
        N_CTRL_AXIS=N_CTRL_AXIS,
        N_CTRL_S=N_CTRL_S,
        m_list=m_list,
        layout=layout,    # list of tuples (name, key, start, stop, shape)
    )
    return theta, meta


def unpack_params(theta: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inverse of pack_params. Produces a dict with the same structure as make_param_dict.
    """
    theta = np.asarray(theta, dtype=float)
    N_CTRL_AXIS = meta["N_CTRL_AXIS"]
    N_CTRL_S = meta["N_CTRL_S"]
    m_list = tuple(meta["m_list"])

    # Prepare outputs
    out = dict(
        N_CTRL_S=N_CTRL_S,
        m_list=m_list,
        axis_ctrl=None,
        a0_cos=None, a0_sin=None,
        alpha0_cos=None, alpha0_sin=None,
        ec_specs={},
        es_specs={},
    )

    for (name, key, start, stop, shape) in meta["layout"]:
        vec = theta[start:stop]
        arr = vec.reshape(shape)
        if name == "axis_ctrl":
            out["axis_ctrl"] = arr
        elif name in ("a0_cos", "a0_sin", "alpha0_cos", "alpha0_sin"):
            out[name] = arr
        elif name in ("ec_specs", "es_specs"):
            m, sub = key
            if name == "ec_specs":
                if m not in out["ec_specs"]:
                    out["ec_specs"][m] = (np.zeros(0), np.zeros(0))
                cos_arr, sin_arr = out["ec_specs"][m]
                if sub == "cos":
                    cos_arr = arr
                else:
                    sin_arr = arr
                out["ec_specs"][m] = (cos_arr, sin_arr)
            else:
                if m not in out["es_specs"]:
                    out["es_specs"][m] = (np.zeros(0), np.zeros(0))
                cos_arr, sin_arr = out["es_specs"][m]
                if sub == "cos":
                    cos_arr = arr
                else:
                    sin_arr = arr
                out["es_specs"][m] = (cos_arr, sin_arr)
        else:
            raise ValueError(f"Unknown field in layout: {name}")

    # Fill any missing m entries with zeros (in case theta omitted them)
    all_ms = set(out["ec_specs"].keys()) | set(out["es_specs"].keys()) | set(m_list)
    for m in all_ms:
        out["ec_specs"].setdefault(m, (np.zeros(0), np.zeros(0)))
        out["es_specs"].setdefault(m, (np.zeros(0), np.zeros(0)))

    return out


def theta_to_surface(
    theta: np.ndarray,
    meta: Dict[str, Any],
    *,
    s_samples: int,
    alpha_samples: int,
) -> Dict[str, Any]:
    """
    Unpack theta → controls_from_fourier → build_surface.

    Returns the usual dict with:
      s, alpha, r0, t_hat, e1, e2, a, X
    """
    P = unpack_params(theta, meta)

    # Build control arrays by sampling Fourier series on the s_ctrl grid
    A0_CTRL, ALPHA0_CTRL, EC_CTRL, ES_CTRL, _s_ctrl = controls_from_fourier(
        P["N_CTRL_S"],
        P["a0_cos"], P["a0_sin"],
        P["alpha0_cos"], P["alpha0_sin"],
        P["ec_specs"], P["es_specs"],
    )

    # Build the actual surface samples
    data = build_surface(
        axis_ctrl=P["axis_ctrl"],
        s_samples=s_samples,
        alpha_samples=alpha_samples,
        a0_ctrl=A0_CTRL,
        alpha0_ctrl=ALPHA0_CTRL,
        ec_ctrl=EC_CTRL,
        es_ctrl=ES_CTRL,
        m_list=P["m_list"],
    )
    return data


def theta_to_points(
    theta: np.ndarray,
    meta: Dict[str, Any],
    *,
    s_samples: int,
    alpha_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience: return just (X, r0, a) for fast plotting or as inputs to a solver.
      X : (S, A, 3) surface points
      r0: (S, 3)     centerline
      a : (S, A)     radius map
    """
    data = theta_to_surface(theta, meta, s_samples=s_samples, alpha_samples=alpha_samples)
    return data["X"], data["r0"], data["a"]


# =============================================================================
# ------------------------- PERIODIC CUBIC B-SPLINE ---------------------------
# =============================================================================

def cubic_bspline_basis_B3(u: np.ndarray) -> np.ndarray:
    """
    Compact uniform cubic B-spline basis B3(u), support |u|<2.

      B3(u) = { (1/6)(4 - 6|u|^2 + 3|u|^3),            |u| < 1
                (1/6)(2 - |u|)^3,                      1 ≤ |u| < 2
                0,                                     otherwise }
    """
    a = np.abs(u)
    out = np.zeros_like(a)
    m1 = a < 1
    m2 = (a >= 1) & (a < 2)
    out[m1] = (1/6.0)*(4 - 6*a[m1]**2 + 3*a[m1]**3)
    out[m2] = (1/6.0)*(2 - a[m2])**3
    return out

def eval_periodic_cubic_bspline(ctrl: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Evaluate periodic uniform cubic B-spline curve from control points.

    ctrl: (N, D)
    s   : (S,) in [0,1)
    returns: (S, D)
    """
    N, D = ctrl.shape
    t = s * N
    j0 = np.floor(t).astype(int) - 1
    js = np.stack([j0 + k for k in range(4)], axis=1)     # (S,4)
    u = t[:, None] - js                                   # (S,4)
    w = cubic_bspline_basis_B3(u)                         # (S,4)
    ctrl4 = ctrl[np.mod(js, N)]                           # (S,4,D)
    x = np.einsum('si,sid->sd', w, ctrl4)
    return x

def eval_periodic_cubic_bspline_1d(ctrl: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    1D periodic cubic B-spline: ctrl (N,), s (S,) → (S,)
    """
    N = ctrl.shape[0]
    t = s * N
    j0 = np.floor(t).astype(int) - 1
    js = np.stack([j0 + k for k in range(4)], axis=1)     # (S,4)
    u = t[:, None] - js
    w = cubic_bspline_basis_B3(u)                         # (S,4)
    ctrl4 = ctrl[np.mod(js, N)]                           # (S,4)
    val = np.einsum('si,si->s', w, ctrl4)
    return val

# =============================================================================
# ------------------------ GEOMETRY & FRAME CONSTRUCTION ----------------------
# =============================================================================

def central_diff_periodic(x: np.ndarray) -> np.ndarray:
    """Central difference along axis=0 with periodic wrap."""
    xp = np.roll(x, -1, axis=0); xm = np.roll(x, +1, axis=0)
    return 0.5 * (xp - xm)

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize vectors along last axis with safety eps."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, 1e12)

def bishop_frame(t_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal-twist (parallel-transport) frame along a closed curve.

    t_hat: (S,3) unit tangent
    returns: e1, e2 (both (S,3))
    """
    S = t_hat.shape[0]
    z = np.array([0.0, 0.0, 1.0])
    e1_0 = np.cross(z, t_hat[0])
    nrm = np.linalg.norm(e1_0)
    e1_0 = np.array([1.0, 0.0, 0.0]) if nrm < 1e-8 else e1_0 / nrm
    e2_0 = np.cross(t_hat[0], e1_0)

    e1 = np.zeros_like(t_hat); e2 = np.zeros_like(t_hat)
    e1[0] = e1_0; e2[0] = e2_0
    for i in range(S - 1):
        t_next = t_hat[(i + 1) % S]
        # project previous e1 onto the normal plane at t_next
        e1_tmp = e1[i] - np.dot(e1[i], t_next) * t_next
        e1[i + 1] = normalize(e1_tmp[None, :])[0]
        e2[i + 1] = np.cross(t_next, e1[i + 1])
    return e1, e2

# =============================================================================
# ----------------------- RADIUS a(s, α) & SURFACE POINTS ---------------------
# =============================================================================

def fourier_eval_on_s(
    s: np.ndarray,
    cos_coeffs: np.ndarray,
    sin_coeffs: np.ndarray,
    two_pi: float = 2.0*np.pi,
) -> np.ndarray:
    """
    Evaluate a real Fourier series on s ∈ [0,1):

        f(s) = cos_coeffs[0]
             + ∑_{n=1}^{Nc-1} cos_coeffs[n] * cos(2π n s)
             + ∑_{n=1}^{Ns}   sin_coeffs[n] * sin(2π n s)

    Notes:
      • cos_coeffs length = Nc >= 1 (c0 is the mean).
      • sin_coeffs length = Ns (Ns may be 0).
      • This returns f(s) sampled at the provided s-grid.
    """
    s = np.asarray(s)
    cos_coeffs = np.asarray(cos_coeffs, dtype=float)
    sin_coeffs = np.asarray(sin_coeffs, dtype=float)

    f = np.zeros_like(s, dtype=float)
    if cos_coeffs.size > 0:
        f = f + cos_coeffs[0]
        for n in range(1, cos_coeffs.size):
            f += cos_coeffs[n] * np.cos(two_pi * n * s)
    for n in range(1, sin_coeffs.size + 1):
        f += sin_coeffs[n-1] * np.sin(two_pi * n * s)
    return f


def controls_from_fourier(
    N_CTRL_S: int,
    a0_cos: np.ndarray, a0_sin: np.ndarray,
    alpha0_cos: np.ndarray, alpha0_sin: np.ndarray,
    ec_specs: Dict[int, Tuple[np.ndarray, np.ndarray]],
    es_specs: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray]:
    """
    Build periodic-spline control arrays by SAMPLING Fourier series at the periodic control grid s_ctrl.

    Inputs
    ------
    N_CTRL_S : int
        Number of periodic control points in s (this defines s_ctrl = linspace(0,1,N_CTRL_S,endpoint=False)).
    a0_cos, a0_sin : array
        Cosine and sine coefficients for a0(s).
    alpha0_cos, alpha0_sin : array
        Cosine and sine coefficients for alpha0(s).
    ec_specs : dict m -> (cos_coeffs, sin_coeffs)
        For each cross-section mode m in α, Fourier series in s for ec_m(s).
    es_specs : dict m -> (cos_coeffs, sin_coeffs)
        For each cross-section mode m in α, Fourier series in s for es_m(s).

    Returns
    -------
    A0_CTRL : (N_CTRL_S,)
    ALPHA0_CTRL : (N_CTRL_S,)
    EC_CTRL : dict m -> (N_CTRL_S,)
    ES_CTRL : dict m -> (N_CTRL_S,)
    s_ctrl : (N_CTRL_S,)
    """
    s_ctrl = np.linspace(0.0, 1.0, N_CTRL_S, endpoint=False)

    A0_CTRL = fourier_eval_on_s(s_ctrl, a0_cos, a0_sin)
    ALPHA0_CTRL = fourier_eval_on_s(s_ctrl, alpha0_cos, alpha0_sin)

    EC_CTRL: Dict[int, np.ndarray] = {}
    ES_CTRL: Dict[int, np.ndarray] = {}

    for m, (c_cos, c_sin) in ec_specs.items():
        EC_CTRL[m] = fourier_eval_on_s(s_ctrl, c_cos, c_sin)
    for m, (c_cos, c_sin) in es_specs.items():
        ES_CTRL[m] = fourier_eval_on_s(s_ctrl, c_cos, c_sin)

    # Ensure every mode in ec has a matching es (and vice-versa); fill zeros if missing
    all_ms = set(EC_CTRL.keys()) | set(ES_CTRL.keys())
    for m in all_ms:
        if m not in EC_CTRL:
            EC_CTRL[m] = np.zeros_like(s_ctrl)
        if m not in ES_CTRL:
            ES_CTRL[m] = np.zeros_like(s_ctrl)

    return A0_CTRL, ALPHA0_CTRL, EC_CTRL, ES_CTRL, s_ctrl


def build_surface_from_fourier(
    axis_ctrl: np.ndarray,
    s_samples: int,
    alpha_samples: int,
    N_CTRL_S: int,
    m_list: Tuple[int, ...],
    a0_cos: np.ndarray, a0_sin: np.ndarray,
    alpha0_cos: np.ndarray, alpha0_sin: np.ndarray,
    ec_specs: Dict[int, Tuple[np.ndarray, np.ndarray]],
    es_specs: Dict[int, Tuple[np.ndarray, np.ndarray]],
    *,
    _build_surface_callable=None,
) -> Dict:
    """
    Convenience wrapper:
      (Fourier numbers) → (periodic control arrays) → call build_surface(...)

    If you renamed/moved build_surface, pass it in via _build_surface_callable.

    Returns the same dict as build_surface(...).
    """
    build_surface_fn = _build_surface_callable or build_surface

    A0_CTRL, ALPHA0_CTRL, EC_CTRL, ES_CTRL, _ = controls_from_fourier(
        N_CTRL_S,
        a0_cos, a0_sin,
        alpha0_cos, alpha0_sin,
        ec_specs, es_specs,
    )

    return build_surface_fn(
        axis_ctrl=axis_ctrl,
        s_samples=s_samples,
        alpha_samples=alpha_samples,
        a0_ctrl=A0_CTRL,
        alpha0_ctrl=ALPHA0_CTRL,
        ec_ctrl=EC_CTRL,
        es_ctrl=ES_CTRL,
        m_list=m_list,
    )

def radius_a(s_grid: np.ndarray,
             alpha_grid: np.ndarray,
             a0_ctrl: np.ndarray,
             alpha0_ctrl: np.ndarray,
             ec_ctrl: Dict[int, np.ndarray],
             es_ctrl: Dict[int, np.ndarray],
             m_list: Tuple[int, ...]) -> np.ndarray:
    """
    a(s,α) = a0(s) * [ 1 + Σ_m ( ec_m(s) cos(m(α - α0(s))) + es_m(s) sin(m(α - α0(s))) ) ].

    Each function of s is specified by periodic cubic B-spline controls.
    """
    a0     = eval_periodic_cubic_bspline_1d(a0_ctrl, s_grid)      # (S,)
    alpha0 = eval_periodic_cubic_bspline_1d(alpha0_ctrl, s_grid)  # (S,)

    S = s_grid.shape[0]; A = alpha_grid.shape[0]
    alpha_mat = alpha_grid[None, :] - alpha0[:, None]             # (S,A)
    acc = np.zeros((S, A))
    for m in m_list:
        ec_m = eval_periodic_cubic_bspline_1d(ec_ctrl[m], s_grid) # (S,)
        es_m = eval_periodic_cubic_bspline_1d(es_ctrl[m], s_grid) # (S,)
        acc += ec_m[:, None]*np.cos(m*alpha_mat) + es_m[:, None]*np.sin(m*alpha_mat)
    return a0[:, None] * (1.0 + acc)

def build_surface(axis_ctrl: np.ndarray,
                  s_samples: int,
                  alpha_samples: int,
                  a0_ctrl: np.ndarray,
                  alpha0_ctrl: np.ndarray,
                  ec_ctrl: Dict[int, np.ndarray],
                  es_ctrl: Dict[int, np.ndarray],
                  m_list: Tuple[int, ...]) -> Dict:
    """
    From numeric inputs → surface samples and helpful fields.
    """
    S = s_samples; A = alpha_samples
    s = np.linspace(0.0, 1.0, S, endpoint=False)
    alpha = np.linspace(0.0, 2*np.pi, A, endpoint=False)

    # Axis and tangent
    r0 = eval_periodic_cubic_bspline(axis_ctrl, s)      # (S,3)
    t_hat = normalize(central_diff_periodic(r0))        # (S,3)

    # Bishop frame
    e1, e2 = bishop_frame(t_hat)

    # Radius field
    a = radius_a(s, alpha, a0_ctrl, alpha0_ctrl, ec_ctrl, es_ctrl, m_list)  # (S,A)

    # Surface points
    ca, sa = np.cos(alpha)[None, :], np.sin(alpha)[None, :]
    X = r0[:, None, :] + a[:, :, None]*(ca[:, :, None]*e1[:, None, :] + sa[:, :, None]*e2[:, None, :])

    return dict(s=s, alpha=alpha, r0=r0, t_hat=t_hat, e1=e1, e2=e2, a=a, X=X)

# =============================================================================
# ----------------------- FIGURE-8 RACETRACK AXIS EXAMPLE ---------------------
# =============================================================================

def rotate_y_to_z(pts: np.ndarray) -> np.ndarray:
    """
    Rotate points about the x-axis by +90 degrees so that:
      y' = -z
      z' =  y
    i.e., the old y-axis maps to the new z-axis.
    pts: (N,3)
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    y_new = -z
    z_new =  y
    return np.stack([x, y_new, z_new], axis=1)

def make_spitzer_like_fig8_ctrl(
    N_CTRL: int,
    L_plateau: float = 1.2,     # half-extent of straight sections in x
    R_bend: float = 0.35,       # lateral lobe size in y
    k_plateau: float = 3.5,     # steepness of tanh plateaus (↑ → straighter)
    z_gap: float = 0.10,        # separation at the waist to avoid self-intersection
    twist_turns: int = 0,       # optional extra z-wobble (integer turns keeps closure)
    enforce_stellarator_sym: bool = True  # rephase/translate to meet your symmetry spec
) -> np.ndarray:
    """
    Spitzer-like figure-8 with racetrack straights and short bends.

    Parametrization (t ∈ [0,1)):
        x(t) = L_plateau * tanh(k_plateau * cos(2π t))
        y(t) = R_bend * sin(4π t) / (1 + 0.3 * sin^2(2π t))
        z(t) = z_gap * sin(2π t) + (0.05*R_bend)*sin(2π*twist_turns*t)  [last term optional]

    With enforce_stellarator_sym=True:
      • Pick the waist crossing where z≈0 and |y| is minimal.
      • Rephase so that this point is s=0.
      • Translate so that this point is exactly (0,0,0).
      ⇒ Then s=0 and s=0.5 both lie on z=0, giving up–down symmetry.
    """
    # Dense parametric samples for a smooth resample
    M = max(4000, 200 * N_CTRL)
    t = np.linspace(0.0, 1.0, M, endpoint=False)

    # Plateaued x (racetrack-like straights)
    x = L_plateau * np.tanh(k_plateau * np.cos(2*np.pi*t))

    # Two lobes in y with flattened mid-sections
    denom = 1.0 + 0.3 * (np.sin(2*np.pi*t)**2)
    y = R_bend * np.sin(4*np.pi*t) / denom

    # Non-self-intersecting waist (and optional extra twist wobble)
    z = z_gap * np.sin(2*np.pi*t)
    if twist_turns:
        z = z + 0.05 * R_bend * np.sin(2*np.pi*twist_turns * t)

    pts = np.stack([x, y, z], axis=1)

    # --- rotate so old y becomes new z ---
    pts = rotate_y_to_z(pts)

    if enforce_stellarator_sym:
        # Find a waist-crossing with z≈0 and |y| minimal (prioritize central crossing)
        # Score each point by |z| + small*|y|, then choose the minimum.
        score = np.abs(z) + 0.1*np.abs(y)
        i0 = int(np.argmin(score))

        # Rephase: rotate the sequence so this crossing is at index 0 (s=0)
        pts = np.vstack([pts[i0:], pts[:i0]])

        # Translate so that this s=0 point is exactly at the origin (0,0,0)
        pts = pts - pts[0]

        # (Optional) ensure s=0.5 is also z≈0:
        # With z(t)=sin(2πt) it already holds: z(t+0.5) = -z(t) ⇒ z=0 at both ends.

    # Uniform-arclength resample to N_CTRL periodic control points
    dif = np.diff(np.vstack([pts, pts[0]]), axis=0)
    seg = np.linalg.norm(dif, axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = s_cum[-1]
    targets = np.linspace(0.0, total, N_CTRL+1)[:-1]

    CTRL = np.zeros((N_CTRL, 3))
    j = 0
    for k, tk in enumerate(targets):
        while s_cum[j+1] < tk:
            j += 1
        t0, t1 = s_cum[j], s_cum[j+1]
        w = 0.0 if t1 == t0 else (tk - t0) / (t1 - t0)
        CTRL[k] = (1-w)*pts[j % M] + w*pts[(j+1) % M]

    return CTRL

# =============================================================================
# ------------------------------ VISUALIZATION --------------------------------
# =============================================================================

def plot_surface_package(data: Dict,
                         n_frame_arrows: int = 24,
                         n_cross_sections: int = 8,
                         figsize=(15, 10)) -> None:
    """
    Three-panel visualization package:
      1) 3D surface (colored by a(s,α)) + axis + Bishop frame arrows
      2) Several cross-sections (XY projection)
      3) Heatmap of a(s,α) vs (s,α)
    """
    s, alpha = data['s'], data['alpha']
    r0, e1, e2, a, X = data['r0'], data['e1'], data['e2'], data['a'], data['X']
    S, A = a.shape

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.2])
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_xsec = fig.add_subplot(gs[0, 1])
    ax_map = fig.add_subplot(gs[1, :])

    # --- 1) 3D surface ---
    ax3d.plot(r0[:,0], r0[:,1], r0[:,2], 'k-', lw=2, label='axis')

    # frame arrows
    idxs = np.linspace(0, S-1, n_frame_arrows, endpoint=False).astype(int)
    scale = 0.35*np.max(a) if np.isfinite(np.max(a)) else 0.1
    for i in idxs:
        p = r0[i]
        ax3d.quiver(p[0], p[1], p[2], e1[i,0], e1[i,1], e1[i,2], length=scale, color='C1', linewidth=1)
        ax3d.quiver(p[0], p[1], p[2], e2[i,0], e2[i,1], e2[i,2], length=scale, color='C2', linewidth=1)

    # surface (downsample for speed if large)
    skip_s = max(1, S//160)
    skip_a = max(1, A//180)
    Xs = X[::skip_s, ::skip_a, :]
    a_col = a[::skip_s, ::skip_a]
    rng = np.ptp(a_col) + 1e-12
    norm = (a_col - np.min(a_col)) / rng
    facecolors = plt.cm.viridis(norm)
    ax3d.plot_surface(Xs[:,:,0], Xs[:,:,1], Xs[:,:,2],
                      facecolors=facecolors, rstride=1, cstride=1,
                      linewidth=0, antialiased=False, alpha=0.35)
    m = plt.cm.ScalarMappable(cmap='viridis'); m.set_array(a)
    cb = plt.colorbar(m, ax=ax3d, shrink=0.6, pad=0.1); cb.set_label('a(s, α)')

    ax3d.set_title("Surface and Bishop frame")
    ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
    ax3d.view_init(elev=28, azim=38)

    # --- 2) Cross-sections (local (e1,e2) plane) ---
    ax_xsec.set_aspect('equal', adjustable='box')
    idx_cs = np.linspace(0, S-1, n_cross_sections, endpoint=False).astype(int)
    colors = plt.cm.tab10(np.linspace(0, 1, len(idx_cs)))

    # helper: project a section into its local cross-section plane
    def local_uv_section(i):
        # Xsec: (A,3), center r0[i], basis (e1[i], e2[i])
        rel = X[i, :, :] - r0[i][None, :]            # (A,3)
        u = rel @ e1[i]                               # (A,)
        v = rel @ e2[i]                               # (A,)
        return u, v

    # set symmetric limits based on max radius in the selected sections
    max_a_sel = 0.0
    for i in idx_cs:
        max_a_sel = max(max_a_sel, float(np.max(a[i])))
    lim = 1.1 * max_a_sel if np.isfinite(max_a_sel) and max_a_sel > 0 else 0.1
    ax_xsec.set_xlim(-lim, lim)
    ax_xsec.set_ylim(-lim, lim)

    for k, i in enumerate(idx_cs):
        u, v = local_uv_section(i)
        ax_xsec.plot(u, v, '-', color=colors[k], lw=1.6, label=f's={s[i]:.2f}')

        # draw local basis arrows and a reference circle of radius a0 (mean) if you want
        # estimate a0(s_i) as the mean radius in the section:
        a0_i = float(np.mean(np.sqrt(u**2 + v**2)))
        ax_xsec.add_artist(plt.Circle((0, 0), a0_i, fill=False, color=colors[k], alpha=0.25, lw=1.0))

        # draw local axes (e1,e2) arrows in uv coordinates
        ax_xsec.arrow(0.0, 0.0, 0.6*a0_i, 0.0, color=colors[k], head_width=0.02*lim,
                      length_includes_head=True, alpha=0.6)
        ax_xsec.arrow(0.0, 0.0, 0.0, 0.6*a0_i, color=colors[k], head_width=0.02*lim,
                      length_includes_head=True, alpha=0.6)

    ax_xsec.set_title("Cross-sections (local (e1,e2) plane; no foreshortening)")
    ax_xsec.legend(ncol=2, fontsize=8, loc='upper right')
    ax_xsec.grid(True, alpha=0.3)
    ax_xsec.set_xlabel('u (along e1)'); ax_xsec.set_ylabel('v (along e2)')

    # --- 3) a(s, α) heatmap ---
    im = ax_map.imshow(a.T, origin='lower', aspect='auto',
                       extent=[s[0], s[-1] + (s[1]-s[0]), 0, 2*np.pi], cmap='viridis')
    plt.colorbar(im, ax=ax_map, orientation='horizontal', pad=0.2, label='a(s, α)')
    ax_map.set_xlabel('s'); ax_map.set_ylabel('α (rad)')
    ax_map.set_title('Radius map a(s, α)')

    plt.tight_layout(); plt.show()
