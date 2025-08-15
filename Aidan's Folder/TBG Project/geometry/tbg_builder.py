# geometry/tbg_builder.py
import numpy as np
from .grapheneSheetGen import generate_graphene_sheet

def rotz(theta_deg):
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]])
    return R

def _cart_to_frac(R, H, HinvT=None):
    R = np.asarray(R, float)
    H = np.asarray(H, float)
    if HinvT is None:
        HinvT = np.linalg.inv(H).T
    return R @ HinvT

def _frac_to_cart(S, H):
    return np.asarray(S, float) @ np.asarray(H, float).T

def _wrap01(S, wrap_z=False):
    S = np.asarray(S, float).copy()
    S[..., :2] -= np.floor(S[..., :2])
    if wrap_z:
        S[..., 2] -= np.floor(S[..., 2])
    return S

def dupeCheck(R, tol=1e-6):
## Checks for duplicate atoms near periodic boundary
## Vectorize with numpy if performance is an issue
    if len(R) == 0:
        return R
    keep = [0]
    for i in range(1, len(R)):
        if np.min(np.sum((R[i] - R[keep])**2, axis=1)) > tol**2:
            keep.append(i)
    return R[np.array(keep)]

def build_tbg(nx, ny, a=2.46, d=3.35, theta_deg=0.0, registry="AB", vacuum=20.0, pad=2.0):
    ''' 
    Registry: controls which atoms sit on top of each other. AA, AB, or BA. AA sits top a over bottom a, AB sits top a over bottom b, etc. 
    pad: scales up the top layer so that we don't get holes when it is rotated.
    '''

    # Target cell from  generator 
    R_bot, H = generate_graphene_sheet(a=a, nx=nx, ny=ny, vacuum=vacuum)
    R_bot = R_bot.copy()
    R_bot[:, 2] = -0.5 * d  # set z

    # Build oversized top layer BEFORE rotation
    nx_pad = max(nx, int(np.ceil(nx * pad)) + 2)
    ny_pad = max(ny, int(np.ceil(ny * pad)) + 2)
    R_top_big, H_pad = generate_graphene_sheet(a=a, nx=nx_pad, ny=ny_pad, vacuum=vacuum)
    R_top_big = R_top_big.copy()
    R_top_big[:, 2] = +0.5 * d

    # shift using primitive lattice
    a1 = a * np.array([1.0, 0.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3)/2, 0.0])
    if registry.upper() == "AA":
        shift = np.zeros(3)
    elif registry.upper() == "AB":
        shift = (a1 + a2) / 3.0
    else:
        shift = np.zeros(3)

    # Make rotation matrix Q, then apply Q and shift
    Q = rotz(theta_deg) 
    R_top_rot = (R_top_big @ Q.T) + shift

    # Clip top layer to make sure we only keep atoms that are meant to be in the cell they're in
    # Mask applies our clip to make sure there's no weirdness caused by periodic boundaries
    HinvT = np.linalg.inv(H).T
    S_top = _cart_to_frac(R_top_rot, H, HinvT=HinvT)
    eps = 1e-6
    mask = (S_top[:, 0] >= -eps) & (S_top[:, 0] < 1.0 + eps) & (S_top[:, 1] >= -eps) & (S_top[:, 1] < 1.0 + eps)

    # Apply wrapping to deal with stragglers after clipping
    S_top = _wrap01(S_top[mask], wrap_z=False)

    # Convert back to cartesian for graphing, then make sure layers are spaced correctly
    R_top = _frac_to_cart(S_top, H)
    R_top[:, 2] = +0.5 * d

    # boundary check
    R_bot = dupeCheck(R_bot, tol=1e-6)
    R_top = dupeCheck(R_top, tol=1e-6)

    # Stack into one array
    R_all = np.vstack([R_bot, R_top])

    # Dictionary for quick reference
    meta = {
        "n_bottom": len(R_bot),
        "n_top": len(R_top),
        "theta_deg": float(theta_deg),
        "registry": registry.upper(),
        "cell": H.copy(),
    }
    return R_all, H, meta
