import numpy as np

def rotz(theta_deg : float) -> np.ndarray:
# Constructs rotation matrix about Z axis by 'theta_deg' degrees.
    angle = np.deg2rad(theta_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]])
    return R

def cartToFrac(R : np.ndarray, H: np.ndarray, HinvT: np.ndarray | None = None) -> np.ndarray:
# If you need to call this function many times, precompute HinvT for performance
    R = np.asarray(R, float)
    H = np.asarray(H, float)

    if HinvT is None:
        HinvT = np.linalg.inv(H).T

    return R @ HinvT

def fracToCart(S, H):
    S = np.asarray(S, dtype = float)
    H = np.asarray(H, dtype = float)
    return S @ H.T

def colsToRowsCell(H_cols: np.ndarray) -> np.ndarray:
    H = np.asarray(H_cols, float)
    assert H.shape == (3, 3)
    return H.T

def rowsToColsCell(H_rows: np.ndarray) -> np.ndarray:
    H = np.asarray(H_rows, float)
    assert H.shape == (3, 3)
    return H.T
