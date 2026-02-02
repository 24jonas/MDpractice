# tbg/utils/geom.py
from __future__ import annotations

import numpy as np


# -----------------------------
# Rotations (explicit units)
# -----------------------------
def rotz_deg(theta_deg: float) -> np.ndarray:
    """3×3 rotation about +z by theta in degrees."""
    th = np.deg2rad(float(theta_deg))
    c, s = float(np.cos(th)), float(np.sin(th))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def rotz_rad(theta_rad: float) -> np.ndarray:
    """3×3 rotation about +z by theta in radians."""
    th = float(theta_rad)
    c, s = float(np.cos(th)), float(np.sin(th))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


# -----------------------------
# Coordinate transforms
# Cell convention: row-stacked (ASE)
#   R_cart = S_frac @ cell_rows
#   S_frac = R_cart @ inv(cell_rows)
# -----------------------------
def cart_to_frac(R: np.ndarray, cell_rows: np.ndarray) -> np.ndarray:
    """
    Convert cartesian positions R (N,3) to fractional S (N,3) using row-stacked cell.

    For ASE-style row-stacked cell:
      R = S @ cell
      => S = R @ inv(cell)
    """
    R = np.asarray(R, dtype=float)
    cell = np.asarray(cell_rows, dtype=float)

    if cell.shape != (3, 3):
        raise ValueError(f"cell_rows must be shape (3,3), got {cell.shape}")
    if R.ndim != 2 or R.shape[1] != 3:
        raise ValueError(f"R must be shape (N,3), got {R.shape}")

    inv_cell = np.linalg.inv(cell)
    return R @ inv_cell


def frac_to_cart(S: np.ndarray, cell_rows: np.ndarray) -> np.ndarray:
    """
    Convert fractional coordinates S (N,3) to cartesian positions R (N,3)
    using row-stacked cell: R = S @ cell.
    """
    S = np.asarray(S, dtype=float)
    cell = np.asarray(cell_rows, dtype=float)

    if cell.shape != (3, 3):
        raise ValueError(f"cell_rows must be shape (3,3), got {cell.shape}")
    if S.ndim != 2 or S.shape[1] != 3:
        raise ValueError(f"S must be shape (N,3), got {S.shape}")

    return S @ cell


# -----------------------------
# Wrapping
# -----------------------------
def wrap_frac_xy(S: np.ndarray) -> np.ndarray:
    """
    Wrap fractional coords in x,y into [0,1). Leaves z unchanged.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[1] != 3:
        raise ValueError(f"S must be shape (N,3), got {S.shape}")

    out = S.copy()
    out[:, 0] = out[:, 0] - np.floor(out[:, 0])
    out[:, 1] = out[:, 1] - np.floor(out[:, 1])
    return out


def wrap_positions_xy(R: np.ndarray, cell_rows: np.ndarray) -> np.ndarray:
    """
    Wrap cartesian positions in x,y into the unit cell using fractional wrap.
    Leaves z unchanged.
    """
    S = cart_to_frac(R, cell_rows)
    Sw = wrap_frac_xy(S)
    return frac_to_cart(Sw, cell_rows)


# -----------------------------
# Cell metrics (2D slab focus)
# -----------------------------
def cell_metrics(cell_rows: np.ndarray) -> dict:
    """
    Return basic in-plane cell metrics for a slab:
    - |a1|, |a2|
    - angle between a1 and a2 (deg)
    - in-plane area
    - cell_z length
    """
    cell = np.asarray(cell_rows, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(f"cell_rows must be shape (3,3), got {cell.shape}")

    a1 = cell[0, :]
    a2 = cell[1, :]
    a3 = cell[2, :]

    L1 = float(np.linalg.norm(a1))
    L2 = float(np.linalg.norm(a2))

    denom = (L1 * L2) if (L1 > 0 and L2 > 0) else 1.0
    cosang = float(np.clip(np.dot(a1, a2) / denom, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cosang)))

    area = float(np.linalg.norm(np.cross(a1, a2)))
    cell_z = float(np.linalg.norm(a3))

    return {"|a1|": L1, "|a2|": L2, "angle_deg": angle_deg, "area": area, "cell_z": cell_z}


# -----------------------------
# Minimal-image deltas (xy)
# -----------------------------
def min_image_delta_frac_xy(dS: np.ndarray) -> np.ndarray:
    """
    Apply minimal-image convention in fractional x,y:
      dS_xy -= round(dS_xy)
    Leaves z unchanged.
    """
    dS = np.asarray(dS, dtype=float)
    out = dS.copy()
    out[..., 0] -= np.round(out[..., 0])
    out[..., 1] -= np.round(out[..., 1])
    return out


# -----------------------------
# Cheap sampled diagnostics
# -----------------------------
def sample_indices(N: int, max_atoms: int, seed: int = 0) -> np.ndarray:
    """Choose up to max_atoms unique indices from range(N)."""
    N = int(N)
    max_atoms = int(max_atoms)
    if max_atoms <= 0 or N <= 0:
        return np.array([], dtype=int)
    if N <= max_atoms:
        return np.arange(N, dtype=int)
    rng = np.random.default_rng(seed)
    return rng.choice(N, size=max_atoms, replace=False)


def min_distance_xy_sample(R: np.ndarray, cell_rows: np.ndarray, max_atoms: int = 3000, seed: int = 0) -> float:
    """
    Sampled minimum pair distance considering PBC in x,y (z included in distance).
    Uses a simple O(M^2) broadcast on the sampled set; keep max_atoms modest.
    """
    R = np.asarray(R, dtype=float)
    idx = sample_indices(len(R), max_atoms, seed=seed)
    if len(idx) < 2:
        return float("inf")

    Rs = R[idx]
    S = cart_to_frac(Rs, cell_rows)  # (M,3)
    dS = S[None, :, :] - S[:, None, :]  # (M,M,3)
    dS = min_image_delta_frac_xy(dS)
    dR = frac_to_cart(dS.reshape(-1, 3), cell_rows).reshape(dS.shape)  # (M,M,3)

    d2 = np.einsum("ijk,ijk->ij", dR, dR)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min()))


def neighbor_counts_xy_sample(
    R: np.ndarray,
    cell_rows: np.ndarray,
    cutoff: float,
    max_atoms: int = 3000,
    seed: int = 0,
) -> np.ndarray:
    """
    Sampled neighbor count within cutoff using PBC in x,y (z included).
    Returns counts for the sampled atoms only.
    """
    R = np.asarray(R, dtype=float)
    cutoff = float(cutoff)
    idx = sample_indices(len(R), max_atoms, seed=seed)
    if len(idx) == 0:
        return np.array([], dtype=int)

    Rs = R[idx]
    S = cart_to_frac(Rs, cell_rows)
    dS = S[None, :, :] - S[:, None, :]
    dS = min_image_delta_frac_xy(dS)
    dR = frac_to_cart(dS.reshape(-1, 3), cell_rows).reshape(dS.shape)

    d2 = np.einsum("ijk,ijk->ij", dR, dR)
    np.fill_diagonal(d2, np.inf)
    return np.sum(d2 <= cutoff * cutoff, axis=1).astype(int)
