# tbg_rebuild/validate/dupes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from tbg_rebuild.utils.geom import cart_to_frac, wrap_frac_xy, frac_to_cart


@dataclass(frozen=True, slots=True)
class DupeSummary:
    n_atoms: int
    n_dupe_atoms: int                 # atoms involved in any duplicate collision
    n_dupe_groups: int                # number of hash collisions (groups)
    groups: List[List[int]]           # index lists (each group size >= 2)
    sample: List[Dict]                # small sample diagnostics


def _quantize_frac_xy_z(
    frac: np.ndarray,
    z: np.ndarray,
    *,
    cell_rows: np.ndarray,
    tol_xy_ang: float,
    tol_z_ang: float,
) -> np.ndarray:
    """
    Create integer keys for hashing with ~tol in Å.

    Strategy:
    - convert tol_xy_ang (Å) to a conservative fractional tolerance using the *shortest*
      in-plane lattice vector length.
    - quantize wrapped fractional x,y by that step.
    - quantize z directly in Å by tol_z_ang.

    Returns keys as (N,3) int64.
    """
    # In-plane scale in Å: use min(|a1|, |a2|) for conservative fraction step
    a1 = cell_rows[0, :2]
    a2 = cell_rows[1, :2]
    L1 = float(np.linalg.norm(a1))
    L2 = float(np.linalg.norm(a2))
    Lmin = max(min(L1, L2), 1e-12)

    tol_frac = float(tol_xy_ang) / Lmin
    tol_frac = max(tol_frac, 1e-12)

    qx = np.floor(frac[:, 0] / tol_frac + 0.5).astype(np.int64)
    qy = np.floor(frac[:, 1] / tol_frac + 0.5).astype(np.int64)
    qz = np.floor(z / float(tol_z_ang) + 0.5).astype(np.int64)

    return np.stack([qx, qy, qz], axis=1)


def find_duplicates_periodic_xy(
    positions: np.ndarray,
    cell_rows: np.ndarray,
    *,
    tol_xy_ang: float = 1e-3,
    tol_z_ang: float = 1e-3,
    max_groups_report: int = 10,
) -> DupeSummary:
    """
    Detect duplicates in a slab with periodicity in x,y and nonperiodic z.

    Duplicate definition:
    - Two atoms are "duplicates" if their wrapped fractional (x,y) coordinates match
      within tol_xy_ang (converted conservatively to fractional), AND their z matches
      within tol_z_ang.

    Complexity:
    - O(N) hashing.

    Notes:
    - This is a *gate* check (catch obvious duplication/cropping bugs early).
    - Later, for twist/crop you may also want a near-duplicate spatial check, but this
      is the correct first line of defense and is very fast.
    """
    R = np.asarray(positions, dtype=float)
    cell = np.asarray(cell_rows, dtype=float)
    if R.ndim != 2 or R.shape[1] != 3:
        raise ValueError(f"positions must be (N,3), got {R.shape}")
    if cell.shape != (3, 3):
        raise ValueError(f"cell_rows must be (3,3), got {cell.shape}")

    N = R.shape[0]
    frac = cart_to_frac(R, cell)          # (N,3)
    frac_w = wrap_frac_xy(frac)           # wrap x,y into [0,1)
    z = R[:, 2].astype(float)

    keys = _quantize_frac_xy_z(frac_w, z, cell_rows=cell, tol_xy_ang=tol_xy_ang, tol_z_ang=tol_z_ang)

    # Hash keys -> groups
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(N):
        k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
        buckets.setdefault(k, []).append(i)

    groups = [idxs for idxs in buckets.values() if len(idxs) >= 2]
    n_dupe_groups = len(groups)
    dupe_atoms = sorted({i for g in groups for i in g})
    n_dupe_atoms = len(dupe_atoms)

    # Build sample diagnostics
    sample = []
    for g in groups[:max_groups_report]:
        i0, i1 = g[0], g[1]
        # compute the actual wrapped xy difference in Å for the first pair
        dS = frac_w[i1] - frac_w[i0]
        dS[0] -= np.round(dS[0])
        dS[1] -= np.round(dS[1])
        dR = frac_to_cart(dS.reshape(1, 3), cell)[0]
        sample.append(
            {
                "group_size": len(g),
                "indices": g[:10],
                "dxy_ang_first_pair": float(np.linalg.norm(dR[:2])),
                "dz_ang_first_pair": float(abs(R[i1, 2] - R[i0, 2])),
                "frac0": frac_w[i0].tolist(),
                "frac1": frac_w[i1].tolist(),
            }
        )

    return DupeSummary(
        n_atoms=N,
        n_dupe_atoms=n_dupe_atoms,
        n_dupe_groups=n_dupe_groups,
        groups=groups,
        sample=sample,
    )
