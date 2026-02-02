######## Deprecated ########

from __future__ import annotations

import numpy as np
from tbg_rebuild.utils.adapter import Structure


def apply_inplane_deformation(struct: Structure, F2: np.ndarray) -> Structure:
    """
    (Deprecated) Apply an in-plane deformation gradient to a Structure.

    This helper applies a 2x2 deformation gradient `F2` to:
      - atomic xy coordinates: positions[:, :2]
      - in-plane cell vectors (row-stacked): cell[0, :2], cell[1, :2]

    Mathematical convention used here:
      - Atomic xy positions are treated as column vectors:
          x' = F2 x
        implemented as:
          R2[:, :2] = (F2 @ R[:, :2].T).T

      - Cell is stored row-stacked (ASE convention). Each row vector v transforms as:
          v' = v F2^T
        because row vectors multiply on the right:
          cell2[i, :2] = cell[i, :2] @ F2.T   for i in {0, 1}

    Out-of-plane behavior:
      - z-coordinates and the out-of-plane cell vector (cell[2, :]) are unchanged.

    Metadata:
      - Adds meta["strain_applied"] = True in the returned Structure.

    Notes:
      - This function is marked deprecated because the project now prefers the
        non-deprecated strain/affine application utilities in `tbg_rebuild.strain`.
      - Kept for reference and backwards compatibility in older scripts.
    """
    F2 = np.asarray(F2, dtype=float)
    if F2.shape != (2, 2):
        raise ValueError(f"F2 must be (2,2), got {F2.shape}")

    R = np.asarray(struct.positions, dtype=float)
    cell = np.asarray(struct.cell, dtype=float)

    # Deform xy positions
    R2 = R.copy()
    R2[:, :2] = (F2 @ R[:, :2].T).T

    # Deform in-plane cell rows (row vectors transform with F^T)
    cell2 = cell.copy()
    cell2[0, :2] = cell[0, :2] @ F2.T
    cell2[1, :2] = cell[1, :2] @ F2.T
    # cell2[2, :] unchanged

    meta = dict(struct.meta)
    meta["strain_applied"] = True

    return Structure(
        positions=R2,
        cell=cell2,
        pbc=np.asarray(struct.pbc, dtype=bool).copy(),
        species=list(struct.species),
        arrays={k: np.asarray(v).copy() for k, v in struct.arrays.items()},
        meta=meta,
    )
