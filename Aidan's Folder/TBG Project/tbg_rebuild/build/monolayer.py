# tbg/build/monolayer.py
from __future__ import annotations

from typing import Tuple, Dict, Any
import numpy as np

from tbg_rebuild.materials.catalog import get_material, Material
from tbg_rebuild.materials.primitives import primitive_2d
from tbg_rebuild.utils.adapter import Structure
from tbg_rebuild.utils.geom import wrap_positions_xy


def build_monolayer(
    material_id: str,
    repeats: Tuple[int, int],
    *,
    vacuum: float = 24.0,
    z0: float = 0.0,
    pbc: Tuple[bool, bool, bool] = (True, True, False),
    stage: str = "built",
    job_id: str | None = None,
) -> Structure:
    """
    Build a tiled monolayer sheet for a hexagonal 2D material.

    What this does:
    - Builds the primitive 2D slab cell for `material_id` (includes a z-vacuum height).
    - Tiles it `repeats=(n1, n2)` times along the primitive in-plane vectors (a1, a2).
    - Sets all atom z-coordinates to `z0` (a simple way to position the layer in a stack).
    - Wraps x/y positions into the periodic cell for safety.

    Outputs (Structure fields):
    - cell: (3,3) row-stacked cell vectors with vacuum in z
    - positions: (N,3) cartesian Å
    - species: list of chemical symbols
    - arrays:
        - layer_id: all zeros (monolayer is layer 0 in Part I)
        - sublattice: A/B index (0/1) repeated from the primitive basis

    Notes:
    - This is the baseline "no twist, no registry shift" builder used in Part I sanity checks.
    - Twists/registry are handled at the bilayer/stack level, not here.
    """
    n1, n2 = int(repeats[0]), int(repeats[1])
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")

    mat: Material = get_material(material_id)
    prim = primitive_2d(mat, vacuum=vacuum)

    # Primitive cell vectors (row-stacked). Ensure vacuum is present in z.
    cell_prim = prim.cell.copy()
    if vacuum and cell_prim[2, 2] == 0.0:
        cell_prim[2, 2] = float(vacuum)

    # Supercell vectors: scale a1 by n1 and a2 by n2 (simple rectangular tiling in a1/a2 basis).
    cell = cell_prim.copy()
    cell[0, :] = n1 * cell_prim[0, :]
    cell[1, :] = n2 * cell_prim[1, :]
    # cell[2] already contains vacuum

    # Primitive basis positions (set slab height to z0).
    basis_pos = prim.positions.copy()
    basis_pos[:, 2] = float(z0)

    basis_species = list(prim.species)
    basis_sub = np.asarray(prim.sublattice, dtype=int)

    positions = []
    species = []
    sublattice = []

    # Translation vectors in cartesian: shift = i*a1 + j*a2.
    a1 = cell_prim[0, :]
    a2 = cell_prim[1, :]

    for i in range(n1):
        for j in range(n2):
            shift = i * a1 + j * a2
            p = basis_pos + shift[None, :]
            positions.append(p)
            species.extend(basis_species)
            sublattice.extend(basis_sub.tolist())

    positions = np.vstack(positions).astype(float)
    species = list(species)
    sublattice = np.asarray(sublattice, dtype=int)

    # Wrap into cell in xy. For perfect tiling this is unnecessary, but it is harmless and robust.
    positions = wrap_positions_xy(positions, cell)

    arrays: Dict[str, np.ndarray] = {
        "layer_id": np.zeros(len(positions), dtype=int),
        "sublattice": sublattice,
    }

    meta: Dict[str, Any] = {
        "stage": stage,
        "material_id": material_id,
        "repeats": (n1, n2),
        "vacuum": float(vacuum),
        "z0": float(z0),
    }
    if job_id is not None:
        meta["job_id"] = str(job_id)

    return Structure(
        positions=positions,
        cell=cell,
        pbc=np.asarray(pbc, dtype=bool),
        species=species,
        arrays=arrays,
        meta=meta,
    )


def lattice_points_in_supercell(M: np.ndarray) -> np.ndarray:
    """
    Enumerate translation lattice points for a 2D integer supercell.

    Inputs:
    - M: (2,2) integer matrix defining the supercell in a primitive lattice basis.
         The number of unique translations is det(M) (absolute value).

    Returns:
    - pts: (det,2) fractional coordinates (u,v) in the *supercell basis*.

    Implementation notes:
    - Brute-force enumerates integer lattice points in a bounding box and keeps those that
      map into the unit square [0,1)^2 under inv(M).
    - Deduplicates with rounding and returns exactly det(M) points (or errors if not possible).

    This function is a "classic" approach but can be brittle if floating tolerances interact
    badly with a particular M; see lattice_points_by_clipping() for a more robust method.
    """
    M = np.asarray(M, int)
    det = int(round(abs(np.linalg.det(M))))
    if det <= 0:
        raise ValueError(f"Bad supercell matrix det={det} for M={M}")

    # Brute force enumerate points in a bounding box and select those in [0,1)^2.
    invM = np.linalg.inv(M.astype(float))
    pts = []
    rng = range(-det, det + 1)
    for i in rng:
        for j in rng:
            uv = invM @ np.array([i, j], float)
            if (-1e-12 <= uv[0] < 1 - 1e-12) and (-1e-12 <= uv[1] < 1 - 1e-12):
                pts.append(uv)

    pts = np.array(pts, float)

    # Must have at least det points before deduplication.
    if len(pts) < det:
        raise RuntimeError("Failed to enumerate enough lattice points.")

    # Dedupe using a tight rounding key.
    key = np.round(pts / 1e-10).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    pts = pts[idx]

    # If we didn't land on exactly det after dedupe, fall back to a stable truncation.
    if len(pts) != det:
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))][:det]

    return pts


def _wrap01(x: np.ndarray) -> np.ndarray:
    """
    Wrap values into [0, 1) elementwise.
    """
    return x - np.floor(x)


def lattice_points_by_clipping(
    *,
    cell_super: np.ndarray,   # (3,3) row-stacked supercell vectors
    a1: np.ndarray,           # primitive a1 (3,)
    a2: np.ndarray,           # primitive a2 (3,)
    nmax: int,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Robust enumeration of supercell translation lattice points by geometric clipping.

    Idea:
    - Generate many primitive translations r = i*a1 + j*a2 in a bounding box (|i|,|j| <= nmax).
    - Convert each r to *supercell fractional coords* s via cart_to_frac(r, cell_super).
    - Wrap the in-plane part into [0,1) and keep points inside the unit square.

    Returns:
    - pts: (npts, 2) supercell fractional coords (u,v) of unique translations.

    This avoids some edge cases where lattice_points_in_supercell(M) can miss points due to
    numerical tolerance issues in inv(M) for certain matrices.
    """
    from tbg_rebuild.utils.geom import cart_to_frac

    pts = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            r = i * a1 + j * a2                         # translation in cartesian (3,)
            s = cart_to_frac(r[None, :], cell_super)[0] # fractional in supercell basis (3,)
            uv = _wrap01(s[:2])

            # Keep those that are truly inside [0,1) up to tolerance.
            if (-tol <= uv[0] < 1 - tol) and (-tol <= uv[1] < 1 - tol):
                pts.append(uv)

    if not pts:
        raise RuntimeError("No lattice points found by clipping; nmax too small or bad cell.")

    pts = np.asarray(pts, float)

    # Dedupe via integer key on tol grid.
    key = np.round(pts / tol).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    pts = pts[idx]

    return pts


def build_monolayer_supercell(
    material_id: str,
    *,
    cell: np.ndarray,             # (3,3) row-stacked commensurate cell
    M: np.ndarray,                # (2,2) integer map from supercell -> primitive lattice (stored for provenance)
    vacuum: float = 24.0,
    z0: float = 0.0,
    pbc=(True, True, False),
    stage="built",
    job_id: str | None = None,
) -> Structure:
    """
    Build a monolayer that matches a provided commensurate supercell cell.

    Intended use:
    - supercell-core planning produces a commensurate in-plane cell for a target twist angle.
    - This function populates that cell with the correct number of primitive translations and basis atoms.

    How translations are generated:
    - Use lattice_points_by_clipping() (robust) to compute all distinct primitive translation
      vectors that map into the commensurate supercell.
    - Place the primitive basis at each translation and wrap into the cell.

    Sanity check:
    - We verify that the number of translations is consistent with the area ratio
      (A_super / A_prim) to catch bad enumeration early.
    """
    mat = get_material(material_id)
    prim = primitive_2d(mat, vacuum=vacuum)

    # Primitive cell (ensure vacuum is present).
    cell_prim = prim.cell.copy()
    cell_prim[2, 2] = float(vacuum)

    # Primitive in-plane vectors (cartesian).
    a1 = cell_prim[0, :].copy()
    a2 = cell_prim[1, :].copy()

    # Choose a conservative nmax that should cover the supercell parallelogram.
    # We over-shoot slightly because the clip step deduplicates and filters.
    L1 = float(np.linalg.norm(cell[:2, :2][0]))
    L2 = float(np.linalg.norm(cell[:2, :2][1]))
    lp1 = float(np.linalg.norm(a1[:2]))
    lp2 = float(np.linalg.norm(a2[:2]))
    nmax = int(np.ceil((L1 + L2) / max(lp1, lp2))) + 3

    pts = lattice_points_by_clipping(
        cell_super=np.asarray(cell, float),
        a1=a1,
        a2=a2,
        nmax=nmax,
        tol=1e-8,
    )

    # Optional sanity: translation count should approximately match the in-plane area ratio.
    from tbg_rebuild.utils.geom import cell_metrics as _cell_metrics

    A_super = float(_cell_metrics(np.asarray(cell, float))["area"])
    A_prim = float(_cell_metrics(np.asarray(cell_prim, float))["area"])
    n_expected = int(round(A_super / A_prim))
    if abs(len(pts) - n_expected) > 2:
        raise RuntimeError(
            f"Clipping produced {len(pts)} lattice translations, expected ~{n_expected} "
            f"(A_super/A_prim={A_super/A_prim:.6g})."
        )

    # Primitive basis positions (cartesian); set slab height to z0.
    basis_pos = prim.positions.copy()
    basis_pos[:, 2] = float(z0)
    basis_species = list(prim.species)
    basis_sub = np.asarray(prim.sublattice, int)

    # Target supercell in-plane vectors (cartesian).
    c1 = np.asarray(cell[0, :], float)
    c2 = np.asarray(cell[1, :], float)

    positions = []
    species = []
    sublattice = []

    # For each translation (u,v) in supercell fractional coords, shift by u*c1 + v*c2.
    for uv in pts:
        shift = uv[0] * c1 + uv[1] * c2
        p = basis_pos + shift[None, :]
        positions.append(p)
        species.extend(basis_species)
        sublattice.extend(basis_sub.tolist())

    positions = np.vstack(positions).astype(float)
    positions = wrap_positions_xy(positions, cell)

    arrays = {
        "layer_id": np.zeros(len(positions), dtype=int),
        "sublattice": np.asarray(sublattice, int),
    }
    meta = {"stage": stage, "material_id": material_id, "M": np.asarray(M, int).tolist()}
    if job_id is not None:
        meta["job_id"] = str(job_id)

    return Structure(
        positions=positions,
        cell=np.asarray(cell, float),
        pbc=np.asarray(pbc, bool),
        species=list(species),
        arrays=arrays,
        meta=meta,
    )
