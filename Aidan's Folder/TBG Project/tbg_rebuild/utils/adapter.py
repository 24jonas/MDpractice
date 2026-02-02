# tbg/utils/adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from ase import Atoms


@dataclass(slots=True)
class Structure:
    """
    Canonical NumPy-first structure container.

    Conventions:
    - positions: (N,3) Å
    - cell: (3,3) row-stacked (ASE convention) Å
    - pbc: (3,) bool
    - species: (N,) list[str]
    - arrays: dict[str, np.ndarray] each length N
    - meta: dict for provenance/stage info
    """
    positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    species: list[str]
    arrays: Dict[str, np.ndarray]
    meta: Dict[str, Any]


def structure_to_atoms(struct: Structure) -> Atoms:
    """Losslessly wrap a Structure into an ASE Atoms object (no math transformations)."""
    pos = np.asarray(struct.positions, dtype=float)
    cell = np.asarray(struct.cell, dtype=float)
    pbc = np.asarray(struct.pbc, dtype=bool)

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (N,3), got {pos.shape}")
    if cell.shape != (3, 3):
        raise ValueError(f"cell must be (3,3), got {cell.shape}")
    if pbc.shape != (3,):
        raise ValueError(f"pbc must be (3,), got {pbc.shape}")
    if len(struct.species) != len(pos):
        raise ValueError("species length must match positions length")

    atoms = Atoms(symbols=list(struct.species), positions=pos, cell=cell, pbc=tuple(bool(x) for x in pbc))

    # Attach per-atom arrays (ASE requires correct lengths)
    for k, v in (struct.arrays or {}).items():
        arr = np.asarray(v)
        if len(arr) != len(atoms):
            raise ValueError(f"atoms.arrays[{k!r}] length {len(arr)} != N {len(atoms)}")
        atoms.arrays[k] = arr.copy()

    # Store meta in Atoms.info
    atoms.info = dict(struct.meta or {})
    return atoms


def atoms_to_structure(atoms: Atoms, *, meta_override: Optional[Dict[str, Any]] = None) -> Structure:
    """
    Extract a Structure from ASE Atoms (no math transformations).

    Note: ASE Atoms may contain lots of arrays; we copy all arrays except internal ones.
    """
    pos = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell().array, dtype=float)  # row-stacked
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    species = list(atoms.get_chemical_symbols())

    skip = {"numbers", "positions"}
    arrays: Dict[str, np.ndarray] = {}
    for k, v in atoms.arrays.items():
        # Skip ASE internal arrays if desired; by default keep everything user-facing
        if k in skip:
            continue
        arrays[k] = np.asarray(v).copy()

    meta: Dict[str, Any] = dict(atoms.info or {})
    if meta_override:
        meta.update(meta_override)

    return Structure(
        positions=pos,
        cell=cell,
        pbc=pbc,
        species=species,
        arrays=arrays,
        meta=meta,
    )
