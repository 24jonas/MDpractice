# tbg_rebuild/materials/catalog.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass(frozen=True, slots=True)
class Material:
    """
    Minimal material definition for 2D hexagonal sheets.

    This is a *geometry-only* description used in Part I for structure generation.
    No electronic parameters appear here.

    Conventions:
    - `a` is the in-plane hex lattice constant (Å).
    - `basis_frac` are (u, v) fractional coordinates in the 2D primitive cell.
    - `basis_species` lists the chemical symbol for each basis site.
    - `basis_sublattice` labels basis sites as A/B (0/1), preserved through tiling
      and supercell construction for visualization and later analysis.
    """
    id: str
    formula: str
    lattice: str                     # currently only "hex" is supported
    a: float                         # Å
    basis_frac: np.ndarray           # shape (nbasis, 2)
    basis_species: List[str]         # length nbasis
    basis_sublattice: List[int]      # length nbasis, values 0/1
    default_interlayer: float        # Å, typical bilayer spacing
    tags: Dict[str, str] = field(default_factory=dict)


def _as_basis(basis_frac: list[list[float]]) -> np.ndarray:
    """
    Validate and convert a list-of-lists basis into a (nbasis, 2) float array.
    """
    arr = np.array(basis_frac, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"basis_frac must be shape (nbasis,2), got {arr.shape}")
    return arr


# --------------------------------------------------------------------
# Built-in material library
# --------------------------------------------------------------------
#
# Canonical primitive bases for honeycomb lattices in hex coordinates.
# Many equivalent choices exist; internal consistency is what matters.
#
# Fractional coordinates are given in the primitive (a1, a2) basis.
#
MATLIB: Dict[str, Material] = {
    "graphene": Material(
        id="graphene",
        formula="C2",
        lattice="hex",
        a=2.46,  # Å (standard graphene lattice constant)
        basis_frac=_as_basis([[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]),
        basis_species=["C", "C"],
        basis_sublattice=[0, 1],
        default_interlayer=3.35,
        tags={"notes": "Graphene primitive honeycomb basis; a ≈ 2.46 Å."},
    ),
    "hbn": Material(
        id="hbn",
        formula="BN",
        lattice="hex",
        a=2.50,  # Å (typical approximate value)
        basis_frac=_as_basis([[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]),
        basis_species=["B", "N"],
        basis_sublattice=[0, 1],
        default_interlayer=3.33,
        tags={"notes": "h-BN honeycomb basis with B/N on A/B sites."},
    ),
}


def list_materials() -> list[str]:
    """
    Return a sorted list of registered material IDs.
    """
    return sorted(MATLIB.keys())


def get_material(material_id: str) -> Material:
    """
    Retrieve a Material definition by ID.

    Raises:
        KeyError if the material is not registered.
    """
    try:
        return MATLIB[material_id]
    except KeyError as e:
        raise KeyError(
            f"Unknown material_id={material_id!r}. Options: {list_materials()}"
        ) from e


def register_material(material: Material, *, overwrite: bool = False) -> None:
    """
    Register a new material definition at runtime.

    This is intended for experimentation or downstream extensions
    (e.g., strained lattices, alternative basis choices).

    Parameters:
        material:
            Material object to register.
        overwrite:
            If False (default), raises if the ID already exists.
    """
    if (material.id in MATLIB) and (not overwrite):
        raise KeyError(
            f"Material {material.id!r} already exists. Use overwrite=True to replace."
        )
    MATLIB[material.id] = material
