# tbg/materials/primitives.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tbg_rebuild.materials.catalog import Material


@dataclass(frozen=True, slots=True)
class Primitive2D:
    """
    Minimal 2D primitive cell description used for structure construction.

    This is intentionally smaller than the full `Structure` object:
    - `cell` is an ASE-style row-stacked (3x3) cell matrix in Å.
    - `positions` are Cartesian (nbasis, 3) in Å, with z initially set to 0.
    - `species` and `sublattice` track per-basis metadata used downstream.
    - `pbc` defaults to slab-style periodicity: (True, True, False).
    """
    cell: np.ndarray           # (3,3) row-stacked (ASE convention)
    positions: np.ndarray      # (nbasis,3) Cartesian Å
    species: list[str]         # length nbasis
    sublattice: np.ndarray     # (nbasis,) integer labels (e.g., A/B = 0/1)
    pbc: tuple[bool, bool, bool]


def hex_cell_rows(a: float, *, cell_z: float = 0.0) -> np.ndarray:
    """
    Construct the 3D hexagonal primitive cell (row-stacked, ASE convention).

    In-plane vectors:
      a1 = (a, 0, 0)
      a2 = (a/2, (sqrt(3)/2)*a, 0)

    Out-of-plane vector:
      a3 = (0, 0, cell_z)

    Notes:
    - `cell_z` may be left as 0.0 and set later by higher-level build routines.
    - Units are Å throughout.
    """
    a = float(a)
    cell = np.zeros((3, 3), dtype=float)
    cell[0, :] = (a, 0.0, 0.0)
    cell[1, :] = (0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0)
    cell[2, :] = (0.0, 0.0, float(cell_z))
    return cell


def frac_to_cart_2d(frac_uv: np.ndarray, cell_rows: np.ndarray) -> np.ndarray:
    """
    Convert 2D fractional coordinates (u, v) into Cartesian xy coordinates.

    Convention:
      r_xy = u * a1_xy + v * a2_xy

    Parameters:
      frac_uv:
        Array of shape (n, 2) containing (u, v) fractional coordinates.
      cell_rows:
        Row-stacked cell matrix (3x3). Only the xy parts of a1/a2 are used.

    Returns:
      Array of shape (n, 2) containing Cartesian xy positions in Å.
    """
    uv = np.asarray(frac_uv, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"frac_uv must be shape (n,2), got {uv.shape}")

    a1 = cell_rows[0, :2]
    a2 = cell_rows[1, :2]
    xy = uv[:, 0:1] * a1[None, :] + uv[:, 1:2] * a2[None, :]
    return xy


def primitive_2d(material: Material, *, vacuum: float = 0.0) -> Primitive2D:
    """
    Build a `Primitive2D` from a `Material` definition.

    - Positions are placed at z=0.
    - If `vacuum` is provided, it becomes the cell z-length (Å).
    - PBC is slab-style by default: (True, True, False).

    This function does not perform any tiling, twisting, registry shifts, or strain.
    Those operations are handled by higher-level build stages.
    """
    if material.lattice != "hex":
        raise ValueError(f"Only hex supported in v1; got {material.lattice!r}")

    cell = hex_cell_rows(material.a, cell_z=float(vacuum) if vacuum else 0.0)
    xy = frac_to_cart_2d(material.basis_frac, cell)

    pos = np.zeros((xy.shape[0], 3), dtype=float)
    pos[:, :2] = xy

    sub = np.asarray(material.basis_sublattice, dtype=int)

    return Primitive2D(
        cell=cell,
        positions=pos,
        species=list(material.basis_species),
        sublattice=sub,
        pbc=(True, True, False),
    )
