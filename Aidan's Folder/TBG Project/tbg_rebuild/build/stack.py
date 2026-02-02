# tbg_rebuild/build/stack.py
from __future__ import annotations

from typing import Dict, Any

import numpy as np

from tbg_rebuild.utils.geom import wrap_positions_xy
from tbg_rebuild.explore.specs import ConfigSpec
from tbg_rebuild.build.domain import DomainPlan
from tbg_rebuild.build.monolayer import build_monolayer
from tbg_rebuild.utils.adapter import Structure
from tbg_rebuild.build.monolayer import build_monolayer_supercell


def build_bilayer_aa_from_spec(spec: ConfigSpec, plan: DomainPlan) -> Structure:
    """
    Build a simple AA-stacked bilayer in tiled mode.

    This is the "baseline bilayer" path used for Part I:
    - No twist (theta = 0 for both layers)
    - No registry shift (AA stacking)
    - Uses the tiled domain (repeats-based supercell)

    Procedure:
    1) Build two identical tiled monolayers in the same in-plane cell.
    2) Place them at z = z_center ± interlayer_distance/2.
    3) Concatenate atoms and assign per-atom arrays:
       - layer_id: 0 for bottom layer, 1 for top layer
       - sublattice: carried through from the monolayer basis tiling

    Assumptions (v1):
    - spec.layers has length 2
    - both layers use the same material_id
    """
    if plan.status != "ok":
        raise ValueError(f"DomainPlan must be ok; got status={plan.status!r} reason={plan.reason!r}")
    if len(spec.layers) != 2:
        raise ValueError(f"Bilayer spec must have 2 layers, got {len(spec.layers)}")

    mat0 = spec.layers[0].material_id
    mat1 = spec.layers[1].material_id
    if mat0 != mat1:
        raise ValueError(
            "v1 build_bilayer_aa_from_spec expects identical materials for both layers. "
            f"Got {mat0!r} and {mat1!r}."
        )

    n1, n2 = spec.geometry.repeats
    vacuum = float(spec.geometry.vacuum)
    pbc = spec.geometry.pbc
    d = float(spec.geometry.interlayer_distance)
    zc = float(spec.geometry.z_center)

    # Layer z-positions (symmetric around z_center).
    z0 = zc - 0.5 * d
    z1 = zc + 0.5 * d

    # Build each layer (monolayer builder handles basis + tiling + sublattice).
    layer0 = build_monolayer(
        material_id=mat0,
        repeats=(n1, n2),
        vacuum=vacuum,
        z0=z0,
        pbc=pbc,
        stage="built",
        job_id=spec.job_name,
    )
    layer1 = build_monolayer(
        material_id=mat0,
        repeats=(n1, n2),
        vacuum=vacuum,
        z0=z1,
        pbc=pbc,
        stage="built",
        job_id=spec.job_name,
    )

    # Concatenate per-atom data.
    positions = np.vstack([layer0.positions, layer1.positions])
    species = list(layer0.species) + list(layer1.species)

    sub0 = np.asarray(layer0.arrays.get("sublattice"), dtype=int)
    sub1 = np.asarray(layer1.arrays.get("sublattice"), dtype=int)
    sublattice = np.concatenate([sub0, sub1], axis=0)

    layer_id = np.concatenate(
        [np.zeros(len(layer0.species), dtype=int), np.ones(len(layer1.species), dtype=int)],
        axis=0,
    )

    arrays: Dict[str, np.ndarray] = {
        "layer_id": layer_id,
        "sublattice": sublattice,
    }

    meta: Dict[str, Any] = {
        "stage": "built",
        "job_id": spec.job_name,
        "stack": "bilayer_aa",
        "material_id": mat0,
        "repeats": (n1, n2),
        "interlayer_distance": d,
        "z_center": zc,
        "vacuum": vacuum,
        "domain_reason": plan.reason,
        "N_est": plan.N_est,
    }

    return Structure(
        positions=positions,
        cell=layer0.cell.copy(),  # both layers share the same cell
        pbc=np.asarray(pbc, dtype=bool),
        species=species,
        arrays=arrays,
        meta=meta,
    )


def _rot2(theta_deg: float) -> np.ndarray:
    """
    2D rotation matrix in degrees.

    NOTE: This helper is currently unused in the supercell-core path.
    It is kept as a utility for future "explicit rotation" builders.
    """
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]], float)


def _apply_F2(positions: np.ndarray, F2: np.ndarray) -> np.ndarray:
    """
    Apply a 2x2 in-plane deformation gradient to the xy components of positions.

    - positions: (N,3) cartesian Å
    - F2: (2,2) deformation gradient acting on row-vectors (xy @ F2^T)

    Returns a new array; does not modify input in-place.
    """
    xy = positions[:, :2] @ np.asarray(F2, float).T
    out = positions.copy()
    out[:, :2] = xy
    return out


def build_bilayer_supercell_core_from_spec(spec: ConfigSpec, plan: DomainPlan) -> Structure:
    """
    Build a bilayer using a supercell-core commensurate plan.

    Inputs:
    - plan.cell: the commensurate supercell vectors (row-stacked)
    - plan.commensurate_meta:
        - layer_Ms: integer (2,2) supercell matrices for each layer
        - strain_tensors (optional): small in-plane strain tensors from supercell-core

    Behavior:
    - Builds each layer with build_monolayer_supercell() using the planned cell + per-layer M.
    - If strain_tensors are provided, applies them as in-plane deformation gradients:
        F = I + strain_tensor
      (either to the top layer only, or to both if both are provided).
    - Does NOT apply an additional rigid rotation by theta here; supercell-core already
      selected a commensurate configuration consistent with the search angle.

    Output:
    - Structure with concatenated positions/species
    - arrays:
        - layer_id: 0/1
        - sublattice: concatenated from each monolayer
    - meta includes plan correction flags and the original commensurate_meta payload
    """
    if plan.status != "ok":
        raise ValueError(f"DomainPlan must be ok; got status={plan.status!r} reason={plan.reason!r}")
    if len(spec.layers) != 2:
        raise ValueError(f"supercell-core bilayer build requires exactly 2 layers; got {len(spec.layers)}")

    meta = plan.commensurate_meta or {}

    # supercell-core planning should provide one (or two) integer supercell matrices.
    layer_Ms = meta.get("layer_Ms")
    if layer_Ms is None:
        raise RuntimeError("Missing layer_Ms from supercell-core result; cannot build layers.")

    # Supported formats:
    # - [M_sub, M_top]
    # - [M] (then reuse the same matrix for both layers)
    layer_Ms = list(layer_Ms)
    if len(layer_Ms) == 1:
        M0 = np.array(layer_Ms[0], dtype=int)
        M1 = np.array(layer_Ms[0], dtype=int)
    else:
        M0 = np.array(layer_Ms[0], dtype=int)
        M1 = np.array(layer_Ms[1], dtype=int)

    vacuum = float(spec.geometry.vacuum)
    d = float(spec.geometry.interlayer_distance)
    zc = float(spec.geometry.z_center)
    z0 = zc - 0.5 * d
    z1 = zc + 0.5 * d

    # Build both layers in the planned commensurate cell.
    layer0 = build_monolayer_supercell(
        spec.layers[0].material_id,
        cell=plan.cell,
        M=M0,
        vacuum=vacuum,
        z0=z0,
        pbc=spec.geometry.pbc,
        stage="built",
        job_id=spec.job_name,
    )
    layer1 = build_monolayer_supercell(
        spec.layers[1].material_id,
        cell=plan.cell,
        M=M1,
        vacuum=vacuum,
        z0=z1,
        pbc=spec.geometry.pbc,
        stage="built",
        job_id=spec.job_name,
    )

    # Apply solver-provided strain tensors if present.
    # Convention: supercell-core typically returns a tensor for the "top" layer (substrate fixed).
    # We interpret each as a small strain tensor and apply F = I + strain_tensor in the xy plane.
    strain_tensors = meta.get("strain_tensors", None)
    if strain_tensors is not None:
        strain_tensors = np.asarray(strain_tensors, float)

        if strain_tensors.shape[0] == 1:
            F_top = np.eye(2) + strain_tensors[0]
            layer1.positions = _apply_F2(layer1.positions, F_top)
            layer1.positions = wrap_positions_xy(layer1.positions, plan.cell)

        elif strain_tensors.shape[0] >= 2:
            F0 = np.eye(2) + strain_tensors[0]
            F1 = np.eye(2) + strain_tensors[1]
            layer0.positions = _apply_F2(layer0.positions, F0)
            layer1.positions = _apply_F2(layer1.positions, F1)
            layer0.positions = wrap_positions_xy(layer0.positions, plan.cell)
            layer1.positions = wrap_positions_xy(layer1.positions, plan.cell)

    # Tag each layer explicitly.
    layer0.arrays["layer_id"] = np.zeros(len(layer0.positions), dtype=int)
    layer1.arrays["layer_id"] = np.ones(len(layer1.positions), dtype=int)

    # Concatenate into a single Structure.
    positions = np.vstack([layer0.positions, layer1.positions])
    species = list(layer0.species) + list(layer1.species)

    arrays: Dict[str, np.ndarray] = {
        "layer_id": np.concatenate([layer0.arrays["layer_id"], layer1.arrays["layer_id"]], axis=0),
        "sublattice": np.concatenate([layer0.arrays["sublattice"], layer1.arrays["sublattice"]], axis=0),
    }

    outmeta: Dict[str, Any] = dict(layer0.meta)
    outmeta.update(
        {
            "stack": "bilayer_supercell_core",
            "corrected": bool(plan.corrected),
            "max_strain": float(plan.max_strain),
            "commensurate_meta": meta,
            "domain_reason": plan.reason,
        }
    )

    return Structure(
        positions=wrap_positions_xy(positions, plan.cell),
        cell=plan.cell.copy(),
        pbc=np.asarray(spec.geometry.pbc, dtype=bool),
        species=species,
        arrays=arrays,
        meta=outmeta,
    )
