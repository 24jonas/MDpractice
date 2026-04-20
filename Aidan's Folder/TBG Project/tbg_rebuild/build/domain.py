# tbg_rebuild/build/domain.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from tbg_rebuild.build.supercell_core_adapter import plan_with_supercell_core
from tbg_rebuild.materials.catalog import get_material
from tbg_rebuild.materials.primitives import hex_cell_rows
from tbg_rebuild.utils.geom import cell_metrics
from tbg_rebuild.explore.specs import ConfigSpec
from tbg_rebuild.explore.caps import enforce_caps, CapDecision


@dataclass(frozen=True, slots=True)
class DomainPlan:
    """
    Result of the "domain planning" step.

    This is the *only* place where we decide the periodic simulation cell (in-plane vectors + vacuum).
    Building atoms comes later.

    The plan also carries an atom-count estimate (N_est) and a visualization hint (downsample_views),
    both computed via enforce_caps() to prevent accidentally constructing huge systems.

    Notes on supercell-core mode:
    - supercell-core may apply a small strain / correction to achieve commensurability near a target
      twist angle. We record that in (corrected, max_strain) and pass through any solver metadata.
    """
    status: Literal["ok", "skip"]
    reason: str
    cell: np.ndarray            # (3, 3) row-stacked cell vectors; cell[2,2] contains vacuum
    N_est: int                  # estimated atom count used for caps + logging
    downsample_views: bool      # visualization policy (not a build gate)

    corrected: bool = False     # supercell-core: whether commensurating strain was applied
    max_strain: float = 0.0     # supercell-core: maximum strain magnitude reported by solver
    commensurate_meta: Optional[dict] = None  # supercell-core: solver metadata (M, layer_Ms, etc.)


def plan_domain(spec: ConfigSpec) -> DomainPlan:
    """
    Choose the simulation cell for this spec, then enforce caps before building atoms.

    Supported periodicity modes:
      - "tiled":
          Use repeats=(n1,n2) to tile the material primitive cell along a1/a2.
      - "supercell_core":
          Use supercell-core to find a nearby commensurate bilayer supercell around a target twist.
          The returned cell is commensurate by construction, but may require a small strain correction.

    Returns:
      DomainPlan with status="ok" if within caps, otherwise status="skip" with a reason string.

    Important:
      enforce_caps() may require precomputed cell metrics (e.g. area, |a1|, |a2|), so we compute
      metrics from the planned cell before asking caps to decide.
    """
    mode = spec.periodicity.mode

    if mode == "tiled":
        n1, n2 = spec.geometry.repeats
        if n1 <= 0 or n2 <= 0:
            return DomainPlan("skip", f"Invalid repeats={spec.geometry.repeats}", np.zeros((3, 3)), 0, True)

        # Build the primitive hex cell for the layer-0 material and scale a1/a2 by (n1,n2).
        mat0 = get_material(spec.layers[0].material_id)
        cell_prim = hex_cell_rows(mat0.a, cell_z=spec.geometry.vacuum)

        cell = cell_prim.copy()
        cell[0, :] = n1 * cell_prim[0, :]
        cell[1, :] = n2 * cell_prim[1, :]

        # Caps: estimate atom count + optional geometry gates (e.g. max cell length).
        metrics = cell_metrics(cell)
        cap: CapDecision = enforce_caps(spec, cell_metrics=metrics)

        if cap.decision == "skip":
            return DomainPlan("skip", cap.reason, cell, cap.N_est, True)

        return DomainPlan("ok", cap.reason, cell, cap.N_est, cap.downsample_views)

    elif mode == "supercell_core":
        # Ask supercell-core for a commensurate supercell near the target twist angle.
        # This is expected to be used for bilayers (graphene on graphene in v1).
        sc_plan = plan_with_supercell_core(spec)

        # Caps: for supercell-core, estimate uses planned cell metrics (area scaling).
        metrics = cell_metrics(sc_plan.cell)
        cap: CapDecision = enforce_caps(spec, cell_metrics=metrics)

        if cap.decision == "skip":
            return DomainPlan(
                "skip",
                cap.reason,
                sc_plan.cell,
                cap.N_est,
                True,
                corrected=sc_plan.corrected,
                max_strain=sc_plan.max_strain,
                commensurate_meta=sc_plan.meta,
            )

        return DomainPlan(
            "ok",
            f"supercell-core: max_strain={sc_plan.max_strain:.6g} corrected={sc_plan.corrected}",
            sc_plan.cell,
            cap.N_est,
            cap.downsample_views,
            corrected=sc_plan.corrected,
            max_strain=sc_plan.max_strain,
            commensurate_meta=sc_plan.meta,
        )

    else:
        # Keep this as a hard skip so unsupported modes fail fast and visibly.
        return DomainPlan("skip", f"Unsupported periodicity mode: {mode!r}", np.zeros((3, 3)), 0, True)
