# tbg_rebuild/explore/caps.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from tbg_rebuild.explore.specs import ConfigSpec

from tbg_rebuild.materials.catalog import get_material
from tbg_rebuild.materials.primitives import hex_cell_rows
from tbg_rebuild.utils.geom import cell_metrics as _cell_metrics  # used to compute primitive-cell area


Decision = Literal["ok", "skip"]


@dataclass(frozen=True, slots=True)
class CapDecision:
    """
    Result of applying resource/sanity caps before building atoms.

    - decision="ok": proceed with build
    - decision="skip": do not build; `reason` explains why

    N_est is an estimated atom count used only for gating (not a guarantee).
    downsample_views suggests whether to downsample visualization outputs for large systems.
    """
    decision: Decision
    reason: str
    N_est: int
    downsample_views: bool


def estimate_atoms_tiled(spec: ConfigSpec) -> int:
    """
    Exact atom count estimator for tiled mode.

    For tiled periodicity, total atoms are:
        N = (sum over layers of nbasis(layer)) * (n1 * n2)

    Returns 0 for invalid repeats.
    """
    n1, n2 = spec.geometry.repeats
    if n1 <= 0 or n2 <= 0:
        return 0

    # Tiled mode can support heterostructures: sum basis sizes across layers.
    nbasis_total = 0
    for layer in spec.layers:
        mat = get_material(layer.material_id)
        nbasis_total += len(mat.basis_species)

    return int(nbasis_total * n1 * n2)


def estimate_atoms(spec: ConfigSpec, *, cell_metrics: Optional[dict] = None) -> int:
    """
    Unified atom-count estimator based on periodicity.mode.

    - tiled: exact count from repeats and basis sizes
    - supercell_core: estimated from (supercell area / primitive area)

    For supercell_core, callers must provide `cell_metrics` for the planned supercell
    (must include an 'area' entry).
    """
    mode = spec.periodicity.mode

    if mode == "tiled":
        return estimate_atoms_tiled(spec)

    if mode == "supercell_core":
        if cell_metrics is None:
            # Domain planning should supply metrics; treat missing metrics as invalid.
            return 0
        return estimate_atoms_supercell_core(spec, cell_metrics=cell_metrics)

    # Unknown mode => upstream should skip.
    return 0


def estimate_atoms_supercell_core(spec: ConfigSpec, *, cell_metrics: dict) -> int:
    """
    Estimate atoms for supercell_core mode using area ratio.

    We approximate the number of primitive cells inside the supercell as:
        n_prim ≈ round(A_super / A_prim)

    Then total atoms are:
        N ≈ sum_layer nbasis(layer) * n_prim(layer)

    Assumptions:
      - hex material with a well-defined primitive cell
    """
    A_super = float(cell_metrics.get("area", 0.0))
    if A_super <= 0:
        return 0

    total = 0
    for layer in spec.layers:
        mat = get_material(layer.material_id)

        # Build a primitive cell consistent with the material and vacuum choice,
        # then compute its in-plane area.
        prim = hex_cell_rows(mat.a, cell_z=spec.geometry.vacuum)
        A_prim = float(_cell_metrics(prim)["area"])
        if A_prim <= 0:
            return 0

        n_prim = int(round(A_super / A_prim))
        if n_prim <= 0:
            return 0

        total += len(mat.basis_species) * n_prim

    return int(total)


def enforce_caps(spec: ConfigSpec, *, cell_metrics: Optional[dict] = None) -> CapDecision:
    """
    Apply resource/sanity caps before building atoms.

    This is the main gate used by planning/stages. It:
      1) rejects unsupported periodicity modes,
      2) estimates atom count (mode-specific),
      3) checks max_atoms and (optionally) max_cell_len,
      4) suggests view downsampling for large builds.

    If `cell_metrics` is provided, it is also used for the max_cell_len gate and for
    supercell_core atom estimation.
    """
    caps = spec.periodicity.caps
    mode = spec.periodicity.mode

    # Gate unsupported modes early (avoids silent N_est=0 weirdness downstream).
    if mode not in ("tiled", "supercell_core"):
        return CapDecision("skip", f"Unsupported periodicity mode: {mode!r}", N_est=0, downsample_views=True)

    N_est = estimate_atoms(spec, cell_metrics=cell_metrics)

    # Reject invalid estimates (usually invalid spec parameters or missing metrics).
    if N_est <= 0:
        if mode == "supercell_core":
            return CapDecision(
                "skip",
                "Invalid supercell-core estimate (need planned cell_metrics with area).",
                N_est=N_est,
                downsample_views=False,
            )
        return CapDecision("skip", "Invalid repeats or empty estimate.", N_est=N_est, downsample_views=False)

    # Atom-count cap.
    if N_est > caps.max_atoms:
        return CapDecision(
            "skip",
            f"Estimated atoms {N_est} exceeds cap max_atoms={caps.max_atoms}.",
            N_est=N_est,
            downsample_views=True,
        )

    # Optional cell-length cap (requires cell_metrics from plan_domain).
    if caps.max_cell_len is not None and cell_metrics is not None:
        max_len = max(cell_metrics.get("|a1|", 0.0), cell_metrics.get("|a2|", 0.0))
        if max_len > float(caps.max_cell_len):
            return CapDecision(
                "skip",
                f"Cell length {max_len:.3f} Å exceeds cap max_cell_len={caps.max_cell_len}.",
                N_est=N_est,
                downsample_views=(N_est > caps.downsample_views_above),
            )

    downsample = (N_est > caps.downsample_views_above)
    return CapDecision("ok", "Within caps.", N_est=N_est, downsample_views=downsample)
