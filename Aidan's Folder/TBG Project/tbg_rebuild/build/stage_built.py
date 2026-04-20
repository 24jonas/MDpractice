# tbg_rebuild/build/stage_built.py
from __future__ import annotations
import numpy as np

from dataclasses import dataclass
from pathlib import Path

from tbg_rebuild.explore.specs import ConfigSpec
from tbg_rebuild.build.domain import plan_domain, DomainPlan
from tbg_rebuild.build.monolayer import build_monolayer
from tbg_rebuild.build.stack import build_bilayer_aa_from_spec
from tbg_rebuild.build.stack import build_bilayer_supercell_core_from_spec
from tbg_rebuild.utils.adapter import Structure, structure_to_atoms, atoms_to_structure
from tbg_rebuild.validate.pipeline import validate_or_raise
from tbg_rebuild.validate.errors import ValidationError, ValidationReport
from tbg_rebuild.visualize.quick import save_3d_views_by_layer

@dataclass(frozen=True, slots=True)
class BuiltStageResult:
    """
    Return type for the canonical "built" stage.

    - plan: the DomainPlan produced by plan_domain()
    - struct: the built Structure (NumPy-native representation)
    - report: validation report for the built structure
    - view_png: path to the saved 3D visualization (if requested)
    - ase_roundtrip_ok: whether Structure <-> ASE Atoms roundtrip preserved all data exactly
    """
    plan: DomainPlan
    struct: Structure
    report: ValidationReport
    view_png: Path
    ase_roundtrip_ok: bool




def _build_from_spec(spec: ConfigSpec, plan: DomainPlan) -> Structure:
    """
    Internal builder dispatch for the "built" stage.

    Dispatch is controlled entirely by spec.periodicity.mode:
      - "tiled":
          * 1 layer  -> build_monolayer()
          * 2 layers -> build_bilayer_aa_from_spec()
      - "supercell_core":
          * 2 layers only -> build_bilayer_supercell_core_from_spec()

    This function does NOT enforce caps; plan_domain() must be called first.
    """
    mode = spec.periodicity.mode

    if mode == "tiled":
        if len(spec.layers) == 1:
            return build_monolayer(
                material_id=spec.layers[0].material_id,
                repeats=spec.geometry.repeats,
                vacuum=spec.geometry.vacuum,
                z0=spec.geometry.z_center,
                pbc=spec.geometry.pbc,
                stage="built",
                job_id=spec.job_name,
            )
        elif len(spec.layers) == 2:
            return build_bilayer_aa_from_spec(spec, plan)
        else:
            raise ValueError(f"tiled built stage supports 1 or 2 layers only; got {len(spec.layers)}")

    elif mode == "supercell_core":
        if len(spec.layers) != 2:
            raise ValueError(f"supercell_core built stage requires exactly 2 layers; got {len(spec.layers)}")
        return build_bilayer_supercell_core_from_spec(spec, plan)

    else:
        raise ValueError(f"Unsupported periodicity mode in built stage: {mode!r}")

def _ase_roundtrip_integrity(struct: Structure) -> bool:
    """
    Verify exact Structure -> ASE Atoms -> Structure fidelity.

    This is a strict byte-for-byte style integrity check for:
    - species ordering
    - positions and cell values
    - pbc flags
    - per-atom arrays (keys and values)

    If this fails, downstream GPAW workflows should be considered unsafe.
    """
    atoms = structure_to_atoms(struct)
    struct_rt = atoms_to_structure(atoms)

    # Exact equality checks (same as smoke scripts)
    if struct.species != struct_rt.species:
        return False
    if (struct.positions.shape != struct_rt.positions.shape) or (struct.cell.shape != struct_rt.cell.shape):
        return False
    if not np.array_equal(struct.pbc, struct_rt.pbc):
        return False
    if (abs(struct.positions - struct_rt.positions).max() != 0.0) or (abs(struct.cell - struct_rt.cell).max() != 0.0):
        return False

    if set(struct.arrays.keys()) != set(struct_rt.arrays.keys()):
        return False
    for k in struct.arrays.keys():
        if not np.array_equal(struct.arrays[k], struct_rt.arrays[k]):
            return False

    return True


def run_stage_built(
    spec: ConfigSpec,
    *,
    outdir: str | Path = "runs",
    make_views: bool = False,
    check_ase_roundtrip: bool = True,
) -> BuiltStageResult:
    """
    Canonical Part I "built" stage runner.

    Pipeline:
      1) plan_domain(spec)         (caps + domain gating)
      2) _build_from_spec(spec, plan)
      3) optionally write a 3D view PNG
      4) validate_or_raise(...)
      5) optionally run strict ASE round-trip integrity check

    Guarantees:
    - The caps/domain gate happens before building atoms.
    - If make_views=True, a 3D view artifact is written before validation.
    - On validation failure, raises ValidationError with a detailed report,
      and (optionally) writes an additional failure-tagged view.
    """
    outdir = Path(outdir)
    stage_dir = outdir / spec.job_name / "built"
    stage_dir.mkdir(parents=True, exist_ok=True)

    plan = plan_domain(spec)
    if plan.status != "ok":
        raise RuntimeError(f"[{spec.job_name}] domain gate: {plan.reason}")

    # 2) Build (mode-aware dispatch)
    struct = _build_from_spec(spec, plan)

    # Mirror plan flags into struct.meta so downstream pipeline stages can read them.
    struct.meta["domain_reason"] = plan.reason
    struct.meta["corrected"] = bool(getattr(plan, "corrected", False))
    struct.meta["max_strain"] = float(getattr(plan, "max_strain", 0.0))
    if getattr(plan, "commensurate_meta", None) is not None:
        struct.meta["commensurate_meta"] = plan.commensurate_meta
                                    
    # NOTE: If a builder does not provide layer_id, we fall back to all zeros.
    # This keeps visualization and validation code robust to monolayer/tiled paths.
    layer = struct.arrays.get("layer_id")
    if layer is None:
        struct.arrays["layer_id"] = np.zeros(len(struct.species), dtype=int)

    # 3) Save a 3D view (optional). This is the primary human-inspection artifact.
    if make_views:
        view_png = save_3d_views_by_layer(
            struct.positions,
            struct.arrays.get("layer_id", [0] * len(struct.species)),
            stage_dir,
            prefix="built_numpy",
            max_points=50_000,
        )
    else:
        view_png = stage_dir / "built_numpy_3d_layer.png"

    # 4) Validation gate (raise on fail, but keep artifacts)
    try:
        report = validate_or_raise(struct, spec, stage="built")
    except ValidationError:
        # Write a failure-tagged view (helps browsing run dirs)
        if make_views:
            save_3d_views_by_layer(
                struct.positions,
                struct.arrays.get("layer_id", [0] * len(struct.species)),
                stage_dir,
                prefix="FAILED_built_numpy",
                max_points=50_000,
            )
        raise

    # 5) Optional ASE round-trip
    ase_ok = True
    if check_ase_roundtrip:
        ase_ok = _ase_roundtrip_integrity(struct)
        if not ase_ok:
            raise RuntimeError(
                "ASE round-trip integrity failed. "
                "This indicates adapter or array propagation problems; do not proceed to GPAW."
            )

    return BuiltStageResult(plan=plan, struct=struct, report=report, view_png=view_png, ase_roundtrip_ok=ase_ok)

def build_from_spec(spec: ConfigSpec, plan: DomainPlan) -> Structure:
    """
    Public helper used by the unified pipeline (pipeline_part1).

    This mirrors run_stage_built()'s build behavior, but does NOT:
    - create directories
    - create views
    - run validation
    - run ASE roundtrip checks

    It does, however, copy key DomainPlan flags into struct.meta to keep the
    pipeline and stage-runner outputs consistent.
    """
    if plan.status != "ok":
        raise RuntimeError(f"[{spec.job_name}] domain gate: {plan.reason}")

    struct = _build_from_spec(spec, plan)

    # mirror plan flags into struct.meta (so pipeline + stage runner match)
    struct.meta["domain_reason"] = plan.reason
    struct.meta["corrected"] = bool(getattr(plan, "corrected", False))
    struct.meta["max_strain"] = float(getattr(plan, "max_strain", 0.0))
    if getattr(plan, "commensurate_meta", None) is not None:
        struct.meta["commensurate_meta"] = plan.commensurate_meta

    return struct
