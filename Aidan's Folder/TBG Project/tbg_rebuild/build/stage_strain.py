# tbg_rebuild/build/stage_strain.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tbg_rebuild.explore.specs import ConfigSpec
from tbg_rebuild.build.stage_built import run_stage_built
from tbg_rebuild.strain.models import directed_small_strain, random_small_strain
from tbg_rebuild.strain.apply import apply_inplane_deformation
from tbg_rebuild.validate.pipeline import validate_or_raise
from tbg_rebuild.visualize.quick import save_3d_views_by_layer


@dataclass(frozen=True, slots=True)
class StrainStageResult:
    """
    Return type for the canonical "strained" stage.

    - built_dir: directory where the built-stage artifacts live
    - strain_dir: directory where strained-stage artifacts live
    - strain_info: small dict describing the strain parameters that were applied
    - struct_strained: the strained Structure (NumPy-native representation)
    - report: validation report for the strained structure
    - views: dict of view-name -> image path (e.g. {"layer": "...png"})
    """
    built_dir: Path
    strain_dir: Path
    strain_info: dict
    struct_strained: object
    report: object
    views: dict[str, Path]


def run_stage_strain(
    spec: ConfigSpec,
    *,
    outdir: str | Path = "runs",
    make_views: bool = False,
    check_ase_roundtrip: bool = True,
) -> StrainStageResult:
    """
    Part I strain stage: built → apply strain → validate (+ optional views).

    What this does:
      1) Runs the built stage first (run_stage_built) to guarantee a valid Structure.
      2) Constructs a strain "plan" from spec.strain (directed_small or random_small).
      3) Applies the in-plane deformation affinely (xy only) to produce a new Structure.
      4) Writes a 3D view of the strained configuration.
      5) Validates the strained structure (same invariants as built should still hold
         for small strains: no duplicates, sane distances, coordination, etc.).

    Requirements:
      - spec.strain.enabled must be True
      - spec.strain.model in {"directed_small","random_small"}
      - spec.strain.params contains required parameters for the chosen model
    """
    outdir = Path(outdir)
    job_dir = outdir / spec.job_name
    built_dir = job_dir / "built"
    strain_dir = job_dir / "strained"
    strain_dir.mkdir(parents=True, exist_ok=True)

    # Ensure we have a valid built structure first (domain/caps + validation gate).
    built = run_stage_built(spec, outdir=outdir, make_views=make_views, check_ase_roundtrip=check_ase_roundtrip)

    if not spec.strain.enabled:
        raise ValueError("spec.strain.enabled is False; refusing to run strain stage.")

    model = spec.strain.model
    params = dict(spec.strain.params or {})

    # Build strain plan (a small in-plane deformation gradient).
    if model == "directed_small":
        # required: eps, theta_deg; optional: poisson
        plan = directed_small_strain(
            eps=float(params["eps"]),
            theta_deg=float(params["theta_deg"]),
            poisson=float(params.get("poisson", 0.165)),
        )
    elif model == "random_small":
        # required: eps_max; optional: shear_max; optional: seed
        plan = random_small_strain(
            eps_max=float(params["eps_max"]),
            shear_max=(float(params["shear_max"]) if "shear_max" in params else None),
            seed=int(params.get("seed", 0)),
        )
    else:
        raise ValueError(f"Unsupported strain model: {model!r}")

    # Apply strain (affine in-plane deformation).
    strained = apply_inplane_deformation(built.struct, plan.F2)
    strained.meta["strain_model"] = plan.model
    strained.meta["strain_info"] = plan.info

    # Always write strain views for quick inspection.
    views = {}
    views["layer"] = save_3d_views_by_layer(
        strained.positions,
        strained.arrays["layer_id"],
        strain_dir,
        prefix="strained_numpy",
    )

    # Gate again after strain (same invariants should still hold for small strains).
    report = validate_or_raise(strained, spec, stage="strained")

    return StrainStageResult(
        built_dir=built_dir,
        strain_dir=strain_dir,
        strain_info=plan.info,
        struct_strained=strained,
        report=report,
        views=views,
    )
