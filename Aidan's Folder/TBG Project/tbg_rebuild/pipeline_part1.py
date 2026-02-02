# tbg_rebuild/pipeline_part1.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tbg_rebuild.explore.specs import ConfigSpec
from tbg_rebuild.build.domain import plan_domain, DomainPlan
from tbg_rebuild.build.stage_built import build_from_spec  # or expose a public build function
from tbg_rebuild.strain.models import directed_small_strain, random_small_strain
from tbg_rebuild.strain.apply import apply_inplane_deformation
from tbg_rebuild.validate.pipeline import run_validation, validate_or_raise
from tbg_rebuild.validate.errors import ValidationReport
from tbg_rebuild.visualize.quick import save_3d_views_by_layer
from tbg_rebuild.utils.adapter import Structure


@dataclass(frozen=True, slots=True)
class Part1Artifacts:
    """
    Paths to filesystem artifacts produced by the Part I pipeline.

    outdir:
      Per-job output directory (e.g. out_part1/<job_name>/).

    built_view / strained_view:
      Optional PNGs for quick visual sanity checking, written only if:
        - spec.output_policy.make_views is True, and
        - the Structure has a "layer_id" per-atom array (used for coloring/grouping).
    """
    outdir: Path
    built_view: Optional[Path]
    strained_view: Optional[Path]


@dataclass(frozen=True, slots=True)
class Part1Result:
    """
    In-memory results returned from Part I.

    spec:
      The input ConfigSpec (fully defines the run).

    plan:
      DomainPlan produced by plan_domain() (cell choice + caps gate outcome).

    built / strained:
      Structure objects containing positions/cell/species/arrays/meta.
      "strained" is None if strain was not enabled.

    report_built / report_strained:
      Validation reports for the corresponding Structures.
    """
    spec: ConfigSpec
    plan: DomainPlan
    built: Structure
    strained: Optional[Structure]
    report_built: ValidationReport
    report_strained: Optional[ValidationReport]
    artifacts: Part1Artifacts


def run_part1(
    spec: ConfigSpec,
    *,
    outdir: Path = Path("out_part1"),
    strict: bool = True,
) -> Part1Result:
    """
    Run the unified "Part I" pipeline:

      1) plan_domain(spec)
         - chooses periodic cell/domain from spec.periodicity.mode
         - enforces caps early (max atoms, optional max cell length)

      2) build_from_spec(spec, plan)
         - constructs a Structure (monolayer or bilayer) consistent with the planned domain

      3) validate built Structure
         - strict=True: raise ValidationError on failure
         - strict=False: return a report the caller can inspect

      4) optionally apply a *synthetic* in-plane strain (for pipeline testing)
         - only runs if spec.strain.enabled is True and spec.strain.model != "none"
         - validates again after strain

      5) optionally write quick 3D PNG views (built and strained)
         - controlled by spec.output_policy.make_views
         - requires "layer_id" array to exist for sensible coloring

    This stage is meant to be a lightweight correctness gate before doing any
    expensive ASE/GPAW relaxations on a cluster.
    """
    # Each job writes into out_part1/<job_name> by default.
    outdir = Path(outdir) / spec.job_name
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 1) Domain planning / caps gate ---
    plan = plan_domain(spec)
    if plan.status != "ok":
        # Plan status is used as a gate: if it's not ok, do not proceed to building atoms.
        raise RuntimeError(f"Domain planning failed: {plan.reason}")

    # --- 2) Build (mode-aware in stage_built) ---
    built = build_from_spec(spec, plan)

    # Mirror plan metadata into the output Structure for downstream consumers.
    built.meta["domain_reason"] = plan.reason
    built.meta["corrected"] = bool(getattr(plan, "corrected", False))
    built.meta["max_strain"] = float(getattr(plan, "max_strain", 0.0))
    if getattr(plan, "commensurate_meta", None) is not None:
        built.meta["commensurate_meta"] = plan.commensurate_meta

    # --- 3) Validate built ---
    if strict:
        report_built = validate_or_raise(built, spec, stage="built")
    else:
        report_built = run_validation(built, spec, stage="built")

    # --- 4) Built view (optional) ---
    built_view = None
    if spec.output_policy.make_views and "layer_id" in built.arrays:
        built_view = save_3d_views_by_layer(
            positions=built.positions,
            layer_id=built.arrays["layer_id"],
            outdir=outdir,
            prefix=f"{spec.job_name}_built",
            max_points=50_000,
        )

    # --- 5) Optional strain stage ---
    strained = None
    report_strained = None
    strained_view = None

    # Synthetic strain is optional and is not used for supercell-core commensurate builds
    # (where strain is already part of the solver plan). It remains useful for exercising
    # the strain pipeline on tiled-mode jobs.
    if spec.strain.enabled and spec.strain.model != "none":
        if spec.strain.model == "directed_small":
            # Required: eps, theta_deg; optional: poisson
            sp = directed_small_strain(
                eps=float(spec.strain.params["eps"]),
                theta_deg=float(spec.strain.params["theta_deg"]),
                poisson=float(spec.strain.params.get("poisson", 0.165)),
            )
        elif spec.strain.model == "random_small":
            # Required: eps_max; optional: shear_max, seed
            sp = random_small_strain(
                eps_max=float(spec.strain.params["eps_max"]),
                shear_max=spec.strain.params.get("shear_max", None),
                seed=int(spec.strain.params.get("seed", 0)),
            )
        else:
            raise ValueError(f"Unknown strain model: {spec.strain.model!r}")

        # Apply affine deformation in-plane (xy) to both positions and cell.
        strained = apply_inplane_deformation(built, sp.F2)

        # Validate strained
        if strict:
            report_strained = validate_or_raise(strained, spec, stage="strained")
        else:
            report_strained = run_validation(strained, spec, stage="strained")

        # Strained view (optional)
        if spec.output_policy.make_views and "layer_id" in strained.arrays:
            strained_view = save_3d_views_by_layer(
                positions=strained.positions,
                layer_id=strained.arrays["layer_id"],
                outdir=outdir,
                prefix=f"{spec.job_name}_strained",
                max_points=50_000,
            )

    return Part1Result(
        spec=spec,
        plan=plan,
        built=built,
        strained=strained,
        report_built=report_built,
        report_strained=report_strained,
        artifacts=Part1Artifacts(outdir=outdir, built_view=built_view, strained_view=strained_view),
    )
