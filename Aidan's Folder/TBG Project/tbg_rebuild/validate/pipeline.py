# tbg_rebuild/validate/pipeline.py
from __future__ import annotations

from typing import List
from tbg_rebuild.utils.adapter import Structure
from tbg_rebuild.explore.specs import ConfigSpec
from .errors import ValidationReport, ValidationError, CheckResult
from .checks import (
    check_required_fields,
    check_required_arrays,
    check_cell_slab,
    check_inplane_bond_and_coordination,
    check_bilayer_separation,
    check_duplicates_periodic_xy
)


def run_validation(struct: Structure, spec: ConfigSpec, *, stage: str) -> ValidationReport:
    checks: List[CheckResult] = []

    checks.append(check_required_fields(struct))
    checks.append(check_required_arrays(struct, required=("layer_id", "sublattice")))
    checks.append(check_cell_slab(struct, min_vacuum=10.0))

    # honeycomb sanity (cheap + catches many issues early)
    checks.extend(check_inplane_bond_and_coordination(struct, cutoff=1.6))

    checks.append(check_duplicates_periodic_xy(struct, tol_xy_ang=1e-3, tol_z_ang=1e-3))

    # if bilayer, check separation using spec geometry
    if len(spec.layers) == 2:
        checks.append(check_bilayer_separation(struct, expected_d=spec.geometry.interlayer_distance, tol=0.05))

    return ValidationReport(
        stage=stage,
        job_name=spec.job_name,
        n_atoms=len(struct.species),
        checks=checks,
    )


def validate_or_raise(struct: Structure, spec: ConfigSpec, *, stage: str) -> ValidationReport:
    report = run_validation(struct, spec, stage=stage)
    if not report.passed:
        raise ValidationError(report)
    return report
