from pathlib import Path

from tbg_rebuild.explore.specs import CapsPolicy, make_spec_tbg_supercell_core, make_spec_bilayer
from tbg_rebuild.build.domain import plan_domain
from tbg_rebuild.build.stage_built import run_stage_built

def test_plan_domain_tiled_ok():
    spec = make_spec_bilayer(job_name="t", material_id="graphene", repeats=(2,2))
    plan = plan_domain(spec)
    assert plan.status == "ok"

def test_plan_domain_supercell_core_ok():
    spec = make_spec_tbg_supercell_core(
        job_name="sc",
        material_id="graphene",
        theta_deg=2.0,
        max_el=12,
        theta_window_deg=0.25,
        theta_step_deg=0.05,
        strain_tol=5e-4,
        caps=CapsPolicy(max_atoms=200_000),
    )
    plan = plan_domain(spec)
    assert plan.status == "ok"
    assert hasattr(plan, "corrected")
    assert hasattr(plan, "max_strain")

def test_stage_built_surfaces_flags(tmp_path: Path):
    spec = make_spec_tbg_supercell_core(
        job_name="sc_stage",
        material_id="graphene",
        theta_deg=2.0,
        max_el=12,
        theta_window_deg=0.25,
        theta_step_deg=0.05,
        strain_tol=5e-4,
    )
    res = run_stage_built(spec, outdir=tmp_path, make_views=False, check_ase_roundtrip=False)
    assert res.plan is not None
    assert "corrected" in res.struct.meta
    assert "max_strain" in res.struct.meta
