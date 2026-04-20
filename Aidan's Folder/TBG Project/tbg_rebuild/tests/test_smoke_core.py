from pathlib import Path

from tbg_rebuild.explore.specs import CapsPolicy, make_spec_tbg_supercell_core, make_spec_bilayer
from tbg_rebuild.build.domain import plan_domain
from tbg_rebuild.build.monolayer import lattice_points_in_supercell
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

def test_hbn_tiled_geometry_validates():
    spec = make_spec_bilayer(job_name="hbn_tiled", material_id="hbn", repeats=(4, 4))
    plan = plan_domain(spec)
    assert plan.status == "ok"

    res = run_stage_built(spec, make_views=False, check_ase_roundtrip=False)
    assert res.report.passed

def test_plan_domain_heterostructure_supercell_core_ok():
    spec = make_spec_tbg_supercell_core(
        job_name="graphene_hbn_sc",
        material_id="graphene",
        top_material_id="hbn",
        theta_deg=2.0,
        max_el=12,
        theta_window_deg=0.25,
        theta_step_deg=0.05,
        strain_tol=5e-4,
        caps=CapsPolicy(max_atoms=200_000),
    )
    plan = plan_domain(spec)
    assert plan.status == "ok"
    assert plan.commensurate_meta["material_ids"] == ["graphene", "hbn"]

def test_matrix_lattice_points_round_float_integer_matrices():
    pts = lattice_points_in_supercell(
        [
            [28.999999999, -2.000000001],
            [-2.000000001, -27.000000001],
        ]
    )
    assert len(pts) == 787

def test_graphene_hbn_supercell_build_validates():
    spec = make_spec_tbg_supercell_core(
        job_name="graphene_hbn_build",
        material_id="graphene",
        top_material_id="hbn",
        theta_deg=5.51,
        max_el=40,
        theta_window_deg=0.25,
        theta_step_deg=0.02,
        strain_tol=5e-4,
        caps=CapsPolicy(max_atoms=200_000),
    )
    res = run_stage_built(spec, make_views=False, check_ase_roundtrip=False)
    assert res.report.passed

def test_graphene_hbn_matrix_basis_build_validates():
    spec = make_spec_tbg_supercell_core(
        job_name="graphene_hbn_matrix_basis",
        material_id="graphene",
        top_material_id="hbn",
        theta_deg=7.93,
        max_el=80,
        theta_window_deg=0.25,
        theta_step_deg=0.01,
        strain_tol=5e-4,
        caps=CapsPolicy(max_atoms=200_000),
    )
    res = run_stage_built(spec, make_views=False, check_ase_roundtrip=False)
    assert res.report.passed
