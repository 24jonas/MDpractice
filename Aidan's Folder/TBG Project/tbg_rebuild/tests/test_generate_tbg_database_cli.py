from argparse import Namespace
from pathlib import Path

from tbg_rebuild.cli.generate_tbg_database import (
    _angle_token,
    _make_jobs,
    _chunk_jobs,
    _ledger_paths,
    _parse_angle_file,
    _selected_angles,
)
from tbg_rebuild.explore.specs import make_spec_tbg_supercell_core


def test_parse_angle_file_allows_commas_comments_and_blank_lines(tmp_path: Path):
    angle_file = tmp_path / "angles.txt"
    angle_file.write_text(
        """
        1.1, 2.0

        3.5   # local smoke
        29.84
        """,
        encoding="utf-8",
    )

    assert _parse_angle_file(angle_file) == [1.1, 2.0, 3.5, 29.84]


def test_selected_angles_dedupes_preserving_order(tmp_path: Path):
    angle_file = tmp_path / "angles.txt"
    angle_file.write_text("2.0\n1.1\n2.00000000001\n", encoding="utf-8")
    args = Namespace(angles=None, angle_file=str(angle_file))

    assert _selected_angles(args) == [2.0, 1.1]


def test_chunk_jobs_uses_modulo_partitioning():
    jobs = [
        make_spec_tbg_supercell_core(job_name=f"job_{i}", material_id="graphene", theta_deg=2.0 + i)
        for i in range(5)
    ]

    assert [job.job_name for job in _chunk_jobs(jobs, chunk_index=1, chunk_count=2)] == ["job_1", "job_3"]


def test_chunked_ledger_paths_include_chunk_identity(tmp_path: Path):
    ledger, failures = _ledger_paths(tmp_path, chunk_index=3, chunk_count=10)

    assert ledger == tmp_path / "dataset_results_chunk_0003_of_0010.jsonl"
    assert failures == tmp_path / "dataset_failures_chunk_0003_of_0010.jsonl"


def test_angle_token_preserves_integer_trailing_zeroes():
    assert _angle_token(30.0) == "30"
    assert _angle_token(2.0) == "2"
    assert _angle_token(2.50) == "2p5"


def test_make_jobs_supports_hbn_defaults():
    args = Namespace(
        angles="2.0",
        angle_file=None,
        material_id="hbn",
        top_material_id=None,
        vacuum=24.0,
        interlayer_distance=None,
        max_el=12,
        theta_window_deg=0.25,
        theta_step_deg=0.05,
        strain_tol=5e-4,
        max_atoms=200_000,
        downsample_views_above=25_000,
        no_views=True,
    )

    jobs = _make_jobs(args)

    assert [job.job_name for job in jobs] == ["tb_hbn_theta_2deg"]
    assert [layer.material_id for layer in jobs[0].layers] == ["hbn", "hbn"]
    assert jobs[0].geometry.interlayer_distance == 3.33


def test_make_jobs_supports_heterostructure_materials():
    args = Namespace(
        angles="2.0",
        angle_file=None,
        material_id="graphene",
        top_material_id="hbn",
        vacuum=24.0,
        interlayer_distance=None,
        max_el=12,
        theta_window_deg=0.25,
        theta_step_deg=0.05,
        strain_tol=5e-4,
        max_atoms=200_000,
        downsample_views_above=25_000,
        no_views=True,
    )

    jobs = _make_jobs(args)

    assert [job.job_name for job in jobs] == ["tb_graphene_hbn_theta_2deg"]
    assert [layer.material_id for layer in jobs[0].layers] == ["graphene", "hbn"]
    assert jobs[0].geometry.interlayer_distance == 3.34
