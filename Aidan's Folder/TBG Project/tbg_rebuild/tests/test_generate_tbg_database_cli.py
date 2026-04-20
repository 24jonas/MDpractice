from argparse import Namespace
from pathlib import Path

from tbg_rebuild.cli.generate_tbg_database import (
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
