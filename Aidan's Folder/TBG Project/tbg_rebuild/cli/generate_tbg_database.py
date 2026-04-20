from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from tbg_rebuild.explore.specs import CapsPolicy, ConfigSpec, OutputPolicy, make_spec_tbg_supercell_core
from tbg_rebuild.materials.catalog import get_material, list_materials


# ---------------------------------------------------------------------------
# Put any twisted bilayer angles to generate in degrees. The CLI will build
# one full supercell-core structure per angle unless --angles is provided.
# ---------------------------------------------------------------------------
ANGLE_LIST_DEG = [
    2.2800000000000002,
    2.45,
    2.6500000000000004,
    2.8800000000000003,
    3.1500000000000004,
    3.48,
    4.720000000000001,
    5.51,
    6.61,
    6.84,
    7.93,
    8.61,
    10.149999999999999,
    10.42,
    11.37,
    11.989999999999998,
    12.209999999999999,
    14.11,
    14.62,
    14.309999999999999,
    15.649999999999999,
    15.870000000000001,
    17.03,
    17.28,
    18.37,
    19.27,
    19.650000000000002,
    19.93,
    20.15,
    23.3,
    23.48,
    23.71,
    24.02,
    25.46,
    26.47,
    26.75,
    27.05,
    28.51,
    28.78,
    29.84,
]


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _fmt_bool(x: object) -> str:
    return "Y" if bool(x) else "N"


def _maybe_float(x: object) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.4g}"
    except Exception:
        return str(x)


def _json_default(value: object) -> object:
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _angle_token(theta_deg: float) -> str:
    token = f"{theta_deg:.8g}".replace("-", "m").replace(".", "p")
    if "p" in token:
        token = token.rstrip("0").rstrip("p")
    return token


def _job_prefix(material_id: str, top_material_id: str | None = None) -> str:
    top_material_id = material_id if top_material_id is None else top_material_id
    if material_id == top_material_id == "graphene":
        return "tbg"
    if material_id == top_material_id:
        return f"tb_{material_id}"
    return f"tb_{material_id}_{top_material_id}"


def _parse_angle_text(value: str, *, source: str = "--angles") -> list[float]:
    angles: list[float] = []
    for lineno, raw_line in enumerate(value.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].replace(",", " ").strip()
        if not line:
            continue
        for part in line.split():
            try:
                angles.append(float(part))
            except ValueError as exc:
                raise ValueError(f"{source}:{lineno}: invalid angle {part!r}") from exc
    return angles


def _parse_angle_csv(value: str) -> list[float]:
    return _parse_angle_text(value, source="--angles")


def _parse_angle_file(path: Path) -> list[float]:
    return _parse_angle_text(path.read_text(encoding="utf-8"), source=str(path))


def _selected_angles(args: argparse.Namespace) -> list[float]:
    if args.angles and args.angle_file:
        raise ValueError("Use either --angles or --angle-file, not both.")

    if args.angle_file:
        selected = _parse_angle_file(Path(args.angle_file))
    elif args.angles:
        selected = _parse_angle_csv(args.angles)
    else:
        selected = [float(theta) for theta in ANGLE_LIST_DEG]

    seen: set[float] = set()
    deduped: list[float] = []
    for theta in selected:
        key = round(theta, 10)
        if key not in seen:
            seen.add(key)
            deduped.append(theta)
    return deduped


def _chunk_jobs(jobs: Sequence[ConfigSpec], *, chunk_index: int, chunk_count: int) -> list[ConfigSpec]:
    if chunk_count < 1:
        raise ValueError("--chunk-count must be >= 1")
    if chunk_index < 0 or chunk_index >= chunk_count:
        raise ValueError("--chunk-index must satisfy 0 <= chunk_index < chunk_count")
    return [job for i, job in enumerate(jobs) if i % chunk_count == chunk_index]


def _ledger_paths(outdir: Path, *, chunk_index: int, chunk_count: int) -> tuple[Path, Path]:
    if chunk_count > 1:
        suffix = f"_chunk_{chunk_index:04d}_of_{chunk_count:04d}.jsonl"
        return outdir / f"dataset_results{suffix}", outdir / f"dataset_failures{suffix}"
    return outdir / "dataset_results.jsonl", outdir / "dataset_failures.jsonl"


def _make_jobs(args: argparse.Namespace) -> list[ConfigSpec]:
    top_material_id = args.material_id if args.top_material_id is None else args.top_material_id
    material = get_material(args.material_id)
    top_material = get_material(top_material_id)
    interlayer_distance = (
        0.5 * (material.default_interlayer + top_material.default_interlayer)
        if args.interlayer_distance is None
        else float(args.interlayer_distance)
    )
    caps = CapsPolicy(
        max_atoms=int(args.max_atoms),
        downsample_views_above=int(args.downsample_views_above),
        max_cell_len=None,
    )
    output_policy = OutputPolicy(
        make_views=not args.no_views,
        downsample_views_above=int(args.downsample_views_above),
    )

    jobs: list[ConfigSpec] = []
    for theta in _selected_angles(args):
        token = _angle_token(theta)
        spec = make_spec_tbg_supercell_core(
            job_name=f"{_job_prefix(args.material_id, top_material_id)}_theta_{token}deg",
            material_id=args.material_id,
            top_material_id=top_material_id,
            theta_deg=theta,
            vacuum=args.vacuum,
            interlayer_distance=interlayer_distance,
            max_el=args.max_el,
            theta_window_deg=args.theta_window_deg,
            theta_step_deg=args.theta_step_deg,
            strain_tol=args.strain_tol,
            caps=caps,
            notes=(
                f"HPRC full-generation twisted bilayer {material.id}/{top_material.id} dataset job. "
                "Large structures are generated; corrected/max_strain remain dataset metadata."
            ),
        )
        jobs.append(replace(spec, output_policy=output_policy))

    return jobs


def _write_structure(path: Path, spec: ConfigSpec, result) -> Path:
    from ase.io import write

    from tbg_rebuild.utils.adapter import structure_to_atoms

    path.parent.mkdir(parents=True, exist_ok=True)
    atoms = structure_to_atoms(result.built)
    commensurate_meta = atoms.info.pop("commensurate_meta", None)
    if commensurate_meta is not None:
        atoms.info["commensurate_meta_json"] = json.dumps(
            commensurate_meta,
            default=_json_default,
            sort_keys=True,
        )
    atoms.info.update(
        {
            "job": spec.job_name,
            "theta_deg": spec.periodicity.solver.get("theta_deg"),
            "theta_used_deg": (commensurate_meta or {}).get("theta_used_deg"),
            "theta_delta_deg": (commensurate_meta or {}).get("theta_delta_deg"),
            "strain_tol": spec.periodicity.solver.get("strain_tol"),
            "corrected": bool(getattr(result.plan, "corrected", False)),
            "strain_exceeds_tol": bool(
                (getattr(result.plan, "commensurate_meta", None) or {}).get("strain_exceeds_tol", False)
            ),
            "max_strain": float(getattr(result.plan, "max_strain", 0.0)),
            "correction_max_abs": (commensurate_meta or {}).get("correction_max_abs"),
        }
    )
    write(path, atoms)
    return path


def _slurm_chunk_defaults(args: argparse.Namespace) -> None:
    if args.array_task_id is None:
        env_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_task is not None:
            args.array_task_id = int(env_task)
    if args.array_task_count is None:
        env_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
        if env_count is not None:
            args.array_task_count = int(env_count)

    if args.array_task_id is not None:
        args.chunk_index = int(args.array_task_id)
    if args.array_task_count is not None:
        args.chunk_count = int(args.array_task_count)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tbg_rebuild.cli.generate_tbg_database",
        description="Generate full twisted bilayer structure datasets for HPRC angle sweeps.",
    )
    parser.add_argument("--outdir", default="out_tbg_database", help="Output dataset directory.")
    parser.add_argument(
        "--angles",
        default=None,
        help="Comma-separated twist angles in degrees. If omitted, uses ANGLE_LIST_DEG in this file.",
    )
    parser.add_argument(
        "--angle-file",
        default=None,
        help="Text file of twist angles. Whitespace, commas, blank lines, and # comments are allowed.",
    )
    parser.add_argument("--material-id", default="graphene", choices=list_materials())
    parser.add_argument(
        "--top-material-id",
        default=None,
        choices=list_materials(),
        help="Top/twisted layer material. Defaults to --material-id.",
    )
    parser.add_argument("--vacuum", type=float, default=24.0)
    parser.add_argument(
        "--interlayer-distance",
        type=float,
        default=None,
        help="Layer spacing in Å. Defaults to the selected material's catalog value.",
    )
    parser.add_argument("--max-el", type=int, default=80, help="supercell-core search limit for production jobs.")
    parser.add_argument("--theta-window-deg", type=float, default=0.25)
    parser.add_argument("--theta-step-deg", type=float, default=0.01)
    parser.add_argument("--strain-tol", type=float, default=5e-4)
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=999_999_999,
        help="Practical HPRC safety ceiling. Set high to avoid local-demo skips.",
    )
    parser.add_argument("--downsample-views-above", type=int, default=25_000)
    parser.add_argument("--no-views", action="store_true", help="Do not write PNG diagnostic views.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip jobs whose structure file already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without generating structures.")
    parser.add_argument("--chunk-index", type=int, default=0, help="Run only jobs where index %% chunk_count matches this.")
    parser.add_argument("--chunk-count", type=int, default=1, help="Split the sweep into this many chunks.")
    parser.add_argument("--array-task-id", type=int, default=None, help="Override chunk index from Slurm array task id.")
    parser.add_argument("--array-task-count", type=int, default=None, help="Override chunk count from Slurm array size.")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Raise on validation failures (default).",
    )
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    _slurm_chunk_defaults(args)

    outdir = Path(args.outdir)
    structures_dir = outdir / "structures"
    ledger, failures = _ledger_paths(outdir, chunk_index=args.chunk_index, chunk_count=args.chunk_count)

    try:
        jobs_all = _make_jobs(args)
        jobs = _chunk_jobs(jobs_all, chunk_index=args.chunk_index, chunk_count=args.chunk_count)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if not args.quiet:
        print(
            f"PLANNED {len(jobs)} of {len(jobs_all)} jobs "
            f"(chunk {args.chunk_index}/{args.chunk_count}) -> {outdir}"
        )

    if args.dry_run:
        for i, spec in enumerate(jobs):
            s = spec.periodicity.solver
            print(
                f"{i:04d} {spec.job_name} theta={s.get('theta_deg')} "
                f"max_el={s.get('max_el')} strain_tol={s.get('strain_tol')}"
            )
        return

    for spec in jobs:
        structure_path = structures_dir / f"{spec.job_name}.extxyz"
        if args.skip_existing and structure_path.exists():
            if not args.quiet:
                print(f"  SKIP {spec.job_name:<30} existing={structure_path}")
            continue

        try:
            from tbg_rebuild.pipeline_part1 import run_part1

            res = run_part1(spec, outdir=outdir / "artifacts", strict=args.strict)
            structure_path = _write_structure(structure_path, spec, res)

            corrected = bool(getattr(res.plan, "corrected", False))
            max_strain = float(getattr(res.plan, "max_strain", 0.0))
            commensurate_meta = getattr(res.plan, "commensurate_meta", None) or {}
            strain_exceeds_tol = bool(commensurate_meta.get("strain_exceeds_tol", False))
            correction_max_abs = commensurate_meta.get("correction_max_abs")
            theta_used_deg = commensurate_meta.get("theta_used_deg")
            theta_delta_deg = commensurate_meta.get("theta_delta_deg")
            theta_deg = spec.periodicity.solver.get("theta_deg")
            strain_tol = spec.periodicity.solver.get("strain_tol")
            n_built = len(res.built.species)

            _append_jsonl(
                ledger,
                {
                    "job": spec.job_name,
                    "mode": spec.periodicity.mode,
                    "theta_deg": theta_deg,
                    "theta_used_deg": theta_used_deg,
                    "theta_delta_deg": theta_delta_deg,
                    "strain_tol": strain_tol,
                    "corrected": corrected,
                    "strain_exceeds_tol": strain_exceeds_tol,
                    "max_strain": max_strain,
                    "correction_max_abs": correction_max_abs,
                    "domain_reason": res.plan.reason,
                    "n_atoms": n_built,
                    "max_el": spec.periodicity.solver.get("max_el"),
                    "theta_window_deg": spec.periodicity.solver.get("theta_window_deg"),
                    "theta_step_deg": spec.periodicity.solver.get("theta_step_deg"),
                    "structure_path": str(structure_path),
                    "built_ok": True,
                    "validation_built_ok": bool(res.report_built.passed),
                },
            )

            if not args.quiet:
                print(
                    f"  OK   {spec.job_name:<30} N={n_built:<8} "
                    f"corr={_fmt_bool(corrected)} tol={_fmt_bool(strain_exceeds_tol)} "
                    f"strain={_maybe_float(max_strain)}"
                )

        except Exception as e:
            _append_jsonl(
                failures,
                {
                    "job": spec.job_name,
                    "mode": spec.periodicity.mode,
                    "theta_deg": spec.periodicity.solver.get("theta_deg"),
                    "strain_tol": spec.periodicity.solver.get("strain_tol"),
                    "error": str(e),
                    "built_ok": False,
                },
            )
            print(f"  FAIL {spec.job_name}: {e}")
            if args.strict:
                raise

    if not args.quiet:
        print("Done.")
    print(f"Ledger: {ledger.resolve()}")


if __name__ == "__main__":
    main()
