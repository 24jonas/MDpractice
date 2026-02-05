# tbg_rebuild/cli/run_part1.py
from __future__ import annotations

from pathlib import Path
import json
import argparse

from tbg_rebuild.pipeline_part1 import run_part1
from tbg_rebuild.explore.specs import (
    make_spec_bilayer,
    with_directed_strain,
    make_spec_tbg_supercell_core,
)


def _append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record to a .jsonl ledger (one object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _fmt_bool(x: object) -> str:
    """Format truthy/falsey values as Y/N for compact CLI output."""
    return "Y" if bool(x) else "N"


def _maybe_float(x: object) -> str:
    """Format numeric values compactly; fall back to string; use em dash for None."""
    if x is None:
        return "—"
    try:
        return f"{float(x):.4g}"
    except Exception:
        return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tbg_rebuild.cli.run_part1",
        description="Run Part I pipeline jobs and write artifacts + a JSONL ledger.",
    )
    parser.add_argument("--outdir", default="out_part1", help="Output directory root.")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Raise on validation failures (default).",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Do not raise on validation failures.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full validation summaries (detailed output).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print failures + final ledger path.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ledger = outdir / "results.jsonl"








    # Fixed 40*1 = 40 
    RRAmatBig =      [2.2800000000000002,
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
                        29.84]
    

    # Fixed 10*1 + Tnoise 10*6 + strain (10 + 10*6)*4  = 350
    RRAmatMed =      [3.89,
                  4.41,
                  5.09,
                  8.26,
                  10.99, 
                  11.64, 
                  15.18, 
                  18.73, 
                  24.43, 
                  25.04]

    # Fixed 10*1 + Tnoise 10*12 + strain (10 + 10*12)*4 = 650
    RRAmatSmall =      [6.01,
                  7.34,
                  9.43,  
                  13.17, 
                  16.43, 
                  17.9,   
                  21.79,  #Smallest system, very high magnitude ranges
                  26.01,
                  27.8,   
                  29.41]

    # Fixed 3*1 + Tnoise 3*16 + strain (3 + 3*16)*6 = 306
    RRAmatStr =   ["bigAA",
                  "bigAB",
                  "bigASP"] #Same as BSP if using just carbon, but different if using boron/nitrogen
    
    # Fixed 1 + Tnoise 24 + strain (1 + 16)*6 + ripple (1 + 16)*7*6 = 833
    RRAmatMono = ["mono"]


    #TOTAL = 40 + 350 + 650 + 306 + 833 = 2179













    jobs = []

    # Core deliverable: solver-assisted commensurate bilayer jobs (supercell-core).
    jobs.append(
        make_spec_tbg_supercell_core(
            job_name="sc_theta_2deg",
            material_id="graphene",
            theta_deg=2.0,
            vacuum=24.0,
            interlayer_distance=3.35,
            max_el=12,
            theta_window_deg=0.25,
            theta_step_deg=0.01,
            strain_tol=5e-4,
            notes="supercell-core opt() commensurate plan; flag if strain applied",
        )
    )
    jobs.append(
        make_spec_tbg_supercell_core(
            job_name="sc_theta_1p1deg",
            material_id="graphene",
            theta_deg=1.1,
            vacuum=24.0,
            interlayer_distance=3.35,
            max_el=20,
            theta_window_deg=0.25,
            theta_step_deg=0.01,
            strain_tol=5e-4,
            notes="supercell-core opt() commensurate plan; flag if strain applied",
        )
    )

    # Baseline sanity: tiled AA bilayer.
    jobs.append(make_spec_bilayer(job_name="aa_tiled_small", material_id="graphene", repeats=(5, 5)))

    # Optional: exercise the strain stage on a small tiled system.
    jobs.append(
        with_directed_strain(
            make_spec_bilayer(job_name="aa_tiled_small_strain", material_id="graphene", repeats=(5, 5)),
            eps=0.005,
            theta_deg=15.0,
            poisson=0.165,
        )
    )

    # Demonstration: same job, but loose strain tolerance => corrected flag likely False.
    jobs.append(
        make_spec_tbg_supercell_core(
            job_name="sc_theta_2deg_loose_tol",
            material_id="graphene",
            theta_deg=2.0,
            vacuum=24.0,
            interlayer_distance=3.35,
            max_el=12,
            theta_window_deg=0.25,
            theta_step_deg=0.01,
            strain_tol=0.1,
            notes="Same as sc_theta_2deg, only tol changed.",
        )
    )

    if not args.quiet:
        if args.verbose:
            print(f"RUNNING {len(jobs)} jobs → {outdir}\n")
        else:
            print(f"RUNNING {len(jobs)} jobs → {outdir}")

    for spec in jobs:
        try:
            res = run_part1(spec, outdir=outdir, strict=args.strict)

            corrected = getattr(res.plan, "corrected", None)
            max_strain = getattr(res.plan, "max_strain", None)

            # Built size is always available for a successful run_part1 call.
            n_built = len(res.built.species)

            # Strain stage is optional; in this pipeline it is present iff res.strained is not None.
            strained_ok = (res.strained is not None)

            # Views are optional artifacts controlled by OutputPolicy.
            views = []
            if res.artifacts.built_view:
                views.append("built")
            if res.artifacts.strained_view:
                views.append("strained")
            views_s = "+".join(views) if views else "—"

            if args.verbose and not args.quiet:
                # Detailed output: keep your full summaries and artifact paths.
                print(f"\n=== RUN {spec.job_name} ===")
                if spec.periodicity.mode == "supercell_core":
                    print(f"[{spec.job_name}] corrected={corrected} max_strain={max_strain}")

                print(res.report_built.summary())
                if res.report_strained is not None:
                    print(res.report_strained.summary())

                print(f"Artifacts in: {res.artifacts.outdir}")
                if res.artifacts.built_view:
                    print(f"  built view:   {res.artifacts.built_view}")
                if res.artifacts.strained_view:
                    print(f"  strained view:{res.artifacts.strained_view}")

            elif not args.quiet:
                # Compact output: one single summary line per job.
                if spec.periodicity.mode == "supercell_core":
                    extra = f" corr={_fmt_bool(corrected)} strain={_maybe_float(max_strain)}"
                else:
                    extra = ""

                print(
                    f"  ✓ {spec.job_name:<22} mode={spec.periodicity.mode:<13}"
                    f" N={n_built:<6} strained={'OK' if strained_ok else '—':<2}"
                    f"{extra} views={views_s}"
                )

            # Ledger always records successful runs (even if --quiet).
            _append_jsonl(
                ledger,
                {
                    "job": spec.job_name,
                    "mode": spec.periodicity.mode,
                    "theta_deg": spec.periodicity.solver.get("theta_deg")
                    if spec.periodicity.mode == "supercell_core"
                    else None,
                    "corrected": corrected,
                    "max_strain": max_strain,
                    "domain_reason": res.plan.reason,
                    "built_ok": True,
                    "strained_ok": strained_ok,
                },
            )

        except Exception as e:
            # Failure path: always show something even in quiet mode.
            print(f"  ✗ FAILED: {spec.job_name}: {e}")
            if args.strict:
                raise

    if not args.quiet:
        print("Done.")
    print(f"Ledger: {ledger.resolve()}")


if __name__ == "__main__":
    main()
