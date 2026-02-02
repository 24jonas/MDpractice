# tbg_rebuild/cli/smoke.py
from __future__ import annotations

from pathlib import Path
import json

from tbg_rebuild.explore.specs import (
    CapsPolicy,
    make_spec_bilayer,
    with_directed_strain,
    make_spec_tbg_supercell_core,
)
from tbg_rebuild.build.stage_built import run_stage_built
from tbg_rebuild.build.stage_strain import run_stage_strain


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    outdir = Path("smoke_out_unified")
    outdir.mkdir(parents=True, exist_ok=True)
    ledger = outdir / "smoke_results.jsonl"

    caps = CapsPolicy(max_atoms=200_000, downsample_views_above=25_000)

    # ---- Spec: solver-commensurate (supercell-core) ----
    sc_spec = make_spec_tbg_supercell_core(
        job_name="smoke_sc_graphene_theta_2deg",
        material_id="graphene",
        theta_deg=2.0,
        interlayer_distance=3.35,
        vacuum=24.0,
        max_el=12,
        theta_window_deg=0.25,
        theta_step_deg=0.01,
        strain_tol=5e-4,
        caps=caps,
        notes="Unified smoke: supercell-core commensurate build + flag correction.",
    )

    sc_built = run_stage_built(
        sc_spec,
        outdir=outdir,
        make_views=True,
        check_ase_roundtrip=True,
    )

    corrected = sc_built.struct.meta.get("corrected")
    max_strain = sc_built.struct.meta.get("max_strain")

    print("[smoke] SUPERCELL-CORE BUILT: PASSED validation gate")
    print(sc_built.report.summary())
    print("[smoke] SC corrected:", corrected, "max_strain:", max_strain)
    print("[smoke] SC view:", sc_built.view_png)

    _append_jsonl(ledger, {
        "job": sc_spec.job_name,
        "mode": sc_spec.periodicity.mode,
        "theta_deg": sc_spec.periodicity.solver.get("theta_deg"),
        "corrected": corrected,
        "max_strain": max_strain,
        "status": "ok",
    })

    # ---- Spec: tiled bilayer ----
    spec = make_spec_bilayer(
        job_name="smoke_bilayer_graphene_20x20",
        material_id="graphene",
        repeats=(20, 20),
        interlayer_distance=3.35,
        vacuum=24.0,
        caps=caps,
        notes="Unified smoke: build bilayer then apply small directed strain.",
    )

    built = run_stage_built(
        spec,
        outdir=outdir,
        make_views=True,
        check_ase_roundtrip=True,
    )

    print("[smoke] BUILT: PASSED validation gate")
    print(built.report.summary())
    print("[smoke] BUILT view:", built.view_png)

    spec_strain = with_directed_strain(spec, eps=0.05, theta_deg=0.0, poisson=0.165)

    strained = run_stage_strain(
        spec_strain,
        outdir=outdir,
        make_views=True,
        check_ase_roundtrip=True,
    )

    print("[smoke] STRAINED: PASSED validation gate")
    print(strained.report.summary())
    if hasattr(strained, "views"):
        print("[smoke] STRAINED views:", strained.views)
    elif hasattr(strained, "view_png"):
        print("[smoke] STRAINED view:", strained.view_png)

    _append_jsonl(ledger, {
        "job": spec.job_name,
        "mode": spec.periodicity.mode,
        "corrected": None,
        "max_strain": None,
        "status": "ok",
    })

    print(f"[smoke] Done. Outputs in: {outdir.resolve()}")
    print(f"[smoke] Ledger: {ledger.resolve()}")


if __name__ == "__main__":
    main()
