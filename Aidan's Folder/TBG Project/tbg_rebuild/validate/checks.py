# tbg_rebuild/validate/checks.py
from __future__ import annotations

from typing import Tuple, List
import numpy as np

from tbg_rebuild.utils.adapter import Structure
from tbg_rebuild.utils.geom import cell_metrics, min_distance_xy_sample, neighbor_counts_xy_sample
from tbg_rebuild.validate.dupes import find_duplicates_periodic_xy
from .errors import CheckResult


def _ok(name: str, msg: str, **details) -> CheckResult:
    return CheckResult(name=name, passed=True, severity="ERROR", message=msg, details=details)


def _fail(name: str, msg: str, **details) -> CheckResult:
    return CheckResult(name=name, passed=False, severity="ERROR", message=msg, details=details)


def _warn(name: str, msg: str, **details) -> CheckResult:
    return CheckResult(name=name, passed=False, severity="WARN", message=msg, details=details)


def check_required_fields(struct: Structure) -> CheckResult:
    N = len(struct.species)
    if struct.positions.shape != (N, 3):
        return _fail("required_fields", "positions shape must be (N,3)", got=struct.positions.shape, N=N)
    if struct.cell.shape != (3, 3):
        return _fail("required_fields", "cell shape must be (3,3)", got=struct.cell.shape)
    if np.asarray(struct.pbc).shape != (3,):
        return _fail("required_fields", "pbc shape must be (3,)", got=np.asarray(struct.pbc).shape)
    return _ok("required_fields", "Structure shapes OK.")


def check_required_arrays(struct: Structure, required: Tuple[str, ...]) -> CheckResult:
    missing = [k for k in required if k not in struct.arrays]
    if missing:
        return _fail(
            "required_arrays",
            f"Missing required per-atom arrays: {missing}.",
            missing=missing,
            present=sorted(list(struct.arrays.keys())),
        )
    # also check lengths
    N = len(struct.species)
    bad = {k: int(len(np.asarray(struct.arrays[k]))) for k in required if len(np.asarray(struct.arrays[k])) != N}
    if bad:
        return _fail("required_arrays", "Required arrays have wrong length.", bad_lengths=bad, N=N)
    return _ok("required_arrays", "Required per-atom arrays present and sized correctly.", required=list(required))


def check_cell_slab(struct: Structure, *, min_vacuum: float = 10.0) -> CheckResult:
    m = cell_metrics(struct.cell)
    if m["cell_z"] <= 0.0:
        return _fail("cell_slab", "cell_z must be > 0 (vacuum must be set).", metrics=m)
    if m["cell_z"] < min_vacuum:
        return _warn("cell_slab", f"vacuum seems small (cell_z={m['cell_z']:.3f} Å).", metrics=m)
    if not (struct.pbc[0] and struct.pbc[1]) or struct.pbc[2]:
        return _warn("cell_slab", f"Expected slab pbc=(T,T,F); got {tuple(bool(x) for x in struct.pbc)}.", metrics=m)
    return _ok("cell_slab", "Cell/vacuum and PBC look sane for a slab.", metrics=m)


def check_inplane_bond_and_coordination(
    struct: Structure,
    *,
    cutoff: float = 1.6,
    expected_nn: int = 3,
    expected_min_dist: float = 1.42,
    min_dist_tol: float = 0.15,
    sample_max: int = 4000,
) -> List[CheckResult]:
    """
    Cheap gate for honeycomb correctness.
    Uses sampled min distance + sampled coordination counts.
    """
    results: List[CheckResult] = []

    mind = min_distance_xy_sample(struct.positions, struct.cell, max_atoms=sample_max, seed=0)
    if not (expected_min_dist - min_dist_tol <= mind <= expected_min_dist + min_dist_tol):
        results.append(
            _fail(
                "inplane_min_distance",
                f"Unexpected nearest-neighbor distance (sampled) ~ {mind:.4f} Å. "
                f"Expected ~{expected_min_dist}±{min_dist_tol} Å. "
                "This often means the primitive basis is wrong or the tiling vectors are inconsistent.",
                min_distance=mind,
                expected=expected_min_dist,
                tol=min_dist_tol,
            )
        )
    else:
        results.append(_ok("inplane_min_distance", f"Min distance looks OK ({mind:.4f} Å).", min_distance=mind))

    nn = neighbor_counts_xy_sample(struct.positions, struct.cell, cutoff=cutoff, max_atoms=sample_max, seed=0)
    vals, counts = np.unique(nn, return_counts=True)
    hist = {int(v): int(c) for v, c in zip(vals, counts)}
    frac_expected = float(hist.get(expected_nn, 0)) / float(len(nn) if len(nn) else 1)

    if frac_expected < 0.95:
        results.append(
            _fail(
                "inplane_coordination",
                f"Coordination (sampled, cutoff={cutoff}) is not mostly {expected_nn}. "
                f"Only {frac_expected*100:.1f}% matched. "
                "This usually indicates wrapping/cell issues or incorrect geometry.",
                cutoff=cutoff,
                hist=hist,
                frac_expected=frac_expected,
            )
        )
    else:
        results.append(
            _ok(
                "inplane_coordination",
                f"Coordination looks OK: {frac_expected*100:.1f}% have {expected_nn} neighbors (sampled).",
                cutoff=cutoff,
                hist=hist,
            )
        )

    return results


def check_bilayer_separation(struct: Structure, *, expected_d: float, tol: float = 0.05) -> CheckResult:
    if "layer_id" not in struct.arrays:
        return _fail("bilayer_separation", "layer_id array missing; cannot assess bilayer separation.")
    layer_id = np.asarray(struct.arrays["layer_id"], dtype=int)
    layers = np.unique(layer_id)
    if len(layers) != 2:
        return _fail("bilayer_separation", f"Expected exactly 2 layers; found {len(layers)}.", layers=layers.tolist())

    z = np.asarray(struct.positions[:, 2], dtype=float)
    z0 = z[layer_id == layers[0]]
    z1 = z[layer_id == layers[1]]
    dz = float(np.mean(z1) - np.mean(z0))

    if not (expected_d - tol <= abs(dz) <= expected_d + tol):
        return _fail(
            "bilayer_separation",
            f"Bilayer separation mean Δz={dz:.4f} Å, expected ~{expected_d}±{tol} Å.",
            dz_mean=dz,
            expected=expected_d,
            tol=tol,
            z0_mean=float(np.mean(z0)),
            z1_mean=float(np.mean(z1)),
        )
    return _ok("bilayer_separation", f"Bilayer separation OK (Δz≈{dz:.4f} Å).", dz_mean=dz)

def check_duplicates_periodic_xy(
    struct: Structure,
    *,
    tol_xy_ang: float = 1e-3,
    tol_z_ang: float = 1e-3,
) -> CheckResult:
    dup = find_duplicates_periodic_xy(
        struct.positions,
        struct.cell,
        tol_xy_ang=tol_xy_ang,
        tol_z_ang=tol_z_ang,
        max_groups_report=10,
    )

    if dup.n_dupe_groups == 0:
        return _ok(
            "duplicates_periodic_xy",
            f"No duplicates detected (tol_xy={tol_xy_ang} Å, tol_z={tol_z_ang} Å).",
            n_atoms=dup.n_atoms,
        )

    # Actionable message with likely causes
    return _fail(
        "duplicates_periodic_xy",
        f"Detected {dup.n_dupe_groups} duplicate groups involving {dup.n_dupe_atoms} atoms "
        f"(tol_xy={tol_xy_ang} Å, tol_z={tol_z_ang} Å). "
        "This usually indicates a crop/wrap/repeat bug or accidental re-insertion of atoms.",
        n_atoms=dup.n_atoms,
        n_dupe_groups=dup.n_dupe_groups,
        n_dupe_atoms=dup.n_dupe_atoms,
        sample=dup.sample,
        first_groups=[g[:10] for g in dup.groups[:5]],
    )