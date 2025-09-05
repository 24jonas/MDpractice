#!/usr/bin/env python3
"""
Explore twist/stack/strain configs on HPRC:
- Build commensurate supercells with supercell_core (fast/direct)
- Enforce atom/strain caps; log skips
- Relax (LCAO by default; PW compatible via calc_kwargs)
- Do PW band-structure line (Γ–M–K–Γ) on the relaxed geometry
- Save compact, reproducible artifacts per config

Usage example (local or via SLURM):
  python scripts/explore_hprc.py \
    --stacks graphene,graphene graphene,hbn,graphene \
    --angles 0:30:3 \
    --mode fast \
    --strain-cap 0.03 \
    --atom-cap 2000 \
    --outdir explorer_runs \
    --seed 42
"""

from __future__ import annotations
import json, os, sys, time, itertools as it
from pathlib import Path
import numpy as np
import supercell_core as sc
from ase import Atoms
from ase.dft.kpoints import bandpath
from gpaw import GPAW, FermiDirac, PW

# ---- import your project helpers ----
from tbg.Materials import MATLIB
from tbg.SupercellCoreAdapter import lattice_from_material
from tbg.Multilayer import build_stack_from_result_exact  # uses your robust K/HNF logic
from tbg.Relax import relaxDFT_and_export
from tbg.MathUtils import colsToRowsCell

# ---------- CLI ----------
import argparse
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stacks", nargs="+", required=True,
                   help="Space-separated stack specs; each is comma-separated layer names (e.g. graphene,hbn,graphene)")
    p.add_argument("--angles", required=True,
                   help="Angle spec in degrees for added layers: 'start:stop:step' (substrate fixed at 0). E.g. 0:30:3")
    p.add_argument("--mode", choices=["fast","direct"], default="fast",
                   help="supercell_core algorithm: 'fast' allows strain; 'direct' = exact commensurate only")
    p.add_argument("--strain-cap", type=float, default=0.02,
                   help="Max strain (norm) for fast mode (ignored for direct).")
    p.add_argument("--atom-cap", type=int, default=2000)
    p.add_argument("--vacuum", type=float, default=24.0)
    p.add_argument("--d", type=float, default=3.35, help="Interlayer spacing (Å)")
    p.add_argument("--outdir", type=str, default="explorer_runs")
    p.add_argument("--seed", type=int, default=0)
    # Relax & bands knobs (reasonable defaults for sweep)
    p.add_argument("--relax-fmax", type=float, default=0.05)
    p.add_argument("--relax-steps", type=int, default=80)
    p.add_argument("--kmesh-relax", type=int, nargs=3, default=[4,4,1])
    p.add_argument("--pw-ecut", type=float, default=500.0)
    p.add_argument("--kmesh-pw", type=int, nargs=3, default=[12,12,1])
    p.add_argument("--bands-npts", type=int, default=240)
    p.add_argument("--dry-run", action="store_true", help="Build+filter only; skip DFT")
    return p.parse_args()

# ---------- helpers ----------
def parse_stack(spec: str):
    names = [s.strip().lower() for s in spec.split(",")]
    mats = []
    for n in names:
        if n not in MATLIB:
            raise ValueError(f"Unknown material '{n}' in stack '{spec}'")
        mats.append(MATLIB[n])
    return mats

def angle_grid(spec: str):
    # "start:stop:step" for added layers; substrate is 0
    start, stop, step = [float(x) for x in spec.split(":")]
    vals = np.arange(start, stop+1e-9, step)
    return vals

def make_config_id(stack_key: str, thetas_deg: list[float], strain_seed: int):
    # e.g. g_hbn_g__a0-24-12__s042
    ang = "-".join([f"{t:.2f}".replace(".","p") for t in thetas_deg])
    return f"{stack_key}__a{ang}__s{strain_seed:03d}"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def atoms_from_cols(R, H_cols, species):
    return Atoms(symbols=species, positions=R, cell=colsToRowsCell(H_cols), pbc=(True,True,False))

def run_pw_bands(R, H_cols, species, ecut, kmesh, npts, out_prefix):
    atoms = atoms_from_cols(R, H_cols, species)
    # SCF (PW)
    calc = GPAW(mode=PW(ecut), xc="PBE", kpts=tuple(kmesh),
                occupations=FermiDirac(0.02), nbands="150%", txt=f"{out_prefix}.log")
    atoms.calc = calc
    _ = atoms.get_potential_energy()

    # mini-BZ path: Γ–M–K–Γ (labels “GMKG” in ASE)
    bp = bandpath("GMKG", cell=atoms.cell, npoints=npts)
    x, xticks, xlabels = bp.get_linear_kpoint_axis()
    kpts = bp.kpts

    bs = calc.fixed_density(kpts=kpts, symmetry="off")
    nb = bs.get_number_of_bands()
    nk = len(kpts)
    E = np.empty((nb, nk))
    for ik in range(nk):
        E[:, ik] = bs.get_eigenvalues(kpt=ik, spin=0)
    Ef = bs.get_fermi_level()

    np.savez(f"{out_prefix}_eigs.npz", E=E, Ef=Ef, kpoints=kpts)
    write_json(Path(f"{out_prefix}_kpath.json"), {"labels": xlabels, "ticks": xticks, "x": x})
    # quick plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4.2,3.2), dpi=140)
        for i in range(min(nb, 64)):
            plt.plot(x, E[i]-Ef, lw=0.8)
        for t in xticks: plt.axvline(t, lw=0.5, ls="--")
        plt.xticks(xticks, xlabels); plt.ylabel("E − $E_F$ (eV)"); plt.tight_layout()
        plt.savefig(f"{out_prefix}.png"); plt.close()
    except Exception as e:
        print(f"[bands] plot skipped: {e}")

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    outroot = Path(args.outdir)
    ensure_dir(outroot)

    angle_vals = angle_grid(args.angles)
    stacks = [parse_stack(s) for s in args.stacks]

    # sweep
    for mats in stacks:
        stack_key = "_".join([m.name[0] for m in mats])  # e.g. g_hbn_g
        for theta in angle_vals:
            # random small symmetric strain per added layer if fast-mode
            rand_strains = []
            if args.mode == "fast":
                for _ in mats[1:]:
                    # small symmetric tensor with Fro norm ~ O(args.strain_cap)
                    S = rng.normal(scale=args.strain_cap/4, size=(2,2))
                    eps = 0.5*(S+S.T)
                    if np.linalg.norm(eps) > args.strain_cap:
                        eps *= (args.strain_cap / np.linalg.norm(eps))
                    rand_strains.append(eps)
            else:
                rand_strains = [np.zeros((2,2)) for _ in mats[1:]]

            # ---- supercell_core: build heterostructure ----
            hs = sc.heterostructure().set_substrate(lattice_from_material(mats[0]))
            for L in mats[1:]:
                hs = hs.add_layer(lattice_from_material(L))

            thetas_deg = [0.0] + [theta]*(len(mats)-1)
            thetas_rad = [t*sc.DEGREE for t in thetas_deg[1:]]

            # call solver (we keep calc for fixed angle; opt could scan)
            try:
                res = hs.calc(M=[[1,0],[0,1]], thetas=thetas_rad) if args.mode=="direct" else hs.calc(M=[[1,0],[0,1]], thetas=thetas_rad)
            except Exception as e:
                print(f"[skip] solver failed at {stack_key} θ={theta:.2f}: {e}")
                continue

            # exact tiling → positions, cell, species, meta (your robust builder)
            try:
                R, H, species, meta = build_stack_from_result_exact(
                    res, mats, thetas_deg, d=args.d, vacuum=args.vacuum
                )
            except Exception as e:
                print(f"[skip] build failed at {stack_key} θ={theta:.2f}: {e}")
                continue

            n_atoms = len(R)
            if n_atoms > args.atom_cap:
                print(f"[skip] atoms {n_atoms} > cap {args.atom_cap} for {stack_key} θ={theta:.2f}")
                continue

            # output dir
            cfg_id = make_config_id(stack_key, thetas_deg, rng.integers(0, 1000))
            odir = outroot / stack_key / cfg_id
            ensure_dir(odir)

            # metadata (build-level)
            build_meta = dict(
                stack=[m.name for m in mats],
                thetas_deg=thetas_deg,
                detK_per_layer=meta.get("detK_per_layer", []),
                n_atoms=n_atoms,
                cell_cols=H.tolist(),
                solver={"algorithm": args.mode, "version": getattr(sc, "__version__", "unknown")},
                caps={"atom_cap": args.atom_cap, "strain_cap": args.strain_cap if args.mode=="fast" else 0.0},
                time_start=time.strftime("%Y-%m-%d %H:%M:%S"),
                status="built",
            )
            write_json(odir/"metadata.json", build_meta)

            # save initial structure
            np.save(odir/"structure_initial.npy", R)
            np.save(odir/"cell_initial.npy", H)
            with open(odir/"species.txt","w") as f:
                for s in species: f.write(f"{s}\n")

            if args.dry_run:
                print(f"[dry] saved build {stack_key} θ={theta:.2f} → {n_atoms} atoms")
                continue

            # ---- relax (LCAO default, PW possible via kwargs below) ----
            try:
                Rf, Hf = relaxDFT_and_export(
                    positions=R, cell_cols=H, species=species,
                    relax_cell=False, max_force=args.relax_fmax, steps=args.relax_steps,
                    pbc=(True,True,False),
                    log_prefix=str(odir/"relax"),
                    out_basename=str(odir/"relaxed"),
                    calc_kwargs=dict(  # LCAO default; swap to PW(...) if desired
                        mode="lcao",
                        basis="dzp",
                        xc="PBE",
                        kpts=tuple(args.kmesh_relax),
                        occupations=FermiDirac(0.05),
                        nbands="120%",
                    ),
                )
            except Exception as e:
                build_meta["status"] = f"failed_relax: {e.__class__.__name__}"
                write_json(odir/"metadata.json", build_meta)
                print(f"[fail] relax: {e}")
                continue

            np.save(odir/"structure_relaxed.npy", Rf)
            np.save(odir/"cell_relaxed.npy", Hf)
            build_meta["status"] = "relaxed"
            write_json(odir/"metadata.json", build_meta)

            # ---- PW band structure on relaxed geom ----
            try:
                run_pw_bands(Rf, Hf, species, ecut=args.pw_ecut,
                             kmesh=args.kmesh_pw, npts=args.bands_npts,
                             out_prefix=str(odir/"bands_pw"))
                build_meta["status"] = "bands_done"
                write_json(odir/"metadata.json", build_meta)
                print(f"[ok] {stack_key} θ={theta:.2f}: {n_atoms} atoms → bands done.")
            except Exception as e:
                build_meta["status"] = f"failed_bands: {e.__class__.__name__}"
                write_json(odir/"metadata.json", build_meta)
                print(f"[fail] bands: {e}")

if __name__ == "__main__":
    main()
