import sys
import json
import time
import numpy as np
from pathlib import Path
from tbg.Materials import MATLIB
from tbg.SupercellCoreAdapter import lattice_from_material
from tbg.Multilayer import build_stack_from_result_exact
import supercell_core as sc
from gpaw.utilities import h2gpts
from ase import Atoms
from ase.optimize import LBFGS
from ase.dft.kpoints import bandpath
from gpaw import GPAW, FermiDirac


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--layers", help="Comma-separated materials (substrate first). Example: graphene,hbn,graphene")
    p.add_argument("--solver", choices=["fast", "direct"], default="fast")
    p.add_argument("--angle-grid", help='Angle sweep in degrees "start:end:step" (applies to all added layers)')
    p.add_argument("--angle-rand", type=int, help="Number of random angles in [0,30) to try (applies to all added layers)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for random angles")
    p.add_argument("--d", type=float, default=3.35, help="Interlayer spacing (Å)")
    p.add_argument("--vacuum", type=float, default=24.0, help="Vacuum (Å)")
    p.add_argument("--atom-cap", type=int, default=1200, help="Skip if total atoms exceed this cap")
    p.add_argument("--relax", action="store_true", help="Run a quick LCAO relaxation")
    p.add_argument("--bands", action="store_true", help="Run non-SCF band sweep along Γ–M–K–Γ")
    p.add_argument("--mode", choices=["lcao", "pw"], default="pw", help="GPAW mode for relax/bands (default: pw)")
    p.add_argument("--basis", default="dzp", help="LCAO basis (if mode=lcao)")
    p.add_argument("--kpt-density", type=float, default=2.0, help="k-point density for relax")
    p.add_argument("--nbands", default="120%", help='Number of bands, e.g. "120%%" or 200')
    p.add_argument("--smear", type=float, default=0.05, help="Fermi-Dirac smearing (eV)")
    p.add_argument("--band-npts", type=int, default=200, help="Number of points for Γ–M–K–Γ path")
    p.add_argument("--outdir", default="runs", help="Output root directory")
    args = p.parse_args()

    if not args.layers:
        print("tbg_explore: missing required --layers.\n")
        print("Example:")
        print("  python -m tbg.scripts.tbg_explore \\")
        print("    --layers graphene,hbn,graphene \\")
        print("    --solver fast \\")
        print("    --angle-grid \"0:30:0.5\" \\")
        print("    --atom-cap 1200 \\")
        print("    --relax --bands \\")
        print("    --outdir runs/demo")
        sys.exit(2)

    return args


def _angle_list(args):
    if args.angle_grid:
        a, b, s = map(float, args.angle_grid.split(":"))
        return list(np.arange(a, b + 1e-9, s))
    if args.angle_rand:
        rng = np.random.default_rng(args.seed)
        return list(rng.uniform(0.0, 30.0, size=args.angle_rand))
    # default: just test 0°
    return [0.0]


def _materials_from_names(names_csv):
    mats = []
    for name in names_csv.split(","):
        key = name.strip().lower()
        if key not in MATLIB:
            raise SystemExit(f"Unknown material '{name}'. Known: {', '.join(MATLIB.keys())}")
        mats.append(MATLIB[key])
    return mats


def _build_result_dir(outroot: Path, layers, theta_deg):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag_layers = "_".join(m.name for m in layers)
    tag = f"{stamp}__{tag_layers}__ang{theta_deg:.3f}"
    d = outroot / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_structure(run_dir: Path, R, H, species, meta):
    np.save(run_dir / "structure_positions.npy", R)
    np.save(run_dir / "structure_cell.npy", H)
    with open(run_dir / "species.txt", "w") as f:
        f.write("\n".join(species))
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def _relax_inline(run_dir: Path, R, H, species, args):
    atoms = Atoms(
        symbols=species,
        positions=np.asarray(R, float),
        cell=np.asarray(H, float).T,
        pbc=(True, True, False),
    )

    calc_kwargs = dict(
        xc="PBE",
        occupations=FermiDirac(args.smear),
        txt=str(run_dir / "relax.log"),
        convergence={"energy": 2e-4},
        nbands=args.nbands,
        kpts={"density": args.kpt_density, "gamma": True},
    )

    if getattr(args, "mode", "pw") == "lcao":
        calc_kwargs.update(dict(mode="lcao", basis=args.basis))
        gpts = h2gpts(0.22, atoms.cell, idiv = 8)
        calc_kwargs["gpts"] = gpts

    else:
        from gpaw import PW
        calc_kwargs.update(dict(mode=PW(400)))  


    atoms.calc = GPAW(**calc_kwargs)

    # ----- Quick, loose geometry relax -----
    opt = LBFGS(atoms, logfile=str(run_dir / "relax_opt.log"))
    opt.run(fmax=0.10, steps=60)

    # Save checkpoint and relaxed arrays
    atoms.calc.write(str(run_dir / "relax.gpw"), mode="all")
    np.save(run_dir / "relaxed_positions.npy", atoms.get_positions())
    np.save(run_dir / "relaxed_cell.npy", atoms.cell.array.T) 
    return str(run_dir / "relax.gpw")


def _bands_inline(run_dir: Path, gpw_path: str, band_npts: int):
    calc = GPAW(gpw_path, txt=str(run_dir / "bands.log"))

    sympath = "G-M-K-G"
    atoms = calc.atoms
    kpath = bandpath(sympath, cell=atoms.cell, npoints=band_npts)
    kpts_frac = np.array(kpath.kpts) 
    kdist, _ = kpath.get_linear_kpoint_axis()

    eigs = []
    for k in kpts_frac:
        calc.set(kpts=[k], symmetry={"point_group": False, "time_reversal": False})
        _ = calc.get_potential_energy()  
        ek = calc.get_eigenvalues()    
        eigs.append(ek)
    E = np.array(eigs).T  
    Ef = calc.get_fermi_level()

    np.savez(run_dir / "bands.npz", kpts=kpts_frac, kdist=kdist, energies=E, Ef=Ef)
    return


def main():
    args = _parse_args()
    outroot = Path(args.outdir); outroot.mkdir(parents=True, exist_ok=True)
    layers = _materials_from_names(args.layers)
    if len(layers) < 1:
        raise SystemExit("Need at least one layer in --layers.")

    substrate = lattice_from_material(layers[0])
    h = sc.heterostructure().set_substrate(substrate)
    for mat in layers[1:]:
        h = h.add_layer(lattice_from_material(mat))

    angles = _angle_list(args)

    for theta_deg in angles:
        thetas = [np.array([theta_deg * sc.DEGREE])] * max(1, len(layers) - 1)

        print(f"\n== Explore: layers={','.join(m.name for m in layers)}  theta={theta_deg:.3f}°  solver={args.solver}")
        res = h.opt(algorithm=args.solver, max_el=20, thetas=thetas)

        R, H, species, meta = build_stack_from_result_exact(
            res, layers, d=args.d, vacuum=args.vacuum
        )

        nat = len(R)
        print(f"   → supercell atoms: {nat}")
        if nat > args.atom_cap:
            print(f"   … skip (>{args.atom_cap} atoms).")
            continue

        run_dir = _build_result_dir(outroot, layers, theta_deg)
        meta_out = {
            "layers": [m.name for m in layers],
            "theta_deg": theta_deg,
            "solver": args.solver,
            "atom_count": nat,
            "detK_per_layer": meta.get("detK_per_layer", []),
            "thetas_deg_all": meta.get("thetas_deg", []),
            "d": args.d,
            "vacuum": args.vacuum,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_structure(run_dir, R, H, species, meta_out)
        gpw_path = None

        if args.relax:
            print("   → relax (pw by default)…")
            gpw_path = _relax_inline(run_dir, R, H, species, args)

        if args.bands:
            # If relax not done, create a  SCF checkpoint first
            if gpw_path is None:
                print("   → quick checkpoint for bands…")
                atoms = Atoms(symbols=species, positions=R, cell=H.T, pbc=(True, True, False))
                calc = GPAW(mode="lcao", basis=args.basis, xc="PBE",
                            kpts={"density": args.kpt_density, "gamma": True},
                            occupations=FermiDirac(args.smear),
                            txt=str(run_dir / "scf_for_bands.log"),
                            convergence={"energy": 2e-4}, nbands=args.nbands)
                atoms.calc = calc
                _ = atoms.get_potential_energy()
                gpw_path = str(run_dir / "relax.gpw")
                calc.write(gpw_path, mode="all")
            print("   → bands Γ–M–K–Γ…")
            _bands_inline(run_dir, gpw_path, args.band_npts)
            print("   → wrote bands.npz")

        print(f"✓ Done: {run_dir}")

if __name__ == "__main__":
    main()
