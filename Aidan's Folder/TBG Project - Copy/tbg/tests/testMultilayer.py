# tbg/tests/testRelax_good_angle.py

import numpy as np
import supercell_core as sc
from tbg.Materials import GRAPHENE
from tbg.SupercellCoreAdapter import lattice_from_material
from tbg.Multilayer import build_stack_from_result_exact
from tbg.Relax import relaxDFT_and_export

ATOM_CAP = 120  # was having issues with duplicates

def main():


    
    # Bilayer Graphene simple test
    A = lattice_from_material(GRAPHENE)
    B = lattice_from_material(GRAPHENE)
    h = sc.heterostructure().set_substrate(A).add_layer(B)

    # Searching around theta0
    theta0 = 21.787
    sweep = [ (theta0 - 0.03), theta0, (theta0 + 0.03) ]
    res = h.opt(algorithm="direct", max_el=12, thetas=[np.array(sweep)*sc.DEGREE])  

    # Build exactly from Result (no oversize/wrap)
    R, H, species, meta = build_stack_from_result_exact(
        res, [GRAPHENE, GRAPHENE], [0.0, theta0], d=3.35, vacuum=24.0
    )
    print(f"[good-angle] atoms={len(R)} detK={meta['detK_per_layer']} thetas={meta['thetas_deg']}")
    def min_dist_2d(R, H):
        import numpy as np
        # fractional coords
        S = R @ np.linalg.inv(H).T
        # wrap x,y only
        S[:, :2] -= np.floor(S[:, :2])
        # all pairwise
        dmin = 1e9
        for i in range(len(R)):
            dS = S - S[i]
            dS[:, :2] -= np.round(dS[:, :2])
            dR = dS @ H.T
            d2 = np.sum(dR**2, axis=1); d2[i] = np.inf
            dmin = min(dmin, float(np.min(d2))**0.5)
        return dmin

    print("min |Δr| (2D PBC):", min_dist_2d(R, H))
    # pin bottom half
    fix_mask = np.zeros(len(R), dtype=bool)
    fix_mask[:len(R)//2] = True

    from gpaw import FermiDirac, PW, KohnShamConvergenceError
    from gpaw.mixer import Mixer

    # Primary (cheap & robust) LCAO settings for moiré relax
    from gpaw.mixer import Mixer

    relax_easy = dict(
        mode=PW(300),
        xc='PBE',
        kpts=(2,2,1),                     # a notch above Γ; helps forces
        occupations=FermiDirac(0.10),
        nbands='140%',
        mixer=Mixer(beta=0.07, nmaxold=8, weight=100.0),  # more damping
        symmetry='off',
        convergence={'energy':5e-4,'density':3e-4,'eigenstates':1e-8,'maximum iterations':250},
    )

# Last-resort fallback: low-cutoff PW (often the most robust SCF)
    relax_pw_rescue = dict(
        mode=PW(300),                 # light cutoff; fast and stable
        xc="PBE",
        kpts=(1, 1, 1),
        occupations=FermiDirac(0.12),
        nbands="140%",
        convergence={"energy": 8e-4, "density": 5e-4, "maximum iterations": 200},
        symmetry="off",
    )

    try:
        Rf, Hf = relaxDFT_and_export(
            positions=R, cell_cols=H, species=species,
            relax_cell=False, max_force=0.10, steps=40,
            pbc=(True, True, False),
            log_prefix="relax_test", out_basename="relax_test",
            calc_kwargs=relax_easy,
        )
    except KohnShamConvergenceError:
        print("[relax] SCF failed — retrying with PW rescue settings…")
        Rf, Hf = relaxDFT_and_export(
            positions=R, cell_cols=H, species=species,
            relax_cell=False, max_force=0.10, steps=40,
            pbc=(True, True, False),
            log_prefix="relax_test_pw", out_basename="relax_test_pw",
            calc_kwargs=relax_pw_rescue,
        )
if __name__ == "__main__":
    main()
