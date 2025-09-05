import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.filters import ExpCellFilter
from ase.optimize import LBFGS
from gpaw import GPAW, FermiDirac

from .MathUtils import colsToRowsCell, rowsToColsCell

def relaxDFT_and_export(
    positions: np.ndarray,
    cell_cols: np.ndarray,                    # (3,3) col stacked
    *,
    calc_kwargs: dict | None = None,
    species: list[str] | None = None,
    fix_mask: np.ndarray | None = None,       # True = atom fixed
    relax_cell: bool = False,                 # relax a,b only
    max_force: float = 0.03,                  # eV/A
    steps: int = 200,
    pbc: tuple[bool, bool, bool] = (True, True, False),
    log_prefix: str = "relax",
    out_basename: str = "relaxed",            # write relaxed_positions.npy, relaxed_cell.npy
) -> tuple[np.ndarray, np.ndarray]:

    R0 = np.asarray(positions, float)
    H_rows = colsToRowsCell(cell_cols)

    sym = species if species is not None else (["C"] * len(R0))
    atoms = Atoms(symbols=sym, positions=R0, cell=H_rows, pbc=pbc)
    
    if fix_mask is not None:
        atoms.set_constraint(FixAtoms(mask=np.asarray(fix_mask, bool)))

    # GPAW defaults, overrided by calc_kwargs if provided
    kwargs = dict(
        mode="lcao",                           # lcao for speed, PW for final
        basis="dzp",
        xc="PBE",
        kpts={"density": 2.0, "gamma": True},
        occupations=FermiDirac(0.05),
        txt=f"{log_prefix}.log",
        convergence={"energy": 2e-4},
        nbands="120%",
    )
    if calc_kwargs:
        kwargs.update(calc_kwargs)
    atoms.calc = GPAW(**kwargs)

    # relax a and b only
    filt = ExpCellFilter(atoms, mask=[1, 1, 0, 0, 0, 0], scalar_pressure=0.0) if relax_cell else atoms

    opt = LBFGS(filt, logfile=f"{log_prefix}_opt.log")
    opt.run(fmax=max_force, steps=steps)

    Rf = atoms.get_positions()
    Hf_cols = rowsToColsCell(atoms.cell.array)

    np.save(f"{out_basename}_positions.npy", Rf)
    np.save(f"{out_basename}_cell.npy", Hf_cols)

    return Rf, Hf_cols
