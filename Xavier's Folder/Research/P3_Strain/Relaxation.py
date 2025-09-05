# This file will not run do to a missing import 'Rescuplus'. This is not resolvable as accessing that import requires permission from a
# separate organization.

"""
from ase.build import bulk
from ase.optimize import BFGS
from nanotools.ase.calculators.rescuplus import Rescuplus
import os

nprc = 20
a = 5.43  # lattice constant in ang
atoms = bulk("Si", "diamond", a=a)
atoms.rattle(stdev=0.05, seed=1)  # move atoms around a bit
# rescuplus calculator
# Nanobase PP Si_AtomicData.mat should be found on path RESCUPLUS_PSEUDO (env variable)
inp = {"system": {"cell": {"resolution": 0.16}, "kpoint": {"grid": [5, 5, 5]}}}
inp["energy"] = {"forces_return": True, "stress_return": True}
inp["solver"] = {
    "mix": {"alpha": 0.5},
    "restart": {"DMRPath": "nano_scf_out.h5"},
    "mpidist": {"kptprc": nprc},
}
cmd = f"mpiexec -n {nprc} rescuplus -i PREFIX.rsi > resculog.out && cp nano_scf_out.json PREFIX.rso"
atoms.calc = Rescuplus(command=cmd, input_data=inp)
# relaxation calculation
os.system("rm -f nano_scf_out.h5")
opt = BFGS(atoms, trajectory="si2.traj", logfile="relax.out")
opt.run(fmax=0.01)
"""
