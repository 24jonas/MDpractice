# Build twisted bilayer graphene, apply strain to supercells, relax supercells and find resultant positions.

# Imports
from Supercell import *
from ase import Atoms
from gpaw import GPAW, PW
from ase.optimize import QuasiNewton

# Apply strain to the top layer.
co2 = np.dot(co2, SM)

# Convert coordinates to ASE.
co = np.vstack([co1, co2])
symbols = ["C"] * co.shape[0]
bilayer = Atoms(symbols, positions=co, cell=supercell, pbc=True)

# GPAW Calculator
calc = GPAW(mode=PW(),
            xc='PBE',
            kpts=(2, 2, 2),
            txt='bilayer.txt')
bilayer.calc = calc

# Optimize atom positions.
relax = QuasiNewton(bilayer, logfile='bilayer.log')
relax.run(fmax=0.05)

energy = bilayer.get_potential_energy()
print("Final relaxed energy:", energy, "eV")

# Record results.
calc.write('bilayer-relaxed.gpw')
bilayer.write('bilayer-relaxed.xyz')

# View Structure
from ase.visualize import view
view(bilayer)
