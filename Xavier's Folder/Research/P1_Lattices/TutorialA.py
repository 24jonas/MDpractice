# This is a copy of 'bandstructure.py' from the GPAW tutorial with some edits for functionality and to complete the tasks.
# The initial parameters were for silicon using the 'easy approach'.

from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac

# Perform standard ground state calculation (with plane wave basis)
# si = bulk('Si', 'diamond', 5.43)  Specifications for silicon.
si = bulk('Ag', 'fcc', 4.085)    # Specifications for silver.
calc = GPAW(mode=PW(200),
            xc='PBE',
            kpts=(8, 8, 8),
            random=True,  # random guess (needed if many empty bands required)
            occupations=FermiDirac(0.01),
            txt='Si_gs.txt')
si.calc = calc
si.get_potential_energy()
ef = calc.get_fermi_level()
calc.write('Si_gs.gpw')

# Calculate eigenvalues along a high symmetry path in the Brillouin zone
calc = GPAW('Si_gs.gpw').fixed_density(
    nbands=16,
    symmetry='off',
    kpts={'path': 'GXWKL', 'npoints': 60},
    convergence={'bands': 8})

# Plot the bandstructure
bs = calc.band_structure()
bs.plot(filename='bandstructure.png', show=True, emax=10.0)
