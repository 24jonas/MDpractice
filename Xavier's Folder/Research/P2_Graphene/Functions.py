# Functions for 'Hexagonal.py"

# Imports
import numpy as np

# Variables
Lattice_Constant = 2.46
Repetitions = 10
A_Coordinates = np.zeros((Repetitions**2, 2))
B_Coordinates = np.zeros((Repetitions**2, 2))
primitive_lattice_vector_1 = np.zeros((1, 2))
primitive_lattice_vector_2 = np.zeros((1, 2))
B_Offset = np.array([Lattice_Constant/2, Lattice_Constant/(2*np.sqrt(3))])

# Renaming
a = Lattice_Constant
r = Repetitions
co1 = A_Coordinates
co2 = B_Coordinates
plv1 = primitive_lattice_vector_1
plv2 = primitive_lattice_vector_2
b = B_Offset

# Functions
def PLV(a, plv1, plv2):
    plv1[0] = [a, 0]
    plv2[0] = [a/2, a*np.sqrt(3)/2]
    return plv1, plv2

def Build(x, y, b, co1, co2, plv1, plv2, c):
    co1[c] = x*plv1 + y*plv2
    co2[c] = co1[c] + b
    return co1, co2
