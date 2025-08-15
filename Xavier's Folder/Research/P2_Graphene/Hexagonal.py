# Generate a plot of hexagonal cells utilizing primitive lattice vectors and two particles per cell.

# Imports
import matplotlib.pyplot as plt
from Functions import *

# Construct the PLVs.
plv1, plv2 = PLV(a, plv1, plv2)

# Build the lattice.
c = count = 0
for y in range(r):
    for x in range(r):
        co1, co2 = Build(x, y, b, co1, co2, plv1, plv2, c)
        c += 1

# Plot the coordinates.
x1, y1 = co1[:, 0], co1[:, 1]
x2, y2 = co2[:, 0], co2[:, 1]
plt.scatter(x1, y1, color='blue', label='A Atoms')
plt.scatter(x2, y2, color='red', label='B Atoms')
plt.title("Graphene Lattice")
plt.xlabel("X Distance (Angstroms)")
plt.ylabel("Y Distance (Angstroms)")
plt.gca().set_aspect('equal')
plt.legend()
plt.grid(True)
plt.show()
