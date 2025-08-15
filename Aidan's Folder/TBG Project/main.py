import numpy as np
import matplotlib
matplotlib.use('Agg') ## generates static image instead of gui. Used for wsl
import matplotlib.pyplot as plt

from geometry import generate_graphene_sheet
from misc import count_nearest_neighbors

## main.py ##

# Build the sheet #
a = 2.46
nx, ny = 15, 15
positions, supercell = generate_graphene_sheet(a=a, nx=nx, ny=ny)

# Test Plot #
plt.figure(figsize=(6, 6))
plt.scatter(positions[:, 0], positions[:, 1], c=['blue' if i % 2 == 0 else 'orange' for i in range(len(positions))])
plt.gca().set_aspect('equal') # Set aspect ration to equal to make sure graph bugs are from calculations
plt.title(f"Graphene sheet")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.savefig('graphene.png')

# Count nearest neighbors #
neighbor_counts = count_nearest_neighbors(positions, supercell, cutoff=1.6)
print(f"Atoms: {len(positions)} | Average nearest neighbors: {np.mean(neighbor_counts):.3f}")

