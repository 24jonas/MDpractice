import numpy as np
# main.py
from grapheneSheetGen import generate_graphene_sheet
import matplotlib
matplotlib.use('Agg') ## generates static image instead of gui. Used for wsl
import matplotlib.pyplot as plt

a = 2.46
nx, ny = 10, 10
positions, supercell = generate_graphene_sheet(a=a, nx=nx, ny=ny)

# Test Plot
plt.figure(figsize=(6, 6))
plt.scatter(positions[:, 0], positions[:, 1], c=['blue' if i % 2 == 0 else 'orange' for i in range(len(positions))])
plt.gca().set_aspect('equal') # Set aspect ration to equal to make sure graph bugs are from calculations
plt.title(f"Graphene sheet")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.savefig('graphene.png')


## Helped a lot with debugging
def count_nearest_neighbors(positions, cutoff=1.6):
    """
    Count how many neighbors each atom has within the cutoff distance.

    Parameters:
        positions (np.ndarray): Nx2 array of atomic positions (Cartesian).
        cutoff (float): Distance cutoff for neighbor search (e.g., ~1.42 Å bond length).

    Returns:
        neighbors_counts (np.ndarray): Array of length N with neighbor counts.
    """
    N = len(positions)
    neighbors_counts = np.zeros(N, dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= cutoff:
                neighbors_counts[i] += 1
                neighbors_counts[j] += 1

    return neighbors_counts

neighbor_counts = count_nearest_neighbors(positions)

print(f"Average nearest neighbors per atom: {np.mean(neighbor_counts)}")