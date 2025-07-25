# main.py
from grapheneSheetGen import generate_graphene_sheet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

a = 2.46
nx, ny = 10, 10
positions, supercell = generate_graphene_sheet(a=a, nx=nx, ny=ny)

# Test Plot
plt.figure(figsize=(6, 6))
plt.scatter(positions[:, 0], positions[:, 1], c=['blue' if i % 2 == 0 else 'orange' for i in range(len(positions))])
plt.gca().set_aspect('equal')
plt.title(f"Graphene sheet")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.savefig('graphene.png')