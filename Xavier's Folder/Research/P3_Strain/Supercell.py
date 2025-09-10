# Find supercell vectors.

# Imports
from Bilayer import *
from Angle import m, n

# Calculate supercell lattice vectors.
# if n % 3 == 0:
#     slv1, slv2 = SLVA(m, n, plv1, plv2, slv1, slv2)
# else:
#     slv1, slv2 = SLVB(m, n, plv1, plv2, slv1, slv2)

############################################
plv1 = plv1[0]
plv2 = plv2[0]
unit_cell = np.array([plv1[:2],plv2[:2]])
slv1, slv2 = SLV(m, n, unit_cell)

slv1 = np.append(slv1,0)
slv2 = np.append(slv2,0)
############################################

# Isolate one supercell.
co4, angle = Reorient(co4, slv1, slv2)
for i in range(2*Repetitions**2):
    co1 = Twist(co1, -angle, i)
for i in range(2*Repetitions**2):
    co2 = Twist(co2, -angle, i)
for i in range(2*Repetitions**2):
    dc1 = Shave(co1, co4, dc1, i)
for i in range(2*Repetitions**2):
    dc2 = Shave(co2, co4, dc2, i)
dc1.reverse()
dc2.reverse()
for i in dc1:
    co1 = np.delete(co1, i, axis=0)
for i in dc2:
    co2 = np.delete(co2, i, axis=0)

# Supercell for ASE.
supercell = np.zeros((3,3))
supercell[0] = co4[1] - co4[0]
supercell[1] = co4[2] - co4[0]
supercell[2] = 2*z

print(np.linalg.norm(supercell[0]))

# Plot the supercell.
x1, y1, z1 = co1[:,0], co1[:,1], co1[:,2]
x2, y2, z2 = co2[:,0], co2[:,1], co2[:,2]
x3, y3, z3 = co3[:,0], co3[:,1], co3[:,2]
x4, y4, z4 = co4[:,0], co4[:,1], co4[:,2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, y1, z1, c='blue', s=50)
ax.scatter(x2, y2, z2, c='red', s=50)
ax.scatter(x3, y3, z3, c='purple', s=50)
# ax.scatter(x4, y4, z4, c='purple', s=50)  # Outline of the supercell.
plt.show()

# How many atoms in the supercell?
print(f"The are {len(x1) + len(x2)} atoms in the supercell.")
