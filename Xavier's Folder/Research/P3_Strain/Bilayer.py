# Build the coordinates of the twisted bilayer.

# Imports
from Functions import *
import matplotlib.pyplot as plt

# Create the PLVs.
plv1, plv2 = PLV(a, plv1, plv2)
s = Initial(s, r, plv1, plv2)

# Build the lattices.
c = count = 0
for y in range(r):
    for x in range(r):
        co1, co2 = Build(x, y, s, z, b, co1, co2, plv1, plv2, c)
        c += 1

# Twist the second one.
for i in range(2*Repetitions**2):
    co2 = Twist(co2, theta, i)

# Scale the axes with vertex points. 50 x 50 x 50.
p1 = np.array([-ws/2, ws/2, 0])
p2 = np.array([-ws/2, -ws/2, ws])
p3 = np.array([ws/2, -ws/2, 0])
co3[0], co3[1], co3[2] = p1, p2, p3

# Plot the coordinates.
x1, y1, z1 = co1[:,0], co1[:,1], co1[:,2]
x2, y2, z2 = co2[:,0], co2[:,1], co2[:,2]
x3, y3, z3 = co3[:,0], co3[:,1], co3[:,2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, y1, z1, c='blue', s=50)
ax.scatter(x2, y2, z2, c='red', s=50)
ax.scatter(x3, y3, z3, c='purple', s=50)
plt.show()
