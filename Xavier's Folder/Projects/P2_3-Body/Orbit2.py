# Simulates the behavior of 3 masses all of which are mobile. Ues the PV algorithm.

# Imports
import matplotlib.pyplot as plt
from Functions2 import *

# Construct the trajectory array.
for i in range(l):
    ao = Acceleration(ro, r1, r2, m1, m2)
    ro, vo = Position_Verlet(ro, vo, ao, dt)
    r1, r2 = Star_Positions(t)
    co[i] = ro
    c1[i] = r1
    c2[i] = r2
    t += dt

# Plot the trajectory.
x = co[:, 0]
y = co[:, 1]

x1 = c1[:, 0]
y1 = c1[:, 1]

x2 = c2[:, 0]
y2 = c2[:, 1]

#plt.plot(x, y, label='Trajectory', color='black')
plt.plot(x1, y1, 'ro', label='Star1', color='red')
plt.plot(x2, x2, 'ro', label='Star2', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Orbit2')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

print("----------------------------------------------------------------------------------------------------------------------------------------")

print("Complete")
