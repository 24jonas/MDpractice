# Collision dynamics of two argon atoms using LJ potential and VV algorithm. Still 2 dimensional.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from FunctionsB import *

# Adjust Variables
l = 5000
co1 = np.zeros((l,2))
co2 = np.zeros((l,2))
x = np.zeros((l,1))
v1a = np.zeros((l,1))
v2a = np.zeros((l,1))
te = np.zeros((l,1))
pe = np.zeros((l,1))
ke = np.zeros((l,1))

# Construct Coordinate Arrays
for i in range(l):
    a, b = Check1(r1, blx, bly)
    c, d = Check2(r2, blx, bly)
    r1, r2, v1, v2, a1, a2, lj = Velocity_Verlet(r1, r2, v1, v2, a1, a2, dt)
    v1, v2, mag1, mag2 = Speed_Limit(v1, v2, sl)
    k1, k2 = Kinetic_Energy(mag1, mag2, m1, m2)

    if a == True:
        v1[0] = -v1[0]
        if b == True:
            r1[0] = or1
        elif b == False:
            r1[0] = blx - or1
    elif a == False:
        v1[1] = -v1[1]
        if b == True:
            r1[1] = or1
        elif b == False:
            r1[1] = bly - or1

    if c == True:
        v2[0] = -v2[0]
        if d == True:
            r2[0] = or2
        elif d == False:
            r2[0] = blx - or2
    elif c == False:
        v2[1] = -v2[1]
        if d == True:
            r2[1] = or2
        elif d == False:
            r2[1] = bly - or2
    
    x[i] = i
    co1[i] = r1
    co2[i] = r2
    v1a[i] = mag1
    v2a[i] = mag2
    te[i] = lj + k1 + k2
    pe[i] = lj
    ke[i] = k1 + k2

# Plot Coordinate Arrays
x1 = co1[:, 0]
y1 = co1[:, 1]
x2 = co2[:, 0]
y2 = co2[:, 1]

plt.plot(x1, y1, label='Trajectory', color='red')
plt.plot(x2, y2, label='Trajectory', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('LJ Collision')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

# Plot x distances over time
plt.plot(x, x1, label="Position", color='red')
plt.plot(x, x2, label="Position", color='blue')
plt.xlabel('Time')
plt.ylabel('X Position')
plt.title('LJ Collision')
plt.grid(True)
plt.legend()
plt.show()

# Plot velocities over time
plt.plot(x, v1a, label="Velocity", color='red')
plt.plot(x, v2a, label="Velocity", color='blue')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('LJ Collision')
plt.grid(True)
plt.legend()
plt.show()

# Plot energy types over time
plt.plot(x, te, label="Total Energy", color='red')
plt.plot(x, pe, label="Potential Energy", color='blue')
plt.plot(x, ke, label="Kinetic Energy", color='purple')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('LJ Collision')
plt.grid(True)
plt.legend()
plt.show()
