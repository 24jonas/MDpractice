# Produces an orbit based on the version of the Verlet algorithm shown in the finished homework. - Corrected.

# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from VerletF import *       # A few of the functions in 'VerletF' don't get used.

# Variables
x = 10
y = 0
v_x = 0
v_y = 0.1
l = 100000
array1 = np.zeros((l,2))
array2 = np.zeros((l,2))

# Solving for variables in the equation for the radius.
p = slr(x, v_y)
e_0 = energy_0(x, v_y)
a = sma(e_0)
ecc = eccentricity(p, a)
t = period(a)
dt = t/10000
r = radius(x, y)

# Constructing the array with coordinates.
for i in range(l):
    v = vel(v_x, v_y)
    e = energy(v, r)
    err = error(e, e_0)
    array1[i] = [x, y]
    array2[i] = [i, err]

    x, y, v_x, v_y = dr(x, y, v_x, v_y, dt)

# Plots the ordered pairs of the array.
x3 = array1[:, 0]
y3 = array1[:, 1]
xe = array2[:, 0]
ye = array2[:, 1]

plt.plot(x3, y3, label='Trajectory', color='blue')
plt.plot(0, 0, 'ro', label='Planet', color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.title('OrbitF')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
