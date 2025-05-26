# Produces an orbit based on the version of the Verlet algorithm shown in the finished homework.

# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from VerletC import *       # A few of the functions in 'VerletC' don't get used.

# Variables
x = 10
y = 0
v_x = 0
v_y = 0.1
l = 10000
array1 = np.zeros((l,2))

# Solving for variables in the equation for the radius.
p = slr(x, v_y)
e_0 = energy(x, v_y)
a = sma(e_0)
e = eccentricity(p, a)
t = period(a)
dt = t/10000
print(dt)

# Constructing the array with coordinates.
for i in range(l):
    array1[i] = [x, y]

    theta = angle(x, y)
    r = radius(x, y)
    a_x = acc_x(r, theta, x)
    a_y = acc_y(r, theta, y)
    x, v_x = dx(x, v_x, a_x, dt)
    y, v_y = dy(y, v_y, a_y, dt)

# Plots the ordered pairs of the array.
x3 = array1[:, 0]
y3 = array1[:, 1]

plt.plot(x3, y3, label='Trajectory', color='blue')
plt.plot(0, 0, 'ro', label='Planet', color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.title('OrbitC')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
