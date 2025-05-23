# This file produces an orbit path based on the verlet algorithm given initial values specified in the homework. You must save VerletA at least 
# once in order to run this file. Save VerletA after it's edited.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from VerletB import *

# Array
l = 10000
a = np.zeros((l,2))

# Initial
vi_x = 0
vi_y = 0.1
x = 10
y = 0
ai_x = -0.01
ai_y = 0

# Constructs the array with ordered pairs.
for i in range(l):
    a[i] = [x, y]

    r = radius(x, y)
    theta = angle(x, y)
    a_x = acc_x(r, theta, x)
    a_y = acc_y(r, theta)
    v_x = vel_x(vi_x, ai_x)
    v_y = vel_y(vi_y, ai_y)
    x_f = dx(x, v_x, a_x)
    y_f = dx(y, v_y, a_y)

    x = x_f
    y = y_f
    vi_x = v_x
    vi_y = v_y
    ai_x = a_x
    ai_y = a_y

# Plots the ordered pairs of the array.
x = a[:, 0]
y = a[:, 1]

plt.plot(x, y, label='Trajectory', color='blue')
plt.plot(0, 0, 'ro', label='Planet', color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.title('OrbitB')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
