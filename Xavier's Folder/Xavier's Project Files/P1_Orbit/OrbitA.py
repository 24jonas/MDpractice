# This file produces an orbit path based on the verlet algorithm given initial values specified in the homework. You must save VerletA at least 
# once in order to run this file. Save VerletA after it's edited.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from VerletA import *

# Array
a = np.zeros((5000,2))

# Initial
x_i = 10
y_i = -0.1
# x_i = 9.98479 & y_i = -0.174285based on InitialA but this cause an early approximation error to propogate.
x = 10
y = 0

# Constructs the array with ordered pairs.
for i in range(5000):
    a[i] = [x, y]

    r1 = radius1(x, y)
    theta1 = angle1(x, y)
    a1_x = acc1_x(r1, theta1, x)
    a1_y = acc1_y(r1, theta1)
    x_f = dx(x, x_i, a1_x)
    y_f = dy(y, y_i, a1_y)

    x_i = x
    y_i = y
    x = x_f
    y = y_f

# Plots the ordered pairs of the array.
x = a[:, 0]
y = a[:, 1]

plt.plot(x, y, label='Trajectory', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('OrbitA')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

# So far the orbit is not correct. The trajectory looks like an strange kind of slingshot.
