# This is a more costumizable version of 'OrbitC'. Note that the initial position and velocity values only work for x position and y velocity.
# The period and increments functions from 'Keplerian.py' are not called. I calculated radius as a function of angle, not time.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from Keplerian import *

# Variables
array1 = np.zeros((360, 2))
x_0 = 10
v_0 = 0.1
l = 500

# Solving for variables in the equation for the radius.
p = slr(x_0, v_0)
e_0 = energy(x_0, v_0)
a = sma(e_0)
e = eccentricity(p, a)

for i in range(360):
    theta = i*np.pi/180
    r = radius(p, e, theta)
    x = x_comp(r, theta)
    y = y_comp(r, theta)
    array1[i] = [x, y]

x = array1[:, 0]
y = array1[:, 1]

plt.plot(x, y, label='Trajectory', color='blue')
plt.plot(0, 0, 'ro', label='Planet', color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.title('OrbitD')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
