# This is a more costumizable version of 'OrbitC'. Note that the initial position and velocity values only work for x position and y velocity.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from Keplerian import *

# Variables
theta = np.linspace(0, 2*np.pi ,360)
array1 = np.zeros((360, 2))
x_0 = 10
v_0 = 0.1
l = 500

# Solving for variables in the equation for the radius.
p = slr(x_0, v_0)
e_0 = energy(x_0, v_0)
a = sma(e_0)
e = eccentricity(p, a)
t = period(a)
dt = increments(t, l)

for i in range(360):
    theta = i
    r = radius(p, e, theta)
    x = x_comp(r, theta)
    y = y_comp(r, theta)
    array1[i] = [x, y]

print(array1)