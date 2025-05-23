# Produces an array with ordered pairs calculated from the given equation for radius.

import matplotlib.pyplot as plt
import numpy as np

array1 = np.zeros((360,2))

def initial():
    for i in range(360):
        x = (np.cos(i*np.pi/180))/(1 - (0.9)*np.cos(i*np.pi/180))
        y = (np.sin(i*np.pi/180))/(1 - (0.9)*np.cos(i*np.pi/180))
        array1[i] = [x, y]
    return array1


# This prints the last coordinate before the orbit repeats.
b = initial()[-1]
print(f"x: {b[0]:.5e}, y: {b[1]:.5e}")

# This prints the coordinate at halfway through the orbit. This is the lowest point radially.
c = initial()[180]
print(f"x: {c[0]:.5e}, y: {c[1]:.5e}")
