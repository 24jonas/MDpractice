# Using the initial conditions provided in the homework, I caclulated eccentricity to be 0.9, and p to be 1.

import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi ,360)
r = (1 - (0.9)*np.cos(theta)) ** (-1)       # The equation for the radius as a function of theta.

# 1 - ecos() is used here because it gives the correct initial x value. Using "1 + ecos()" just flips the orbit about the y-axis.

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
ax.plot(theta, r)
plt.plot(0, 0, 'ro', label='Planet', color='black')
plt.show()
