import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi ,360)
r = (1 - (0.9)*np.cos(theta)) ** (-1)       # Using "1 + ecos()" just flips the orbit about the y-axis.

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
ax.plot(theta, r)
plt.show()

# 1 - ecos() is used here because it gives the correct initial x value.