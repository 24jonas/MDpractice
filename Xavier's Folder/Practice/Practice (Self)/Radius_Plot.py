import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0, 2*np.pi ,360)
r = (1 + (0.9)*np.cos(theta)) ** (-1)

plt.plot(theta, r, label='Radius')
plt.title('Orbital Radius')
plt.xlabel('Degrees')
plt.ylabel('Radius')
plt.legend()
plt.grid(True)
plt.show()
