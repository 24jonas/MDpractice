# This file measures the energy error in 'OrbitF' along the trajectory.

# Imports
import matplotlib.pyplot as plt
import numpy as np
from OrbitF import xe
from OrbitF import ye
from OrbitF import l

xe = np.linspace(0, 2*np.pi, l)

plt.plot(xe, ye, label='Error', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('OrbitF Energy Error')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
