# This file measures the energy error in 'OrbitA' along the trajectory.

# Imports
import matplotlib.pyplot as plt
import numpy as np
from OrbitC import xe
from OrbitC import ye
from OrbitC import l

xe = np.linspace(0, 2*np.pi, l)  # 100 points from 0 to 10

plt.plot(xe, ye, label='Error', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('OrbitC Energy Error')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
