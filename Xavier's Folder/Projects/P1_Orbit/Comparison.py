# This file while plot multiple orbital paths and compare the methods which generated them.

# Imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from OrbitA import x1
from OrbitA import y1
from OrbitB import x2
from OrbitB import y2
from OrbitC import x3
from OrbitC import y3
from OrbitE import x5
from OrbitE import y5

plt.plot(x1, y1, label = 'OrbitA', color = 'red')
plt.plot(x2, y2, label = 'OrbitB', color = 'green')
plt.plot(x3, y3, label = 'OrbitC', color = 'blue')
plt.plot(x5, y5, label = 'OrbitE', color = 'purple')
plt.plot(0, 0, 'ro', label='Planet', color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Orbital Comparison')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
