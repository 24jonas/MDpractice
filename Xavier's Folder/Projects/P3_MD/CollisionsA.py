# Models the behavior of two masses and elastic collisions.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from FunctionsA import *

# Adjust Variables
l = 4000
co1 = np.zeros((l,2))
co2 = np.zeros((l,2))

# Construct Coordinate Arrays
for i in range(l):
    a, dr = Check1(r1, r2, cr)
    b = Check2(r1, blx, bly)
    c = Check3(r2, blx, bly)

    if a == True:
        n1, n2 = Contact(dr, rm)
        v1, v2 = Components(v1, v2, n1, n2)     # Fix boundary behavior by using radius from center.
    elif b == True:
        v1[0] = -v1[0]
    elif b == False:
        v1[1] = -v1[1]
    elif c == True:
        v2[0] = -v2[0]
    elif c == False:
        v2[1] = -v2[1]
    
    r1, r2 = Position(r1, r2, v1, v2, dt)
    co1[i] = r1
    co2[i] = r2

# Plot Coordinate Arrays
x1 = co1[:, 0]
y1 = co1[:, 1]
x2 = co2[:, 0]
y2 = co2[:, 1]

plt.plot(x1, y1, label='Trajectory', color='red')
plt.plot(x2, y2, label='Trajectory', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Collision')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
