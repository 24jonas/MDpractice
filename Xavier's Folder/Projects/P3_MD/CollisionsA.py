# Models the behavior of two masses and elastic collisions.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from FunctionsA import *

# Adjust Variables
l = 5000
co1 = np.zeros((l,2))
co2 = np.zeros((l,2))

# Construct Coordinate Arrays
for i in range(l):
    a, dr = Check1(r1, r2, cr)
    b1, b2 = Check2(r1, blx, bly)
    c1, c2 = Check3(r2, blx, bly)

    if a == True:
        n1, n2 = Contact(dr, rm)
        v1, v2 = Components(v1, v2, n1, n2)
        r1 = co1[i-2]
        r2 = co2[i-2]
    elif b1 == True:
        v1[0] = -v1[0]
        if b2 == True:
            r1[0] = or1
        elif b2 == False:
            r1[0] = blx - or1
    elif b1 == False:
        v1[1] = -v1[1]
        if b2 == True:
            r1[1] = or1
        elif b2 == False:
            r1[1] = bly - or1
    elif c1 == True:
        v2[0] = -v2[0]
        if c2 == True:
            r2[0] = or2
        elif c2 == False:
            r2[0] = blx - or2
    elif c1 == False:
        v2[1] = -v2[1]
        if c2 == True:
            r2[1] = or2
        elif c2 == False:
            r2[1] = bly - or2
    
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
