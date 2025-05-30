# Produces an orbit with two stationary masses and a mobile object. Uses the position-verlet (PV) algorithm and a provided equation of motion.
# The initial velocity and position the object along with the star positions can be changed.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from Functions import *

# Optionally Input Variables
v_0 = 1.7
theta = 60
theta = theta*np.pi/180
vo = np.array([v_0*np.cos(theta), v_0*np.sin(theta)])
l = 100000
co = np.zeros((l,2))

# Construct the trajectory array.
for i in range(l):
    ao = Acceleration(ro, r1, r2)
    ro, vo = Position_Verlet(ro, vo, ao, dt)
    co[i] = ro

# Plot the trajectory.
x = co[:, 0]
y = co[:, 1]

plt.plot(x, y, label='Trajectory', color='blue')
plt.plot(-0.5, 0, 'ro', label='Star1', color='black')
plt.plot(0.5, 0, 'ro', label='star2', color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Orbit1')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
