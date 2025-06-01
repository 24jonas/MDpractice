# Produces an orbit with two stationary masses and a mobile object. Uses the position-verlet (PV) algorithm and a provided equation of motion.
# The initial velocity and direction of the object are determined by a seperate process to find stable initial conditions which takes a while.
# The final graph will be blank, and there's a slight delay between graphs being produced after the last one was closed.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from Functions1 import *
from Optimize1 import solutions

# Cycle through solutions.
for i in range(len(solutions)):
    pair = solutions[i]
    v_0 = pair[0]
    theta = pair[1]

    vo = np.array([v_0*np.cos(theta), v_0*np.sin(theta)])
    co = np.zeros((l,2))
    ro = np.array([x_0, y_0])

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

print("----------------------------------------------------------------------------------------------------------------------------------------")

print("Complete")
