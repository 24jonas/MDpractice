# Animation for 'CollisionsB.py'. The animation code was written by ChatGPT.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from CollisionsB import co1, co2, or1, or2

# co1 and co2 are assumed to be (l, 2) arrays
l = len(co1)

# Setup plot
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.set_title("CollisionB Animation")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

# Objects
mass1, = ax.plot([], [], 'ro')               # red dot
mass2, = ax.plot([], [], 'bo')               # blue dot

# Altering Circles
radius1 = or1
radius2 = or2
mass1 = Circle((0, 0), radius=radius1, fc='red')   # red filled circle
mass2 = Circle((0, 0), radius=radius2, fc='blue')  # blue filled circle
ax.add_patch(mass1)
ax.add_patch(mass2)

def init():
    mass1.center = (-100, -100)  # move it off screen or initialize properly
    mass2.center = (-100, -100)
    return mass1, mass2

def update(frame):
    x1, y1 = co1[frame]
    x2, y2 = co2[frame]

    mass1.center = (x1, y1)
    mass2.center = (x2, y2)

    return mass1, mass2

ani = animation.FuncAnimation(
    fig, update, frames=len(co1),
    init_func=init, blit=False, interval=10
)

plt.show()
