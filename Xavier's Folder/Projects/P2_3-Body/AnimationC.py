# Animate 'OrbitC'. The animation code was written by ChatGPT.

# Imports
from OrbitC import co, c1, c2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Animate the trajectory
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.set_title("Restricted Three-Body Animation")
ax.grid()

# Markers for the bodies
object, = ax.plot([], [], 'ro', label='Object', color='black')
star1, = ax.plot([], [], 'bo', label='Star1', color='red')
star2, = ax.plot([], [], 'go', label='Star2', color='blue')

# Trails for the bodies
trail1, = ax.plot([], [], 'r-', linewidth=0.5, color='black')
trail2, = ax.plot([], [], 'b-', linewidth=0.5, color='red')
trail3, = ax.plot([], [], 'g-', linewidth=0.5, color='blue')

def init():
    object.set_data([], [])
    star1.set_data([], [])
    star2.set_data([], [])
    trail1.set_data([], [])
    trail2.set_data([], [])
    trail3.set_data([], [])
    return object, star1, star2, trail1, trail2, trail3

def update(frame):
    x1, y1 = co[frame]
    x2, y2 = c1[frame]
    x3, y3 = c2[frame]

    object.set_data([x1], [y1])
    star1.set_data([x2], [y2])
    star2.set_data([x3], [y3])

    trail1.set_data(co[:frame+1, 0], co[:frame+1, 1])
    trail2.set_data(c1[:frame+1, 0], c1[:frame+1, 1])
    trail3.set_data(c2[:frame+1, 0], c2[:frame+1, 1])

    return object, star1, star2, trail1, trail2, trail3

ani = animation.FuncAnimation(
    fig, update, frames=len(co),
    init_func=init, blit=True, interval=5
)

plt.legend()
plt.show()
