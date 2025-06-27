# Animation for 'CollisionsC.py'. The animation code was written by ChatGPT. It is not my own.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from CollisionsC import rx, ry, rz, l, ro, blx, q

# === Adjustable parameters ===
num_objects = q
radius = ro                 # Radius of each object
cube_bounds = [0, blx]      # Min and max for x, y, z axes

# === Example data === (Replace with your actual data)
x_vals = rx
y_vals = ry
z_vals = rz

# === Set up figure and axes ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(cube_bounds)
ax.set_ylim(cube_bounds)
ax.set_zlim(cube_bounds)
ax.set_box_aspect([1,1,1])  # Equal aspect ratio

# Initialize scatter plot
scatters = [ax.plot([], [], [], 'o', markersize=radius * 10, label=f'Obj {i+1}')[0] for i in range(num_objects)]

# Progress Bar Setup
total_frames = l  # or however many frames you have
progress = tqdm(total=total_frames)

# === Animation function ===
def update(frame):
    for i, scatter in enumerate(scatters):
        scatter.set_data([x_vals[frame, i]], [y_vals[frame, i]])
        scatter.set_3d_properties([z_vals[frame, i]])
    progress.update(1)  # Update the bar
    return scatters

# Create animation
ani = FuncAnimation(fig, update, frames=l, interval=30, blit=False)

plt.title("3D Object Motion Animation")
# plt.legend(loc='upper right', fontsize='x-small', ncol=3)
plt.tight_layout()
plt.show()

print("---------------------------------------------------------------------------------------------------------------------------------------------------")
