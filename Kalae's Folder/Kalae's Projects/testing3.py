import numpy as np
import matplotlib.pyplot as plt

# Constants
dr = 0.05
box_size = 6.0
num_particles = 27
volume = box_size**3
nbins = int(box_size / 2 / dr)
B = np.zeros(nbins)

# Create staggered particle points
particle_points = []
for i in range(1, 6, 2):
    for j in range(1, 6, 2):
        for k in range(1, 6, 2):
            if k == 3:
                particle_points.append((i, j + 0.75, k))
            else:
                particle_points.append((i, j, k))

particle_points = np.array(particle_points)

# Accumulate g(r) over many steps
steps = 100
for step in range(steps):
    # Loop over unique pairs only (i < j)
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            dx = particle_points[i][0] - particle_points[j][0]
            dy = particle_points[i][1] - particle_points[j][1]
            

            r = np.sqrt(dx**2 + dy**2)
            bin_index = int(r / dr)

            if bin_index < nbins:
                B[bin_index] += 2  # count both (i,j) and (j,i)

# Normalize to get g(r)
r_vals = np.arange(nbins) * dr
shell_volumes = 4 * np.pi * (r_vals**2) * dr
density = num_particles / volume
g_r = B / (steps * num_particles * density * shell_volumes)

# Plot
plt.figure(figsize=(6, 5))
plt.plot(r_vals, g_r, label="g(r)", color='blue')
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Pair Correlation Function g(r)")
plt.grid(True)
plt.tight_layout()
plt.show()
