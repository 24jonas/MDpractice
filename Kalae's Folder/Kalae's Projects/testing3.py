import math
import matplotlib.pyplot as plt
import numpy as np 

# constants
box_size = 6.0
dr = 0.05
Num_particles = 27
box_length = 6
Volume = box_length**3
nbin = int((box_size/2) / dr)
origin_particle = 0

# arrays
Bn_bins = np.zeros(nbin)
r_values = np.arange(nbin) * dr

# create the particle points
particle_points = []
for i in range(1, 6, 2):
    for j in range(1, 6, 2):
        for k in range(1, 6, 2):
            y_val = j + 0.75 if k == 3 else j
            particle_points.append((i, y_val, k))

# Calculate all pairwise distances (excluding self-pairs)
for origin_particle in range(Num_particles):
    for point in range(Num_particles):
        if origin_particle != point:
            rx = particle_points[origin_particle][0] - particle_points[point][0]
            ry = particle_points[origin_particle][1] - particle_points[point][1]
            rz = particle_points[origin_particle][2] - particle_points[point][2]
            r = (rx**2 + ry**2 + rz**2)**0.5
            bin_index = int(r / dr)
            if bin_index < nbin:
                Bn_bins[bin_index] += 1

# Normalize to get g(r)
g_r = []
for i in range(nbin):
    r_outer = (i + 1) * dr
    r_inner = i * dr
    shell_vol = (4/3) * np.pi * (r_outer**3 - r_inner**3)
    #ideal = Num_particles * (Num_particles - 1) / Volume * shell_vol
    g_r.append((shell_vol/Num_particles)*)             #Bn_bins[i] / ideal if ideal > 0 else 0)

# Plot
plt.figure(figsize=(6, 5))
plt.plot(r_values, g_r, label="g(r)", color='blue')
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Pair Correlation Function g(r)")
plt.grid(True)
plt.tight_layout()
plt.show()