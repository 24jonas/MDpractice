import math
import matplotlib.pyplot as plt
import numpy as np 

box_size = 6.0
steps = 100
dr =0.05
Num_particles=27
box_length = 6
Volume = box_length**3
origin_particle =0

particle_points = []
for i in range(1, 6, 2):
    for j in range(1, 6, 2):
        for k in range(1, 6, 2):
            y_val = j + 0.75 if k == 3 else j
            particle_points.append((i, y_val, k))

Num_particles = len(particle_points)
origin_particle = 0
r_xj = []
B_nbin = [0]
r_values = []
g_r = []

for step in range(100):
    point = step % Num_particles

    if point == 0 and step != 0:
        origin_particle += 1
        if origin_particle >= Num_particles:
            origin_particle = 0

    rx = particle_points[origin_particle][0] - particle_points[point][0]
    ry = particle_points[origin_particle][1] - particle_points[point][1]
    r = (rx**2 + ry**2)**0.5
    r_xj.append(r)

    nbin = int(r / dr) + 1
    if step == 0:
        B_nbin.append(nbin + 1)
    else:
        B_nbin.append((nbin + 1) + B_nbin[step])

    r_values.append(nbin + step)

    # Avoid division by zero
    if r_values[step] != 0:
        g_r.append((Volume / Num_particles) * (B_nbin[step + 1]) / (Num_particles * 4 * np.pi * (r_values[step] ** 2) * dr))
    else:
        g_r.append(0)

    plt.figure(figsize=(8, 5))
plt.plot(r_values, g_r, label="g(r)", color='green')
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Pair Correlation Function g(r)")
plt.grid(True)
plt.tight_layout()
plt.show()
