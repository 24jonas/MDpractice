
import math
import matplotlib.pyplot as plt
import numpy as np 

# Simulation Settings 
box_size = 6.0
dt = 0.005
steps = 10000
sample_per = 10  # (this will be used for plotting intervals)


# constants
steps = 100
dr =0.05
Num_particles=27
Volume = box_size**3
origin_particle =0
counter = 0
nbin = 100


# arrays
B_nbin =  np.zeros(nbin)
r_xj = []
r_values = []
g_r = [] 
nbins = [] 

# Step 1: Initialize particles in a staggered checkerboard 
particle_points = []
for i in range(1, 6, 2):
    for j in range(1, 6, 2):
        for k in range(1, 6, 2):
            if k == 3:
                particle_points.append((i, j + 0.75, k))
            else:
                particle_points.append((i, j, k))

N = len(particle_points)

# Initial velocities: all zero
vx = [0.0] * N
vy = [0.0] * N
vz = [0.0] * N

# Energy trackers
kinetic_energy_list = []
potential_energy_list = []

# Main Simulation Loop
for step in range(steps):
    ax = [0.0] * N
    ay = [0.0] * N
    az = [0.0] * N
    potential_energy = 0.0
    kinetic_energy = 0.0

    # Lennard-Jones force calculation
    for i in range(N):
        for j in range(i + 1, N):
            rx = particle_points[i][0] - particle_points[j][0]
            ry = particle_points[i][1] - particle_points[j][1]
            rz = particle_points[i][2] - particle_points[j][2]
            r = math.sqrt(rx**2 + ry**2 + rz**2)

            if r != 0:
                inv_r6 = (1 / r)**6
                inv_r12 = inv_r6**2
                f_mag = 24 * (2 * inv_r12 - inv_r6) / r

                fx = f_mag * (rx / r)
                fy = f_mag * (ry / r)
                fz = f_mag * (rz / r)

                ax[i] += fx
                ay[i] += fy
                az[i] += fz

                ax[j] -= fx  # Apply Newton's third law
                ay[j] -= fy
                az[j] -= fz

                potential_energy += 4 * (inv_r12 - inv_r6)

    # Update velocity and position
    new_positions = []
    for i in range(N):
        vx[i] += ax[i] * dt
        vy[i] += ay[i] * dt
        vz[i] += az[i] * dt

        kinetic_energy += 0.5 * (vx[i]**2 + vy[i]**2 + vz[i]**2)

        # Update positions
        x_new = particle_points[i][0] + vx[i] * dt
        y_new = particle_points[i][1] + vy[i] * dt
        z_new = particle_points[i][2] + vz[i] * dt

           # Periodic Conditions.
        if x_new <= 0:
            x_new = box_size
            
        elif x_new >= box_size:
            x_new = 0
            
        if y_new <= 0:
            y_new = box_size
            
        elif y_new >= box_size:
            y_new = 0
            
        if z_new <= 0:
            z_new = box_size
            
        elif z_new >= box_size:
            z_new = 0

        """
        # Wall bounce conditions
        if x_new <= 0 or x_new >= box_size:
            vx[i] *= -1
            x_new = max(0, min(x_new, box_size))

        if y_new <= 0 or y_new >= box_size:
            vy[i] *= -1
            y_new = max(0, min(y_new, box_size))

        if z_new <= 0 or z_new >= box_size:
            vz[i] *= -1
            z_new = max(0, min(z_new, box_size))

        """

        new_positions.append((x_new, y_new, z_new))  # Correct list

    particle_points = new_positions  # Update all positions at once

    # Sample energies at regular intervals
    if step % sample_per == 0:
        kinetic_energy_list.append(kinetic_energy)
        potential_energy_list.append(potential_energy)


        #for i in range(steps):
for step in range(100):

        
        point  = step % 27

        if point == 0:
            origin_particle +=1

        if origin_particle != point:
             
            print("j:", point, "origin_particle", origin_particle)
            rx = particle_points[origin_particle][0] - particle_points[point][0]
            ry = particle_points[origin_particle][1] - particle_points[point][1]
            rz = particle_points[origin_particle][2] - particle_points[point][2]
        
            r_xj.append((rx**2 + ry**2+rz**2)**(0.5))


            #nbin = int(r_xj[counter]/dr)+1
            nbin = counter + 1
            B_nbin[nbin] = B_nbin[nbin]+1
            r_values.append(nbin*dr)

            r_outer = (counter+1)*dr
            r_inner = counter*dr

            shell_vol = 4*np.pi*(r_outer**3-r_inner**3)/3


            #p_r = (B_nbin[counter])/(Num_particles * 4*np.pi*((r_values[counter]*dr)**2) * dr)
            nbin
            p_r = (B_nbin[counter])/(Num_particles * 4*np.pi*((nbin*dr)**3) * dr)
            g_r.append(((shell_vol)/Num_particles)*p_r)
            counter += 1




plt.figure(figsize=(6, 5))
plt.plot(r_values, g_r, label="g(r)", color='blue')
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Pair Correlation Function g(r)")
plt.grid(True)
plt.tight_layout()
plt.show()

