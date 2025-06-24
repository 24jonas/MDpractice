import math
import matplotlib.pyplot as plt
import numpy as np 





# constants
box_size = 6.0
steps = 100
dr =0.05
Num_particles=27
box_length = 6
Volume = box_length**3
origin_particle =0


# arrays
B_nbin = [0]
r_xj = []
r_values = []
g_r = [] # equivalent to (V/N)* p(r)
 

# create the particle points.

particle_points = []
for i in range(1, 6, 2):
    for j in range(1, 6, 2):
        for k in range(1, 6, 2):
            if k == 3:
                particle_points.append((i, j + 0.75, k))
            else:
                particle_points.append((i, j, k))
#for i in range(steps):
    for step in range(100):

        
        point  = step % 27

        if point == 0:
            origin_particle +=1

        print("j:", step, "len(particle_points):", len(particle_points))
        rx = particle_points[origin_particle][0] - particle_points[point][0]
        ry = particle_points[origin_particle][1] - particle_points[point][1]
        
        r_xj.append((rx**2 + ry**2)**(0.5))

        nbin = int(r_xj[step]/dr) +1
        B_nbin.append((nbin+1)+B_nbin[step-1])    
        r_values.append(nbin+step)

        g_r.append((Volume/Num_particles)*(B_nbin[step])/(Num_particles*4*np.pi*(r_values[step]**2) * dr))



plt.figure(figsize=(8, 5))
plt.plot(r_values, g_r, label="g(r)", color='green')
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Pair Correlation Function g(r)")
plt.grid(True)
plt.tight_layout()
plt.show()


