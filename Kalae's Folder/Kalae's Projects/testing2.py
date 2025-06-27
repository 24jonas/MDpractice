import math
import matplotlib.pyplot as plt
import numpy as np 





# constants
box_size = 6.0
steps = 100
dr =0.05
Num_particles=27
box_length = 6
Volume = box_length**2
origin_particle =0
counter = 0
nbin = 100


# arrays
B_nbin =  np.zeros(nbin)
r_xj = []
r_values = []
g_r = [] 
nbins = []                # equivalent to (V/N)* p(r)
 

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

        if origin_particle != point:
             
            print("j:", point, "origin_particle", origin_particle)
            rx = particle_points[origin_particle][0] - particle_points[point][0]
            ry = particle_points[origin_particle][1] - particle_points[point][1]
            rz = particle_points[origin_particle][2] - particle_points[point][2]
        
            r_xj.append(rx**2 + ry**2+rz**2)


            nbins.append(int(r_xj[counter]/dr) +1)
            B_nbin[counter] = nbins[counter]+counter
            r_values.append(nbins[counter]*dr)

            shell_vol = 4*np.pi*((dr*(counter+1))**3-(dr*counter)**3)/3

            g_r.append((shell_vol)*(B_nbin[counter])/(Num_particles**2 * 4*np.pi*((r_values[counter]*dr)**2) * dr))
            counter += 1




plt.figure(figsize=(6, 5))
plt.plot(r_values, g_r, label="g(r)", color='blue')
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Pair Correlation Function g(r)")
plt.grid(True)
plt.tight_layout()
plt.show()


