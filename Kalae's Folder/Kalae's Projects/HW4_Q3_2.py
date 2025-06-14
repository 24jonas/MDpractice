import numpy as np  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#constants
             
n_bin = 1
dr = 0.05
B(n_bin) = []

n_bin = int(1/dr)+1
box_length = 6.0
Volume = box_length**3
Num_particles = 27
Pair_density = B(n_bin)/(4*np.pi*(np.arange(n_bin)*dr)**2*dr)



g_r = (Volume/Num_particles)*Pair_density

plt.plot(np.arange(n_bin)*dr, g_r, label='g(r)')
plt.title("Pair Distribution Function g(r)")
plt.xlabel("Distance (r)")
plt.ylabel("g(r)")
plt.grid(True)
plt.legend()
plt.show()
# This code calculates and plots the pair distribution function g(r) for a system of particles in a cubic box.
# It uses a histogram to count the number of pairs of particles at different distances and normalizes it to obtain g(r).
# The pair distribution function is a measure of how particle density varies as a function of distance from a reference particle.
# The code initializes parameters such as the number of bins, bin width, box length, and number of particles.
# It then calculates the pair density for each bin and plots g(r) against distance.         
