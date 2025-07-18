import numpy as np  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

"""
Devoted to testing blocks of my code.
Kalae's Projects
"""

N= 27
SX =6.0
SY = 6.0
SZ = 6.0
dt = 0.01
timesteps = 10,000
steps = 0
kinetic_energy =0
potential_energy =0
save_every = 10 
box_size = 6 
#sigma =1
#epsilon = 1 # taken out of calculation to make it easier to read

# arrrays to store particle positions and energies
rx = []
ry = []
rz = []
kinetic_energy_list = []
potential_energy_list = []
forces = []

ax = [0]*N
ay = [0]*N
az = [0]*N
running = True
iterations = 0
sub_iterations = 0
vx =[0]*N
vy =[0]*N
vz =[0]*N
x=0
y=0
z=0






particle_points = []
for i in range(1,6,2):
    for j in range(1,6,2):
        for k in range(1,6,2):
            if k == 3:
                j += 0.75
                particle_points.append((i,j,k))
                j = j - 0.75
            else:
                particle_points.append((i,j,k))
while (running == True):



    for particle in range(N):
        for i in range(i+1, N): #calculates the distances between particles needed for the acceleration
            rx =particle_points[particle][0]-particle_points[i][0]
            ry =particle_points[particle][1]-particle_points[i][1]
            rz =particle_points[particle][2]-particle_points[i][2]
            r=((rx**2 + ry**2 + rz**2)**0.5)

            if r != 0:
                inv_r6 = (1/r)**6
                inv_r12 = (inv_r6)**2

                force_total = 24*(2*inv_r12-inv_r6)/r 
    

                fx = force_total * (rx / r)
                fy = force_total * (ry / r)
                fz = force_total * (rz / r)

                ax[i] += fx
                ay[i] += fy
                az[i] += fz

                potential_energy += 4 * (inv_r12 - inv_r6)

             

    new_positions=[]
    for i in range(N):
        vx[i] += ax[i] * dt
        vy[i] += ay[i] * dt
        vz[i] += az[i] * dt

        kinetic_energy += 0.5 * (vx[i]**2 + vy[i]**2 + vz[i]**2)

        x_new = particle_points[i][0]+vx[i]*dt
        y_new = particle_points[i][1]+vy[i]*dt
        z_new = particle_points[i][2]+vz[i]*dt

        if x_new <= 0 or x_new >= box_size:
            vx[i] *= -1
            x_new = max(0, min(x_new, box_size))

        if y_new <= 0 or y_new >= box_size:
            vy[i] *= -1
            y_new = max(0, min(y_new, box_size))

        if z_new <= 0 or z_new >= box_size:
            vz[i] *= -1
            z_new = max(0, min(z_new, box_size))

        new_positions.append((x_new, y_new, z_new))



    particle_points = new_positions
    steps += 1
    print(steps)
    if steps % 20 == 0: 
    
        kinetic_energy_list.append(kinetic_energy)
        potential_energy_list.append(potential_energy)

    if steps == 10000:

        running = False


plt.figure(figsize=(10, 6))
plt.plot(kinetic_energy_list, label="Kinetic Energy", color='blue')
plt.plot(potential_energy_list, label="Potential Energy", color='red')
plt.plot(
    [k + v for k, v in zip(kinetic_energy_list, potential_energy_list)],
    label="Total Energy", linestyle='--', color='black'
)
plt.xlabel("Time Step")
plt.ylabel("Energy")
plt.title("Energy vs. Time ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

        


#for radius in range(len(particle_points )):
#   total_particle_r.append((rx[radius]**2 + ry[radius]**2 + rz[radius]**2)**0.5)

# Create a loop for the varlet algorithm

"""


 
    


#print("rx:", rx)
#print("ry:", ry)
#print("rz:", rz)
# This code initializes a list of particle positions in a cubic box, where particles are staggered in the z-direction.



                   potential_energy.append(4 * ((1/total_particle_r[i])**12 - (1/total_particle_r[i])**6))
                   forces.append(4 * ((12 / (total_particle_r[i]**13)) - (6  / (total_particle_r[i]**7)))) 

                   ax[i] = forces[i] * (rx[i] / total_particle_r[i])
                   ay[i] = forces[i] * (ry[i] / total_particle_r[i])
                   az[i] = forces[i] * (rz[i] / total_particle_r[i])
                   
                  
            else:
                ax.append(0), ay.append(0), az.append(0)  # Avoid division by zero
                forces.append(0)

                vx[i] += ax[i] * dt
                vy[i] += ay[i] * dt
                vz[i] += az[i] * dt


        sub_iterations += 1
        print("Sub-Iteration:", sub_iterations)    

        kinetic_energy = 0.5 * (vx[i]**2 + vy[i]**2 + vz[i]**2)
        particle_points.append((particle_points[particle][0] + vx[i] * dt,
                                    particle_points[particle][1] + vy[i] * dt, 
                                    particle_points[particle][2] + vz[i] * dt))

            
            


    iterations += 1
    print("Iteration:", iterations)
    if iterations >= timesteps:
        running = False




"""