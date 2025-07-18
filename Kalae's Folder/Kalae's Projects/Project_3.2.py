import numpy as np 
import matplotlib.pyplot as plt

# -1/2 * [(r-r_1)/(d_1)**3]-1/2 * 
# [(r-r_2)/(d_2)**3] = 0
#time
t = 0

r_1 = [-0.5*np.cos(t),-0.5*np.sin(t)]
r_1_absolutevalue = 0.5
r_x= []
r_2 = [0.5*np.cos(t),0.5*np.sin(t)]
r_2_absolutevalue = 0.5
m_1 = 1/2
m_2 = 1/2
#v_0= [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
#theta_0= [np.pi/1.8, np.pi/2, np.pi/2.2, np.pi/2.4, np.pi/1.7, np.pi/2.3, np.pi/2.1]


v_0=.6
theta_0 = np.pi/1.7 # Initial angle of the velocity of the planet
v_x = [v_0*np.cos(t)]  # Initial x velocity
v_y = [v_0*np.sin(t)]  # Initial y velocity
v_001 = [v_x, v_y]  # Initial velocity vector
steps = 2000
dt = 0.001
x = [0]  # Initial x position
y = [0]  # Initial y position
d_1 = np.linalg.norm(np.array(r_1) - np.array([x[0], y[0]]))  # Distance to star 1
d_2 = np.linalg.norm(np.array(r_2) - np.array([x[0], y[0]]))  # Distance to star 2
for i in range(steps):
        d_1 = np.linalg.norm(np.array(r_1) - np.array([x[i], y[i]]))  # Distance to star 1
        d_2 = np.linalg.norm(np.array(r_2) - np.array([x[i], y[i]]))  # Distance to star 2
        Acceleration_x = (-1/2 * (x[i]-r_1_absolutevalue)/(d_1)**3-1/2 * (x[i]-r_2_absolutevalue)/(d_2)**3) 
        Acceleration_y = (-1/2 * (y[i])/(d_1)**3-1/2 * (y[i])/(d_2)**3) 
        v_x.append(v_x[i] + Acceleration_x * dt)
        v_y.append(v_y[i] + Acceleration_y * dt)
        x.append(x[i] + v_x[i+1] * dt)
        y.append(y[i] + v_y[i+1] * dt)
        r_1 = [-0.5*np.cos(t),-0.5*np.sin(t)]
        r_2 = [0.5*np.cos(t),0.5*np.sin(t)]
        t+= 1



#two stars of equal mass orbiting eachother.
#third body at the origin.

plt.plot(r_1[0], r_1[1], 'ro', label='Star 1')  # Star 1 at (-0.5, 0)
plt.plot(r_2[0], r_2[1], 'bo', label='Star 2')  # Star 2 at (0.5, 0)
plt.plot(x, y, label='Orbit', color='green')
plt.title( "Two stationary stars")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.grid(True)
plt.legend()
plt.axis('equal')  # Equal scaling for x and y axes
plt.show()
