import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Constants From previous problem. 

x_0=10
yv_0= 1/10
h=x_0*yv_0
Energy_0= 1/2*yv_0**2-1/x_0
a= -1/(2*Energy_0)
p=h**2
# Constants for integration
dt = p/500
T= 10
steps = 100000 #int(T / dt)
G= 1 # simplified gravitational constant.
m=1 # simplified mass of body.
e=np.sqrt(1-p/a)
theta= 0  
radius = np.sqrt(x_0**2 + 0**2)  # Initial radius
r_minus = p/(1-e*np.cos(theta))
Acceleration_x= G*m/r_minus**2
Acceleration_y= G*m/r_minus**2

#Initial conditions
x = [x_0]
y = [0]
v_y = [yv_0]
v_x=[0]


# Euler-Cromer Integration
for i in range(steps):
    if (x[i] > 0):
        Acceleration_x = (G * m / radius**2) * np.cos(theta)
        Acceleration_y = (G * m / radius**2) * np.sin(theta)
        v_x.append((v_x[i]) - Acceleration_x * dt)
        v_y.append((v_y[i]) -Acceleration_y * dt)
        x.append(x[i] + v_x[i+1] * dt)
        y.append(y[i] + v_y[i+1] * dt)
        theta= np.arctan(y[i+1]/x[i+1])  # Update theta based on new position
        radius = np.sqrt(x[i+1]**2 + y[i+1]**2)

    elif (x[i] <= 0):
        Acceleration_x = (G * m / radius**2) * np.cos(theta)
        Acceleration_y = (G * m / radius**2) * np.sin(theta)
        v_x.append((v_x[i]) - Acceleration_x * dt)
        v_y.append((v_y[i]) - Acceleration_y * dt)
        x.append(x[i] + v_x[i+1] * dt)
        y.append(y[i] + v_y[i+1] * dt)
        theta = np.arctan(y[i+1]/x[i+1])+np.pi  # Update theta based on new position
        radius = np.sqrt(x[i+1]**2 + y[i+1]**2)
# Plot
plt.plot(x, y, label='Orbit', color='blue')
plt.plot(0,0, 'ro', label='Central Body')  # Central body at origin
plt.title("Orbit Using Euler-Cromer Method")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.grid(True)
plt.axis('equal')
plt.show()

