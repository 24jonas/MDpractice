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
steps = int(T / dt)
G= 1
m=1
e=np.sqrt(1-p/a)
theta= np.linspace(0, 2*np.pi, 100) 
r_minus = p/(1-e*np.cos(theta))
Acceleration= G*m/r_minus**2

#Initial conditions
x = np.zeros(steps)
v = np.zeros(steps)
x[0] = [x_0,0]
v[0] = [0,yv_0]


# Euler-Cromer Integration
for i in range(steps - 1):
    Acceleration = G * m / r_minus**2
    v[i + 1] = v[i] + Acceleration * dt
    x[i + 1] = x[i] + v[i + 1] * dt

# Plot
plt.plot(x[:, 0], x[:, 1], label='Orbit', color='blue')
plt.title("Orbit Using Euler-Cromer Method")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.grid(True)
plt.axis('equal')
plt.show()

