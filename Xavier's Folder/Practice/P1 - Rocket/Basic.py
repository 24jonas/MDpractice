# Simulate a rocket's flight path with no atmosphere.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from FunctionsA import *

# Input Variables
rocket_mass = 5     # kg
fuel_mass = 5       # kg
burn_rate = 1       # kg/s
gravitational_acceleration = 9.81   # m/s^2
thrust = 300        # N
angle = 89          # Degrees
initial_x = 0       # m
initial_y = 0       # m
initial_vx = 0      # m/s
initial_vy = 0      # m/s
coordinate_array = np.array([[initial_x, initial_y]])   # Ordered Pairs (x, y)
precision = 0.01     # s
time = 0
total_mass = rocket_mass + fuel_mass

# Alter Variables
m = total_mass
f = fuel_mass
br = burn_rate
g = gravitational_acceleration
th = thrust
theta = angle*np.pi/180
co = coordinate_array
dt = precision
x = initial_x
y = initial_y
v_x = initial_vx
v_y = initial_vy
fg = G_Force(m, g)
t = time

# Build the array.
while y >= 0:
    t = t + dt
    m, f, th = Fuel(f, t, br, m, th)
    th_x = Thrust_x(th, theta)
    th_y = Thrust_y(th, theta)
    fy = Net_y(fg, th_y)
    a_x = Acc_x(m, th_x)
    a_y = Acc_y(m, fy)
    v_x = Vel_x(v_x, a_x, dt)
    v_y = Vel_y(v_y, a_y, dt)
    x = Dx(x, v_x, dt)
    y = Dy(y, v_y, dt)
    theta = Angle(v_x, v_y)
    co = np.append(co, [[x, y]], axis = 0)

# Plot the coordinate pairs.
x = co[:, 0]
y = co[:, 1]

plt.plot(x, y, label='Trajectory', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Basic Rocket')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
