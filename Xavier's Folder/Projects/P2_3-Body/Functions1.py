# Functions for 'Orbit1.py'.

# Imports
import numpy as np

# Input Variables
calculations = 1000000          # unitless
time_unit = 0.001               # seconds
object_initial_x_position = 0   # meters
object_initial_y_position = 0   # meters
initial_velocity = 1            # meters/second
initial_heading = 60            # degrees
star1_x_position = -0.5         # meters
star1_y_position = 0            # meters
star2_x_position = 0.5          # meters
star2_y_position = 0            # meters
star1_mass = 0.5                # kg
star2_mass = 0.5                # kg
ro = "Object Position"          # array: meters, meters
r1 = "Star1 Position"           # array: meters, meters
r2 = "Star2 Position"           # array: meters, meters
vo = "Object Velocity"          # array: meters/second, meters/second
co = "Coordinate Array"         # array: meters, meters

# Alter Variables
x_0 = object_initial_x_position
y_0 = object_initial_y_position
v_0 = initial_velocity
theta = initial_heading
theta = theta*np.pi/180
x1_0 = star1_x_position
y1_0 = star1_y_position
x2_0 = star2_x_position
y2_0 = star2_y_position
m1 = star1_mass
m2 = star2_mass
dt = time_unit
l = calculations
ro = np.array([x_0, y_0])
r1 = np.array([x1_0, y1_0])
r2 = np.array([x2_0, y2_0])
vo = np.array([v_0*np.cos(theta), v_0*np.sin(theta)])
co = np.zeros((l,2))

# Functions
def Radius(ro):
    r = np.sqrt((ro[0]**2) + (ro[1]**2))
    return r

def Angle(x, y):
    if x > 0:
        theta = np.arctan(y/x)
    elif x < 0:
        theta = np.arctan(-y/x)
    else:
        theta = (np.abs(y)/y)*np.pi/2
    return theta

def X_Acceleration(r, theta, x):
    a_x = "x acceleration"
    return a_x

def Y_Acceleration(r, theta):
    a_y = "y acceleration"
    return a_y

def Acceleration(ro, r1, r2):
    d1 = np.abs(ro - r1)
    d2 = np.abs(ro - r2)
    mag1 = np.sqrt((d1[0]**2)+(d1[1]**2))
    mag2 = np.sqrt((d2[0]**2)+(d2[1]**2))
    dr1 = ro - r1
    dr2 = ro - r2
    ao = (-0.5)*(dr1)/(mag1**3) + (-0.5)*(dr2)/(mag2**3)
    return ao

def Position_Verlet(ro, vo, ao, dt):
    ro = ro + (0.5)*vo*dt
    vo = vo + ao*dt
    ro = ro + (0.5)*vo*dt
    return ro, vo
