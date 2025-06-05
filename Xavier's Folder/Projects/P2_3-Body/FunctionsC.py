# Functions for 'OrbitC.py'.

# Imports
import numpy as np

# Input Variables
calculations = 5655                 # unitless
time_unit = 0.005                   # seconds
time_elapsed = 0                    # seconds
object_initial_x_position = 0       # meters
object_initial_y_position = 0.058   # meters
initial_velocity_x = 0.49           # meters/second
initial_velocity_y = 0              # meters/second
star1_x_position = -0.5             # meters
star1_y_position = 0                # meters
star2_x_position = 0.5              # meters
star2_y_position = 0                # meters
star1_mass = 0.5                    # kg
star2_mass = 0.5                    # kg
ro = "Object Position"              # array: meters, meters
r1 = "Star1 Position"               # array: meters, meters
r2 = "Star2 Position"               # array: meters, meters
vo = "Object Velocity"              # array: meters/second, meters/second
co = "Coordinate Array"             # array: meters, meters

# Alter Variables
x_0 = object_initial_x_position
y_0 = object_initial_y_position
v_x = initial_velocity_x
v_y = initial_velocity_y
x1_0 = star1_x_position
y1_0 = star1_y_position
x2_0 = star2_x_position
y2_0 = star2_y_position
m1 = star1_mass
m2 = star2_mass
dt = time_unit
t = time_elapsed
l = calculations
ro = np.array([x_0, y_0])
r1 = np.array([x1_0, y1_0])
r2 = np.array([x2_0, y2_0])
vo = np.array([v_x, v_y])
co = np.zeros((l,2))
c1 = np.zeros((l,2))
c2 = np.zeros((l,2))

# Functions
def Acceleration(ro, r1, r2, m1, m2):
    d1 = np.abs(ro - r1)
    d2 = np.abs(ro - r2)
    mag1 = np.sqrt((d1[0]**2)+(d1[1]**2))
    mag2 = np.sqrt((d2[0]**2)+(d2[1]**2))
    ao = (-m1)*(ro - r1)/(mag1**3) + (-m2)*(ro - r2)/(mag2**3)
    return ao

def Position_VerletA(ro, vo, dt, r1, r2, m1, m2):
    dt = dt/(0.74007895)
    ro = ro + (0.5)*vo*dt
    ao = Acceleration(ro, r1, r2, m1, m2)
    vo = vo + ao*dt
    ro = ro + (0.5)*vo*dt
    return ro, vo

def Position_VerletB(ro, vo, dt, r1, r2, m1, m2):
    dt = dt/(-1.70241438)
    ro = ro + (0.5)*vo*dt
    ao = Acceleration(ro, r1, r2, m1, m2)
    vo = vo + ao*dt
    ro = ro + (0.5)*vo*dt
    return ro, vo

def Star_Positions(t):
    r1 = [(-0.5)*np.cos(t), (-0.5)*np.sin(t)]
    r2 = [(0.5)*np.cos(t), (0.5)*np.sin(t)]
    return r1, r2

def Jacobai_Constant(ro, r1, r2, vo, t):
    ja = (vo**2) - 1/(ro - r1) - 1/(ro - r2) - 2*ro*vo
    jo = np.sqrt((ja[0]**2)+(ja[1]**2))
    return jo
