# Fucntions for 'CollisionsB.py'.

# Imports
import numpy as np

# Input Variables
time_increment = 0.01
time_passed = 0
number_of_increments = 10000
object1_initial_x_position = 7
object1_initial_y_position = 6
object1_initial_x_velocity = 0
object1_initial_y_velocity = 0
object2_initial_x_position = 3
object2_initial_y_position = 6
object2_initial_x_velocity = 0.4
object2_initial_y_velocity = 0
object_1_mass = 6.63*10**(-26)      # Mass of an argon atom.
object_2_mass = 6.63*10**(-26)      # Mass of an argon atom.
object_1_mass = 1
object_2_mass = 1
object1_initial_acceleration = 0
object2_initial_acceleration = 0
r1 = "Object 1 position"
r2 = "Object 2 position"
v1 = "Object 1 velocity"
v2 = "Object 2 velocity"
rm = "Rotation Matrix"
cr = "Combined Radii"
co1 = "Object 1 Coordinate Array"
co2 = "Object 2 Coordinate Array"
object1_radius = 0.5
object2_radius = 0.5
box_x_length = 10
box_y_length = 10
speed_limit = 25

# Rename Variables
dt = time_increment
l = number_of_increments
x1 = object1_initial_x_position
y1 = object1_initial_y_position
x2 = object2_initial_x_position
y2 = object2_initial_y_position
v1x = object1_initial_x_velocity
v1y = object1_initial_y_velocity
v2x = object2_initial_x_velocity
v2y = object2_initial_y_velocity
a1 = object1_initial_acceleration
a2 = object2_initial_acceleration
m1 = object_1_mass
m2 = object_2_mass
r1 = np.array([x1,y1])
r2 = np.array([x2,y2])
v1 = np.array([v1x,v1y])
v2 = np.array([v2x,v2y])
rm = np.array([[0,-1],[1,0]])
co1 = np.zeros((l,2))
co2 = np.zeros((l,2))
or1 = object1_radius
or2 = object2_radius
blx = box_x_length
bly = box_y_length
cr = or1 + or2
sl = speed_limit

# Functions
def Check1(r1, blx, bly):
    if r1[0] < or1:
        return True, True
    elif r1[0] > blx-or1:
        return True, False
    elif r1[1] < or1:
        return False, True
    elif r1[1] > bly-or1:
        return False, False
    else:
        return None, None

def Check2(r2, blx, bly):
    if r2[0] < or2:
        return True, True
    elif r2[0] > blx-or2:
        return True, False
    elif r2[1] < or2:
        return False, True
    elif r2[1] > bly-or2:
        return False, False
    else:
        return None, None

def Lennard_Jones(mag):
    lj = 4*((1/mag)**12 - (1/mag)**6)
    pf = 24*mag**(-7) - 48*mag**(-13)
    return lj, pf

def Vectors(r1, r2):
    dr1 = r2 - r1
    dr2 = r1 - r2
    mag = np.sqrt(dr1[0]**2 + dr1[1]**2)
    n1 = dr1/mag
    n2 = dr2/mag
    return n1, n2, mag

def Acceleration(pf, n1, n2, m1, m2):
    a1 = n1*pf/m1
    a2 = n2*pf/m2
    return a1, a2

def Velocity_Verlet(r1, r2, v1, v2, a1, a2, dt):
    v1 = v1 + 0.5*a1*dt
    v2 = v2 + 0.5*a2*dt
    r1 = r1 + v1*dt
    r2 = r2 + v2*dt
    n1, n2, mag = Vectors(r1, r2)
    lj, pf = Lennard_Jones(mag)
    a1, a2 = Acceleration(pf, n1, n2, m1, m2)
    v1 = v1 + 0.5*a1*dt
    v2 = v2 + 0.5*a2*dt
    return r1, r2, v1, v2, a1, a2, lj

def Speed_Limit(v1, v2, sl):
    mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 > sl:
        v1 = v1*sl/mag1
    if mag2 > sl:
        v2 = v2*sl/mag2
    return v1, v2, mag1, mag2

def Kinetic_Energy(mag1, mag2, m1, m2):
    k1 = 0.5*m1*mag1**2
    k2 = 0.5*m2*mag2**2
    return k1, k2
