# Functions for the first molecular dynamics problem.

# Imports
import numpy as np

# Input Variables
time_increment = 0.01
number_of_increments = 10000
object1_initial_x_position = 7
object1_initial_y_position = 6
object1_initial_x_velocity = 0
object1_initial_y_velocity = 0
object2_initial_x_position = 3
object2_initial_y_position = 5.75
object2_initial_x_velocity = 5
object2_initial_y_velocity = 0
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

# Functions
def Norm(vr):
    n = np.sqrt((vr[0]**2)+(vr[1]**2))
    return n

def Check1(r1, r2, cr):
    dr = r1 - r2
    n = Norm(dr)
    if n <= cr:
        return True, dr
    else:
        return False, dr

def Check2(r1, blx, bly):
    if r1[0] < or1 or r1[0] > blx-or1:
        return True
    elif r1[1] < or1 or r1[1] > bly-or1:
        return False
    else:
        return None

def Check3(r2, blx, bly):
    if r2[0] < or2 or r2[0] > blx-or2:
        return True
    elif r2[1] < or2 or r2[1] > bly-or2:
        return False
    else:
        return None

def Position(r1, r2, v1, v2, dt):
    r1 = r1 + v1*dt
    r2 = r2 + v2*dt
    return r1, r2

def Contact(dr, rm):
    n1 = Norm(dr)
    n2 = rm@dr
    return n1, n2

def Components(v1, v2, n1, n2):
    va = n2*np.dot(v1,n2)
    vb = n1*np.dot(v1,n1)
    vc = n2*np.dot(v2,n2)
    vd = n1*np.dot(v2,n1)
    v1 = va + vd
    v2 = vb + vc
    return v1, v2
