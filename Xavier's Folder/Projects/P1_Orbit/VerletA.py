# These functions are accessed by OrbitA.py to produce an orbital path. Verlet equations sourced from physics.udel.edu. This file uses the
# position equation, or eqn (21).

# Constants
dt = 0.151734
m = 1.4983*10**9
g = 6.6743*10**(-11)

# Imports
import numpy as np

# Functions
def radius1(x, y):
    r1 = np.sqrt((x**2) + (y**2))
    return r1

def angle1(x, y):
    if x > 0:
        theta1 = np.arctan(y/x)
    elif x < 0:
        theta1 = np.arctan(-y/x)
    else:
        theta1 = (np.abs(y)/y)*np.pi/2
    return theta1

def acc1_x(r1, theta1, x):
    acc = -g*m/r1
    if x > 0:
        a1_x = acc*np.cos(theta1)
    elif x < 0:
        a1_x = -acc*np.cos(theta1)
    else:
        a1_x = 0
    return a1_x

def acc1_y(r1, theta1):
    acc = -g*m/r1
    a1_y = acc*np.sin(theta1)
    return a1_y

def dx(x, x_i, a1_x):
    x_f = 2*x - x_i + a1_x*(dt)**2
    return x_f

def dy(y, y_i, a1_y):
    y_f = 2*y - y_i + a1_y*(dt)**2
    return y_f

# The functions beyond this point are not accessed by 'OrbitA', because they are used for calculating velocity according to the Verlet algorithm.

def radius2(x_f, y_f):
    r2 = np.sqrt((x_f**2) + (y_f**2))
    return r2

def angle2(x_f, y_f):
    if x_f > 0:
        theta2 = np.arctan(y_f/x_f)
    elif x_f < 0:
        theta2 = np.arctan(-y_f/x_f)
    else:
        theta2 = (np.abs(y_f)/y_f)*np.pi/2
    return theta2

def acc2_x(r2, theta2, x):
    acc = -g*m/r2
    if x > 0:
        a2_x = acc*np.cos(theta2)
    elif x < 0:
        a2_x = -acc*np.cos(theta2)
    else:
        a2_x = 0
    return a2_x

def acc2_y(r2, theta2):
    acc = -g*m/r2
    a2_y = acc*np.sin(theta2)
    return a2_y

def vel2_x(v1_x, a1_x, a2_x):
    v2_x = v1_x + (0.5)*(a1_x + a2_x)*(dt)
    return v2_x

def vel2_y(v1_y, a1_y, a2_y):
    v2_y = v1_y + (0.5)*(a1_y + a2_y)*(dt)
    return v2_y
