# These functions are accessed by OrbitB.py to produce an orbital path. The constants where calculated or sourced separately.

# Constants
dt = 0.151734*0.1
m = 1.4983*10**9
g = 6.6743*10**(-11)

# Imports
import numpy as np

# Functions
def radius(x, y):
    r = np.sqrt((x**2) + (y**2))
    return r

def angle(x, y):
    if x > 0:
        theta = np.arctan(y/x)
    elif x < 0:
        theta = -np.arctan(y/x)
    else:
        theta = (np.abs(y)/y)*np.pi/2
    return theta

def acc_x(r, theta, x):
    acc = -g*m/r
    if x > 0:
        a_x = acc*np.cos(theta)
    elif x < 0:
        a_x = -acc*np.cos(theta)
    else:
        a_x = 0
    return a_x

def acc_y(r, theta):
    acc = -g*m/r
    a_y = acc*np.sin(theta)
    return a_y

"""
def vel_x(x, x_i):
    dx = x - x_i
    v_x = dx/dt
    return v_x

def vel_y(y, y_i):
    dy = y - y_i
    v_y = dy/dt
    return v_y
"""

def vel_x(vi_x, ai_x):
    v_x = vi_x + ai_x*dt
    return v_x

def vel_y(vi_y, ai_y):
    v_y = vi_y + ai_y*dt
    return v_y

def dx(x, v_x, a_x):
    x_f = x + v_x*dt + (0.5)*a_x*(dt**2)
    return x_f

def dx(y, v_y, a_y):
    y_f = y + v_y*dt + (0.5)*a_y*(dt**2)
    return y_f
