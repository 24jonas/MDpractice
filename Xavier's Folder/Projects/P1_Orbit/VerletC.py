# These functions are accessed by 'OrbitC.py' using a different version of the Verlet algorithm.

# Imports
import numpy as np

# Functions
def slr(x, v_y):
    p = (x*v_y)**2
    return p

def energy_0(x, v_y):
    e_0 = (0.5)*(v_y)**2 - (x)**(-1)
    return e_0

def sma(e_0):
    a = -(2*e_0)**(-1)
    return a

def eccentricity(p, a):
    e = np.sqrt(1 - p/a)
    return e

def period(a):
    dt = 2*np.pi*(a)**(1.5)
    return dt

def radius(x, y):
    r = np.sqrt((x**2) + (y**2))
    return r

def angle(x, y):
    if x > 0:
        theta = np.arctan(y/x)
    elif x < 0:
        theta = np.arctan(-y/x)
    else:
        theta = (np.abs(y)/y)*np.pi/2
    return theta

def acc_x(r, theta, x):
    acc = -x/(r**3)
    a_x = acc*np.cos(theta)
    return a_x

def acc_y(r, theta, y):
    acc = -y/(r**3)

    if y != 0:
        co = np.abs(y)/y
    else:
        co = 1
    
    a_y = acc*np.sin(theta)*co
    return a_y

def dx(x, v_x, a_x, dt):
    x = x + (0.5)*dt*v_x
    v_x = v_x + dt*a_x
    x = x + (0.5)*dt*v_x
    return x, v_x

def dy(y, v_y, a_y, dt):
    y = y + (0.5)*dt*v_y
    v_y = v_y + dt*a_y
    y = y + (0.5)*dt*v_y
    return y, v_y

def energy(v, r):
    e = (0.5)*(v**2) - r**(-1)
    return e

def error(e, e_0):
    err = e/e_0 - 1
    return err

def vel(v_x, v_y):
    v = np.sqrt((v_x**2)+(v_y**2))
    return v
