# The functions here are accessed by 'OrbitD'.

# Imports
import numpy as np

def slr(x_0, v_0):
    p = (x_0*v_0)**2
    return p

def energy(x_0, v_0):
    e_0 = (0.5)*(v_0)**2 - (x_0)**(-1)
    return e_0

def sma(e_0):
    a = -(2*e_0)**(-1)
    return a

def eccentricity(p, a):
    e = np.sqrt(1 - p/a)
    return e

def period(a):
    t = 2*np.pi*(a)**(1.5)
    return t

def radius(p, e, theta):
    r = p / (1 - e*np.cos(theta))
    return r

def x_comp(r, theta):
    x = r*np.cos(theta)
    return x

def y_comp(r, theta):
    y = r*np.sin(theta)
    return y

def increments(t, l):
    dt = t/l
    return dt
