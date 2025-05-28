# This file contains the functions used in the 'Rocket1' file.

# Imports
import numpy as np

def G_Force(m, g):
    fg = -m*g
    return fg

def Fuel(f, t, br, m, th):
    if f > 0:
        f = f - br*t
        m = m - br*t
    else:
        m = m
        f = 0
        th = 0
    return m, f, th

def Thrust_x(th, theta):
    th_x = th*np.cos(theta)
    return th_x

def Thrust_y(th, theta):
    th_y = th*np.sin(theta)
    return th_y

def Net_y(fg, th_y):
    fy = fg + th_y
    return fy

def Acc_x(m, th_x):
    a_x = th_x/m
    return a_x

def Acc_y(m, fy):
    a_y = fy/m
    return a_y

def Vel_x(v_x, a_x, dt):
    v_x = v_x + a_x*dt
    return v_x

def Vel_y(v_y, a_y, dt):
    v_y = v_y + a_y*dt
    return v_y

def Dx(x, v_x, dt):
    x = x + v_x*dt
    return x

def Dy(y, v_y, dt):
    y = y + v_y*dt
    return y

def Angle(v_x, v_y):
    if v_x > 0:
        theta = np.arctan(v_y/v_x)
    elif v_x < 0:
        theta = -np.arctan(v_y/v_x)
    else:
        theta = (0.5)*np.pi*np.abs(v_y)/v_y
    return theta
