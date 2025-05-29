from acc_func_3b import *
from variables_3b import r_1, r_2, s

import numpy as np

## Second Order Verlet Function ##

def verlet(R, V, dt, m_1, m_2, t):
    
    R = R + 0.5 * dt * V
    t += 0.5
    R_1 = r_1(t)
    R_2 = r_2(t)
    A = acc(R, R_1, R_2, m_1, m_2)
    V = V + dt * A
    R = R + 0.5 * dt * V
    t += 0.5

    return R, V, t

## Fourth Order Forest Ruth Function ##

def For_Rut(R, V, dt, m_1, m_2, t):

    dt_1 = dt / (2 - s)
    dt_2 = (-s * dt) / (2 - s)
    t_1 = t + dt / (2 - s)
    t_2 = t_1 - s * dt / (2 - s)
    
    verlet(R, V, dt_1, m_1, m_2, t_2)
    verlet(R, V, dt_2, m_1, m_2, t_1)
    verlet(R, V, dt_1, m_1, m_2, t)

    return