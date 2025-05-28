import numpy as np
from acc_func import *

## Position Verlet Function ##

def verlet(R, V, dt, R_1, R_2, m_1, m_2):
    
    R = R + 0.5 * dt * V
    A = acc(R, R_1, R_2, m_1, m_2)
    V = V + dt * A
    R = R + 0.5 * dt * V

    return R, V