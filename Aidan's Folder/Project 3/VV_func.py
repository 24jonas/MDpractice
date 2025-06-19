from acc_func import *
import numpy as np
from init import border

def verlet(V, R, dt):

    A = acc(R)
    V = V + 0.5 * A * dt
    R = R + V * dt

    borders_arr = np.array(border)
    if borders_arr.size == 1:
        borders_arr = np.full(R.shape[1], borders_arr.item())

    R = R % border
    A = acc(R)
    V = V + 0.5 * A * dt

    return (V, R)