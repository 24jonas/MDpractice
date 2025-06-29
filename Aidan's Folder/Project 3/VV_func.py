from acc_func import *
import numpy as np
from init import border

def verlet(V, R, dt):

    A = acc(R)
    R = R + V * dt + 0.5 * A * dt ** 2

    borders_arr = np.array(border)
    if borders_arr.size == 1:
        borders_arr = np.full(R.shape[1], borders_arr.item())

    R = R % border
    V = V + 0.5 * (A + acc(R)) * dt

    return (V, R)