import numpy as np
from init import border
def kinetic(V):
    return (0.5 * np.sum(V ** 2))

def potential(R):
    num_par = len(R)
    P = 0.0
    borders_arr = np.array(border)
    if borders_arr.size == 1:
        borders_arr = np.full(R.shape[1], borders_arr.item())
    for i in range(num_par):
        for j in range(i + 1, num_par):
            R_ij = R[i] - R[j]
            R_ij = R_ij - borders_arr * np.round(R_ij / borders_arr)
            r = np.linalg.norm(R_ij)
            if (r == 0):
                continue
            r_term = (1 / r) ** 6
            P += 4 * (r_term ** 2 - r_term)
    return P