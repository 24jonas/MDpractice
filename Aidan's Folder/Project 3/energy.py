import numpy as np

def kinetic(V):
    return (0.5 * np.sum(V ** 2))

def potential(R):
    num_par = len(R)
    P = 0.0
    for i in range(num_par):
        for j in range(i + 1, num_par):
            R_ij = R[i] - R[j]
            r = np.linalg.norm(R_ij)
            if (r == 0):
                continue
            r_term = (1 / r) ** 6
            P += 4 * (r_term ** 2 - r_term)
    return P