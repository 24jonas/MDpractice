import numpy as np

from init import border

def acc(R):

    num_particles, dim = R.shape
    A = np.zeros_like(R)
    borders_arr = np.array(border)

    if (borders_arr.size == 1):
        borders_arr = np.full(dim, borders_arr.item())
    elif (borders_arr.size != dim):
        raise ValueError("Dimensions of simularion border and R don't match!")

    for i in range(num_particles):
        R_i = R[i]
        for j in range(i + 1, num_particles):
            R_ij = R_i - R[j]
            R_ij = R_ij - borders_arr * np.round(R_ij / borders_arr)
            r = np.dot(R_ij, R_ij)
            if r == 0:
                continue

            r2_inv =  1.0 / r
            r6 = r2_inv ** 3
            r12 = r6 ** 2
            F_mag = 24 * (2 * r12 - r6) * r2_inv
            F_vec = F_mag * R_ij

            A[i] += F_vec
            A[j] -= F_vec
    return A