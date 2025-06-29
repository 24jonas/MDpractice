import numpy as np
from init import border

def acc(R):
    dim = R.shape[1]
    A = np.zeros_like(R)
    borders_arr = np.array(border)

    if borders_arr.size == 1:
        borders_arr = np.full(dim, borders_arr.item())
    elif borders_arr.size != dim:
        raise ValueError("Dimensions of border and R don't match")

    R_ij = R[:, np.newaxis, :] - R[np.newaxis, :, :]  # Basically makes a rank 3 tensor to calculate the distance matrix with
    R_ij -= borders_arr * np.round(R_ij / borders_arr) # Checks if the nearest particle is across the box border or not
    R_ij_sq = np.sum(R_ij ** 2, axis = -1)
    np.fill_diagonal(R_ij_sq, np.inf) # Will result in diagonal being zero without causeing div by zero error later

    r2_inv = 1.0 / R_ij_sq
    r6 = r2_inv ** 3
    r12 = r6 ** 2
    F_mag = 24 * (2 * r12 - r6) * r2_inv
    np.fill_diagonal(F_mag, 0.0)
    F_vec = F_mag[:,:, np.newaxis] * R_ij

    A = np.sum(F_vec, axis = 1)

    return A
