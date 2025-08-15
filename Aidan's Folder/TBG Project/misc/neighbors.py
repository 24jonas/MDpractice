import numpy as np

import numpy as np

def _cart_to_frac(R, H):
    return np.linalg.solve(H, np.asarray(R, float).T).T

def _frac_to_cart(S, H):
    S = np.asarray(S, float)
    H = np.asarray(H, float)
    return np.einsum('ij,...j->...i', H, S)

def count_nearest_neighbors(positions, cell, cutoff=1.6, use_z=False):
    ## Changed to use Periodic Boundary
    R = np.asarray(positions, float)
    H = np.asarray(cell, float)

    S = _cart_to_frac(R, H)                        
    dS = S[None, :, :] - S[:, None, :]              
    if use_z:
        dS -= np.round(dS)
    else:
        dS[:, :, :2] -= np.round(dS[:, :, :2])      

    dR = _frac_to_cart(dS, H)                     
    d2 = np.einsum('ijk,ijk->ij', dR, dR)
    np.fill_diagonal(d2, np.inf)

    return np.sum(d2 <= cutoff**2, axis=1).astype(int)
