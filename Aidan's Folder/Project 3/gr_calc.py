import numpy as np

def update_gr(R, B, dr, nbin, border):
    N = len(R)
    dR = R[:, np.newaxis, :] - R[np.newaxis, :, :]  
    dR -= border * np.round(dR / border)            
    dists = np.linalg.norm(dR, axis=2)              

    upper_indices = np.triu_indices(N, k=1) # Only indices above the diagonal
    r_vals = dists[upper_indices]

    bin_indices = (r_vals / dr).astype(int)
    valid = bin_indices < nbin
    np.add.at(B, bin_indices[valid], 2)
