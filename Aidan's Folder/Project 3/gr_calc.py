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

import numpy as np

def normalize_gr(B, dr, nbin, gr_sample_count, R_arr, border, dims):
    V = np.prod(border)
    N = len(R_arr)
    rho = N / V
    r_vals = (np.arange(nbin) + 0.5) * dr
    g = np.zeros_like(B)

    for n in range(nbin):
        if dims == 2:
            shell_vol = np.pi * (((dr * (n + 1)) ** 2) - ((dr * n) ** 2))
        else:
            shell_vol = (4/3) * np.pi * (((dr * (n + 1)) ** 3) - ((dr * n) ** 3))
        
        denom = gr_sample_count * N * shell_vol * rho
        if denom > 0:
            g[n] = B[n] / denom
        else:
            g[n] = 0.0

    return r_vals, g
