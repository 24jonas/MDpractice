import numpy as np

## Strain Operations ##

def cart_to_frac(R, H):
    return np.dot(R, np.linalg.inv(H).T)

def frac_to_cart(S, H):
    return np.dot(S, H.T)

def wrap01(S, use_z=False):
    S = np.asarray(S, float).copy()
    S[..., :2] = S[..., :2] - np.floor(S[..., :2])
    if use_z:
        S[..., 2] = S[..., 2] - np.floor(S[..., 2])
    return S

def apply_strain(positions, cell, eps_x=0.0, eps_y=0.0, wrap_z=False, return_F=False):
    R = np.asarray(positions, float)
    H = np.asarray(cell, float)

    # deformation gradient
    F = np.diag([1.0 + eps_x, 1.0 + eps_y, 1.0])

    # apply deformation gradient
    Hp = np.dot(F, H)
    Rp = np.dot(R, F.T)

    # wrap atoms under new cell
    Sp = cart_to_frac(Rp, Hp)
    Sp = wrap01(Sp, use_z=wrap_z)
    Rp_wrapped = frac_to_cart(Sp, Hp)

    if return_F:
        return Rp_wrapped, Hp, F
    
    return Rp_wrapped, Hp