import numpy as np
from .MathUtils import cartToFrac, fracToCart
from .Geom import wrapFrac

def applyStrain(
        positions: np.ndarray, 
        cell: np.ndarray, 
        eps_x = 0.0, 
        eps_y = 0.0, 
        wrap_z = False, 
        return_F = False
        ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    R = np.asarray(positions, dtype = float)
    H = np.asarray(cell, dtype = float) 

    # deformation gradient
    F = np.diag([1.0 + eps_x, 1.0 + eps_y, 1.0])

    # apply deformation gradient
    H_strained = F @ H
    R_strained = R @ F.T

    # wrap atoms under new cell
    S = cartToFrac(R_strained, H_strained)
    S = wrapFrac(S, wrap_z = wrap_z)
    R_wrapped = fracToCart(S, H_strained)

    if return_F:
        return R_wrapped, H_strained, F
    return R_wrapped, H_strained