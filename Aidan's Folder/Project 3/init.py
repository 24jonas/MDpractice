import numpy as np

### Initial Variables and Constnats Declaration ###

## 2D init ##
Two_Dim = True
if (Two_Dim):
    R_arr = np.array([[3.0, 6.0],
                     [7.0, 6.0]])

    V_arr = np.array([[0.0, 0.0],
                     [0.4, 0.0]])
    border = 10

## 3D init ##
def initialize_lattice(border):
    layers = 3
    spacing = border[0] / layers ## Assuming cubic borders
    R = []

    for i in range(layers):
        for j in range(layers):
            for k in range(layers):
                init_pos = np.array([i, j, k]) * spacing
                offset = 0.5 * spacing * ((i + j + k) % 2)

                R_val = init_pos + offset
                R.append(R_val)
                
    R_arr = np.array(R)

    return R_arr

Three_dim = False
if (Three_dim):
    SX = 6.0
    SY = 6.0
    SZ = 6.0
    border = np.array([SX, SY, SZ])
    R_arr = initialize_lattice(border)
    V_arr = np.zeros_like(R_arr)

## Global init ##
dt = 0.01
num_step = 15000
R_sto = [R_arr.copy()]
V_sto = [V_arr.copy()]


