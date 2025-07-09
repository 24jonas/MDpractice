import numpy as np
### Initial Variables and Constants Declaration ###

## Inputs ##
def get_input(mesg, default, cast_func):
    inp = input(f"{mesg} (default: {default}): ")
    if (inp.strip() == ''):
        return default
    else:
        return cast_func(inp)
while(True):
    Dim_check = input("Enter 2 for 2D, 3 for 3D, or 0 to exit.\n")
    if(Dim_check == "2"):
        print("Starting 2D Simulation.")
        Dim_check = 2
        break
    elif(Dim_check == "3"):
        print("Starting 3D Simulation.")
        Dim_check = 3
        break
    elif(Dim_check == "0"):
        print("Exiting Simulation.")
        exit()

## 2D init ##
if (Dim_check == 2):
    R_arr = np.array([[3.0, 6.0],
                     [7.0, 6.0]])

    V_arr = np.array([[0.0, 0.0],
                     [0.4, 0.0]])
    border = 10

## 3D init ##
def initialize_lattice(border):
    layers = int(input("Input number of particles as cube root: "))
    spacing = border[0] / (layers) ## Assuming cubic borders
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

if (Dim_check == 3):
    SX = 6.0
    SY = 6.0
    SZ = 6.0
    border = np.array([SX, SY, SZ])
    R_arr = initialize_lattice(border)
    np.random.seed(0)
    V_arr = np.random.normal(0, 0.5, size = R_arr.shape)
    V_arr -= V_arr.mean(axis = 0)

## g(r) config ##
dr = 0.05
nbin = 100
B = np.zeros(nbin)
equil_step = 5000
sample_rate_gr = 20
gr_sample_count = 0

## Global init ##
dt = get_input("Input dt", 0.01, float)
num_step = get_input("Input num_step", 10000, int)
R_sto = [R_arr.copy()]
V_sto = [V_arr.copy()]
sample_rate = get_input("Sample Rate", 10, int)


