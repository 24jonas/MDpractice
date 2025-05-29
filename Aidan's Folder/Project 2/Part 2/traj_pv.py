from ver_func_3b import *
from variables_3b import *

import matplotlib.pyplot as plt

## Main Loop for Position Verlet ##
for i in range(num * n):
    
    R, V, t, J_PV = verlet(R, V, dt, m_1, m_2, t)
    pos_list_1 = np.append(pos_list_1, R, axis = 0)
    vel_list_1 = np.append(vel_list_1, V, axis = 0)
    J_PV_list = np.append(J_FR_list, J_PV, axis = 0)




## Make x and y Lists for Plot ##
x_PV = [i[0] for i in pos_list_1]
y_PV = [i[1] for i in pos_list_1]

## Plot x and y Values ##
if __name__ == "__main__":
    plt.plot(x_PV,y_PV)
    plt.show()