from ver_func_3b import *
from variables_3b import *

import matplotlib.pyplot as plt

## Main Loop for Forest Ruth ##
for i in range(num * n):

    R, V, t, J_FR = For_Rut(R, V, dt, m_1, m_2, t)
    pos_list_2 = np.append(pos_list_2, R, axis = 0)
    vel_list_2 = np.append(vel_list_2, V, axis = 0)
    J_FR_list = np.append(J_FR_list, J_FR, axis = 0)

## Make x and y Lists for Plot ##
x_FR = [i[0] for i in pos_list_2]
y_FR = [i[1] for i in pos_list_2]

## Plot x and y Values ##
if __name__ == "__main__":
    plt.plot(x_FR,y_FR)
    plt.show()