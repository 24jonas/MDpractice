from ver_func_3b import *
from variables_3b import *

import matplotlib.pyplot as plt

## Main Loop for Position Verlet ##
for i in range(num * n):
    
    R, V, t = verlet(R, V, dt, m_1, m_2, t)
    pos_list_1 = np.append(pos_list_1, R, axis = 0)
    vel_list_1 = np.append(vel_list_1, V, axis = 0)

## Make x and y Lists for Plot ##
x = [i[0] for i in pos_list_1]
y = [i[1] for i in pos_list_1]

## Plot x and y Values ##
plt.plot(x,y)
plt.show()

