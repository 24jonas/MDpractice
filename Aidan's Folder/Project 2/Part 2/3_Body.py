from ver_func_3b import *
from variables_3b import *

import matplotlib.pyplot as plt

## Main Loop ##

for i in range(num * n):
    
    R, V, t = verlet(R, V, dt, m_1, m_2, t)
    pos_list = np.append(pos_list, R, axis = 0)
    vel_list = np.append(vel_list, V, axis = 0)

## Make x and y Lists for Plot ##
x = [i[0] for i in pos_list]
y = [i[1] for i in pos_list]

## Plot x and y Values ##
plt.plot(x,y)
plt.show()