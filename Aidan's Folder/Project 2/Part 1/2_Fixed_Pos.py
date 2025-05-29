from ver_func_2b import *
from variables_2b import *

import numpy as np
import matplotlib.pyplot as plt

## 2 Fixed Bodies Positions Problem ##

## Main Loop ##
for i in range(num * n):

    R, V = verlet(R, V, dt, R_1, R_2, m_1, m_2)

    pos_list = np.append(pos_list, R, axis = 0)
    vel_list = np.append(vel_list, V, axis = 0)


## Create x and y Values to plot ##
x = [i[0] for i in pos_list]
y = [i[1] for i in pos_list]

## Use Matplotlib to Create a Graph ##
plt.plot(x, y)
plt.show()
