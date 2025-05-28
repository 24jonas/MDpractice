from Verlet_Project1 import *
import numpy as np
import matplotlib.pyplot as plt

E_list = [0]
t_list = [0]
v_x = [i[0] for i in vel_output]
v_y = [i[1] for i in vel_output]

for i in range(len(x)):
    
    r = np.sqrt(x[i] ** 2 + y[i] ** 2)
    v = v_x[i] ** 2 + v_y[i] ** 2

    ## Using wrong v?
    E = (v ** 2) / 2 - (1 / r)
    E_ratio = ((E) / (abs(E_0))) - 1
    E_list.append(E_ratio)
    t_list.append(t_list[i] + 1)

    if i > 500:
    ##round(0.55 * per + 1):
        break




##plt.plot(t_list[round(0.45 * per) : round(0.55 * per + 1)], E_list[round(0.45 * per) : round(0.55 * per + 1)])
plt.plot(t_list[round(0.45 * per) : 500], E_list[round(0.45 * per) : 500])

plt.show()

