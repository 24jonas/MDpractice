from traj_pv import x_PV, y_PV
from traj_fr import x_FR, y_FR

import matplotlib.pyplot as plt

plt.plot(x_FR, y_FR, label = 'Forest-Ruth')
plt.plot(x_PV, y_PV, label = 'Position Verlet')
plt.xlabel('x')
plt.ylabel('y')
plt.title('FR vs. PV')
plt.legend()
plt.show()

#####
from variables_3b import J_0
from traj_fr import J_FR_list, J_PV_list
import numpy as np

J_mag0 = np.sqrt(J_0[0, 0] ** 2 + J_0[0, 1] ** 2)
x_JF = [i[0] for i in J_FR_list]
y_JF = [i[1] for i in J_FR_list]
x_JP = [i[0] for i in J_PV_list]
y_JP = [i[1] for i in J_PV_list]

j_mag_PV = []
j_mag_FR = []

for i in range(len(x_JP)):

    j_mag_PV.append(np.sqrt(x_JP[i] ** 2 + y_PV[i] ** 2))
    j_mag_FR.append(np.sqrt(x_JF[i] ** 2 + y_JF[i] ** 2))

print(j_mag_PV)


