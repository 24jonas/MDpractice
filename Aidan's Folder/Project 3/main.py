import numpy as np
import matplotlib.pyplot as plt

from VV_func import verlet
from energy import kinetic, potential
from init import *

R = R_arr.copy()
V = V_arr.copy()
KE_sto = [kinetic(V)]
PE_sto = [potential(R)]
E_sto = [kinetic(V) + potential(R)]


for i in range(num_step):
    V, R = verlet(V, R, dt)
    R_sto.append(R.copy())
    V_sto.append(V.copy())

    K = kinetic(V)
    P = potential(R)
    KE_sto.append(K)
    PE_sto.append(P)
    E_sto.append(K + P)

R_sto = np.array(R_sto)
V_sto = np.array(V_sto)

dims = R_sto.shape[2] ## 3D compatibility


# Plot trajectories
plt.subplot(1, 3, 1)
for i in range(R_sto.shape[1]):
    plt.plot(R_sto[:,i,0], label = f'Particle {i + 1}')


##plt.plot(R_sto[:, 0, 0], label='Particle 1')
##plt.plot(R_sto[:, 1, 0], label='Particle 2')

plt.title("Particle trajectories")

# Plot velocity
plt.subplot(1, 3, 2)
t = np.arange(len(V_sto)) * dt
for i in range(V_sto.shape[1]):
    for d in range(min(dims, 3)):
        plt.plot(t, V_sto[:, i, d], label=f'P{i+1} V{d}')
plt.title("Vel Graph")
plt.legend()

# Plot Energies
plt.subplot(1,3,3)
t = np.arange(len(KE_sto)) * dt
plt.plot(t, KE_sto, label = 'Kinetic')
plt.plot(t, PE_sto, label = 'Potential')
plt.plot(t, E_sto, label = 'total')
plt.title("Energies")
plt.legend()

plt.tight_layout()
plt.show()