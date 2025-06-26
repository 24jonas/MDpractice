import numpy as np
import matplotlib.pyplot as plt

from VV_func import verlet
from energy import kinetic, potential
from init import *
from gr_calc import update_gr

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

    if i >= equil_step and (i - equil_step) % sample_rate == 0:
        update_gr(R, B, dr, nbin, np.array(border))
        gr_sample_count += 1

    if i % (num_step // 10) == 0:
        print(f"Progress: {int(i / num_step * 100)}%...")

# g(r) Normalization
V = np.prod(border) # Calculate volume
N = len(R_arr)
rho = N / V
r_vals = (np.arange(nbin) + 0.5) * dr # +0.5 instead of +1.0 to get to middle of bin
g = np.zeros_like(B)

for n in range(nbin):
    shell_vol = 4 * np.pi * (r_vals[n] ** 2) * dr
    denom = gr_sample_count * N * shell_vol * rho
    if denom > 0:
        g[n] = B[n] / denom
    else:
        g[n] = 0

# Preparing for plotting
R_sto = np.array(R_sto)
V_sto = np.array(V_sto)
dims = R_sto.shape[2] ## 3D compatibility



# Plot trajectories
plt.subplot(1, 3, 1)
plt.plot(R_sto[:, 0, 0], label = "Particle 1")
plt.title("Particle trajectories")
plt.legend()

# Plot velocity
plt.subplot(1, 3, 2)
for d in range(min(dims, 3)):
    plt.plot(V_sto[:, 0, d], label=f'V{d}')
plt.title("Vel Graph")
plt.legend()

# Plot Energies
plt.subplot(1,3,3)
plt.plot(KE_sto, label = 'Kinetic')
plt.plot(PE_sto, label = 'Potential')
plt.plot(E_sto, label = 'total')
plt.title("Energies")
plt.legend()

plt.tight_layout()
plt.show()

# Plot g(r)
plt.figure()
plt.plot(r_vals, g)
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Pair Correlation Function")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Progress : 100.0%\nProgram ended successfully.")