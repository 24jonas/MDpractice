from VV_func import verlet
from energy import kinetic, potential
from init import *
from plots import *
from gr_calc import update_gr

R = R_arr.copy()
V = V_arr.copy()
KE_sto = [kinetic(V)]
PE_sto = [potential(R)]
E_sto = [kinetic(V) + potential(R)]


for i in range(num_step):
    if i % sample_rate == 0:
        V, R = verlet(V, R, dt)
        R_sto.append(R.copy())
        V_sto.append(V.copy())

        K = kinetic(V)
        P = potential(R)
        KE_sto.append(K)
        PE_sto.append(P)
        E_sto.append(K + P)

    if i >= equil_step and (i - equil_step) % sample_rate_gr == 0:
        update_gr(R, B, dr, nbin, np.array(border))
        gr_sample_count += 1

    if i % (num_step // 10) == 0:
        print(f"Progress: {int(i / num_step * 100)}%...")

# g(r) Normalization
## Move to plots, no need to keep here for 2d case.
V = np.prod(border) # Calculate volume
N = len(R_arr)
rho = N / V
r_vals = (np.arange(nbin) + 0.5) * dr # +0.5 instead of +1.0 to get to middle of bin
g = np.zeros_like(B)

for n in range(nbin):
    if (Dim_check == '2'):
        shell_vol = 4 * np.pi * (r_vals[n] ** 2) * dr
    else:
        shell_vol = 4 * np.pi * ((dr * (n + 1)) ** 3 - (dr * n) ** 3) / 3
    denom = gr_sample_count * N * shell_vol * rho
    if denom > 0:
        g[n] = B[n] / denom
    else:
        g[n] = 0

# Preparing for plotting
R_sto = np.array(R_sto)
V_sto = np.array(V_sto)
## 3D Compatibility
dims = R_sto.shape[2]

plot_master(R_sto, V_sto, KE_sto, PE_sto, E_sto, border, r_vals, g, dt, num_step, Dim_check, R_arr)

print("Progress : 100.0%\nProgram ended successfully.")