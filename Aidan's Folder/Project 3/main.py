from VV_func import verlet
from energy import kinetic, potential
from init import *
from plots import *
from gr_calc import update_gr, normalize_gr

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

# Preparing for plotting
R_sto = np.array(R_sto)
V_sto = np.array(V_sto)
## g(r) Normalization
r_vals, g = normalize_gr(B, dr, nbin, gr_sample_count, R_arr, border, Dim_check)


plot_master(R_sto, V_sto, KE_sto, PE_sto, E_sto, border, r_vals, g, dt, num_step, Dim_check, R_arr)

print("Progress : 100.0%\nProgram ended successfully.")