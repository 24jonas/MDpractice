# 3d with variable molecules in cubic pattern. LJ collision dynamics except for classical dynamics with walls. Does not use the VV algorithm.
# This file plots radial particle density based on 'CollisionsC.py'.
# This program should be run with the initial conditions in 'FunctionsC.py' of q = 27, dt = 0.01, l = 10000, io = 0.5, iv = 0, m = 0.1, ro = 0.5
# bf = 5, isn = 2, sl = 100, el = 100, and bs = 0.5. Running other initial conditions may cause an error.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from FunctionsC import *

# Generate initial coordinates.
n = 0
s1 = s2 = s3 = io
for i in range(p):
    s1 = -s1
    for j in range(p):
        s2 = -s2
        for k in range(p):
            s3 = -s3
            rx[0][n] = isn*i + bf + s1
            ry[0][n] = isn*j + bf + s2
            rz[0][n] = isn*k + bf + s3
            n += 1

# Generate initial velocities. Optional.
for i in range(q):
    vx[0][i] = iv
    vy[0][i] = iv
    vz[0][i] = iv

# Generate coordinate arrays.
count = 0
for i in range(l):
    rx, vx = Wall_x(rx, vx, ro, i)
    ry, vy = Wall_y(ry, vy, ro, i)
    rz, vz = Wall_z(rz, vz, ro, i)

    acc1 = []
    for j in range(q):
        r1 = np.array([rx[i][j],ry[i][j],rz[i][j]])
        v = np.array([vx[i][j],vy[i][j],vz[i][j]])

        acc2 = []
        for k in range(q):
            r2 = np.array([rx[i][k],ry[i][k],rz[i][k]])

            if r1[0] == r2[0] and r1[1] == r2[1] and r1[2] == r2[2]:
                a = np.zeros((1,3))
                lj = 0
                r = 0
            else:
                n, r = Vectors(r1, r2)
                lj, a = Lennard_Jones(r, n, m)
            
            if i+1 > br:
                rl.append(r)
            acc2.append(a)
            count += 1
        
        sum1 = np.zeros((1,3))
        for w1 in range(q):
            sum1 += acc2[w1]
        acc1.append(sum1)
    
    rx, ry, rz, vx, vy, vz = Update(rx, ry, rz, vx, vy, vz, acc1, dt, i, l, sl)
    print(i+1)

for r in rl:
    nbin = int(r/bs) + 1
    bb[nbin] = bb[nbin] + 1

rpd[0] = [0,0]
arpd[0] = [0,0]
for i in range(dd):
    nbin = i + 1
    bb[nbin] = bb[nbin]/(l-br)/q
    gr = Radial_Density(bb, nbin, bs, q, blx)
    rpd[nbin] = [gr,nbin]
    if i < add:
        arpd[nbin] = [gr,nbin]

print("------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Complete")

# Plot RPD
x = rpd[:,1]
y = rpd[:,0]

plt.plot(x, y, label="RPD", color='red')
plt.xlabel('r')
plt.ylabel('g(r)')
plt.title('Radial Particle Density')
plt.grid(True)
plt.legend()
plt.show()

# Plot Abbreviated RPD
x = arpd[:,1]
y = arpd[:,0]

plt.plot(x, y, label="RPD", color='red')
plt.xlabel('r')
plt.ylabel('g(r)')
plt.title('Abbreviated Radial Particle Density')
plt.grid(True)
plt.legend()
plt.show()
