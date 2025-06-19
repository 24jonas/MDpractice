# 3d with variable molecules in cubic pattern. LJ collision dynamics except for classical dynamics with walls. Does not use the VV algorithm.

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
for i in range(l):
    rx, vx = Wall_x(rx, vx, ro, i)
    ry, vy = Wall_y(ry, vy, ro, i)
    rz, vz = Wall_z(rz, vz, ro, i)

    pot1 = []
    acc1 = []
    for j in range(q):
        r1 = np.array([rx[i][j],ry[i][j],rz[i][j]])
        v = np.array([vx[i][j],vy[i][j],vz[i][j]])

        pot2 = []
        acc2 = []
        for k in range(q):
            r2 = np.array([rx[i][k],ry[i][k],rz[i][k]])

            if r1[0] == r2[0] and r1[1] == r2[1] and r1[2] == r2[2]:
                a = np.zeros((1,3))
                lj = 0
            else:
                n, r = Vectors(r1, r2)
                lj, a = Lennard_Jones(r, n, m)
            
            acc2.append(a)
            pot2.append(lj/2)
        
        sum1 = np.zeros((1,3))
        for w1 in range(q):
            sum1 += acc2[w1]
        acc1.append(sum1)

        sum2 = 0
        for w2 in range(q):
            sum2 += pot2[w2]
        pot1.append(sum2)

    sum3 = 0
    for w3 in range(q):
        k = Kinetic_Energy(vx, vy, vz, m, i, w3)
        if k > el or k < -el:
            k = el
        sum3 += k
    ke[i] = [sum3]

    sum4 = 0
    for w4 in range(q):
        if pot1[w4] > el or pot1[w4] < -el:
            pot1[w4] = el
        sum4 += pot1[w4]
    pe[i] = [sum4]

    te[i] = [sum3 + sum4, i]

    ate[i] = np.sum(te[:,0], axis=0)/i
    ape[i] = np.sum(pe, axis=0)/i
    ake[i] = np.sum(ke, axis=0)/i

    rx, ry, rz, vx, vy, vz = Update(rx, ry, rz, vx, vy, vz, acc1, dt, i, l, sl)
    print(i+1)

print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Complete")

# Plot energy types over time
x = te[:,1]
y1 = te[:,0]
y2 = pe
y3 = ke
y4 = ate
y5 = ape
y6 = ake

plt.plot(x, y1, label="Total Energy", color='purple')
plt.plot(x, y2, label="Potential Energy", color='blue')
plt.plot(x, y3, label="Kinetic Energy", color='red')
plt.plot(x, y4, label="Total Energy Average", color='purple')
plt.plot(x, y5, label="Potential Energy Average", color='blue')
plt.plot(x, y6, label="Kinetic Energy Average", color='red')
plt.xlabel('Time Increments')
plt.ylabel('Energy')
plt.title('LJ Collision')
plt.grid(True)
#plt.legend()      # Optionally plot the legend.
plt.show()
