# Find the best pairing of initial heading and initial velocity to maintain a stable orbit in 'Orbit1'.
# Tests initial velocities between 0 and 2, and initial headings between 0 and 90 degrees.
# There are other methods of doing this which I may try in the future.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from Functions import *

# Generate combinations.
combinations = []
for i in range(200):
    for j in range(90):
        v_0 = i/100
        theta = j*np.pi/180
        combinations.append((v_0, theta))

print("----------------------------------------------------------------------------------------------------------------------------------------")

# Initial variables.
l = 1000
check1 = []
trim1 = []
count = 0

# Exclude highly erroneous trajectories.
print("trim1 part 1")
for pair in combinations:
    v_0 = pair[0]
    theta = pair[1]

    vo = np.array([v_0*np.cos(theta), v_0*np.sin(theta)])
    co = np.zeros((l,2))
    ro = np.array([x_0, y_0])

    for i in range(l):
        ao = Acceleration(ro, r1, r2)
        ro, vo = Position_Verlet(ro, vo, ao, dt)
        co[i] = ro

    r = Radius(ro)
    check1.append((r, v_0, theta))
    count += 1
    print(count)

print("trim1 part 2")
count = 0

for i in range(len(check1)):
    if check1[i][0] < 5 and check1[i][1] != 0 and check1[i][2] != 0:
        trim1.append((check1[i][1], check1[i][2]))
    count += 1
    print(count)

print(trim1)

print("----------------------------------------------------------------------------------------------------------------------------------------")

# New variables.
l = 10000
check2 = []
trim2 = []
count = 0

# Exclude somewhat erroneous trajectories.
print("trim2 part 1")
for pair in trim1:
    v_0 = pair[0]
    theta = pair[1]

    vo = np.array([v_0*np.cos(theta), v_0*np.sin(theta)])
    co = np.zeros((l,2))
    ro = np.array([x_0, y_0])

    for i in range(l):
        ao = Acceleration(ro, r1, r2)
        ro, vo = Position_Verlet(ro, vo, ao, dt)
        co[i] = ro
        
    r = Radius(ro)
    check2.append((r, v_0, theta))
    count += 1
    print(count)

print("trim2 part 2")
count = 0

for i in range(len(check2)):
    if check2[i][0] < 5:
        trim2.append((check2[i][1], check2[i][2]))
        count += 1
        print(count)

print(trim2)

print("----------------------------------------------------------------------------------------------------------------------------------------")

# New variables.
l = 100000
check3 = []
trim3 = []
count = 0

# Exclude slightly erroneous trajectories.
print("trim3 part 1")
for pair in trim2:
    v_0 = pair[0]
    theta = pair[1]

    vo = np.array([v_0*np.cos(theta), v_0*np.sin(theta)])
    co = np.zeros((l,2))
    ro = np.array([x_0, y_0])

    for i in range(l):
        ao = Acceleration(ro, r1, r2)
        ro, vo = Position_Verlet(ro, vo, ao, dt)
        co[i] = ro
        
    r = Radius(ro)
    check3.append((r, v_0, theta))
    count += 1
    print(count)

print("trim3 part 2")
count = 0

for i in range(len(check3)):
    if check3[i][0] < 5:
        trim3.append((check3[i][1], check3[i][2]))
        count += 1
        print(count)

print(trim3)

print("----------------------------------------------------------------------------------------------------------------------------------------")

# New variables.
l = 1000000
check4 = []
solutions = []
count = 0

# Leave only very stable trajectories.
print("Solution part 1")
for pair in trim3:
    v_0 = pair[0]
    theta = pair[1]

    vo = np.array([v_0*np.cos(theta), v_0*np.sin(theta)])
    co = np.zeros((l,2))
    ro = np.array([x_0, y_0])

    for i in range(l):
        ao = Acceleration(ro, r1, r2)
        ro, vo = Position_Verlet(ro, vo, ao, dt)
        co[i] = ro
        
    r = Radius(ro)
    check4.append((r, v_0, theta))

print("Solution part 2")
count = 0

for i in range(len(check4)):
    if check4[i][0] < 5:
        solutions.append((check4[i][1], check4[i][2]))
        count += 1
        print(count)

print(solutions)

print("----------------------------------------------------------------------------------------------------------------------------------------")

print("Complete")

solutions.append((0,0))
