# Simulates the behavior of 3 masses all of which are mobile. Ues the PV algorithm.

# Imports
import matplotlib.pyplot as plt
from FunctionsB import *

# Adjust Variables
l = 5655
x_0 = 0
y_0 = 0.058
v_x = 0.49
v_y = 0
ro = np.array([x_0, y_0])
vo = np.array([v_x, v_y])
co = np.zeros((l,2))
c1 = np.zeros((l,2))
c2 = np.zeros((l,2))
jc = np.zeros((l))
jc_0 = np.zeros((l))

# Construct the trajectory array.
jo = Jacobai_Constant(ro, r1, r2, vo)
jo_0 = jo
for i in range(l):
    co[i] = ro
    c1[i] = r1
    c2[i] = r2
    jc[i] = jo
    jc_0[i] = jo_0
    r1, r2 = Star_Positions(t)
    ro, vo = Position_Verlet(ro, vo, dt, r1, r2, m1, m2)
    jo = Jacobai_Constant(ro, r1, r2, vo)
    t += dt

# Plot the trajectory.
x = co[:, 0]
y = co[:, 1]

x1 = r1[0]
y1 = r1[1]

x2 = r2[0]
y2 = r2[1]

plt.plot(x, y, label='Trajectory', color='black')
plt.plot(x1, y1, 'ro', label='Star1', color='red')
plt.plot(x2, y2, 'ro', label='Star2', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('OrbitB')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

print("----------------------------------------------------------------------------------------------------------------------------------------")

print("Complete")
