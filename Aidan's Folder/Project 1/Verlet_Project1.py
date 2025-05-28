import numpy as np
import matplotlib.pyplot as plt

##### Part 1 #####

## Declaring variables ##
num = 500
x_0 = 10
vy_0 = 1 / 10
E_0 = 0.5 * vy_0 ** 2 - 1 / x_0
smaj = -0.5 / E_0
per = 2 * np.pi * smaj ** 1.5
h = x_0 * vy_0
p = h * h
e = np.sqrt(1 - p / smaj)

## Placeholder Variables to Store Results ##
result = np.empty((0,2))
x_val = []
y_val = []

## Main Loop to Analytically Represent the Orbit ##
for i in range(num + 1):

    theta = (i * 2 * np.pi) / num
    rad = p / (1 - e * np.cos(theta))

    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    
    x_val.append(x)
    y_val.append(y)
    result = np.append(result, np.array([[x, y]]), axis = 0)

## Graphing the Orbit and Printing Array ##
plt.plot(x_val, y_val)
plt.show()
print(result)




##### Part 2 #####

## Acceleration Function ##
def acc(R):

    r_mag = np.sqrt(R[0, 0] ** 2 + R[0, 1] ** 2)
    A = -R / (r_mag ** 3)

    return A

## Verlet Algorithm Function ##
def verlet(R, V, dt):

    R = R + 0.5 * dt * V
    A = acc(R)
    V = V + dt * A
    R = R + 0.5 * dt * V
    
    pos = R
    vel = V

    return pos, vel

## Declaring Variables ##
dt = per / 500
R = [x_0, 0.0]
V = [0.0, vy_0]
n = 4
pos = pos_output = np.array([R])
vel = vel_output = np.array([V])

## Main Loop ##
for i in range(num * n):

    pos, vel = verlet(pos, vel, dt)


    pos_output = np.append(pos_output, pos, axis = 0)
    vel_output = np.append(vel_output, vel, axis = 0)

## Printing the Array ##
print(pos_output, "\n", vel_output)

## Making a list out of each column of the Array ##
x = [i[0] for i in pos_output]
y = [i[1] for i in pos_output]

## Plotting the graph ##
plt.plot(x,y)
plt.show()