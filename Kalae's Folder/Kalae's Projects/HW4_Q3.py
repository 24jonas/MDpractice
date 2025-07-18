import numpy as np  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

"""
Here 27 atoms where simulated, with box length size S = 6.0. They have a cold start with 0 velocity and
they are arranged in a checkerboard pattern, also known as FCC structure.

"""
N = int(input("Enter the number of particles of choice N= 27 or N=64: "))
if N == 27:
    vertices = [
        [0, 0, 0], [6, 0, 0], [6, 6, 0], [0, 6, 0],
        [0, 0, 6], [6, 0, 6], [6, 6, 6], [0, 6, 6]
               ]

    edges = [
    [vertices[0], vertices[1]],  # Edge from vertex 1 to 2
    [vertices[1], vertices[2]],  # Edge from vertex 2 to 3
    [vertices[2], vertices[3]],  # Edge from vertex 3 to 4
    [vertices[3], vertices[0]],  # Edge from vertex 4 to 1
    [vertices[4], vertices[5]],  # Edge from vertex 5 to 6
    [vertices[5], vertices[6]],  # Edge from vertex 6 to 7
    [vertices[6], vertices[7]],  # Edge from vertex 7 to 8
    [vertices[7], vertices[4]],  # Edge from vertex 8 to 5
    [vertices[0], vertices[4]],  # Edge from vertex 1 to 5
    [vertices[1], vertices[5]],  # Edge from vertex 2 to 6
    [vertices[2], vertices[6]],  # Edge from vertex 3 to 7
    [vertices[3], vertices[7]]   # Edge from vertex 4 to 8
]

particle_points = []
for i in range(1,6,2):
    for j in range(1,6,2):
        for k in range(1,6,2):
            if k == 3:
                j += 0.75
                particle_points.append((i, j, k))
                j = j - 0.75
            else:
                particle_points.append((i, j, k))

print(len(particle_points), "particles in total")


if N == 64:
    vertices = [
        [0, 0, 0],  # Vertex 1
        [10, 0, 0],  # Vertex 2
        [10, 10, 0],  # Vertex 3
        [0, 10, 0],  # Vertex 4
        [0, 0, 10],  # Vertex 5
        [10, 0, 10],  # Vertex 6
        [10, 10, 10],  # Vertex 7
        [0, 10, 10]   # Vertex 8
    ]

    edges = [
    [vertices[0], vertices[1]],  # Edge from vertex 1 to 2
    [vertices[1], vertices[2]],  # Edge from vertex 2 to 3
    [vertices[2], vertices[3]],  # Edge from vertex 3 to 4
    [vertices[3], vertices[0]],  # Edge from vertex 4 to 1
    [vertices[4], vertices[5]],  # Edge from vertex 5 to 6
    [vertices[5], vertices[6]],  # Edge from vertex 6 to 7
    [vertices[6], vertices[7]],  # Edge from vertex 7 to 8
    [vertices[7], vertices[4]],  # Edge from vertex 8 to 5
    [vertices[0], vertices[4]],  # Edge from vertex 1 to 5
    [vertices[1], vertices[5]],  # Edge from vertex 2 to 6
    [vertices[2], vertices[6]],  # Edge from vertex 3 to 7
    [vertices[3], vertices[7]]   # Edge from vertex 4 to 8
   ]




    particle_points = []
    for i in range(1, 10, 3):
        for j in range(1, 10, 3):
            for k in range(1, 10, 3):
                if k == 4:
                    particle_points.append((i, j+1.5, k))
                elif k == 1 or k == 7:
                    particle_points.append((i, j, k))




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

particle_points = np.array(particle_points)
x_coords = particle_points[:, 0]
y_coords = particle_points[:, 1]
z_coords = particle_points[:, 2]


ax.scatter3D(x_coords, y_coords, z_coords, color='green', s=75)


for edge in edges:
    x_coords = [edge[0][0], edge[1][0]]
    y_coords = [edge[0][1], edge[1][1]]
    z_coords = [edge[0][2], edge[1][2]]
    ax.plot(x_coords, y_coords, z_coords, color='blue')

    ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Box Plot')

plt.show()