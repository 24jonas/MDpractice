import numpy as np  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

"""
Here 27 atoms where simulated, with box length size S = 6.0. They have a cold start with 0 velocity and
they are arranged in a checkerboard pattern, also known as FCC structure.

"""
vertices = [
    [0, 0, 0],  # Vertex 1
    [6, 0, 0],  # Vertex 2
    [6, 6, 0],  # Vertex 3
    [0, 6, 0],  # Vertex 4
    [0, 0, 6],  # Vertex 5
    [6, 0, 6],  # Vertex 6
    [6, 6, 6],  # Vertex 7
    [0, 6, 6]   # Vertex 8
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
Num_particles = 27 
points_to_plot = []

for i in range(Num_particles):
    for j in range(4):
        for k in range(4):
            points_to_plot.append([i * 2, j * 2, k * 2])  # Create vertices at intervals of 2


"""
points_to_plot = np.array([
    [2, 0, 2],
    [4, 0, 2],
    [6, 0, 2],
    [2, 2.5, 2],
    [4, 2.5, 2],
    [6, 2.5, 2],
    [2, 5, 2],
    [4, 5, 2],
    [6, 5, 2],
    # set 1
    [2, 0, 4],
    [4, 0, 4],
    [6, 0, 4],
    [2, 3, 4],
    [4, 3, 4],
    [6, 3, 4],
    [2, 6, 4],
    [4, 6, 4],
    [6, 6, 4],
    # set 2
     [2, 0, 6],
    [4, 0, 6],
    [6, 0, 6],
    [2, 2.5, 6],
    [4, 2.5, 6],
    [6, 2.5, 6],
    [2, 5, 6],
    [4, 5, 6],
    [6, 5, 6],
    # set 3

])
"""
x_coords = [point[0] for point in points_to_plot]
y_coords = [point[1] for point in points_to_plot]
z_coords = [point[2] for point in points_to_plot]

x_coords = points_to_plot[:, 0]
y_coords = points_to_plot[:, 1]
z_coords = points_to_plot[:, 2]
    




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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