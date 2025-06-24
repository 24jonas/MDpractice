particle_points = []
for i in range(1, 6, 2):
    for j in range(1, 6, 2):
        for k in range(1, 6, 2):
            if k == 3:
                particle_points.append((i, j + 0.75, k))
            else:
                particle_points.append((i, j, k))


print(len(particle_points))