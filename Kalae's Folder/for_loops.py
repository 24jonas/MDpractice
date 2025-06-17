#example 1
'''
for i in range(1, 11, 2):
    print(i)
'''
#example 2: loops can be used in strings.
'''
credit_card = "1234-5678-9012-3456"

for i in credit_card:
    print(i)
'''
#example 3: skip a number in a for loop
for i in range(1,11):
    if i ==5:
        continue
    print(i)


N =27
particle_points = []
for i in range(1,6,2):
    if i == 6:
        j = 6
    for j in range(1,6,2):
        if j == 3:
            j += 0.5
        if j == 5.5:
            j = 3
        for k in range(1,6,2):
            particle_points.append((i, j, k))

print(particle_points)