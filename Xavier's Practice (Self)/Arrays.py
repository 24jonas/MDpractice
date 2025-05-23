import numpy as np

a = np.array([1, 2, 3, 4, 5])

print(a)

print(type(a))

print(a[2])

a[2] = 10

print(a)

a[2] = 3

print(a.shape)

a = a + 5

print(a)

b = np.array([11, 12, 13, 14, 15])

print(a + b)

c = np.random.rand(10, 2)
d = np.zeros((10, 2))

print(c)
print(d)

for i in range(10):
    x = i
    y = i + 1
    d[i] = [x,y]

print(d)
