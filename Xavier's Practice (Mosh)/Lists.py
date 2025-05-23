numbers = [5, 4, 3, 7, 5, 10, 5, 9, 9, 7, 8, 4, 1]
max = numbers[0]

for n in numbers:
    if n > max:
        max = n
print(max)

# 2D Lists

"""
(0,0) (0,1) (0,2)
(1,0) (1,1) (1,2)
(2,0) (2,1) (2,2)
"""

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print(matrix[0][1])
print("____________")

# List Methods

numbers = [4, 3, 3, 7, 6, 8, 5, 5, 1, 1, 1, 0, 1, 6, 7]
print(numbers.count(1))
numbers.insert(4, 5)
numbers.remove(1)
numbers.pop()
print(numbers)

numbers.sort()
print(numbers)

print(numbers.index(5))
print("B" in numbers)
numbers.clear()
print(numbers)
print("______________________")

numbers = [4, 3, 3, 7, 6, 8, 5, 5, 1, 1, 1, 0, 1, 6, 7]

for n in numbers:
    x = numbers.count(n)
    for i in range(x-1):
        numbers.remove(n)

print(numbers)