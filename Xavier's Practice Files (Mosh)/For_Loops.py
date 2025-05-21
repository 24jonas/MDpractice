for i in range(10):
    print(i)

for i in range(5,10):
    print(i)

for i in range(2,10,2):
    print(i)

prices = [10, 20, 30]

total = 0
for price in prices:
    total += price
print(f"Total: {total}")

# Nested Loops

for x in range(4):
    for y in range(3):
        print(f'({x}, {y})')

numbers = [5, 2, 5, 2, 2]
line = ""

for n in numbers:
    for i in range(n):
        line = line + "x"
    print(line)
    line = ""
