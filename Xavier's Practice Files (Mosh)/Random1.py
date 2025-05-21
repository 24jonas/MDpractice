from Package1.dice import Dice
import random

"""
for i in range(3):
    print(random.random())

for i in range(3):
    print(random.randint(10, 20))
"""

x = Dice()
y = x.roll()
# print(y)

print(Dice().roll())
