class Mammal:
    def walk(self):
        print("Walk")

class Dog(Mammal):
    def bark(self):
        print("bark")

class Cat(Mammal):
    pass

dog1 = Dog()
dog1.bark()

print("______________________")

import Converters
from Converters import kg_lbs

print(Converters.kg_lbs(70))

print("____________________")

from utils import find_max

numbers = [1, 5, 5, 4, 3, 7, 8, 9, 4, 5, 2, 7, 7, 6, 7, 0, 2, 6, 0]

print(find_max(numbers))
