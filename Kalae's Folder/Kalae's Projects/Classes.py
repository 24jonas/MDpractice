import numpy as np
import matplotlib.pyplot as plt 

"""
Kala'e Abrams Following along with a classes tutorial 
for the context of machine learning.
"""

class Pet:
# __init__ method to initialize the pet's name and species
    def __init__(self, age, collar_number):
        self.age = age
        self.collar_number = collar_number
    def method(self):
        print(f"This is a {self.species} named {self.name}.")

age =5
collar_number = "12345"
dog = Pet(age, collar_number)

print((int(dog.collar_number)-50))



angles = np.arange(0, 360, 1)


print(angles[359])