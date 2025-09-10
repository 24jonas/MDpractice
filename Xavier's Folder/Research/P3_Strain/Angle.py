# Find integer values of m and n from chosen angle theta.

# Imports
import numpy as np
import math
from Functions import theta

# Variables
search = 10
a = b = c = 1
d = round(np.cos(theta) , 3)

# Find m and n values
for i in range(search):
    for j in range(search):
        e = round((a**2 + 4*a*b + b**2)/(2*a**2 + 2*a*b + 2*b**2) , 3)
        if d == e and math.gcd(a, b) == 1:
            c = 0
            break
        b += 1
    if c == 0:
        break
    a += 1
    b = 1

m = a
n = b
