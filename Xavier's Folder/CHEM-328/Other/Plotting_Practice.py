# CHEM-328 Plotting Practice

# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

# Generate Data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot
plt.plot(x, y, label = "y = sin(x)")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of Sin(x)')
plt.axis('equal')
plt.legend()
plt.show()
