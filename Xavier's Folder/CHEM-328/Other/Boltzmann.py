# Plotting the Maxwell-Boltzmann distribution.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k, N_A, pi

# Variables
t = 300.0
mm = 28e-3
m = mm / N_A

# Function
def mbs(v, t, m):
    prefactor = 4*pi*(m/(2*pi*k*t))**(1.5)
    return prefactor*(v**2)*np.exp(-m*(v**2)/(2*k*t))

# Generate Data
v = np.linspace(0.0, 2000.0, 1200)
f = mbs(v, t, m)

# Charactereistic Speeds
v_mp = np.sqrt(2*k*t/m)
v_mean = np.sqrt(8*k*t/(pi*m))
v_rms = np.sqrt(3*k*t/m)

# Plot
plt.plot(v, f, label = "MB Speed PDF")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of Sin(x)')
plt.axis('equal')
plt.legend()
plt.show()
# plt.savefig("Boltzmann N2 300K.png", dpi =300)    # Saves the plot.

# Print Speeds
print(f"Most probable speed = {v_mp:0.2f} m/s")
