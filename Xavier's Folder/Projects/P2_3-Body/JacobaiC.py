# Plot the jacobai constant over time.

# Imports
import matplotlib.pyplot as plt
import numpy as np
from OrbitC import jc, jo_0

x = np.linspace(0, 1, 5655)
y = jc - jo_0

plt.plot(x, y, label='Jacobai Constant', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Jacobai Constant OrbitB')
plt.grid(True)
plt.legend()
plt.show()
