# Plot the jacobai constant over time.

# Imports
import matplotlib.pyplot as plt
import numpy as np
from OrbitB import jc, jo_0

x = jc[:, 0]
y = jc[:, 1] - jo_0

plt.plot(x, y, label='Jacobai Constant', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Jacobai Constant OrbitB')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
