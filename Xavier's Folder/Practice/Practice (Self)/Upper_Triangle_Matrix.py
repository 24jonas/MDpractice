# Solve system of equations. Input array and output the upper triangular form. Row index is i and column index is j.

# Imports
import numpy as np
import copy

# Matrix
a = np.array([[0.0, 1.0, 1.0, 1.0, 0.0],       # Objects must be entered as floats. For example, enter 3 as 3.0
             [3.0, 0.0, 3.0, -4.0, 7.0],
             [1.0, 1.0, 1.0, 2.0, 6.0],
             [3.0, 3.0, 1.0, 3.0, 6.0]])
m = a.shape[0]
n = a.shape[1] - 1

# Create upper triangular form.
for j in range(n):
    for i in range(m):
        if a[i][j] != 0 and i >= j:
            r = np.copy(a[i])
            a[i] = a[j]
            a[j] = r
            for i in range(m-j-1):
                a[i+j+1] = a[i+j+1] - (a[i+j+1][j]/r[j])*r

# Print resultaant matrix
print(a)

"""
# Solve the system of equations.
sol = []
if m == n-1:
    x4 = a[m][n]/a[m][n-1]
    x3 = (a[m-1][n] - a[m-1][n-1]*x4)/a[m-1][n-2]
    x2 = a[m-2][n]
    x1 = a[m-3][n]
"""