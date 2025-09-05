# Functions for 'Strain.py"

# Imports
import numpy as np

# Variables
Lattice_Constant = 2.46
Repetitions = 15
Twist_Angle = 21.79
Window_Size = 100
Strain_x = 1.1
Strain_y = 1
Strain_z = 1
Deleted_Coordinates1 = []
Deleted_Coordinates2 = []
Coordinates1 = np.zeros((2*Repetitions**2, 3))
Coordinates2 = np.zeros((2*Repetitions**2, 3))
Coordinates3 = np.zeros((3, 3))
Coordinates4 = np.zeros((4, 3))
primitive_lattice_vector_1 = np.zeros((1,3))
primitive_lattice_vector_2 = np.zeros((1,3))
supercell_lattice_vector1 = np.zeros((1,3))
supercell_lattice_vector2 = np.zeros((1,3))
S_Offset = np.zeros((1,3))
B_Offset = np.array([Lattice_Constant/np.sqrt(3), 0, 0])
Z_Offset = np.array([0, 0, Lattice_Constant])
Strain_Matrix = np.array([[Strain_x, 0, 0],
                          [0, Strain_y, 0],
                          [0, 0, Strain_z]])

# Renaming
a = Lattice_Constant
r = Repetitions
theta = Twist_Angle*np.pi/180
ws = Window_Size
dc1 = Deleted_Coordinates1
dc2 = Deleted_Coordinates2
co1 = Coordinates1
co2 = Coordinates2
co3 = Coordinates3
co4 = Coordinates4
plv1 = primitive_lattice_vector_1
plv2 = primitive_lattice_vector_2
slv1 = supercell_lattice_vector1
slv2 = supercell_lattice_vector2
s = S_Offset
b = B_Offset
z = Z_Offset
SM = Strain_Matrix

# Functions
def PLV(a, plv1, plv2):
    plv1[0] = [a*np.sqrt(3)/2, a/2, 0]
    plv2[0] = [a*np.sqrt(3)/2, -a/2, 0]
    return plv1, plv2

def Initial(s, r, plv1, plv2):
    s = (r-1)*plv1/2 + (r-1)*plv2/2
    return s

def SLVA(m, n, plv1, plv2, slv1, slv2):
    mod = n/3
    slv1 = (m+mod)*plv1 + mod*plv2
    slv2 = -mod*plv1 + (m+2*mod)*plv2
    return slv1, slv2

def SLVB(m, n, plv1, plv2, slv1, slv2):
    slv1 = m*plv1 + (m+n)*plv2
    slv2 = -(m+n)*plv1 + (2*m+n)*plv2
    return slv1, slv2

def Build(x, y, s, z, b, co1, co2, plv1, plv2, c):
    co1[2*c] = x*plv1 + y*plv2 - s
    co1[2*c+1] = co1[2*c] + b
    co2[2*c] = co1[2*c] + z
    co2[2*c+1] = co1[2*c+1] + z
    return co1, co2

def Twist(co2, theta, i):
    x, y = co2[i][0], co2[i][1]
    r = np.sqrt(x**2 + y**2)
    angle = np.arctan(y/x)
    if x == 0 and y == 0:
        angle = 0
    if x < 0:
        angle += np.pi
    x = r*np.cos(angle+theta)
    y = r*np.sin(angle+theta)
    co2[i][0], co2[i][1] = x, y
    return co2

def Reorient(co4, slv1, slv2):
    co4[0] = (-slv1 - slv2)/2
    co4[1] = (slv1 - slv2)/2
    co4[2] = (-slv1 + slv2)/2
    co4[3] = (slv1 + slv2)/2
    x, y = co4[3][0], co4[3][1]
    angle = np.arctan(y/x)
    if x < 0:
        angle += np.pi
    for i in range(4):
        x, y = co4[i][0], co4[i][1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan(y/x)
        if x < 0:
            theta += np.pi
        x = r*np.cos(theta - angle)
        y = r*np.sin(theta - angle)
        co4[i][0], co4[i][1] = x, y
    if co4[1][1] < co4[2][1]:
        copy = np.copy(co4[2])
        co4[2] = co4[1]
        co4[1] = copy
    return co4, angle

def Shave(co, co4, dc, i):
    v1 = co4[3] - co4[1]
    v2 = co4[1] - co4[0]
    v3 = co4[2] - co4[0]
    v4 = co4[3] - co4[2]
    x, y = co[i][0], co[i][1]
    if x >= 0 and y > 0:
        slope = v1[1]/v1[0]
        int = co4[1][1]
        f = slope*x + int
        if y > f:
            dc.append(i)
        pass
    elif x < 0 and y >= 0:
        slope = v2[1]/v2[0]
        int = co4[1][1]
        f = slope*x + int
        if y > f:
            dc.append(i)
        pass
    elif x <= 0 and y < 0:
        slope = v3[1]/v3[0]
        int = co4[2][1]
        f = slope*x + int
        if y < f:
            dc.append(i)
        pass
    elif x > 0 and y <= 0:
        slope = v4[1]/v4[0]
        int = co4[2][1]
        f = slope*x + int
        if y < f:
            dc.append(i)
        pass
    return dc
