import matplotlib.pyplot as plt
import numpy as np

"""
theta = np.linspace(0, 2*np.pi, 360)
x = (np.cos(theta))/(1 + (0.9)*np.cos(theta))
y = (np.sin(theta))/(1 + (0.9)*np.cos(theta))
"""

a = np.zeros((360,2))

for i in range(360):
    x = (np.cos(i*np.pi/180))/(1 - (0.9)*np.cos(i*np.pi/180))
    y = (np.sin(i*np.pi/180))/(1 - (0.9)*np.cos(i*np.pi/180))
    a[i] = [x, y]

print(a)
