import numpy as np

## Declaration of Variables for Project 2 ##

## Star 1 ##
R_1 = np.array([[-0.5, 0]])
m_1 = 0.5

## Star 2 ##
R_2 = np.array([[0.5, 0]])
m_2 = 0.5

## Third Body ##
theta = 45
theta = theta * 2 * np.pi / 360
v_0 = 1.0

R = pos_list = np.array([[0, 0]])
V = vel_list = np.array([[v_0 * np.cos(theta), v_0 * np.sin(theta)]])

## Other Variables ##
dt = 0.005              # Step Size
num = 450               # Number of Steps
n = 5                   # Number of Periods
