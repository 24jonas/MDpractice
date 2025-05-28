import numpy as np

## Declaration of Variables for Project 2 ##

## Star 1 ##
R_1 = np.array([[-0.5, 0]])
m_1 = 0.5

## Star 2 ##
R_2 = np.array([[0.5, 0]])
m_2 = 0.5

## Third Body ##
theta = 71.939
theta = theta * 2 * np.pi / 360
v_0 = 1.2

R = pos_list = np.array([[0, 0]])
V = vel_list = np.array([[v_0 * np.cos(theta), v_0 * np.sin(theta)]])

## Other Variables ##
dt = 0.005              # Step Size
num = 1800              # Number of Steps
n = 4                   # Number of Periods
