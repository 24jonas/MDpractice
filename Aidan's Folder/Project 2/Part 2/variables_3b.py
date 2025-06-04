import numpy as np

## Declaration of Variables and Constants for Project 2 Part 2 ##

## Moving Stars Functions ##
def r_1(t):

    r_x = -0.5 * np.cos(t)
    r_y = -0.5 * np.sin(t)
    R_1 = np.array([[r_x, r_y]])

    return R_1

def r_2(t):

    r_x = 0.5 * np.cos(t)
    r_y = 0.5 * np.sin(t)
    R_2 = np.array([[r_x, r_y]])
    
    return R_2

## Initial Conditions for Moving Particle ##
R = pos_list_1 = pos_list_2 = np.array([[0.0, 0.058]])
V = vel_list_1 = vel_list_2 = np.array([[0.49, 0]])

## Initial Conditions for Stars ##
m_1 = 0.5
m_2 = 0.5

## Run Parameters ##
P = 9 * np.pi
dt = 0.005
num = round(P / dt)
n = 1
t = 0

## Other Variables ##
s = 2 ** (1 / 3)

## Jacobi Constant Function ##
def J(t):
    J = V ** 2 - 1 / (R - r_1(t)) - 1 / (R-r_2(t) - (np.cross(2 * R, V)))

    return J

J_0 = J_FR_list = J_PV_list = J(0)
