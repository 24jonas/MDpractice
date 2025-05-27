import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

""" This is my Newtons method practice code. It is code that I got
off an online tutorial so I can understand how to use the Varlet method.""" 

def newtonsmethod(func, funcderiv, x,n):
    def f(x):
        f= eval(func)
        return f
    def df(x):
        df = eval(funcderiv)
        return df
    
    for intercepts in range(1,n):
        i = x -f(x)/df(x)
        x=i

    print(f"The root is: {x}")


newtonsmethod("x**2-2", "2*x", 1, 10)
# This is a simple implementation of Newton's method for finding roots of a function.