## Project #1 Keplarian Orbits.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Create x values
def function_Q1():
    
    x_0=10
    v_0= 1/10
    h=x_0*v_0
    Energy_0= 1/2*v_0**2-1/x_0
    a= -1/(2*Energy_0)
    Period = 2*np.pi*a**(3/2)
    print("The semi-major axis is: ", {Period})
    p=h**2
    e=np.sqrt(1-p/a)
    theta= np.linspace(0, 2*np.pi, 100) 
    #r_plus = p/(1+e*np.cos(theta))
    r_minus = p/(1-e*np.cos(theta))
   # r_final= np.sqrt(r_plus**2+r_minus**2)
    x_final = r_minus * np.cos(theta)
    y_final = r_minus * np.sin(theta)
    # This section calculates the foci of the ellipse
    #foci = np.sqrt(a**2 - b**2)
    a  = int((r_minus * np.cos(2*np.pi)+r_minus * np.cos(np.pi))/2)
    print(a)
    # Create the plot
    plt.plot(x_final, y_final, label=' Orbit 1 (e < 1)', color='blue')
    plt.plot()
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
   

    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.title('Keplerian Orbit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')  # Equal scaling for x and y axes
    plt.legend()
    plt.show()
# Create y values


function_Q1()




""" Solve the ODE using solve_ivp
    solution = solve_ivp(dydt, t_span, y0, t_eval=t_eval)

    Plot the solution
    plt.plot(solution.t, solution.y[0], label='y(t)', color='blue')
    plt.title('Solution of dy/dt = -2y with y(0) = 5')
    plt.xlabel('Time t')
   plt.ylabel('y(t)')
   plt.grid(True)
   plt.legend()
    plt.show()"""

