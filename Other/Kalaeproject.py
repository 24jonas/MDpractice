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
    r_plus = p/(1+e*np.cos(theta))
    r_minus = p/(1-e*np.cos(theta))
    x_plus = r_plus * np.cos(theta)
    y_plus = r_plus * np.sin(theta)
    x_minus = r_minus * np.cos(theta)
    y_minus = r_minus * np.sin(theta)
    # Create the plot
    plt.plot(x_plus, y_plus, label=' Orbit 1 (e < 1)', color='blue')
    plt.plot(x_minus, y_minus, label='Orbit 2 (e > 1)', color='red')
    plt.plot(0, 0, 'ro', label='Central Body')  # Central body at origin
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




## Solve the ODE using solve_ivp
   ## solution = solve_ivp(dydt, t_span, y0, t_eval=t_eval)

    # Plot the solution
   ## plt.plot(solution.t, solution.y[0], label='y(t)', color='blue')
   ## plt.title('Solution of dy/dt = -2y with y(0) = 5')
   ## plt.xlabel('Time t')
   #### plt.ylabel('y(t)')
   ## plt.grid(True)
   ## plt.legend()
    ##plt.show()##