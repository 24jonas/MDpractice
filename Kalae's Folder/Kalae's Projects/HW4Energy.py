"""
1.)
To check your program, do the following very simple run. Let particle 1 be initially resting at (7.0, 6.0). Let
particle 2 collide into it with velocity (0.4,0.0) from the initial position (3.0, 6.0). Take DT=0.01. Run about 10,000
time step and graph the x-coordinate of both particle every 50 or 100 time-steps. (Can you animate this so that
you can see what’s going on? Represent each particle by plotting a circle of radius 0.5 ) Graph also the velocity of
both particles. Finally, plot out also the total energy, the kinetic energy, and the potential energy all in one graph.
Compared to the initial energy, how accurately is energy conserved? What kind of scattering is being simulated?
"""





import numpy as np
import matplotlib.pyplot as plt 

n=1
steps_list=[]
#Particle 1
energy1= []
x1=700
y1=600
vx1= 0
radius1=20
color_1=(255, 255, 255)

#Particle 2
energy2= []
x2=300
y2=600
vx2= 40
radius2=20
color_2=(0, 0, 255)

#total energy
energy_total= []


rate =100
dt = 1/rate
space_size=1000
steps =10000

Backround=(0, 0, 0)

times_run= 0

# Simple pygame program

# Import and initialize the pygame library
import pygame
pygame.init()

# Set up the drawing window
screen = pygame.display.set_mode([space_size, space_size])

# Run until the user asks to quit

clock =pygame.time.Clock()

energy1.append((1/2)*vx1**2)
energy2.append((1/2)*vx2**2)
energy_total.append(0.5*((vx1**2)+vx2**2))
steps_list.append(times_run)

running = True
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill(Backround)

    # Draw a solid blue circle in the center
    pygame.draw.circle(screen,color_1, (float(x1),400), radius1)

    pygame.draw.circle(screen,color_2, (float(x2),400), radius2)

    #collisions with wall
    if(x1-radius1 < 0):
        x1 = 1000-radius1
    elif(x1+radius1 >space_size):
        x1=0+radius1

    if(x2-radius2 < 0):
        x2 = 10000-radius2

    elif(x2+radius2 >space_size):
        x2=0+radius2

    #particle collisions
    if(abs(x2-x1) < (radius1+radius2)):
        vx1, vx2 = vx2,vx1

    # Flip the display
    pygame.display.flip()

    #kinematics
    x1 = x1+vx1*dt

    x2 =x2+vx2*dt
    #Creating an array of the kinetic and potential energy

    clock.tick(rate)

    times_run += 1
    
    #once it runs steps amount of time it will set the velocity zero for both.
    if times_run > steps:
        vx1,vx2 = 0,0

    if times_run == 50*n:
        n+=1
        energy1.append((1/2)*vx1**2)
        energy2.append((1/2)*vx2**2)
        energy_total.append(0.5*((vx1**2)+vx2**2))
        steps_list.append(times_run)


plt.plot(steps_list, energy1, label='Kinetic Energy 1', color='blue')
plt.plot(steps_list, energy2, label='Kinetic Energy 2', color='red')
plt.plot(steps_list, energy_total, label= 'total energy ', color='green')
plt.xlabel("Step number")
plt.ylabel("Kinetic Energy")
plt.grid(True)
plt.legend()
plt.show()