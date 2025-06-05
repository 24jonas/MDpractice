"""
1.)
To check your program, do the following very simple run. Let particle 1 be initially resting at (7.0, 6.0). Let
particle 2 collide into it with velocity (0.4,0.0) from the initial position (3.0, 6.0). Take DT=0.01. Run about 10,000
time step and graph the x-coordinate of both particle every 50 or 100 time-steps. (Can you animate this so that
you can see what’s going on? Represent each particle by plotting a circle of radius 0.5 ) Graph also the velocity of
both particles. Finally, plot out also the total energy, the kinetic energy, and the potential energy all in one graph.
Compared to the initial energy, how accurately is energy conserved? What kind of scattering is being simulated?
"""
# This is the animation
import numpy as np
import matplotlib.pyplot as plt     
import pygame, pymunk, math
from random import randint

#Particle 1
particle1= []
x1=200
vx1= 200
radius1=20
color_1=(255, 255, 255)

#Particle 2
particle2= []
x2=600
vx2= 0
radius2=20
color_2=(0, 0, 255)



rate =60
dt = 1/rate
space_size=800

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
    if(x1-radius1 < 0) or (x1+radius1 >space_size):
        vx1 = -vx1

    if(x2-radius2 < 0) or (x2+radius2 >space_size):
        vx2 = -vx2

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

# Done! Time to quit.
pygame.quit()