# Functions for 'CollisionsC.py'.

# Imports
import numpy as np

# Input Variables
number_of_objects = 27      # Must have an integer cube root.
time_increment = 0.01
number_of_increments = 10000
initial_offset = 0.5
initial_velocity = 0        # In each dimension.
object_mass = 0.1
object_radius = 0.5
buffer = 5
initial_separation = 2
box_length = initial_separation*(np.cbrt(number_of_objects)-1) + 2*buffer
box_x_length = box_length
box_y_length = box_length
box_z_length = box_length
speed_limit = 100
energy_limit = 100
bin_size = 0.5
begin_radii = 1000
density_domain = 100
abbreviated_density_domain = 20
rx = "X Position Array"
ry = "Y Position Array"
rz = "Z Position Array"
vx = "X Velocity Array"
vy = "Y Velocity Array"
vz = "Z Velocity Array"
te = "Total Energy"
pe = "Potential Energy"
ke = "Kinetic Energy"
ate = "Average Total Energy"
ape = "Average Potential Energy"
ake = "Average Kinetic Energy"
bb = "nbin Array"
rl = "Radii List"
rpd = "Density As A Functino Of Radius"
arpd = "Abbrevaited RPD"

# Rename Variables
q = number_of_objects
p = int(np.cbrt(number_of_objects))
dt = time_increment
l = number_of_increments
io = initial_offset
iv = initial_velocity
isn = initial_separation
m = object_mass
ro = object_radius
bf = buffer
blx = box_x_length
bly = box_y_length
blz = box_z_length
sl = speed_limit
el = energy_limit
bs = bin_size
br = begin_radii
dd = density_domain
add = abbreviated_density_domain
rx = np.zeros((l,q))
ry = np.zeros((l,q))
rz = np.zeros((l,q))
vx = np.zeros((l,q))
vy = np.zeros((l,q))
vz = np.zeros((l,q))
te = np.zeros((l,2))
pe = np.zeros((l,1))
ke = np.zeros((l,1))
ate = np.zeros((l,1))
ape = np.zeros((l,1))
ake = np.zeros((l,1))
bb = np.zeros(dd+1)
rl = []
rpd = np.zeros((dd+1,2))
arpd = np.zeros((add+1,2))

# Functions
def Wall_x(rx, vx, ro, i):
    for j in range(q):
        if rx[i][j] < ro:
            rx[i][j] = ro
            vx[i][j] = -vx[i][j]
        elif rx[i][j] > blx - ro:
            rx[i][j] = blx - ro
            vx[i][j] = -vx[i][j]
    return rx, vx

def Wall_y(ry, vy, ro, i):
    for j in range(q):
        if ry[i][j] < ro:
            ry[i][j] = ro
            vy[i][j] = -vy[i][j]
        elif ry[i][j] > bly - ro:
            ry[i][j] = bly - ro
            vy[i][j] = -vy[i][j]
    return ry, vy

def Wall_z(rz, vz, ro, i):
    for j in range(q):
        if rz[i][j] < ro:
            rz[i][j] = ro
            vz[i][j] = -vz[i][j]
        elif rz[i][j] > blz - ro:
            rz[i][j] = blz - ro
            vz[i][j] = -vz[i][j]
    return rz, vz

def Vectors(r1, r2):
    dr = r2 - r1
    r = np.linalg.norm(dr)
    n = dr/r
    return n, r

def Lennard_Jones(r, n, m):
    lj = 4*((1/r)**12 - (1/r)**6)
    f = 24*r**(-7) - 48*r**(-13)
    a = n*f/m
    return lj, a

def Update(rx, ry, rz, vx, vy, vz, acc1, dt, i, l, sl):
    for g in range(q):
        r = np.array([[rx[i][g],ry[i][g],rz[i][g]]])
        v = np.array([[vx[i][g],vy[i][g],vz[i][g]]])
        a = acc1[g]

        v = v + a*dt
        r = r + v*dt
        v = Speed_Limit(v, sl)

        if i < l-1:
            rx[i+1][g] = r[0][0]
            ry[i+1][g] = r[0][1]
            rz[i+1][g] = r[0][2]
            vx[i+1][g] = v[0][0]
            vy[i+1][g] = v[0][1]
            vz[i+1][g] = v[0][2]
    return rx, ry, rz, vx, vy, vz

def Speed_Limit(v, sl):
    mag = np.linalg.norm(v)
    if mag > sl:
        v = v*sl/mag
    return v

def Kinetic_Energy(vx, vy, vz, m, i, w3):
    v = np.array((vx[i][w3], vy[i][w3], vz[i][w3]))
    mag = np.linalg.norm(v)
    k = 0.5*m*mag**2
    return k

def Radial_Density(bb, nbin, bs, q, blx):
    ppr = bb[nbin]/(q*4*(np.pi)*bs*(nbin*bs)**2)
    gr = (blx**3)*ppr/q
    return gr
