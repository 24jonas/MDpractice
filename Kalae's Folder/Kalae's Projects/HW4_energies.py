import numpy as np
import matplotlib.pyplot as plt



# Parameters
N = 27
SX = SY = SZ = 6.0
dt = 0.005
timesteps = 15000
save_every = 10  # only save every 10 steps to reduce plotting data

# Lennard-Jones parameters (reduced units: epsilon = sigma = 1)
def compute_forces(positions, box_size):
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rij = positions[i] - positions[j]
            rij -= box_size * np.round(rij / box_size)  # periodic BCs
            r2 = np.dot(rij, rij)
            if r2 < 3.0**2:  # cutoff
                inv_r6 = 1.0 / r2**3
                inv_r12 = inv_r6**2
                f = 48 * inv_r12 - 24 * inv_r6
                force = f * rij / r2
                forces[i] += force
                forces[j] -= force
                potential_energy += 4 * (inv_r12 - inv_r6)
    return forces, potential_energy

# Initial positions in 3x3x3 checkerboard cube
def initialize_positions():
    pos = []
    spacing = SX / 3.0
    for x in range(3):
        for y in range(3):
            for z in range(3):
                offset = ((x + y + z) % 2) * 0.2  # stagger
                pos.append([x * spacing + offset,
                            y * spacing + offset,
                            z * spacing + offset])
    return np.array(pos)

# Initial velocities (Maxwell-Boltzmann-like, zero net momentum)
def initialize_velocities():
    velocities = np.random.randn(N, 3)
    velocities -= np.mean(velocities, axis=0)  # zero total momentum
    return velocities

# Velocity Verlet integration
positions = initialize_positions()
velocities = initialize_velocities()
box_size = np.array([SX, SY, SZ])
forces, _ = compute_forces(positions, box_size)

kinetic_data = []
potential_data = []
total_data = []
kinetic_avg = []

cumulative_kinetic = 0

for step in range(timesteps):
    # First half step
    velocities += 0.5 * dt * forces
    positions += dt * velocities
    positions %= box_size  # periodic BCs

    # Compute new forces
    forces, potential = compute_forces(positions, box_size)

    # Second half step
    velocities += 0.5 * dt * forces

    # Energies
    kinetic = 0.5 * np.sum(velocities**2)
    total = kinetic + potential

    cumulative_kinetic += kinetic
    kinetic_avg_val = cumulative_kinetic / (step + 1)

    if step % save_every == 0:
        kinetic_data.append(kinetic)
        potential_data.append(potential)
        total_data.append(total)
        kinetic_avg.append(kinetic_avg_val)

# Plotting
time = np.arange(0, timesteps, save_every)

plt.figure(figsize=(10, 6))
plt.plot(time, total_data, label="Total energy", color='orchid')
plt.plot(time, kinetic_data, label="Kinetic energy", color='mediumseagreen')
plt.plot(time, potential_data, label="Potential energy", color='skyblue')
plt.plot(time, kinetic_avg, label="Avg. Kinetic energy", color='orange', linestyle='--')
plt.xlabel("Time step")
plt.ylabel("Energy")
plt.title("Energy vs Time in MD Simulation")
plt.legend()
plt.tight_layout()
plt.show()

"""

# Constants
N= 27
SX =6.0
SY = 6.0
SZ = 6.0
dt = 0.005
timesteps = 15000
save_every = 10  

"""