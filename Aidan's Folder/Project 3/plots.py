import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from init import sample_rate
from matplotlib.animation import FuncAnimation
# Plot trajectories
def plot_trajectories(R_sto, border, dim_check, R_arr):
    if(dim_check == '2'):
        plt.subplot(1, 3, 1)
        t = np.arange(len(R_sto)) * sample_rate  # steps, not time
        plt.plot(t, R_sto[:, 0, 0], label = "Particle 1")
        plt.title("Particle trajectories")
        plt.legend()
    else:
        fig = plt.figure(figsize = (12, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection = '3d')
        ax2 = fig.add_subplot(1, 2, 2, projection = '3d')

        plot_lattice(ax1, R_arr, border)

        ax2.set_title("3D Trajectory: Particle 1")
        traj = R_sto[:, 0]
        ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], label = "Particle 1")
        ax2.scatter(*traj[0], color = 'green', label = 'Start', s = 40)
        ax2.scatter(*traj[-1], color = 'red', label = 'End', s = 40)
        x = [0, border[0]]
        y = [0, border[1]]
        z = [0, border[2]]
        for s, e in combinations(product(x, y, z), 2):
           if sum(np.abs(np.array(s) - np.array(e)) > 0) == 1:
               ax2.plot(*zip(s, e), color='black', linewidth=0.5)
        ax2.set_xlim(0, border[0])
        ax2.set_ylim(0, border[1])
        ax2.set_zlim(0, border[2])
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.legend()

        plt.tight_layout()
        plt.show()
         


# Plot velocity
def plot_vel(V_sto, dims):
    plt.subplot(1, 3, 2)
    t = np.arange(len(V_sto)) * sample_rate  # steps, not time
    
    for d in range(min(dims, 3)):
        plt.plot(t, V_sto[:, 0, d], label=f'V{d}')
    plt.title("Vel Graph")
    plt.legend()

# Plot Energies
def plot_energy(KE_sto, PE_sto, E_sto):
    plt.subplot(1,3,3)
    t = np.arange(len(KE_sto)) * sample_rate  # steps, not time
    plt.plot(t, KE_sto, label = 'Kinetic')
    plt.plot(t, PE_sto, label = 'Potential')
    plt.plot(t, E_sto, label = 'total')
    plt.title("Energies")
    plt.legend()

# Plot g(r)
def plot_gr(r_vals, g):
    plt.figure()
    plt.plot(r_vals, g)
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title("Pair Correlation Function")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot Lattice Setup
def plot_lattice(ax, R_arr, border):
    ax.set_title("3D Lattice Setup")

    ax.scatter(R_arr[:,0], R_arr[:,1], R_arr[:, 2], color = 'royalblue', s = 50)

    x = [0, border[0]]
    y = [0, border[1]]
    z = [0, border[2]]
    for s, e in combinations(np.array(list(product(x, y, z))), 2):
        if np.sum(np.abs(s-e) > 0) == 1:
            ax.plot(*zip(s, e), color='black', linewidth=1)

    ax.set_xlim(0, border[0])
    ax.set_ylim(0, border[1])
    ax.set_zlim(0, border[2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


# 2D Animation
def animate_2D(R_sto, border, dt, num_step):
    fig, ax = plt.subplots()
    if np.isscalar(border):
        ax.set_xlim(0, border)
        ax.set_ylim(0, border)
    else:
        ax.set_xlim(0, border[0])
        ax.set_ylim(0, border[1])   
    ax.set_aspect('equal')
    ax.set_title("2D Particle Animation")

    scatter = ax.scatter([], [], s = 50, color = 'royalblue')

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return scatter,
    
    def update(frame):
               pos = R_sto[frame]
               scatter.set_offsets(pos)
               return scatter,

    sim_time = num_step * dt
    num_frames = len(R_sto)
    interval = (sim_time * 1000) / num_frames
    anim = FuncAnimation(fig, update, frames = len(R_sto), init_func = init, blit = True, interval = interval)

    plt.show()

def plot_master(R_sto, V_sto, KE_sto, PE_sto, E_sto, border, r_vals, g, dt, num_step, dim_check, R_arr):
    if dim_check == '2':
        plt.figure(figsize=(18, 5))
        plot_trajectories(R_sto, border, dim_check, R_arr)
        plot_vel(V_sto, R_sto.shape[2])
        plot_energy(KE_sto, PE_sto, E_sto)
        plt.tight_layout()
        plt.show()

        plot_gr(r_vals, g)
        animate_2D(R_sto, border, dt, num_step)    
    else:
        plot_trajectories(R_sto, border, dim_check, R_arr)
        plot_gr(r_vals, g)
        print("3D animation will go here.")