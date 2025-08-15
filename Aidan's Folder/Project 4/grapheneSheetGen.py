import numpy as np

def generate_graphene_sheet(a = 2.46, nx = 4, ny = 4, z = 0.0, vacuum_padding = 24):
    
    # Lattice Vectors
    a1 = np.array([a, 0.0, z], dtype = float)
    a2 = np.array([0.5 * a, (np.sqrt(3) / 2) * a, z], dtype = float)    # a2 rotated 60 degrees from a1

    # Scaling distance between A and nearest B atom to sqrt(3).
    B_offset = (1/3) * (a1 + a2) 

    # Making positions array by generating in order of A, B, A, B, ... etc.
    positions = []
    for i in range(nx):
        for j in range(ny):
            R = i * a1 + j * a2
            positions.append([R[0], R[1], z])
            positions.append([R[0] + B_offset[0], 
                             R[1] + B_offset[1], 
                             z
                             ])
    positions = np.array(positions, dtype = float)

    # Final Vectors
    supercell = np.array([
        nx * a1,
        ny * a2, 
        [0, 0, vacuum_padding] ## default z is around 10a to make sure any two layers are far away by default
    ], dtype = float)

    return positions, supercell
