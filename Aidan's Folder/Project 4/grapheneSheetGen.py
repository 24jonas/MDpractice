import numpy as np

def generate_graphene_sheet(a = 2.46, nx = 4, ny = 4):
    
    # Lattice Vectors
    a1 = np.array([a, 0], dtype = float)
    a2 = np.array([0.5 * a, (np.sqrt(3) / 2) * a], dtype = float)    # a2 rotated 60 degrees from a1

    # Basis definition- Atom A is origin, so B is at offset
    ##
    B_offset = (1/3) * a1 + (1/3) * a2 

    # Making positions array by generating in order of A, B, A, B, ... etc.
    positions = []
    for i in range(nx):
        for j in range(ny):
            R = i * a1 + j * a2
            positions.append(R)
            positions.append(R + B_offset)
    positions = np.array(positions)

    # Final Vectors
    supercell = np.array([
        nx * a1,
        ny * a2
    ])

    return positions, supercell
