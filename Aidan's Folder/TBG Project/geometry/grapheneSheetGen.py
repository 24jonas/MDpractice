import numpy as np

def generate_graphene_sheet(a = 2.46, nx = 4, ny = 4, z = 0.0, vacuum = 24.0):
    
    # Lattice Vectors
    a1 = a * np.array([1.0, 0.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3)/2, 0.0])
    a3 = np.array([0.0, 0.0, vacuum])
    A  = np.column_stack([a1, a2, a3])

    # Scaling distance between A and nearest B atom to sqrt(3).
    basis_frac = np.array([[0.0, 0.0, 0.0],
                           [1/3, 1/3, 0.0]], 
                           dtype=float)
    basis_cart = (A @ basis_frac.T).T  # (2,3)

    # Tile indices
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    IJ = np.stack([I.ravel(), J.ravel()], axis=1)  # (nx*ny, 2)

    # Cell translations (broadcast)
    T = IJ[:, [0]] * a1 + IJ[:, [1]] * a2          # (nx*ny, 3)

    # Add basis and reshape
    R = (T[:, None, :] + basis_cart[None, :, :]).reshape(-1, 3)

    # Supercell matrix (columns)
    supercell = np.column_stack([nx * a1, ny * a2, a3])

    return R, supercell