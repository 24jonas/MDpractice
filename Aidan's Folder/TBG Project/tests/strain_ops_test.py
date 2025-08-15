
from strain.strain_ops import apply_strain
import numpy as np
from geometry import generate_graphene_sheet

def test_biaxial_strain_area_scaling():
    positions, cell = generate_graphene_sheet(a=2.46, nx=5, ny=5)

    eps = 0.005
    pos_s, cell_s = apply_strain(positions, cell, eps_x=eps, eps_y=eps)

    # Compute areas
    area0 = np.linalg.norm(np.cross(cell[:, 0], cell[:, 1]))
    area1 = np.linalg.norm(np.cross(cell_s[:, 0], cell_s[:, 1]))

    # Expected area scaling factor
    print(f"Area scale: {area1/area0:.6f} (expected ≈ {(1+eps)*(1+eps):.6f})")