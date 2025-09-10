import supercell_core as sc
from .Materials import primitive_cols

def lattice_from_material(mat, vacuum=24.0):
    H_cols = primitive_cols(mat.a, vacuum=vacuum)           # col-stacked
    A2 = H_cols[:2, :2]                                      # 2D in-plane
    L = sc.lattice().set_vectors(A2[:,0].tolist(), A2[:,1].tolist())
    for sym, frac in zip(mat.species, mat.basis_frac):
        xy = (A2 @ frac[:2]).tolist()
        L.add_atom(sym, (xy[0], xy[1], 0.0))
    return L
