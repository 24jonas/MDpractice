import numpy as np
from .Materials import primitive_cols, basis_cart
from .MathUtils import rotz, cartToFrac, fracToCart


def _normalize_K_orientation(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, int).reshape(2, 2)
    detK = int(round(np.linalg.det(K)))
    if detK < 0:
        K = K @ np.array([[-1, 0], [0, 1]], dtype=int)
    return K

def hnf_2x2_strict(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, int).reshape(2, 2)
    a, b, c, d = K.ravel()
    def egcd(x, y):
        if y == 0:
            return (abs(x), 1 if x > 0 else -1, 0)
        g, u, v = egcd(y, x % y)
        return g, v, u - (x // y) * v

    g, u, v = egcd(a, c)  # make lower-left zero
    U1 = np.array([[u, v], [-c // g, a // g]], dtype=int)
    K1 = U1 @ K 

    h11, h12, h22 = int(K1[0, 0]), int(K1[0, 1]), int(K1[1, 1])

    # force positive diagonals
    if h11 < 0:
        h11, h12 = -h11, -h12
    if h22 < 0:
        h22, h12 = -h22, -h12

    if h11 <= 0 or h22 <= 0:
        raise ValueError(f"HNF: invalid diagonals (h11={h11}, h22={h22})")

    # reduce h12 to [0, h22)
    h12m = ((h12 % h22) + h22) % h22
    return np.array([[h11, h12m], [0, h22]], dtype=int)

def enumerate_reps(H: np.ndarray) -> list[np.ndarray]:
    H = np.asarray(H, int)
    h11, h12, h22 = int(H[0, 0]), int(H[0, 1]), int(H[1, 1])
    reps = []
    for j in range(h22):
        off = (j * h12) % h11
        for i in range(h11):
            reps.append(np.array([(i + off) % h11, j], int))
    return reps
def build_stack_from_result_exact(result, materials, thetas_deg, *,
                                  d=3.35, vacuum=24.0, registry_fracs=None):
    if registry_fracs is None:
        registry_fracs = [(0.0, 0.0)] * len(materials)

# Supercell
    superlat = result.superlattice()
    v = superlat.vectors()
    c1 = np.asarray(v[0])[:2]; c2 = np.asarray(v[1])[:2]
    H_cols = np.column_stack([[c1[0], c1[1], 0.0],
                              [c2[0], c2[1], 0.0],
                              [0.0,   0.0,   float(vacuum)]])

    # Per-layer data from result
    thetas_added = list(result.thetas())                 # radians, len = n_added
    eps_added    = list(result.strain_tensors()) if hasattr(result, "strain_tensors") else []

    thetas_rad = [0.0] + thetas_added
    eps_full   = [np.zeros((2,2))] + [np.array(E, float) for E in eps_added]

    R_layers = []; species_layers = []; detKs = []
    detKs = []   # before the loop
    for idx, (mat, (u,v)) in enumerate(zip(materials, registry_fracs)):
        # Primitive lattice for material in 2D
        P = primitive_cols(mat.a, vacuum=vacuum)
        A2 = P[:2, :2]

        # Strain and rotation
        eps = eps_full[idx]
        A2s = (np.eye(2) + eps) @ A2                    # strained
        theta = thetas_rad[idx]
        Q2 = rotz(theta)[:2, :2]                        # rotation
        A2_eff = Q2 @ A2s                               # rotated + strained basis

    # Integer K for this layer from solver
        if idx == 0:
            # substrate
            K_raw = np.array(result.M(), int)
        else:
            # added layer
            K_raw = np.array(result.layer_Ms()[idx-1], int)

        # normalize orientation
        K = _normalize_K_orientation(K_raw)
        detK = int(round(np.linalg.det(K)))
        detKs.append(detK)
        # Enumerate translations once with hnf
        Hnf = hnf_2x2_strict(K)
        reps = enumerate_reps(Hnf)
        # Basis atoms under A2_eff (already rotated & strained)
        P_eff = P.copy(); P_eff[:2, :2] = A2_eff
        B = basis_cart(P_eff, mat.basis_frac)   # (nbasis, 3)

        # Registry shift in supercell basis
        a1g, a2g = H_cols[:,0], H_cols[:,1]
        shift = u * a1g + v * a2g

        atoms = []
        for r in reps:
            t2 = (A2_eff @ r).astype(float)
            t = np.array([t2[0], t2[1], 0.0])
            for b in B:
                x = b + t
                x[:2] += shift[:2]
                atoms.append(x)
        R = np.vstack(atoms)

        # Place layer height in fractional z so it sticks
        S = cartToFrac(R, H_cols)
        S[:, :2] -= np.floor(S[:, :2])     # wrap XY only (exact tiling; this is just hygiene)
        S[:, 2] = ((-0.5 * d) + idx * d) / float(vacuum)
        R = fracToCart(S, H_cols)

        species = list(mat.species) * (len(R) // len(mat.species))
        R_layers.append(R); species_layers.append(species)

    R_all = np.vstack(R_layers)
    species_all = sum(species_layers, [])

    meta = {
        "thetas_deg": [float(t)*180.0/np.pi for t in thetas_rad],
        "detK_per_layer": detKs,
        "n_atoms": len(R_all),
        "cell": H_cols.copy(),
    }
    return R_all, H_cols, species_all, meta