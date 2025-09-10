import numpy as np
from .Materials import Material, primitive_cols, basis_cart
from .MathUtils import rotz, cartToFrac, fracToCart
from .Geom import dupeCheck

def _normalize_K_orientation(K: np.ndarray) -> np.ndarray:
    """Make det(K) positive by a unimodular flip; tiles the same supercell."""
    K = np.asarray(K, int).reshape(2, 2)
    if int(round(np.linalg.det(K))) < 0:
        # flip first column sign -> det changes sign
        K = K @ np.array([[-1, 0], [0, 1]], dtype=int)
    return K

def _get_layer_Ms_any(result) -> list[np.ndarray]:
    """Fetch per-layer integer matrices (2x2) across supercell-core versions."""
    for name in ("layer_Ms", "layer_integer_matrices", "layer_repetition_matrices", "Ms"):
        if hasattr(result, name):
            Ks = getattr(result, name)()
            return [np.array(K, int) for K in Ks]
    raise KeyError("No per-layer integer matrices on Result (API variant).")

def _get_thetas_rad_any(result) -> list[float]:
    """Fetch layer angles (radians) across supercell-core versions."""
    for name in ("thetas", "angles", "layer_thetas"):
        if hasattr(result, name):
            vals = getattr(result, name)()
            return [float(x) for x in np.atleast_1d(vals)]
    raise KeyError("No layer angles found on Result.")

def _get_strains_any(result) -> list[np.ndarray] | None:
    """Fetch optional small 2x2 strain tensors per layer (if present)."""
    for name in ("strain_tensors", "layer_strain_tensors", "strains"):
        if hasattr(result, name):
            E = getattr(result, name)()
            return [np.array(e, float)[:2, :2] for e in E]
    return None

def _egcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended GCD: returns (g,u,v) s.t. u*a + v*b = g = gcd(a,b) with g >= 0."""
    if b == 0:
        # choose u sign so g>=0; avoid (0,0,0)
        return (abs(a), 1 if a >= 0 else -1, 0)
    g, u, v = _egcd(b, a % b)
    return g, v, u - (a // b) * v

def hnf_2x2_strict(K: np.ndarray) -> np.ndarray:
    """Return Hermite Normal Form H (upper-triangular) of 2x2 integer matrix K.
       H = [[h11, h12], [0, h22]] with h11>0, h22>0, 0<=h12<h22.
       Raises if K is singular."""
    K = np.asarray(K, int).reshape(2, 2)
    a, b, c, d = K.ravel()

    # Make lower-left = 0 via left-multiplying by unimodular matrix U1
    g, u, v = _egcd(a, c)
    if g == 0:
        # a=c=0 -> singular
        raise ValueError(f"HNF: singular matrix K=\n{K}")
    U1 = np.array([[u, v], [-c // g, a // g]], dtype=int)
    K1 = U1 @ K  # [[h11, h12], [0, h22]] up to signs

    h11, h12, h22 = int(K1[0, 0]), int(K1[0, 1]), int(K1[1, 1])

    # force positive diagonals
    if h11 < 0:
        h11, h12 = -h11, -h12
    if h22 < 0:
        h22, h12 = -h22, -h12

    if h11 <= 0 or h22 <= 0:
        raise ValueError(f"HNF: invalid diagonals (h11={h11}, h22={h22}) for K=\n{K}")

    # reduce h12 into [0, h22)
    h12m = ((h12 % h22) + h22) % h22
    return np.array([[h11, h12m], [0, h22]], dtype=int)

def enumerate_reps(H: np.ndarray) -> list[np.ndarray]:
    """Enumerate representative integer translations r=(i,j) inside the HNF cell.
       Guarantees exactly det(H)=h11*h22 reps with consistent ordering."""
    H = np.asarray(H, int)
    h11, h12, h22 = int(H[0, 0]), int(H[0, 1]), int(H[1, 1])
    reps: list[np.ndarray] = []
    for j in range(h22):
        off = (j * h12) % h11
        for i in range(h11):
            reps.append(np.array([(i + off) % h11, j], int))
    return reps

def build_stack_from_result_exact(
    result,
    materials: list[Material],
    *,
    d: float = 3.35,
    vacuum: float = 24.0,
    # Registry shifts expressed in supercell fractional coords (u,v) per layer.
    # Default = no shift. If you prefer AB in the *layer* basis, convert before wrap.
    registry_fracs: list[tuple[float, float]] | None = None,
    use_solver_strain: bool = True,
    extra_random_strain: list[np.ndarray] | None = None,  # optional small 2x2 per layer
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    L = len(materials)
    if registry_fracs is None:
        registry_fracs = [(0.0, 0.0)] * L
    if extra_random_strain is not None:
        assert len(extra_random_strain) == L, "extra_random_strain must match #layers"

    # --- Common supercell from solver ---
    C2 = np.asarray(result.superlattice().vectors(), float)  # 2x2 in-plane
    H_cols = np.column_stack([
        [C2[0, 0], C2[0, 1], 0.0],
        [C2[1, 0], C2[1, 1], 0.0],
        [0.0,       0.0,     float(vacuum)]
    ])
    HinvT = np.linalg.inv(H_cols).T

    thetas_added = _get_thetas_rad_any(result)       # for non-substrate layer
    strains_added = _get_strains_any(result) if use_solver_strain else None
    Ks_layers = _get_layer_Ms_any(result)            # for non-substrate layer
    K_sub = np.array(result.M(), int)                # substrate K

    thetas_rad = [0.0] + thetas_added
    if strains_added is None:
        eps_full = [np.zeros((2, 2))] + [np.zeros((2, 2)) for _ in Ks_layers]
    else:
        eps_full = [np.zeros((2, 2))] + [np.array(E, float) for E in strains_added]
    Ks_all = [K_sub] + Ks_layers

    if not (len(Ks_all) == len(thetas_rad) == L):
        raise ValueError(
            f"Layer count mismatch: materials={L}, Ks={len(Ks_all)}, thetas={len(thetas_rad)}"
        )

    def _apply_extra(F: np.ndarray, idx: int) -> np.ndarray:
        if extra_random_strain is None:
            return F
        return (np.eye(2) + np.asarray(extra_random_strain[idx], float)) @ F

    R_layers: list[np.ndarray] = []
    species_layers: list[list[str]] = []
    detKs: list[int] = []
    n_per_layer: list[int] = []

    # center stack at z ~ 0
    z_center = 0.0
    z_levels = [z_center + (idx - (L - 1) / 2.0) * d for idx in range(L)]

    for idx, mat in enumerate(materials):
        # primitive
        P3 = primitive_cols(mat.a, vacuum=vacuum)  
        A2 = P3[:2, :2]                             

        # rotation + strain
        eps = eps_full[idx]
        F = (np.eye(2) + eps) @ A2
        F = _apply_extra(F, idx)                    
        Q2 = rotz(thetas_rad[idx])[:2, :2]
        A2_eff = Q2 @ F                            

        K_raw = np.array(Ks_all[idx], int)
        K = _normalize_K_orientation(K_raw)
        detK = int(round(np.linalg.det(K)))
        if detK <= 0:
            raise ValueError(f"Layer {idx}: singular/invalid det(K)={detK} for K=\n{K_raw}")
        detKs.append(detK)


        # HNF enumeration 
        Hnf = hnf_2x2_strict(K_raw)
        reps = enumerate_reps(Hnf)                  # len == detK

        P_eff = P3.copy()
        P_eff[:2, :2] = A2_eff
        B = basis_cart(P_eff, mat.basis_frac)    

        u, v = registry_fracs[idx]
        shift_cart = u * H_cols[:, 0] + v * H_cols[:, 1]

        atoms = []
        for r in reps:
            t2 = (A2_eff @ r).astype(float)
            t = np.array([t2[0], t2[1], 0.0])
            for b in B:
                x = b + t
                x[:2] += shift_cart[:2]
                atoms.append(x)
        R = np.asarray(atoms, float)

        # Wrap xy into the common cell and place z level
        S = cartToFrac(R, H_cols, HinvT=HinvT)
        S[:, :2] -= np.floor(S[:, :2])             
        S[:, 2] = z_levels[idx] / float(vacuum)
        R = fracToCart(S, H_cols)

        R = dupeCheck(R, tol=1e-8)

        # Species repeated to match atom count
        nb = len(mat.species)
        species = [mat.species[i % nb] for i in range(len(R))]
        n_per_layer.append(len(R))
        species_layers.append(species)
        R_layers.append(R)

    # Concatenate layers
    R_all = np.vstack(R_layers)
    species_all = sum(species_layers, [])

    meta = {
        "thetas_deg": [float(np.degrees(t)) for t in thetas_rad],
        "detK_per_layer": detKs,
        "n_atoms": int(len(R_all)),
        "n_per_layer": n_per_layer,
        "materials": [m.name for m in materials],
        "cell": H_cols.copy(),  # 3x3 column-stacked
        "vacuum": float(vacuum),
        "interlayer_d": float(d),
    }
    return R_all, H_cols, species_all, meta
