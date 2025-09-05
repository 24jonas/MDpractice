import numpy as np
from .MathUtils import rotz, fracToCart, cartToFrac
from .Materials import Material, MATLIB, primitive_cols, basis_cart

def generateSheet(material: Material, nx: int = 4, ny: int = 4,
                  z: float = 0.0, vacuum: float = 24.0
                  ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    A = primitive_cols(material.a, vacuum)       # (3,3)
    a1, a2 = A[:, 0], A[:, 1]
    B_basis_cart = basis_cart(A, material.basis_frac)

    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    IJ = np.stack([I.ravel(), J.ravel()], axis=1)
    T = IJ[:, [0]] * a1 + IJ[:, [1]] * a2

    R = (T[:, None, :] + B_basis_cart[None, :, :]).reshape(-1, 3)
    R[:, 2] = z

    supercell = np.column_stack([nx * a1, ny * a2, A[:, 2]])
    species = list(material.species) * (len(R) // len(material.species))
    return R, supercell, species

def generateGrapheneSheet(a=2.46, nx=4, ny=4, z=0.0, vacuum=24.0):
    from .Materials import GRAPHENE
    if abs(a - GRAPHENE.a) > 1e-9:
        from dataclasses import replace
        mat = replace(GRAPHENE, a=float(a))
    else:
        mat = GRAPHENE
    R, H, _ = generateSheet(mat, nx=nx, ny=ny, z=z, vacuum=vacuum)
    return R, H

def countNearestNeighbors(
        positions: np.ndarray, 
        cell: np.ndarray, 
        cutoff = 1.6, 
        use_z = False
        ) -> np.ndarray:

    R = np.asarray(positions, dtype = float)
    H = np.asarray(cell, dtype = float)
    cutoff2 = float(cutoff) * float(cutoff)

    S = cartToFrac(R, H)

    # Using broadcasting to get fractional vectors
    dS = S[None, :, :] - S[:, None, :]

    if use_z:
        dS -= np.round(dS)
    else:
        dS[:, :, :2] -= np.round(dS[:, :, :2])      

    dR = fracToCart(dS, H)           

    # Product of i and j with themselves over k axis            
    d2 = np.einsum('ijk,ijk->ij', dR, dR)

    # Ignore distance to self
    np.fill_diagonal(d2, np.inf)

    return np.sum(d2 <= cutoff2, axis=1).astype(int)

def wrapFrac(S: np.ndarray, wrap_z = False) -> np.ndarray:
    S = np.asarray(S, float).copy()

    # Always wrap x and y
    S[..., :2] -= np.floor(S[..., :2])

    if wrap_z:
        S[..., 2] -= np.floor(S[..., 2])

    return S

def dupeCheck(R: np.ndarray, tol=1e-6):
# Remove duplicate atoms within a distance tolerance.
    R = np.asarray(R)
    keep = [0]
    for i in range(1, len(R)):
        d2 = np.sum((R[i] - R[keep])**2, axis=1)
        if np.min(d2) > tol**2:
            keep.append(i)
    return R[np.array(keep)]

def build_tbg(nx: int,
              ny: int,
              a=2.46,
              d=3.35,
              theta_deg=0.0,
              registry="AB",
              vacuum=20.0,
              pad=1.0,
              bottom_material: str = "graphene",
              top_material: str = "graphene"
              ) -> tuple[np.ndarray, np.ndarray, dict, list[str]]:

    from .Materials import MATLIB
    from .MathUtils import rotz, cartToFrac, fracToCart
    from .Geom import wrapFrac, dupeCheck
    import numpy as np

    # Materials
    mat_bot = MATLIB[bottom_material.lower()]
    mat_top = MATLIB[top_material.lower()]

    # Bottom layer: generator sets z for us
    R_bot, H, _ = generateSheet(mat_bot, nx=nx, ny=ny, z=-0.5 * d, vacuum=vacuum)

    # Oversized top layer before rotation
    nx_pad = max(nx, int(np.ceil(nx * pad)) + 2)
    ny_pad = max(ny, int(np.ceil(ny * pad)) + 2)
    R_top_big, _, _ = generateSheet(mat_top, nx=nx_pad, ny=ny_pad, z=+0.5 * d, vacuum=vacuum)

    # Registry shift
    a1 = mat_bot.a * np.array([1.0, 0.0, 0.0])
    a2 = mat_bot.a * np.array([0.5, np.sqrt(3)/2, 0.0])
    if registry.upper() == "AA":
        shift = np.zeros(3)
    elif registry.upper() == "AB":
        shift = (a1 + a2) / 3.0
    else:
        shift = np.zeros(3)

    # Rotate + shift top
    Q = rotz(theta_deg)
    R_top_rot = (R_top_big @ Q.T) + shift

    # Clip to target cell, wrap into [0,1)
    HinvT = np.linalg.inv(H).T
    S_top = cartToFrac(R_top_rot, H, HinvT=HinvT)
    eps = 1e-6
    mask = (
        (S_top[:, 0] >= -eps) & (S_top[:, 0] < 1.0 + eps) &
        (S_top[:, 1] >= -eps) & (S_top[:, 1] < 1.0 + eps)
    )
    S_top = wrapFrac(S_top[mask], wrap_z=False)

    # Convert back to Cartesian, place at +z
    R_top = fracToCart(S_top, H)
    R_top[:, 2] = +0.5 * d

    R_bot = dupeCheck(R_bot, tol=1e-6)
    R_top = dupeCheck(R_top, tol=1e-6)

    # Species AFTER dupeCheck so lengths match
    nb_bot = len(mat_bot.species)
    species_bot = [mat_bot.species[i % nb_bot] for i in range(len(R_bot))]
    nb_top = len(mat_top.species)
    species_top = [mat_top.species[i % nb_top] for i in range(len(R_top))]
    species_all = species_bot + species_top

    # Combine and meta
    R_all = np.vstack([R_bot, R_top])
    meta = {
        "n_bottom": len(R_bot),
        "n_top": len(R_top),
        "theta_deg": float(theta_deg),
        "registry": registry.upper(),
        "materials": {"bottom": mat_bot.name, "top": mat_top.name},
        "cell": H.copy(),
    }

    return R_all, H, meta, species_all