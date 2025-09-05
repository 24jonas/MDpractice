import numpy as np
from .MathUtils import rotz, fracToCart, cartToFrac
from .Geom import wrapFrac, dupeCheck
from .Materials import Material, primitive_cols, basis_cart, MATLIB

def commensurate_cell(
        a=2.46,
        m: int = 1, 
        n: int = 2, 
        vacuum=20.0, 
        return_meta=False
        ) -> np.ndarray | tuple[np.ndarray, dict]:
    # Construct the moire cell for (m,n) using primitives a1 & a2
    
    a1 = a * np.array([1.0, 0.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3)/2, 0.0])

    # Integer matrix whose columns are the coeffs of t1,t2 in {a1,a2}
    #   t1 = n a1 + m a2
    #   t2 = -m a1 + (n+m) a2
    M = np.array([[n,  m],
                  [-m, n+m]], 
                  dtype=int)

    # Build moire cell
    T = np.column_stack([a1, a2]) @ M  # 3x2
    t1, t2 = T[:, 0], T[:, 1]
    H = np.column_stack([t1, t2, [0.0, 0.0, vacuum]])

    if not return_meta:
        return H

    # S is determinant of M; 2S gives # atoms per layer for choice of (n, m)
    S = n*n + m*n + m*m
    theta = np.degrees(np.arccos((n*n + 4*m*n + m*m) / (2.0*S)))
    meta = {
        "S": S,
        "theta_deg": theta,
        "L": np.linalg.norm(t1),
        "area": np.linalg.norm(np.cross(t1, t2)),
        "M": M,
    }
    return H, meta

def build_tbg_commensurate(
        m: int,
        n: int,
        material: str | Material = "graphene",
        d: float = 3.35,
        vacuum: float = 20.0,
        registry: str = "AB",
        theta_deg: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:

    mat = MATLIB[material] if isinstance(material, str) else material

    # Moiré cell + meta
    H, meta = commensurate_cell(a=mat.a, m=m, n=n, vacuum=vacuum, return_meta=True)
    S = int(meta["S"])
    M = meta["M"]

    if theta_deg is None:
        theta_deg = float(meta["theta_deg"])

    # Primitive and basis
    a1 = mat.a * np.array([1.0, 0.0, 0.0])
    a2 = mat.a * np.array([0.5, np.sqrt(3)/2, 0.0])
    B_cart = basis_cart(primitive_cols(mat.a, vacuum), mat.basis_frac)

    # Tile indices
    corners = np.array([[0,0],[S,0],[0,S],[S,S]], dtype=int)
    IJ_corners = corners @ np.linalg.inv(M).T
    i_min = int(np.floor(IJ_corners[:,0].min())) - 2
    i_max = int(np.ceil( IJ_corners[:,0].max())) + 2
    j_min = int(np.floor(IJ_corners[:,1].min())) - 2
    j_max = int(np.ceil( IJ_corners[:,1].max())) + 2

    ij_list = []
    for i in range(i_min, i_max+1):
        for j in range(j_min, j_max+1):
            u = (n+m)*i - m*j
            v = m*i + n*j
            if 0 <= u < S and 0 <= v < S:
                ij_list.append((i,j))
    ij = np.array(ij_list, dtype=int)
    assert len(ij) == S, f"Expected {S} tiles, got {len(ij)}"

    # Bottom layer
    T_prim = ij[:,[0]]*a1 + ij[:,[1]]*a2
    R_bot = (T_prim[:,None,:] + B_cart[None,:,:]).reshape(-1,3)
    R_bot = fracToCart(wrapFrac(cartToFrac(R_bot,H), wrap_z=False), H)
    R_bot[:,2] = -0.5*d
    R_bot = dupeCheck(R_bot, tol=1e-8)

    # Top layer
    if registry.upper()=="AA":
        shift = np.zeros(3)
    elif registry.upper()=="AB":
        shift = (a1+a2)/3.0
    else:
        shift = np.zeros(3)

    Q = rotz(theta_deg)
    R_top_raw = (R_bot @ Q.T) + shift
    R_top = fracToCart(wrapFrac(cartToFrac(R_top_raw,H), wrap_z=False), H)
    R_top[:,2] = +0.5*d
    R_top = dupeCheck(R_top, tol=1e-8)

    # Stack + meta
    R_all = np.vstack([R_bot, R_top])
    meta_out = {
        "m": m, "n": n, "S": S,
        "theta_deg": float(theta_deg),
        "registry": registry.upper(),
        "material": mat.name,
        "n_bottom": len(R_bot),
        "n_top": len(R_top),
        "expected_per_layer": len(mat.basis_frac)*S,
        "expected_total": 2*len(mat.basis_frac)*S,
    }
    return R_all, H, meta_out