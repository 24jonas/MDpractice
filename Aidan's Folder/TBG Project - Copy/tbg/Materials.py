import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Material:
    name: str
    a: float      
    basis_frac: np.ndarray
    species: tuple[str, ...]

def primitive_cols(a: float, vacuum: float = 24.0) -> np.ndarray:
    a1 = a * np.array([1.0, 0.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3)/2.0, 0.0])
    a3 = np.array([0.0, 0.0, vacuum])
    return np.column_stack([a1, a2, a3])

def basis_cart(H_cols: np.ndarray, basis_frac: np.ndarray) -> np.ndarray:
    H = np.asarray(H_cols, float)
    Bf = np.asarray(basis_frac, float)
    return (H @ Bf.T).T

GRAPHENE = Material(
    name="graphene",
    a=2.46,
    basis_frac=np.array([[0.0,     0.0, 0.0],
                         [1.0/3.0, 1.0/3.0, 0.0]]),
    species=("C", "C"),
)
HBN = Material(
    name="hbn",
    a=2.504,  
    basis_frac=np.array([[0.0,     0.0, 0.0],   # B at origin
                         [1.0/3.0, 1.0/3.0, 0.0]]),  # N
    species=("B", "N"),
)
MATLIB: dict[str, Material] = {
    "graphene": GRAPHENE,
    "hbn": HBN,
}
