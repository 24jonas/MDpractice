# tbg_rebuild/commensurate/supercell_core_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import os


from tbg_rebuild.materials.catalog import get_material
from tbg_rebuild.materials.primitives import hex_cell_rows

@dataclass(frozen=True, slots=True)
class SupercellCorePlan:
    cell: np.ndarray                 # (3,3) row-stacked with vacuum on z
    theta_deg: float                 # chosen / evaluated twist
    max_strain: float
    strain_tensors: np.ndarray       # shape (n_layers, 2,2) (typically 1 layer above substrate)
    corrected: bool                  # max_strain > tol
    meta: Dict[str, Any]             # store M, layer_Ms, etc.

def _deg2rad(x: float) -> float:
    return float(x) * np.pi / 180.0

def _debug(msg: str) -> None:
    if os.environ.get("TBG_DEBUG", ""):
        print(msg)

def plan_with_supercell_core(spec) -> SupercellCorePlan:
    """
    Uses supercell-core to find a commensurate supercell for bilayer graphene-like systems.

    Assumptions for now (matches your meeting scope):
    - 2 layers total, same material_id (graphene on graphene)
    - layer0 is substrate, layer1 is twisted
    """
    try:
        import supercell_core as sc
    except Exception as e:
        raise RuntimeError(
            "supercell_core is required for periodicity.mode='supercell_core'. "
            "Install it in this env (and on HPRC)."
        ) from e

    if len(spec.layers) != 2:
        raise ValueError("supercell-core mode currently supports exactly 2 layers.")
    if spec.layers[0].material_id != spec.layers[1].material_id:
        raise ValueError("supercell-core mode currently assumes identical materials for both layers.")

    mat = get_material(spec.layers[0].material_id)

    # supercell-core uses 2D vectors; we build them in Å in the same convention as your project
    # Your hex primitives typically use:
    # a1=(a,0), a2=(a/2, sqrt(3)/2 a) or the rotated equivalent.
    # Use your canonical helper to avoid mismatch.
    cell_prim = hex_cell_rows(mat.a, cell_z=spec.geometry.vacuum)
    a1 = cell_prim[0, :2]
    a2 = cell_prim[1, :2]

    # Build supercell-core Lattice for graphene primitive.
    # Important: include both basis atoms for graphene if your material has nbasis=2.
    lat = sc.lattice().set_vectors(tuple(a1), tuple(a2))

    # Use the material basis (fractional uv) to populate the primitive in sc coords.
    # supercell-core expects positions in the cell; easiest is crystal coords (fractional) or angstrom.
    # We’ll pass Angstrom positions by converting uv -> cart in 2D.
    basis_uv = np.asarray(mat.basis_frac, float)  # (nbasis,2)
    basis_cart = basis_uv[:, 0, None]*np.array(a1)[None, :] + basis_uv[:, 1, None]*np.array(a2)[None, :]
    for i, sym in enumerate(mat.basis_species):
        x, y = basis_cart[i]
        lat.add_atom(sym, (float(x), float(y), 0.0))

    h = sc.heterostructure().set_substrate(lat).add_layer(lat)

    s = dict(spec.periodicity.solver or {})
    method = s.get("method", "opt")
    theta_deg = float(s["theta_deg"])
    theta = _deg2rad(theta_deg)
    max_el = int(s.get("max_el", 20))

    # Theta scan around requested angle (this is the “almost commensurate” → “commensurate” search)
    w = float(s.get("theta_window_deg", 0.2))
    step = float(s.get("theta_step_deg", 0.01))
    thetas = np.arange(theta - _deg2rad(w), theta + _deg2rad(w) + 1e-15, _deg2rad(step))

    if method == "opt":
        res = h.opt(max_el=max_el, thetas=[thetas])
    elif method == "calc":
        M = s.get("M")
        if M is None:
            raise ValueError("solver.method='calc' requires solver['M'] to be set.")
        res = h.calc(M=M, thetas=[theta])
    else:
        raise ValueError(f"Unknown solver.method={method!r}")

# Extract commensurate in-plane cell vectors from superlattice
    superlat = res.superlattice()
    v = superlat.vectors()

    c1 = np.asarray(v[0], dtype=float).ravel()
    c2 = np.asarray(v[1], dtype=float).ravel()

# supercell-core may return (x,y) or (x,y,z). We want in-plane only.
    if c1.size < 2 or c2.size < 2:
        raise RuntimeError(f"supercell-core returned invalid vectors: {v}")

    c1xy = c1[:2]
    c2xy = c2[:2]

    _debug(f"[supercell-core] superlattice vectors: {c1} {c2}")

    cell = np.zeros((3, 3), dtype=float)
    cell[0, :2] = c1xy
    cell[1, :2] = c2xy
    cell[2, 2] = float(spec.geometry.vacuum)
    _debug(f"[supercell-core] superlattice vectors: {c1} {c2}")


    max_strain = float(res.max_strain())
    strain_tensors = np.asarray(res.strain_tensors(), float)  # per layer above substrate
    tol = float(s.get("strain_tol", 5e-4))
    corrected = bool(max_strain > tol)

    def _maybe_call(x):
        return x() if callable(x) else x

    M_obj = getattr(res, "M", None)
    layer_Ms_obj = getattr(res, "layer_Ms", None)

    M_val = _maybe_call(M_obj)
    layer_Ms_val = _maybe_call(layer_Ms_obj)

    meta = {
        "method": method,
        "theta_req_deg": theta_deg,
        "theta_grid_deg": [float(x * 180.0 / np.pi) for x in thetas],
        "max_el": max_el,
        "max_strain": max_strain,
        "M": M_val,
        "layer_Ms": layer_Ms_val,
}

    return SupercellCorePlan(
        cell=cell,
        theta_deg=theta_deg,
        max_strain=max_strain,
        strain_tensors=strain_tensors,
        corrected=corrected,
        meta=meta,
    )
