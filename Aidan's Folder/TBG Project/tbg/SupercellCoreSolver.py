import supercell_core as sc
import numpy as np

def _as_thetas_iterables(thetas_deg):
    return [np.array([td * sc.DEGREE], dtype=float) for td in thetas_deg]

def supercellCoreSolve(substrate_L, layer_L_list, thetas_deg, *,
                        algorithm: str, max_el: int):
    h = sc.heterostructure().set_substrate(substrate_L)
    for L in layer_L_list:
        h = h.add_layer(L)
    thetas = _as_thetas_iterables(thetas_deg)
    return h.opt(algorithm=algorithm, max_el=max_el, thetas=thetas)