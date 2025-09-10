# This example creates a 9.43 degree twisted system by first computing the proper p,q, then using these as inputs to make_graphene. All of the 
# properties from the shifted case which can be set here also allow the same variety of input formats (scalar, list, etc.) as above.

import ase
from ase.visualize import view
import flatgraphene as fg

p_found, q_found, theta_comp = fg.twist.find_p_q(21.79)
atoms=fg.twist.make_graphene(cell_type='hex',n_layer=2,
                             p=p_found,q=q_found,lat_con=0.0,a_nn=1.5,
                             sep=3.35,h_vac=3)
ase.visualize.view(atoms)
