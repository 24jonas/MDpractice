# This example gives the same result as the above, but specifies the relevant properties per layer using lists (instead of with a single value 
# to be assumed for all layers). When lists are used, all must have length n_layer. Interlayer separation is relative to the layer below, with 
# sep[i] giving spacing between layer i and i+1. The last element last element defines the spacing between top layer and top of supercell box 
# (if no vacuum is added). See the documentation in the section below for more information on fine grained options.

import ase
from ase.visualize import view
import flatgraphene as fg
#the comments from the above example apply here as well
atoms=fg.shift.make_graphene(stacking=['A','B','A'],cell_type='rect',n_1=3,n_2=3,
                             lat_con=0.0,n_layer=3,sep=[2.0,2.0,2.0],a_nn=1.5,
                             sym=['C','C','C'],mass=[12,12,12],h_vac=3.0)
ase.visualize.view(atoms)
