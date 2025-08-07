# This example creates an ABA trilayer graphene system with rectangular unit cell. Any stacking designation (A,B,SP) is defined relative to the 
# origin of the unit cell.

import ase
from ase.visualize import view
import flatgraphene as fg
#note the inputs are all given with variable name for clarity,
#  but this is not necessary for required inputs
#the nearest neighbor distance (in-plane) a_nn is optional, and
#  overrides the lat_con variable  meaning the value of lat_con is unused
atoms=fg.shift.make_graphene(stacking=['A','B','A'],cell_type='rect',n_1=3,n_2=3,
                             lat_con=0.0,n_layer=3,sep=2.0,a_nn=1.5,
                             sym='C',mass=12,h_vac=3.0)
ase.visualize.view(atoms)
