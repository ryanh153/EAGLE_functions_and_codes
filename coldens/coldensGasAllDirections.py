### Getting output and some plots of the snapshot030 galaxy
### Author: Ryan Horton 
### Created 4/25/2016


#  allocating memory for 1918053300 tree nodes
# MaxNodes = 1918053300
# failed to allocate memory for 1918053300 tree-nodes (87801.5 MB).

### Imports

import EagleFunctions
import coldens
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as plt3d
import sys
import h5py
import glob
import glob

### User Inputs

snap_directory = sys.argv[1] # directory of all the snap files to be analyzed

### Constants (all your data is in cgs!!!)

parsec = 3.0857e18 # cm
G = 6.674e-8
M_sol = 1.98855e33 # g

### Code

snap_files = EagleFunctions.get_snap_files(snap_directory)

particles_included_keyword = "snap_noneq_030_z000p205"
group_included_keyword = "group_tab_030_z000p205"
subfind_included_keyword = "eagle_subfind_tab_030_z000p205"

#### Extracting attributes

box_size = EagleFunctions.read_attribute(snap_files,'Header','BoxSize',include_file_keyword=particles_included_keyword)
print 'box size is'
print box_size
print ''
print ''
expansion_factor = EagleFunctions.read_attribute(snap_files,'Header','ExpansionFactor',include_file_keyword=particles_included_keyword)
hubble_param = EagleFunctions.read_attribute(snap_files,'Header','HubbleParam',include_file_keyword=particles_included_keyword)

### Extracting simulations parameter arrays

### Extracting group/FOF data

gal_coords = EagleFunctions.read_array(snap_files,"FOF/CentreOfMass",include_file_keyword=group_included_keyword)[0] # zero is to get most massives halo
gal_center = np.asarray([0.0,0.0,0.0]) # I transform all coordinates into gal centered coordinates
gal_velocity = EagleFunctions.read_array(snap_files,"FOF/Velocity",include_file_keyword=group_included_keyword)[0]
gal_M200 = EagleFunctions.read_array(snap_files,"FOF/Group_M_Crit200",include_file_keyword=subfind_included_keyword)[0]
gal_R200 = EagleFunctions.read_array(snap_files,"FOF/Group_R_Crit200",include_file_keyword=subfind_included_keyword)[0]

### Gas (particle type 0)

gas_coords = EagleFunctions.read_array(snap_files,'PartType0/Coordinates',include_file_keyword=particles_included_keyword)
gas_coords = EagleFunctions.gal_centered_coords(gas_coords,gal_coords,box_size,expansion_factor,hubble_param)
gas_distance = np.sqrt(gas_coords[:,0]**2.0+gas_coords[:,1]**2.0+gas_coords[:,2]**2.0)

gas_mass = EagleFunctions.read_array(snap_files,'PartType0/Mass',include_file_keyword=particles_included_keyword)
gasSmoothingLength = EagleFunctions.read_array(snap_files,'PartType0/SmoothingLength',include_file_keyword=particles_included_keyword)

H_m = 1.0 # mass of hydrogen relative to mass of hygrogen. means not corrected because gas is almost entirely H

### Radius looking at
R = 1.0*gal_R200

### call coldens
res = 500 # resoltuion of coldense output (pixels in each direction)
SPH_neighbors = 58  # Ben uses 58
theta = 0.0
psi = 0.0

L_z = 2.0e0*R/(1.e6*parsec)

result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), gas_mass/M_sol, H_m, gal_center, 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), L_z, res, res, SPH_neighbors, box_size, fig_name = "GasColdens_Rvir_zhat", theta=theta,psi=psi)

theta = 90
result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), gas_mass/M_sol, H_m, gal_center, 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), L_z, res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "GasColdens_Rvir_yhat", theta=theta,psi=psi)

psi = 90
result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), gas_mass/M_sol, H_m, gal_center, 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), L_z, res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "GasColdens_Rvir_xhat", theta=theta,psi=psi)









