### Getting output and some plots of the snapshot030 galaxy
### Author: Ryan Horton 
### Created 4/25/2016

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
expansion_factor = EagleFunctions.read_attribute(snap_files,'Header','ExpansionFactor',include_file_keyword=particles_included_keyword)
hubble_param = EagleFunctions.read_attribute(snap_files,'Header','HubbleParam',include_file_keyword=particles_included_keyword)

### Extracting simulations parameter arrays

### Extracting group/FOF data

gal_coords = EagleFunctions.read_array(snap_files,"FOF/CentreOfMass",include_file_keyword=group_included_keyword)[0] # zero is to get most massives halo
gal_center = np.asarray([0.0,0.0,0.0]) # I transform all coordinates into gal centered coordinates
gal_velocity = EagleFunctions.read_array(snap_files,"FOF/Velocity",include_file_keyword=group_included_keyword)[0]
gal_M200 = EagleFunctions.read_array(snap_files,"FOF/Group_M_Crit200",include_file_keyword=subfind_included_keyword)[0]
gal_R200 = EagleFunctions.read_array(snap_files,"FOF/Group_R_Crit200",include_file_keyword=subfind_included_keyword)[0]

### Stars (particle type 4)

star_coords = EagleFunctions.read_array(snap_files,'PartType4/Coordinates',include_file_keyword=particles_included_keyword)
star_coords = EagleFunctions.gal_centered_coords(star_coords,gal_coords,box_size,expansion_factor,hubble_param)

star_Z = EagleFunctions.read_array(snap_files,'PartType4/Metallicity',include_file_keyword=particles_included_keyword)
log_star_Z = np.log10(star_Z)

### Gas (particle type 0)

gas_coords = EagleFunctions.read_array(snap_files,'PartType0/Coordinates',include_file_keyword=particles_included_keyword)
gas_coords = EagleFunctions.gal_centered_coords(gas_coords,gal_coords,box_size,expansion_factor,hubble_param)
gas_distance = np.sqrt(gas_coords[:,0]**2.0+gas_coords[:,1]**2.0+gas_coords[:,2]**2.0)

gas_mass = EagleFunctions.read_array(snap_files,'PartType0/Mass',include_file_keyword=particles_included_keyword)
gasSmoothingLength = EagleFunctions.read_array(snap_files,'PartType0/SmoothingLength',include_file_keyword=particles_included_keyword)

H_mass = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Hydrogen',include_file_keyword=particles_included_keyword)*gas_mass
He_mass = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Helium',include_file_keyword=particles_included_keyword)*gas_mass
C_mass = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Carbon',include_file_keyword=particles_included_keyword)*gas_mass
Fe_mass = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Iron',include_file_keyword=particles_included_keyword)*gas_mass
Mg_mass = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Magnesium',include_file_keyword=particles_included_keyword)*gas_mass
Ne_mass = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Neon',include_file_keyword=particles_included_keyword)*gas_mass
N_mass = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Nitrogen',include_file_keyword=particles_included_keyword)*gas_mass
O_mass = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Oxygen',include_file_keyword=particles_included_keyword)*gas_mass
Si_mass = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Silicon',include_file_keyword=particles_included_keyword)*gas_mass

### Radius looking at
R = 1.0*gal_R200

### compare with your own plot
gas_coords_in_R = EagleFunctions.particles_within_R(gas_coords, gas_coords,R)
gas_mass_in_R = EagleFunctions.particles_within_R(gas_mass,gas_coords,R)
H_mass_in_R = EagleFunctions.particles_within_R(H_mass,gas_coords,R)
He_mass_in_R = EagleFunctions.particles_within_R(He_mass,gas_coords,R)
C_mass_in_R = EagleFunctions.particles_within_R(C_mass,gas_coords,R)
Fe_mass_in_R = EagleFunctions.particles_within_R(Fe_mass,gas_coords,R)
Mg_mass_in_R = EagleFunctions.particles_within_R(Mg_mass,gas_coords,R)
Ne_mass_in_R = EagleFunctions.particles_within_R(Ne_mass,gas_coords,R)
N_mass_in_R = EagleFunctions.particles_within_R(N_mass,gas_coords,R)
O_mass_in_R = EagleFunctions.particles_within_R(O_mass,gas_coords,R)
Si_mass_in_R = EagleFunctions.particles_within_R(Si_mass,gas_coords,R)

### Masses of elements
H_m = 1.0
He_m = 4.0
C_m = 12.0
Fe_m = 56.0
Mg_m = 24.0
Ne_m = 20.0
N_m = 14.0
O_m = 16.0
Si_m = 28.0

### call coldens
res = 3 # resoltuion of coldense output (pixels in each direction)
SPH_neighbors = 1  # Ben uses 58
theta = 0.0
psi = 0.0

result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), gas_mass/M_sol, H_m, gal_center, 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "GasColumnDensity_R=R_vir", theta=theta,psi=psi)
result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), H_mass/M_sol, H_m, gal_center, 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "HColumnDensity_R=R_vir", theta=theta,psi=psi)
result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), He_mass/M_sol, He_m, gal_center, 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "HeColumnDensity_R=R_vir", theta=theta,psi=psi)
result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), C_mass/M_sol, C_m, gal_center, 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "CColumnDensity_R=R_vir", theta=theta,psi=psi)
result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), Fe_mass/M_sol, Fe_m, gal_center, 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "FeColumnDensity_R=R_vir", theta=theta,psi=psi)
result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), Mg_mass/M_sol, Mg_m, gal_center, 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "MgColumnDensity_R=R_vir", theta=theta,psi=psi)
result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), Ne_mass/M_sol, Ne_m, gal_center, 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "NeColumnDensity_R=R_vir", theta=theta,psi=psi)
result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), N_mass/M_sol, N_m, gal_center, 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "NColumnDensity_R=R_vir", theta=theta,psi=psi)
result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), O_mass/M_sol, O_m, gal_center, 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "OColumnDensity_R=R_vir", theta=theta,psi=psi)
result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), Si_mass/M_sol, Si_m, gal_center, 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "SiColumnDensity_R=R_vir", theta=theta,psi=psi)




