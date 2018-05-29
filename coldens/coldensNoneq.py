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
group_number = int(sys.argv[2])
# what galaxy, and how much of it we want to look at 
R_in_vir = float(sys.argv[2]) # radius in virial radii

### Constants (all your data is in cgs!!!)

parsec = 3.0857e18 # cm
G = 6.674e-8
M_sol = 1.98855e33 # g
H_atom_mass = 1.67372e-24 # grams

### Code

snap_files = EagleFunctions.get_snap_files(snap_directory)

particles_included_keyword = "snap_noneq_030_z000p205"
group_included_keyword = "group_tab_030_z000p205"
subfind_included_keyword = "eagle_subfind_tab_030_z000p205"

#### Extracting attributes

# get basic properties of simulation and galaxy
print 'getting basic props\n'
box_size, expansion_factor, hubble_param, gal_coords, gal_velocity, gal_M200, gal_R200, radius, gal_speed = \
EagleFunctions.get_basic_props(snap_directory, R_in_vir, group_number, particles_included_keyword, group_included_keyword, subfind_included_keyword)

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

chemAbundances = EagleFunctions.read_array(snap_files,'PartType0/ChemistryAbundances',include_file_keyword=particles_included_keyword)

H_mass = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Hydrogen',include_file_keyword=particles_included_keyword)*gas_mass



h1 = chemAbundances[:,1]  # number densities of each relative to hydrogen number density
c4 = chemAbundances[:,10]
o6 = chemAbundances[:,28]
o7 = chemAbundances[:,29]
o8 = chemAbundances[:,30]

### Radius looking at
R =R_in_vir*gal_R200

### compare with your own plot
gas_coords_in_R = EagleFunctions.particles_within_R(gas_coords, gas_coords,R)
gas_mass_in_R = EagleFunctions.particles_within_R(gas_mass,gas_coords,R)


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

### Masses of each ion in each SPH particle
O6_mass = (O_m/H_m)*o6*H_mass

#O6_mass_in_R = EagleFunctions.particles_within_R(gas_coords,O6_mass,R)

### call coldens
res = 500 # resoltuion of coldense output (pixels in each direction)
SPH_neighbors = 58  # Ben uses 58
theta = 0.0
psi = 0.0
gal_center = np.asarray([0.0,0.0,0.0]) # I transform all coordinates into gal centered coordinates

result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), O6_mass/M_sol, O_m, gal_center, 2.0*R/(1.e6*parsec), \
2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "OVI_ColumnDensity_R=R_vir", theta=theta,psi=psi)

result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), gas_mass/M_sol, H_m, gal_center, 2.0*R/(1.e6*parsec), \
2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "GasColumnDensity_R=R_vir", theta=theta,psi=psi)

result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), H_mass/M_sol, H_m, gal_center, 2.0*R/(1.e6*parsec), \
2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "HColumnDensity_R=R_vir", theta=theta,psi=psi)

result = coldens.main(gas_coords/(1.e6*parsec), gasSmoothingLength/(1.e6*parsec), He_mass/M_sol, He_m, gal_center, 2.0*R/(1.e6*parsec), \
2.0*R/(1.e6*parsec), 2.0*R/(1.e6*parsec), res, res, SPH_neighbors, box_size/(1.e6*parsec), fig_name = "HeColumnDensity_R=R_vir", theta=theta,psi=psi)




