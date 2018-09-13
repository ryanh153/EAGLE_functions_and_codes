### Imports

import EagleFunctions
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as plt3d
import sys
import h5py
import glob

snap_directory = sys.argv[1] # directory of all the snap files to be analyzed

snap_files = EagleFunctions.get_snap_files(snap_directory)

particles_included_keyword = "snap_noneq_030_z000p205"
group_included_keyword = "group_tab_030_z000p205"
subfind_included_keyword = "eagle_subfind_tab_030_z000p205"

### Extracting attributes

box_size = EagleFunctions.read_attribute(snap_files,'Header','BoxSize',include_file_keyword=particles_included_keyword)
expansion_factor = EagleFunctions.read_attribute(snap_files,'Header','ExpansionFactor',include_file_keyword=particles_included_keyword)
hubble_param = EagleFunctions.read_attribute(snap_files,'Header','HubbleParam',include_file_keyword=particles_included_keyword)


### Extracting group/FOF data

gal_coords = EagleFunctions.read_array(snap_files,"FOF/CentreOfMass",include_file_keyword=group_included_keyword)[0] # zero is to get most massives halo
gal_velocity = EagleFunctions.read_array(snap_files,"FOF/Velocity",include_file_keyword=group_included_keyword)[0]
gal_M200 = EagleFunctions.read_array(snap_files,"FOF/Group_M_Crit200",include_file_keyword=subfind_included_keyword)[0]
gal_R200 = EagleFunctions.read_array(snap_files,"FOF/Group_R_Crit200",include_file_keyword=subfind_included_keyword)[0]

### Stars (particle type 4)

star_coords = EagleFunctions.read_array(snap_files,'PartType4/Coordinates',include_file_keyword=particles_included_keyword)
# gal_coords = np.median(star_coords,axis = 0) # can be used if gal_coords from FOF file is inaccurate
star_coords = EagleFunctions.gal_centered_coords(star_coords,gal_coords,box_size,expansion_factor,hubble_param)

star_Z = EagleFunctions.read_array(snap_files,'PartType4/Metallicity',include_file_keyword=particles_included_keyword)
log_star_Z = np.log10(star_Z)

### Gas (particle type 0)

gas_coords = EagleFunctions.read_array(snap_files,'PartType0/Coordinates',include_file_keyword=particles_included_keyword)
gas_coords = EagleFunctions.gal_centered_coords(gas_coords,gal_coords,box_size,expansion_factor,hubble_param)
gas_distance = np.sqrt(gas_coords[:,0]**2.0+gas_coords[:,1]**2.0+gas_coords[:,2]**2.0)

gas_mass = EagleFunctions.read_array(snap_files,'PartType0/Mass',include_file_keyword=particles_included_keyword)

gas_T = EagleFunctions.read_array(snap_files,'PartType0/Temperature',include_file_keyword=particles_included_keyword)
log_gas_T = np.log10(gas_T)

gas_density = EagleFunctions.read_array(snap_files,'/PartType0/Density',include_file_keyword=particles_included_keyword)
log_gas_density = np.log10(gas_density)

gas_Z = EagleFunctions.read_array(snap_files,'PartType0/Metallicity',include_file_keyword=particles_included_keyword)
log_gas_Z = np.log10(gas_Z)

gas_velocity = EagleFunctions.read_array(snap_files,"/PartType0/Velocity",include_file_keyword=particles_included_keyword)
gas_velocity = gas_velocity-gal_velocity
gas_speed = np.sqrt(gas_velocity[:,0]**2.0+gas_velocity[:,1]**2.0+gas_velocity[:,2]**2.0)

H_mass_frac = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Hydrogen',include_file_keyword=particles_included_keyword)
He_mass_frac = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Helium',include_file_keyword=particles_included_keyword)
C_mass_frac = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Carbon',include_file_keyword=particles_included_keyword)
Fe_mass_frac = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Iron',include_file_keyword=particles_included_keyword)
Mg_mass_frac = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Magnesium',include_file_keyword=particles_included_keyword)
Ne_mass_frac = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Neon',include_file_keyword=particles_included_keyword)
N_mass_frac = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Nitrogen',include_file_keyword=particles_included_keyword)
O_mass_frac = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Oxygen',include_file_keyword=particles_included_keyword)
Si_mass_frac = EagleFunctions.read_array(snap_files,'PartType0/ElementAbundance/Silicon',include_file_keyword=particles_included_keyword)

mass_frac_array = np.asarray([H_mass_frac,He_mass_frac,C_mass_frac,Fe_mass_frac,Mg_mass_frac,Ne_mass_frac,N_mass_frac,O_mass_frac,Si_mass_frac])

mu = np.dot(atom_mass_array[None,:],mass_frac_array).ravel()

gas_volume = gas_mass/gas_density
gas_number_of_particles = gas_mass/mu
gas_pressure = (gas_number_of_particles*kB*gas_T)/(gas_volume)

### Dark Matter (particle type 1)

DM_coords = EagleFunctions.read_array(snap_files,'PartType1/Coordinates',include_file_keyword=particles_included_keyword)
DM_coords = EagleFunctions.gal_centered_coords(DM_coords,gal_coords,box_size,expansion_factor,hubble_param)
DM_distance = np.sqrt(DM_coords[:,0]**2.0+DM_coords[:,1]**2.0+DM_coords[:,2]**2.0)

# special. All DM1 particles have same mass. Pull from header. units are 10^10/hubble_param in solar masses... no idea why
DM_mass = EagleFunctions.read_attribute(snap_files,'Header','MassTable',include_file_keyword=particles_included_keyword)[1]*(1.e10/hubble_param)*M_sol
DM_mass = np.zeros(np.size(DM_coords))+DM_mass

DM_velocity = EagleFunctions.read_array(snap_files,'PartType1/Velocity',include_file_keyword=particles_included_keyword)
DM_velocity = DM_velocity-gal_velocity
DM_speed = np.sqrt(DM_velocity[:,0]**2.0+DM_velocity[:,1]**2.0+DM_velocity[:,2]**2.0)
