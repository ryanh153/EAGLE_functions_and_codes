import EagleFunctions
import numpy as np
import sys

snap_directory = sys.argv[1] # directory of all the snap files to be analyzed
group_number = int(sys.argv[2])
# what galaxy, and how much of it we want to look at 
R_in_vir = float(sys.argv[3]) # radius in virial radii
snapshot_file_end = sys.argv[4]

### Keywords to get correct files
particles_included_keyword = "snap_noneq_" + snapshot_file_end
group_included_keyword = "group_tab_" + snapshot_file_end
subfind_included_keyword = "eagle_subfind_tab_" + snapshot_file_end

# get basic properties of simulation and galaxy
print 'getting basic props\n'
box_size, expansion_factor, hubble_param, gal_coords, gal_velocity, gal_M200, gal_R200, radius, gal_speed, gal_stellar_mass, gal_SFR = \
EagleFunctions.get_basic_props(snap_directory, R_in_vir, group_number, particles_included_keyword, group_included_keyword, subfind_included_keyword)
# array of output radii
radii = np.linspace(0.0,R_in_vir,100*R_in_vir)*gal_R200
arr_size = np.size(radii)-1 # in data set creation for output this is how many data points are created in each data set

# get properties of gas in the volume we want to look at
print 'getting gas props\n'
gas_coords_in_R, gas_distance_in_R, gas_mass_in_R, gas_velocity_in_R, gas_speed_in_R = \
EagleFunctions.get_gas_props_for_spectra_runs(snap_directory, radius, group_number, particles_included_keyword, group_included_keyword, subfind_included_keyword,
box_size, expansion_factor, hubble_param, gal_coords, gal_velocity)

# get properties of DM in volume we want to look at
print 'getting DM props\n'
DM_coords_in_R, DM_distance_in_R, DM_mass_in_R, DM_velocity_in_R, DM_speed_in_R = \
EagleFunctions.get_DM_props(snap_directory, radius, group_number, particles_included_keyword, group_included_keyword, subfind_included_keyword,
box_size, expansion_factor, hubble_param, gal_coords, gal_velocity)

# Convert velocities from cartesian to spherical coordinates [radial, theta, phi]
print 'spherical stuff\n'
gas_velocity_sph_in_R = EagleFunctions.cartesian_to_spherical_velocity(gas_velocity_in_R,gas_coords_in_R)

DM_velocity_sph_in_R = EagleFunctions.cartesian_to_spherical_velocity(DM_velocity_in_R,DM_coords_in_R)

### get velocity anisotropies
print 'getting betas\n'
gas_sigma_radial, gas_sigma_theta, gas_sigma_phi, gas_sigma_tan, gas_beta, DM_sigma_radial, DM_sigma_theta, DM_sigma_phi, DM_sigma_tan, DM_beta = \
EagleFunctions.get_betas(gas_velocity_sph_in_R, gas_coords_in_R, DM_velocity_sph_in_R, DM_coords_in_R, radii, output_all = True)

### Add this galaxies data to output file
print 'making file\n'
file_name = "output_%s_%s_%d.hdf5" % ('mybox', particles_included_keyword, group_number) # name of file

# adds all relelvant data to file
EagleFunctions.create_gal_data(file_name, arr_size, snap_directory, particles_included_keyword, group_number, radii[1::], gal_M200, gal_stellar_mass, 
							gal_SFR, gal_R200, box_size, hubble_param, expansion_factor, gal_coords, gal_velocity, gas_sigma_radial, gas_sigma_theta, gas_sigma_phi, gas_beta, 
	                        DM_sigma_radial, DM_sigma_theta, DM_sigma_phi, DM_beta)



