import EagleFunctions
import numpy as np
import sys

snap_directory = sys.argv[1] # directory of all the snap files to be analyzed
# what galaxy, and how much of it we want to look at 
R_in_vir = float(sys.argv[2]) # radius in virial radii
gal_num = int(sys.argv[3])

### Keywords to get correct files
particles_included_keyword = "snap_noneq_030_z000p205"
group_included_keyword = "group_tab_030_z000p205"
subfind_included_keyword = "eagle_subfind_tab_030_z000p205"

# output_file = EagleFunctions.make_output_file("output_%s_%s_%d.txt" % (snap_directory[0:-1], particles_included_keyword, gal_num))

# get basic properties of simulation and galaxy
box_size, expansion_factor, hubble_param, gal_coords, gal_velocity, gal_M200, gal_R200, radius, gal_speed = \
EagleFunctions.get_basic_props(snap_directory, R_in_vir, gal_num, particles_included_keyword, group_included_keyword, subfind_included_keyword)

# get properties of gas in the volume we want to look at
gas_coords_in_R, gas_distance_in_R, gas_mass_in_R, gas_density_in_R, gas_velocity_in_R, gas_speed_in_R, gas_T_in_R, mu_in_R, gas_volume_in_R, \
gas_number_of_particles_in_R, gas_pressure_in_R = \
EagleFunctions.get_gas_props(snap_directory, radius, gal_num, particles_included_keyword, group_included_keyword, subfind_included_keyword,
box_size, expansion_factor, hubble_param, gal_coords, gal_velocity)

# get properties of DM in volume we want to look at
DM_coords_in_R, DM_distance_in_R, DM_mass_in_R, DM_velocity_in_R, DM_speed_in_R = \
EagleFunctions.get_DM_props(snap_directory, radius, gal_num, particles_included_keyword, group_included_keyword, subfind_included_keyword,
box_size, expansion_factor, hubble_param, gal_coords, gal_velocity)

# Convert velocities from cartesian to spherical coordinates [radial, theta, phi]
gas_velocity_sph_in_R = EagleFunctions.cartesian_to_spherical_velocity(gas_velocity_in_R,gas_coords_in_R)

DM_velocity_sph_in_R = EagleFunctions.cartesian_to_spherical_velocity(DM_velocity_in_R,DM_coords_in_R)

### get velocity anisotropies

gas_sigma_radial, gas_sigma_theta, gas_sigma_phi, gas_sigma_tan, gas_beta, DM_sigma_radial, DM_sigma_theta, DM_sigma_phi, DM_sigma_tan, DM_beta = \
EagleFunctions.get_betas(gas_velocity_sph_in_R, DM_velocity_sph_in_R, output_all = True)

### tesing virial ratio outputs
radii = np.asarray([0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0])*gal_R200
gas_virial_LHS, gas_KE_in_radii, gas_T_KE, gas_virial_ratio, DM_virial_LHS, DM_KE_in_radii, DM_virial_ratio = \
EagleFunctions.get_virial_ratios(radii, gas_mass_in_R, DM_mass_in_R, gas_speed_in_R, gas_coords_in_R, gas_T_in_R, DM_speed_in_R, DM_coords_in_R, mu_in_R)

### add to file
file_name = "output_%s_%s_%d.hdf5" % ('mybox', particles_included_keyword, gal_num) # name of file

EagleFunctions.add_dataset_to_hdf5(file_name, 'gas_virial_LHS', gas_virial_LHS)
EagleFunctions.add_dataset_to_hdf5(file_name, 'gas_KE', gas_KE_in_radii)
EagleFunctions.add_dataset_to_hdf5(file_name, 'gas_T_KE', gas_T_KE)
EagleFunctions.add_dataset_to_hdf5(file_name, 'gas_virial_ratio', gas_virial_ratio)
EagleFunctions.add_dataset_to_hdf5(file_name, 'DM_virial_LHS', DM_virial_LHS)
EagleFunctions.add_dataset_to_hdf5(file_name, 'DM_KE', DM_KE_in_radii)
EagleFunctions.add_dataset_to_hdf5(file_name, 'DM_virial_ratio', DM_virial_ratio)

print "gas ratios are"
print gas_virial_ratio