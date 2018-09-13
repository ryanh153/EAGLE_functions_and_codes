import EaglePlottingFunctions
import sys

snap_directory = sys.argv[1] # directory of all the snap files to be analyzed
R_in_vir = sys.argv[2] # radius in virial radii
gal_num = sys.argv[3]

particles_included_keyword = "snap_noneq_030_z000p205"
group_included_keyword = "group_tab_030_z000p205"
subfind_included_keyword = "eagle_subfind_tab_030_z000p205"

# output_file = open('data.txt', 'w')
# output_file.write("snap_directory FOF_index radius gal_mass virial_radius gal_v_x gal_v_y gal_v_z gal_speed gas_beta_DM_beta\n")
# output_file.write("%s %d %.2e " % (snap_directory, gal_num, R))

### Plotting functions

gal_mass, virial_radius, gal_v_x, gal_v_y, gal_v_z, gal_speed = \
EaglePlottingFunctions.velocity_dispersions(snap_directory,R_in_vir,gal_num,particles_included_keyword,group_included_keyword,subfind_included_keyword)

# gal_mass, virial_radius, gal_v_x, gal_v_y, gal_v_z, gal_speed = \
# EaglePlottingFunctions.spherical_velocity(snap_directory,R,gal_num,particles_included_keyword,group_included_keyword,subfind_included_keyword)

# gal_mass, virial_radius, gal_v_x, gal_v_y, gal_v_z, gal_speed = \
# EaglePlottingFunctions.force_balance_and_virial(snap_directory,R,gal_num,particles_included_keyword,group_included_keyword,subfind_included_keyword)

# output_file.write("%.2e %.2e %.2e %.2e %.2e %.2e " % (gal_mass, virial_radius, gal_v_x, gal_v_y, gal_v_z, gal_speed))

### Obtain values/data functions

gas_beta, DM_beta = \
EaglePlottingFunctions.velocity_anisotropy(snap_directory,R,gal_num,particles_included_keyword,group_included_keyword,subfind_included_keyword)

# output_file.write("%.2e %.2e\n" % (gas_beta, DM_beta))

EagleFunctions.add_gal_data(True, snap_directory, gal_num, R_in_vir, gal_mass, virial_radius, gal_v_x, gal_v_y, gal_v_z, gal_speed,
	                                   gas_sigma_radial, gas_sigma_theta, gas_sigma_phi, gas_sigma_tan, gas_beta,
	                                   DM_sigma_radial, DM_sigma_theta, DM_sigma_phi, DM_sigma_tan, DM_beta)