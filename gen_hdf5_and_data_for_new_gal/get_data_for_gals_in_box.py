import h5py 
import numpy as np
import SpecwizardFunctions
import EagleFunctions
import sys
import os
import subprocess
import glob
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import survey_realization_functions as cos_functions
import real_data

#constants
M_sol = 1.98855e33 # g
secs_per_year = 3.154e7 

#### Bool stuff? 

cos_smass_data, cos_ssfr_data, cos_radii_data, cos_h1_equ_widths, cos_h1_equ_widths_radii, cos_h1_cols, cos_h1_cols_radii, cos_si3_equ_widths, cos_si3_equ_widths_radii, cos_si3_cols, cos_si3_cols_radii, cos_o6_cols, cos_o6_cols_radii, cos_o6_equ_widths, cos_o6_equ_widths_radii, cos_AGN, which_survey = real_data.select_data_for_run(cos_gass=True, cos_gto=True, cos_halos_blue=True, cos_halos_red=False, cos_dwarfs=True, cos_AGN=True)

real_data_to_match = np.array([cos_smass_data, cos_ssfr_data])
tols = np.array([0.2, 0.3])
hi_lo_tols = np.array([0.3,0.6])
names_in_gal_outputs = np.array(['log10_smass', 'log10_sSFR'])
names_for_real_data = np.array(['cos_smass', 'cos_sSFR'])

# for AGN, add in a fake cos gal w/ zero AGN lum so the no AGN run is also picked. 
# cos_smass_data = np.concatenate((cos_smass_data, [10.0]))
# cos_ssfr_data = np.concatenate((cos_ssfr_data, [-9.0]))
# cos_AGN = np.concatenate((cos_AGN, [0.0]))
# print (np.random.random(1)*140.+20.)
# cos_radii_data = np.concatenate((cos_radii_data, (np.random.random(1)*140.+20.)))
# which_survey = np.concatenate((which_survey, ['agn']))

# real_data_to_match = np.asarray([cos_smass_data, cos_ssfr_data, cos_AGN])
# tols = np.array([5., 5., 0.25])
# hi_lo_tols = np.array([5., 5., 0.25])
# names_in_gal_outputs = np.array(['log10_smass', 'log10_sSFR', 'AGN_lum'])
# names_for_real_data = np.array(['cos_smass', 'cos_sSFR', 'cos_AGN'])


c_kms = 2.9979e+05 ### speed of light in km/s

cos_smass_data -= 0.2
cos_comparisons_file = 'cos_data_comparisons.hdf5'
directory_with_gal_folders = '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/'
spec_output_directory = '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/spec_outputs/'
R_in_vir = 2.0
AGN_bool = False
ion = 'si3'
smass_or_ssfr = 'smass'
 

starting_gal_id = int(0)
realizations = 3

real_survery_radii = cos_functions.make_cos_data_comparison_file(cos_comparisons_file, real_data_to_match, names_in_gal_outputs, names_for_real_data, cos_radii_data)

where_matched_bools = cos_functions.populate_hdf5_with_matching_gals(cos_comparisons_file, directory_with_gal_folders, starting_gal_id, real_data_to_match, tols, hi_lo_tols, names_in_gal_outputs, which_survey)

unmatched_cos_smass = cos_smass_data[where_matched_bools == False]
unmatched_cos_ssfr = cos_ssfr_data[where_matched_bools == False]
unmatched_real_data	= np.array([unmatched_cos_smass, unmatched_cos_ssfr])

### Box stuff

snap_directory = '/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023' # directory of all the snap files to be analyzed
# what galaxy, and how much of it we want to look at 
R_in_vir = 2.0 # radius in virial radii
snapshot_file_end = '039_z000p205'

### Keywords to get correct files
particles_included_keyword = "snap_noneq_" + snapshot_file_end
group_included_keyword = "group_tab_" + snapshot_file_end
subfind_included_keyword = "eagle_subfind_tab_" + snapshot_file_end

### For AGN Gals (different naming convention)
# particles_included_keyword = "snap_AGN445_" + snapshot_file_end
# group_included_keyword = "group_tab_" + snapshot_file_end
# subfind_included_keyword = "eagle_subfind_tab_" + snapshot_file_end

group_numbers = np.array(range(1,150))
smass_arr = np.zeros(np.size(group_numbers))
ssfr_arr = np.zeros(np.size(group_numbers))

print group_numbers

for i in range(0,np.size(group_numbers)):
	# get basic properties of simulation and galaxy
	box_size, expansion_factor, hubble_param, gal_coords, gal_velocity, gal_M200, gal_R200, radius, gal_speed, gal_stellar_mass, gal_SFR = \
	EagleFunctions.get_basic_props(snap_directory, R_in_vir, group_numbers[i], particles_included_keyword, group_included_keyword, subfind_included_keyword)

	smass_arr[i] = gal_stellar_mass/M_sol
	ssfr_arr[i] = (gal_SFR*secs_per_year)/gal_stellar_mass


plt.plot(np.log10(smass_arr), np.log10(ssfr_arr), 'r.')
plt.hold(True)

sim_data_to_match = np.array([np.log10(smass_arr), np.log10(ssfr_arr)])
sim_gals_that_match_unmatched_cos_gal = np.array([])
smass_data_of_gals_that_match_unmatched_cos_gals = np.array([])
ssfr_data_of_gals_that_match_unmatched_cos_gals = np.array([])

for n in range(0,np.size(sim_data_to_match[0])):
	match_found = False
	for i in range(0,np.size(unmatched_real_data[0])):
		successes = 0
		for j in range(0,np.shape(unmatched_real_data)[0]):
			if ((unmatched_real_data[1][i] < -11.4) or (unmatched_real_data[1][i] > -9.4)):
				if ((unmatched_real_data[j][i] > sim_data_to_match[j][n]-hi_lo_tols[j]) & (unmatched_real_data[j][i] < sim_data_to_match[j][n] +hi_lo_tols[j])):
					successes += 1
			else:	
				if ((unmatched_real_data[j][i] > sim_data_to_match[j][n]-tols[j]) & (unmatched_real_data[j][i] < sim_data_to_match[j][n] +tols[j])):
					successes += 1

		if successes == np.shape(unmatched_real_data)[0]:
			match_found = True
			
	if match_found:
		sim_gals_that_match_unmatched_cos_gal =  np.append(sim_gals_that_match_unmatched_cos_gal,group_numbers[n])
		smass_data_of_gals_that_match_unmatched_cos_gals = np.append(smass_data_of_gals_that_match_unmatched_cos_gals,sim_data_to_match[0][n])
		ssfr_data_of_gals_that_match_unmatched_cos_gals =  np.append(ssfr_data_of_gals_that_match_unmatched_cos_gals, sim_data_to_match[1][n])

print sim_gals_that_match_unmatched_cos_gal
print ''
print smass_data_of_gals_that_match_unmatched_cos_gals
print ''
print ssfr_data_of_gals_that_match_unmatched_cos_gals

# plt.plot(smass_data_of_gals_that_match_unmatched_cos_gals, ssfr_data_of_gals_that_match_unmatched_cos_gals, 'b.')
plt.hold(False)
plt.savefig('stuff_in_box.png')
plt.close()




