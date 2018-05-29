#### Full: 79 out of 179
####: Gass: 14 out of 45
####: GTO: 14 out of 48
####: Halos 17 out of 44
####: Dwarfs 
####: Halos/Dwarfs 51 out of 86


### make col dense maps (example maps)

### high and low ssfr for the mean spectra plot when binned by mass (-11 cut in ssfr? for red vs blue)
### Low ssfr has slighlty more H1 (dips to 0.7, no 0.8) still can't save it

### Look for tests to compare fits with Rongmon
### also, look at a fit vs ssfr there. Are the two quantities that ronngmon fits against actually more significant that sSFR? 
### Keep plugging away at AGN stuff
### Look into the fact that troughs in stacks are not zero centered? 
### Split into near/far side? 
### split into red/blue shifted? 

### Gass 1 rel: 3hr 52min??? (EW LyA, EW Si3)
### 3 rel: 3hr 8min

### Dwarf 1 rel: 1hr 44min (All EW, Si2,3,4; H1; N5; C2,4)
### 5 rel: 2hr 38min

### Halos blue 1 rel: 1hr 19min (EW and Col for H1, all others are Col, C2,3,4; Mg2; N5; O1,6; Si2,3,4)
### 5 rel: 2hr 17min
### 10 rel: 3hr 28min

### Halos red 1 re: 1hr 18min (EW and Col for H1, all others are Col, C2,3,4; Mg2; N5; O1,6; Si2,3,4)
### 10 rel: 5hr 6min

### AGN 1 rel: 6min? (EW for H1 and Si3)

### Goals: 10 single runs of each category for statistiacal comparisons (KS)
### 50 realizations of each category for stacks etc. 
### Do again for some form of randomized radii? Spread about real values or select randomly from full range? Probably fomrer


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

cos_smass_data, cos_ssfr_data, cos_radii_data, cos_h1_equ_widths, cos_h1_equ_widths_radii, cos_h1_cols, cos_h1_cols_radii, cos_si3_equ_widths, cos_si3_equ_widths_radii, cos_si3_cols, cos_si3_cols_radii, cos_o6_cols, cos_o6_cols_radii, cos_o6_equ_widths, cos_o6_equ_widths_radii, cos_AGN, which_survey = real_data.select_data_for_run(cos_gass=False, cos_gto=False, cos_halos_blue=False, cos_halos_red= False, cos_dwarfs=False, cos_AGN=True)

# real_data_to_match = np.array([cos_smass_data, cos_ssfr_data])
# tols = np.array([0.2, 0.3])
# hi_lo_tols = np.array([0.3,0.6])
# names_in_gal_outputs = np.array(['log10_smass', 'log10_sSFR'])
# names_for_real_data = np.array(['cos_smass', 'cos_sSFR'])

### Only strong AGN? 
original_AGN = cos_AGN
original_radii = cos_radii_data

# cos_smass_data = cos_smass_data[(original_AGN >= 43.0) & (original_radii <= 150.)]
# cos_ssfr_data = cos_ssfr_data[(original_AGN >= 43.0) & (original_radii <= 150.)]
# cos_AGN = cos_AGN[(original_AGN >= 43.0) & (original_radii <= 150.)]
# cos_radii_data = cos_radii_data[(original_AGN >= 43.0) & (original_radii <= 150.)]

# cos_smass_data = cos_smass_data[(original_AGN >= 43.0)]
# cos_ssfr_data = cos_ssfr_data[(original_AGN >= 43.0)]
# cos_AGN = cos_AGN[(original_AGN >= 43.0)]
# cos_radii_data = cos_radii_data[(original_AGN >= 43.0)]

# for AGN, add in a fake cos gal w/ zero AGN lum so the no AGN run is also picked. 
cos_smass_data = np.concatenate((cos_smass_data, cos_smass_data))
cos_ssfr_data = np.concatenate((cos_ssfr_data, cos_ssfr_data))
cos_AGN = np.concatenate((cos_AGN, np.zeros(np.size(cos_AGN))))
cos_radii_data = np.concatenate((cos_radii_data, cos_radii_data))
which_survey = np.concatenate((which_survey, which_survey))

real_data_to_match = np.asarray([cos_smass_data, cos_ssfr_data, cos_AGN])
tols = np.array([0.8, 1.0, 0.25])
hi_lo_tols = np.array([0.8, 1.0, 0.25])
names_in_gal_outputs = np.array(['log10_smass', 'log10_sSFR', 'AGN_lum'])
names_for_real_data = np.array(['cos_smass', 'cos_sSFR', 'cos_AGN'])


c_kms = 2.9979e+05 ### speed of light in km/s

cos_smass_data -= 0.2
cos_comparisons_file = 'cos_data_comparisons_newAGN50rel.hdf5'

### Super important to make sure these are right before each run. Don't overwrite stuff for the love of god. Especially on long runs

# this is the location of the gal output files that contain all the data you need from the eagle galaxies
directory_with_gal_folders = '/gpfs/data/analyse/rhorton/opp_research/data/AGN_gals/with_AGN'
# where all that data is put and where created los live. Basically the home for the run you are doing. 
# don't make it the same for multiple runs and make sure if you're just running plotting stuff it looks in the right place
spec_output_directory = np.array(['/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/newAGN_50rel_2'])
# spec_output_directory = np.array(['/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/halo_dwarf_1rel/',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/halo_dwarf_1rel_2/'])#,
								  # '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/halo_dwarf_1rel_3/',
								  # '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/halo_dwarf_1rel_4/'])

combined_plots_folder = None

# so it knows where to grab things since it might not be living in Ali_spec_src anymore. Check these before each run!
# otherwise you'll get confusion between runs. Run something new, but pull old data etc. 
path_to_param_template = '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/mybox.par'
path_to_specwizard_executable = '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/specwizard'
path_to_cos_comparisons = '/gpfs/data/analyse/rhorton/opp_research/snapshots/%s' % (cos_comparisons_file)


R_in_vir = 2.0
AGN_bool = True
semi_random_radius = False # radii will be drawn from gauss with center being survey values and 10kpc std
random_radii = False # radii will be random between max and min radius (passed)
ions = np.array(['h1','si2','si3','si4','c4','n5','o6'])
covering_frac_vals = np.array([.124,.197,.234,.218,.272,.151,.1])
line_wavelengths = np.array([1215.67, 1260.42, 1206.5, 1393.76, 1548.20, 1238.82, 1031.93])
# ions = np.array(['h1'])
# covering_frac_vals = np.array([0.124])
# line_wavelengths = np.array([1215.67])
smass_or_ssfr = 'smass'
 

starting_gal_id = int(0)
realizations = 50

real_survery_radii = cos_functions.make_cos_data_comparison_file(cos_comparisons_file, real_data_to_match, names_in_gal_outputs, names_for_real_data, cos_radii_data)

where_matched_bools = cos_functions.populate_hdf5_with_matching_gals(cos_comparisons_file, directory_with_gal_folders, starting_gal_id, real_data_to_match, real_survery_radii, tols, hi_lo_tols, names_in_gal_outputs, which_survey)

# calls, gals_matched_for_each_cos_gal = cos_functions.run_specwizard_for_matching_gals(cos_comparisons_file, directory_with_gal_folders, spec_output_directory, path_to_param_template, path_to_specwizard_executable, path_to_cos_comparisons, real_survery_radii, semi_random_radius, realizations, AGN_bool, random_radii)

# print 'Number of Real Gals Passed'
# print np.size(gals_matched_for_each_cos_gal)
# print 'Number of gals Matched'
# print np.size(gals_matched_for_each_cos_gal[gals_matched_for_each_cos_gal!=0])


if AGN_bool == False:
	cos_h1_equ_widths_radii = cos_h1_equ_widths_radii[((cos_h1_equ_widths > 0.) & (where_matched_bools == True) & (real_data_to_match[1] != 1000))]
	cos_h1_equ_widths = cos_h1_equ_widths[((cos_h1_equ_widths > 0.) & (where_matched_bools == True) & (real_data_to_match[1] != 1000))]
	cos_h1_cols_radii = cos_h1_cols_radii[((cos_h1_cols > 0.) & (where_matched_bools == True) & (real_data_to_match[1] != 1000))]
	cos_h1_cols = cos_h1_cols[((cos_h1_cols > 0.) & (where_matched_bools == True) & (real_data_to_match[1] != 1000))]

	cos_si3_equ_widths_radii = cos_si3_equ_widths_radii[((cos_si3_equ_widths > 0.) & (where_matched_bools == True) & (real_data_to_match[1] != 1000))]
	cos_si3_equ_widths = cos_si3_equ_widths[((cos_si3_equ_widths > 0.) & (where_matched_bools == True) & (real_data_to_match[1] != 1000))]
	cos_si3_cols_radii = cos_si3_cols_radii[((cos_si3_cols > 0.) & (where_matched_bools == True) & (real_data_to_match[1] != 1000))]
	cos_si3_cols = cos_si3_cols[((cos_si3_cols > 0.) & (where_matched_bools == True) & (real_data_to_match[1] != 1000))]

	cos_o6_equ_widths_radii = cos_o6_equ_widths_radii[((cos_o6_equ_widths > 0.) & (where_matched_bools == True) & (real_data_to_match[1] != 1000))]
	cos_o6_equ_widths = cos_o6_equ_widths[((cos_o6_equ_widths > 0.) & (where_matched_bools == True) & (real_data_to_match[1] != 1000))]
	cos_o6_cols_radii = cos_o6_cols_radii[((cos_o6_cols > 0.) & (where_matched_bools == True) & (real_data_to_match[1] != 1000))]
	cos_o6_cols = cos_o6_cols[((cos_o6_cols > 0.) & (where_matched_bools == True) & (real_data_to_match[1] != 1000))]

else:
	num_real_gals = int(np.size(cos_smass_data)/2.0)
	cos_h1_equ_widths_radii = cos_h1_equ_widths_radii[((cos_h1_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (real_data_to_match[1][0:num_real_gals] != 1000))]
	cos_h1_equ_widths = cos_h1_equ_widths[((cos_h1_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (real_data_to_match[1][0:num_real_gals] != 1000))]
	cos_h1_cols_radii = cos_h1_cols_radii[((cos_h1_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (real_data_to_match[1][0:num_real_gals] != 1000))]
	cos_h1_cols = cos_h1_cols[((cos_h1_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (real_data_to_match[1][0:num_real_gals] != 1000))]

	cos_si3_equ_widths_radii = cos_si3_equ_widths_radii[((cos_si3_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (real_data_to_match[1][0:num_real_gals] != 1000))]
	cos_si3_equ_widths = cos_si3_equ_widths[((cos_si3_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (real_data_to_match[1][0:num_real_gals] != 1000))]
	cos_si3_cols_radii = cos_si3_cols_radii[((cos_si3_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (real_data_to_match[1][0:num_real_gals] != 1000))]
	cos_si3_cols = cos_si3_cols[((cos_si3_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (real_data_to_match[1][0:num_real_gals] != 1000))]

	cos_o6_equ_widths_radii = cos_o6_equ_widths_radii[((cos_o6_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (real_data_to_match[1][0:num_real_gals] != 1000))]
	cos_o6_equ_widths = cos_o6_equ_widths[((cos_o6_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (real_data_to_match[1][0:num_real_gals] != 1000))]
	cos_o6_cols_radii = cos_o6_cols_radii[((cos_o6_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (real_data_to_match[1][0:num_real_gals] != 1000))]
	cos_o6_cols = cos_o6_cols[((cos_o6_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (real_data_to_match[1][0:num_real_gals] != 1000))]	

cos_h1_equ_widths[cos_h1_equ_widths > 5.] /= 1000.
cos_si3_equ_widths[cos_si3_equ_widths > 5.] /= 1000.
cos_o6_equ_widths_radii[cos_o6_equ_widths > 5.] /= 1000.

for i in range(0,np.size(ions)):
	ion = ions[i]
	covering_frac_val = covering_frac_vals[i]
	lambda_line = line_wavelengths[i]
	if ion == 'h1':
		plot_cols = cos_h1_cols
		plot_cols_radii = cos_h1_cols_radii
		plot_equ_widths = cos_h1_equ_widths
		plot_equ_widths_radii = cos_h1_equ_widths_radii
	elif ion == 'si3':
		plot_cols = cos_si3_cols
		plot_cols_radii = cos_si3_cols_radii
		plot_equ_widths = cos_si3_equ_widths
		plot_equ_widths_radii = cos_si3_equ_widths_radii
	elif ion == 'o6':
		plot_cols = cos_o6_cols
		plot_cols_radii = cos_o6_cols_radii
		plot_equ_widths = cos_o6_equ_widths
		plot_equ_widths_radii = cos_o6_equ_widths_radii
	else: 
		plot_cols =  np.array([]) 
		plot_cols_radii =  np.array([]) 
		plot_equ_widths =  np.array([]) 
		plot_equ_widths_radii =  np.array([]) 


	os.chdir(spec_output_directory)

	# cos_functions.make_col_dense_plots(ion, spec_output_directory, combined_plots_folder, plot_cols, plot_cols_radii, covering_frac_bool = True, covering_frac_val = 14.0, smass_or_ssfr = smass_or_ssfr, bins_for_median=3, offset = starting_gal_id)

	# cos_functions.make_equ_width_plots(ion, spec_output_directory, combined_plots_folder, plot_equ_widths, plot_equ_widths_radii, covering_frac_bool = True, covering_frac_val = 14.0, lambda_line, smass_or_ssfr = smass_or_ssfr, bins_for_median=3, offset = starting_gal_id) # line in Angst

	radii_bins_for_mean_equ_widths = np.array([0.,163.9,300.])
	cos_functions.mean_equ_widts_binned_by_radius(ion, spec_output_directory, real_survery_radii, combined_plots_folder, lambda_line, AGN_bool =AGN_bool, radii_bins = radii_bins_for_mean_equ_widths, offset = starting_gal_id) # line in Angst
	# cos_functions.covering_fractions_binned_by_radius(ion, spec_output_directory, combined_plots_folder, lambda_line, AGN_bool =AGN_bool, radii_bins = radii_bins_for_mean_equ_widths, covering_frac_val=covering_frac_val, offset = starting_gal_id) # line in Angst


	# smass_or_ssfr = 'ssfr'

	# cos_functions.make_col_dense_plots(ion, spec_output_directory, combined_plots_folder, plot_cols, plot_cols_radii, covering_frac_bool = True, covering_frac_val = 14.0, smass_or_ssfr = smass_or_ssfr, bins_for_median = 6, offset = starting_gal_id)

	# cos_functions.make_equ_width_plots(ion, spec_output_directory, combined_plots_folder, plot_equ_widths, plot_equ_widths_radii, covering_frac_bool = True, covering_frac_val = 14.0, lambda_line, smass_or_ssfr = smass_or_ssfr, bins_for_median=3, offset = starting_gal_id) # line in Angst

	# cos_functions.make_contour_col_dense_plots(ion, spec_output_directory, combined_plots_folder, cos_h1_cols, cos_h1_cols_radii, cos_si3_cols, cos_si3_cols_radii, plot_cols, plot_cols_radii, covering_frac_bool = True, covering_frac_val = 14.0, offset = starting_gal_id)

	# cos_functions.make_equ_width_contour_plots(ion, spec_output_directory, combined_plots_folder, plot_equ_widths, plot_equ_widths_radii, covering_frac_bool = True, covering_frac_val = 14.0, lambda_line, offset = starting_gal_id)

	radii_bins = [0.,55.,100.,160.,250,500] #kpc
	radii_colors = ['r','g','b','m','k']
	mass_bins = [8.,9.4,9.85,10.6,12.0]
	# AGN_lum_bins = [-1.0, 41.0, 45.0]	
	AGN_lum_bins = [-1.0, 41.0, 45.0]
	# radii_bins = [0.0,0.4,0.8,1.2,1.6] # virial radii
	# print 'before by radius'
	# cos_functions.plot_for_multiple_gals_by_radius(spec_output_directory, combined_plots_folder, ion, radii_bins, radii_colors, max_abs_vel = None,
	# 	virial_vel_bool=True, virial_radii_bool=False, mean_spectra_bool=True, min_halo_mass=10.0, max_halo_mass=14.0, min_radius=20., max_radius = 1000., plot_chance = 0.05, offset = starting_gal_id)

	# cos_functions.plot_for_multiple_gals_by_mass(spec_output_directory, combined_plots_folder, ion, mass_bins, max_abs_vel = 700,
	# 	virial_vel_bool=False, virial_radii_bool=False, mean_spectra_bool=True, min_halo_mass=10.0, max_halo_mass=14.0, min_radius=20., max_radius = 1000., min_ssfr = -15, max_ssfr = -5.0, offset = starting_gal_id)

	# cos_functions.plot_for_multiple_gals_by_radius(spec_output_directory, combined_plots_folder, ion, radii_bins, radii_colors, max_abs_vel = 1000.,
	# 	virial_vel_bool=False, virial_radii_bool=False, mean_spectra_bool=True, min_halo_mass=10.0, max_halo_mass=14.0, min_radius=20., max_radius = 1000., plot_chance = 0.1, offset = starting_gal_id)

	# cos_functions.plot_for_multiple_gals_by_mass(spec_output_directory, combined_plots_folder, ion, mass_bins, max_abs_vel = 380.,
	# 	virial_vel_bool=False, virial_radii_bool=False, mean_spectra_bool=True, min_halo_mass=10.0, max_halo_mass=14.0, min_radius=20., max_radius = 1000., min_ssfr = -15, max_ssfr = -11.0, offset = starting_gal_id)



	# cos_functions.plot_for_multiple_gals_by_AGN_lum(spec_output_directory, combined_plots_folder, ion, AGN_lum_bins, text_output=True, max_abs_vel = 500.,
	# 	virial_vel_bool=False, virial_radii_bool=False, mean_spectra_bool=True, min_halo_mass=10.0, max_halo_mass=14.0, min_radius=163.9, max_radius = 2000.0, offset = starting_gal_id)

	# cos_functions.fits(ion, which_survey, spec_output_directory, combined_plots_folder, plot_equ_widths, plot_equ_widths_radii, covering_frac_bool = True, covering_frac_val = 14.0, lambda_line, smass_or_ssfr = smass_or_ssfr, offset = starting_gal_id) # line in Angst
