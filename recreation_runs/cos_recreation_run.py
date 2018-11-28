#### Full: 79 out of 179
####: Gass: 28 out of 45 (33 matched, 6 no/bad COS measurements, 1 overlap?)
####: GTO: 14 out of 48
####: Halos 33 out of 44 (40 matched, 7 no/bad COS measurements)
####: Dwarfs: 31 out of 43 (34 matched, 3 have no/bad COS measurements)
####: Halos/Dwarfs 51 out of 86
### Otters@78911

### make col dense maps (example maps)

### Populations (in COS) smass 9.9, ssfr -11 are the cuts

### Look for tests to compare fits with Rongmon
### also, look at a fit vs ssfr there. Are the two quantities that ronngmon fits against actually more significant that sSFR? 
### Keep plugging away at AGN stuff
### Look into the fact that troughs in stacks are not zero centered? 
### Split into near/far side? 
### split into red/blue shifted? 

### semi-rand gass1 has 6 extra?

# No triangles on EWs
# mA

# % no frac for tracking
# break down to entered ism and left, currently in ism, is a star


##################################
# Imports

import h5py 
import numpy as np
import sys
import os
import subprocess
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import real_data

# my function libraries
import SpecwizardFunctions
import EagleFunctions
import survey_realization_functions as cos_functions
import particle_tracking_functions

#################################

### default plotting params
plt.rcParams['axes.labelsize'], plt.rcParams['axes.titlesize'], plt.rcParams['legend.fontsize'], plt.rcParams['xtick.labelsize'], plt.rcParams['ytick.labelsize'] = 13., 14., 11., 13, 13

### If dividing into populations that aren't the surveys use these to pick out galaxies you want

# cos_comparisons_file = 'cos_part_track_tests_dwarfs.hdf5'
cos_comparisons_file = 'cos_data.hdf5'

# ### Super important to make sure these are right before each run. Don't overwrite stuff for the love of god. Especially on long runs

# ### old cosma locations

# # this is the location of the gal output files that contain all the data you need from the eagle galaxies
# # directory_with_gal_folders = '/cosma/home/analyse/rhorton/opp_research/data/AGN_gals/with_AGN'
# directory_with_gal_folders = '/cosma/home/analyse/rhorton/opp_research/data/end_summer_gals'
# lookup_files = '/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/IonizationTables/HM01G+C+SSH/'
# # where all that data is put and where created los live. Basically the home for the run you are doing. 
# # don't make it the same for multiple runs and make sure if you're just running plotting stuff it looks in the right place

# folders = glob.glob('/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/with_partIDs/test_new_spec')

# # folders = glob.glob('/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/with_partIDs/masters_reruns/matching_radii/halos_5rel_1')
# # folders.append(glob.glob('/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/with_partIDs/masters_reruns/matching_radii/gass_5rel_1'))
# # folders.append(glob.glob('/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/with_partIDs/masters_reruns/matching_radii/dwarfs_5rel_1'))

# # folders = glob.glob('/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/with_partIDs/masters_reruns/semi_rand_radii/dwarfs*')
# # folders.append(glob.glob('/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/with_partIDs/masters_reruns/semi_rand_radii/gass*'))
# # folders.append(glob.glob('/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/with_partIDs/masters_reruns/semi_rand_radii/halos*'))
# # folders.append(glob.glob('/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/with_partIDs/masters_reruns/matching_radii/dwarf*'))
# # folders.append(glob.glob('/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/with_partIDs/masters_reruns/matching_radii/halos*'))
# # folders.append(glob.glob('/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/with_partIDs/masters_reruns/matching_radii/gass*'))

# spec_output_directory = np.hstack(folders)

# # combined_plots_folder = None
# combined_plots_folder = '/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/with_partIDs/gal_paper_plots'

# # so it knows where to grab things since it might not be living in Ali_spec_src anymore. Check these before each run!
# # otherwise you'll get confusion between runs. Run something new, but pull old data etc. 
# path_to_param_template = '/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/CGM_template.par'
# path_to_specwizard_executable = '/cosma/home/analyse/rhorton/opp_research/Ali_Spec_src/specwizard'
# path_to_cos_comparisons = '/cosma/home/analyse/rhorton/opp_research/snapshots/%s' % (cos_comparisons_file)

### New rc locations

# this is the location of the gal output files that contain all the data you need from the eagle galaxies
# directory_with_gal_folders = '/projects/ryho3446/data/AGN_gals/with_AGN'
directory_with_gal_folders = '/projects/ryho3446/data/end_summer_gals'
lookup_files = '/projects/ryho3446/Ali_Spec_src/IonizationTables/HM01G+C+SSH/'
# where all that data is put and where created los live. Basically the home for the run you are doing. 
# don't make it the same for multiple runs and make sure if you're just running plotting stuff it looks in the right place

folders = glob.glob('/projects/ryho3446/Ali_Spec_src/with_partIDs/masters_reruns/semi_rand_radii/halos_5rel_1')
folders.append(glob.glob('/projects/ryho3446/Ali_Spec_src/with_partIDs/masters_reruns/semi_rand_radii/gass_5rel_1'))
folders.append(glob.glob('/projects/ryho3446/Ali_Spec_src/with_partIDs/masters_reruns/semi_rand_radii/dwarfs_5rel_1'))

# folders = glob.glob('/projects/ryho3446/Ali_Spec_src/with_partIDs/masters_reruns/semi_rand_radii/dwarfs*')
# folders.append(glob.glob('/projects/ryho3446/Ali_Spec_src/with_partIDs/masters_reruns/semi_rand_radii/gass*'))
# folders.append(glob.glob('/projects/ryho3446/Ali_Spec_src/with_partIDs/masters_reruns/semi_rand_radii/halos*'))
# folders.append(glob.glob('/projects/ryho3446/Ali_Spec_src/with_partIDs/masters_reruns/matching_radii/dwarf*'))
# folders.append(glob.glob('/projects/ryho3446/Ali_Spec_src/with_partIDs/masters_reruns/matching_radii/halos*'))
# folders.append(glob.glob('/projects/ryho3446/Ali_Spec_src/with_partIDs/masters_reruns/matching_radii/gass*'))

spec_output_directory = np.hstack(folders)

# combined_plots_folder = None
combined_plots_folder = '/projects/ryho3446/Ali_Spec_src/with_partIDs/gal_paper_plots'

# so it knows where to grab things since it might not be living in Ali_spec_src anymore. Check these before each run!
# otherwise you'll get confusion between runs. Run something new, but pull old data etc. 
path_to_param_template = '/projects/ryho3446/Ali_Spec_src/CGM_template.par'
path_to_specwizard_executable = '/projects/ryho3446/Ali_Spec_src/specwizard'
path_to_cos_comparisons = '/projects/ryho3446/snapshots/%s' % (cos_comparisons_file)

R_in_vir = 2.0
colorbar = 'hmass'

cos_gass_bool = True
cos_halos_bool = True
cos_dwarfs_bool = True
cos_AGN_bool = False
cos_gto_bool = False
single_gal_for_tests = False

run_specwizard = False
make_new_los_files = False
AGN_bool = False
semi_random_radius = False # radii will be drawn from gauss with center being survey values and 10kpc std
random_radii = False # radii will be random between max and min radius (passed)
covering_frac_bool = False # have covering fraction at a certain value on col dense plots
covering_frac_val = 14.0
virial_vel_bool = False
mean_spectra_bool = False
halo_mass_bool = False # for mass bins in flux plots. Use halo or stellar mass
mass_estimates_bool = False # try to get mass of H along line from lookup tables. 
kinematics_bool = False # look at properties of individual lines
pop_str = 'old'
proch_bool = False
mean_total_mass_bool = False
virial_radii_bool = True
make_realistic_bool = False
new_lines = False
equ_widths_bool = True
log_plots = True

# cut is 9.9 smass in cos, this will be 9.7 after read ins and 9.7 in eagle gals (IMF)
if run_specwizard:
	# makes sure I'm not an idiot and don't run specwizard for everything
	max_smass = 15.
	min_smass = 5.0
	max_ssfr = -5.0
	min_ssfr = -15.
else:
	# make whatever you want to select a subsample
	max_smass = 15.
	min_smass = 5.
	max_ssfr = -5.
	min_ssfr = -15.

starting_gal_id = int(0)
realizations = 1
bins_for_median= 2 # in scatter or maybe hist plots number of bins used to make running median/percentile lines
max_abs_vel = 500. # km/s
min_halo_mass = 10. # for plots. 
max_halo_mass = 14.
plt.rcParams['axes.labelsize'], plt.rcParams['axes.titlesize'], plt.rcParams['legend.fontsize'] = 18., 20., 16.

### for flux plots
radii_bins = [0.,80.,100.,160.,250,500] #kpc
# radii_bins = [0.0,0.4,0.8,1.2,1.6] # virial radii
radii_colors = ['r','g','b','m','k']

mass_colors = ['r','g','b','m']
mass_bins = [8.,9.4,9.85,10.6,12.0] # stellar

# AGN_lum_bins = [-1.0,41.75, 42.25, 42.75, 43.25, 43.75, 44.25, 44.75]
AGN_lum_bins = [-1.0,45.]


# ions = np.array(['HydrogenI','SiliconII','SiliconIII','SiliconIV', 'CarbonII', 'CarbonIV', 'NitrogenV','OxygenVI'])
# ions_short = np.array(['h1','si2','si3','si4', 'c2', 'c4','n5','o6'])
# elements = np.array(['hydrogen', 'silicon', 'silicon', 'silicon', 'carbon', 'carbon', 'nitrogen', 'oxygen'])
# covering_frac_vals = np.array([.124,.197,.234,.218,.272, .272,.151,.1])
# line_wavelengths = np.array([1215.67, 1260.42, 1206.5, 1393.76, 1334.5, 1548.20, 1238.82, 1031.93])

# ions = np.array(['OxygenVI'])
# ions_short = np.array(['o6'])
# elements = np.array(['oxygen'])
# covering_frac_vals = np.array([0.1])
# line_wavelengths = np.array([1031.93])

ions = np.array(['HydrogenI'])
ions_short = np.array(['h1'])
elements = np.array(['hydrogen'])
covering_frac_vals = np.array([.124])
line_wavelengths = np.array([1215.67])

# ions = np.array(['HydrogenI', 'OxygenVI'])
# ions_short = np.array(['h1', 'o6'])
# elements = np.array(['hydrogen', 'oxygen'])
# covering_frac_vals = np.array([.124, 0.1])
# line_wavelengths = np.array([1215.67, 1031.93])


# Which Halo gals are in prochaska? Use for cum mask plots when we want to match Proch gals. Only usable when halos is the only survey selected
# this block and first line below last cos data filter are what need to be added/removed to do this
unmatched_indices = np.array([0,8,10,11,19,20,30,34,35,38,40,41,42])
proch_mask = np.invert(np.in1d(range(0,44), unmatched_indices))
halo_ids = np.array(range(157,201))
proch_ids = halo_ids[proch_mask == True]

#########################################
# This is where the fun begins

# Get COS Data
cos_smass_data, cos_ssfr_data, cos_radii_data, cos_h1_equ_widths, cos_h1_W_errs, cos_h1_W_flags, cos_h1_equ_widths_radii, cos_h1_cols, cos_h1_cols_errs, cos_h1_cols_flags, cos_h1_cols_radii, cos_si3_equ_widths, cos_si3_equ_widths_radii, cos_si3_cols, cos_si3_cols_radii, cos_o6_cols, cos_o6_cols_radii, cos_o6_equ_widths, cos_o6_equ_widths_radii, cos_c4_cols, cos_c4_cols_radii, cos_c4_equ_widths, cos_c4_equ_widths_radii, cos_AGN, which_survey, cos_id_arr = real_data.select_data_for_run(max_smass = max_smass, min_smass = min_smass, max_ssfr = max_ssfr, min_ssfr = min_ssfr, cos_gass=cos_gass_bool, cos_gto=cos_gto_bool, cos_halos = cos_halos_bool, cos_dwarfs=cos_dwarfs_bool, cos_AGN=cos_AGN_bool, single_gal_for_tests=single_gal_for_tests)
ordered_cos_radii = np.sort(cos_radii_data)

cos_smass_data -= 0.2
max_smass -= 0.2
min_smass -= 0.2

### This duplicates the AGN survey, but with and AGN luminosity of zero, so AGN on and off will be matched and can be compared
if AGN_bool:
	cos_smass_data = np.concatenate((cos_smass_data, cos_smass_data))
	cos_ssfr_data = np.concatenate((cos_ssfr_data, cos_ssfr_data))
	cos_AGN = np.concatenate((cos_AGN, np.zeros(np.size(cos_AGN))))
	cos_radii_data = np.concatenate((cos_radii_data, cos_radii_data))
	which_survey = np.concatenate((which_survey, which_survey))
	cos_id_arr = np.concatenate((cos_id_arr, np.arange(203,203+int(np.size(cos_id_arr)),1)))

	real_data_to_match = np.asarray([cos_smass_data, cos_ssfr_data, cos_AGN])
	names_in_gal_outputs = np.array(['log10_smass', 'log10_sSFR', 'AGN_lum'])
	names_for_real_data = np.array(['cos_smass', 'cos_sSFR', 'cos_AGN'])
	tols = np.array([2.0, 2.0, 0.25])
	hi_lo_tols = np.array([2.0, 2.0, 0.25])
	
else:
	real_data_to_match = np.array([cos_smass_data, cos_ssfr_data])
	names_in_gal_outputs = np.array(['log10_smass', 'log10_sSFR'])
	names_for_real_data = np.array(['cos_smass', 'cos_sSFR'])
	tols = np.array([0.2, 0.3])
	hi_lo_tols = np.array([0.3,0.6])

real_survey_radii = cos_functions.make_cos_data_comparison_file(path_to_cos_comparisons, real_data_to_match, names_in_gal_outputs, names_for_real_data, cos_radii_data, which_survey)

where_matched_bools = cos_functions.populate_hdf5_with_matching_gals(path_to_cos_comparisons, directory_with_gal_folders, starting_gal_id, real_data_to_match, real_survey_radii, tols, hi_lo_tols, names_in_gal_outputs, which_survey, combined_plots_folder, spec_output_directory, cos_halos_bool, cos_dwarfs_bool, cos_gass_bool, cos_AGN_bool, cos_gto_bool)

if run_specwizard:
	calls, gals_matched_for_each_cos_gal = cos_functions.run_specwizard_for_matching_gals(path_to_cos_comparisons, directory_with_gal_folders, spec_output_directory, path_to_param_template, path_to_specwizard_executable, path_to_cos_comparisons, real_survey_radii, cos_id_arr, semi_random_radius, realizations, AGN_bool, random_radii, make_new_los_files=make_new_los_files)

	print 'Number of Real Gals Passed'
	print np.size(gals_matched_for_each_cos_gal)
	print 'Number of gals Matched'
	print np.size(gals_matched_for_each_cos_gal[gals_matched_for_each_cos_gal!=0])

h1_cols_indices, h1_W_indices, si3_cols_indices, si3_equ_widths_indices, o6_cols_indices, o6_equ_widths_indices, c4_cols_indices, c4_equ_widths_indices = real_data.cos_where_matched_in_EAGLE(AGN_bool, proch_bool, proch_ids, where_matched_bools, cos_smass_data, cos_ssfr_data, cos_radii_data, cos_id_arr, cos_h1_equ_widths, cos_h1_equ_widths_radii, cos_h1_cols, cos_h1_cols_radii, cos_si3_equ_widths, cos_si3_equ_widths_radii, cos_si3_cols, cos_si3_cols_radii, cos_o6_cols, cos_o6_cols_radii, cos_o6_equ_widths, cos_o6_equ_widths_radii, cos_c4_cols, cos_c4_cols_radii, cos_c4_equ_widths, cos_c4_equ_widths_radii, cos_AGN)

### If you want KS tests and histrograms for each realization run in all directories passed (each KS test is on a single rel even if a folder is multiple)
# if equ_widths_bool:
# 	cos_functions.handle_single_realization_statistics(spec_output_directory, cos_smass_data[h1_W_indices], cos_ssfr_data[h1_W_indices], cos_id_arr[h1_W_indices],\
# 		cos_h1_equ_widths[h1_W_indices], cos_h1_W_flags[h1_W_indices], cos_h1_equ_widths_radii[h1_W_indices], cos_h1_cols[h1_W_indices], cos_h1_cols_flags[h1_W_indices], \
# 		cos_h1_cols_radii[h1_W_indices], equ_widths_bool)

# else:
# 	cos_functions.handle_single_realization_statistics(spec_output_directory, cos_smass_data[h1_cols_indices], cos_ssfr_data[h1_cols_indices], cos_id_arr[h1_cols_indices],\
# 		cos_h1_equ_widths[h1_cols_indices], cos_h1_W_flags[h1_cols_indices], cos_h1_equ_widths_radii[h1_cols_indices], cos_h1_cols[h1_cols_indices], cos_h1_cols_flags[h1_cols_indices], \
# 		cos_h1_cols_radii[h1_cols_indices], equ_widths_bool)

if combined_plots_folder != None:
	os.chdir(combined_plots_folder)
else:
	os.chdir(spec_output_directory)

for i in range(0,np.size(ions)):
	print 'doing new ion'
	print ions[i]
	print ''
	ion = ions[i]
	lookup_file = lookup_files+ions_short[i]+'.hdf5'
	covering_frac_val = covering_frac_vals[i]
	lambda_line = line_wavelengths[i]
	
	if ion == 'HydrogenI':
		plot_cols = cos_h1_cols[h1_cols_indices]
		plot_cols_err = cos_h1_cols_errs[h1_cols_indices]
		plot_cols_flags = cos_h1_cols_flags[h1_cols_indices]
		plot_cols_radii = cos_h1_cols_radii[h1_cols_indices]
		plot_equ_widths = cos_h1_equ_widths[h1_W_indices]
		plot_W_errs = cos_h1_W_errs[h1_W_indices]
		plot_W_flags = cos_h1_W_flags[h1_W_indices]
		plot_equ_widths_radii = cos_h1_equ_widths_radii[h1_W_indices]
		if equ_widths_bool:
			curr_cos_id_arr = cos_id_arr[h1_W_indices]
			curr_cos_smass = cos_smass_data[h1_W_indices]
			curr_cos_ssfr = cos_ssfr_data[h1_W_indices]
		else:
			curr_cos_id_arr = cos_id_arr[h1_cols_indices]
			curr_cos_id_arr = cos_id_arr[h1_cols_indices]
			curr_cos_smass = cos_smass_data[h1_cols_indices]
			curr_cos_ssfr = cos_ssfr_data[h1_cols_indices]
	elif ion == 'SiliconIII':
		plot_cols = cos_si3_cols[si3_cols_indices]
		plot_cols_radii = cos_si3_cols_radii[si3_cols_indices]
		plot_equ_widths = cos_si3_equ_widths[si3_equ_widths_indices]
		plot_equ_widths_radii = cos_si3_equ_widths_radii[si3_equ_widths_indices]
		if equ_widths_bool:
			curr_cos_id_arr = cos_id_arr[si3_equ_widths_indices]
		else:
			curr_cos_id_arr = cos_id_arr[si3_cols_indices]
	elif ion == 'OxygenVI':
		plot_cols = cos_o6_cols[o6_cols_indices]
		plot_cols_radii = cos_o6_cols_radii[o6_cols_indices]
		plot_equ_widths = cos_o6_equ_widths[o6_equ_widths_indices]
		plot_equ_widths_radii = cos_o6_equ_widths_radii[o6_equ_widths_indices]
		if equ_widths_bool:
			curr_cos_id_arr = cos_id_arr[o6_equ_widths_indices]
		else:
			curr_cos_id_arr = cos_id_arr[o6_cols_indices]
	elif ion == 'CarbonIV':
		plot_cols = cos_c4_cols[c4_cols_indices]
		plot_cols_radii = cos_c4_cols_radii[c4_cols_indices]
		plot_equ_widths = cos_c4_equ_widths[c4_equ_widths_indices]
		plot_equ_widths_radii = cos_c4_equ_widths_radii[c4_equ_widths_indices]
		if equ_widths_bool:
			curr_cos_id_arr = cos_id_arr[c4_equ_widths_indices]
		else:
			curr_cos_id_arr = cos_id_arr[c4_cols_indices]
	else: 
		plot_cols = np.array([])
		plot_cols_radii = np.array([])
		plot_equ_widths = np.array([])
		plot_equ_widths_radii = np.array([])
		if equ_widths_bool:
			curr_cos_id_arr = np.array([])
		else:
			curr_cos_id_arr = np.array([])
		print 'warning: No COS data prepared for this ion'

	# EagleFunctions.actual_cumulative_mass_for_EAGLE_gals('/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/')

	covered, total, ssfr, masses, smasses, redshifts, radii, virial_radii, R200, cols, equ_widths, eagle_ids, flux_for_stacks, vel_for_stacks, virial_vel_for_stacks, cols, H_cols, num_minimas, depths, FWHMs, centroid_vels, temps, line_ion_densities, line_nHs, escape_vels, virial_radii_for_kin, halo_masses_for_kin, stellar_masses_for_kin, ssfr_for_kin, redshifts_for_kin, ion_num_densities, temperatues, gas_densities = cos_functions.get_EAGLE_data_for_plots(ions_short[i], lambda_line, curr_cos_id_arr, lambda_line, spec_output_directory, lookup_file, max_smass, min_smass, max_ssfr, min_ssfr, tols, hi_lo_tols, ordered_cos_radii, covering_frac_bool, covering_frac_val, max_abs_vel, mass_estimates_bool, kinematics_bool, make_realistic_bool = make_realistic_bool, offset = starting_gal_id)

	# # can do both to compare num_minima with and without realistic spectra
	# covered, total, ssfr, masses, smasses, redshifts, radii, virial_radii, R200, cols, equ_widths, eagle_ids, flux_for_stacks, vel_for_stacks, virial_vel_for_stacks, cols, H_cols, real_num_minimas, depths, FWHMs, centroid_vels, temps, line_ion_densities, line_nHs, escape_vels, virial_radii_for_kin, halo_masses_for_kin, stellar_masses_for_kin, ssfr_for_kin, redshifts_for_kin, ion_num_densities, temperatues, gas_densities = cos_functions.get_EAGLE_data_for_plots(ions_short[i], lambda_line, curr_cos_id_arr, lambda_line, spec_output_directory, lookup_file, max_smass, min_smass, max_ssfr, min_ssfr, tols, hi_lo_tols, ordered_cos_radii, covering_frac_bool, covering_frac_val, max_abs_vel, mass_estimates_bool, kinematics_bool, make_realistic_bool = True, offset = starting_gal_id)

	# bins = np.arange(0,np.max(num_minimas)+2) - 0.5
	# plt.hist(num_minimas, bins, alpha=0.5, color='b', label='Idealized Spectra')
	# plt.hold(True)
	# plt.hist(real_num_minimas, bins, alpha=0.5, color ='#00FF00', label='Instrumental Spectra')
	# plt.hold(False)
	# plt.legend(loc='upper right')
	# plt.title('Number of Minima per Sightline')
	# plt.xlabel('Number of Minima')
	# plt.ylabel('Number of Sightlines (x1000)')
	# ax = plt.gca()
	# ax.set_yticklabels((ax.get_yticks()/1000.).astype(int))
	# plt.tight_layout()
	# plt.savefig('num_minima_comp.pdf')
	# plt.close()

	# print 'analysis of components'
	# print np.size(num_minimas)
	# print np.size(num_minimas[num_minimas==0])
	# print np.size(num_minimas[num_minimas==1])
	# print np.size(num_minimas[num_minimas==2])
	# print np.size(num_minimas[num_minimas==3])
	# print ''

	# np.savez('/projects/ryho3446/snapshots/kin_outputs.npz', num_minimas, centroid_vels, depths, FWHMs, radii, temps, line_ion_densities, line_nHs, escape_vels, virial_radii_for_kin, halo_masses_for_kin, stellar_masses_for_kin, ssfr_for_kin, redshifts_for_kin)

	# npzfile = np.load('/projects/ryho3446/snapshots/kin_outputs.npz')
 # 	num_minimas, centroid_vels, depths, FWHMs, radii, temps, line_ion_densities, line_nHs, escape_vels, virial_radii_for_kin, halo_masses_for_kin, stellar_masses_for_kin, ssfr_for_kin, redshifts_for_kin =  npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2'], npzfile['arr_3'], npzfile['arr_4'], npzfile['arr_5'], npzfile['arr_6'], npzfile['arr_7'], npzfile['arr_8'], npzfile['arr_9'], npzfile['arr_10'], npzfile['arr_11'], npzfile['arr_12'], npzfile['arr_13'] # [npzfile['arr_%s' % (i)] for i in range(np.size(npzfile))]

	# cos_functions.kinematic_plots(num_minimas, centroid_vels, depths, FWHMs, radii, temps, line_ion_densities, line_nHs, escape_vels, virial_radii_for_kin, halo_masses_for_kin, stellar_masses_for_kin, ssfr_for_kin, redshifts_for_kin, bins_for_median)
	
	# cos_functions.neutral_columns_plot(cols, H_cols, radii, virial_radii, R200, smasses, masses, ssfr, ion_num_densities, gas_densities, temperatues, mean_total_mass_bool, virial_radii_bool, pop_str)

	if equ_widths_bool:
		# cos_functions.make_equ_width_plots(ions_short[i], ssfr, masses, smasses, radii, virial_radii, equ_widths, eagle_ids, curr_cos_id_arr, plot_equ_widths, plot_W_errs, plot_W_flags, plot_equ_widths_radii, colorbar, bins_for_median, log_plots) # line in Angst

		cos_functions.make_equ_width_contour_plots(ions_short[i], radii, virial_radii, equ_widths, smasses, ssfr, plot_equ_widths, plot_W_errs, plot_W_flags, plot_equ_widths_radii, curr_cos_smass, curr_cos_ssfr, virial_radii_bool, log_plots)

	else:
		# cos_functions.make_col_dense_plots(ions_short[i], covered, total, ssfr, masses, smasses, radii, virial_radii, cols, eagle_ids, curr_cos_id_arr, plot_cols, plot_cols_err, plot_cols_flags, plot_cols_radii, covering_frac_val, colorbar, bins_for_median)

		cos_functions.make_contour_col_dense_plots(ions_short[i], radii, virial_radii, cols, plot_cols, plot_cols_err, plot_cols_flags, plot_cols_radii, smasses, ssfr, virial_radii_bool)

	# cos_functions.plot_for_multiple_gals_by_radius(ions_short[i], radii_bins, radii_colors, virial_vel_bool, virial_radii_bool, halo_mass_bool, mean_spectra_bool, radii, virial_radii, masses, smasses, flux_for_stacks, vel_for_stacks, virial_vel_for_stacks, min_halo_mass, max_halo_mass)

	# cos_functions.plot_for_multiple_gals_by_mass(ions_short[i], mass_bins, mass_colors, virial_vel_bool, virial_radii_bool, halo_mass_bool, mean_spectra_bool, masses, smasses, radii, virial_radii, flux_for_stacks, vel_for_stacks, virial_vel_for_stacks, min_halo_mass, max_halo_mass)

	# cos_functions.plot_for_multiple_gals_by_radius(spec_output_directory, combined_plots_folder, ions_short[i], radii_bins, radii_colors, max_abs_vel = 1000.,
	# 	virial_vel_bool=False, virial_radii_bool=False, mean_spectra_bool=True, min_halo_mass=10.0, max_halo_mass=14.0, min_radius=20., max_radius = 1000., plot_chance = plot_chance, offset = starting_gal_id)

	# cos_functions.plot_for_multiple_gals_by_mass(spec_output_directory, combined_plots_folder, ions_short[i], mass_bins, max_abs_vel = 380.,
	# 	virial_vel_bool=False, virial_radii_bool=False, mean_spectra_bool=True, min_halo_mass=10.0, max_halo_mass=14.0, min_radius=20., max_radius = 1000., min_ssfr = -15, max_ssfr = -5.0, offset = starting_gal_id)



	# radii_bins_for_mean_equ_widths = np.array([0.,163.9,300.])
	# cos_functions.mean_equ_widts_binned_by_radius(ions_short[i], spec_output_directory, real_survey_radii, combined_plots_folder, lambda_line, AGN_bool =AGN_bool, radii_bins = radii_bins_for_mean_equ_widths, offset = starting_gal_id) # line in Angst

	# cos_functions.plot_for_multiple_gals_by_AGN_lum(spec_output_directory, combined_plots_folder, ions_short[i], AGN_lum_bins, radii_bins=radii_bins_for_mean_equ_widths, text_output=True, max_abs_vel = None,
	# 	virial_vel_bool=False, virial_radii_bool=False, mean_spectra_bool=True, min_halo_mass=10.0, max_halo_mass=14.0, min_radius=20., max_radius = 1000., offset = starting_gal_id)

# 	# cos_functions.fits(ions_short[i], which_survey, spec_output_directory, combined_plots_folder, plot_equ_widths, plot_equ_widths_radii, covering_frac_bool = True, covering_frac_val = 14.0, lambda_line = lambda_line, colorbar = colorbar, offset = starting_gal_id) # line in Angst

# list_for_all_id_data = particle_tracking_functions.get_all_id_data(spec_output_directory)
# print 'got list'
# print list_for_all_id_data
# print ''
# particle_tracking_functions.get_particle_properties(list_for_all_id_data, ions, ions_short, elements, lookup_files, new_lines)


# ########################################

# # directory_with_gal_folders = '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals'
# # spec_output_directory = np.array(['/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/spec_outputs'])
# # # spec_output_directory = np.array(['/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/halo_dwarf_1rel/',
# # # 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/halo_dwarf_1rel_2/'])#,
# # 								  # '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/halo_dwarf_1rel_3/',
# # 								  # '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/halo_dwarf_1rel_4/'])
# # R_in_vir = 2.0
# # AGN_bool = False
# # ion = 'h1'
# # colorbar = 'smass'
 

# # starting_gal_id = int(0)
# # realizations = 25


# # real_survey_radii = cos_functions.make_cos_data_comparison_file(cos_comparisons_file, real_data_to_match, names_in_gal_outputs, names_for_real_data, cos_radii_data)

# # where_matched_bools = cos_functions.populate_hdf5_with_matching_gals(cos_comparisons_file, directory_with_gal_folders, starting_gal_id, real_data_to_match, real_survey_radii, tols, hi_lo_tols, names_in_gal_outputs, which_survey)

# # calls, gals_matched_for_each_cos_gal = cos_functions.run_specwizard_for_matching_gals(cos_comparisons_file, directory_with_gal_folders, real_survey_radii, realizations, AGN_bool = AGN_bool, random_radii=False)




