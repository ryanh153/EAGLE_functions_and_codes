import h5py 
import numpy as np
import SpecwizardFunctions
import EagleFunctions
import sys
import os
import subprocess
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import bisect
from matplotlib.colors import LogNorm
from multiprocessing import Pool as pool
import multiprocessing
import gen_lsf
import math
import scipy.stats
from matplotlib import rc


### constants
c_kms = 3.0e5
parsec_to_cm = 3.0857e18 # cm
G = 6.67e-8 # cgs
sol_mass_to_g = 1.99e33
m_e = 9.10938356e-28 # g
m_p = 1.6726219e-24 # g
m_H = m_e+m_p
mu = 1.3 # Hydrogen to total mass correction factor
pi = 3.1415
h = 0.667
x_H = 0.752
omega_b = 0.04825
rho_bar_norm = 1.88e-29

### 22 calls, 10 finished, 34 total sightlines, 44 times "file written" printed, 


def make_cos_data_comparison_file(cos_comparisons_file, real_data_to_match, names_in_gal_outputs, names_for_real_data, real_survey_radii, which_survey):
	with h5py.File(cos_comparisons_file, 'w') as hf:
		cos_gals = hf.create_group('cos_gals')
		for i in range(0,np.shape(real_data_to_match)[0]):
			temp_cos_name = cos_gals.create_dataset(names_for_real_data[i], (np.size(real_data_to_match[i]),), maxshape=(None,), data = real_data_to_match[i])

		real_survey_radii = np.array(cos_gals.create_dataset('real_survey_radii', (np.size(real_survey_radii),), maxshape=(None,), data = real_survey_radii))

		which_survey = np.array(cos_gals.create_dataset('which_survey', (np.size(which_survey),), maxshape=(None,), data = which_survey))

		matching_gals = cos_gals.create_group('matching_gals')
		for i in range(0,np.size(real_data_to_match[0])):
			matching_gals.create_dataset('gal_'+str(i),(0,), maxshape=(None,))

		my_gals = hf.create_group('my_gals')

		for i in range(0,np.shape(real_data_to_match)[0]):
			temp_gal_name = my_gals.create_dataset(names_in_gal_outputs[i], (0,), maxshape=(None,))

		data_type = h5py.special_dtype(vlen=str)
		my_directory = my_gals.create_dataset('my_directory', (0,), maxshape=(None,), dtype = data_type)

		my_subhalo = my_gals.create_dataset('my_subhalo', (0,), maxshape=(None,))
		my_gal_id = my_gals.create_dataset('my_gal_id', (0,), maxshape=(None,))


		my_keyword = my_gals.create_dataset('my_keyword', (0,), maxshape=(None,),dtype = data_type)

		return real_survey_radii
	
def populate_hdf5_with_matching_gals(cos_comparisons_file, directory_with_gal_folders, starting_gal_id, real_data_to_match, real_survey_radii, tols, hi_lo_tols, names_in_gal_outputs, which_survey, combined_plots_folder, spec_output_directory, cos_halos_bool, cos_dwarfs_bool, cos_gass_bool, cos_AGN_bool, cos_gto_bool):

	if directory_with_gal_folders[-1] != '/':
		directory_with_gal_folders += '/'
	output_files = glob.glob(directory_with_gal_folders + '*/output*.hdf5')

	sim_data_to_match = np.zeros([np.shape(real_data_to_match)[0],np.size(output_files)])

	files_looked_at = 0
	matched_gals = 0

	snap_directories = np.empty(np.size(output_files), dtype = object)
	group_numbers = np.zeros(np.size(output_files))
	keyword_ends = np.empty(np.size(output_files), dtype=object)

	smass_of_cos_gals_matched = np.array([])
	ssfr_of_cos_gals_matched = np.array([])

	where_matched_bools = [False]*np.shape(real_data_to_match)[1]

	real_smass_plot = real_data_to_match[0]
	real_ssfr_plot = real_data_to_match[1]

	for file in output_files:
		with h5py.File(file, 'r') as hf:
			GalaxyProperties = hf.get('GalaxyProperties')
			snap_directory = np.array(GalaxyProperties.get('snap_directory'))[0]
			group_number = np.array(GalaxyProperties.get('group_number'))[0]
			keyword_end = np.array(GalaxyProperties.get('file_keyword'))[0][-12::]
			halo_mass = np.array(GalaxyProperties.get('gal_mass'))[0]

			snap_directories[files_looked_at] = snap_directory
			group_numbers[files_looked_at] = group_number
			keyword_ends[files_looked_at] = keyword_end

			for i in range(0,np.shape(sim_data_to_match)[0]):
				sim_data_to_match[i][files_looked_at] = np.array(GalaxyProperties.get(names_in_gal_outputs[i]))[0]


		with h5py.File(cos_comparisons_file, 'r+') as hf:
			cos_gals = hf.get('cos_gals')
			matching_gals = cos_gals.get('matching_gals')

			my_gals = hf.get('my_gals')
			my_directory = my_gals.get('my_directory')
			my_subhalo = my_gals.get('my_subhalo')
			my_gal_id = my_gals.get('my_gal_id')
			my_keyword = my_gals.get('my_keyword')

			gals = np.shape(real_data_to_match)[1]

			for i in range(0,gals):
				successes = 0

				for j in range(0,np.shape(real_data_to_match)[0]):

					if ((real_data_to_match[1][i] < -11.4) or (real_data_to_match[1][i] > -9.4)):
						if ((real_data_to_match[j][i] > sim_data_to_match[j][files_looked_at]-hi_lo_tols[j]) & (real_data_to_match[j][i] < sim_data_to_match[j][files_looked_at] +hi_lo_tols[j])):
							successes += 1
							if successes == np.shape(real_data_to_match)[0]:
								where_matched_bools[i] = True
					else:	
						if ((real_data_to_match[j][i] > sim_data_to_match[j][files_looked_at]-tols[j]) & (real_data_to_match[j][i] < sim_data_to_match[j][files_looked_at] +tols[j])):
							successes += 1
							if successes == np.shape(real_data_to_match)[0]:
								where_matched_bools[i] = True

				if successes == np.shape(real_data_to_match)[0]:
					matched_gals += 1

					curr_directory = snap_directory
					curr_subhalo = group_number

					if (np.size(my_directory) == 0):
						with h5py.File(cos_comparisons_file, 'r+') as hf:
							my_gals = hf.get('my_gals')

							for m in range(0,np.shape(sim_data_to_match)[0]):
								temp_data_arrays_for_checks = my_gals.get(names_in_gal_outputs[m])
								temp_data_arrays_for_checks.resize(np.size(temp_data_arrays_for_checks)+1, axis=0)
								temp_data_arrays_for_checks[-1] = sim_data_to_match[m][files_looked_at]

						my_directory.resize(np.size(my_directory)+1, axis = 0)
						my_directory[-1] = snap_directory

						my_subhalo.resize(np.size(my_subhalo)+1, axis = 0)
						my_subhalo[-1] = group_number

						my_keyword.resize(np.size(my_keyword)+1, axis= 0)
						my_keyword[-1] = keyword_end

						my_gal_id.resize(np.size(my_gal_id)+1, axis=0)
						my_gal_id[-1] = starting_gal_id
						curr_gal_id = my_gal_id[-1]

						curr_gal = matching_gals.get('gal_'+str(i))
						curr_gal.resize(np.size(curr_gal)+1, axis=0)
						curr_gal[-1] = curr_gal_id

						my_directories = np.array(my_directory)
						my_gals_to_pass = np.array(my_gals)
						my_gal_ids = np.array(my_gal_id)
						my_subhalos = np.array(my_subhalo)

					else:
						if np.size(np.argwhere((curr_directory == np.array(my_directory))&(curr_subhalo == np.array(my_subhalo)))) != 0:
							index = np.argwhere((curr_directory == np.array(my_directory))&(curr_subhalo == np.array(my_subhalo)))[0][0]
							curr_gal_id = np.array(my_gal_id)[index]

						else:

							for m in range(0,np.shape(sim_data_to_match)[0]):
								temp_data_arrays_for_checks = my_gals.get(names_in_gal_outputs[m])
								temp_data_arrays_for_checks.resize(np.size(temp_data_arrays_for_checks)+1, axis=0)
								temp_data_arrays_for_checks[-1] = sim_data_to_match[m][files_looked_at]

							my_directory.resize(np.size(my_directory)+1, axis = 0)
							my_directory[-1] = snap_directory

							my_subhalo.resize(np.size(my_subhalo)+1, axis = 0)
							my_subhalo[-1] = group_number

							my_keyword.resize(np.size(my_keyword)+1, axis= 0)
							my_keyword[-1] = keyword_end

							my_gal_id.resize(np.size(my_gal_id)+1, axis=0)
							my_gal_id[-1] = my_gal_id[-2]+1
							curr_gal_id = my_gal_id[-1]

						curr_gal = matching_gals.get('gal_'+str(i))
						curr_gal.resize(np.size(curr_gal)+1, axis=0)
						curr_gal[-1] = curr_gal_id

						my_directories = np.array(my_directory)
						my_gals_to_pass = np.array(my_gals)
						my_gal_ids = np.array(my_gal_id)
						my_subhalos = np.array(my_subhalo)
					# break
			

		files_looked_at += 1

	where_matched_bools = np.asarray(where_matched_bools)
	where_matched_bools_unaltered = where_matched_bools

	real_smass_plot = real_smass_plot[real_ssfr_plot != 1000]
	where_matched_bools = where_matched_bools[real_ssfr_plot != 1000]
	which_survey = which_survey[real_ssfr_plot != 1000]
	real_ssfr_plot = real_ssfr_plot[real_ssfr_plot != 1000]

	cos_halos_smass_matched = real_smass_plot[(where_matched_bools==True) & (which_survey == 'halos')]
	cos_halos_ssfr_matched = real_ssfr_plot[(where_matched_bools==True) & (which_survey == 'halos')]
	cos_halos_smass = real_smass_plot[(which_survey == 'halos')]
	cos_halos_ssfr = real_ssfr_plot[(which_survey == 'halos')]

	cos_dwarfs_smass_matched = real_smass_plot[(where_matched_bools==True) & (which_survey == 'dwarfs')]
	cos_dwarfs_ssfr_matched = real_ssfr_plot[(where_matched_bools==True) & (which_survey == 'dwarfs')]
	cos_dwarfs_smass = real_smass_plot[(which_survey == 'dwarfs')]
	cos_dwarfs_ssfr = real_ssfr_plot[(which_survey == 'dwarfs')]

	cos_gto_smass_matched = real_smass_plot[(where_matched_bools==True) & (which_survey == 'gto')]
	cos_gto_ssfr_matched = real_ssfr_plot[(where_matched_bools==True) & (which_survey == 'gto')]
	cos_gto_smass = real_smass_plot[(which_survey == 'gto')]
	cos_gto_ssfr = real_ssfr_plot[(which_survey == 'gto')]

	cos_gass_smass_matched = real_smass_plot[(where_matched_bools==True) & (which_survey == 'gass')]
	cos_gass_ssfr_matched = real_ssfr_plot[(where_matched_bools==True) & (which_survey == 'gass')]
	cos_gass_smass = real_smass_plot[(which_survey == 'gass')]
	cos_gass_ssfr = real_ssfr_plot[(which_survey == 'gass')]

	cos_agn_smass_matched = real_smass_plot[(where_matched_bools==True) & (which_survey == 'agn')]
	cos_agn_ssfr_matched = real_ssfr_plot[(where_matched_bools==True) & (which_survey == 'agn')]
	cos_agn_smass = real_smass_plot[(which_survey == 'agn')]
	cos_agn_ssfr = real_ssfr_plot[(which_survey == 'agn')]

	if combined_plots_folder != None:
		os.chdir(combined_plots_folder)
	else:
		os.chdir(spec_output_directory)

	if cos_halos_bool:
		plt.plot(cos_halos_smass, cos_halos_ssfr,'b*', markersize=6)
		plt.hold(True)
		plt.plot(cos_halos_smass_matched, cos_halos_ssfr_matched,'b*', label = 'cos-Halos', zorder=5, markersize=9)

	if cos_dwarfs_bool:
		plt.plot(cos_dwarfs_smass, cos_dwarfs_ssfr,'r*', markersize=6)
		plt.plot(cos_dwarfs_smass_matched, cos_dwarfs_ssfr_matched,'r*', label = 'cos-Dwarfs', zorder=4, markersize=9)

	if cos_gto_bool:
		plt.plot(cos_gto_smass, cos_gto_ssfr,'k*', markersize=6)
		plt.plot(cos_gto_smass_matched, cos_gto_ssfr_matched,'k*', label = 'cos-GTO', zorder=3, markersize=9)

	if cos_gass_bool:
		plt.plot(cos_gass_smass, cos_gass_ssfr,'m*', markersize=6)
		plt.plot(cos_gass_smass_matched, cos_gass_ssfr_matched,'m*', label = 'cos-GASS', zorder=2, markersize=9)

	if cos_AGN_bool:
		plt.plot(cos_agn_smass, cos_agn_ssfr,'c*', markersize=6)
		plt.plot(cos_agn_smass_matched, cos_agn_ssfr_matched,'c*', label = 'cos-AGN', zorder=1, markersize=9)

	

	ax = plt.subplot(1,1,1)
	plt.scatter(sim_data_to_match[0][:], sim_data_to_match[1][:], c='#00FF00', label = 'EAGLE galaxies', zorder=10, s=20, edgecolor='black', linewidth=0.5)
	# plt.axhline(-11.0,c='k')
	# plt.axvline(9.7,c='k')
	handles, labels = ax.get_legend_handles_labels()
	handles = np.concatenate(([handles[-1]], handles[0:np.size(handles)-1]))
	labels = np.concatenate(([labels[-1]], labels[0:np.size(labels)-1]))
	l = plt.legend(handles, labels, fontsize=16, loc='lower left')
	l.set_zorder(20)
	plt.title('Stellar Mass vs SSFR Comparison', fontsize=18)
	plt.xlabel(r'$log_{10}\left(\frac{M_*}{M_{\odot}}\right)$', fontsize=16)
	plt.ylabel(r'$log_{10}(sSFR)$', fontsize=16)
	# plt.text(7.5, -9., 'Low Mass', fontsize = 16)
	# plt.text( 10., -9., 'Active', fontsize = 16)
	# plt.text( 9.75,-11.5, 'Passive', fontsize = 16)
	plt.tight_layout()
	plt.hold(False)
	plt.savefig('ssfr_vs_smass.pdf')
	plt.close()
 
	return where_matched_bools_unaltered


def run_specwizard_for_matching_gals(cos_comparisons_file, directory_with_gal_folders, spec_output_directory, path_to_param_template, path_to_specwizard_executable, path_to_cos_comparisons, real_survey_radii, cos_id_arr, semi_random_radius, realizations, AGN_bool, random_radii, min_radius=None,max_radius=None):
	if directory_with_gal_folders[-1] != '/':
		directory_with_gal_folders += '/'
	output_files = glob.glob(directory_with_gal_folders + '*/output*.hdf5')

	with h5py.File(cos_comparisons_file, 'r+') as hf:
		cos_gals = hf.get('cos_gals')
		matching_gals = cos_gals.get('matching_gals')

		my_gals = hf.get('my_gals')
		my_gal_id = np.array(my_gals.get('my_gal_id'))
		my_directory = np.array(my_gals.get('my_directory'))
		my_subhalo = np.array(my_gals.get('my_subhalo'))

		calls = 0
		gals_used = 0
		gals_matched_for_each_cos_gal = np.zeros(np.size(matching_gals.keys()))

		# ### Makes coordinate files
		# for n in range(0,realizations):

		# 	num_cos_gals = np.size(matching_gals.keys())
		# 	for i in range(0,num_cos_gals):
		# 		curr_gal =  np.array(matching_gals.get('gal_'+str(i)))
		# 		if n == 0:
		# 			gals_matched_for_each_cos_gal[i] = np.size(curr_gal)

		# 		if np.size(curr_gal) > 0:
		# 			gal_id = np.random.choice(np.array(curr_gal))
		# 			gal_index = np.argwhere(gal_id==np.array(my_gal_id))[0][0]
		# 			gal_directory = my_directory[gal_index]
		# 			gal_subhalo = my_subhalo[gal_index]

		# 			### we're going to see if it's somehow catching two galaxies occasionally
		# 			output_files_matched = 0
		# 			for j in range(0,np.size(output_files)):
		# 				with h5py.File(output_files[j], 'r') as hf1:
		# 					GalaxyProperties = hf1.get('GalaxyProperties')
		# 					temp_snap_directory = np.array(GalaxyProperties.get('snap_directory'))[0]
		# 					temp_subhalo_num = np.array(GalaxyProperties.get('group_number'))[0]

		# 					if ((temp_snap_directory==gal_directory) & (temp_subhalo_num==gal_subhalo)):
		# 						output_files_matched += 1
		# 						temp_smass = np.array(GalaxyProperties.get('gal_stellar_mass'))[0]
		# 						keyword_end = np.array(GalaxyProperties.get('file_keyword'))[0][-12::]
								

		# 						if AGN_bool:	
		# 							AGN_val = np.array(GalaxyProperties.get('AGN_lum'))[0][0]
		# 							if AGN_val > 0.1:
		# 								gal_folder = 'snapshot_'+keyword_end
		# 								particles_included_keyword = "snap_AGN" + str(AGN_val)[0:2] + str(AGN_val)[-1] + "_" + keyword_end
		# 								group_included_keyword = "group_tab_" + keyword_end
		# 								subfind_included_keyword = "eagle_subfind_tab_" + keyword_end
		# 							else:
		# 								gal_folder = 'snapshot_noneq_'+keyword_end
		# 								particles_included_keyword = 'snap_noneq_' + keyword_end
		# 								group_included_keyword = 'group_tab_' + keyword_end
		# 								subfind_included_keyword = 'eagle_subfind_tab_' + keyword_end

		# 						else:
		# 							gal_folder = 'snapshot_noneq_'+keyword_end
		# 							particles_included_keyword = 'snap_noneq_' + keyword_end
		# 							group_included_keyword = 'group_tab_' + keyword_end
		# 							subfind_included_keyword = 'eagle_subfind_tab_' + keyword_end

		# 						snap_base = 'snap_noneq_'+keyword_end
		# 						axis = np.random.choice(np.asarray([0,1,2]))
		# 						filename = 'los_'+str(int(gal_id))+'.txt'
		# 						gals_used += 1
		# 						subprocess.call('cp %s gal_output_%s.hdf5' % (str(output_files[j]), str(int(gal_id))), shell=True)

		# 						if os.path.isfile(filename):
		# 							SpecwizardFunctions.add_los_to_text(filename,gal_directory+'/',gal_subhalo, particles_included_keyword,
		# 								                                group_included_keyword, subfind_included_keyword, real_survey_radii[i], cos_id_arr[i], semi_random_radius, axis)
		# 						else:
		# 							if random_radii:
		# 								SpecwizardFunctions.create_one_los_text_random(filename,gal_directory+'/',gal_subhalo, particles_included_keyword,
		# 									                                group_included_keyword, subfind_included_keyword, cos_id_arr[i], min_radius,max_radius, axis)
		# 							else:
		# 								SpecwizardFunctions.create_one_los_text(filename,gal_directory+'/',gal_subhalo, particles_included_keyword,
		# 								                                group_included_keyword, subfind_included_keyword, real_survey_radii[i], cos_id_arr[i], semi_random_radius, axis)
		

		los_coord_files = glob.glob('los_*.txt')
		for i in range(0,np.size(los_coord_files)):
			with h5py.File(path_to_cos_comparisons, 'r+') as hf:
				my_gals = hf.get('my_gals')
				my_directory = my_gals.get('my_directory')
				my_gal_id = my_gals.get('my_gal_id')
				my_keyword = my_gals.get('my_keyword')
				my_group_numbers = my_gals.get('my_subhalo')

				curr_id = int(los_coord_files[i][4:-4])

				try: 
					index = np.argwhere(curr_id == np.array(my_gal_id))[0][0]
				except: 
					print 'could not match curr_id to gal_id'
					print 'curr_id = ' + str(curr_id)
					print 'gal_id = ' + str(np.array(my_gal_id))
					print ''
				gal_directory = np.asarray(my_directory)[index]
				gal_subhalo = np.asarray(my_group_numbers)[index]
				keyword_end = np.asarray(my_keyword)[index]
				gal_folder = 'snapshot_'+str(keyword_end)

				if AGN_bool:
					AGN_values = my_gals.get('AGN_lum')
					AGN_val = np.asarray(AGN_values)[index]
					if AGN_val > 0.1: 
						gal_folder = 'snapshot_'+str(keyword_end)
						snap_base = 'snap_AGN' + str(gal_directory)[-12:-9]  + '_' + str(keyword_end)
					else:
						gal_folder = 'snapshot_noneq_'+str(keyword_end)
						snap_base = 'snap_noneq_' + str(keyword_end)
				else:
					gal_folder = 'snapshot_noneq_'+str(keyword_end)
					snap_base = 'snap_noneq_' + str(keyword_end)
				gal_id = int(np.asarray(my_gal_id)[index])

				if os.path.isfile('spec.snap_%s.hdf5' % (str(gal_id))): # check if we already ran this sightline, means we can restart a crashed job
					continue

				for j in range(0,np.size(output_files)):
					found_file = False
					with h5py.File(output_files[j], 'r') as hf1:
						GalaxyProperties = hf1.get('GalaxyProperties')
						temp_snap_directory = np.array(GalaxyProperties.get('snap_directory'))[0]
						temp_subhalo_num = np.array(GalaxyProperties.get('group_number'))[0]

						if ((temp_snap_directory==gal_directory) & (temp_subhalo_num==gal_subhalo)):
							found_file = True
							temp_smass = np.array(GalaxyProperties.get('gal_stellar_mass'))[0]
					if found_file:
						break

				# dir_for_this_spec_run = 'specwizard_run_%s/' % (str(gal_id))
				# subprocess.call("mkdir %s" % (dir_for_this_spec_run), shell=True)

				# param_keywords = ['datadir','snap_base','los_coordinates_file', 'outputdir']
				param_keywords = ['datadir','snap_base','los_coordinates_file']
				param_replacements = [None]*np.size(param_keywords)
				param_replacements[0] = 'datadir = ' + gal_directory + '/' + gal_folder + '/'
				param_replacements[1] = 'snap_base = ' + snap_base
				param_replacements[2] = 'los_coordinates_file = ./'+str(los_coord_files[i])
				# param_replacements[3] = 'outputdir = %s' % (dir_for_this_spec_run) 

				EagleFunctions.edit_text(path_to_param_template, 'curr_params_%s.par' % (str(gal_id)), param_keywords, param_replacements)
				# subprocess.call("mpirun -np 2 %s curr_params_%s.par" % (path_to_specwizard_executable, str(gal_id)), shell=True)
				subprocess.call("%s curr_params_%s.par" % (path_to_specwizard_executable, str(gal_id)), shell=True)
				subprocess.call("mv spec.snap_noneq* spec.snap_%s.hdf5" % (str(gal_id)), shell=True)
				subprocess.call("mkdir particle_id_files_"+str(gal_id), shell=True)
				subprocess.call("mv eagle_particles_hit* particle_id_files_"+str(gal_id), shell=True)
				calls += 1
				# if calls >= 2:
					# raise ValueError('Stoped after two specwizard runs')

	print 'num of calls'
	print calls
	return calls, gals_matched_for_each_cos_gal


def get_EAGLE_data_for_plots(ion, rest_wavelength, cos_id_arr, lambda_line, spec_output_directory, lookup_file, max_smass, min_smass, max_ssfr, min_ssfr, tols, hi_lo_tols, ordered_cos_radii, covering_frac_bool, covering_frac_val, max_abs_vel, mass_estimates_bool, kinematics_bool, make_realistic_bool, offset = 0):
	masses = np.array([])
	smasses = np.array([])
	ssfr = np.array([])
	radii = np.array([])
	virial_radii = np.array([])
	R200 = np.array([])
	cols = np.array([])
	H_cols = np.array([])
	deltas = np.array([])
	equ_widths = np.array([])
	ion_densities = np.array([])
	gas_densities = np.array([])
	temperatures = np.array([])
	flux_for_stacks = []
	vel_for_stacks = []
	virial_vel_for_stacks = []
	num_minimas = []
	centroid_vels = []
	FWHMs = []
	depths = []
	temps = []
	eagle_ids = []
	escape_vels = []
	virial_radii_for_kin = []
	halo_masses_for_kin = []
	stellar_masses_for_kin = []
	ssfr_for_kin = []

	covered = 0
	total = 0
	directory_with_COS_LSF = '/gpfs/data/analyse/rhorton/opp_research/snapshots/'

	for folder in spec_output_directory:
		if folder[-1] != '/':
			folder += '/'

		los_files = glob.glob(folder + 'los*')
		buffer_size = len(folder)
		los_nums = np.zeros(np.size(los_files))
		for i in range(0,np.size(los_files)):
			los_nums[i] = los_files[i][int(buffer_size+4):-4]

		for i in range(0,int(1+np.max(los_nums))):	

			los_file = folder + 'los_'+str(i+offset)+'.txt'
			gal_output_file = folder + 'gal_output_'+str(i+offset)+'.hdf5'
			spec_output_file = folder + 'spec.snap_'+str(i+offset)+'.hdf5'
			if os.path.isfile(spec_output_file) == False:
				spec_output_file = folder + 'spec.snap_'+str(i+offset)+'.hdf5'
				if os.path.isfile(spec_output_file) == False:
					print spec_output_file
					print i + offset
					print 'no spec file for that number'
					continue


			with h5py.File(gal_output_file, 'r') as hf:
				galaxy_properties = hf.get("GalaxyProperties")
				gal_directory = np.array(galaxy_properties.get('snap_directory'))[0]
				file_keyword = np.array(galaxy_properties.get('file_keyword'))[0]
				gal_coords = np.array(galaxy_properties.get('gal_coords'))[0]
				box_size = np.array(galaxy_properties.get("box_size"))[0]
				gal_mass = np.array(galaxy_properties.get('gal_mass'))[0]
				gal_stellar_mass = np.array(galaxy_properties.get('gal_stellar_mass'))[0]
				gal_vel = np.array(galaxy_properties.get("gal_velocity"))[0]
				gal_R200 = np.array(galaxy_properties.get('gal_R200'))[0]
				gal_ssfr = np.array(galaxy_properties.get('log10_sSFR'))[0]

				virial_vel = (np.sqrt((G*gal_mass*sol_mass_to_g)/(gal_R200*parsec_to_cm*1.e3)))*1.e-5 # km/s
				escape_vel = np.sqrt(2.)*virial_vel 

			redshift = float(file_keyword[-7:-4]+'.'+file_keyword[-3::])
			lines = np.genfromtxt(los_file, skip_header=1)
			gal = np.array([gal_coords[0], gal_coords[1], gal_coords[2]])/box_size

			if np.size(lines) > 5:
				spec_num = 0
				for line in lines:
					radius = get_correct_radius_of_line(line,gal)*box_size

					gal_passes = False
					for id in cos_id_arr:
						if line[4] == id:
							gal_passes = True
							break

					if gal_passes == True:
						with h5py.File(spec_output_file,'r') as hf:
							spec_hubble_velocity = hf.get('VHubble_KMpS')
							delta_v = np.abs(spec_hubble_velocity[1]-spec_hubble_velocity[0])
							spectrum = hf.get('Spectrum'+str(spec_num))
							try:
								curr_ion = spectrum.get(ion)
							except:
								print spec_output_file
								print ion
								raise ValueError('missing a spectrum?')
							optical_depth = np.array(curr_ion.get('OpticalDepth'))
							col_dense = np.array(curr_ion.get('LogTotalIonColumnDensity'))
							flux = np.array(curr_ion.get('Flux'))
							equ_width = np.sum(1.-flux)*(delta_v/c_kms)*lambda_line
							optical_depth_weighted = curr_ion.get('RedshiftSpaceOpticalDepthWeighted')
							temperature = np.array(optical_depth_weighted.get('Temperature_K'))
							density = np.array(optical_depth_weighted.get('NIon_CM3'))
							overdensity = np.array(optical_depth_weighted.get('OverDensity'))

							gas_density = overdensity*((rho_bar_norm*h**2.*(1.+redshift)**3.*x_H*omega_b)/(m_H))
							
							total += 1
							if covering_frac_bool:
								if col_dense > covering_frac_val:
									covered += 1

							length_spectra = np.size(spec_hubble_velocity)
							max_box_vel = spec_hubble_velocity[-1]*(length_spectra+1)/(length_spectra)
							H = max_box_vel/(box_size/1.e3) # stuff with H in Mpc
							gal_hubble_vel = (gal_coords[2]/1.e3)*H # switched gal coords to Mpc
							gal_vel_z = gal_vel[2]

							spec_hubble_velocity = spec_hubble_velocity - (max_box_vel/2.0) # add in peculiar velocity 
							spec_hubble_velocity = np.where(spec_hubble_velocity > max_box_vel/2.0, spec_hubble_velocity-max_box_vel, spec_hubble_velocity)
							spec_hubble_velocity = np.where(spec_hubble_velocity < (-1.0)*max_box_vel/2.0, spec_hubble_velocity+max_box_vel, spec_hubble_velocity)

							# ### if makings fits outputs for Jess
							# if total < 100:
							# 	fits_arrays = [spec_hubble_velocity[np.abs(spec_hubble_velocity) < max_abs_vel], flux[np.abs(spec_hubble_velocity) < max_abs_vel]]
							# 	fits_file_name = 'spectra_%d.fits' % (total)
							# 	make_fits_file_for_flux(fits_arrays, fits_file_name)

							# 	plt.plot(spec_hubble_velocity[np.abs(spec_hubble_velocity) <= max_abs_vel], flux[np.abs(spec_hubble_velocity) < max_abs_vel])
							# 	plt.xlabel('Velocity (km/s)')
							# 	plt.ylabel('Flux')
							# 	plt.legend(loc='lower right')
							# 	plt.savefig('sing_lines_%d_%d.pdf' % (i+offset, spec_num))
							# 	plt.close()
							# else:
							# 	raise ValueError('Stopped After 100')

							if mass_estimates_bool:
								H_col, delta = mass_estimates(lookup_file, spec_output_file, 'Spectrum'+str(spec_num), ion, redshift)

							if kinematics_bool:
								# ### if want to make .txt output to do a full plot sequence of the LSF convolution later
								# if ((i+offset == 9) & (spec_num == 0)):
								# 	SpecwizardFunctions.make_txt_output(spec_hubble_velocity[np.abs(spec_hubble_velocity)<= max_abs_vel], flux[np.abs(spec_hubble_velocity) <= max_abs_vel], optical_depth[np.abs(spec_hubble_velocity)<=max_abs_vel], gal_directory, col_dense, 'Spectrum'+str(spec_num), ion, radius, gal_R200, gal_mass)
								# 	raise ValueError('it worked? Check file')
								num_minima, centroid_vel, FWHM, depth, temp = get_line_kinematics(flux[np.abs(spec_hubble_velocity) <= max_abs_vel], spec_hubble_velocity[np.abs(spec_hubble_velocity)<= max_abs_vel], temperature[np.abs(spec_hubble_velocity)<=max_abs_vel], optical_depth[np.abs(spec_hubble_velocity)<=max_abs_vel], i+offset, spec_num, radius, gal_mass, gal_ssfr, make_realistic_bool=make_realistic_bool, rest_wavelength=rest_wavelength, redshift=redshift, directory_with_COS_LSF=directory_with_COS_LSF)
								if np.size(FWHM) != np.size(centroid_vel):
									print spec_output_file
									print ''
									print centroid_vel
									print np.shape(centroid_vel)
									print ''
									print FWHM
									print np.shape(FWHM)
									print ''
									raise ValueError('some kinematic arrays are differe sizes!')
								curr_escape_vel = np.zeros(np.size(centroid_vel)) + escape_vel
								curr_virial_radii = np.zeros(np.size(centroid_vel)) + gal_R200
								curr_halos_masses = np.zeros(np.size(centroid_vel)) + gal_mass
								curr_ssfr = np.zeros(np.size(centroid_vel)) + gal_ssfr
								curr_stellar_mass = np.zeros(np.size(centroid_vel)) + gal_stellar_mass

							try:
								ssfr = np.concatenate((ssfr,np.array(gal_ssfr)))
							except:
								ssfr = np.concatenate((ssfr,np.array([gal_ssfr])))

							try:
								masses = np.concatenate((masses,np.array(gal_mass)))
							except:
								masses = np.concatenate((masses,np.array([gal_mass])))

							try:
								smasses = np.concatenate((smasses,np.array(gal_stellar_mass)))
							except:
								smasses = np.concatenate((smasses,np.array([gal_stellar_mass])))

							try:
								radii = np.concatenate((radii,np.array(radius)))
							except:
								radii = np.concatenate((radii,np.array([radius])))

							try:
								virial_radii = np.concatenate((virial_radii,np.array(radius/gal_R200)))
							except:
								virial_radii = np.concatenate((virial_radii,np.array([radius/gal_R200])))

							try:
								R200 = np.concatenate((R200,np.array(gal_R200)))
							except:
								R200 = np.concatenate((R200,np.array([gal_R200])))

							try:
								equ_widths = np.concatenate((equ_widths,np.array(equ_width)))
							except:
								equ_widths = np.concatenate((equ_widths,np.array([equ_width])))

							try:
								cols = np.concatenate((cols,np.array(col_dense)))
							except:
								cols = np.concatenate((cols,np.array([col_dense])))

							eagle_ids.append(line[4])

							if mass_estimates_bool:
								try:
									H_cols = np.concatenate((H_cols,np.array(H_col)))
								except:
									H_cols = np.concatenate((H_cols,np.array([H_col])))

								try:
									deltas = np.concatenate((deltas,np.array(delta)))
								except:
									deltas = np.concatenate((deltas,np.array([delta])))

								temp_density = density[((np.abs(spec_hubble_velocity) <= max_abs_vel) & (optical_depth >= 0.01))]
								temp_optical_depth = optical_depth[((np.abs(spec_hubble_velocity) <= max_abs_vel) & (optical_depth >= 0.01))]
								weighted_density = np.sum(temp_density*temp_optical_depth)/np.sum(temp_optical_depth)
								try:
									ion_densities = np.concatenate((ion_densities,weighted_density))
								except:
									ion_densities = np.concatenate((ion_densities,np.array([weighted_density])))

								temp_gas_density = gas_density[((np.abs(spec_hubble_velocity) <= max_abs_vel) & (optical_depth >= 0.01))]
								temp_optical_depth = optical_depth[((np.abs(spec_hubble_velocity) <= max_abs_vel) & (optical_depth >= 0.01))]
								weighted_gas_density = np.sum(temp_gas_density*temp_optical_depth)/np.sum(temp_optical_depth)
								try:
									gas_densities = np.concatenate((gas_densities,weighted_gas_density))
								except:
									gas_densities = np.concatenate((gas_densities,np.array([weighted_gas_density])))

								temp_temperature = temperature[((np.abs(spec_hubble_velocity) <= max_abs_vel) & (optical_depth >= 0.01))]
								weighted_temperature = np.sum(temp_temperature*temp_optical_depth)/np.sum(temp_optical_depth)
								try:
									temperatures = np.concatenate((temperatures,weighted_temperature))
								except:
									temperatures = np.concatenate((temperatures,np.array([weighted_temperature])))

							if kinematics_bool:
								num_minimas.append(num_minima)
								centroid_vels.append(centroid_vel)
								FWHMs.append(FWHM)
								depths.append(depth)
								temps.append(temp)
								escape_vels.append(curr_escape_vel)
								virial_radii_for_kin.append(curr_virial_radii)
								halo_masses_for_kin.append(curr_halos_masses)
								stellar_masses_for_kin.append(curr_stellar_mass)
								ssfr_for_kin.append(curr_ssfr)


							flux_for_stacks.append(flux[np.abs(spec_hubble_velocity) <= max_abs_vel])

							vel_for_stacks.append(spec_hubble_velocity[np.abs(spec_hubble_velocity) <= max_abs_vel])

							virial_vel_for_stacks.append(spec_hubble_velocity[np.abs(spec_hubble_velocity) <= max_abs_vel]/virial_vel)

					spec_num += 1

			else:
				spec_num = 0
				radius = get_correct_radius_of_line(lines,gal)*box_size

				gal_passes = False

				for id in cos_id_arr:
					if lines[4] == id:
						gal_passes = True
						break
				
				if gal_passes == True:

					with h5py.File(spec_output_file,'r') as hf:
						spec_hubble_velocity = hf.get('VHubble_KMpS')
						delta_v = np.abs(spec_hubble_velocity[1]-spec_hubble_velocity[0])
						spectrum = hf.get('Spectrum'+str(spec_num))
						curr_ion = spectrum.get(ion)
						optical_depth = np.array(curr_ion.get('OpticalDepth'))
						col_dense = np.array(curr_ion.get('LogTotalIonColumnDensity'))
						flux = np.array(curr_ion.get('Flux'))
						equ_width = np.sum(1.-flux)*(delta_v/c_kms)*lambda_line
						optical_depth_weighted = curr_ion.get('RedshiftSpaceOpticalDepthWeighted')
						temperature = np.array(optical_depth_weighted.get('Temperature_K'))
						density = np.array(optical_depth_weighted.get('NIon_CM3'))
						overdensity = np.array(optical_depth_weighted.get('OverDensity'))

						gas_density = overdensity*((rho_bar_norm*h**2.*(1.+redshift)**3.*x_H*omega_b)/(m_H))

						total += 1
						if covering_frac_bool:
							if col_dense > covering_frac_val:
								covered += 1

						length_spectra = np.size(spec_hubble_velocity)
						max_box_vel = spec_hubble_velocity[-1]*(length_spectra+1)/(length_spectra)
						H = max_box_vel/(box_size/1.e3) # stuff with H in Mpc
						gal_hubble_vel = (gal_coords[2]/1.e3)*H # switched gal coords to Mpc
						gal_vel_z = gal_vel[2]

						spec_hubble_velocity = spec_hubble_velocity - (max_box_vel/2.0) # add in peculiar velocity 
						spec_hubble_velocity = np.where(spec_hubble_velocity > max_box_vel/2.0, spec_hubble_velocity-max_box_vel, spec_hubble_velocity)
						spec_hubble_velocity = np.where(spec_hubble_velocity < (-1.0)*max_box_vel/2.0, spec_hubble_velocity+max_box_vel, spec_hubble_velocity)

						# ### if makings fits outputs for Jess
						# if total < 100:
						# 	fits_arrays = [spec_hubble_velocity[np.abs(spec_hubble_velocity) < max_abs_vel], flux[np.abs(spec_hubble_velocity) < max_abs_vel]]
						# 	fits_file_name = 'spectra_%d.fits' % (total)
						# 	make_fits_file_for_flux(fits_arrays, fits_file_name)

						# 	plt.plot(spec_hubble_velocity[np.abs(spec_hubble_velocity) <= max_abs_vel], flux[np.abs(spec_hubble_velocity) < max_abs_vel])
						# 	plt.xlabel('Velocity (km/s)')
						# 	plt.ylabel('Flux')
						# 	plt.legend(loc='lower right')
						# 	plt.savefig('sing_lines_%d_%d.pdf' % (i+offset, spec_num))
						# 	plt.close()
						# else:
						# 	raise ValueError('Stopped After 100')

						if mass_estimates_bool:
							H_col, delta = mass_estimates(lookup_file, spec_output_file, 'Spectrum'+str(spec_num), ion, redshift)

						if kinematics_bool:
							# ### if want to make .txt output to do a full plot sequence of the LSF convolution later
							# if ((i+offset == 9) & (spec_num == 0)):
							# 	SpecwizardFunctions.make_txt_output(spec_hubble_velocity[np.abs(spec_hubble_velocity)<= max_abs_vel], flux[np.abs(spec_hubble_velocity) <= max_abs_vel], optical_depth[np.abs(spec_hubble_velocity)<=max_abs_vel], gal_directory, col_dense, 'Spectrum'+str(spec_num), ion, radius, gal_R200, gal_mass)
							# 	raise ValueError('it worked? Check file')
							num_minima, centroid_vel, FWHM, depth, temp = get_line_kinematics(flux[np.abs(spec_hubble_velocity) <= max_abs_vel], spec_hubble_velocity[np.abs(spec_hubble_velocity)<= max_abs_vel], temperature[np.abs(spec_hubble_velocity)<=max_abs_vel], optical_depth[np.abs(spec_hubble_velocity)<=max_abs_vel], i+offset, spec_num, radius, gal_mass, gal_ssfr, make_realistic_bool=make_realistic_bool, rest_wavelength=rest_wavelength, redshift=redshift, directory_with_COS_LSF=directory_with_COS_LSF)
							if np.size(FWHM) != np.size(centroid_vel):
								print spec_output_file
								print ''
								print centroid_vel
								print np.shape(centroid_vel)
								print ''
								print FWHM
								print np.shape(FWHM)
								print ''
								raise ValueError('some kinematic arrays are differe sizes!')
							curr_escape_vel = np.zeros(np.size(centroid_vel)) + escape_vel
							curr_virial_radii = np.zeros(np.size(centroid_vel)) + gal_R200
							curr_halos_masses = np.zeros(np.size(centroid_vel)) + gal_mass
							curr_ssfr = np.zeros(np.size(centroid_vel)) + gal_ssfr
							curr_stellar_mass = np.zeros(np.size(centroid_vel)) + gal_stellar_mass

						try:
							ssfr = np.concatenate((ssfr,np.array(gal_ssfr)))
						except:
							ssfr = np.concatenate((ssfr,np.array([gal_ssfr])))

						try:
							masses = np.concatenate((masses,np.array(gal_mass)))
						except:
							masses = np.concatenate((masses,np.array([gal_mass])))

						try:
							smasses = np.concatenate((smasses,np.array(gal_stellar_mass)))
						except:
							smasses = np.concatenate((smasses,np.array([gal_stellar_mass])))

						try:
							radii = np.concatenate((radii,np.array(radius)))
						except:
							radii = np.concatenate((radii,np.array([radius])))

						try:
							virial_radii = np.concatenate((virial_radii,np.array(radius/gal_R200)))
						except:
							virial_radii = np.concatenate((virial_radii,np.array([radius/gal_R200])))

						try:
							R200 = np.concatenate((R200,np.array(gal_R200)))
						except:
							R200 = np.concatenate((R200,np.array([gal_R200])))

						try:
							equ_widths = np.concatenate((equ_widths,np.array(equ_width)))
						except:
							equ_widths = np.concatenate((equ_widths,np.array([equ_width])))

						try:
							cols = np.concatenate((cols,np.array(col_dense)))
						except:
							cols = np.concatenate((cols,np.array([col_dense])))

						eagle_ids.append(lines[4])

						if mass_estimates_bool:
							try:
								H_cols = np.concatenate((H_cols,np.array(H_col)))
							except:
								H_cols = np.concatenate((H_cols,np.array([H_col])))

							try:
								deltas = np.concatenate((deltas,np.array(delta)))
							except:
								deltas = np.concatenate((deltas,np.array([delta])))

							temp_density = density[((np.abs(spec_hubble_velocity) <= max_abs_vel) & (optical_depth >= 0.01))]
							temp_optical_depth = optical_depth[((np.abs(spec_hubble_velocity) <= max_abs_vel) & (optical_depth >= 0.01))]
							weighted_density = np.sum(temp_density*temp_optical_depth)/np.sum(temp_optical_depth)
							try:
								ion_densities = np.concatenate((ion_densities,weighted_density))
							except:
								ion_densities = np.concatenate((ion_densities,np.array([weighted_density])))

							temp_gas_density = gas_density[((np.abs(spec_hubble_velocity) <= max_abs_vel) & (optical_depth >= 0.01))]
							temp_optical_depth = optical_depth[((np.abs(spec_hubble_velocity) <= max_abs_vel) & (optical_depth >= 0.01))]
							weighted_gas_density = np.sum(temp_gas_density*temp_optical_depth)/np.sum(temp_optical_depth)
							try:
								gas_densities = np.concatenate((gas_densities,weighted_gas_density))
							except:
								gas_densities = np.concatenate((gas_densities,np.array([weighted_gas_density])))

							temp_temperature = temperature[((np.abs(spec_hubble_velocity) <= max_abs_vel) & (optical_depth >= 0.01))]
							weighted_temperature = np.sum(temp_temperature*temp_optical_depth)/np.sum(temp_optical_depth)
							try:
								temperatures = np.concatenate((temperatures,weighted_temperature))
							except:
								temperatures = np.concatenate((temperatures,np.array([weighted_temperature])))

						if kinematics_bool:
							num_minimas.append(num_minima)
							centroid_vels.append(centroid_vel)
							FWHMs.append(FWHM)
							depths.append(depth)
							temps.append(temp)
							escape_vels.append(curr_escape_vel)
							virial_radii_for_kin.append(curr_virial_radii)
							halo_masses_for_kin.append(curr_halos_masses)
							stellar_masses_for_kin.append(curr_stellar_mass)
							ssfr_for_kin.append(curr_ssfr)

						flux_for_stacks.append(flux[np.abs(spec_hubble_velocity) <= max_abs_vel])

						vel_for_stacks.append(spec_hubble_velocity[np.abs(spec_hubble_velocity) <= max_abs_vel])

						virial_vel_for_stacks.append(spec_hubble_velocity[np.abs(spec_hubble_velocity) <= max_abs_vel]/virial_vel)

	if kinematics_bool:
		return covered, total, ssfr, masses, smasses, radii, virial_radii, R200, cols, equ_widths, eagle_ids, flux_for_stacks, vel_for_stacks, virial_vel_for_stacks, cols, H_cols, deltas, np.hstack(num_minimas), np.hstack(depths), np.hstack(FWHMs), np.hstack(centroid_vels), np.hstack(temps), np.hstack(escape_vels), np.hstack(virial_radii_for_kin), np.hstack(halo_masses_for_kin), np.hstack(stellar_masses_for_kin), np.hstack(ssfr_for_kin), ion_densities, temperatures, gas_densities
	else:
		return covered, total, ssfr, masses, smasses, radii, virial_radii, R200, cols, equ_widths, eagle_ids, flux_for_stacks, vel_for_stacks, virial_vel_for_stacks, cols, H_cols, deltas, num_minimas, depths, FWHMs, centroid_vels, temps, escape_vels, virial_radii_for_kin, halo_masses_for_kin, stellar_masses_for_kin, ssfr_for_kin, ion_densities, temperatures, gas_densities

def make_col_dense_plots(ion, covered, total, ssfr, masses, smasses, radii, virial_radii, cols, eagle_ids, cos_id_arr, plot_cols, plot_cols_err, plot_cols_flags, plot_cols_radii, covering_frac_val, colorbar,bins_for_median):
	one_sig_top = []
	one_sig_bot = []
	two_sig_top = []
	two_sig_bot = []
	median = []
	plot_radii = []
	plot_radii_err = []
	cos_percentiles = []

	eagle_ids = np.array(eagle_ids)

	# because not all surveys have column densities
	cos_id_arr = cos_id_arr[plot_cols != 1.]
	plot_cols_radii = plot_cols_radii[plot_cols != 1.]
	plot_cols = plot_cols[plot_cols != 1.]
	### Errors and percentiles for EAGLE as well as finding out how much of an outlier each COS data point is. 

	### temporory until columns are obtained for all surveys
	temp_eagle_ids = []
	temp_radii = []
	temp_virial_radii = []
	temp_cols = []
	temp_masses = []
	for i in range(0,np.size(eagle_ids)):
		if np.size(cos_id_arr[cos_id_arr == eagle_ids[i]]) > 0:
			temp_eagle_ids.append(eagle_ids[i])
			temp_radii.append(radii[i])
			temp_virial_radii.append(virial_radii[i])
			temp_cols.append(cols[i])
			temp_masses.append(masses[i])

	eagle_ids = np.array(temp_eagle_ids)
	radii = np.array(temp_radii)
	virial_radii = np.array(temp_virial_radii)
	cols = np.array(temp_cols)
	masses = np.array(temp_masses)

	### end temp part
	for i in range(0,np.size(cos_id_arr)):
		curr_cols = cols[eagle_ids == cos_id_arr[i]]
		curr_radii = radii[eagle_ids == cos_id_arr[i]]
		if np.size(curr_cols) != 0:
			cos_percentiles.append(100.*float(np.size(curr_cols[curr_cols <= plot_cols[i]]))/np.size(curr_cols))
			one_sig_top.append(np.percentile(curr_cols,84))
			one_sig_bot.append(np.percentile(curr_cols, 16))
			two_sig_top.append(np.percentile(curr_cols, 95))
			two_sig_bot.append(np.percentile(curr_cols, 5))
			median.append(np.percentile(curr_cols, 50))
			plot_radii.append(np.mean(curr_radii))
			plot_radii_err.append(np.std(curr_radii, ddof=1))

	one_sig_bot = np.array(one_sig_bot)
	one_sig_top = np.array(one_sig_top)
	two_sig_bot = np.array(two_sig_bot)
	two_sig_top = np.array(two_sig_top)
	median = np.array(median)

	cos_percentiles = np.hstack(cos_percentiles)
	low_frac = float(np.size(cos_percentiles[cos_percentiles < 5.]))/np.size(cos_percentiles)
	high_frac = float(np.size(cos_percentiles[cos_percentiles > 95.]))/np.size(cos_percentiles)

	### Linear fits for EAGLE and COS
	eagle_params, eagle_errs = np.polyfit(plot_cols_radii, median, 1 ,cov=True)
	eagle_fit_arr = eagle_params[0]*plot_cols_radii+eagle_params[1]

	cos_params, cos_errs = np.polyfit(plot_cols_radii, plot_cols, 1, cov=True)
	cos_fit_arr = cos_params[0]*plot_cols_radii + cos_params[1]

	if np.size(plot_cols) > 0:
		obs_radius_max, obs_value_max, sim_radius_max, sim_value_max, obs_quad, sim_quad, delta, reject_val = KS_2D(plot_cols_radii, plot_cols, radii, cols)	

		try:
			with open(str(ion)+'_KS_col_%.1f.txt' % (reject_val), 'w') as file:
				file.write('reject_val delta obs_radius_max obs_value_max sim_radius_max sim_value_max obs_quad sim_quad\n')
				file.write('%.1f %.3f %.2f %.2f %.2f %.2f %s %s' % (reject_val, delta, obs_radius_max, obs_value_max, sim_radius_max, sim_value_max, obs_quad, sim_quad))
				file.close()
		except:
			with open(str(ion)+'_KS_col_%s.txt' % ('not_rejectred'), 'w') as file:
				file.write('reject_val delta obs_radius_max obs_value_max sim_radius_max sim_value_max obs_quad sim_quad\n')
				file.write('%s %.3f %.2f %.2f %.2f %.2f %s %s' % (reject_val, delta, obs_radius_max, obs_value_max, sim_radius_max, sim_value_max, obs_quad, sim_quad))
				file.close()

		# ### scatter/single rel block
		# if colorbar == 'smass':
		# 	plt.scatter(radii, cols, s=20., label = 'EAGLE, %.2f' % (delta), c=np.log10(smasses), cmap='RdBu_r', edgecolor = 'black', linewidth = 0.5)
		# 	cb = plt.colorbar()
		# 	cb.set_label(r'$log_{10}(M_{star})$')
		# elcolorbar == 'hmass':
		# 	plt.scatter(radii, cols, s=20., label = 'EAGLE, %.2f' % (delta), c=np.log10(masses), cmap='RdBu_r', edgecolor = 'black', linewidth = 0.5)
		# 	cb = plt.colorbar()
		# 	cb.set_label(r'$log_{10}(M_{halo})$')
		# elif colorbar == 'ssfr':
		# 	plt.scatter(radii, cols, c=ssfr, s=20., cmap='RdBu', edgecolor = 'black', linewidth = 0.5, label = 'EAGLE, %.2f' % (delta))
		# 	cb = plt.colorbar()
		# 	cb.set_label(r'$log_{10}(sSFR)$')
		# plt.hold(True)

		### err bars/multi rel block
		bins = []
		indices = np.argsort(plot_cols_radii)
		bins_radii_arr = plot_cols_radii[indices]
		for i in range(0,np.size(plot_cols_radii)):
			if i == 0:
				bins.append(int(round(bins_radii_arr[i]))-0.5)
			elif int(round(bins_radii_arr[i])) != int(round(bins_radii_arr[i-1])):
				bins.append(int(round(bins_radii_arr[i]))-0.5)
		bins.append(int(round(bins_radii_arr[-1]))+0.5)
		plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, radii, cols)

		plt.errorbar(plot_radii, median, yerr=[median-two_sig_bot, two_sig_top-median], color='k', fmt = '.', ecolor = 'r', label='EAGLE')
		plt.hold(True)
		plt.errorbar(plot_radii, median, yerr=[median-one_sig_bot, one_sig_top-median], xerr = plot_radii_err, color='k', fmt = '.', ecolor = 'b')

		### scatter/single rel block
		bins = np.linspace(np.min(radii), np.max(radii),bins_for_median)
		plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, radii, cols)
		plt.plot(plot_x, plot_median, color = 'k', linestyle = '-')
		plt.plot(plot_x, plot_84,  color = 'k', linestyle = '-.')
		plt.plot(plot_x, plot_16,  color = 'k', linestyle = '-.')

		bins = np.linspace(np.min(plot_cols_radii), np.max(plot_cols_radii),bins_for_median)# 
		plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, plot_cols_radii, plot_cols)
		plt.plot(plot_x, plot_median,c='#00FF00', linestyle = '-')
		plt.plot(plot_x, plot_84, c='#00FF00', linestyle = '-.')
		plt.plot(plot_x, plot_16, c='#00FF00', linestyle = '-.')

		# ### KS test stuff
		# plt.plot(obs_radius_max, obs_value_max, 'k*', markersize = 15., label = obs_quad)
		# plt.plot(sim_radius_max, sim_value_max, 'k.', markersize = 15., label = sim_quad)

		### Linear fit stuff
		plt.plot(plot_cols_radii, eagle_fit_arr, color='k', label=r'm=%.2e $\pm$ %.0e' % (eagle_params[0], np.sqrt(eagle_errs[0,0])) + '\n' + r'b=%.2e $\pm$ %.0e' % (np.power(10,eagle_params[1]), np.sqrt(np.power(10,eagle_errs[1,1]))))
		plt.plot(plot_cols_radii, cos_fit_arr, color='#00FF00', label=r'm=%.2e $\pm$ %.0e' % (cos_params[0], cos_errs[0,0]) + '\n' + r'b=%.2e $\pm$ %.0e' % (np.power(10,cos_params[1]), np.sqrt(np.power(10,cos_errs[1,1]))))

		### COS data

		norm_radii = plot_cols_radii[plot_cols_flags == 1.]
		norm_cols = plot_cols[plot_cols_flags == 1.]
		norm_err = plot_cols_err[plot_cols_flags == 1.]
		plt.errorbar(norm_radii, norm_cols, yerr=norm_err, fmt='*', c='#00FF00', markersize = 8.5, markeredgecolor = 'k', markeredgewidth = 0.5, label = 'COS: high=%.2f, low=%.2f' % (high_frac, low_frac))
		
		upper_lim_cols = plot_cols[plot_cols_flags == 3]
		upper_lim_radii = plot_cols_radii[plot_cols_flags == 3]
		lower_lim_cols = plot_cols[plot_cols_flags == 2]
		lower_lim_radii = plot_cols_radii[plot_cols_flags == 2]
		plt.plot(upper_lim_radii, upper_lim_cols, marker='v', c='#00FF00', linestyle='None', markeredgecolor = 'k', markeredgewidth = 0.5, markersize=8.5)
		plt.plot(lower_lim_radii, lower_lim_cols, marker='^', c='#00FF00', linestyle='None', markeredgecolor = 'k', markeredgewidth = 0.5, markersize=8.5)
		plt.hold(False)
		plt.title('Passive: N vs b for %s' % (ion), fontsize=16.)

	else:
		if colorbar == 'smass':
			plt.scatter(radii, cols, s=20., c=np.log10(smasses), cmap='RdBu_r', edgecolor = 'black', linewidth = 0.5, label = 'EAGLE', fontsize=16.)
			cb = plt.colorbar()
			cb.set_label(r'$log_{10}(M_{star})$', fontsize=16.)
		elif colorbar == 'hmass':
			plt.scatter(radii, cols, s=20., c=np.log10(masses), cmap='RdBu_r', edgecolor = 'black', linewidth = 0.5, label = 'EAGLE', fontsize=16.)
			cb = plt.colorbar()
			cb.set_label(r'$log_{10}(M_{halo})$', fontsize=16.)
		elif colorbar == 'ssfr':
			plt.scatter(radii, cols, c=ssfr, s=20., cmap='RdBu', edgecolor = 'black', linewidth = 0.5, label = 'EAGLE', fontsize=16.)
			cb = plt.colorbar()
			cb.set_label(r'$log_{10}(sSFR)$', fontsize=16.)
		if np.size(plot_cols) > 0:
			plt.hold(True)
			plt.plot(plot_cols_radii,plot_cols, '*', c='#00FF00', markersize = 8.5, markeredgecolor = 'k', markeredgewidth = 0.5, label = 'COS')
			plt.hold(False)

		bins = np.linspace(np.min(x_arr), np.max(x_arr),bins_for_median)
		plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, radii, cols)

		plt.hold(True)
		plt.plot(obs_radius_max, obs_value_max, 'k*', markersize = 15., label = obs_quad)
		plt.plot(sim_radius_max, sim_value_max, 'k.', markersize = 15., label = sim_quad)
		# plt.plot(plot_x, plot_median,'k')
		# plt.plot(plot_x, plot_68, 'g')
		# plt.plot(plot_x, plot_32, 'g')
		# plt.plot(plot_x, plot_90, 'r')
		# plt.plot(plot_x, plot_10, 'r')
		plt.hold(False)

		plt.title('Passive: W vs b for %s' % (ion), fontsize=16.)

	plt.xlabel('Impact Parameter (kpc)', fontsize=16.)
	plt.ylabel(r'$log_{10}(N_{%s}) cm^{-2}$' % (ion), fontsize=16.)
	plt.legend(loc = 'upper right', fontsize=12.)
	plt.grid()
	if colorbar == 'smass':
		plt.savefig(ion+'col_with_color_smass.pdf')
	elif colorbar == 'hmass':
		plt.savefig(ion+'col_with_color_hmass.pdf')
	elif colorbar == 'ssfr':
		plt.savefig(ion+'col_with_color_ssfr.pdf')
	plt.close()


	if colorbar == 'smass':
		fig = plt.scatter(virial_radii, cols, s=20., c=np.log10(smasses), cmap='RdBu_r', edgecolor = 'black', linewidth = 0.5)
		cb = plt.colorbar()
		cb.set_label(r'$log_{10}(M_{star})$', fontsize=16.)
	elif colorbar == 'hmass':
		fig = plt.scatter(virial_radii, cols, s=20., c=np.log10(masses), cmap='RdBu_r', edgecolor = 'black', linewidth = 0.5)
		cb = plt.colorbar()
		cb.set_label(r'$log_{10}(M_{halo})$', fontsize=16.)		
	elif colorbar == 'ssfr':
		plt.scatter(virial_radii, cols, c=ssfr, s=20., cmap='RdBu', edgecolor = 'black', linewidth = 0.5)
		cb = plt.colorbar()
		cb.set_label(r'$log_{10}(sSFR)$', fontsize=16.)

	bins = np.linspace(np.min(virial_radii), np.max(virial_radii),bins_for_median)
	plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, virial_radii, cols)
	plt.hold(True)
	plt.plot(plot_x, plot_median,'k')
	plt.plot(plot_x, plot_84, 'g')
	plt.plot(plot_x, plot_16, 'g')
	plt.plot(plot_x, plot_95, 'r')
	plt.plot(plot_x, plot_5, 'r')
	plt.hold(False)

	try:
		plt.title('Passive: N vs b for %s' % (ion), fontsize=18.)
	except:
		plt.title('Passive: N vs b for %s' % (ion), fontsize=18.)
	plt.xlabel('Impact Parameter (in virial radii)', fontsize=16.)
	plt.ylabel(r'$log_{10}(N_{%s}) cm^{-2}$' % (ion), fontsize=16.)
	plt.grid()
	if colorbar == 'smass':
		plt.savefig(ion+'col_virial_with_color_smass.pdf')
	elif colorbar == 'hmass':
		plt.savefig(ion+'col_virial_with_color_hmass.pdf')
	elif colorbar == 'ssfr':
		plt.savefig(ion+'col_virial_with_color_ssfr.pdf')

	plt.close()

	# if np.shape(plot_cols) != np.shape(cols):
	# 	print np.shape(plot_cols)
	# 	print np.shape(cols)
	# 	raise ValueError('size of cos data and eagle data are different!')


def make_equ_width_plots(ion, ssfr, masses, smasses, radii, virial_radii, equ_widths, eagle_ids, cos_id_arr, plot_equ_widths, plot_W_errs, plot_W_flags, plot_equ_widths_radii, colorbar, bins_for_median, log_plots):

	one_sig_top = []
	one_sig_bot = []
	two_sig_top = []
	two_sig_bot = []
	median = []
	plot_radii = []
	plot_radii_err = []
	cos_percentiles = []

	### if moving to mA
	equ_widths *= 1.e3
	plot_equ_widths *= 1.e3
	plot_W_errs *= 1.e3

	# log equivalent widths for fits 
	if log_plots:
		temp_equ_widths = np.log10(equ_widths)
		temp_plot_equ_widths = np.log10(plot_equ_widths)
	else:
		temp_equ_widths = equ_widths
		temp_plot_equ_widths = plot_equ_widths

	### Errors and percentiles for EAGLE as well as finding out how much of an outlier each COS data point is. 
	for i in range(0,np.size(cos_id_arr)):
		curr_equ_widths = temp_equ_widths[eagle_ids == cos_id_arr[i]]
		curr_radii = radii[eagle_ids == cos_id_arr[i]]
		if np.size(curr_equ_widths) != 0:
			cos_percentiles.append(100.*float(np.size(curr_equ_widths[curr_equ_widths <= temp_plot_equ_widths[i]]))/np.size(curr_equ_widths))
			one_sig_top.append(np.percentile(curr_equ_widths,84))
			one_sig_bot.append(np.percentile(curr_equ_widths, 16))
			two_sig_top.append(np.percentile(curr_equ_widths, 95))
			two_sig_bot.append(np.percentile(curr_equ_widths, 5))
			median.append(np.percentile(curr_equ_widths, 50))
			plot_radii.append(np.mean(curr_radii))
			plot_radii_err.append(np.std(curr_radii, ddof=1))

	if log_plots:
		one_sig_bot = np.power(10,np.array(one_sig_bot))
		one_sig_top = np.power(10,np.array(one_sig_top))
		two_sig_bot = np.power(10,np.array(two_sig_bot))
		two_sig_top = np.power(10,np.array(two_sig_top))
	else:
		one_sig_bot = np.array(one_sig_bot)
		one_sig_top = np.array(one_sig_top)
		two_sig_bot = np.array(two_sig_bot)
		two_sig_top = np.array(two_sig_top)
	median = np.array(median)
	cos_percentiles = np.hstack(cos_percentiles)


	low_frac = float(np.size(cos_percentiles[cos_percentiles < 5.]))/np.size(cos_percentiles)
	high_frac = float(np.size(cos_percentiles[cos_percentiles > 95.]))/np.size(cos_percentiles)

	### Linear fits for EAGLE and COS
	eagle_params, eagle_errs = np.polyfit(plot_equ_widths_radii, median, 1 ,cov=True)
	eagle_fit_arr = eagle_params[0]*plot_equ_widths_radii+eagle_params[1]

	cos_params, cos_errs = np.polyfit(plot_equ_widths_radii, temp_plot_equ_widths, 1, cov=True)
	cos_fit_arr = cos_params[0]*plot_equ_widths_radii + cos_params[1]

	### move back to linear units because the plot will automatically to logy for everthing that goes into it. 
	if log_plots:
		median = np.power(10,median)
		eagle_fit_arr = np.power(10,eagle_fit_arr)
		cos_fit_arr = np.power(10,cos_fit_arr)

	### temporary (hopefully) some where mA some Angstroms
	if np.size(plot_equ_widths) > 0:
		for i in range(0,np.size(plot_equ_widths)):
			if plot_equ_widths[i] >= 100.:
				plot_equ_widths[i] /= 1000.

	if np.size(plot_equ_widths) > 0:
		obs_radius_max, obs_value_max, sim_radius_max, sim_value_max, obs_quad, sim_quad, delta, reject_val = KS_2D(plot_equ_widths_radii, plot_equ_widths, radii, equ_widths)

		try:
			with open(str(ion)+'_KS_EW_%.1f.txt' % (reject_val), 'w') as file:
				file.write('reject_val delta obs_radius_max obs_value_max sim_radius_max sim_value_max obs_quad sim_quad\n')
				file.write('%.1f %.3f %.2f %.2f %.2f %.2f %s %s' % (reject_val, delta, obs_radius_max, obs_value_max, sim_radius_max, sim_value_max, obs_quad, sim_quad))
				file.close()
		except:
			with open(str(ion)+'_KS_EW_%s.txt' % ('not_rejected'), 'w') as file:
				file.write('reject_val delta obs_radius_max obs_value_max sim_radius_max sim_value_max obs_quad sim_quad\n')
				file.write('%s %.3f %.2f %.2f %.2f %.2f %s %s' % (reject_val, delta, obs_radius_max, obs_value_max, sim_radius_max, sim_value_max, obs_quad, sim_quad))
				file.close()

		# ## scatter/single rel block
		# if colorbar == 'smass':
		# 	plt.scatter(radii, equ_widths, s=20., c=np.log10(smasses), cmap='RdBu_r', edgecolor = 'black', linewidth = 0.5)
		# 	if log_plots:
		# 		plt.yscale('log')
		# 	cb = plt.colorbar()
		# 	cb.set_label(r'$log_{10}(M_{star})$', fontsize=16.)
		# elif colorbar == 'hmass':
		# 	plt.scatter(radii, equ_widths, s=20., c=np.log10(masses), cmap='RdBu_r', edgecolor = 'black', linewidth = 0.5)
		# 	if log_plots:
		# 		plt.yscale('log')
		# 	cb = plt.colorbar()
		# 	cb.set_label(r'$log_{10}(M_{halo})$', fontsize=16.)
		# elif colorbar == 'ssfr':
		# 	plt.scatter(radii, equ_widths, c=ssfr, s=20., cmap='RdBu', edgecolor = 'black', linewidth = 0.5)
		# 	if log_plots:
		# 		plt.yscale('log')
		# 	cb = plt.colorbar()
		# 	cb.set_label(r'$log_{10}(sSFR)$', fontsize=16.)
		# plt.hold(True)

		### err bars/multi rel block
		bins = []
		indices = np.argsort(plot_equ_widths_radii)
		bins_radii_arr = plot_equ_widths_radii[indices]
		for i in range(0,np.size(plot_equ_widths_radii)):
			if i == 0:
				bins.append(int(round(bins_radii_arr[i]))-0.5)
			elif int(round(bins_radii_arr[i])) != int(round(bins_radii_arr[i-1])):
				bins.append(int(round(bins_radii_arr[i]))-0.5)
		bins.append(int(round(bins_radii_arr[-1]))+0.5)

		plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, radii, equ_widths)

		plt.errorbar(plot_radii, median, yerr=[median-two_sig_bot, two_sig_top-median], color='k', fmt = '.', ecolor = 'r')
		if log_plots:
			plt.yscale('log')
		plt.hold(True)
		plt.errorbar(plot_radii, median, yerr=[median-one_sig_bot, one_sig_top-median], xerr = plot_radii_err, color='k', fmt = '.', ecolor = 'b', label='EAGLE')

		# ### scatter/single rel block
		# bins = np.linspace(np.min(radii), np.max(radii),bins_for_median)
		# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, radii, equ_widths)
		# plt.plot(plot_x, plot_median, color = 'k', linestyle = '-')
		# plt.plot(plot_x, plot_84,  color = 'k', linestyle = '-.')
		# plt.plot(plot_x, plot_16,  color = 'k', linestyle = '-.')

		# bins = np.linspace(np.min(plot_equ_widths_radii), np.max(plot_equ_widths_radii),bins_for_median)# 
		# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, plot_equ_widths_radii, plot_equ_widths)
		# plt.plot(plot_x, plot_median,c='#00FF00', linestyle = '-')
		# plt.plot(plot_x, plot_84, c='#00FF00', linestyle = '-.')
		# plt.plot(plot_x, plot_16, c='#00FF00', linestyle = '-.')

		norm_radii = plot_equ_widths_radii[plot_W_flags > 0.]
		norm_equ_widths = plot_equ_widths[plot_W_flags > 0.]
		norm_err = plot_W_errs[plot_W_flags > 0.]
		plt.errorbar(norm_radii, norm_equ_widths, yerr=norm_err, fmt='*', c='#00FF00', markersize = 8.5, markeredgecolor = 'k', markeredgewidth = 0.5, label = 'COS: high=%.2f, low=%.2f' % (high_frac, low_frac))

		# # KS test stuff
		# plt.plot(obs_radius_max, obs_value_max, 'k*', markersize = 15., label = obs_quad)
		# plt.plot(sim_radius_max, sim_value_max, 'k.', markersize = 15., label = sim_quad)

		### Linear fit stuff
		# plt.plot(plot_equ_widths_radii, eagle_fit_arr, color='k', label=r'm=%.2e $\pm$ %.0e' % (eagle_params[0], np.sqrt(eagle_errs[0,0])) + '\n' + r'b=%.2e $\pm$ %.0e' % (np.power(10.,eagle_params[1]), (1.0/(np.abs(eagle_params[1])*np.log(10.)))*np.sqrt(eagle_errs[1,1])))
		# plt.plot(plot_equ_widths_radii, cos_fit_arr, color='#00FF00', label=r'm=%.2e $\pm$ %.0e' % (cos_params[0], np.sqrt(cos_errs[0,0])) + '\n' + r'b=%.2e $\pm$ %.0e' % (np.power(10.,cos_params[1]), (1.0/(np.abs(cos_params[1])*np.log(10.)))*np.sqrt(cos_errs[1,1])))
		plt.plot(plot_equ_widths_radii, eagle_fit_arr, color='k', label=r'm=%.1e $\pm$ %.0e' % (eagle_params[0], np.sqrt(eagle_errs[0,0])) + '\n' + r'b=%.1e $\pm$ %.0e' % (eagle_params[1], np.sqrt(eagle_errs[1,1])))
		plt.plot(plot_equ_widths_radii, cos_fit_arr, color='#00FF00', label=r'm=%.1e $\pm$ %.0e' % (cos_params[0], np.sqrt(cos_errs[0,0])) + '\n' + r'b=%.1e $\pm$ %.0e' % (cos_params[1], np.sqrt(cos_errs[1,1])))
		plt.hold(False)

		# plt.title('Red W vs b, Rejection at: %s' % (str(reject_val)))
		plt.title('Passive: W vs b for %s' % (ion), fontsize=16.)

	else:
		if colorbar == 'smass':
			plt.scatter(radii, equ_widths, s=20., c=np.log10(smasses), cmap='RdBu_r', edgecolor = 'black', linewidth = 0.5, label = 'EAGLE')
			cb = plt.colorbar()
			cb.set_label(r'$log_{10}(M_{star})$', fontsize=16.)
		elif colorbar == 'hmass':
			plt.scatter(radii, equ_widths, s=20., c=np.log10(masses), cmap='RdBu_r', edgecolor = 'black', linewidth = 0.5, label = 'EAGLE')
			cb = plt.colorbar()
			cb.set_label(r'$log_{10}(M_{halo})$', fontsize=16.)
		elif colorbar == 'ssfr':
			plt.scatter(radii, equ_widths, c=ssfr, s=20., cmap='RdBu', edgecolor = 'black', linewidth = 0.5, label = 'EAGLE')
			cb = plt.colorbar()
			cb.set_label(r'$log_{10}(sSFR)$', fontsize=16.)
		if np.size(plot_equ_widths) > 0:
			plt.hold(True)
			plt.plot(plot_equ_widths_radii,plot_equ_widths, '*', c='#00FF00', markersize = 8.5, markeredgecolor = 'k', markeredgewidth = 0.5, label = 'COS', fontsize=16.)
			plt.hold(False)

		bins = np.linspace(np.min(x_arr), np.max(x_arr),bins_for_median)
		plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, radii, equ_widths)

		plt.hold(True)
		plt.plot(obs_radius_max, obs_value_max, 'k*', markersize = 15., label = obs_quad)
		plt.plot(sim_radius_max, sim_value_max, 'k.', markersize = 15., label = sim_quad)
		# plt.plot(plot_x, plot_median,'k')
		# plt.plot(plot_x, plot_68, 'g')
		# plt.plot(plot_x, plot_32, 'g')
		# plt.plot(plot_x, plot_90, 'r')
		# plt.plot(plot_x, plot_10, 'r')
		plt.hold(False)

		plt.title('Passive: W vs b for %s' % (ion), fontsize=16.)

	plt.xlabel('Impact Parameter (kpc)', fontsize=16.)
	plt.ylabel(r'Equivalent Width ($\AA{}$)', fontsize=16.)
	if log_plots:
		plt.legend(loc = 'lower left', fontsize=12.)
	else:
		plt.legend(loc = 'upper right', fontsize=12.)
	plt.grid()
	if colorbar == 'smass':
		plt.savefig(ion+'_equ_width_with_color_smass.pdf')
	elif colorbar == 'hmass':
		plt.savefig(ion+'_equ_width_with_color_hmass.pdf')
	elif colorbar == 'ssfr':
		plt.savefig(ion+'_equ_width_with_color_ssfr.pdf')
	plt.close()


	if colorbar == 'smass':
		fig = plt.scatter(virial_radii, equ_widths, s=20., c=np.log10(smasses), cmap='RdBu_r', edgecolor = 'black', linewidth = 0.5)
		if log_plots:
			plt.yscale('log')
		cb = plt.colorbar()
		cb.set_label(r'$log_{10}(M_{star})$', fontsize=16.)
	elif colorbar == 'hmass':
		fig = plt.scatter(virial_radii, equ_widths, s=20., c=np.log10(masses), cmap='RdBu_r', edgecolor = 'black', linewidth = 0.5)
		if log_plots:
			plt.yscale('log')
		cb = plt.colorbar()
		cb.set_label(r'$log_{10}(M_{halo})$', fontsize=16.)	
	elif colorbar == 'ssfr':
		plt.scatter(virial_radii, equ_widths, c=ssfr, s=20., cmap='RdBu', edgecolor = 'black', linewidth = 0.5)
		if log_plots:
			plt.yscale('log')
		cb = plt.colorbar()
		cb.set_label(r'$log_{10}(sSFR)$', fontsize=16.)

	bins = np.linspace(np.min(virial_radii), np.max(virial_radii),bins_for_median)
	plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, virial_radii, equ_widths)
	plt.hold(True)
	plt.plot(plot_x, plot_median,'k')
	plt.plot(plot_x, plot_84, 'g')
	plt.plot(plot_x, plot_16, 'g')
	plt.plot(plot_x, plot_95, 'r')
	plt.plot(plot_x, plot_5, 'r')
	plt.hold(False)

	try:
		plt.title('Passive: W vs b for %s' % (ion), fontsize=18.)
	except:
		plt.title('Passive: W vs b for %s' % (ion), fontsize=16.)
	plt.xlabel('Impact Parameter (in virial radii)', fontsize=16.)
	plt.ylabel(r'Equivalent Width ($\AA{}$)', fontsize=16.)
	plt.grid()
	if colorbar == 'smass':
		plt.savefig(ion+'equ_width_virial_with_color_smass.pdf')
	elif colorbar == 'hmass':
		plt.savefig(ion+'equ_width_virial_with_color_hmass.pdf')
	elif colorbar == 'ssfr':
		plt.savefig(ion+'equ_width_virial_with_color_ssfr.pdf')

	plt.close()

	# if np.shape(plot_equ_widths) != np.shape(equ_widths):
	# 	print np.shape(plot_equ_widths)
	# 	print np.shape(equ_widths)
	# 	raise ValueError('size of cos data and eagle data are different!')

def make_contour_col_dense_plots(ion, radii, virial_radii, cols, plot_cols, plot_cols_err, plot_cols_flags, plot_cols_radii, smasses, ssfr, virial_radii_bool):

	ymin, ymax = return_proper_min_max(cols, np.concatenate((plot_cols, [16.,12])))
	xmin, xmax = return_proper_min_max(radii, np.concatenate((plot_cols_radii, [25,195])))
	xmin -= 5.
	xmax += 5.
	ymin -= 0.1
	ymax += 0.1

	### fixed these values for final plots. If yo you want to go back uncomment section above and in the if virial bool section change ymin, ymax in extent to min/max cols
	# xmin = 0.0
	# xmax = 300.
	# ymin = -0.1
	# ymax = 2.5

	### For halo mass
	# upper_mass = np.array([9.7, 15., 15.])
	# lower_mass = np.array([5., 9.7, 9.7])

	# upper_ssfr = np.array([-5., -5., -11.])
	# lower_ssfr = np.array([-15., -11., -15.])

	# colors = ['m', 'b', 'r']
	# stagger = [0.0, -1.5, 1.5]
	# labels = ['low mass', 'blue', 'red']

	upper_mass = np.array([15.])
	lower_mass = np.array([5.])

	upper_ssfr = np.array([-5.])
	lower_ssfr = np.array([-15.])

	colors = ['m']
	stagger = [0.0]
	labels = ['low mass']
	# labels = ['EAGLE fit']

	if virial_radii_bool:
		height, xedges, yedges = np.histogram2d(virial_radii, cols, bins = 50, range=[[np.min(virial_radii), np.max(virial_radii)], [ymin,ymax]])
		plt.imshow(height.transpose(), extent = (np.min(virial_radii), np.max(virial_radii), ymin, ymax), aspect = 'auto', origin='lower', cmap = 'gray_r')
		cb = plt.colorbar()
		cb.set_label('points per bin', fontsize=14.)
		plt.xlabel('Impact Parameter (virial radii)', fontsize=14.)

		plt.hold(True)
		for i in range(0,np.size(upper_mass)):
			curr_radii = virial_radii[((np.log10(smasses) < upper_mass[i]) & (np.log10(smasses) > lower_mass[i]) & (ssfr < upper_ssfr[i]) & (ssfr > lower_ssfr[i]))]
			curr_cols = cols[((np.log10(smasses) < upper_mass[i]) & (np.log10(smasses) > lower_mass[i]) & (ssfr < upper_ssfr[i]) & (ssfr > lower_ssfr[i]))]
			eagle_params, eagle_errs = np.polyfit(curr_radii, curr_cols, 1 ,cov=True)
			eagle_fit_arr = eagle_params[0]*virial_radii+eagle_params[1]
			plt.plot(virial_radii[eagle_fit_arr >=0.0], eagle_fit_arr[eagle_fit_arr >=0.0], color=colors[i], label=r'm=%.1e $\pm$ %.0e' %(eagle_params[0], np.sqrt(eagle_errs[0,0])) + '\n' + r'b=%.1e $\pm$ %.0e' % (eagle_params[1], np.sqrt(eagle_errs[1,1])))

		plt.hold(False)
		plt.legend(fontsize=12.)


	else:
		height, xedges, yedges = np.histogram2d(radii, cols, bins = 50, range=[[xmin, xmax], [ymin,ymax]])
		plt.imshow(height.transpose(), extent = (xmin, xmax, ymin, ymax), aspect = 'auto', origin='lower', cmap = 'gray_r')
		plt.hold(True)
		cb = plt.colorbar()
		cb.set_label('points per bin', fontsize=14.)
		plt.xlabel('Impact Parameter (kpc)', fontsize=14.)

		plt.hold(True)
		for i in range(0,np.size(upper_mass)):
			curr_radii = radii[((np.log10(smasses) < upper_mass[i]) & (np.log10(smasses) > lower_mass[i]) & (ssfr < upper_ssfr[i]) & (ssfr > lower_ssfr[i]))]
			curr_cols = cols[((np.log10(smasses) < upper_mass[i]) & (np.log10(smasses) > lower_mass[i]) & (ssfr < upper_ssfr[i]) & (ssfr > lower_ssfr[i]))]
			eagle_params, eagle_errs = np.polyfit(curr_radii, curr_cols, 1 ,cov=True)
			eagle_fit_arr = eagle_params[0]*radii+eagle_params[1]
			plt.plot(radii[eagle_fit_arr >=0.0], eagle_fit_arr[eagle_fit_arr >=0.0], color=colors[i], label=r'm=%.1e $\pm$ %.0e' %(eagle_params[0], np.sqrt(eagle_errs[0,0])) + '\n' + r'b=%.1e $\pm$ %.0e' % (eagle_params[1], np.sqrt(eagle_errs[1,1])))

		plt.hold(False)

		# ### If using just one
		# eagle_params, eagle_errs = np.polyfit(radii, cols, 1 ,cov=True)
		# eagle_fit_arr = eagle_params[0]*radii+eagle_params[1]
		# plt.plot(radii[eagle_fit_arr >=0.0], eagle_fit_arr[eagle_fit_arr >=0.0])

	plt.ylabel(r'$log_{10}(N_{%s}) cm^{-2}$' % (ion), fontsize=14.)
	plt.title('Passive: N vs b for %s' % (ion), fontsize=16.)

	#### overplot all data points if testing stuff
	# plt.hold(True)
	# plt.plot(radii, cols, 'k.')
	# plt.hold(False)

	### overplot cos data
	if virial_radii_bool == False:
		print 'something up with flags?'
		print plot_cols_flags
		print ''
		plt.hold(True)
		norm_cols = plot_cols[plot_cols_flags == 1.]
		norm_radii = plot_cols_radii[plot_cols_flags == 1.]
		upper_cols = plot_cols[plot_cols_flags == 3.]
		upper_radii = plot_cols_radii[plot_cols_flags == 3.]
		lower_cols = plot_cols[plot_cols_flags == 2.]
		lower_radii = plot_cols_radii[plot_cols_flags == 2.]


		plt.errorbar(norm_radii, norm_cols, yerr=plot_cols_err[plot_cols_flags == 1.],linestyle='None', marker='*', c='#00FF00', markersize=6., label='COS Data')
		plt.errorbar(upper_radii, upper_cols, yerr=plot_cols_err[plot_cols_flags == 3.],linestyle='None', marker='v', c='#00FF00', markersize=6.)
		plt.errorbar(lower_radii, lower_cols, yerr=plot_cols_err[plot_cols_flags == 2.],linestyle='None', marker='^', c='#00FF00', markersize=6.)
		plt.hold(False)

	plt.legend(fontsize=12.)
	if virial_radii_bool:
		plt.savefig(ion + '_col_contour_test_virial.pdf')
	else:
		plt.savefig(ion + '_col_contour_test_kpc.pdf')
	plt.close()

	# if virial_radii_bool:
	# 	height, xedges, yedges = np.histogram2d(virial_radii, cols, bins = 50)
	# 	plt.imshow(height.transpose(), extent = (np.min(virial_radii), np.max(virial_radii), np.min(cols), np.max(cols)), origin='lower', aspect = 'auto', cmap = 'gray')
	# 	plt.xlabel('Impact Parameter (virial radii)')
	# else:
	# 	height, xedges, yedges = np.histogram2d(radii, cols, bins = 50)
	# 	plt.imshow(height.transpose(), extent = (np.min(radii), np.max(radii), np.min(cols), np.max(cols)), origin='lower', aspect = 'auto', cmap = 'gray')
	# 	plt.xlabel('Impact Parameter (kpc)')

	# plt.ylabel(r'$log_{10}(N_{%s})$ cm^-2' % (ion))
	# plt.title('Impact Parameter vs Column Density')

	# #### overplot all data points if testing stuff
	# # plt.hold(True)
	# # plt.plot(plot_cols_radii, plot_cols, 'k.')
	# # plt.hold(False)
	# plt.savefig(ion + '_col_dense_contour_test.pdf')
	# plt.close()

def make_equ_width_contour_plots(ion, radii, virial_radii, equ_widths, plot_equ_widths, plot_W_errs, plot_W_flags, plot_equ_widths_radii, smasses, ssfr, virial_radii_bool, log_plots):

	ymin, ymax = return_proper_min_max(equ_widths, np.concatenate((plot_equ_widths, [1.0e-2,1.])))
	xmin, xmax = return_proper_min_max(radii, np.concatenate((plot_equ_widths_radii, [25,245])))
	xmin -= 5.
	xmax += 5.
	ymax += 0.1

	### fixed these values for final plots. If yo you want to go back uncomment section above and in the if virial bool section change ymin, ymax in extent to min/max equ_widths
	# xmin = 0.0
	# xmax = 300.
	# ymin = -0.1
	# ymax = 2.5

	### For halo mass
	# upper_mass = np.array([9.7, 15., 15.])
	# lower_mass = np.array([5., 9.7, 9.7])

	# upper_ssfr = np.array([-5., -5., -11.])
	# lower_ssfr = np.array([-15., -11., -15.])

	# colors = ['m', 'b', 'r']
	# stagger = [0.0, -1.5, 1.5]
	# labels = ['low mass', 'blue', 'red']

	upper_mass = np.array([15.])
	lower_mass = np.array([5.])

	upper_ssfr = np.array([-5.])
	lower_ssfr = np.array([-15.])

	colors = ['m']
	stagger = [0.0]
	labels = ['low mass']
	# labels = ['EAGLE fit']

	if virial_radii_bool:
		height, xedges, yedges = np.histogram2d(virial_radii, equ_widths, bins = 50, range=[[np.min(virial_radii), np.max(virial_radii)], [ymin,ymax]])
		plt.imshow(height.transpose(), extent = (np.min(virial_radii), np.max(virial_radii), ymin, ymax), aspect = 'auto', origin='lower', cmap = 'gray_r')
		if log_plots:
			plt.yscale('log')
		cb = plt.colorbar()
		cb.set_label('points per bin', fontsize=16.)
		plt.xlabel('Impact Parameter (virial radii)', fontsize=16.)

		plt.hold(True)
		for i in range(0,np.size(upper_mass)):
			curr_radii = virial_radii[((np.log10(smasses) < upper_mass[i]) & (np.log10(smasses) > lower_mass[i]) & (ssfr < upper_ssfr[i]) & (ssfr > lower_ssfr[i]))]
			curr_equ_widths = equ_widths[((np.log10(smasses) < upper_mass[i]) & (np.log10(smasses) > lower_mass[i]) & (ssfr < upper_ssfr[i]) & (ssfr > lower_ssfr[i]))]
			if log_plots:
				eagle_params, eagle_errs = np.polyfit(curr_radii, np.log10(curr_equ_widths), 1 ,cov=True)
				eagle_fit_arr = np.power(10,eagle_params[0]*virial_radii+eagle_params[1])
			else:
				eagle_params, eagle_errs = np.polyfit(curr_radii, curr_equ_widths, 1 ,cov=True)
				eagle_fit_arr = eagle_params[0]*virial_radii+eagle_params[1]
			plt.plot(virial_radii[eagle_fit_arr >=0.0], eagle_fit_arr[eagle_fit_arr >=0.0], color=colors[i], label=r'm=%.1e $\pm$ %.0e' %(eagle_params[0], np.sqrt(eagle_errs[0,0])) + '\n' + r'b=%.1e $\pm$ %.0e' % (eagle_params[1], np.sqrt(eagle_errs[1,1])))


	else:
		height, xedges, yedges = np.histogram2d(radii, equ_widths, bins = 50, range=[[xmin, xmax], [ymin,ymax]])
		plt.imshow(height.transpose(), extent = (xmin, xmax, ymin, ymax), aspect = 'auto', origin='lower', cmap = 'gray_r')
		if log_plots:
			plt.yscale('log')
		plt.hold(True)
		cb = plt.colorbar()
		cb.set_label('points per bin', fontsize=16.)
		plt.xlabel('Impact Parameter (kpc)', fontsize=16.)

		for i in range(0,np.size(upper_mass)):
			curr_radii = radii[((np.log10(smasses) < upper_mass[i]) & (np.log10(smasses) > lower_mass[i]) & (ssfr < upper_ssfr[i]) & (ssfr > lower_ssfr[i]))]
			curr_equ_widths = equ_widths[((np.log10(smasses) < upper_mass[i]) & (np.log10(smasses) > lower_mass[i]) & (ssfr < upper_ssfr[i]) & (ssfr > lower_ssfr[i]))]
			if log_plots:
				eagle_params, eagle_errs = np.polyfit(curr_radii, np.log10(curr_equ_widths), 1 ,cov=True)
				eagle_fit_arr = np.power(10,eagle_params[0]*radii+eagle_params[1])
			else:
				eagle_params, eagle_errs = np.polyfit(curr_radii, curr_equ_widths, 1 ,cov=True)
				eagle_fit_arr = eagle_params[0]*radii+eagle_params[1]
			plt.plot(radii[eagle_fit_arr >=0.0], eagle_fit_arr[eagle_fit_arr >=0.0], color=colors[i], label=r'm=%.1e $\pm$ %.0e' %(eagle_params[0], np.sqrt(eagle_errs[0,0])) + '\n' + r'b=%.1e $\pm$ %.0e' % (eagle_params[1], np.sqrt(eagle_errs[1,1])))

		# ### If using just one
		# eagle_params, eagle_errs = np.polyfit(radii, equ_widths, 1 ,cov=True)
		# eagle_fit_arr = eagle_params[0]*radii+eagle_params[1]
		# plt.plot(radii[eagle_fit_arr >=0.0], eagle_fit_arr[eagle_fit_arr >=0.0])

	plt.ylabel(r'Equivalent Width ($\AA{}$)', fontsize=16.)
	print ymin
	print ymax
	plt.ylim((0.001, 3.0))
	plt.title('Passive: W vs b for %s' % (ion), fontsize=16.)

	### overplot all data points if testing stuff
	# plt.hold(True)
	# plt.plot(radii, equ_widths, 'k.')
	# plt.hold(False)

	### overplot cos data
	if virial_radii_bool == False:
		norm_W = plot_equ_widths[plot_W_flags > 0.]
		norm_radii = plot_equ_widths_radii[plot_W_flags > 0.]

		plt.errorbar(norm_radii, norm_W, yerr=plot_W_errs[plot_W_flags > 0.],linestyle='None', marker='*', c='#00FF00', markersize=6., label='COS Data')
		plt.hold(False)

	plt.legend(fontsize=16., loc='lower left')
	if virial_radii_bool:
		plt.savefig(ion + '_equ_width_contour_test_virial.pdf')
	else:
		plt.savefig(ion + '_equ_width_contour_test_kpc.pdf')
	plt.close()


def return_proper_min_max(array1, array2):
	if np.min(array1) < np.min(array2):
		return_min = np.min(array1)
	else:
		return_min = np.min(array2)

	if np.max(array1) > np.max(array2):
		return_max = np.max(array1)
	else:
		return_max = np.max(array2)

	return return_min, return_max

def KS_2D(obs_radii, obs_values, sim_radii, sim_values):
	obs_num = np.size(obs_radii)
	sim_num = np.size(sim_radii)

	if ((obs_num <= 0) or (sim_num <= 0)):
		raise ValueError('No observational data points')

	delta_obs = 0.0
	obs_radius_max = 0.0
	obs_value_max = 0.0
	obs_quad = 'na'
	for i in range(0,obs_num):

		# quad 1 (>, >)
		obs_frac = np.size(obs_radii[(obs_radii > obs_radii[i]) & (obs_values > obs_values[i])])/float(obs_num)
		sim_frac = np.size(sim_radii[(sim_radii > obs_radii[i]) & (sim_values > obs_values[i])])/float(sim_num)
		if np.abs(obs_frac - sim_frac) > delta_obs:
			delta_obs = np.abs(obs_frac - sim_frac)
			obs_radius_max = obs_radii[i]
			obs_value_max = obs_values[i]
			obs_quad = 'b>,W>: o %.2f s %.2f' % (obs_frac, sim_frac)

		# quad 2 (>, <)
		obs_frac = np.size(obs_radii[(obs_radii > obs_radii[i]) & (obs_values < obs_values[i])])/float(obs_num)
		sim_frac = np.size(sim_radii[(sim_radii > obs_radii[i]) & (sim_values < obs_values[i])])/float(sim_num)
		if np.abs(obs_frac - sim_frac) > delta_obs:
			delta_obs = np.abs(obs_frac - sim_frac)
			obs_radius_max = obs_radii[i]
			obs_value_max = obs_values[i]
			obs_quad = 'b>,W<: o %.2f s %.2f' % (obs_frac, sim_frac)

		# quad 3 (<, >)
		obs_frac = np.size(obs_radii[(obs_radii < obs_radii[i]) & (obs_values > obs_values[i])])/float(obs_num)
		sim_frac = np.size(sim_radii[(sim_radii < obs_radii[i]) & (sim_values > obs_values[i])])/float(sim_num)
		if np.abs(obs_frac - sim_frac) > delta_obs:
			delta_obs = np.abs(obs_frac - sim_frac)
			obs_radius_max = obs_radii[i]
			obs_value_max = obs_values[i]
			obs_quad = 'b<,W>: o %.2f s %.2f' % (obs_frac, sim_frac)

		# quad 4 (<, <)
		obs_frac = np.size(obs_radii[(obs_radii < obs_radii[i]) & (obs_values < obs_values[i])])/float(obs_num)
		sim_frac = np.size(sim_radii[(sim_radii < obs_radii[i]) & (sim_values < obs_values[i])])/float(sim_num)
		if np.abs(obs_frac - sim_frac) > delta_obs:
			delta_obs = np.abs(obs_frac - sim_frac)
			obs_radius_max = obs_radii[i]
			obs_value_max = obs_values[i]
			obs_quad = 'b<,W<: o %.2f s %.2f' % (obs_frac, sim_frac)

	delta_sim = 0.0
	sim_radius_max = 0.0
	sim_value_max = 0.0
	sim_quad = 'na'
	for i in range(0,sim_num):

		# quad 1 (>, >)
		obs_frac = np.size(obs_radii[(obs_radii > sim_radii[i]) & (obs_values > sim_values[i])])/float(obs_num)
		sim_frac = np.size(sim_radii[(sim_radii > sim_radii[i]) & (sim_values > sim_values[i])])/float(sim_num)
		if np.abs(obs_frac - sim_frac) > delta_sim:
			delta_sim = np.abs(obs_frac - sim_frac)
			sim_radius_max = sim_radii[i]
			sim_value_max = sim_values[i]
			sim_quad = 'b>,W>: o %.2f s %.2f' % (obs_frac, sim_frac)

		# quad 2 (>, <)
		obs_frac = np.size(obs_radii[(obs_radii > sim_radii[i]) & (obs_values < sim_values[i])])/float(obs_num)
		sim_frac = np.size(sim_radii[(sim_radii > sim_radii[i]) & (sim_values < sim_values[i])])/float(sim_num)
		if np.abs(obs_frac - sim_frac) > delta_sim:
			delta_sim = np.abs(obs_frac - sim_frac)
			sim_radius_max = sim_radii[i]
			sim_value_max = sim_values[i]
			sim_quad = 'b>,W<: o %.2f s %.2f' % (obs_frac, sim_frac)

		# quad 3 (<, >)
		obs_frac = np.size(obs_radii[(obs_radii < sim_radii[i]) & (obs_values > sim_values[i])])/float(obs_num)
		sim_frac = np.size(sim_radii[(sim_radii < sim_radii[i]) & (sim_values > sim_values[i])])/float(sim_num)
		if np.abs(obs_frac - sim_frac) > delta_sim:
			delta_sim = np.abs(obs_frac - sim_frac)
			sim_radius_max = sim_radii[i]
			sim_value_max = sim_values[i]
			sim_quad = 'b<,W>: o %.2f s %.2f' % (obs_frac, sim_frac)

		# quad 4 (<, <)
		obs_frac = np.size(obs_radii[(obs_radii < sim_radii[i]) & (obs_values < sim_values[i])])/float(obs_num)
		sim_frac = np.size(sim_radii[(sim_radii < sim_radii[i]) & (sim_values < sim_values[i])])/float(sim_num)
		if np.abs(obs_frac - sim_frac) > delta_sim:
			delta_sim = np.abs(obs_frac - sim_frac)
			sim_radius_max = sim_radii[i]
			sim_value_max = sim_values[i]
			sim_quad = 'b<,W<: o %.2f s %.2f' % (obs_frac, sim_frac)

	delta = (delta_sim+delta_obs)/2.0
	n_eff = (obs_num*sim_num)/(obs_num + sim_num)

	confidence_vals = np.arange(.999, 0.000, -0.001) 

	for i in range(0,np.size(confidence_vals)):
		if delta > np.sqrt(-0.5*np.log((1.-confidence_vals[i])/2.)*n_eff**(-1.0)):
			reject_val = confidence_vals[i]*100.
			break
		if confidence_vals[i] <= 0.1:
			reject_val = 'Not Rejected'

	obs_r = pearson_r(obs_radii, obs_values)
	sim_r = pearson_r(sim_radii, sim_values)

	r = np.sqrt(1.0-0.5*(obs_r**2.0+sim_r**2.0))

	value_for_KS = (np.sqrt(n_eff*delta))/(1.+np.sqrt(1.-r**2.0)*(0.25-0.75/np.sqrt(n_eff)))
	tol = 1.e-6 # tolerance for convergence of cumulative KS thing
	significance = Q_ks(value_for_KS, tol)

	return obs_radius_max, obs_value_max, sim_radius_max, sim_value_max, obs_quad, sim_quad, delta, reject_val



def pearson_r(radii, vals):
	ave_radii = np.average(radii)
	ave_vals = np.average(vals)

	top = np.sum((radii-ave_radii)*(vals-ave_vals))
	bottom = np.sqrt(np.sum((radii-ave_radii)**2.0))*np.sqrt(np.sum((vals-ave_vals)**2.0))


	return top/bottom

def Q_ks(value, tol):
	final_sum = 0.0
	for j in range(1,10):
		previous_sum = final_sum
		final_sum += (-1.0)**(j-1)*np.exp(-2.0*j**2.0*value**2.0)
		if np.abs(previous_sum-final_sum) <= tol:
			return 2.0*final_sum
	if j == 99:
		print 'Q_ks did not converge returning 1'
		return 1.0

def select_virial_or_not(radii, virial_radii, virial_radii_bool, vel, virial_vel, virial_vel_bool, masses, smasses, halo_mass_bool):
	if virial_radii_bool:
		used_radii = virial_radii
	else:
		used_radii = radii

	if virial_vel_bool:
		used_velocities = virial_vel
	else:
		used_velocities = vel

	if halo_mass_bool:
		used_masses = masses
	else:
		used_masses = smasses

	return used_radii, used_velocities, used_masses

def plot_for_multiple_gals_by_radius(ion, radii_bins, plot_colors, virial_vel_bool, virial_radii_bool, halo_mass_bool, mean_spectra_bool, radii, virial_radii, masses, smasses, flux_for_stacks, vel_for_stacks, virial_vel_for_stacks, min_halo_mass, max_halo_mass):

	used_radii, used_velocities, used_masses = select_virial_or_not(radii, virial_radii, virial_radii_bool, vel_for_stacks, virial_vel_for_stacks, virial_vel_bool, masses, smasses, halo_mass_bool)

	for j in range(0,np.size(radii_bins)-1):
		num_in_bin = 0
		final_ion_flux = np.array([])
		final_velocities = np.array([])

		for i in range(0,np.size(used_radii)):
			if ((radii_bins[j] < used_radii[i]) & (radii_bins[j+1] > used_radii[i])):
				num_in_bin += 1
				if np.size(final_ion_flux) == 0:
					final_ion_flux = flux_for_stacks[i]
					final_velocities = used_velocities[i]
				else:
					final_ion_flux = np.concatenate((final_ion_flux, flux_for_stacks[i]))
					final_velocities = np.concatenate((final_velocities, used_velocities[i]))
	
		
		try:
			if virial_vel_bool:
				vel_bins = np.linspace(np.min(final_velocities), np.max(final_velocities), (np.max(final_velocities)-np.min(final_velocities))/0.1)
			else:
				vel_bins = np.linspace(np.min(final_velocities), np.max(final_velocities), (np.max(final_velocities)-np.min(final_velocities))/5.0)
			plot_velocities = np.zeros(np.size(vel_bins)-1)
			plot_fluxes = np.zeros(np.size(vel_bins)-1)
			for i in range(0,np.size(vel_bins)-1):
				if mean_spectra_bool:
					plot_fluxes[i] = np.mean(final_ion_flux[np.where((final_velocities>vel_bins[i]) & (final_velocities<vel_bins[i+1]))])
				else:
					plot_fluxes[i] = np.median(final_ion_flux[np.where((final_velocities>vel_bins[i]) & (final_velocities<vel_bins[i+1]))])
				plot_velocities[i] = np.median([vel_bins[i],vel_bins[i+1]])

			if virial_radii_bool:
				plt.plot(plot_velocities, plot_fluxes, plot_colors[j] + '-', label = '%.1f-%.1fR_vir:(n=%d)' %(radii_bins[j], radii_bins[j+1], num_in_bin))
			else:
				plt.hold(True)
				plt.plot(plot_velocities, plot_fluxes, plot_colors[j] + '-', label = '%.0f-%.0fkpc:(n=%d)' %(radii_bins[j], radii_bins[j+1], num_in_bin))
				plt.hold(True)
			plt.hold(True)
		except:
			print 'plot fail for these radii. maybe None in bin? %.1f-%.1f' % (radii_bins[j], radii_bins[j+1])

	plt.legend(loc='lower left')
	plt.grid()
	plt.hold(False)
	plt.ylim(ymin=-0.2, ymax=1)
	plt.title('Flux vs Speed Relative to Central Galaxy for %s [%.1f-%.1f]' % (ion, min_halo_mass,max_halo_mass) )
	plt.ylabel('normalized flux')
	if virial_vel_bool:
		plt.xlim(xmin=-3, xmax = 3)
		plt.xlabel('vel (vel/virial velocity)')
	else:
		plt.xlabel('vel (km/s)')
	if virial_radii_bool:
		if virial_vel_bool:
			if mean_spectra_bool:
				plt.savefig(ion +'radius_bins_mean_spectra_virial_radius_virial_vel.pdf')
			else:
				plt.savefig(ion +'radius_bins_median_spectra_virial_radius_virial_vel.pdf')
		else:
			if mean_spectra_bool:
				plt.savefig(ion +'radius_bins_mean_spectra_virial_radius_physical_vel.pdf')
			else:
				plt.savefig(ion +'radius_bins_median_spectra_virial_radius_physical_vel.pdf')
	else:
		if virial_vel_bool:
			if mean_spectra_bool:
				plt.savefig(ion +'radius_bins_mean_spectra_physical_radius_virial_vel.pdf')
			else:
				plt.savefig(ion +'radius_bins_median_spectra_physical_radius_virial_vel.pdf')
		else:
			if mean_spectra_bool:
				plt.savefig(ion +'radius_bins_mean_spectra_physical_radius_physical_vel.pdf')
			else:
				plt.savefig(ion +'radius_bins_median_spectra_physical_radius_physical_vel.pdf')

	plt.close()

def plot_for_multiple_gals_by_mass(ion, mass_bins, plot_colors, virial_vel_bool, virial_radii_bool, halo_mass_bool, mean_spectra_bool, masses, smasses, radii, virial_radii, flux_for_stacks, vel_for_stacks, virial_vel_for_stacks, min_halo_mass, max_halo_mass):
	
	used_radii, used_velocities, used_masses = select_virial_or_not(radii, virial_radii, virial_radii_bool, vel_for_stacks, virial_vel_for_stacks, virial_vel_bool, masses, smasses, halo_mass_bool)
	used_masses = np.log10(used_masses)

	for j in range(0,np.size(mass_bins)-1):
		num_in_bin = 0
		final_ion_flux = np.array([])
		final_velocities = np.array([])

		for i in range(0,np.size(used_radii)):
			if ((mass_bins[j] < used_masses[i]) & (mass_bins[j+1] > used_masses[i])):
				num_in_bin += 1
				if np.size(final_ion_flux) == 0:
					final_ion_flux = flux_for_stacks[i]
					final_velocities = used_velocities[i]
				else:
					final_ion_flux = np.concatenate((final_ion_flux, flux_for_stacks[i]))
					final_velocities = np.concatenate((final_velocities, used_velocities[i]))

		try:
			if virial_vel_bool:
				vel_bins = np.linspace(np.min(final_velocities), np.max(final_velocities), (np.max(final_velocities)-np.min(final_velocities))/0.1)
			else:
				vel_bins = np.linspace(np.min(final_velocities), np.max(final_velocities), (np.max(final_velocities)-np.min(final_velocities))/5.0)
			plot_velocities = np.zeros(np.size(vel_bins)-1)
			plot_fluxes = np.zeros(np.size(vel_bins)-1)
			for i in range(0,np.size(vel_bins)-1):
				if mean_spectra_bool:
					plot_fluxes[i] = np.mean(final_ion_flux[np.where((final_velocities>vel_bins[i]) & (final_velocities<vel_bins[i+1]))])
				else:
					plot_fluxes[i] = np.median(final_ion_flux[np.where((final_velocities>vel_bins[i]) & (final_velocities<vel_bins[i+1]))])
				plot_velocities[i] = np.median([vel_bins[i],vel_bins[i+1]])

			if virial_radii_bool:
				plt.plot(plot_velocities, plot_fluxes, '-', label = '%.1f-%.1flog10(M_sol):(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
			else:
				plt.plot(plot_velocities, plot_fluxes, '-', label = '%.1f-%.1f log10(M_sol):(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
			plt.hold(True)
		except:
			print 'plot fail for these masses. maybe None in bin? %.1f-%.1f' % (mass_bins[j], mass_bins[j+1])
	plt.legend(loc='lower left')
	plt.grid()
	plt.ylim(ymin=-0.2, ymax=1)
	plt.title('Flux vs Speed Relative to Central Galaxy for %s [%.1f-%.1f]' % (ion, min_halo_mass,max_halo_mass) )
	plt.ylabel('normalized flux')
	plt.hold(False)
	if virial_vel_bool:
		plt.xlim(xmin=-3, xmax = 3)
		plt.xlabel('vel (vel/virial velocity)')
	else:
		plt.xlabel('vel (km/s)')
	if virial_radii_bool:
		if virial_vel_bool:
			if mean_spectra_bool:
				plt.savefig(ion +'mass_bins_mean_spectra_virial_radius_virial_vel.pdf')
			else:
				plt.savefig(ion +'mass_bins_median_spectra_virial_radius_virial_vel.pdf')
		else:
			if mean_spectra_bool:
				plt.savefig(ion +'mass_bins_mean_spectra_virial_radius_physical_vel.pdf')
			else:
				plt.savefig(ion +'mass_bins_median_spectra_virial_radius_physical_vel.pdf')
	else:
		if virial_vel_bool:
			if mean_spectra_bool:
				plt.savefig(ion +'mass_bins_mean_spectra_physical_radius_virial_vel.pdf')
			else:
				plt.savefig(ion +'mass_bins_median_spectra_physical_radius_virial_vel.pdf')
		else:
			if mean_spectra_bool:
				plt.savefig(ion +'mass_bins_mean_spectra_physical_radius_physical_vel.pdf')
			else:
				plt.savefig(ion +'mass_bins_median_spectra_physical_radius_physical_vel.pdf')

	plt.close()

def plot_for_multiple_gals_by_AGN_lum(spec_output_directory, combined_plots_folder, ion, AGN_lum_bins, radii_bins, text_output, max_abs_vel, virial_vel_bool, virial_radii_bool, 
	mean_spectra_bool, min_halo_mass,max_halo_mass, min_radius, max_radius, plot_chance = 0.0, offset = 0):
	total_num = 0.
	num_covered = 0.
	col_dense_arr = np.array([])
	single_line_plots_made = 0
	no_AGN_all_flux = []
	AGN_all_flux = []
	no_AGN_hi_flux = [] # high radii (164 kpc or higher)
	AGN_hi_flux = []
	no_AGN_low_flux = []
	AGN_low_flux = []
	counts = 0

	vel_bins_for_trystyn = np.arange(-500., 500., 15.)
	final_vels_for_trystyn = np.zeros(np.size(vel_bins_for_trystyn)-1)
	for i in range(0,np.size(vel_bins_for_trystyn)-1):
		final_vels_for_trystyn[i] = (vel_bins_for_trystyn[i]+vel_bins_for_trystyn[i+1])/2.0

	if text_output:
		text_outputs = [[],[],[],[]]


	for j in range(0,np.size(AGN_lum_bins)-1):
		num_in_bin = 0
		final_ion_flux = np.array([])
		final_velocities = np.array([])

		for folder in spec_output_directory:
			if folder[-1] != '/':
				folder += '/'		
			los_files = glob.glob(folder + 'los*')
			buffer_size = len(folder)
			los_nums = np.zeros(np.size(los_files))
			for i in range(0,np.size(los_files)):
				los_nums[i] = los_files[i][int(buffer_size+4):-4]

			for i in range(0,int(1+np.max(los_nums))):	

				los_file = folder + 'los_'+str(i+offset)+'.txt'
				gal_output_file = folder + 'gal_output_'+str(i+offset)+'.hdf5'
				spec_output_file = folder + 'spec.snap'+str(i+offset)+'.hdf5'
				if os.path.isfile(spec_output_file) == False:
					spec_output_file = folder + 'spec.snap_'+str(i+offset)+'.hdf5'
					if os.path.isfile(spec_output_file) == False:
						# print i + offset
						# print 'no spec file for that number'
						continue
	 
				with h5py.File(gal_output_file) as hf:
					GalaxyProperties = hf.get("GalaxyProperties")
					box_size = np.array(GalaxyProperties.get("box_size"))[0]
					gal_R200 = np.array(GalaxyProperties.get("gal_R200"))[0]
					gal_mass = np.array(GalaxyProperties.get("gal_mass"))[0]
					gal_AGN_lum = np.array(GalaxyProperties.get("AGN_lum"))[0]
					gal_stellar_mass = np.array(GalaxyProperties.get("gal_stellar_mass"))
					snap_directory = np.array(GalaxyProperties.get("snap_directory"))[0]
					gal_vel = np.array(GalaxyProperties.get("gal_velocity"))[0]
					gal_coords = np.array(GalaxyProperties.get("gal_coords"))[0]

				lines = np.genfromtxt(los_file, skip_header=1)
				gal = np.array([gal_coords[0], gal_coords[1], gal_coords[2]])/box_size
				if np.size(lines) > 5:
					spec_num = 0
					num_lines = 0
					for line in lines:
						radius = get_correct_radius_of_line(line,gal)*box_size

						with h5py.File(spec_output_file) as hf:
							spec_hubble_velocity = np.array(hf.get('VHubble_KMpS'))
							curr_spectra_folder = hf.get('Spectrum'+str(spec_num))
							curr_spectra = curr_spectra_folder.get(ion)
							ion_flux = np.array(curr_spectra.get("Flux"))
							col_dense = np.array(curr_spectra.get("LotTotalIonColumnDensity"))
			
						length_spectra = np.size(spec_hubble_velocity)
						max_box_vel = spec_hubble_velocity[-1]*(length_spectra+1)/(length_spectra)
						spec_num += 1

						H = max_box_vel/(box_size/1.e3) # stuff with H in Mpc
						gal_hubble_vel = (gal_coords[2]/1.e3)*H # switched gal coords to Mpc
						gal_vel_z = gal_vel[2]
						virial_vel = np.sqrt(G*gal_mass*sol_mass_to_g/(gal_R200*parsec_to_cm*1.e3))


						spec_hubble_velocity = spec_hubble_velocity - (max_box_vel/2.0) # add in peculiar velocity 
						#spec_hubble_velocity = spec_hubble_velocity - (gal_hubble_vel+gal_vel_z) # old. Before new spec (now it centers for you)
						spec_hubble_velocity = np.where(spec_hubble_velocity > max_box_vel/2.0, spec_hubble_velocity-max_box_vel, spec_hubble_velocity)
						spec_hubble_velocity = np.where(spec_hubble_velocity < (-1.0)*max_box_vel/2.0, spec_hubble_velocity+max_box_vel, spec_hubble_velocity)


						if virial_radii_bool:
							radius = radius/gal_R200


						if ((radius > min_radius) & (radius < max_radius)):
							if (gal_AGN_lum < AGN_lum_bins[j+1]) and (gal_AGN_lum > AGN_lum_bins[j]):
								if ((np.log10(gal_mass) > min_halo_mass) & (np.log10(gal_mass) < max_halo_mass)):
									rand_num = np.random.random(1)
									if rand_num <= plot_chance:
										
										### For no discernable reasons some single line plots have an artifiact. 
										### It's consistent in terms of which files they come from but I can't find any way to get rid of it
										### If you call plt.close() before though...it's not there
										plt.close()
										if max_abs_vel != None:
											temp_spec_hubble_velocity = spec_hubble_velocity[np.abs(spec_hubble_velocity) <= max_abs_vel]
											temp_ion_flux = ion_flux[np.abs(spec_hubble_velocity) <= max_abs_vel]
										else:
											temp_spec_hubble_velocity = spec_hubble_velocity
											temp_ion_flux = ion_flux


										plt.plot(temp_spec_hubble_velocity, temp_ion_flux)
										plt.ylim(ymin = -0.05, ymax = 1.05)
										plt.title(ion + ' single line spectra AGN:%.1f' % (gal_AGN_lum))
										plt.savefig(ion+'_single_line_AGN_%.1f_%d.pdf' % (gal_AGN_lum, single_line_plots_made))
										plt.close()
										single_line_plots_made += 1

									if (np.size(final_ion_flux > 0)):
										num_in_bin += 1
										final_ion_flux = np.concatenate([final_ion_flux,ion_flux])
										if virial_vel_bool:
											spec_hubble_velocity /= virial_vel
										final_velocities = np.concatenate([final_velocities, spec_hubble_velocity])


									else:
										num_in_bin += 1
										final_ion_flux = ion_flux
										if virial_vel_bool:
											spec_hubble_velocity /= virial_vel
										final_velocities = spec_hubble_velocity
						num_lines += 1

						# This is where we'll make the finalized flux lists
						digitized = np.digitize(spec_hubble_velocity, vel_bins_for_trystyn)
						flux_means = [ion_flux[digitized == i].mean() for i in range(1, len(vel_bins_for_trystyn))]

						# if math.isnan(flux_means[0]):
						# 	nan_count += 1
						if gal_AGN_lum < 10.:
							no_AGN_all_flux.append(flux_means)
							if ((radius > radii_bins[0]) & (radius < radii_bins[1])):
								no_AGN_low_flux.append(flux_means)
							else:
								no_AGN_hi_flux.append(flux_means)
						else:
							AGN_all_flux.append(flux_means)	
							if ((radius > radii_bins[0]) & (radius < radii_bins[1])):
								AGN_low_flux.append(flux_means)
							else:
								AGN_hi_flux.append(flux_means)

				else:
					radius = get_correct_radius_of_line(lines,gal)*box_size

					with h5py.File(spec_output_file) as hf:
						spec_hubble_velocity = np.array(hf.get('VHubble_KMpS'))
						curr_spectra_folder = hf.get('Spectrum0')
						curr_spectra = curr_spectra_folder.get(ion)
						ion_flux = np.array(curr_spectra.get("Flux"))
						col_dense = np.array(curr_spectra.get("LotTotalIonColumnDensity"))
				
					length_spectra = np.size(spec_hubble_velocity)
					max_box_vel = spec_hubble_velocity[-1]*(length_spectra+1)/(length_spectra)

					H = max_box_vel/(box_size/1.e3) # stuff with H in Mpc
					gal_hubble_vel = (gal_coords[2]/1.e3)*H # switched gal coords to Mpc
					gal_vel_z = gal_vel[2]
					virial_vel = np.sqrt(G*gal_mass*sol_mass_to_g/(gal_R200*parsec_to_cm*1.e3))

					final_vel_arr = spec_hubble_velocity - (max_box_vel/2.0)
					#final_vel_arr = spec_hubble_velocity - (gal_hubble_vel+gal_vel_z) # old. Before new spec (now it centers for you)
					final_vel_arr = np.where(final_vel_arr > max_box_vel/2.0, final_vel_arr-max_box_vel, final_vel_arr)
					final_vel_arr = np.where(final_vel_arr < (-1.0)*max_box_vel/2.0, final_vel_arr+max_box_vel, final_vel_arr)


					if virial_radii_bool:
						radius = radius/gal_R200

					if ((radius > min_radius) & (radius < max_radius)):
						if (gal_AGN_lum < AGN_lum_bins[j+1]) and (gal_AGN_lum > AGN_lum_bins[j]):
							if ((np.log10(gal_mass) > min_halo_mass) & (np.log10(gal_mass) < max_halo_mass)):
								rand_num = np.random.random(1)
								if rand_num <= plot_chance:
									
									### For no discernable reasons some single line plots have an artifiact. 
									### It's consistent in terms of which files they come from but I can't find any way to get rid of it
									### If you call plt.close() before though...it's not there
									plt.close()
									if max_abs_vel != None:
										temp_spec_hubble_velocity = spec_hubble_velocity[np.abs(spec_hubble_velocity) <= max_abs_vel]
										temp_ion_flux = ion_flux[np.abs(spec_hubble_velocity) <= max_abs_vel]
									else:
										temp_spec_hubble_velocity = spec_hubble_velocity
										temp_ion_flux = ion_flux


									plt.plot(temp_spec_hubble_velocity, temp_ion_flux)
									plt.ylim(ymin = -0.05, ymax = 1.05)
									plt.title(ion + ' single line spectra AGN:%.1f' % (gal_AGN_lum))
									plt.savefig(ion+'_single_line_AGN_%.1f_%d.pdf' % (gal_AGN_lum, single_line_plots_made))
									plt.close()
									single_line_plots_made += 1

								if (np.size(final_ion_flux > 0)):
									num_in_bin += 1
									final_ion_flux = np.concatenate([final_ion_flux,ion_flux])
									if virial_vel_bool:
										spec_hubble_velocity /= virial_vel
									final_velocities = np.concatenate([final_velocities, spec_hubble_velocity])


								else:
									num_in_bin += 1
									final_ion_flux = ion_flux
									if virial_vel_bool:
										final_velocities = final_vel_arr/virial_vel
									else:
										final_velocities = final_vel_arr

									# This is where we'll make the finalized flux lists
					digitized = np.digitize(spec_hubble_velocity, vel_bins_for_trystyn)
					flux_means = [ion_flux[digitized == i].mean() for i in range(1, len(vel_bins_for_trystyn))]

					# if math.isnan(flux_means[0]):
					# 	nan_count += 1
					if gal_AGN_lum < 10.:
						no_AGN_all_flux.append(flux_means)
						if ((radius > radii_bins[0]) & (radius < radii_bins[1])):
							no_AGN_low_flux.append(flux_means)
						else:
							no_AGN_hi_flux.append(flux_means)
					else:
						AGN_all_flux.append(flux_means)	
						if ((radius > radii_bins[0]) & (radius < radii_bins[1])):
							AGN_low_flux.append(flux_means)
						else:
							AGN_hi_flux.append(flux_means)



		if combined_plots_folder != None:
			os.chdir(combined_plots_folder)

		try:
			if virial_vel_bool:
				vel_bins = np.linspace(np.min(final_velocities), np.max(final_velocities), (np.max(final_velocities)-np.min(final_velocities))/0.1)
			else:
				vel_bins = np.linspace(np.min(final_velocities), np.max(final_velocities), (np.max(final_velocities)-np.min(final_velocities))/15.0)
			plot_velocities = np.zeros(np.size(vel_bins)-1)
			plot_fluxes = np.zeros(np.size(vel_bins)-1)


			for i in range(0,np.size(vel_bins)-1):
				if mean_spectra_bool:
					plot_fluxes[i] = np.mean(final_ion_flux[np.where((final_velocities>vel_bins[i]) & (final_velocities<vel_bins[i+1]))])
				else:
					plot_fluxes[i] = np.median(final_ion_flux[np.where((final_velocities>vel_bins[i]) & (final_velocities<vel_bins[i+1]))])
				plot_velocities[i] = np.median([vel_bins[i],vel_bins[i+1]])
			
			if max_abs_vel != None:
				plot_fluxes = plot_fluxes[np.abs(plot_velocities) < max_abs_vel]
				plot_velocities = plot_velocities[np.abs(plot_velocities) < max_abs_vel]

			if virial_radii_bool:
				plt.plot(plot_velocities, plot_fluxes, '-', label = '%.1f-%.1f AGN:(n=%d)' %(AGN_lum_bins[j], AGN_lum_bins[j+1], num_in_bin))
			else:
				plt.plot(plot_velocities, plot_fluxes, '-', label = '%.1f-%.1fAGN:(n=%d)' %(AGN_lum_bins[j], AGN_lum_bins[j+1], num_in_bin))
			plt.hold(True)
			if j ==0:
				text_outputs[2*j] = plot_velocities
				text_outputs[2*j+1] = plot_fluxes
			else:
				text_outputs[j+1] = plot_fluxes
		except:
			print 'plot fail for these radii. maybe None in bin? %.1f-%.1f' % (AGN_lum_bins[j], AGN_lum_bins[j+1])
	

	final_no_AGN_all_flux = np.nanmean(no_AGN_all_flux, axis=0)
	final_AGN_all_flux = np.nanmean(AGN_all_flux, axis=0)
	final_no_AGN_low_flux = np.nanmean(no_AGN_low_flux, axis=0)
	final_AGN_low_flux = np.nanmean(AGN_low_flux, axis=0)
	final_no_AGN_hi_flux = np.nanmean(no_AGN_hi_flux, axis=0)
	final_AGN_hi_flux = np.nanmean(AGN_hi_flux, axis=0)

	with open(ion+'_flux_output.txt', 'w') as file:
		file.write(' vel_km_s no_AGN_all AGN_all no_AGN_low AGN_low no_AGN_hi AGN_hi\n')
		for i in range(0,np.size(final_no_AGN_all_flux)):
			file.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' % (final_vels_for_trystyn[i], final_no_AGN_all_flux[i], final_AGN_all_flux[i], final_no_AGN_low_flux[i], final_AGN_low_flux[i], final_no_AGN_hi_flux[i], final_AGN_hi_flux[i]))
		file.close()


	plt.legend(loc='lower left')
	plt.grid()
	plt.ylim(ymin=-0.2, ymax=1)
	plt.title('Flux vs Speed Relative to Central Galaxy for %s [%.1f-%.1f]' % (ion, min_halo_mass,max_halo_mass) )
	plt.ylabel('normalized flux')
	plt.hold(False)
	if virial_vel_bool:
		plt.xlim(xmin=-3, xmax = 3)
		plt.xlabel('vel (vel/virial velocity)')
	else:
		if max_abs_vel != None:
			plt.xlim(xmin=-1.0*(max_abs_vel), xmax=max_abs_vel)
		else:
			plt.xlim(xmin=-700, xmax = 700)
		plt.xlabel('vel (km/s)')
	if virial_radii_bool:
		if virial_vel_bool:
			if mean_spectra_bool:
				plt.savefig(ion +'AGN_lum_mean_spectra_virial_radius_virial_vel.pdf')
			else:
				plt.savefig(ion +'AGN_lum_median_spectra_virial_radius_virial_vel.pdf')
		else:
			if mean_spectra_bool:
				plt.savefig(ion +'AGN_lum_mean_spectra_virial_radius_physical_vel.pdf')
			else:
				plt.savefig(ion +'AGN_lum_median_spectra_virial_radius_physical_vel.pdf')
	else:
		if virial_vel_bool:
			if mean_spectra_bool:
				plt.savefig(ion +'AGN_lum_mean_spectra_physical_radius_virial_vel.pdf')
			else:
				plt.savefig(ion +'AGN_lum_median_spectra_physical_radius_virial_vel.pdf')
		else:
			if mean_spectra_bool:
				plt.savefig(ion +'AGN_lum_mean_spectra_physical_radius_physical_vel.pdf')
			else:
				plt.savefig(ion +'AGN_lum_median_spectra_physical_radius_physical_vel.pdf')

	plt.close()

	# with open(ion+'_flux_lum_bins.txt','w') as file:
	# 	file.write('vel_km_s no_AGN_flux low_AGN_flux hi_AGN_flux\n')
	# 	for i in range(0,np.size(text_outputs[0])):
	# 		file.write('%.6f %.6f %.6f %.6f\n' % (text_outputs[0][i], text_outputs[1][i], text_outputs[2][i], text_outputs[3][i]))
	# 	file.close()

	# with open(ion+'_flux_rad_bins_hi.txt','w') as file:
	# 	file.write('vel_km_s no_AGN_hi AGN_hi\n')
	# 	print 'what is the shape?'
	# 	print np.shape(text_outputs)
	# 	print ''
	# 	for i in range(0,np.size(text_outputs[0])):
	# 		file.write('%.6f %.6f %.6f\n' % (text_outputs[0][i], text_outputs[1][i], text_outputs[2][i]))
	# 	file.close()

def fits(ion, which_survey, spec_output_directory, combined_plots_folder, plot_equ_widths, plot_equ_widths_radii, covering_frac_bool, covering_frac_val, lambda_line, colorbar, offset = 0):
	masses = np.array([])
	stellar_masses = np.array([])
	ssfr = np.array([])
	radii = np.array([])
	virial_radii = np.array([])
	equ_widths = np.array([])
	covered = 0
	total = 0

	for folder in spec_output_directory:
		if folder[-1] != '/':
			folder += '/'
		los_files = glob.glob(folder + 'los*')
		buffer_size = len(folder)
		los_nums = np.zeros(np.size(los_files))
		for i in range(0,np.size(los_files)):
			los_nums[i] = los_files[i][int(buffer_size+4):-4]

		for i in range(0,int(1+np.max(los_nums))):	

			los_file = folder + 'los_'+str(i+offset)+'.txt'
			gal_output_file = folder + 'gal_output_'+str(i+offset)+'.hdf5'
			spec_output_file = folder + 'spec.snap'+str(i+offset)+'.hdf5'
			if os.path.isfile(spec_output_file) == False:
				spec_output_file = folder + 'spec.snap_'+str(i+offset)+'.hdf5'
				if os.path.isfile(spec_output_file) == False:
					# print i + offset
					# print 'no spec file for that number'
					continue

			with h5py.File(gal_output_file, 'r') as hf:
				galaxy_properties = hf.get("GalaxyProperties")
				gal_directory = np.array(galaxy_properties.get('snap_directory'))[0]
				gal_coords = np.array(galaxy_properties.get('gal_coords'))[0]
				box_size = np.array(galaxy_properties.get("box_size"))[0]
				gal_mass = np.array(galaxy_properties.get('gal_mass'))[0]
				gal_stellar_mass = np.array(galaxy_properties.get('log10_smass'))
				gal_ssfr = np.array(galaxy_properties.get('log10_sSFR'))[0]
				gal_R200 = np.array(galaxy_properties.get('gal_R200'))[0]

			lines = np.genfromtxt(los_file, skip_header=1)
			gal = np.array([gal_coords[0], gal_coords[1], gal_coords[2]])/box_size
			if np.size(lines) > 5:
				spec_num = 0
				for line in lines:
					radius = np.sqrt(np.sum(np.power((line[0:3]-gal[0:3]),2)))
					if radius > 0.5:
						radius = (1.0-radius)*box_size
					else:
						radius *= box_size

					with h5py.File(spec_output_file,'r') as hf:
						vel = hf.get('VHubble_KMpS')
						delta_v = np.abs(vel[1]-vel[0])
						spectrum = hf.get('Spectrum'+str(spec_num))
						curr_ion = spectrum.get(ion)
						flux = np.array(curr_ion.get('Flux'))
						equ_width = np.sum(1.-flux)*(delta_v/c_kms)*lambda_line
						total += 1
						if covering_frac_bool:
							if equ_width > covering_frac_val:
								covered += 1

					ssfr = np.concatenate((ssfr,np.array([gal_ssfr])))
					masses = np.concatenate((masses,np.array([gal_mass])))
					stellar_masses = np.concatenate((stellar_masses, np.array(gal_stellar_mass)))
					radii = np.concatenate((radii,np.array([radius])))
					virial_radii = np.concatenate((virial_radii,np.array([radius/gal_R200])))
					equ_widths = np.concatenate((equ_widths,np.array([equ_width])))
					spec_num += 1

			else:
				spec_num = 0
				radius = np.sqrt(np.sum(np.power((lines[0:3]-gal[0:3]),2)))
				if radius > 0.5:
					radius = (1.0-radius)*box_size
				else:
					radius *= box_size

				with h5py.File(spec_output_file,'r') as hf:
					vel = hf.get('VHubble_KMpS')
					delta_v = np.abs(vel[1]-vel[0])
					spectrum = hf.get('Spectrum'+str(spec_num))
					curr_ion = spectrum.get(ion)
					flux = np.array(curr_ion.get('Flux'))
					equ_width = np.sum(1.-flux)*(delta_v/c_kms)*lambda_line
					total += 1
					if covering_frac_bool:
						if equ_width > covering_frac_val:
							covered += 1

					ssfr = np.concatenate((ssfr,np.array([gal_ssfr])))
					masses = np.concatenate((masses,np.array([gal_mass])))
					stellar_masses = np.concatenate((stellar_masses, np.array(gal_stellar_mass)))
					radii = np.concatenate((radii,np.array([radius])))
					virial_radii = np.concatenate((virial_radii,np.array([radius/gal_R200])))
					equ_widths = np.concatenate((equ_widths,np.array([equ_width])))
					spec_num += 1
				spec_num += 1

	if combined_plots_folder != None:
		os.chdir(combined_plots_folder)

	ssfr = ssfr[radii < 510.]
	masses = masses[radii < 510.]
	radii = radii[radii < 510.]
	virial_radii = virial_radii[radii < 510.]
	equ_widths = equ_widths[radii < 510.]
	stellar_masses = stellar_masses[radii < 510.]

	### temporary (hopefully) some where mA some Angstroms
	for i in range(0,np.size(plot_equ_widths)):
		if plot_equ_widths[i] >= 100.:
			plot_equ_widths[i] /= 1000.



	### do a linear fit for smass
	m,b = np.polyfit(radii, np.log10(equ_widths*1.0e3), 1)
	fit_radii = np.linspace(np.min(radii), np.max(radii), 1.0e2)
	fit_equ_width = m*fit_radii + b

	fit_dif = np.log10(equ_widths*1.0e3) - (m*radii + b)
	m1,b1 = np.polyfit(stellar_masses, fit_dif,1)
	err_fit_smass = np.linspace(np.min(stellar_masses), np.max(stellar_masses), 1.e2)
	err_fit_residual = m1*err_fit_smass + b1

	plt.plot(radii, np.log10(equ_widths*1.0e3), '.')
	plt.hold(True)
	plt.plot(fit_radii, fit_equ_width)
	plt.hold(False)
	plt.savefig('fit_the_thing.pdf')

	plt.plot(stellar_masses, fit_dif, '.')
	plt.hold(True)
	plt.plot(err_fit_smass,err_fit_residual)
	plt.hold(False)
	plt.savefig('just_fit_the_difference.pdf')

	### Try again over just cos halo/dwarf range? 
	radii = radii[radii <= 160.]
	equ_widths = equ_widths[radii <= 160.]
	stellar_masses = stellar_masses[radii <= 160.]

	m,b = np.polyfit(radii, np.log10(equ_widths*1.0e3), 1)
	fit_radii = np.linspace(np.min(radii), np.max(radii), 1.0e2)
	fit_equ_width = m*fit_radii + b

	fit_dif = np.log10(equ_widths*1.0e3) - (m*radii + b)
	m1,b1 = np.polyfit(stellar_masses, fit_dif,1)
	err_fit_smass = np.linspace(np.min(stellar_masses), np.max(stellar_masses), 1.e2)
	err_fit_residual = m1*err_fit_smass + b1

	plt.plot(radii, np.log10(equ_widths*1.0e3), '.')
	plt.hold(True)
	plt.plot(fit_radii, fit_equ_width)
	plt.hold(False)
	plt.savefig('fit_the_thing_halo_dwarf.pdf')

	plt.plot(stellar_masses, fit_dif, '.')
	plt.hold(True)
	plt.plot(err_fit_smass,err_fit_residual)
	plt.hold(False)
	plt.savefig('just_fit_the_difference_dwarf.pdf')



	### plot error (fit-data) as a function of impact param 

### pass the coordinates of the line and galaxy (in coordinates where 0,0,0 is center, 1,1,1 is a vertex)
def get_correct_radius_of_line(line,gal):
	delta_x = np.abs(line[0]-gal[0])
	if delta_x > 0.5:
		delta_x = np.abs((1.0-delta_x))

	delta_y = np.abs(line[1]-gal[1])
	if delta_y > 0.5:
		delta_y = np.abs((1.0-delta_y))

	delta_z = np.abs(line[2]-gal[2])
	if delta_z > 0.5:
		delta_z = np.abs((1.0-delta_z))
	
	radius = np.sqrt(delta_x**2.0+delta_y**2.0+delta_z**2.0)
	return radius

def mean_equ_widts_binned_by_radius(ion, spec_output_directory, real_survey_radii, combined_plots_folder, lambda_line, AGN_bool, radii_bins, offset = 0):
	masses = np.array([])
	ssfr = np.array([])
	radii = np.array([])
	virial_radii = np.array([])
	equ_widths = np.array([])
	AGN_vals = np.array([])


	for folder in spec_output_directory:
		if folder[-1] != '/':
			folder += '/'	

		los_files = glob.glob(folder + 'los*')
		buffer_size = len(folder)
		los_nums = np.zeros(np.size(los_files))
		for i in range(0,np.size(los_files)):
			los_nums[i] = los_files[i][int(buffer_size+4):-4]

		for i in range(0,int(1+np.max(los_nums))):	

			los_file = folder + 'los_'+str(i+offset)+'.txt'
			gal_output_file = folder + 'gal_output_'+str(i+offset)+'.hdf5'
			spec_output_file = folder + 'spec.snap'+str(i+offset)+'.hdf5'
			if os.path.isfile(spec_output_file) == False:
				spec_output_file = folder + 'spec.snap_'+str(i+offset)+'.hdf5'
				if os.path.isfile(spec_output_file) == False:
					# print i + offset
					# print 'no spec file for that number'
					continue

			with h5py.File(gal_output_file, 'r') as hf:
				galaxy_properties = hf.get("GalaxyProperties")
				gal_directory = np.array(galaxy_properties.get('snap_directory'))[0]
				gal_coords = np.array(galaxy_properties.get('gal_coords'))[0]
				box_size = np.array(galaxy_properties.get("box_size"))[0]
				gal_mass = np.array(galaxy_properties.get('gal_mass'))[0]
				gal_ssfr = np.array(galaxy_properties.get('log10_sSFR'))[0]
				gal_R200 = np.array(galaxy_properties.get('gal_R200'))[0]
				if AGN_bool:
					gal_AGN = np.array(galaxy_properties.get('AGN_lum'))[0]

			lines = np.genfromtxt(los_file, skip_header=1)
			gal = np.array([gal_coords[0], gal_coords[1], gal_coords[2]])/box_size
			if np.size(lines) > 5:
				spec_num = 0
				for line in lines:
					radius = get_correct_radius_of_line(line,gal)*box_size

					if radius > 500.:
						raise ValueError('radius too big. Radius: %.5f, gal_id: %d' % (radius, i+offset))
						print 'radius is too big'
						print radius
						print i+offset
						print ''

					if radius < 0.0:
						raise ValueError('radius is negative... Radius: %.5f, gal_id: %d' % (radius, i+offset))
						print 'wtf radius'
						print radius
						print i+offset
						print ''

					with h5py.File(spec_output_file,'r') as hf:
						vel = hf.get('VHubble_KMpS')
						delta_v = np.abs(vel[1]-vel[0])
						spectrum = hf.get('Spectrum'+str(spec_num))
						curr_ion = spectrum.get(ion)
						flux = np.array(curr_ion.get('Flux'))
						equ_width = np.sum(1.-flux)*(delta_v/c_kms)*lambda_line

					try:
						ssfr = np.concatenate((ssfr,np.array(gal_ssfr)))
					except:
						ssfr = np.concatenate((ssfr,np.array([gal_ssfr])))

					try:
						masses = np.concatenate((masses,np.array(gal_mass)))
					except:
						masses = np.concatenate((masses,np.array([gal_mass])))

					try:
						radii = np.concatenate((radii,np.array(radius)))
					except:
						radii = np.concatenate((radii,np.array([radius])))

					try:
						virial_radii = np.concatenate((virial_radii,np.array(radius/gal_R200)))
					except:
						virial_radii = np.concatenate((virial_radii,np.array([radius/gal_R200])))

					try:
						equ_widths = np.concatenate((equ_widths,np.array(equ_width)))
					except:
						equ_widths = np.concatenate((equ_widths,np.array([equ_width])))

					if AGN_bool:
						try:
							AGN_vals = np.concatenate((AGN_vals,np.array(gal_AGN)))
						except:
							AGN_vals = np.concatenate((AGN_vals,np.array([gal_AGN])))

					spec_num += 1

			else:
				spec_num = 0
				radius = get_correct_radius_of_line(lines,gal)*box_size

				if radius > 500.:
					raise ValueError('radius too big. Radius: %.5f, gal_id: %d' % (radius, i+offset))
					print 'radius is too big'
					print radius
					print i+offset
					print ''

				if radius < 0.0:
					raise ValueError('radius is negative... Radius: %.5f, gal_id: %d' % (radius, i+offset))
					print 'wtf radius'
					print radius
					print i+offset
					print ''

				with h5py.File(spec_output_file,'r') as hf:
					vel = hf.get('VHubble_KMpS')
					delta_v = np.abs(vel[1]-vel[0])
					spectrum = hf.get('Spectrum'+str(spec_num))
					curr_ion = spectrum.get(ion)
					flux = np.array(curr_ion.get('Flux'))
					equ_width = np.sum(1.-flux)*(delta_v/c_kms)*lambda_line

					try:
						ssfr = np.concatenate((ssfr,np.array(gal_ssfr)))
					except:
						ssfr = np.concatenate((ssfr,np.array([gal_ssfr])))

					try:
						masses = np.concatenate((masses,np.array(gal_mass)))
					except:
						masses = np.concatenate((masses,np.array([gal_mass])))

					try:
						radii = np.concatenate((radii,np.array(radius)))
					except:
						radii = np.concatenate((radii,np.array([radius])))

					try:
						virial_radii = np.concatenate((virial_radii,np.array(radius/gal_R200)))
					except:
						virial_radii = np.concatenate((virial_radii,np.array([radius/gal_R200])))

					try:
						equ_widths = np.concatenate((equ_widths,np.array(equ_width)))
					except:
						equ_widths = np.concatenate((equ_widths,np.array([equ_width])))

					if AGN_bool:
						try:
							AGN_vals = np.concatenate((AGN_vals,np.array(gal_AGN)))
						except:
							AGN_vals = np.concatenate((AGN_vals,np.array([gal_AGN])))

					spec_num += 1


	print 'how many gals do we have?'
	print np.size(radii)

	if combined_plots_folder != None:
		os.chdir(combined_plots_folder)

	final_AGN_equ_widths = np.zeros(np.size(radii_bins)-1)
	final_AGN_equ_widths_err = np.zeros(np.size(radii_bins)-1)
	final_noAGN_equ_widths = np.zeros(np.size(radii_bins)-1)
	final_noAGN_equ_widths_err = np.zeros(np.size(radii_bins)-1)
	final_radii = np.zeros(np.size(radii_bins)-1)

	for i in range(0,np.size(radii_bins)-1):
		if AGN_bool:
			curr_AGN_equ_widths_vals = equ_widths[(radii > radii_bins[i]) & (radii < radii_bins[i+1]) & (AGN_vals > 0.1)]
			final_AGN_equ_widths[i] = np.mean(curr_AGN_equ_widths_vals)
			final_AGN_equ_widths_err[i] = np.std(curr_AGN_equ_widths_vals,ddof=1)


		curr_noAGN_equ_widths_vals = equ_widths[(radii > radii_bins[i]) & (radii < radii_bins[i+1]) & (AGN_vals < 0.1)]
		final_noAGN_equ_widths[i] = np.mean(curr_noAGN_equ_widths_vals)
		final_noAGN_equ_widths_err[i] = np.std(curr_noAGN_equ_widths_vals, ddof=1)
		final_radii[i] = (radii_bins[i]+radii_bins[i+1])/2.0

	indices = np.argsort(real_survey_radii)
	ordered_real_survey_radii = real_survey_radii[indices]
	survey_radii = real_survey_radii[0:np.size(real_survey_radii)/2]


	delta_EW_all = np.log10(equ_widths[AGN_vals>0.1])-np.log10(equ_widths[AGN_vals<0.1])
	radii_all = radii[AGN_vals>0.1]
	delta_EW_all_mean, delta_EW_all_top_err, delta_EW_all_bot_err = jackknife(delta_EW_all, ion+'_all', radii_all, survey_radii)

	delta_EW_low_radius = np.log10(equ_widths[((AGN_vals > 0.1) & (radii<=163.9))])-np.log10(equ_widths[((AGN_vals < 0.1) & (radii<=163.9))])
	radii_low = radii[((radii<=163.9) & (AGN_vals > 0.1))]
	delta_EW_low_radius_mean, delta_EW_low_radius_top_err, delta_EW_low_radius_bot_err = jackknife(delta_EW_low_radius, ion+'_low', radii_low, survey_radii)

	delta_EW_high_radius = np.log10(equ_widths[((AGN_vals > 0.1) & (radii>163.9))])-np.log10(equ_widths[((AGN_vals < 0.1) & (radii>163.9))])
	radii_hi = radii[((radii>163.9) & (AGN_vals > 0.1))]
	delta_EW_high_radius_mean, delta_EW_high_radius_top_err, delta_EW_high_radius_bot_err = jackknife(delta_EW_high_radius, ion+'_high', radii_hi, survey_radii)


	plt.errorbar(final_radii+5., final_noAGN_equ_widths, yerr=final_noAGN_equ_widths_err, fmt = 'b.', ecolor = 'b',label = 'noAGN')
	if AGN_bool:
		plt.hold(True)
		plt.errorbar(final_radii-5.,final_AGN_equ_widths, yerr=final_AGN_equ_widths_err, fmt = 'g.', ecolor = 'g', label = 'AGN')
		plt.hold(False)
	plt.title(ion+' mean equ widths with and without AGN')
	plt.legend()
	plt.xlabel('radius (kpc)')
	plt.ylabel('equivalent width (A)')
	plt.xlim(xmin = 20., xmax = 300.)
	plt.savefig(ion + '_mean_equ_widths_radii_binned.pdf')
	plt.close()

	plt.errorbar(0.,(delta_EW_all_mean),yerr = [[delta_EW_all_top_err-delta_EW_all_mean], [delta_EW_all_mean-delta_EW_all_bot_err]], fmt = 'k.',label = 'all')
	plt.hold(True)
	plt.errorbar(0.75,(delta_EW_low_radius_mean), yerr = [[delta_EW_low_radius_top_err-delta_EW_low_radius_mean], [delta_EW_low_radius_mean-delta_EW_low_radius_bot_err]], fmt = 'r.', label = 'r<=164')
	plt.errorbar(1.25,(delta_EW_high_radius_mean), yerr = [[delta_EW_high_radius_top_err-delta_EW_high_radius_mean], [delta_EW_high_radius_mean-delta_EW_high_radius_bot_err]], fmt = 'b.', label = 'r>164')
	plt.hold(False)
	plt.legend()
	plt.title(ion+ ' difference in mean EW between AGN and non-AGN Galaxies')
	plt.ylabel('Difference in EW (log10)')
	plt.xlim([-1.,2.])
	plt.ylim([-0.3,0.3])
	plt.savefig(ion+'_EW_dif.pdf')
	plt.close()

	### write to file for Trystyn
	EW_file = ion+'_EW.txt'
	deltaEW_file = 'delta_EW_all_log10.txt'

	with open(EW_file, 'w') as file:
		file.write('AGN_EW_low AGN_EW_low_err AGN_EW_hi AGN_EW_hi_err noAGN_EW_low noAGN_EW_low_err noAGN_EW_hi noAGN_EW_hi_err\n')
		file.write('%.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e\n' % (final_AGN_equ_widths[0], final_AGN_equ_widths_err[0], final_AGN_equ_widths[1], final_AGN_equ_widths_err[1], final_noAGN_equ_widths[0], final_noAGN_equ_widths_err[0], final_noAGN_equ_widths[1], final_noAGN_equ_widths_err[1]))
		file.close()

	if os.path.isfile(deltaEW_file):
		with open(deltaEW_file, 'r+') as file:
			with open('temp_file.txt', 'w') as temp_file:
				for line in file:
					temp_file.write(line)
				temp_file.write('%s %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e\n' % (ion, delta_EW_all_mean, delta_EW_all_mean-delta_EW_all_bot_err, delta_EW_all_top_err-delta_EW_all_mean, delta_EW_low_radius_mean, delta_EW_low_radius_mean-delta_EW_low_radius_bot_err, delta_EW_low_radius_top_err-delta_EW_low_radius_mean, delta_EW_high_radius_mean, delta_EW_high_radius_mean-delta_EW_high_radius_bot_err, delta_EW_high_radius_top_err-delta_EW_high_radius_mean))
				temp_file.close()
			file.close()
		os.rename('temp_file.txt', deltaEW_file)

	else:
		with open(deltaEW_file, 'w') as file:
			file.write('ion delta_all delta_all_low_err delta_all_hi_err delta_low_radius delta_low_radius_low_err delta_low_radius_hi_err delta_hi_radius delta_hi_radius_low_err delta_hi_radius_low_err\n')
			file.write('%s %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e\n' % (ion, delta_EW_all_mean, delta_EW_all_mean-delta_EW_all_bot_err, delta_EW_all_top_err-delta_EW_all_mean, delta_EW_low_radius_mean, delta_EW_low_radius_mean-delta_EW_low_radius_bot_err, delta_EW_low_radius_top_err-delta_EW_low_radius_mean, delta_EW_high_radius_mean, delta_EW_high_radius_mean-delta_EW_high_radius_bot_err, delta_EW_high_radius_top_err-delta_EW_high_radius_mean))
			file.close()



def covering_fractions_binned_by_radius(ion, spec_output_directory, combined_plots_folder, lambda_line, AGN_bool, radii_bins, covering_frac_val, offset = 0):
	masses = np.array([])
	ssfr = np.array([])
	radii = np.array([])
	virial_radii = np.array([])
	equ_widths = np.array([])
	AGN_vals = np.array([])


	for folder in spec_output_directory:
		if folder[-1] != '/':
			folder += '/'	

		los_files = glob.glob(folder + 'los*')
		buffer_size = len(folder)
		los_nums = np.zeros(np.size(los_files))
		for i in range(0,np.size(los_files)):
			los_nums[i] = los_files[i][int(buffer_size+4):-4]

		for i in range(0,int(1+np.max(los_nums))):	

			los_file = folder + 'los_'+str(i+offset)+'.txt'
			gal_output_file = folder + 'gal_output_'+str(i+offset)+'.hdf5'
			spec_output_file = folder + 'spec.snap'+str(i+offset)+'.hdf5'
			if os.path.isfile(spec_output_file) == False:
				spec_output_file = folder + 'spec.snap_'+str(i+offset)+'.hdf5'
				if os.path.isfile(spec_output_file) == False:
					# print i + offset
					# print 'no spec file for that number'
					continue

			with h5py.File(gal_output_file, 'r') as hf:
				galaxy_properties = hf.get("GalaxyProperties")
				gal_directory = np.array(galaxy_properties.get('snap_directory'))[0]
				gal_coords = np.array(galaxy_properties.get('gal_coords'))[0]
				box_size = np.array(galaxy_properties.get("box_size"))[0]
				gal_mass = np.array(galaxy_properties.get('gal_mass'))[0]
				gal_ssfr = np.array(galaxy_properties.get('log10_sSFR'))[0]
				gal_R200 = np.array(galaxy_properties.get('gal_R200'))[0]
				if AGN_bool:
					gal_AGN = np.array(galaxy_properties.get('AGN_lum'))[0]

			lines = np.genfromtxt(los_file, skip_header=1)
			gal = np.array([gal_coords[0], gal_coords[1], gal_coords[2]])/box_size
			if np.size(lines) > 5:
				spec_num = 0
				for line in lines:
					radius = get_correct_radius_of_line(line,gal)*box_size

					if radius > 500.:
						raise ValueError('radius too big. Radius: %.5f, gal_id: %d' % (radius, i+offset))
						print 'radius is too big'
						print radius
						print i+offset
						print ''

					if radius < 0.0:
						raise ValueError('radius is negative... Radius: %.5f, gal_id: %d' % (radius, i+offset))
						print 'wtf radius'
						print radius
						print i+offset
						print ''

					with h5py.File(spec_output_file,'r') as hf:
						vel = hf.get('VHubble_KMpS')
						delta_v = np.abs(vel[1]-vel[0])
						spectrum = hf.get('Spectrum'+str(spec_num))
						curr_ion = spectrum.get(ion)
						flux = np.array(curr_ion.get('Flux'))
						equ_width = np.sum(1.-flux)*(delta_v/c_kms)*lambda_line

					try:
						ssfr = np.concatenate((ssfr,np.array(gal_ssfr)))
					except:
						ssfr = np.concatenate((ssfr,np.array([gal_ssfr])))

					try:
						masses = np.concatenate((masses,np.array(gal_mass)))
					except:
						masses = np.concatenate((masses,np.array([gal_mass])))

					try:
						radii = np.concatenate((radii,np.array(radius)))
					except:
						radii = np.concatenate((radii,np.array([radius])))

					try:
						virial_radii = np.concatenate((virial_radii,np.array(radius/gal_R200)))
					except:
						virial_radii = np.concatenate((virial_radii,np.array([radius/gal_R200])))

					try:
						equ_widths = np.concatenate((equ_widths,np.array(equ_width)))
					except:
						equ_widths = np.concatenate((equ_widths,np.array([equ_width])))

					if AGN_bool:
						try:
							AGN_vals = np.concatenate((AGN_vals,np.array(gal_AGN)))
						except:
							AGN_vals = np.concatenate((AGN_vals,np.array([gal_AGN])))

					spec_num += 1

			else:
				spec_num = 0
				radius = get_correct_radius_of_line(lines,gal)*box_size

				if radius > 500.:
					raise ValueError('radius too big. Radius: %.5f, gal_id: %d' % (radius, i+offset))
					print 'radius is too big'
					print radius
					print i+offset
					print ''

				if radius < 0.0:
					raise ValueError('radius is negative... Radius: %.5f, gal_id: %d' % (radius, i+offset))
					print 'wtf radius'
					print radius
					print i+offset
					print ''

				with h5py.File(spec_output_file,'r') as hf:
					vel = hf.get('VHubble_KMpS')
					delta_v = np.abs(vel[1]-vel[0])
					spectrum = hf.get('Spectrum'+str(spec_num))
					curr_ion = spectrum.get(ion)
					flux = np.array(curr_ion.get('Flux'))
					equ_width = np.sum(1.-flux)*(delta_v/c_kms)*lambda_line

					try:
						ssfr = np.concatenate((ssfr,np.array(gal_ssfr)))
					except:
						ssfr = np.concatenate((ssfr,np.array([gal_ssfr])))

					try:
						masses = np.concatenate((masses,np.array(gal_mass)))
					except:
						masses = np.concatenate((masses,np.array([gal_mass])))

					try:
						radii = np.concatenate((radii,np.array(radius)))
					except:
						radii = np.concatenate((radii,np.array([radius])))

					try:
						virial_radii = np.concatenate((virial_radii,np.array(radius/gal_R200)))
					except:
						virial_radii = np.concatenate((virial_radii,np.array([radius/gal_R200])))

					try:
						equ_widths = np.concatenate((equ_widths,np.array(equ_width)))
					except:
						equ_widths = np.concatenate((equ_widths,np.array([equ_width])))

					if AGN_bool:
						try:
							AGN_vals = np.concatenate((AGN_vals,np.array(gal_AGN)))
						except:
							AGN_vals = np.concatenate((AGN_vals,np.array([gal_AGN])))

					spec_num += 1


	if combined_plots_folder != None:
		os.chdir(combined_plots_folder)

	AGN_covering_frac = np.zeros(np.size(radii_bins)-1)
	AGN_covering_frac_err = np.zeros(np.size(radii_bins)-1)
	noAGN_covering_frac = np.zeros(np.size(radii_bins)-1)
	noAGN_covering_frac_err = np.zeros(np.size(radii_bins)-1)
	final_radii = np.zeros(np.size(radii_bins)-1)

	for i in range(0,np.size(radii_bins)-1):
		if AGN_bool:
			AGN_covered = equ_widths[(radii > radii_bins[i]) & (radii < radii_bins[i+1]) & (AGN_vals > 0.1) & (equ_widths > covering_frac_val)]
			AGN_total = equ_widths[(radii > radii_bins[i]) & (radii < radii_bins[i+1]) & (AGN_vals > 0.1)]
			try:
				frac_est = float(np.size(AGN_covered))/np.size(AGN_total)
			except:
				frac_est = 0.0
			AGN_covering_frac[i] = frac_est
			try:
				AGN_covering_frac_err[i] = np.sqrt(frac_est*(1.-frac_est)/float(np.size(AGN_total)))
			except:
				AGN_covering_frac_err[i] = 1.0

		noAGN_covered = equ_widths[(radii > radii_bins[i]) & (radii < radii_bins[i+1]) & (AGN_vals < 0.1) & (equ_widths > covering_frac_val)]
		noAGN_total = equ_widths[(radii > radii_bins[i]) & (radii < radii_bins[i+1]) & (AGN_vals < 0.1)]
		try:
			frac_est = float(np.size(noAGN_covered))/np.size(noAGN_total)
		except:
			frac_est = 0.0
		noAGN_covering_frac[i] = frac_est
		try:
			noAGN_covering_frac_err[i] = np.sqrt(frac_est*(1.-frac_est)/float(np.size(noAGN_total)))
		except:
			noAGN_covering_frac_err[i] = 1.0

		final_radii[i] = (radii_bins[i]+radii_bins[i+1])/2.0

	plt.errorbar(final_radii+5., noAGN_covering_frac, yerr=noAGN_covering_frac_err, fmt = 'b.', ecolor = 'b',label = 'noAGN n=%d' % (np.size(noAGN_total)))
	if AGN_bool:
		plt.hold(True)
		plt.errorbar(final_radii-5.,AGN_covering_frac, yerr=AGN_covering_frac_err, fmt = 'g.', ecolor = 'g', label = 'AGN n=%d' % (np.size(AGN_total)))
		plt.hold(False)
	plt.title(ion+' covering fractions with and without AGN')
	plt.xlabel('radius (kpc')
	plt.ylabel('covering fraction')
	plt.ylim(ymin=0.0,ymax=1.0)
	plt.xlim(xmin = 20., xmax = 300.)
	plt.legend()
	plt.savefig(ion + '_covering_frac_radii_binned.pdf')
	plt.close()

	with open(ion+'_covering_frac.txt', 'w') as file:
		file.write('AGN_covering_frac_low_radius AGN_covering_frac_err_low_radius AGN_covering_frac_hi_radius AGN_covering_frac_err_hi_radius noAGN_covering_frac_low_radius noAGN_covering_frac_err_low_radius noAGN_covering_frac_hi_radius noAGN_covering_frac_err_hi_radius\n')
		file.write('%.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e\n' % (AGN_covering_frac[0], AGN_covering_frac_err[0], AGN_covering_frac[1], AGN_covering_frac_err[1], noAGN_covering_frac[0], noAGN_covering_frac_err[0], noAGN_covering_frac[1], noAGN_covering_frac_err[1]))
		file.close()

def jackknife(data, ion, radii, survey_radii):
	estimators = np.zeros(np.size(survey_radii))
	tol = 0.2
	for i in range(0,np.size(survey_radii)):
		mask = np.ones(np.size(data),dtype=bool)
		indices_to_ignore = np.argwhere((radii > survey_radii[i]-tol) & (radii < survey_radii[i]+tol))
		mask[indices_to_ignore] = 0
		subset = data[mask]
		estimators[i] = np.mean(subset)


	# for i in range(np.size(data)):
	# 	mask = np.ones(np.size(data),dtype=bool)
	# 	mask[i] = 0
	# 	subset = data[mask]

	# 	estimators[i] = np.mean(subset)

	jackknife_mean = np.mean(estimators)
	plt.plot(estimators, 'k.')
	plt.title('%.3e %.3e' % (np.max(estimators), np.min(estimators)))
	plt.savefig('%s_delta_estimators.pdf' % (ion))
	plt.close()
	# jackknife_err = ((np.size(data)-1.)/(np.size(data)))*np.sum((estimators-mean)**2.0)
	top_range = np.max(estimators)
	bot_range = np.min(estimators)
	return jackknife_mean, top_range, bot_range

def radius_check(gal_radius, ordered_cos_radii):
	index = bisect.bisect_left(ordered_cos_radii, gal_radius)
	if index == np.size(ordered_cos_radii):
		if gal_radius <  ordered_cos_radii[index-1] + 0.25:
			return True
		else:
			return False

	elif ((gal_radius <  ordered_cos_radii[index-1] + 0.25) or (gal_radius > ordered_cos_radii[index] - 0.25)):
		return True

	else:
		return False

# lookup file: use it to pull the 3d matrix lookup table uses
# spec_output_fil: the file specwizard outputs
# spectrum/ion/redshift: Get passed to the file becuase the code that uses it will iterate over ions/spectra. 
# Example: spectrum = 'Spectrum0', ion = 'h1', redshift = 0.205
def mass_estimates(lookup_file, spec_output_file, spectrum, ion, redshift): 
	tol = 0.3 # only looks at where optical_depth is greater than this value

	# read specwizard file
	with h5py.File(spec_output_file, 'r') as file: 
		curr_spec = file.get(spectrum)
		ion = curr_spec.get(ion)
		optical_depth = np.array(ion.get('OpticalDepth'))
		col_dense = np.array(ion.get('LogTotalIonColumnDensity'))
		vel_space = ion.get('RedshiftSpaceOpticalDepthWeighted')
		temp = np.array(vel_space.get("Temperature_K"))

		h1 = curr_spec.get('h1')
		h1_vel_space = h1.get('RedshiftSpaceOpticalDepthWeighted')
		num_dens = np.array(h1_vel_space.get("NIon_CM3"))
		file.close()

	# Get correction factor and change optical depth array to column density array
	total_optical_depth = np.sum(optical_depth)
	tau_to_col_factor = (10.**col_dense)/total_optical_depth
	col_dense_arr = optical_depth*tau_to_col_factor
	test_col_dense_arr = optical_depth[optical_depth >= tol]*tau_to_col_factor

	# initialize arrays
	neutral_fractions = np.empty(np.size(num_dens))

	# read lookup table
	with h5py.File(lookup_file, 'r') as file:
		ion_bal = np.array(file.get('ionbal'))
		log_dens = np.array(file.get('logd'))
		log_T = np.array(file.get('logt'))
		lookup_redshift = np.array(file.get('redshift'))
		file.close()

	# get nearest redshift table uses (redhisft is the same for all points in spectrum so outside of for loop)
	redshift_index = bisect.bisect_left(lookup_redshift, redshift)
	redshift_index = return_best_index(lookup_redshift, redshift_index, redshift)

	# get nearest number density and temperature for each point in spectra
	# use that to populate neutral fraction array
	for i in range(0,np.size(num_dens)):
		dense_index = bisect.bisect_left(log_dens, num_dens[i])
		dense_index = return_best_index(log_dens, dense_index, num_dens[i])

		temp_index = bisect.bisect_left(log_T, np.log10(temp[i]))
		temp_index = return_best_index(log_T, temp_index, np.log10(temp[i]))

		neutral_fractions[i] = ion_bal[dense_index][temp_index][redshift_index]

	# get neutral column density array
	H_col_dense_arr = col_dense_arr/neutral_fractions
	test_H_col_dense_arr = test_col_dense_arr/neutral_fractions[optical_depth >= tol]

	# get total neutral column
	H_column = np.log10(np.sum(H_col_dense_arr))
	test_H_column = np.log10(np.sum(test_H_col_dense_arr))
	delta = H_column- test_H_column

	return test_H_column, delta


def return_best_index(array, left_index, value):
	delta = np.abs(array[left_index] - value)
	if np.abs(array[left_index+1] -value) < delta:
		return left_index+1
	else:
		return left_index

def get_line_kinematics(flux, velocity, temperature, optical_depth, gals_num, spec_num, radius, gal_mass, gal_ssfr, make_realistic_bool, rest_wavelength=None, redshift=None, directory_with_COS_LSF='./'):
	# parameters for making the spectra realistic
	pix_per_bin = 8
	snr = 10.
	cos_lsf_bool=True
	correlated_pixels = True
	if correlated_pixels:
		snr_pow = 0.38
	else:
		snr_pow = 0.5
	vel_kms=True
	std=20
	num_sigma_in_gauss=3

	# requirements for identifying a line, want it to be visible after convolved to (so, visible by COS)
	depth_tol = 2./snr
	min_val_limit = 1.0-depth_tol
	EAGLE_delta_vel = 0.40
	COS_delta_vel = 2.5
	sim_px_per_cos_px = (COS_delta_vel*pix_per_bin)/(EAGLE_delta_vel)
	seperation_tol = 2*COS_delta_vel*pix_per_bin # km/s, makes it so they must be seperated by twice the width of a bin in COS (trough, not trough, trough. as tight as I can do)
	extra_indices_bool = False

	if make_realistic_bool:
		velocity, wavelengths, flux = gen_lsf.do_it_all(velocity, flux, rest_wavelength, redshift, pix_per_bin, snr, cos_lsf_bool=cos_lsf_bool, correlated_pixels = correlated_pixels, vel_kms=vel_kms, chan=None, std=std, num_sigma_in_gauss=num_sigma_in_gauss, directory_with_COS_LSF=directory_with_COS_LSF)

	# identifies if there are a bunch of zero points. If so adds a minima, because the rest of the script only catches if it's a strict minima (BELOW the ones on either side)
	extra_indices = []
	zero_indices = np.argwhere(flux == 0.0)
	if np.size(zero_indices) > 1:
		first_index = zero_indices[0]
		for i in range(0,np.size(zero_indices)-1):
			previous_index = zero_indices[i]
			next_index = zero_indices[i+1]
			if ((next_index == previous_index + 1) & (i < np.size(zero_indices)-2)):
				continue
			else:
				extra_indices.append(int(np.median([first_index, previous_index])))
				extra_indices_bool = True
				first_index = next_index


	if np.size(flux) != np.size(velocity):
		raise ValueError('in get line kinematics, flux and velocity arrays are different sizes')

	min_indices = [] # identify strict minima
	for i in range(1,np.size(flux)-1):
		if make_realistic_bool:
			if ((flux[i] < min_val_limit) & (flux[i] < flux[i-1]) & (flux[i] < flux[i+1])):
				min_indices.append(i)
		else:
			### if real spectra make the same convolution that it would if it were made realistic, if this is < min_val_limit, let it through
			if ((flux[i] < flux[i-1]) & (flux[i] < flux[i+1]) & (flux[i]-np.random.normal(0, flux[i]/(snr*pix_per_bin**snr_pow), 1) < min_val_limit)):
				min_indices.append(i)



		# if ((flux[i] < flux[i-1]) & (flux[i] < flux[i+1])):
		# 	if ((make_realistic_bool) & (flux[i] < min_val_limit)):
		# 		min_indices.append(i)
		# 	### if real spectra make the same convolution that it would if it were made realistic, if this is < min_val_limit, let it through
		# 	elif ((make_realistic_bool == False) & (flux[i]-np.random.normal(flux[i], flux[i]/snr, 1) < min_val_limit)):
		# 		min_indices.append(i)

	min_indices = np.array(sorted(extra_indices + min_indices)) # merge the two min arrays and keep them sorted
	seperation_mask = np.zeros(np.size(min_indices))+1

	# check that minima meet seperation, else mask weaker one
	if np.size(min_indices) > 1:
		for i in range(0,np.size(min_indices)-1):
			for j in range(i+1,np.size(min_indices)):
				if np.abs(velocity[min_indices[i]]-velocity[min_indices[j]]) < seperation_tol:
					if flux[min_indices[i]] <= flux[min_indices[j]]:
						seperation_mask[j] = 0
					else:
						seperation_mask[i] = 0

	min_indices = min_indices[seperation_mask == 1]
	num_minima = np.size(min_indices)
	depth = np.empty(num_minima)
	FWHM = []
	centroid_vel = np.empty(num_minima)
	prominence_mask = np.zeros(num_minima) + 1

	for i in range(0,np.size(min_indices)):
		centroid_vel[i] = velocity[min_indices[i]]
		depth[i] = 1.-flux[min_indices[i]]

	for i in range(0,num_minima):
		index = min_indices[i]
		# how high does the flux have to climb before next min (i.e. how prominent must the trough be) based on snr at trough (2 sigma) or .1 where signal is low (still want unique trough)
		min_recovery = np.max([flux[index] + 2.*(flux[index]/snr), flux[index] + 0.1])
		missing_half = False

		if index <= 1:
			# print 'no half max intersection found on left side'
			missing_half = True
		elif index >= np.size(flux)-2:
			# print 'no half max intersection found on right side'
			missing_half = True
			
		for j in range(index+1,np.size(flux)-1):
			if ((num_minima > 1) & (i < num_minima -1)):
				if velocity[j] >= centroid_vel[i+1]:

					if depth[i] == depth[i+1]:
						'triggered eq'
						extra_index = int(np.mean([min_indices[i], min_indices[i+1]]))
						prominence_mask[i+1] = 0
						depth[i] = depth[i] # redundant, but clearer
						min_indices[i] = extra_index
						index = extra_index # keeps where we start the backwards search consistent
						centroid_vel[i] = velocity[extra_index]

					elif depth[i] > depth[i+1]:
						prominence_mask[i+1] = 0

					else:
						prominence_mask[i] = 0
						break

			if flux[j+1] >= min_recovery:
				right_index = j
				break

			if j == np.size(flux)-2:
				# print 'no half max intersection found on right side'
				missing_half = True


		for j in np.arange(index-1,0,-1):
			lookback = 1
			if ((num_minima > 1) & (i > 0) & (prominence_mask[i] != 0)):
				if velocity[j] < centroid_vel[i-lookback]:
					if depth[i] < depth[i-lookback]:
						prominence_mask[i] = 0
						break
					if prominence_mask[i-lookback] == 0:
						if i-lookback == 0:
							if j == 1:
								missing_half = True
							else:
								continue
						else:
							lookback += 1
					else:
						plt.plot(velocity, flux)
						plt.savefig('cause_error.pdf')
						plt.close()
						print num_minima
						print min_indices
						print depth
						print centroid_vel
						print ''
						raise ValueError('I thought this was impossible because it would have been caught in the forward look above')

			if flux[j-1] >= min_recovery:
				left_index = j
				break

			if j == 1:
				# print 'no half max intersection found on left side'
				missing_half = True

		if prominence_mask[i] == 0:
			continue
		elif missing_half:
			FWHM.append(1.e6)
		else:
			try:
				FWHM.append(velocity[right_index]-velocity[left_index])
			except:
				plt.plot(velocity, flux)
				plt.savefig('cause_error.pdf')
				plt.close()
				print j
				print lookback
				print num_minima
				print min_indices
				print centroid_vel
				print ''
				raise ValueError('Left not found?')


	min_indices = min_indices[prominence_mask == 1]
	depth = depth[prominence_mask == 1]
	centroid_vel = centroid_vel[prominence_mask == 1]

	temps = np.zeros(np.size(min_indices))
	for i in range(0,np.size(min_indices)):
		temps[i] = np.sum(temperature[optical_depth >= 0.01]*optical_depth[optical_depth >= 0.01])/np.sum(optical_depth[optical_depth >= 0.01])

	num_minima = np.size(min_indices)

	# ### if doing single line plots
	# for i in range(0,np.size(min_indices)):
	# 	plt.plot(velocity, flux, 'k', label = 'velocity=%d, FWHM=%d, depth=%.2f' % (centroid_vel[i], FWHM[i], depth[i]))
	# 	plt.hold(True)
	# 	plt.axvline(centroid_vel[i])
	# plt.hold(False)
	# plt.xlabel('Velocity (km/s)')
	# plt.ylabel('Flux')
	# plt.legend(loc='lower right')
	# if make_realistic_bool:
	# 	plt.title(r'Spectra w/ COS LSF: b=%.0d, $M_{halo}=%.2E, sSFR=%.2f$' % (radius, gal_mass, gal_ssfr))
	# 	plt.savefig('sing_lines_real_%d_%d.pdf' % (gals_num, spec_num))
	# else:
	# 	plt.title(r'Clean Spectra: b=%.0d, $M_{halo}=%.2E, sSFR=%.2f$' % (radius, gal_mass, gal_ssfr))
	# 	plt.savefig('sing_lines_%d_%d.pdf' % (gals_num, spec_num))

	return num_minima, centroid_vel, np.array(FWHM), depth, temps

def neutral_columns_plot(cols, H_cols, delta, radii, virial_radii, R200, smasses, masses, ssfr, dens, gas_densities, temps, mean_bool, virial_radii_bool, pop_str):
	print 'how many points do we have?'
	print np.size(radii)
	print ''
	H_cols = np.where((H_cols)>=22., cols, H_cols)
	### Prochaska data for comparisons
	proch_radii = np.arange(20., 160., 10.)

	tens = np.zeros(np.size(proch_radii)) + 10.
	proch_ann_mass = np.power(tens, np.array([8.9, 9.9, 9.4, 8.2, 10.0, 7.3, 10.6, 9.2, 8.2, 10.2, 9.0, 8.6, 8.1, 8.4]))
	adj_proch_ann_mass = np.power(tens, np.array([8.9, 9.9, 9.4, 8.2, 10.0, 7.3, 9.6, 9.2, 8.2, 9.2, 9.0, 8.6, 8.1, 8.4]))
	proch_mass_err_logged = np.array([0.3, 0.4, 0.3, 0.3, 0.3, 0.3, 0.4, 0.2, 0.2, 0.4, 0.3, 0.3, 0.3, 0.3])
	proch_mass_err = np.power(tens, np.array([8.9, 9.9, 9.4, 8.2, 10.0, 7.3, 9.6, 9.2, 8.2, 9.2, 9.0, 8.6, 8.1, 8.4]))*np.log(10.)*np.array([0.3, 0.4, 0.3, 0.3, 0.3, 0.3, 0.4, 0.2, 0.2, 0.4, 0.3, 0.3, 0.3, 0.3])

	proch_cum_mass = np.zeros(np.size(proch_ann_mass))
	adj_proch_cum_mass = np.zeros(np.size(proch_ann_mass))
	proch_cum_mass_err = np.zeros(np.size(proch_ann_mass))
	proch_cum_mass_err_logged = np.zeros(np.size(proch_ann_mass))
	for i in range(0,np.size(proch_ann_mass)):
		proch_cum_mass[i] = np.sum(proch_ann_mass[0:i+1])
		adj_proch_cum_mass[i] = np.sum(adj_proch_ann_mass[0:i+1])
		proch_cum_mass_err[i] = np.sqrt(np.sum(proch_mass_err[0:i+1]**2.))/np.sqrt(np.size(proch_mass_err[0:i+1]))
		proch_cum_mass_err_logged[i] = np.sqrt(np.sum(proch_mass_err_logged[0:i+1]**2.))/np.sqrt(np.size(proch_mass_err_logged[0:i+1]))

	if pop_str == 'old':
		used_mass = np.log10(smasses)

		### For smass
		upper_mass = np.array([9.7, 15., 15.])
		lower_mass = np.array([5., 9.7, 9.7])

		upper_ssfr = np.array([-5., -5., -11.])
		lower_ssfr = np.array([-15., -11., -15.])

		labels = ['Low Mass', 'Blue', 'Red']
		colors = ['k', 'b', 'r']

	elif pop_str == 'new':
		used_mass = np.log10(masses)

		### For halo mass
		upper_mass = np.array([11.7, 12.5, 14.])
		lower_mass = np.array([10., 11.7, 12.5])

		upper_ssfr = np.array([-5., -5., -5.])
		lower_ssfr = np.array([-15., -15., -15.])

		labels = ['M_halo<11.7', 'M_halo=11.7-12.5', 'M_halo>12.5']
		colors = ['k', 'b', 'r']

	elif pop_str == 'proch':
		used_mass = np.log10(masses)

		upper_mass = np.array([15.])
		lower_mass = np.array([5.])

		upper_ssfr = np.array([-5.])
		lower_ssfr = np.array([-15.])

		labels = ['EAGLE']
		colors = ['k']

	else:
		raise ValueError('population string does not match any preset values (old, new, proch)')


	### initialize array of arrays that will be iterated over to plot
	med_cols_list = []
	col_err_top_list = []
	col_err_bot_list = []
	vol_list = []
	plot_radii_list = []
	mass_annuli_list = []
	mass_annuli_top_err_list = []
	mass_annuli_bot_err_list = []
	cum_mass_list = []
	cum_mass_top_err_list = []
	cum_mass_bot_err_list = []
	temps_list = []
	gas_densities_list = []

	tens = np.zeros(np.size(cols)) + 10.
	cols = np.power(tens, cols)
	H_cols = np.power(tens, H_cols)
	if virial_radii_bool:
		delta_r = 0.1
		stagger = [0.0, -0.01, 0.01]
	else:
		delta_r = 10
		stagger = [0.0, -1.5, 1.5]

	for i in range(0,np.size(upper_ssfr)):
		temps_list.append(temps[((used_mass > lower_mass[i]) & (used_mass < upper_mass[i]) & (ssfr > lower_ssfr[i]) & (ssfr < upper_ssfr[i]))])
		gas_densities_list.append(gas_densities[((used_mass > lower_mass[i]) & (used_mass < upper_mass[i]) & (ssfr > lower_ssfr[i]) & (ssfr < upper_ssfr[i]))])


		## Mass estimates
		if virial_radii_bool:
			bins = np.arange(np.min(virial_radii), np.max(virial_radii), delta_r)
		else:
			bins = np.arange(20., 160., delta_r)
		med_cols = np.zeros(np.size(bins)-1)
		col_err_top = np.zeros(np.size(bins)-1)
		col_err_bot = np.zeros(np.size(bins)-1)
		vol = np.zeros(np.size(bins)-1)
		plot_radii = np.zeros(np.size(bins)-1)
		mass_annuli = np.zeros(np.size(bins)-1)
		mass_annuli_top_err = np.zeros(np.size(bins)-1)
		mass_annuli_bot_err = np.zeros(np.size(bins)-1)
		cum_mass = np.zeros(np.size(bins)-1)
		cum_mass_top_err = np.zeros(np.size(bins)-1)
		cum_mass_bot_err = np.zeros(np.size(bins)-1)


		for j in range(0,np.size(bins)-1):
			if virial_radii_bool:
				temp_cols = H_cols[((virial_radii >= bins[j]) & (virial_radii < bins[j+1]) & (used_mass > lower_mass[i]) & (used_mass < upper_mass[i]) & (ssfr > lower_ssfr[i]) & (ssfr < upper_ssfr[i]))]
				temp_virial_radii = virial_radii[((virial_radii >= bins[j]) & (virial_radii < bins[j+1]) & (used_mass > lower_mass[i]) & (used_mass < upper_mass[i]) & (ssfr > lower_ssfr[i]) & (ssfr < upper_ssfr[i]))]
				temp_R200 = R200[((virial_radii >= bins[j]) & (virial_radii < bins[j+1]) & (used_mass > lower_mass[i]) & (used_mass < upper_mass[i]) & (ssfr > lower_ssfr[i]) & (ssfr < upper_ssfr[i]))]
				temp_masses = masses[((virial_radii >= bins[j]) & (virial_radii < bins[j+1]) & (used_mass > lower_mass[i]) & (used_mass < upper_mass[i]) & (ssfr > lower_ssfr[i]) & (ssfr < upper_ssfr[i]))]
			else:
				temp_cols = H_cols[((radii >= bins[j]) & (radii < bins[j+1]) & (used_mass > lower_mass[i]) & (used_mass < upper_mass[i]) & (ssfr > lower_ssfr[i]) & (ssfr < upper_ssfr[i]))]

			if np.size(temp_cols) > 0:
				if mean_bool == False:
					if virial_radii_bool:
						fiducial_value = np.percentile(temp_R200, 50.)
						fiducial_mass = np.percentile(temp_masses, 50.)
					med_cols[j] = np.median(temp_cols)
					col_err_top[j] = np.percentile(temp_cols, 84.)
					col_err_bot[j] = np.percentile(temp_cols, 16.)
				else:
					if virial_radii_bool:
						fiducial_value = np.mean(temp_R200)
						fiducial_mass = np.mean(temp_masses)
					med_cols[j] = np.mean(temp_cols)
					col_err_top[j] = np.std(temp_cols, ddof=1)#np.size(temp_cols)-1)
					col_err_bot[j] = np.std(temp_cols, ddof=1)#np.size(temp_cols)-1)
					if col_err_top[j] == 0:
						print temp_cols

				if virial_radii_bool:
					vol[j] = pi*((bins[j+1]*fiducial_value*1.e3*parsec_to_cm)**2.-(bins[j]*fiducial_value*1.e3*parsec_to_cm)**2.)
				else:
					vol[j] = pi*((bins[j+1]*1.e3*parsec_to_cm)**2.-(bins[j]*1.e3*parsec_to_cm)**2.)
				plot_radii[j] = (bins[j]+bins[j+1])/2.0

				if virial_radii_bool:
					mass_annuli[j] = ((m_p*mu*med_cols[j]*vol[j])/sol_mass_to_g)/fiducial_mass
					mass_annuli_top_err[j] = ((m_p*mu*col_err_top[j]*vol[j])/sol_mass_to_g)/fiducial_mass
					mass_annuli_bot_err[j] = ((m_p*mu*col_err_bot[j]*vol[j])/sol_mass_to_g)/fiducial_mass
					cum_mass[j] = np.nansum(mass_annuli)
					cum_mass_top_err[j] = np.sqrt(np.nansum(mass_annuli_top_err**2.))
					cum_mass_bot_err[j] = np.sqrt(np.nansum(mass_annuli_bot_err**2.))
				else:
					mass_annuli[j] = (m_p*mu*med_cols[j]*vol[j])/sol_mass_to_g
					mass_annuli_top_err[j] = (m_p*mu*col_err_top[j]*vol[j])/sol_mass_to_g
					mass_annuli_bot_err[j] = (m_p*mu*col_err_bot[j]*vol[j])/sol_mass_to_g
					cum_mass[j] = np.nansum(mass_annuli)
					cum_mass_top_err[j] = np.sqrt(np.nansum(np.power(mass_annuli_top_err,2.)))
					cum_mass_bot_err[j] = np.sqrt(np.nansum(np.power(mass_annuli_bot_err,2.)))
			else:
				plot_radii[j] = (bins[j]+bins[j+1])/2.0

				mass_annuli[j] = np.nan
				mass_annuli_top_err[j] = np.nan
				mass_annuli_bot_err[j] = np.nan
				cum_mass[j] = np.nan
				cum_mass_top_err[j] = np.nan
				cum_mass_bot_err[j] = np.nan

		med_cols_list.append(med_cols)
		vol_list.append(vol)
		plot_radii_list.append(plot_radii)
		mass_annuli_list.append(mass_annuli)
		cum_mass_list.append(cum_mass)

		if mean_bool:
			col_err_top_list.append(0.434*col_err_top/med_cols)
			col_err_bot_list.append(0.434*col_err_bot/med_cols)
			mass_annuli_top_err_list.append(0.434*mass_annuli_top_err/mass_annuli)
			mass_annuli_bot_err_list.append(0.434*mass_annuli_bot_err/mass_annuli)
			for i in range(0,np.size(mass_annuli_bot_err)):
				cum_mass_top_err[i] = np.sqrt(np.nansum(np.power(0.434*mass_annuli_top_err[0:i+1]/mass_annuli[0:i+1],2.)))/np.sqrt(np.size(mass_annuli_top_err[0:i+1]))
				cum_mass_bot_err[i] = np.sqrt(np.nansum(np.power(0.434*mass_annuli_bot_err[0:i+1]/mass_annuli[0:i+1],2.)))/np.sqrt(np.size(mass_annuli_top_err[0:i+1]))

			cum_mass_top_err_list.append(cum_mass_top_err)
			cum_mass_bot_err_list.append(cum_mass_bot_err)

		else:
			col_err_top_list.append(col_err_top)
			col_err_bot_list.append(col_err_bot)
			mass_annuli_top_err_list.append(mass_annuli_top_err)
			mass_annuli_bot_err_list.append(mass_annuli_bot_err)
			cum_mass_top_err_list.append(cum_mass_top_err)
			cum_mass_bot_err_list.append(cum_mass_bot_err)


		# print 'errs'
		# print col_err_top_list
		# print ''
		# print mass_annuli_bot_err_list
		# print ''
		# print cum_mass_bot_err_list
		# print ''
	### Plots
	for i in range(0,np.size(upper_ssfr)):
		if mean_bool:
			plt.errorbar(plot_radii_list[i] +stagger[i], np.log10(mass_annuli_list[i]), yerr=[mass_annuli_bot_err_list[i], mass_annuli_top_err_list[i]], color = colors[i], fmt = '.', ecolor= colors[i], label = labels[i])
		else:
			plt.errorbar(plot_radii_list[i] +stagger[i], mass_annuli_list[i], yerr=[mass_annuli_list[i]-mass_annuli_bot_err_list[i], mass_annuli_top_err_list[i]-mass_annuli_list[i]], color = colors[i], fmt = '.', ecolor= colors[i], label = labels[i])
			plt.yscale('log')
		plt.hold(True)

	if virial_radii_bool:
		plt.xlabel('Radius (fraction of virial radius)')
		plt.ylabel('log10(Mass) (fraction of halo mass))')
	else:
		plt.xlabel('Radius (kpc)')
		plt.ylabel('log10(Mass/M_sol) ')
		if mean_bool:
			plt.errorbar(proch_radii, np.log10(adj_proch_ann_mass), yerr=proch_mass_err_logged, c='#00FF00', marker='*', ecolor='#00FF00')
			plt.errorbar(proch_radii, np.log10(proch_ann_mass), yerr=proch_mass_err_logged, c='g', marker='*', ecolor='g', label='Prochaska 2017')
		else:
			plt.errorbar(proch_radii, adj_proch_ann_mass, yerr=proch_mass_err, c='#00FF00', marker='*', ecolor='#00FF00')
			plt.errorbar(proch_radii, proch_ann_mass, yerr=proch_mass_err, c='g', marker='*', ecolor='g', label='Prochaska 2017')
	plt.hold(False)
	plt.legend()
	plt.title('Gas Mass in Annulur Bins')
	plt.savefig('adj_mass_annuli.pdf')
	plt.close()

	for i in range(0,np.size(upper_ssfr)):
		if mean_bool:
			plt.errorbar(plot_radii_list[i]+stagger[i], np.log10(med_cols_list[i]), yerr=[col_err_bot_list[i], col_err_top_list[i]], color = colors[i], fmt = '.', ecolor= colors[i], label = labels[i])
		else:
			plt.errorbar(plot_radii_list[i]+stagger[i], med_cols_list[i], yerr=[med_cols_list[i]-col_err_bot_list[i], col_err_top_list[i]-med_cols_list[i]], color = colors[i], fmt = '.', ecolor= colors[i], label = labels[i])
			plt.yscale('log')
		plt.hold(True)
	plt.hold(False)
	plt.legend()
	plt.title('Hydrogen Column Densities')
	if virial_radii_bool:
		plt.xlabel('Radius (fraction of virial radius)')
	else:
		plt.xlabel('Radius (kpc)')
	plt.ylabel('log10(N_H) cm^-2')
	plt.savefig('rad_col_mass_est.pdf')
	plt.close()

	print 'cumulative massees'
	print labels
	print np.log10(cum_mass_list[0][-1])
	# print np.log10(cum_mass_list[1][-1])
	# print np.log10(cum_mass_list[2][-1])
	print 'errs'
	if mean_bool:
		print np.log10(cum_mass_bot_err_list[0][-1])
		print np.log10(cum_mass_top_err_list[0][-1])
		# print cum_mass_bot_err_list[1][-1]
		# print cum_mass_top_err_list[1][-1]
		# print cum_mass_bot_err_list[2][-1]
		# print cum_mass_top_err_list[2][-1]
	else:
		print cum_mass_list[0][-1]-cum_mass_bot_err_list[0][-1]
		print cum_mass_top_err_list[0][-1]-cum_mass_list[0][-1]
		# print cum_mass_list[1][-1]-cum_mass_bot_err_list[1][-1]
		# print cum_mass_top_err_list[1][-1]-cum_mass_list[1][-1]
		# print cum_mass_list[2][-1]-cum_mass_bot_err_list[2][-1]
		# print cum_mass_top_err_list[2][-1]-cum_mass_list[2][-1]
	print ''

	for i in range(0,np.size(upper_ssfr)):
		if mean_bool:
			plt.plot(plot_radii_list[i]+stagger[i], np.log10(cum_mass_list[i]), color = colors[i], label = labels[i])
			plt.hold(True)
			plt.fill_between(plot_radii_list[i], np.log10(cum_mass_list[i])-cum_mass_bot_err_list[i], np.log10(cum_mass_list[i])+cum_mass_top_err_list[i], color='k', alpha=0.33)
		else:
			plt.errorbar(plot_radii_list[i]+stagger[i], cum_mass_list[i], yerr=[cum_mass_list[i]-cum_mass_bot_err_list[i], cum_mass_top_err_list[i]-cum_mass_list[i]], color = colors[i], fmt = '.', ecolor= colors[i], label = labels[i])
			plt.yscale('log')
		plt.hold(True)

	if virial_radii_bool:
		plt.xlabel('Radius (fraction of virial radius)')
		plt.ylabel('log10(Mass (fraction of halo mass))')
	else:
		plt.xlabel('Radius (kpc)')
		plt.ylabel(r'$log_{10}\left ( \frac{M_{gas}}{M_{\odot}} \right )$')
		if mean_bool:
			# plt.plot(proch_radii, np.log10(adj_proch_cum_mass), color='#00FF00')
			# plt.fill_between(proch_radii, np.log10(adj_proch_cum_mass)-proch_cum_mass_err_logged, np.log10(adj_proch_cum_mass)+proch_cum_mass_err_logged, color='#00FF00', alpha=0.33)
			# plt.plot(proch_radii, np.log10(proch_cum_mass), color='g', label='Prochaska 2017')
			# plt.fill_between(proch_radii, np.log10(proch_cum_mass)-proch_cum_mass_err_logged, np.log10(proch_cum_mass)+proch_cum_mass_err_logged, color='g', alpha=0.33)
			plt.scatter(160., np.log10(2.1e10), c='m', label='Werk 2014')
		else:
			plt.errorbar(proch_radii, adj_proch_cum_mass, yerr=proch_cum_mass_err, c='#00FF00', marker='*', fmt='-', ecolor='#00FF00')
			plt.errorbar(proch_radii, proch_cum_mass, yerr=proch_cum_mass_err, c='g', marker='*', fmt='-', ecolor='g', label='Prochaska 2017')
			plt.scatter(160., (2.1e10), c='m', label='Werk 2014')
	plt.hold(False)
	plt.legend(loc='lower right')
	plt.title('Cumulative Gas Mass')
	plt.savefig('cum_mass.pdf')
	plt.close()

	if pop_str == 'old':
		titles = ['Low Mass', 'Red', 'Blue']
		xmin = np.log10(np.min([np.min(gas_densities_list[0]),np.min(gas_densities_list[1]),np.min(gas_densities_list[2])]))
		xmax = np.log10(np.max([np.max(gas_densities_list[0]),np.max(gas_densities_list[1]),np.max(gas_densities_list[2])]))
		ymin = np.log10(np.min([np.min(temps_list[0]),np.min(temps_list[1]),np.min(temps_list[2])]))
		ymax = np.log10(np.max([np.max(temps_list[0]),np.max(temps_list[1]),np.max(temps_list[2])]))

		for i in range(0,np.size(upper_ssfr)):
			height, xedges, yedges = np.histogram2d(np.log10(np.array(gas_densities_list[i])), np.log10(np.array(temps_list[i])),range=[[xmin, xmax], [ymin,ymax]], bins = 50)
			plt.imshow(height.transpose(), origin='lower', aspect = 'auto', cmap = 'gray_r', extent = (xmin, xmax, ymin, ymax))
			cb = plt.colorbar()
			cb.set_label('Number of Points')
			# plt.yscale('log')
			# plt.xscale('log')
			plt.title('%s Number Density vs Temperature' % (titles[i]))
			plt.xlabel('log10(n_gas) (cm^-3)')
			plt.ylabel('log10(T) (K)')
			plt.savefig('%s_rho_T.pdf'% (titles[i]))
			plt.close()

	# plt.plot(np.log10(cols), np.log10(H_cols), 'b.')
	# plt.title('H1 columns vs H columns')
	# plt.savefig('columns_comparison.pdf')
	# plt.close()

	plt.scatter(np.log10(cols), np.log10(H_cols), c=(gas_densities), s=6, cmap='gray', edgecolor = 'black', linewidth = 0.3, norm=LogNorm())
	# plt.clim(10.**10.7, 10**13.6)
	cb = plt.colorbar()
	cb.set_label(r'$log_{10}(%s number density)' % (ion))
	plt.title('N_H1 vs N_H columns in EAGLE')
	plt.xlabel('log10(N_H1) cm^-2')
	plt.ylabel('log10(N_H) cm^-2')
	plt.savefig('columns_comparison.pdf')
	plt.close()

	# num_inf = np.size(delta[delta>1.e30])
	# plt.plot(cols, np.log10(delta),'b.')
	# plt.title('log10(delta) infs=%d, tot=%d' % (num_inf, np.size(delta)))
	# plt.savefig('delta.pdf')

	# bins = np.linspace(np.min(np.power(H_cols,10)), np.max(np.power(H_cols,10)),6)
	# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(6, np.power(H_cols,10), np.power(cols,10))	

	# plt.plot(np.power(H_cols, 10)/np.power(cols,10), 'b.')
	# plt.hold(True)
	# plt.plot(plot_x, plot_median, 'k')
	# plt.plot(plot_x,plot_68, 'g')
	# plt.plot(plot_x,plot_32, 'g')
	# plt.plot(plot_x,plot_90, 'r')
	# plt.plot(plot_x,plot_10, 'r')
	# plt.hold(False)
	# plt.title('Ratio of Total to Neutral Column')
	# plt.savefig('col_ratios.pdf')
	# plt.close()

	# bins = np.linspace(np.min(H_cols), np.max(H_cols), 20)
	# plt.hist(H_cols,bins)
	# plt.title('Histogram of Neutral Hydrogen Columns')
	# plt.savefig('H_col_hist.pdf')

	# bins = np.linspace(np.min(cols), np.max(cols), 20)
	# plt.hist(cols,bins)
	# plt.title('Histogram of Neutral Hydrogen Columns')
	# plt.savefig('col_hist.pdf')

	# height, xedges, yedges = np.histogram2d(radii, H_cols, bins=int((np.max(radii)-np.min(radii))/10.))
	# plt.imshow(height.transpose(), extent = (np.min(radii), np.max(radii), np.min(H_cols), np.max(H_cols)), origin = 'lower', aspect = 'auto')
	# plt.savefig('H_col_vs_rad.pdf')

	# height, xedges, yedges = np.histogram2d(cols, H_cols, bins=25)
	# plt.imshow(height.transpose(), extent = (np.min(cols), np.max(cols), np.min(H_cols), np.max(H_cols)), origin = 'lower', aspect = 'auto')
	# plt.savefig('col_vs_H_col.pdf')

	# height, xedges, yedges = np.histogram2d(virial_radii, cols, bins = 50)
	# plt.imshow(height.transpose(), extent = (np.min(virial_radii), np.max(virial_radii), np.min(cols), np.max(cols)), origin='lower', aspect = 'auto')

def kinematic_plots(num_minima, centroid_vel, depth, FWHM, radii, temps, escape_vels, virial_radii_for_kin, halo_masses_for_kin, stellar_masses_for_kin, ssfr_for_kin, bins_for_median):

	### Making radii array that accounts for spectra with multiple minima
	plotting_radii = np.zeros(np.size(centroid_vel))
	index = 0
	for i in range(0,np.size(num_minima)):
		for j in range(0,num_minima[i]):
			plotting_radii[index] = radii[i]
			index += 1

	centroids_in_esc = centroid_vel/escape_vels
	### if using virial velocity!!!!
	centroids_in_esc *= np.sqrt(2.0)

	virial_radii_for_kin = plotting_radii/virial_radii_for_kin # the passed array is really the value of the virial radius for each galaxy
	temps = np.log10(temps) # temperature in log space

	### Get rid of points where line was too close to edge so FWHM couldn't be obtained
	centroid_vel = centroid_vel[FWHM != 1.e6]
	centroids_in_esc = centroids_in_esc[FWHM != 1.e6]
	depth = depth[FWHM != 1.e6]
	plotting_radii = plotting_radii[FWHM != 1.e6]
	temps = temps[FWHM != 1.e6]
	virial_radii_for_kin = virial_radii_for_kin[FWHM != 1.e6]
	halo_masses_for_kin = halo_masses_for_kin[FWHM != 1.e6]
	stellar_masses_for_kin = stellar_masses_for_kin[FWHM != 1.e6]
	ssfr_for_kin = ssfr_for_kin[FWHM != 1.e6]
	FWHM = FWHM[FWHM != 1.e6]

	high_m_esc_gas = centroids_in_esc[((centroids_in_esc > 1.0) & (np.log10(halo_masses_for_kin) >= 12.8))]
	mid_m_esc_gas = centroids_in_esc[((centroids_in_esc > 1.0) & (np.log10(halo_masses_for_kin) >= 11.7) & (np.log10(halo_masses_for_kin) < 12.8))]
	low_m_esc_gas = centroids_in_esc[((centroids_in_esc > 1.0) & (np.log10(halo_masses_for_kin) < 11.7))]

	high_m_frac = float(np.size(high_m_esc_gas))/np.size(centroids_in_esc)
	mid_m_frac = float(np.size(mid_m_esc_gas))/np.size(centroids_in_esc)
	low_m_frac = float(np.size(low_m_esc_gas))/np.size(centroids_in_esc)

	### For stellar
	upper_mass = np.array([9.7, 15., 15.])
	lower_mass = np.array([5., 9.7, 9.7])

	upper_ssfr = np.array([-5., -5., -11.])
	lower_ssfr = np.array([-15., -11., -15.])

	colors = ['k', 'b', 'r']
	labels = ['low mass', 'blue', 'red']

	### Histograms
	bins = [-0.5,0.5,1.5,2.5,3.5,4.5,5.5]
	plt.hist(num_minima, bins)
	plt.hist(num_minima)
	plt.title('number of minima in lines')
	plt.xlabel('Number of Minima')
	plt.savefig('num_minima_hist.pdf')
	plt.close()

	bins = np.linspace(np.min(centroid_vel), np.max(centroid_vel), 20.)
	centroids = [[],[],[]]
	for i in range(0,np.size(upper_mass)):
		centroids[i] = centroid_vel[((np.log10(stellar_masses_for_kin) < upper_mass[i]) & (np.log10(stellar_masses_for_kin) > lower_mass[i]) & (ssfr_for_kin < upper_ssfr[i]) & (ssfr_for_kin > lower_ssfr[i]))]

	plt.hist(centroids, bins, label=labels, color=colors)
	plt.legend()
	plt.title('Velocity Histogram')
	plt.xlabel('Velocity (km/s)')
	plt.savefig('vel_hist.pdf')
	plt.close()

	bins = np.linspace(np.min(centroids_in_esc), np.max(centroids_in_esc), 20.)
	centroids = [[],[],[]]
	for i in range(0,np.size(upper_mass)):
		centroids[i] = centroids_in_esc[((np.log10(stellar_masses_for_kin) < upper_mass[i]) & (np.log10(stellar_masses_for_kin) > lower_mass[i]) & (ssfr_for_kin < upper_ssfr[i]) & (ssfr_for_kin > lower_ssfr[i]))]

	plt.hist((centroids), bins, label=labels, color=colors)
	plt.legend()
	plt.title('Veolocities in Virial Velocity Histogram')
	plt.xlabel('Velocity (v/v_virial)')
	plt.savefig('esc_vel_hist.pdf')
	plt.close()	

	# bins = np.linspace(0, np.max(centroid_vel), 20.)
	# plt.hist(np.abs(centroid_vel), bins)
	# plt.title('speed histogram')
	# plt.savefig('speed_hist.pdf')
	# plt.close()


	# plt.scatter(virial_radii, equ_widths, c=ssfr, s=20., cmap='RdBu', edgecolor = 'black', linewidth = 0.5)
	# cb = plt.colorbar()
	# cb.set_label(r'$log_{10}(sSFR)')

	### Plots
	# bins = np.linspace(np.min(centroid_vel), np.max(centroid_vel),bins_for_median)
	# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, centroid_vel, depth)	

	# plt.plot(centroid_vel, depth, 'b.')
	# plt.hold(True)
	# plt.plot(plot_x, plot_median, 'k')
	# plt.plot(plot_x,plot_68, 'k--')
	# plt.plot(plot_x,plot_32, 'k--')
	# plt.plot(plot_x,plot_90, 'k:')
	# plt.plot(plot_x,plot_10, 'k:')
	# plt.hold(False)
	# plt.title('Vel vs depth')
	# plt.savefig('vel_depth.pdf')
	# plt.close()

	# bins = np.linspace(np.min(centroids_in_esc), np.max(centroids_in_esc),bins_for_median)
	# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, centroids_in_esc, depth)	

	# plt.plot(centroids_in_esc, depth, 'b.')
	# plt.hold(True)
	# plt.plot(plot_x, plot_median, 'k')
	# plt.plot(plot_x,plot_68, 'k--')
	# plt.plot(plot_x,plot_32, 'k--')
	# plt.plot(plot_x,plot_90, 'k:')
	# plt.plot(plot_x,plot_10, 'k:')
	# plt.hold(False)
	# plt.title('Vel in esc vs depth')
	# plt.savefig('esc_vel_depth.pdf')
	# plt.close()

	# bins = np.linspace(np.min(centroid_vel), np.max(centroid_vel),bins_for_median)
	# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, centroid_vel, FWHM)

	# plt.plot(centroid_vel, FWHM, 'b.')
	# plt.hold(True)
	# plt.plot(plot_x, plot_median, 'k')
	# plt.plot(plot_x,plot_68, 'k--')
	# plt.plot(plot_x,plot_32, 'k--')
	# plt.plot(plot_x,plot_90, 'k:')
	# plt.plot(plot_x,plot_10, 'k:')
	# plt.hold(False)
	# plt.title('vel vs FWHM')
	# plt.savefig('vel_FWHM.pdf')
	# plt.close()

	# bins = np.linspace(np.min(centroids_in_esc), np.max(centroids_in_esc),bins_for_median)
	# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, centroids_in_esc, FWHM)

	# plt.plot(centroids_in_esc, FWHM, 'b.')
	# plt.hold(True)
	# plt.plot(plot_x, plot_median, 'k')
	# plt.plot(plot_x,plot_68, 'k--')
	# plt.plot(plot_x,plot_32, 'k--')
	# plt.plot(plot_x,plot_90, 'k:')
	# plt.plot(plot_x,plot_10, 'k:')
	# plt.hold(False)
	# plt.title('vel in exc vs FWHM')
	# plt.savefig('esc_vel_FWHM.pdf')
	# plt.close()

	# bins = np.linspace(np.min(depth), np.max(depth),bins_for_median)
	# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, depth, FWHM)

	# plt.plot(depth, FWHM, 'b.')
	# plt.hold(True)
	# plt.plot(plot_x, plot_median, 'k')
	# plt.plot(plot_x,plot_84, 'k--')
	# plt.plot(plot_x,plot_16, 'k--')
	# plt.plot(plot_x,plot_95, 'k:')
	# plt.plot(plot_x,plot_5, 'k:')
	# plt.hold(False)
	# plt.title('Depth vs FWHM')
	# plt.savefig('depth_FWHM.pdf')
	# plt.close()

	# bins = np.linspace(np.min(plotting_radii), np.max(plotting_radii),bins_for_median)
	# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, plotting_radii, depth)

	# plt.plot(plotting_radii, depth, 'b.')
	# plt.hold(True)
	# plt.plot(plot_x, plot_median, 'k')
	# plt.plot(plot_x,plot_84, 'k--')
	# plt.plot(plot_x,plot_16, 'k--')
	# plt.plot(plot_x,plot_95, 'k:')
	# plt.plot(plot_x,plot_5, 'k:')
	# plt.hold(False)
	# plt.title('radius vs Depth')
	# plt.savefig('radius_depth.pdf')
	# plt.close()

	# bins = np.linspace(np.min(virial_radii_for_kin), np.max(virial_radii_for_kin),bins_for_median)
	# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, virial_radii_for_kin, depth)

	# plt.plot(virial_radii_for_kin, depth, 'b.')
	# plt.hold(True)
	# plt.plot(plot_x, plot_median, 'k')
	# plt.plot(plot_x,plot_84, 'k--')
	# plt.plot(plot_x,plot_16, 'k--')
	# plt.plot(plot_x,plot_95, 'k:')
	# plt.plot(plot_x,plot_5, 'k:')
	# plt.hold(False)
	# plt.title('vir radius vs Depth')
	# plt.savefig('vir_radius_depth.pdf')
	# plt.close()

	bins = np.linspace(np.min(plotting_radii), np.max(plotting_radii),bins_for_median)
	plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, plotting_radii, FWHM)

	plt.scatter(plotting_radii, FWHM, c=halo_masses_for_kin, s=3, cmap='RdBu_r', edgecolor = 'black', linewidth = 0.3, norm=LogNorm())
	plt.clim(10.**10.7, 10**13.6)
	cb = plt.colorbar()
	cb.set_label(r'$log_{10}(M_{halo})$')
	plt.hold(True)
	plt.plot(plot_x, plot_median, c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_84, '--', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_16, '--', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_95, ':', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_5, ':', c='k', linewidth = 1.5)
	plt.hold(False)
	plt.title('Radius vs FWHM')
	plt.xlabel('Radius (kpc)')
	plt.ylabel('FWHM of Line (km/s)')
	plt.savefig('radius_FWHM.pdf')
	plt.close()

	bins = np.linspace(np.min(virial_radii_for_kin), np.max(virial_radii_for_kin),bins_for_median)
	plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, virial_radii_for_kin, FWHM)

	plt.scatter(virial_radii_for_kin, FWHM, c=halo_masses_for_kin, s=3, cmap='RdBu_r', edgecolor = 'black', linewidth = 0.3, norm=LogNorm())
	plt.clim(10.**10.7, 10**13.6)
	cb = plt.colorbar()
	cb.set_label(r'$log_{10}(M_{halo})$')
	plt.hold(True)
	plt.plot(plot_x, plot_median, c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_84, '--', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_16, '--', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_95, ':', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_5, ':', c='k', linewidth = 1.5)
	plt.hold(False)
	plt.title('Radius vs FWHM')
	plt.xlabel('Radius (in virial radii)')
	plt.ylabel('FWHM of Line (km/s)')
	plt.savefig('vir_radius_FWHM.pdf')
	plt.close()

	# bins = np.linspace(np.min(plotting_radii), np.max(plotting_radii),bins_for_median)
	# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, plotting_radii, centroid_vel)

	# plt.plot(plotting_radii, centroid_vel, 'b.')
	# plt.hold(True)
	# plt.plot(plot_x, plot_median, 'k')
	# plt.plot(plot_x,plot_84, 'k--')
	# plt.plot(plot_x,plot_16, 'k--')
	# plt.plot(plot_x,plot_95, 'k:')
	# plt.plot(plot_x,plot_5, 'k:')
	# plt.hold(False)
	# plt.title('rad vs vel')
	# plt.savefig('rad_vel.pdf')
	# plt.close()

	# bins = np.linspace(np.min(virial_radii_for_kin), np.max(virial_radii_for_kin),bins_for_median)
	# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, virial_radii_for_kin, centroids_in_esc)

	# plt.scatter(virial_radii_for_kin, centroids_in_esc, c=halo_masses_for_kin, s=3, cmap='RdBu_r', edgecolor = 'black', linewidth = 0.3, norm=LogNorm())
	# plt.clim(10.**10.7, 10**13.6)
	# cb = plt.colorbar()
	# cb.set_label(r'$log_{10}(M_{halo})$')
	# plt.hold(True)
	# plt.plot(plot_x, plot_median, c='k', linewidth = 1.5)
	# plt.plot(plot_x,plot_84, '--', c='k', linewidth = 1.5)
	# plt.plot(plot_x,plot_16, '--', c='k', linewidth = 1.5)
	# plt.plot(plot_x,plot_95, ':', c='k', linewidth = 1.5)
	# plt.plot(plot_x,plot_5, ':', c='k', linewidth = 1.5)
	# plt.hold(False)
	# plt.title('vir rad vs esc vel')
	# plt.savefig('vir_rad_esc_vel.pdf')
	# plt.close()

	bins = np.linspace(np.min(centroid_vel), np.max(centroid_vel),bins_for_median)
	plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, centroid_vel, temps)

	plt.scatter(centroid_vel, temps, c=halo_masses_for_kin, s=3, cmap='RdBu_r', edgecolor = 'black', linewidth = 0.3, norm=LogNorm())
	plt.clim(10.**10.7, 10**13.6)
	cb = plt.colorbar()
	cb.set_label(r'$log_{10}(M_{halo})$')
	plt.hold(True)
	plt.plot(plot_x, plot_median, c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_84, '--', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_16, '--', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_95, ':', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_5, ':', c='k', linewidth = 1.5)
	plt.hold(False)
	plt.title('Velocity vs Temperature')
	plt.xlabel('Velocity (km/s)')
	plt.ylabel('log10(Temperature) K')
	plt.savefig('vel_temp.pdf')
	plt.close()

	bins = np.linspace(np.min(np.abs(centroids_in_esc)), np.max(np.abs(centroids_in_esc)),bins_for_median)
	plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, np.abs(centroids_in_esc), temps)

	plt.scatter(np.abs(centroids_in_esc), temps, c=halo_masses_for_kin, s=3, cmap='RdBu_r', edgecolor = 'black', linewidth = 0.3, norm=LogNorm())
	plt.clim(10.**10.7, 10**13.6)
	cb = plt.colorbar()
	cb.set_label(r'$log_{10}(M_{halo})$')
	plt.hold(True)
	plt.plot(plot_x, plot_median, c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_84, '--', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_16, '--', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_95, ':', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_5, ':', c='k', linewidth = 1.5)
	plt.hold(False)
	plt.title('Velocity vs Temperature')
	plt.xlabel('Velocity (in fraction of virial velocity)')
	plt.ylabel('log10(Temperature) K')
	plt.savefig('esc_vel_temp.pdf')
	plt.close()


	# bins = np.linspace(np.min(depth), np.max(depth),bins_for_median)
	# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, depth, temps)

	# plt.plot(depth, temps, 'b.')
	# plt.hold(True)
	# plt.plot(plot_x, plot_median, 'k')
	# plt.plot(plot_x,plot_84, 'k--')
	# plt.plot(plot_x,plot_16, 'k--')
	# plt.plot(plot_x,plot_95, 'k:')
	# plt.plot(plot_x,plot_5, 'k:')
	# plt.hold(False)
	# plt.title('depth vs temp')
	# plt.savefig('depth_temp.pdf')
	# plt.close()

	# bins = np.linspace(np.min(plotting_radii), np.max(plotting_radii),bins_for_median)
	# plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, plotting_radii, temps)

	# plt.scatter(plotting_radii, temps, c=halo_masses_for_kin, s=3, cmap='RdBu_r', edgecolor = 'black', linewidth = 0.3, norm=LogNorm())
	# plt.clim(10.**10.7, 10**13.6)
	# cb = plt.colorbar()
	# cb.set_label(r'$log_{10}(M_{halo})$')
	# plt.hold(True)
	# plt.plot(plot_x, plot_median, c='k', linewidth = 1.5)
	# plt.plot(plot_x,plot_84, '--', c='k', linewidth = 1.5)
	# plt.plot(plot_x,plot_16, '--', c='k', linewidth = 1.5)
	# plt.plot(plot_x,plot_95, ':', c='k', linewidth = 1.5)
	# plt.plot(plot_x,plot_5, ':', c='k', linewidth = 1.5)
	# plt.hold(False)
	# plt.title('radius vs temp')
	# plt.savefig('radius_temp.pdf')
	# plt.close()

	bins = np.linspace(np.min(virial_radii_for_kin), np.max(virial_radii_for_kin),bins_for_median)
	plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, virial_radii_for_kin, temps)

	plt.scatter(virial_radii_for_kin, temps, c=halo_masses_for_kin, s=3, cmap='RdBu_r', edgecolor = 'black', linewidth = 0.3, norm=LogNorm())
	plt.clim(10.**10.7, 10**13.6)
	cb = plt.colorbar()
	cb.set_label('log10(halo mass)')
	plt.hold(True)
	plt.plot(plot_x, plot_median, c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_84, '--', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_16, '--', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_95, ':', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_5, ':', c='k', linewidth = 1.5)
	plt.hold(False)
	plt.title('Radius vs Temperature')
	plt.xlabel('Radius (in virial radii)')
	plt.ylabel('log10(Temperature) K')
	plt.savefig('vir_radius_temp.pdf')
	plt.close()

	bins = np.linspace(np.min(np.log10(halo_masses_for_kin)), np.max(np.log10(halo_masses_for_kin)),bins_for_median)
	plot_x, plot_median, plot_84, plot_16, plot_95, plot_5 = percentile_array(bins, np.log10(halo_masses_for_kin), temps)

	tens = np.zeros(np.size(ssfr_for_kin)) + 10.
	plt.scatter(np.log10(halo_masses_for_kin), temps, c=np.power(tens, ssfr_for_kin), s=3, cmap='RdBu', edgecolor = 'black', linewidth = 0.3, norm=LogNorm())
	plt.clim(10**(-12.7), 10.**(-9.0))
	cb = plt.colorbar()
	cb.set_label(r'$log_{10}(sSFR)$')
	plt.hold(True)
	plt.plot(plot_x, plot_median, c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_84, '--', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_16, '--', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_95, ':', c='k', linewidth = 1.5)
	plt.plot(plot_x,plot_5, ':', c='k', linewidth = 1.5)
	plt.hold(False)
	plt.title('Halo Mass vs Temperature')
	plt.xlabel('log10(Halo Mass)')
	plt.ylabel('log10(Temperature) K')
	plt.savefig('mass_temp.pdf')
	plt.close()

def percentile_array(bins, x_arr, y_arr):
	plot_median = np.zeros(np.size(bins)-1)
	plot_84 = np.zeros(np.size(bins)-1)
	plot_16 = np.zeros(np.size(bins)-1)
	plot_95 = np.zeros(np.size(bins)-1)
	plot_5 = np.zeros(np.size(bins)-1)
	plot_x = np.zeros(np.size(bins)-1)

	for i in range(0,np.size(bins)-1):
		temp_y_arr = y_arr[(x_arr > bins[i]) & (x_arr < bins[i+1])]

		plot_median[i] = np.percentile(temp_y_arr, 50)
		plot_84[i] = np.percentile(temp_y_arr, 84)
		plot_16[i] = np.percentile(temp_y_arr, 16)
		plot_95[i] = np.percentile(temp_y_arr, 95)
		plot_5[i] = np.percentile(temp_y_arr, 5)
		plot_x[i] = (bins[i]+bins[i+1])/2.0

	return plot_x, plot_median, plot_84, plot_16, plot_95, plot_5

# makes fits file for flux data with 3 columns, velocity, flux (normed), and err (will be zeros if np.size(names)=2)
def make_fits_file_for_flux(arrays, file_name):
	if np.shape(arrays)[0] == 2:
		arrays.append(np.zeros(np.size(arrays[0])))

	binary_table = pyfits.new_table(pyfits.ColDefs([pyfits.Column(name='velocities', format='E', array=arrays[0]),pyfits.Column(name='flux', format='E', array=arrays[1]),pyfits.Column(name='flux_err', format='E', array=arrays[2])]))

	primary_header = pyfits.Header()
	primary_header['Author'] = 'Ryan Horton'
	primary_header['Comments'] = 'A fits file for a single line spectra through an EAGLE simulation using specwizard'
	primary_header['col_1'] = 'name= velocities, units= km/s, description = line is plotted in velocity at each pixel in the line (relative to central galaxy)'
	primary_header['col_2'] = 'name = flux, units = normalized flux (None), description= flux of the line'
	primary_header['col_3'] = 'name = flux_err, units = normalized flux (None), description= error in each pixel. All zeros means spectra taken directly from specwizard and no error bars have been estimated'
	primary_hdu = pyfits.PrimaryHDU(header=primary_header)

	final_hdu_list = pyfits.HDUList([primary_hdu, binary_table])
	final_hdu_list.writeto(file_name)

	# ### Example of how to open and read some the data
	# hdulist = pyfits.open(file_name)

	# print hdulist.info()
	# print ''
	# print hdulist[0].header
	# print ''
	# print hdulist[1].header
	# print ''

	# hdu_data = hdulist[1].data
	# print hdu_data['velocities']
	# print ''
	# print hdu_data['flux']
	# print ''
	# print hdu_data['flux_err']
	# hdulist.close()

	# raise ValueError('only doing it once right now')


def handle_single_realization_statistics(spec_output_directory, cos_smass_data, cos_ssfr_data, cos_id_arr, cos_h1_equ_widths, cos_h1_W_flags, cos_h1_equ_widths_radii, cos_h1_cols, cos_h1_cols_flags, cos_h1_cols_radii, equ_widths_bool):

	upper_mass = [9.7, 15., 15.]
	lower_mass = [5., 9.7, 9.7]
	upper_ssfr = [-5., -5., -11.]
	lower_ssfr = [-15., -11., -15.]
	groups = ['Low Mass', 'Active', 'Passive']
	reject_vals = [[] for i in range(np.size(upper_mass))]
	colors = ['g', 'b', 'r']
	bins = np.arange(0,110,10)

	for j in range(0,np.size(upper_mass)):
		### work with only cos data we care about for this subset (low mass, active, passive)
		curr_cos_id_arr = cos_id_arr[((cos_smass_data<upper_mass[j]) & (cos_smass_data>lower_mass[j]) & (cos_ssfr_data<upper_ssfr[j]) & (cos_ssfr_data>lower_ssfr[j]))]
		curr_cos_h1_equ_widths = cos_h1_equ_widths[((cos_smass_data<upper_mass[j]) & (cos_smass_data>lower_mass[j]) & (cos_ssfr_data<upper_ssfr[j]) & (cos_ssfr_data>lower_ssfr[j]))]
		curr_cos_h1_W_flags = cos_h1_W_flags[((cos_smass_data<upper_mass[j]) & (cos_smass_data>lower_mass[j]) & (cos_ssfr_data<upper_ssfr[j]) & (cos_ssfr_data>lower_ssfr[j]))]
		curr_cos_h1_equ_widths_radii = cos_h1_equ_widths_radii[((cos_smass_data<upper_mass[j]) & (cos_smass_data>lower_mass[j]) & (cos_ssfr_data<upper_ssfr[j]) & (cos_ssfr_data>lower_ssfr[j]))]
		curr_cos_h1_cols = cos_h1_cols[((cos_smass_data<upper_mass[j]) & (cos_smass_data>lower_mass[j]) & (cos_ssfr_data<upper_ssfr[j]) & (cos_ssfr_data>lower_ssfr[j]))]
		curr_cos_h1_cols_flags = cos_h1_cols_flags[((cos_smass_data<upper_mass[j]) & (cos_smass_data>lower_mass[j]) & (cos_ssfr_data<upper_ssfr[j]) & (cos_ssfr_data>lower_ssfr[j]))]
		curr_cos_h1_cols_radii = cos_h1_cols_radii[((cos_smass_data<upper_mass[j]) & (cos_smass_data>lower_mass[j]) & (cos_ssfr_data<upper_ssfr[j]) & (cos_ssfr_data>lower_ssfr[j]))]

		print groups[j]
		print 'num of cos galaxies in this group'
		print np.size(curr_cos_id_arr)
		print ''

		file_ids, cos_ids, spec_nums, directory_arr, num_runs = decompose_multi_rel_into_single(spec_output_directory, curr_cos_id_arr)
		print 'number of runs is %d' % (num_runs)
		plot_num = 0
		for i in range(0,num_runs):
			curr_file_ids = file_ids[i::num_runs]
			curr_cos_ids = cos_ids[i::num_runs]
			curr_spec_nums = spec_nums[i::num_runs]
			curr_directory_arr = directory_arr[i::num_runs]

			reject_val = get_KS_for_single(curr_directory_arr, curr_file_ids, curr_cos_ids, curr_spec_nums, curr_cos_id_arr, curr_cos_h1_equ_widths, \
				         curr_cos_h1_W_flags, curr_cos_h1_equ_widths_radii, curr_cos_h1_cols, curr_cos_h1_cols_flags, curr_cos_h1_cols_radii, equ_widths_bool,\
				         plot_num, groups[j])
			plot_num += 1
			if reject_val == 'Not Rejected':
				reject_vals[j].append(0.0)
			else:
				reject_vals[j].append(reject_val)
	

	reject_vals = np.transpose(np.array(reject_vals))
	print np.shape(reject_vals)
	print ''

	hist_fig = plt.figure()
	hist_ax = hist_fig.add_subplot(111)
	plt.hold(True)
	hist_ax.hist(reject_vals, bins, color=colors, label=groups)
	plt.hold(False)
	hist_ax.set_title('Histograms of Rejection Certainty', fontsize=16)
	hist_ax.set_xlabel(r'Rejection Certainty (10$\%$ bins)', fontsize=14)
	hist_ax.set_ylabel('Number', fontsize=14)
	hist_ax.set_xticks(bins)
	plt.hold(False)
	hist_ax.legend(fontsize=14)
	hist_ax.grid()
	hist_ax.set_axisbelow(True)
	hist_fig.savefig('better_hists.pdf')
	plt.close(hist_fig)


def decompose_multi_rel_into_single(spec_output_directory, curr_cos_id_arr):

	file_ids = []
	cos_ids = []
	spec_nums = []
	directory_arr = []

	for directory in spec_output_directory:
		los_files = glob.glob(directory + '/los_*')
		spec_files = glob.glob(directory + '/spec.snap*')

		for file in los_files:
			run_num = file[-6:-4]
			if run_num[0] == '_':
				run_num = run_num[1::]

			first_line = True
			with open(file, 'r') as curr_file:
				lines_read = 0
				for line in curr_file:
					if first_line:
						first_line = False
						continue
					else:
						curr_cos_id = line.rsplit(' ',1)[-1]
						
						if int(curr_cos_id) in curr_cos_id_arr:
							file_ids.append(int(run_num))
							cos_ids.append(int(curr_cos_id))
							spec_nums.append(lines_read)
							directory_arr.append(directory)
						lines_read += 1
		temp_cos_ids = np.array(cos_ids)


	cos_ids = np.array(cos_ids)
	file_ids = np.array(file_ids)
	spec_nums = np.array(spec_nums)
	directory_arr = np.array(directory_arr)
	sorted_indices = np.argsort(cos_ids)

	file_ids = file_ids[sorted_indices]
	cos_ids = cos_ids[sorted_indices]
	spec_nums = spec_nums[sorted_indices]
	directory_arr = directory_arr[sorted_indices]
		
	for i in range(0,np.size(cos_ids)):
		if cos_ids[i] != cos_ids[0]:
			num_runs = i
			break
	print 'breakdown'
	print num_runs
	print np.size(cos_ids)
	print np.size(cos_ids)%num_runs
	print ''

	return file_ids, cos_ids, spec_nums, directory_arr, num_runs


def get_KS_for_single(curr_directory_arr, curr_file_ids, curr_cos_ids, curr_spec_nums, cos_id_arr, cos_h1_equ_widths, cos_Ws_flags, \
	cos_h1_equ_widths_radii, cos_h1_cols, cos_h1_cols_flags, cos_h1_cols_radii, equ_widths_bool, plot_num, plot_group):
	### Just set stuff in here
	ion_name = 'h1'
	lambda_line = 1215.67

	### sim line initializations
	sim_Ws = np.array([None]*np.size(curr_file_ids))
	sim_cols = np.array([None]*np.size(curr_file_ids))
	sim_radii = np.array([None]*np.size(curr_file_ids))

	### Actual stuff
	mask = np.in1d(cos_id_arr, curr_cos_ids)
	unmasked_indices = np.argwhere(mask)[:,0]

	cos_Ws = cos_h1_equ_widths[unmasked_indices]
	cos_Ws_flags = cos_Ws_flags[unmasked_indices]
	cos_cols = cos_h1_cols[unmasked_indices]
	cos_cols_flags = cos_h1_cols_flags[unmasked_indices]
	cos_radii = cos_h1_equ_widths_radii[unmasked_indices]

	i = 0
	for file_id in curr_file_ids:

		spec_file = curr_directory_arr[i] + '/spec.snap_%s.hdf5' % (file_id)
		gal_output_file = curr_directory_arr[i] + '/gal_output_%s.hdf5' % (file_id)
		los_file = curr_directory_arr[i] + '/los_%s.txt' % (file_id)

		with h5py.File(spec_file, 'r') as hf:
			spec_hubble_velocity = hf.get('VHubble_KMpS')
			delta_v = np.abs(spec_hubble_velocity[1]-spec_hubble_velocity[0])
			spec = hf.get('Spectrum%s' % (str(curr_spec_nums[i])))
			ion = spec.get(ion_name)
			sim_cols[i] =  np.array(ion.get('LogTotalIonColumnDensity'))
			flux = np.array(ion.get('Flux'))
			sim_Ws[i] = np.sum(1.-flux)*(delta_v/c_kms)*lambda_line

		with h5py.File(gal_output_file, 'r') as hf:
			GalaxyProperties = hf.get('GalaxyProperties')
			box_size = np.array(GalaxyProperties.get('box_size'))
			gal_coords = np.array(GalaxyProperties.get('gal_coords'))[0]
			gal = np.array([gal_coords[0], gal_coords[1], gal_coords[2]])/box_size

		with open(los_file, 'r') as file:
			lines = file.readlines()
			line = lines[curr_spec_nums[i]+1]
			line = line.rsplit(' ')
			line = np.array([float(line_val) for line_val in line])

		sim_radii[i] = get_correct_radius_of_line(line,gal)*box_size

		i += 1

	if equ_widths_bool:
		obs_radius_max, obs_value_max, sim_radius_max, sim_value_max, obs_quad, sim_quad, delta, reject_val = \
		KS_2D(cos_radii, cos_Ws, sim_radii, sim_Ws)
		if reject_val == 'Not Rejected':
			reject_val = 0.0

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(sim_radii, sim_Ws, s=50., marker='.', c='k')
		plt.hold(True)
		ax.scatter(sim_radius_max, sim_value_max, s=100., marker='s', c='k', label='EAGLE: Quad=%s' % (str(sim_quad)))

		norm_radii = cos_radii[cos_Ws_flags == 1]
		norm_Ws = cos_Ws[cos_Ws_flags == 1]
		upper_radii = cos_radii[cos_Ws_flags == 3]
		upper_Ws = cos_Ws[cos_Ws_flags == 3]
		lower_radii = cos_radii[cos_Ws_flags == 2]
		lower_Ws = cos_Ws[cos_Ws_flags == 2]

		ax.scatter(norm_radii, norm_Ws, s=50., marker='*', c='#00FF00')
		ax.scatter(upper_radii, upper_Ws, s=50., marker='v', c='#00FF00')
		ax.scatter(lower_radii, lower_Ws, s=50., marker='^', c='#00FF00')
		ax.scatter(obs_radius_max, obs_value_max, s=100., marker='s', c='#00FF00', label='COS: Quad=%s' % (str(obs_quad)))
		plt.hold(False)
		ax.grid()
		ax.set_axisbelow(True)
		ax.legend(fontsize=14)
		ax.set_title('%s Example: Reject at %s%% certainty' % (plot_group ,str(reject_val)), fontsize=16)
		ax.set_xlabel('Impact Parameter (kpc)', fontsize=14)
		ax.set_ylabel(r'Equivalent Width ($\AA{}$)', fontsize=14)
		fig.savefig('%s_ws_%s.pdf' % (plot_group, str(plot_num)))
		plt.close(fig)

	else:
		obs_radius_max, obs_value_max, sim_radius_max, sim_value_max, obs_quad, sim_quad, delta, reject_val = \
		KS_2D(cos_radii, cos_cols, sim_radii, sim_cols)
		if reject_val == 'Not Rejected':
			reject_val = 0.0

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(sim_radii, sim_cols, s=50., marker='.', c='k')
		plt.hold(True)
		ax.scatter(sim_radius_max, sim_value_max, s=100., marker='s', c='k', label='EAGLE: Quad=%s' % (str(sim_quad)))

		norm_radii = cos_radii[cos_cols_flags == 1]
		norm_cols = cos_cols[cos_cols_flags == 1]
		upper_radii = cos_radii[cos_cols_flags == 3]
		upper_cols = cos_cols[cos_cols_flags == 3]
		lower_radii = cos_radii[cos_cols_flags == 2]
		lower_cols = cos_cols[cos_cols_flags == 2]

		ax.scatter(norm_radii, norm_cols, s=50., marker='*', c='#00FF00')
		ax.scatter(upper_radii, upper_cols, s=50., marker='v', c='#00FF00')
		ax.scatter(lower_radii, lower_cols, s=50., marker='^', c='#00FF00')
		ax.scatter(obs_radius_max, obs_value_max, s=100., marker='s', c='#00FF00', label='COS: Quad=%s' % (str(obs_quad)))
		plt.hold(False)
		ax.grid()
		ax.set_axisbelow(True)
		ax.legend(fontsize=14)
		ax.set_title('%s Example: Reject at %s%% certainty' % (plot_group ,str(reject_val)), fontsize=16)
		ax.set_xlabel('Impact Parameter (kpc)', fontsize=14)
		ax.set_ylabel(r'Column Density ($cm^{-2}$)', fontsize=14)
		fig.savefig('%s_cols_%s.pdf' % (plot_group, str(plot_num)))
		plt.close(fig)

	return reject_val









