### Particle tracking funcitons. 
### These functions use particle IDs output by specwizard (or some other source) to determine properties of specific subsets of particles in EAGLE.
### Ryan Horton 

### Tests: For single_line_test = True (directory = /gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/1_gal_test) we get
### Specwizard says NGas = 25,761,997
### My code says the number of ParticleIDs in PartType0 = 51,523,994

### Specwizard code that I added says 14,695 particles hit 
### When I look throug the snapshot files it only says that 356 of those match (and sometimes that number is much lower)

### I want to change how it looks for the snap files. Have the directory be more specific as in...
### snap_directroy/*snapshot_noneq*/*file_keyword* play around with the middle part (snapshot_noneq) being just read from the file keyword
### Check to make sure it doesn't break AGN.
### halos part_ids 10

### Imports
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
import matplotlib.patches as mpatches
import bisect
from matplotlib.colors import LogNorm
import gen_lsf
import math
from multiprocessing import Pool as pool
import time
import scipy.ndimage

### My other function libraries
import SpecwizardFunctions
import EagleFunctions
import survey_realization_functions


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
element_masses = {'hydrogen' : m_e+m_p, 'carbon' : 1.9944e-23, 'oxygen' : 2.6567e-23, 'silicon' : 4.6637e-23, 'nitrogen' : 2.326e-23}

### Functions

def get_all_id_data(folders_for_recreation_calls):

	if np.size(folders_for_recreation_calls) == 0:
		raise ValueError('folders_for_recreation_calls has zero length (no folders passed)')

	elif np.size(folders_for_recreation_calls) == 1:
		particle_id_folders = glob.glob(str(folders_for_recreation_calls[0]) + '/particle_id_files*')
		num_particle_id_folders = np.size(particle_id_folders)

		list_for_all_id_data = make_2d_list(np.size(folders_for_recreation_calls),2*num_particle_id_folders+1)
		list_for_all_id_data[0][0] = str(folders_for_recreation_calls[0])

		for i in range(0,num_particle_id_folders):
			list_for_all_id_data[0][2*(i+1)-1] = get_gal_id_from_folder_name(particle_id_folders[i])
			list_for_all_id_data[0][2*(i+1)] = np.size(glob.glob(str(particle_id_folders[i]) + '/eagle_particles_hit*.txt'))

	else:
		list_for_all_id_data = []
		for i in range(0,np.size(folders_for_recreation_calls)):
			particle_id_folders = glob.glob(str(folders_for_recreation_calls[i]) + '/particle_id_files*')
			num_particle_id_folders = np.size(particle_id_folders)

			list_for_all_id_data += [[0]*(2*num_particle_id_folders+1)]
			list_for_all_id_data[i][0] = str(folders_for_recreation_calls[i])

			for j in range(0,num_particle_id_folders):
				list_for_all_id_data[i][2*(j+1)-1] = get_gal_id_from_folder_name(particle_id_folders[j])
				list_for_all_id_data[i][2*(j+1)] = np.size(glob.glob(str(particle_id_folders[j]) + '/eagle_particles_hit*.txt'))

	return list_for_all_id_data

def make_2d_list(rows,cols):
	a = []
	for row in xrange(rows): a += [[0]*cols]
	return a


def get_gal_id_from_folder_name(folder):
	gal_id = 'nothing found'
	folder = str(folder)

	for j in range(0,len(folder)):
		if folder[-1*j] == '_':
			gal_id = folder[-1*j+1::]
			break

	if gal_id == 'nothing found':
		raise ValueError('get_gal_id_from_folder failed to find the gal id from the folder containing id files')
		return gal_id
	else:
		return gal_id


def get_particle_properties(list_for_all_id_data, ions, ions_short, elements, lookup_file, new_lines=False):
	not_files = 0
	files = 0
	on_first = True # are we still waiting to start our array concatenation 

	for i in range(0,len(list_for_all_id_data)):

		for j in range(0,(len(list_for_all_id_data[i])-1)/2):

			curr_gal_output = str(list_for_all_id_data[i][0]) + '/gal_output_' + str(list_for_all_id_data[i][2*(j+1)-1]) + '.hdf5'
			curr_spec_output = str(list_for_all_id_data[i][0]) + '/spec.snap_' + str(list_for_all_id_data[i][2*(j+1)-1]) + '.hdf5'

			with h5py.File(curr_gal_output, 'r') as hf1:
				GalaxyProperties = hf1.get('GalaxyProperties')
				snap_directory = check_directory_format(np.array(GalaxyProperties.get('snap_directory'))[0])
				file_keyword = np.array(GalaxyProperties.get('file_keyword'))[0]
				redshift = float(file_keyword[-7:-4]+'.'+file_keyword[-3::])
				group_number = np.array(GalaxyProperties.get('group_number'))[0]
				gal_coords = np.array(GalaxyProperties.get('gal_coords'))[0]
				gal_vel = np.array(GalaxyProperties.get('gal_velocity'))
				gal_R200 = np.array(GalaxyProperties.get('gal_R200'))[0]
				gal_mass = np.array(GalaxyProperties.get('gal_mass'))[0]
				gal_smass = np.array(GalaxyProperties.get('gal_stellar_mass'))[0]
				gal_sSFR = 10.**(np.array(GalaxyProperties.get('log10_sSFR'))[0])

			if file_keyword[-8::] == 'z000p000': # if the sightline was run at z=0 no can't do comparisons do z=0 (one halo I belive)
				print 'sightline was run at z=0'
				print ''
				continue

			virial_vel = (np.sqrt((G*gal_mass*sol_mass_to_g)/(gal_R200*parsec_to_cm*1.e3)))*1.e-5
			curr_snapshot_files = EagleFunctions.get_snap_files(snap_directory, file_keyword)


			if new_lines:
				max_particle_file_tag = get_particle_id_tag(int(list_for_all_id_data[i][2*(j+1)]-1))
				max_filename = str(list_for_all_id_data[i][0]) + '/particle_id_files_' + str(list_for_all_id_data[i][2*(j+1)-1]) + '/eagle_particles_hit_' + max_particle_file_tag + '.hdf5'
				if os.path.isfile(max_filename):
					continue

				gas_ids, gas_coords, gas_vel, particle_mass, time_since_ISM, temperature, density, smoothing_length, metallicity, mass_fracs, nH, ion_fracs, \
				z0_keyword, z0_gal_coords, z0_ids, z0_time_since_ISM, z0_coords, z0_vel, z0_gal_bool = read_in_data(snap_directory, curr_snapshot_files, file_keyword, ions, elements, group_number)

				if z0_gal_bool == False: # if halo was not run to z=0 we can't do comparisons to z=0
					continue


			for k in range(0,list_for_all_id_data[i][2*(j+1)]):
				particle_file_tag = get_particle_id_tag(k)
				
				particles_hit_file = str(list_for_all_id_data[i][0]) + '/particle_id_files_' + str(list_for_all_id_data[i][2*(j+1)-1]) + '/eagle_particles_hit_' + particle_file_tag + '.txt'
				if new_lines:
					if os.path.isfile(particles_hit_file[0:-4]+'.hdf5'):
						continue
					else:
						curr_gas_ids, curr_gas_coords, curr_gas_vel, curr_particle_mass, curr_time_since_ISM, curr_metallicity, curr_temperature, curr_density, curr_smoothing_length, \
						curr_mass_fracs, curr_ion_fracs, curr_lookup_ion_fracs, curr_z0_time_since_ISM, curr_z0_coords, curr_z0_vel \
						= find_matched_particles_and_store_data(particles_hit_file, \
						lookup_file, curr_gal_output, snap_directory, file_keyword, z0_keyword, ions, ions_short, elements, redshift, gas_ids, gas_coords, gal_coords, gas_vel, particle_mass, time_since_ISM, \
						temperature, density, smoothing_length, metallicity, mass_fracs, nH, ion_fracs, z0_gal_coords, z0_ids, z0_time_since_ISM, z0_coords, z0_vel)

				else:

					if os.path.isfile(particles_hit_file[0:-4]+'.hdf5') == False:
						# print 'not a file'
						# print particles_hit_file[0:-4]+'.hdf5'
						# print ''
						not_files += 1
						continue

					files += 1
					# if files > 10:
					# 	raise ValueError('done after 10')
					curr_gas_ids, curr_gas_coords, curr_density, curr_gas_vel, curr_metallicity, curr_particle_mass, curr_smoothing_length, curr_temperature, \
					curr_time_since_ISM, gal_coords, curr_ion_fracs, curr_lookup_ion_fracs, curr_element_fracs, curr_z0_coords, curr_z0_vel, curr_z0_time_since_ISM, \
					z0_gal_coords = \
					read_in_matched_particle_data(particles_hit_file, file_keyword, ions, elements)

					num_particles_hit = np.size(curr_gas_ids)
					frac_never_SF = float(np.size(curr_time_since_ISM[curr_time_since_ISM==0]))/num_particles_hit
					frac_currently_SF = float(np.size(curr_time_since_ISM[curr_time_since_ISM>0]))/num_particles_hit

					indices_recent_SF = np.argwhere((curr_z0_time_since_ISM<0) & ((-1./curr_z0_time_since_ISM)-1<redshift))[:,0]
					z0_frac_never_SF = float(np.size(curr_z0_time_since_ISM[curr_z0_time_since_ISM == 0]))/num_particles_hit
					z0_frac_currently_SF = float(np.size(curr_z0_time_since_ISM[curr_z0_time_since_ISM>0]))/num_particles_hit
					frac_SF_recently = np.size(indices_recent_SF)/num_particles_hit

					curr_particle_radii = np.sqrt(np.sum(np.power(curr_gas_coords-gal_coords,2), axis =1))
					impact_param = np.min(curr_particle_radii)
					curr_z0_particle_radii = np.sqrt(np.sum(np.power(curr_z0_coords-z0_gal_coords,2), axis=1))

					with h5py.File(curr_spec_output, 'r') as hf2:
						curr_spec = hf2.get('Spectrum%s' % (str(k)))
						curr_ion = curr_spec.get('h1') # replace that soon
						col_dense = np.array(curr_ion.get('LogTotalIonColumnDensity'))

					### Possible filters
					if ((col_dense <= 1.5) or (col_dense > 200.5)): # column density filter
						continue

					# plots_for_each_line(ions, gal_mass, col_dense, i, j, k, curr_particle_radii, impact_param, curr_ion_fracs, curr_lookup_ion_fracs, list_for_all_id_data, curr_density, \
					# curr_temperature, element_masses, elements, curr_element_fracs, curr_time_since_ISM, curr_z0_time_since_ISM, redshift, indices_recent_SF, \
					# frac_currently_SF, frac_never_SF, num_particles_hit, curr_z0_particle_radii, z0_frac_never_SF, z0_frac_currently_SF, frac_SF_recently)

					curr_group, curr_radius_bin = get_bins_for_line(gal_smass, gal_sSFR, gal_R200, impact_param)

					if on_first:
						overall_gas_ids, overall_particle_radii, overall_density, overall_gas_vel, overall_metallicity, overall_particle_mass, overall_smoothing_length, overall_temperature, \
		   				overall_time_since_ISM, overall_eagle_ion_fracs, overall_lookup_ion_fracs, overall_element_fracs, overall_z0_particle_radii, \
		   				overall_z0_time_since_ISM, overall_groups, overall_radius_bins \
		   				= create_overall_arrays(curr_gas_ids, curr_particle_radii, curr_density, curr_gas_vel, curr_metallicity, curr_particle_mass, curr_smoothing_length, curr_temperature, \
						curr_time_since_ISM, curr_ion_fracs, curr_lookup_ion_fracs, curr_element_fracs, curr_z0_particle_radii, curr_z0_time_since_ISM, curr_group, curr_radius_bin)

						on_first = False

					else:
						overall_gas_ids, overall_particle_radii, overall_density, overall_gas_vel, overall_metallicity, overall_particle_mass, overall_smoothing_length, overall_temperature, \
			   			overall_time_since_ISM, overall_eagle_ion_fracs, overall_lookup_ion_fracs, overall_element_fracs, overall_z0_particle_radii, overall_z0_time_since_ISM, \
			   			overall_groups, overall_radius_bins \
			   			= appending_all_data(curr_gas_ids, curr_particle_radii, curr_density, curr_gas_vel, curr_metallicity, curr_particle_mass, curr_smoothing_length, curr_temperature, \
						curr_time_since_ISM, curr_ion_fracs, curr_lookup_ion_fracs, curr_element_fracs, curr_z0_particle_radii, curr_z0_time_since_ISM, curr_group, curr_radius_bin, overall_gas_ids, \
						overall_particle_radii, overall_density, overall_gas_vel, overall_metallicity, overall_particle_mass, overall_smoothing_length, overall_temperature, \
			   			overall_time_since_ISM, overall_eagle_ion_fracs, overall_lookup_ion_fracs, overall_element_fracs, overall_z0_particle_radii, overall_z0_time_since_ISM, \
			   			overall_groups, overall_radius_bins)

		   			# plots_with_all_lines(overall_particle_radii, overall_density, overall_gas_vel, overall_metallicity, overall_particle_mass, overall_smoothing_length, overall_temperature, \
			   		# 	overall_time_since_ISM, overall_eagle_ion_fracs, overall_lookup_ion_fracs, overall_element_fracs, overall_z0_particle_radii, overall_z0_time_since_ISM, \
			   		# 	overall_groups, overall_radius_bins)

		   	# if j > 9:
		   	# 	print 'here'
		   	# 	# plots_with_all_lines(overall_particle_radii, overall_density, overall_gas_vel, overall_metallicity, overall_particle_mass, overall_smoothing_length, overall_temperature, \
		   	# 	# 	overall_time_since_ISM, overall_eagle_ion_fracs, overall_lookup_ion_fracs, overall_element_fracs, overall_z0_particle_radii, overall_z0_time_since_ISM, \
		   	# 	# 	overall_groups, overall_radius_bins)
		   	# 	raise ValueError('did 9 gals at least')


	## All lines have now been read in and arrays of data for all particles hit in all sightlines accumulated
	print 'failed and found totals are'
	print not_files
	print files
	print ''
	if new_lines == False:

		plots_with_all_lines(overall_particle_radii, overall_density, overall_gas_vel, overall_metallicity, overall_particle_mass, overall_smoothing_length, overall_temperature, \
			   			overall_time_since_ISM, overall_eagle_ion_fracs, overall_lookup_ion_fracs, overall_element_fracs, overall_z0_particle_radii, overall_z0_time_since_ISM, \
			   			overall_groups, overall_radius_bins)


def get_particle_id_tag(k): # get passed k, the number of the sightline, assign it a 3 digit number for consistency (k=1, returns '001', k=11 returns '011')
	if k < 9:
		particle_file_tag = '00' + str(k+1)
	elif k < 99:
		particle_file_tag = '0' + str(k+1)
	elif k<999:
		particle_file_tag = str(k+1)
	else:
		raise ValueError('k >= 1000 which means it thinks there are more than 1000 sightlines through this galaxy, which is a problem because of how files are named')

	return particle_file_tag

def sorted_search(array, value):
	index = bisect.bisect_left(array,value)
	if index != np.size(array):
		if array[index] == value:
			return index
		else:
			return -1
	else:
		return -1


def lookup_ion_frac_from_specwizard(lookup_file, ion_short, temperature, nHI_density, redshift): 
	# initialize arrays
	ion_fracs = np.empty(np.size(nHI_density))

	# read lookup table
	with h5py.File(lookup_file+ion_short+'.hdf5', 'r') as file:
		ion_bal = np.array(file.get('ionbal'))
		log_dens = np.array(file.get('logd'))
		log_T = np.array(file.get('logt'))
		lookup_redshift = np.array(file.get('redshift'))
		file.close()

	# get nearest redshift table uses (redhisft is the same for all points in spectrum so outside of for loop)
	redshift_index = bisect.bisect_left(lookup_redshift, redshift)
	redshift_index = return_best_index(lookup_redshift, redshift_index, redshift)

	# get nearest nHI_density and temperature for each point in spectra
	# use that to populate neutral fraction array
	for i in range(0,np.size(nHI_density)):
		dense_index = bisect.bisect_left(log_dens, np.log10(nHI_density[i]))
		dense_index = return_best_index(log_dens, dense_index, np.log10(nHI_density[i]))

		temp_index = bisect.bisect_left(log_T, np.log10(temperature[i]))
		temp_index = return_best_index(log_T, temp_index, np.log10(temperature[i]))

		ion_fracs[i] = ion_bal[dense_index][temp_index][redshift_index]

	return ion_fracs


def return_best_index(array, left_index, value):
	if left_index == np.size(array):
		raise ValueError('insertion index was size of array. Value passed was larger than any in array')
	delta = np.abs(array[left_index] - value)
	if np.abs(array[left_index+1] -value) < delta:
		return left_index+1
	else:
		return left_index

def read_in_data(snap_directory, curr_snapshot_files, file_keyword, ions, elements, group_number):
	z0_gal_bool = True

	# Make sure there is a z=0 version of the galaxy
	try:
		z0_file = glob.glob(snap_directory+'snapshot_noneq_***_z000p000')[0]
	except:
		print 'No z=0 galaxy found for ...'
		print snap_directory
		print ''
		z0_gal_bool = False
	
	if z0_gal_bool:
		p = pool(24)
		print 'opened pool'
		### Notes I've made on pool. 1) If you're getting a "NULL result without error in PyObject_Call" error, it is likely because
		### one job (one call of apply_asynch) is taking too much memory. You'll need to reduce its size somehow. My initial thoughts are 
		### breaking arrays with multiple columns into their components or breaking the files up (i.e. for snapshots with 64 hdf5 files make an array
		### for 1-32 and then 33-64. Then concatenate these together after pool is closed. Or maybe do chuncks of 16 since we know that works.
		gas_ids_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/ParticleIDs'], {'include_file_keyword':file_keyword})

		gas_coords_x_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/Coordinates'], {'include_file_keyword':file_keyword, 'column':0})
		gas_coords_y_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/Coordinates'], {'include_file_keyword':file_keyword, 'column':1})
		gas_coords_z_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/Coordinates'], {'include_file_keyword':file_keyword, 'column':2})
		# gas_coords_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/Coordinates'], {'include_file_keyword':file_keyword})

		gas_vel_x_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/Velocity'], {'include_file_keyword':file_keyword, 'column':0})
		gas_vel_y_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/Velocity'], {'include_file_keyword':file_keyword, 'column':1})
		gas_vel_z_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/Velocity'], {'include_file_keyword':file_keyword, 'column':2})
		# gas_vel_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/Velocity'], {'include_file_keyword':file_keyword})

		particle_mass_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/Mass'], {'include_file_keyword':file_keyword})
		time_since_ISM_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/OnEquationOfState'], {'include_file_keyword':file_keyword})
		temperature_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/Temperature'], {'include_file_keyword':file_keyword, 'get_scaling_conversions':False})
		density_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/Density'], {'include_file_keyword':file_keyword})
		smoothing_length_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/SmoothingLength'], {'include_file_keyword':file_keyword})
		metallicity_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/Metallicity'], {'include_file_keyword':file_keyword})
		hydrogen_mass_frac_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/ElementAbundance/Hydrogen'], {'include_file_keyword':file_keyword})
		carbon_mass_frac_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/ElementAbundance/Carbon'], {'include_file_keyword':file_keyword})
		oxygen_mass_frac_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/ElementAbundance/Oxygen'], {'include_file_keyword':file_keyword})
		silicon_mass_frac_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/ElementAbundance/Silicon'], {'include_file_keyword':file_keyword})
		nitrogen_mass_frac_result = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/ElementAbundance/Nitrogen'], {'include_file_keyword':file_keyword})

		ion_fracs_result = [None]*np.size(ions)
		for ion_iter in range(0,np.size(ions)):
			try:
				ion_fracs_result[ion_iter] = p.apply_async(EagleFunctions.read_array, [curr_snapshot_files, 'PartType0/ChemicalAbundances/%s' % (str(ions[ion_iter]))], {'include_file_keyword':file_keyword, 'get_scaling_conversions':False})
			except Exception, e:
				print ''
				print ions[ion_iter]
				print 'ChemicalAbundances not found! Failed at: PartType0/ChemicalAbundances/%s  In %s' % (str(ions[ion_iter]), curr_snapshot_files)
				print ''
				print 'error was %s' % (str(e))
				raise ValueError('ChemicalAbundances not found! Failed at: PartType0/ChemicalAbundances/%s' % (str(ions[ion_iter])))

		z0_suffix = z0_file[-12::]
		z0_keyword = 'snap_noneq_' + z0_suffix 
		subfind_z0_keyword = 'eagle_subfind_tab_' + z0_suffix
		z0_snapshot_files = EagleFunctions.get_snap_files(snap_directory, z0_keyword)
		temp_arr = []
		for file in z0_snapshot_files:
			if 'spec.' in file:
				continue
			else:
				temp_arr.append(file)
		
		z0_snapshot_files = np.asarray(temp_arr)

		z0_gal_coords_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, "Subhalo/CentreOfPotential"], {'include_file_keyword':subfind_z0_keyword})
		GrpIDs_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, "Subhalo/GroupNumber"], {'include_file_keyword':subfind_z0_keyword})
		SubIDs_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, "Subhalo/SubGroupNumber"], {'include_file_keyword':subfind_z0_keyword})
		z0_gas_ids_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, "PartType0/ParticleIDs"], {'include_file_keyword':z0_keyword})
		z0_star_ids_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, "PartType4/ParticleIDs"], {'include_file_keyword':z0_keyword})
		z0_time_since_ISM_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, "PartType0/OnEquationOfState"], {'include_file_keyword':z0_keyword})
		
		z0_gas_coords_x_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, 'PartType0/Coordinates'], {'include_file_keyword':z0_keyword, 'column':0})
		z0_gas_coords_y_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, 'PartType0/Coordinates'], {'include_file_keyword':z0_keyword, 'column':1})
		z0_gas_coords_z_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, 'PartType0/Coordinates'], {'include_file_keyword':z0_keyword, 'column':2})
		# z0_gas_coords_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, "PartType0/Coordinates"], {'include_file_keyword':z0_keyword})
		
		z0_star_coords_x_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, 'PartType4/Coordinates'], {'include_file_keyword':z0_keyword, 'column':0})
		z0_star_coords_y_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, 'PartType4/Coordinates'], {'include_file_keyword':z0_keyword, 'column':1})
		z0_star_coords_z_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, 'PartType4/Coordinates'], {'include_file_keyword':z0_keyword, 'column':2})
		# z0_star_coords_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, "PartType4/Coordinates"], {'include_file_keyword':z0_keyword})
		
		z0_gas_vel_x_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, 'PartType0/Velocity'], {'include_file_keyword':z0_keyword, 'column':0})
		z0_gas_vel_y_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, 'PartType0/Velocity'], {'include_file_keyword':z0_keyword, 'column':1})
		z0_gas_vel_z_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, 'PartType0/Velocity'], {'include_file_keyword':z0_keyword, 'column':2})
		# z0_gas_vel_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, "PartType0/Velocity"], {'include_file_keyword':z0_keyword})
		
		z0_star_vel_x_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, 'PartType4/Velocity'], {'include_file_keyword':z0_keyword, 'column':0})
		z0_star_vel_y_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, 'PartType4/Velocity'], {'include_file_keyword':z0_keyword, 'column':1})
		z0_star_vel_z_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, 'PartType4/Velocity'], {'include_file_keyword':z0_keyword, 'column':2})
		# z0_star_vel_result = p.apply_async(EagleFunctions.read_array, [z0_snapshot_files, "PartType4/Velocity"], {'include_file_keyword':z0_keyword})

		p.close()

		gas_ids = gas_ids_result.get()
		sorted_indices = np.argsort(gas_ids)
		gas_ids = gas_ids[sorted_indices]

		gas_coords_x = gas_coords_x_result.get()[sorted_indices]/(parsec_to_cm*1000)
		gas_coords_y = gas_coords_y_result.get()[sorted_indices]/(parsec_to_cm*1000)
		gas_coords_z = gas_coords_z_result.get()[sorted_indices]/(parsec_to_cm*1000)
		gas_coords = np.column_stack((gas_coords_x, gas_coords_y, gas_coords_z))
		# gas_coords = gas_coords_result.get()[sorted_indices]/(parsec_to_cm*1000)

		gas_vel_x = gas_vel_x_result.get()[sorted_indices]/(1.e5)
		gas_vel_y = gas_vel_y_result.get()[sorted_indices]/(1.e5)
		gas_vel_z = gas_vel_z_result.get()[sorted_indices]/(1.e5)
		gas_vel = np.column_stack((gas_vel_x, gas_vel_y, gas_vel_z))
		# gas_vel = gas_vel_result.get()[sorted_indices]/(1.e5)

		particle_mass = particle_mass_result.get()[sorted_indices]
		time_since_ISM = time_since_ISM_result.get()[sorted_indices]
		temperature = temperature_result.get()[sorted_indices]
		density = density_result.get()[sorted_indices]
		smoothing_length = smoothing_length_result.get()[sorted_indices]/(parsec_to_cm*1000)
		metallicity = metallicity_result.get()[sorted_indices]
		hydrogen_mass_frac = hydrogen_mass_frac_result.get()[sorted_indices]
		carbon_mass_frac = carbon_mass_frac_result.get()[sorted_indices]
		oxygen_mass_frac = oxygen_mass_frac_result.get()[sorted_indices]
		silicon_mass_frac = silicon_mass_frac_result.get()[sorted_indices]
		nitrogen_mass_frac = nitrogen_mass_frac_result.get()[sorted_indices]
		mass_fracs = {'hydrogen' : hydrogen_mass_frac, 'carbon' : carbon_mass_frac, 'oxygen' : oxygen_mass_frac, 'silicon' : silicon_mass_frac, 'nitrogen' : nitrogen_mass_frac}
		nH = density*hydrogen_mass_frac/m_H # number density of all hydrogen (HI and HII)

		ion_fracs = [None]*np.size(ions)
		for ion_iter in range(0,np.size(ions)):
			try:
				ion_fracs[ion_iter] = ion_fracs_result[ion_iter].get()[sorted_indices] # mass fraction of particle that is given ion
			except Exception, r:
				print 'Issue reading out ion_frac array, appending zeros, printing problem galaxy'
				print snap_directory
				print curr_snapshot_files[0::]
				print file_keyword
				print ions[ion_iter]
				print ''
				print ''
				print 'error was %s' % (str(r))
				ion_fracs[ion_iter] = np.zeros(np.size(nH))

		z0_gal_coords = z0_gal_coords_result.get()
		GrpIDs = GrpIDs_result.get()
		SubIDs = SubIDs_result.get()
		index = np.where((GrpIDs == float(group_number)) & (SubIDs == 0))[0] # picks out most massive galaxy in the designated group
		z0_gal_coords = z0_gal_coords[index][0]/(parsec_to_cm*1.e3)

		z0_gas_ids = z0_gas_ids_result.get()
		z0_star_ids = z0_star_ids_result.get()
		z0_ids = np.concatenate((z0_gas_ids, z0_star_ids))
		z0_sorted_indices = np.argsort(z0_ids)
		z0_ids = z0_ids[z0_sorted_indices]

		z0_time_since_ISM = z0_time_since_ISM_result.get()
		z0_time_since_ISM = np.concatenate((z0_time_since_ISM, np.ones(np.size(z0_star_ids))+1.))[z0_sorted_indices]
		
		z0_gas_coords_x = z0_gas_coords_x_result.get()/(parsec_to_cm*1000)
		z0_gas_coords_y = z0_gas_coords_y_result.get()/(parsec_to_cm*1000)
		z0_gas_coords_z = z0_gas_coords_z_result.get()/(parsec_to_cm*1000)
		z0_gas_coords = np.column_stack((z0_gas_coords_x, z0_gas_coords_y, z0_gas_coords_z))
		# z0_gas_coords = z0_gas_coords_result.get()
		
		z0_star_coords_x = z0_star_coords_x_result.get()/(parsec_to_cm*1000)
		z0_star_coords_y = z0_star_coords_y_result.get()/(parsec_to_cm*1000)
		z0_star_coords_z = z0_star_coords_z_result.get()/(parsec_to_cm*1000)
		z0_star_coords = np.column_stack((z0_star_coords_x, z0_star_coords_y, z0_star_coords_z))
		# z0_star_coords = z0_star_coords_result.get()
		
		z0_coords = np.concatenate((z0_gas_coords, z0_star_coords))[z0_sorted_indices]
		
		z0_gas_vel_x = z0_gas_vel_x_result.get()/(1.e5)
		z0_gas_vel_y = z0_gas_vel_y_result.get()/(1.e5)
		z0_gas_vel_z = z0_gas_vel_z_result.get()/(1.e5)
		z0_gas_vel = np.column_stack((z0_gas_vel_x, z0_gas_vel_y, z0_gas_vel_z))
		# z0_gas_vel = z0_gas_vel_result.get()
		
		z0_star_vel_x = z0_star_vel_x_result.get()/(1.e5)
		z0_star_vel_y = z0_star_vel_y_result.get()/(1.e5)
		z0_star_vel_z = z0_star_vel_z_result.get()/(1.e5)
		z0_star_vel = np.column_stack((z0_star_vel_x, z0_star_vel_y, z0_star_vel_z))
		# z0_star_vel = z0_star_vel_result.get()
		
		z0_vel = np.concatenate((z0_gas_vel, z0_star_vel))[z0_sorted_indices]

		return gas_ids, gas_coords, gas_vel, particle_mass, time_since_ISM, temperature, density, smoothing_length, metallicity, mass_fracs, nH, ion_fracs, \
		z0_keyword, z0_gal_coords, z0_ids, z0_time_since_ISM, z0_coords, z0_vel, z0_gal_bool
	else:
		return [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], z0_gal_bool


def find_matched_particles_and_store_data(particles_hit_file, lookup_file, curr_gal_output, snap_directory, file_keyword, z0_keyword, ions, ions_short, elements, redshift, gas_ids, \
gas_coords, gal_coords, gas_vel, particle_mass, time_since_ISM, temperature, density, smoothing_length, metallicity, mass_fracs, nH, ion_fracs, z0_gal_coords, z0_ids, z0_time_since_ISM, z0_coords, z0_vel):

	matched_indices = []
	without_future_stars_indices = []
	z0_matched_indices = []
	lines_read = 0
	with open(particles_hit_file, 'r') as curr_particle_file:
		for line in curr_particle_file:
			curr_id = int(line)
			matched_index = sorted_search(gas_ids, curr_id)
			z0_matched_index = sorted_search(z0_ids, curr_id)

			if ((z0_matched_index != -1) and (matched_index != -1)):
				matched_indices.append(matched_index)
				without_future_stars_indices.append(matched_index)
				z0_matched_indices.append(z0_matched_index)
			else:
				if matched_index == -1:
					raise ValueError('problem: missing gas particle from current redshift gal')
				else:
					raise ValueError('problem: missing gas particle from redshift 0 gal')
			lines_read += 1

	
	if np.size(matched_indices) == 0:
		raise ValueError('No particles were found with matching ids. particles hit file = %s, curr gal output file: %s' % (particles_hit_file, curr_gal_output))

	curr_gas_ids = gas_ids[matched_indices]
	curr_gas_coords = gas_coords[matched_indices]
	curr_gas_vel = gas_vel[matched_indices]
	curr_particle_mass = particle_mass[matched_indices]
	curr_time_since_ISM = time_since_ISM[matched_indices]
	curr_metallicity = metallicity[matched_indices]
	curr_temperature = temperature[matched_indices]
	curr_density = density[matched_indices]
	curr_smoothing_length = smoothing_length[matched_indices]

	curr_mass_fracs = mass_fracs.copy()
	for i in range(len(mass_fracs)):
		curr_mass_fracs.update({curr_mass_fracs.keys()[i] : curr_mass_fracs[curr_mass_fracs.keys()[i]][matched_indices]})

	nH = curr_density*curr_mass_fracs['hydrogen']/m_H # number density of all hydrogen (HI and HII)
	
	curr_ion_fracs = [None]*np.size(ions)
	curr_lookup_ion_fracs = [None]*np.size(ions)

	for ion_iter in range(0,np.size(ions)):
		curr_ion_fracs[ion_iter] = (ion_fracs[ion_iter][matched_indices]*nH*element_masses[elements[ion_iter]])/(curr_mass_fracs[elements[ion_iter]]*curr_density)
		curr_ion_fracs[ion_iter][np.where(curr_mass_fracs[elements[ion_iter]] ==0.0)] = 0
		curr_lookup_ion_fracs[ion_iter] = lookup_ion_frac_from_specwizard(lookup_file, ions_short[ion_iter], curr_temperature, nH, redshift)

	curr_z0_time_since_ISM = z0_time_since_ISM[z0_matched_indices]
	curr_z0_coords = z0_coords[z0_matched_indices]
	curr_z0_vel = z0_vel[z0_matched_indices]

	hdf5_file = particles_hit_file[0:-4]+'.hdf5'
	with h5py.File(hdf5_file,'w') as hf:
		arr_size = np.size(matched_indices)

		hdf5_snap_directory = hf.create_dataset('snap_directory', (1,), maxshape= (None,), data = snap_directory)
		hdf5_snap_directory.attrs['description'] = 'base directory for snap files of this galaxy'

		hdf5_particle_ids = hf.create_dataset('particle_ids', (arr_size,), maxshape= (None,), data=curr_gas_ids.astype(int))
		hdf5_particle_ids.attrs['description'] = 'particle ids of all particles hit by this specwizard line'

		hdf5_curr_redshift_gal = hf.create_group(str(file_keyword[-8::]))

		hdf5_eagle_ion_fracs = hdf5_curr_redshift_gal.create_group('eagle_ion_fracs')
		for ion_iter in range(0,np.size(ions)):
			hdf5_curr_ion_fracs = hdf5_eagle_ion_fracs.create_dataset(ions[ion_iter], (arr_size,), maxshape= (None,), data = curr_ion_fracs[ion_iter])
			hdf5_curr_ion_fracs.attrs['description']='fraction of this particle whose mass is the given ion. Taken from eagle snapshots'

		hdf5_lookup_ion_fracs = hdf5_curr_redshift_gal.create_group('lookup_ion_fracs')
		for ion_iter in range(0,np.size(ions)):
			hdf5_curr_lookup_ion_frac = hdf5_lookup_ion_fracs.create_dataset(ions[ion_iter], (arr_size,), maxshape= (None,), data = curr_lookup_ion_fracs[ion_iter])
			hdf5_curr_lookup_ion_frac.attrs['description']='fraction of this particle whose mass is the given ion. Taken from eagle specwizard lookup tables for redshift,density, and temperature'

		hdf5_element_abundance = hdf5_curr_redshift_gal.create_group('element_abundance')
		for ion_iter in range(0,len(curr_mass_fracs)):
			hdf5_curr_element_abundance = hdf5_element_abundance.create_dataset(curr_mass_fracs.keys()[ion_iter], (arr_size,), maxshape= (None,), data = curr_mass_fracs[curr_mass_fracs.keys()[ion_iter]])
			hdf5_curr_element_abundance.attrs['description'] = 'fraction of the particle whose mass is the given element'

		hdf5_file_keyword=hdf5_curr_redshift_gal.create_dataset('file_keyword', (1,), maxshape= (None,), data = file_keyword)
		hdf5_file_keyword.attrs['description'] = 'keyword that can be used to grab only snap files for the gal at the desired redshift'

		hdf5_gal_coords = hdf5_curr_redshift_gal.create_dataset('gal_coords', (1,3), maxshape= (None,3), data = gal_coords)
		hdf5_gal_coords.attrs['units'] = 'kpc'

		hdf5_gas_coords=hdf5_curr_redshift_gal.create_dataset('gas_coords', (arr_size,3), maxshape= (None,3), data=curr_gas_coords)
		hdf5_gas_coords.attrs['units'] = 'kpc'

		hdf5_gas_vel=hdf5_curr_redshift_gal.create_dataset('gas_vel', (arr_size,3), maxshape= (None,3), data=curr_gas_vel)
		hdf5_gas_vel.attrs['units'] = 'km/s'

		hdf5_particle_mass=hdf5_curr_redshift_gal.create_dataset('particle_mass', (arr_size,), maxshape= (None,), data=curr_particle_mass)
		hdf5_particle_mass.attrs['units'] = 'grams'

		hdf5_time_since_ISM=hdf5_curr_redshift_gal.create_dataset('time_since_ISM', (arr_size,), maxshape= (None,), data=curr_time_since_ISM)
		hdf5_time_since_ISM.attrs['description'] = 'star-formation flag. 0 if it has never formed stars. Positive value if currently star forming, negative value if it has. Then the vlaue indicates the expansion factor (aexp) at which it last formed stars'

		hdf5_temperature=hdf5_curr_redshift_gal.create_dataset('temperature', (arr_size,), maxshape= (None,), data=curr_temperature)
		hdf5_temperature.attrs['units'] = 'Kelvin'

		hdf5_density=hdf5_curr_redshift_gal.create_dataset('density', (arr_size,), maxshape= (None,), data=curr_density)
		hdf5_density.attrs['units'] = 'grams/cm^3'

		hdf5_smoothing_length=hdf5_curr_redshift_gal.create_dataset('smoothing_length', (arr_size,), maxshape= (None,), data=curr_smoothing_length)
		hdf5_smoothing_length.attrs['units'] = 'kpc'

		hdf5_metallicity=hdf5_curr_redshift_gal.create_dataset('metallicity', (arr_size,), maxshape= (None,), data=curr_metallicity)
		hdf5_metallicity.attrs['description'] = 'mass fraction of elements heavier than helium'


		hdf5_zero_redshift_gal = hf.create_group('z000p000')

		hdf5_z0_file_keyword=hdf5_zero_redshift_gal.create_dataset('z0_file_keyword', (1,), maxshape= (None,), data = z0_keyword)
		hdf5_z0_file_keyword.attrs['description'] = 'keyword that can be used to grab only snap files for the gal at the desired redshift'

		hdf5_z0_gal_coords = hdf5_zero_redshift_gal.create_dataset('z0_gal_coords', (1,3), maxshape= (None,3), data=z0_gal_coords)
		hdf5_z0_gal_coords.attrs['units'] = 'kpc'

		hdf5_z0_coords=hdf5_zero_redshift_gal.create_dataset('z0_coords', (arr_size,3), maxshape= (None,3), data=curr_z0_coords)
		hdf5_z0_coords.attrs['units'] = 'kpc'

		hdf5_z0_vel=hdf5_zero_redshift_gal.create_dataset('z0_vel', (arr_size,3), maxshape= (None,3), data=curr_z0_vel)
		hdf5_z0_vel.attrs['units'] = 'kpc'

		hdf5_z0_time_since_ISM=hdf5_zero_redshift_gal.create_dataset('z0_time_since_ISM', (arr_size,), maxshape= (None,), data=curr_z0_time_since_ISM)
		hdf5_z0_time_since_ISM.attrs['description'] = 'star-formation flag. 0 if it has never formed stars. Positive value if currently star forming, negative value if it has. Then the vlaue indicates the expansion factor (aexp) at which it last formed stars'

	return curr_gas_ids, curr_gas_coords, curr_gas_vel, curr_particle_mass, curr_time_since_ISM, curr_metallicity, curr_temperature, curr_density, curr_smoothing_length, \
	curr_mass_fracs, curr_ion_fracs, curr_lookup_ion_fracs, curr_z0_time_since_ISM, curr_z0_coords, curr_z0_vel

def read_in_matched_particle_data(particles_hit_file, file_keyword, ions, elements):
	hdf5_file = particles_hit_file[0:-4]+'.hdf5'
	with h5py.File(hdf5_file, 'r') as hf:

		curr_gas_ids = np.array(hf.get('particle_ids'))
		curr_redshift_gal = hf.get(str(file_keyword[-8::]))

		curr_density = np.array(curr_redshift_gal.get('density'))
		curr_gas_coords = np.array(curr_redshift_gal.get('gas_coords'))
		curr_gas_vel = np.array(curr_redshift_gal.get('gas_vel'))
		curr_metallicity = np.array(curr_redshift_gal.get('metallicity'))
		curr_particle_mass = np.array(curr_redshift_gal.get('particle_mass'))
		curr_smoothing_length = np.array(curr_redshift_gal.get('smoothing_length'))
		curr_temperature = np.array(curr_redshift_gal.get('temperature'))
		curr_time_since_ISM = np.array(curr_redshift_gal.get('time_since_ISM'))
		gal_coords = np.array(curr_redshift_gal.get('gal_coords'))

		eagle_ion_fracs = curr_redshift_gal.get('eagle_ion_fracs')
		lookup_ion_fracs = curr_redshift_gal.get('lookup_ion_fracs')
		element_fracs = curr_redshift_gal.get('element_abundance')

		curr_ion_fracs_keys = []
		curr_element_fracs_keys = []

		for ion_iter in range(0,np.size(ions)):
			curr_ion_fracs_keys.append(ions[ion_iter])
			if ((ion_iter > 0) & (elements[ion_iter] == elements[ion_iter-1])):
				continue
			else:
				curr_element_fracs_keys.append(elements[ion_iter])

		curr_ion_fracs = dict([(key, None) for key in curr_ion_fracs_keys])
		curr_lookup_ion_fracs = dict([(key, None) for key in curr_ion_fracs_keys])
		curr_element_fracs = dict([(key, None) for key in curr_element_fracs_keys])

		for ion_iter in range(0,np.size(ions)):
			curr_ion_fracs[ions[ion_iter]] = np.array(eagle_ion_fracs.get(ions[ion_iter]))
			curr_lookup_ion_fracs[ions[ion_iter]] = np.array(lookup_ion_fracs.get(ions[ion_iter]))
			if ((ion_iter > 0) & (elements[ion_iter] == elements[ion_iter-1])):
				continue
			else:
				curr_element_fracs[elements[ion_iter]] = np.array(element_fracs.get(elements[ion_iter]))

		z0_gal = hf.get('z000p000')

		curr_z0_coords = np.array(z0_gal.get('z0_coords'))
		curr_z0_vel = np.array(z0_gal.get('z0_vel'))
		curr_z0_time_since_ISM = np.array(z0_gal.get('z0_time_since_ISM'))
		z0_gal_coords = np.array(z0_gal.get('z0_gal_coords'))

	return curr_gas_ids, curr_gas_coords, curr_density, curr_gas_vel, curr_metallicity, curr_particle_mass, curr_smoothing_length, curr_temperature, \
	curr_time_since_ISM, gal_coords, curr_ion_fracs, curr_lookup_ion_fracs, curr_element_fracs, curr_z0_coords, curr_z0_vel, curr_z0_time_since_ISM, z0_gal_coords


def plots_for_each_line(ions, gal_mass, col_dense, i, j, k, curr_particle_radii, impact_param, curr_ion_fracs, curr_lookup_ion_fracs, list_for_all_id_data, curr_density, curr_temperature, element_masses, elements, curr_element_fracs, \
	curr_time_since_ISM, curr_z0_time_since_ISM, redshift, indices_recent_SF, frac_currently_SF, frac_never_SF, num_particles_hit, curr_z0_particle_radii, z0_frac_never_SF, \
	z0_frac_currently_SF, frac_SF_recently):
	close_indices = np.where(curr_particle_radii < 500)
	max_plot_radius = 400.
	rvir_frac = 3.

	# num_grid_pts = 30.

	T_max = 8.0
	T_min = 2.5
	nH_max = 0.0
	nH_min = -8.5
	h1_frac_max = -2.0
	h1_frac_min = -8.0
	n_C_max = -7.0
	n_C_min = -21.0
	n_O_max = -6.0
	n_O_min = -20.0

	### plotting params
	plt.rcParams['axes.labelsize'], plt.rcParams['axes.titlesize'], plt.rcParams['legend.fontsize'], plt.rcParams['xtick.labelsize'],  plt.rcParams['ytick.labelsize'] = 16., 20., 13., 14., 14.

	### Star forming in data stuff, z=0.205 is 2.504 Gyr before present, 2.504 Gry before z=0.205 was z=0.489
	### z=0 -> 1.0, z=0.205 -> a = 0.830, z=0.489 -> a=0.672
	will_be_ISM, new_accretion, recycled_accretion, were_ISM_recently, were_and_will_be_ISM = track_ISM(curr_z0_time_since_ISM, curr_time_since_ISM)

	### testing plots H
	nH = curr_density*curr_element_fracs['hydrogen']/m_H
	fig, ax = plt.subplots(1,1)
	cbax = ax.scatter(np.log10(nH), np.log10(curr_temperature), s=0.25, c=np.log10(curr_ion_fracs['HydrogenI']), vmin = h1_frac_min, vmax = h1_frac_max)
	plt.hold(True)
	cb = fig.colorbar(cbax, extend='both')
	cb.set_label(r'$f_{HI}$')
	ax.scatter(np.log10(nH[will_be_ISM]), np.log10(curr_temperature[will_be_ISM]), s=2.5, \
		marker='*', c='k', label = 'Future Accretion')
	ax.scatter(np.log10(nH[were_ISM_recently]), np.log10(curr_temperature[were_ISM_recently]), s=2.5, \
		marker='*', c='r', label = 'Previous ISM')
	ax.scatter(np.log10(nH[were_and_will_be_ISM]), np.log10(curr_temperature[were_and_will_be_ISM]), s=2.5, \
		marker='*', c='g', label = 'Recycling Gas')
	ax.set(ylim = [T_min, T_max], xlim = [nH_min, nH_max])
	ax.set(title=r'$n_{H}$ vs T')
	ax.text(0.5,0.03, r'b=%.0f, $M_{Halo}$=%.1f, $N_{HI}$=%.1f' % (impact_param, np.log10(gal_mass), col_dense), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14)
	ax.set(xlabel=r'$log_{10}(n_H) \, {\rm cm^{-3}}$')
	ax.set(ylabel=r'l$og_{10}(T) \, {\rm K}$')
	ax.legend(loc='upper left', borderpad=0.1, handletextpad=0.01)
	plt.hold(False)
	subprocess.call('mkdir particle_id_files_%s' % (list_for_all_id_data[i][2*(j+1)-1]),shell=True)
	fig.savefig('particle_id_files_%s/n_H_T_H_weight_%s.pdf' % (list_for_all_id_data[i][2*(j+1)-1], k), bbox_inches='tight')
	plt.close(fig)

	# nH_close = nH[close_indices]
	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# cbax = ax.scatter(np.log10(nH_close), np.log10(curr_temperature[close_indices]), s=0.25, c=np.log10(curr_ion_fracs['HydrogenI'][close_indices]))
	# cb = fig.colorbar(cbax)
	# cb.set_label('Neutral Fraction')
	# ax.set(ylim = [T_min, T_max], xlim = [nH_min, nH_max])
	# ax.set(title='n_H T')
	# ax.set(xlabel='n_H')
	# ax.set(ylabel='T (K)')
	# fig.savefig('particle_id_files_%s/n_H_T_close_%s.pdf' % (list_for_all_id_data[i][2*(j+1)-1], k), bbox_inches='tight')
	# plt.close(fig)

	# ### OVI
	# n_O = curr_density*curr_element_fracs['oxygen']/element_masses['oxygen']
	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# cbax = ax.scatter(np.log10(n_O), np.log10(curr_temperature), s=0.25, c=np.log10(curr_ion_fracs['OxygenVI']))
	# cb = fig.colorbar(cbax)
	# cb.set_label('OVI/O')
	# ax.set(ylim = [T_min, T_max], xlim = [n_O_min, n_O_max])
	# ax.set(title='n_O T')
	# ax.set(xlabel='n_O')
	# ax.set(ylabel='T (K)')
	# fig.savefig('particle_id_files_%s/n_O_T_%s.pdf' % (list_for_all_id_data[i][2*(j+1)-1], k), bbox_inches='tight')
	# plt.close(fig)

	# n_O_close = n_O[close_indices]
	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# cbax = ax.scatter(np.log10(n_O_close), np.log10(curr_temperature[close_indices]), s=0.25, c=np.log10(curr_ion_fracs['OxygenVI'][close_indices]))
	# cb = fig.colorbar(cbax)
	# cb.set_label('OVI/O')
	# ax.set(ylim = [T_min, T_max], xlim = [n_O_min, n_O_max])
	# ax.set(title='n_O T')
	# ax.set(xlabel='n_O')
	# ax.set(ylabel='T (K)')
	# fig.savefig('particle_id_files_%s/n_O_T_close_%s.pdf' % (list_for_all_id_data[i][2*(j+1)-1], k), bbox_inches='tight')
	# plt.close(fig)


def check_directory_format(directory):
	if directory[-1] != '/': # checks is directory passed included a '\' character
		directory += '/'
	return directory

def create_overall_arrays(curr_gas_ids, curr_particle_radii, curr_density, curr_gas_vel, curr_metallicity, curr_particle_mass, curr_smoothing_length, curr_temperature, \
						  curr_time_since_ISM, curr_ion_fracs, curr_lookup_ion_fracs, curr_element_fracs, curr_z0_particle_radii, curr_z0_time_since_ISM, \
						  curr_group, curr_radius_bin):
	overall_gas_ids = curr_gas_ids
	overall_particle_radii = curr_particle_radii
	overall_density = curr_density
	overall_gas_vel = curr_gas_vel # n x 3 array
	overall_metallicity = curr_metallicity
	overall_particle_mass = curr_particle_mass
	overall_smoothing_length = curr_smoothing_length
	overall_temperature = curr_temperature
	overall_time_since_ISM = curr_time_since_ISM
	overall_eagle_ion_fracs = curr_ion_fracs.copy() # dictionary with keys = ['HydrogenI', 'CarbonIV', etc]
	overall_lookup_ion_fracs = curr_lookup_ion_fracs.copy() # same as above
	overall_element_fracs = curr_element_fracs.copy() # dictionary with keys = ['hydrogen', 'carbon', etc]
	overall_z0_particle_radii = curr_z0_particle_radii
	overall_z0_time_since_ISM = curr_z0_time_since_ISM
	### per line properties
	num_parts = np.size(curr_gas_ids)
	overall_groups = np.zeros(num_parts) + curr_group
	overall_radius_bins = np.zeros(num_parts) + curr_radius_bin

	return overall_gas_ids, overall_particle_radii, overall_density, overall_gas_vel, overall_metallicity, overall_particle_mass, overall_smoothing_length, overall_temperature, \
		   overall_time_since_ISM, overall_eagle_ion_fracs, overall_lookup_ion_fracs, overall_element_fracs, overall_z0_particle_radii, overall_z0_time_since_ISM, \
		   overall_groups, overall_radius_bins


def appending_all_data(curr_gas_ids, curr_particle_radii, curr_density, curr_gas_vel, curr_metallicity, curr_particle_mass, curr_smoothing_length, curr_temperature, \
					curr_time_since_ISM, curr_ion_fracs, curr_lookup_ion_fracs, curr_element_fracs, curr_z0_particle_radii, curr_z0_time_since_ISM, curr_group, curr_radius_bin, overall_gas_ids, \
					overall_particle_radii, overall_density, overall_gas_vel, overall_metallicity, overall_particle_mass, overall_smoothing_length, overall_temperature, \
		   			overall_time_since_ISM, overall_eagle_ion_fracs, overall_lookup_ion_fracs, overall_element_fracs, overall_z0_particle_radii, overall_z0_time_since_ISM, \
		   			overall_groups, overall_radius_bins):

	overall_gas_ids = np.concatenate((overall_gas_ids,curr_gas_ids))
	overall_particle_radii = np.concatenate((overall_particle_radii, curr_particle_radii))
	overall_density = np.concatenate((overall_density,curr_density))
	overall_gas_vel = np.concatenate((overall_gas_vel, curr_gas_vel), axis = 0) # n x 3 array
	overall_metallicity = np.concatenate((overall_metallicity ,curr_metallicity))
	overall_particle_mass = np.concatenate((overall_particle_mass ,curr_particle_mass))
	overall_smoothing_length = np.concatenate((overall_smoothing_length ,curr_smoothing_length))
	overall_temperature = np.concatenate((overall_temperature, curr_temperature))
	overall_time_since_ISM = np.concatenate((overall_time_since_ISM ,curr_time_since_ISM))
	overall_z0_particle_radii = np.concatenate((overall_z0_particle_radii ,curr_z0_particle_radii))
	overall_z0_time_since_ISM = np.concatenate((overall_z0_time_since_ISM ,curr_z0_time_since_ISM))
	### features of the galaxy/line
	num_parts = np.size(curr_gas_ids)
	overall_groups = np.concatenate((overall_groups, np.zeros(num_parts) + curr_group))
	overall_radius_bins = np.concatenate((overall_radius_bins, np.zeros(num_parts) + curr_radius_bin))

	for key in curr_ion_fracs.keys():
		overall_eagle_ion_fracs[key] = np.concatenate((overall_eagle_ion_fracs[key], curr_ion_fracs[key]))
		overall_lookup_ion_fracs[key] = np.concatenate((overall_lookup_ion_fracs[key], curr_lookup_ion_fracs[key]))

	for key in curr_element_fracs.keys():
		overall_element_fracs[key] = np.concatenate((overall_element_fracs[key], curr_element_fracs[key]))

	return overall_gas_ids, overall_particle_radii, overall_density, overall_gas_vel, overall_metallicity, overall_particle_mass, overall_smoothing_length, overall_temperature, \
		   overall_time_since_ISM, overall_eagle_ion_fracs, overall_lookup_ion_fracs, overall_element_fracs, overall_z0_particle_radii, overall_z0_time_since_ISM, \
		   overall_groups, overall_radius_bins

def plots_with_all_lines(overall_particle_radii, overall_density, overall_gas_vel, overall_metallicity, overall_particle_mass, overall_smoothing_length, overall_temperature, \
		   				 overall_time_since_ISM, overall_eagle_ion_fracs, overall_lookup_ion_fracs, overall_element_fracs, overall_z0_particle_radii, overall_z0_time_since_ISM, 
		   				 overall_groups, overall_radius_bins):
	
	### set bounds
	close_indices = np.where(overall_particle_radii < 500.)
	T_max = 8.
	T_min = 2.5
	nH_max = 0.
	nH_min = -8.5
	f_HI_max = -1.
	f_HI_min = -7.0
	f_OVI_max = 0.0
	f_OVI_min = -6.0

	### plotting params
	plt.rcParams['axes.labelsize'], plt.rcParams['axes.titlesize'], plt.rcParams['legend.fontsize'], plt.rcParams['xtick.labelsize'],  plt.rcParams['ytick.labelsize'] = 17., 17., 13., 14., 16.

	### If I want to filter plots by where they're close
	[overall_z0_time_since_ISM, overall_time_since_ISM, overall_particle_radii, overall_particle_mass, overall_density, overall_temperature, overall_lookup_ion_fracs['HydrogenI'], overall_element_fracs['hydrogen'], overall_groups, overall_radius_bins] \
	= return_where_close(radius=500., radii=overall_particle_radii, arrays_to_filter=[overall_z0_time_since_ISM, overall_time_since_ISM, \
		overall_particle_radii, overall_particle_mass, overall_density, overall_temperature, overall_lookup_ion_fracs['HydrogenI'], overall_element_fracs['hydrogen'], overall_groups, overall_radius_bins])

	### get indices for past and future ISM interaction
	overall_will_be_ISM, overall_new_accretion, overall_recycled_accretion, overall_were_ISM, overall_were_and_will_be_ISM = track_ISM(overall_z0_time_since_ISM, overall_time_since_ISM)

	### get number of particles and extract nH
	num_parts = float(np.size(overall_particle_radii))
	overall_nH = overall_density*overall_element_fracs['hydrogen']/m_H
	frac_will_be_ISM, frac_were_ISM, frac_both = np.array([np.size(overall_will_be_ISM), np.size(overall_were_ISM), np.size(overall_were_and_will_be_ISM)])/num_parts

	num_levels = 2 # really number of levels minus one becuase of how contour handles this value
	hist_bins = 50 # for the 2d hists this is the number of bins PER AXIS (ex: hist_bins=10 -> 10 x 10 grid)
	all_bar_x_vals = np.array([1,11,21])
	all_bar_heights = np.array([frac_will_be_ISM, frac_were_ISM, frac_both])
	all_bar_heights = np.where(all_bar_heights == 0., 1.0e-4, all_bar_heights)

	print "Overall H Fracs: Will %.2e, were %.2e, both %.2e" % (frac_will_be_ISM, frac_were_ISM, frac_both)
	print ""

	colors = np.array(['g','b','r'])
	color_labels = ['Low Mass', 'Active', 'Passive']
	edge_styles = np.array(['-', '--', ':'])
	edge_labels = [r'$r/R_{vir}$ $\leq$ 0.5', r'0.5 < $r/R_{vir}$ $\leq$ 1.0', r'$r/R_{vir}$ > 1.0']
	filename_edgle_labels = ['low_r', 'mid_r', 'high_r']
	patches_list1 = [mpatches.Patch(color='k', label = 'All')]
	patches_list2 = []
	for i in range(3):
		patches_list1.append(mpatches.Patch(color=colors[i], label = color_labels[i]))

	for i in range(3):
		patches_list2.append(mpatches.Patch(facecolor = 'w', edgecolor = 'k', linestyle = edge_styles[i], label = edge_labels[i]))

	fig, ax = plt.subplots(1)
	ax.bar(all_bar_x_vals, all_bar_heights, color='k')
	plt.hold(True)

	for group_identifier in range(3): # this is low mass, active, passive
		curr_group_indices = np.where(overall_groups == group_identifier)
		for radius_bin_identifier in range(3): # this is within 0.5, 0.5-1, and >1. R_vir
			curr_radius_indices = np.where(overall_radius_bins == radius_bin_identifier)

			curr_indices = np.intersect1d(curr_group_indices, curr_radius_indices)
			curr_num_parts = float(np.size(curr_indices))

			curr_will, curr_new, curr_recycled, curr_were, curr_both = track_ISM(overall_z0_time_since_ISM[curr_indices], overall_time_since_ISM[curr_indices])

			curr_frac_will, curr_frac_new, curr_frac_recycled, curr_frac_were, curr_frac_both = np.array([np.size(curr_will), np.size(curr_new), np.size(curr_recycled), np.size(curr_were), np.size(curr_both)])/curr_num_parts
			
			if curr_num_parts != 0.0:
				curr_nH, curr_T = [overall_nH[curr_indices], overall_temperature[curr_indices]]
				height, xedges, yedges = np.histogram2d(np.log10(curr_nH), np.log10(curr_T), range=[[nH_min, nH_max], [T_min, T_max]], bins=hist_bins) #, weights = overall_lookup_ion_fracs['HydrogenI'][curr_indices])
				height = np.log10(height)
				if curr_frac_were != 0.0:
					height_were, xedges_were, yedges_were = np.histogram2d(np.log10(curr_nH[curr_were]), np.log10(curr_T[curr_were]), range=[[nH_min, nH_max], [T_min, T_max]], bins=hist_bins)
					height_will, xedges_will, yedges_will = np.histogram2d(np.log10(curr_nH[curr_will]), np.log10(curr_T[curr_will]), range=[[nH_min, nH_max], [T_min, T_max]], bins=hist_bins)
					# replace zeros with ones, one goes to zero when we take the log and enough particles that making the floor value 1 is fine
					height_were, height_will = [np.where(height_were==0, 1, height_were), np.where(height_will==0, 1, height_will)]
					height_were, height_will = [np.log10(height_were), np.log10(height_will)] # move to log space
					were_max, will_max = [np.max(np.max(height_were)), np.max(np.max(height_will))] # Get the max height, if not > 10 pts this will cause issues for levels in plot
					were_bool, will_bool = [True if were_max > -10. else False, True if will_max > -10. else False] # so we don't contour if this is <= 1 (log10(10)=1)

				n_t_fig = plt.figure()
				n_t_ax = plt.gca()
				image = n_t_ax.imshow(height.transpose(), origin='lower', aspect = 'auto', cmap = 'gray_r', extent = (nH_min, nH_max, T_min, T_max))

				if curr_frac_were != 0.0:
					if were_bool:
						n_t_ax.contour(height_were.transpose(), levels = [were_max-1.0, were_max-0.5], extent=(np.min(xedges), np.max(xedges), np.min(yedges), np.max(yedges)), colors='r', linestyles=edge_styles[-2::-1])
					if will_bool:
						n_t_ax.contour(height_will.transpose(), levels = [will_max-1.0, will_max-0.5], extent=(np.min(xedges), np.max(xedges), np.min(yedges), np.max(yedges)), colors='b', linestyles=edge_styles[-2::-1])

				cb = n_t_fig.colorbar(image)
				cb.set_label(r'$log_{10}(N_{particles})$') # \times f_{HI})$')
				n_t_ax.set_title(r'$n_{H}$ vs T: %s %s' % (color_labels[group_identifier], edge_labels[radius_bin_identifier]))
				n_t_ax.set_xlabel(r'$log_{10}(n_H)$ $cm^{-3}$')
				n_t_ax.set_ylabel(r'$log_{10}(T)$ K')
				plt.tight_layout()
				# n_t_fig.savefig('n_t_hist_%s_%s.pdf' % (color_labels[group_identifier], filename_edgle_labels[radius_bin_identifier]))
				plt.close(n_t_fig)
				if ((group_identifier == 1) & (radius_bin_identifier == 0)):
					print "fracs for active, low b gals"
					print curr_frac_will
					print curr_frac_were
					print curr_frac_both
					print ''


			curr_bar_xvals = all_bar_x_vals + radius_bin_identifier+1 + (group_identifier)*3
			curr_bar_xvals = np.concatenate((curr_bar_xvals, [curr_bar_xvals[0]]))
			curr_bar_heights = np.array([curr_frac_will, curr_frac_were, curr_frac_both])
			curr_bar_heights = np.concatenate((curr_bar_heights, [curr_frac_new]))
			curr_bar_heights = np.where(curr_bar_heights <= 1.0e-5, 1.0e-5, curr_bar_heights)

			ax.bar(curr_bar_xvals, curr_bar_heights, color=colors[group_identifier], edgecolor = 'k', linestyle=edge_styles[radius_bin_identifier])

	plt.hold(False)
	ax.set_title(r'Origin and Fate of Gas') #: ${\rm log}_{10}(N_{HI})$ $<$ 16.5')
	ax.set_ylim(ymin = 1.e-5,ymax=500.0)
	ax.set_ylabel(r'${\rm log}_{10}(f)$')
	plt.xticks([0.5, 5, 10.5, 15, 20.5, 25, 30.5], ['','Future Accretion','', 'Previous ISM','', 'Recycling Gas',''])
	ax.axvline(0.5, color='k', linewidth=0.5)
	ax.axvline(10.5, color='k', linewidth=0.5)
	ax.axvline(20.5, color='k', linewidth=0.5)
	ax.axvline(30.5, color='k', linewidth=0.5)
	legend1 = ax.legend(loc='upper left', handles = patches_list1) # [0.05,0.68]
	legend2 = ax.legend(loc='upper right', handles = patches_list2) # [0.53,0.76]
	ax.add_artist(legend1)
	ax.add_artist(legend2)
	ax.set_yscale('log')
	ax.set_yticklabels(np.log10(ax.get_yticks()).astype(int))
	fig.savefig('ISM_hists.pdf')
	plt.close(fig)

	### HI traced stuff

	overall_HI_masses = overall_particle_mass*overall_lookup_ion_fracs["HydrogenI"]
	total_HI_mass = np.sum(overall_HI_masses)
	frac_HI_will_be_ISM, frac_HI_were_ISM, frac_HI_both = np.array([np.sum(overall_HI_masses[overall_will_be_ISM]), np.sum(overall_HI_masses[overall_were_ISM]), np.sum(overall_HI_masses[overall_were_and_will_be_ISM])])/total_HI_mass
	print "Overall HI Fracs: Will %.2e, were %.2e, both %.2e" % (frac_HI_will_be_ISM, frac_HI_were_ISM, frac_HI_both)
	print ""
	all_HI_bar_heights = np.array([frac_HI_will_be_ISM, frac_HI_were_ISM, frac_HI_both])
	all_HI_bar_heights = np.where(all_HI_bar_heights <= 1.0e-5, 1.0e-5, all_HI_bar_heights)

	HI_fig, HI_ax = plt.subplots(1)
	HI_ax.bar(all_bar_x_vals, all_HI_bar_heights, color='k')
	plt.hold(True)

	for group_identifier in range(3): # this is low mass, active, passive
		curr_group_indices = np.where(overall_groups == group_identifier)
		for radius_bin_identifier in range(3): # this is within 0.5, 0.5-1, and >1. R_vir
			curr_radius_indices = np.where(overall_radius_bins == radius_bin_identifier)
			curr_indices = np.intersect1d(curr_group_indices, curr_radius_indices)
			curr_HI_masses = overall_HI_masses[curr_indices]
			curr_total_HI_mass = np.sum(curr_HI_masses)

			curr_will, curr_new, curr_recycled, curr_were, curr_both = track_ISM(overall_z0_time_since_ISM[curr_indices], overall_time_since_ISM[curr_indices])

			curr_HI_mass_frac_will, curr_HI_mass_frac_new, curr_HI_mass_frac_recycled, curr_HI_mass_frac_were, curr_HI_mass_frac_both = np.array([np.sum(curr_HI_masses[curr_will]),
				np.sum(curr_HI_masses[curr_new]), np.sum(curr_HI_masses[curr_recycled]), np.sum(curr_HI_masses[curr_were]), np.sum(curr_HI_masses[curr_both])])/curr_total_HI_mass

			if ((group_identifier == 1) & (radius_bin_identifier == 0)):
				print "fracs for active, low b gals HI weighted"
				print curr_HI_mass_frac_will
				print curr_HI_mass_frac_were
				print curr_HI_mass_frac_both
				print ''

			curr_bar_xvals = all_bar_x_vals + radius_bin_identifier+1 + (group_identifier)*3
			curr_bar_xvals = np.concatenate((curr_bar_xvals, [curr_bar_xvals[0]]))
			curr_HI_mass_bar_heights = np.array([curr_HI_mass_frac_will, curr_HI_mass_frac_were, curr_HI_mass_frac_both])
			curr_HI_mass_bar_heights = np.concatenate((curr_HI_mass_bar_heights, [curr_HI_mass_frac_new]))
			curr_HI_mass_bar_heights = np.where(curr_HI_mass_bar_heights <= 1.0e-5, 1.0e-5, curr_HI_mass_bar_heights)

			HI_ax.bar(curr_bar_xvals, curr_HI_mass_bar_heights, color=colors[group_identifier], edgecolor = 'k', linestyle=edge_styles[radius_bin_identifier])

	plt.hold(False)
	HI_ax.set_title(r'Origin and Fate of HI Gas') #: ${\rm log}_{10}(N_{HI})$ $<$ 16.5')
	HI_ax.set_ylim(ymin = 1.e-5,ymax=500.0)
	HI_ax.set_ylabel(r'${\rm log}_{10}(f_{M_{HI}})$')
	plt.xticks([0.5, 5, 10.5, 15, 20.5, 25, 30.5], ['','Future Accretion','', 'Previous ISM','', 'Recycling Gas',''])
	HI_ax.axvline(0.5, color='k', linewidth=0.5)
	HI_ax.axvline(10.5, color='k', linewidth=0.5)
	HI_ax.axvline(20.5, color='k', linewidth=0.5)
	HI_ax.axvline(30.5, color='k', linewidth=0.5)
	legend1 = HI_ax.legend(ncol=1,loc='upper left', handles = patches_list1) # [0.05,0.68]
	legend2 = HI_ax.legend(loc='upper right', handles = patches_list2) # [0.53,0.76]
	HI_ax.add_artist(legend1)
	HI_ax.add_artist(legend2)
	HI_ax.set_yscale('log')
	HI_ax.set_yticklabels(np.log10(HI_ax.get_yticks()).astype(int))
	HI_fig.savefig('ISM_HI_hists.pdf')
	plt.close(HI_fig)

	# ### Hydrogen
	# nH = overall_density*overall_element_fracs['hydrogen']/m_H
	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# cbax = ax.scatter(np.log10(nH), np.log10(overall_temperature), s=0.25, c=np.log10(overall_eagle_ion_fracs['HydrogenI']))
	# cb = fig.colorbar(cbax)
	# cb.set_label('Neutral Fraction')
	# cb.set_clim([f_HI_min, f_HI_max])
	# cb.set_ticks(np.arange(f_HI_min, f_HI_max))
	# ax.set(ylim = [T_min, T_max], xlim = [nH_min, nH_max])
	# ax.set(title='n_H T')
	# ax.set(xlabel='n_H')
	# ax.set(ylabel='T (K)')
	# fig.savefig('n_H_T_all_lines_eagle.pdf', bbox_inches='tight')
	# plt.close(fig)

	# nH_close = nH[close_indices]
	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# cbax = ax.scatter(np.log10(nH_close), np.log10(overall_temperature[close_indices]), s=0.25, c=np.log10(overall_eagle_ion_fracs['HydrogenI'][close_indices]))
	# cb = fig.colorbar(cbax)
	# cb.set_label('Neutral Fraction')
	# cb.set_clim([f_HI_min, f_HI_max])
	# cb.set_ticks(np.arange(f_HI_min, f_HI_max))
	# ax.set(ylim = [T_min, T_max], xlim = [nH_min, nH_max])
	# ax.set(title='n_H T')
	# ax.set(xlabel='n_H')
	# ax.set(ylabel='T (K)')
	# fig.savefig('n_H_T_all_lines_eagle_close.pdf', bbox_inches='tight')
	# plt.close(fig)

	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# cbax = ax.scatter(np.log10(nH), np.log10(overall_temperature), s=0.25, c=np.log10(overall_lookup_ion_fracs['HydrogenI']))
	# cb = fig.colorbar(cbax)
	# cb.set_label('Neutral Fraction')
	# cb.set_clim([f_HI_min, f_HI_max])
	# cb.set_ticks(np.arange(f_HI_min, f_HI_max))
	# ax.set(ylim = [T_min, T_max], xlim = [nH_min, nH_max])
	# ax.set(title='n_H T')
	# ax.set(xlabel='n_H')
	# ax.set(ylabel='T (K)')
	# fig.savefig('n_H_T_all_lines_lookup.pdf', bbox_inches='tight')
	# plt.close(fig)

	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# cbax = ax.scatter(np.log10(nH_close), np.log10(overall_temperature[close_indices]), s=0.25, c=np.log10(overall_lookup_ion_fracs['HydrogenI'][close_indices]))
	# cb = fig.colorbar(cbax)
	# cb.set_label('Neutral Fraction')
	# cb.set_clim([f_HI_min, f_HI_max])
	# cb.set_ticks(np.arange(f_HI_min, f_HI_max))
	# ax.set(ylim = [T_min, T_max], xlim = [nH_min, nH_max])
	# ax.set(title='n_H T')
	# ax.set(xlabel='n_H')
	# ax.set(ylabel='T (K)')
	# fig.savefig('n_H_T_all_lines_lookup_close.pdf', bbox_inches='tight')
	# plt.close(fig)

	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# ax.plot(overall_eagle_ion_fracs['HydrogenI']/overall_lookup_ion_fracs['HydrogenI'], markersize=1.0, marker='.', markerfacecolor='blue', linestyle='None')
	# ax.set(xlabel='particle index in array (meaningless', ylabel='eagle/lookup ratio', title='Eagle to Lookup Table Ratio For Hydrogen')
	# fig.savefig('hydrogen_eagle_lookup_ratio.pdf')

	# ### Oxygen
	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# cbax = ax.scatter(np.log10(nH), np.log10(overall_temperature), s=0.25, c=np.log10(overall_eagle_ion_fracs['OxygenVI']))
	# cb = fig.colorbar(cbax)
	# cb.set_label('OVI/O')
	# cb.set_clim([f_OVI_min, f_OVI_max])
	# cb.set_ticks(np.arange(f_OVI_min, f_OVI_max))
	# ax.set(ylim = [T_min, T_max], xlim = [nH_min, nH_max])
	# ax.set(title='%s n_O T' % (impact_param))
	# ax.set(xlabel='n_O')
	# ax.set(ylabel='T (K)')
	# fig.savefig('n_O_T_all_lines_eagle.pdf', bbox_inches='tight')
	# plt.close(fig)

	# nH_close = nH[close_indices]
	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# cbax = ax.scatter(np.log10(nH_close), np.log10(overall_temperature[close_indices]), s=0.25, c=np.log10(overall_eagle_ion_fracs['OxygenVI'][close_indices]))
	# cb = fig.colorbar(cbax)
	# cb.set_label('OVI/O')
	# cb.set_clim([f_OVI_min, f_OVI_max])
	# cb.set_ticks(np.arange(f_OVI_min, f_OVI_max))
	# ax.set(ylim = [T_min, T_max], xlim = [nH_min, nH_max])
	# ax.set(title='%s n_O T' % (impact_param))
	# ax.set(xlabel='n_O')
	# ax.set(ylabel='T (K)')
	# fig.savefig('n_O_T_all_lines_eagle_close.pdf', bbox_inches='tight')
	# plt.close(fig)

	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# cbax = ax.scatter(np.log10(nH), np.log10(overall_temperature), s=0.25, c=np.log10(overall_lookup_ion_fracs['OxygenVI']))
	# cb = fig.colorbar(cbax)
	# cb.set_label('OVI/O')
	# cb.set_clim([f_OVI_min, f_OVI_max])
	# cb.set_ticks(np.arange(f_OVI_min, f_OVI_max))
	# ax.set(ylim = [T_min, T_max], xlim = [nH_min, nH_max])
	# ax.set(title='%s n_O T' % (impact_param))
	# ax.set(xlabel='n_O')
	# ax.set(ylabel='T (K)')
	# fig.savefig('n_O_T_all_lines_lookup.pdf', bbox_inches='tight')
	# plt.close(fig)

	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# cbax = ax.scatter(np.log10(nH_close), np.log10(overall_temperature[close_indices]), s=0.25, c=np.log10(overall_lookup_ion_fracs['OxygenVI'][close_indices]))
	# cb = fig.colorbar(cbax)
	# cb.set_label('OVI/O')
	# cb.set_clim([f_OVI_min, f_OVI_max])
	# cb.set_ticks(np.arange(f_OVI_min, f_OVI_max))
	# ax.set(ylim = [T_min, T_max], xlim = [nH_min, nH_max])
	# ax.set(title='%s n_O T' % (impact_param))
	# ax.set(xlabel='n_O')
	# ax.set(ylabel='T (K)')
	# fig.savefig('n_O_T_all_lines_lookup_close.pdf', bbox_inches='tight')
	# plt.close(fig)

	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# ax.semilogy(overall_eagle_ion_fracs['OxygenVI']/overall_lookup_ion_fracs['OxygenVI'],  markersize=1.0, marker='.', markerfacecolor='blue', linestyle='None')
	# ax.set(xlabel='particle index in array (meaningless', ylabel='eagle/lookup ratio', title='Eagle to Lookup Table Ratio For Oxygen')
	# fig.savefig('oxygen_eagle_lookup_ratio.pdf')

def track_ISM(z0_time_since_ISM, time_since_ISM):
	will_be_ISM = np.where(((z0_time_since_ISM <= -0.830) & (z0_time_since_ISM < 0.)))[0] 
	is_a_star_indices = np.where((z0_time_since_ISM > 0.))[0]
	will_be_ISM = np.concatenate((will_be_ISM, is_a_star_indices))

	new_accretion = np.where(time_since_ISM[will_be_ISM] == 0)
	recycled_accretion = np.where(time_since_ISM[will_be_ISM] < 0)

	were_ISM_recently = np.where(((time_since_ISM <= -0.672) & (time_since_ISM > -0.830) & (time_since_ISM < 0.)))[0] 

	were_and_will_be_ISM = np.intersect1d(will_be_ISM, were_ISM_recently)

	return will_be_ISM, new_accretion, recycled_accretion, were_ISM_recently, were_and_will_be_ISM


def get_bins_for_line(gal_smass, gal_sSFR, gal_R200, impact_param):
	### figure out which subgroup this line/galaxy is in options are
	### group: low_mass (gal_smass < 9.7 in EAGLE), actie, passive (divided at ssfr = -11.) [0,1,2]
	### radius_bin: impact param less than 0.5, 1.0, inf r_vir [0,1,2]
	if np.log10(gal_smass) < 9.7:
		curr_group = 0
	elif np.log10(gal_sSFR) < -11.0:
		curr_group = 2
	else:
		curr_group = 1

	virial_impact_param = impact_param/gal_R200
	if virial_impact_param <= 0.5:
		curr_radius_bin = 0
	elif virial_impact_param <= 1.:
		curr_radius_bin = 1
	else:
		curr_radius_bin = 2

	return curr_group, curr_radius_bin

def return_where_close(radius, radii, arrays_to_filter):
	indices = np.where(radii<=radius)
	for i in range(0,len(arrays_to_filter)):
		arrays_to_filter[i] = arrays_to_filter[i][indices]

	return arrays_to_filter



