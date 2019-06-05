### Custom functions for importing and dealing with Eagle arrays/attributes as well as some useful functions
### Author: Ryan Horton 
### Created 4/25/2016

### Imports

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as plt3d
import sys
import h5py
import glob
import warnings
import time
from multiprocessing import Pool as pool
import survey_realization_functions

### Constants
parsec_in_cm = 3.0857e18 # cm
G = 6.674e-8
M_sol = 1.98855e33 # g
secs_per_year = 3.15e7
kB = 1.380648e-16

### Atom Constants
atomic_mass_unit = 1.66053904e-24 # g
#atom masses given in atomic mass units
H_atom_mass = 1.0
He_atom_mass = 4.0
C_atom_mass = 12.0
Fe_atom_mass = 56.0
Mg_atom_mass = 24.0
Ne_atom_mass = 20.0
N_atom_mass = 14.0
O_atom_mass = 16.0
Si_atom_mass = 28.0
e_mass = 1.0/1836.0

atom_mass_array = np.asarray([H_atom_mass,He_atom_mass,C_atom_mass,Fe_atom_mass,Mg_atom_mass,Ne_atom_mass,N_atom_mass,O_atom_mass,Si_atom_mass,e_mass])

### functions
# get snap files from a directory that have the particles included keyword. 
# usually we only want the specific directory passes (snap_directory+particles_included_keyword basically) but sometimes a weird naming conveneiton (rot for example)
# means we want to get everything with the right name/redshift. all_directories should be set to true in this case. 
def get_snap_files(snap_directory, particles_included_keyword, all_directories=False):
	glob_snap_directory = str(snap_directory) # makes directory a string so it can be passed to glob (python function)
	if glob_snap_directory[-1] != '/': # checks is directory passed included a '\' character and then makes sure it only looks at .hdf5 files
		glob_snap_directory += '/'

	snap_files = glob.glob(glob_snap_directory+"*hdf5") # list of snap files in the directory
	# removed a wildcard in the two lines of code below this to get rid of directories ending in .ioneq
	# if this causes problems look into another solution. 
	# old line
	if all_directories:
		snap_files = np.concatenate([snap_files,glob.glob(glob_snap_directory+str(particles_included_keyword[0:4])+'shot'+str(particles_included_keyword[4::])+'*/'+str(particles_included_keyword)+'*')])
	else:
		snap_files = np.concatenate([snap_files,glob.glob(glob_snap_directory+str(particles_included_keyword[0:4])+'shot'+str(particles_included_keyword[4::])+'/'+str(particles_included_keyword)+'*')])
	# in case there are _noneq_ files in a folder without the noneq keyword. It was done apparently before the naming convention was there
	snap_files = np.concatenate([snap_files,glob.glob(glob_snap_directory+str(particles_included_keyword[0:4])+'shot'+str(particles_included_keyword[4::])+'/'+ str(particles_included_keyword[0:4])+'_noneq'+str(particles_included_keyword[4::])+'*')])
	snap_files = np.concatenate([snap_files, glob.glob(glob_snap_directory+'groups_'+str(particles_included_keyword[-12::])+'/*hdf5')])
	return snap_files

def read_attribute(snap_files,array_name,attribute,include_file_keyword=""): # Extracts the atrribute from the snap filess passed
	# file type is a keyword that is in the file name. Ex: subfind, group, ioneq. If none it searches all files in the directory
	#snap_files = glob.glob(glob_snap_directory)
	attr = None
	if include_file_keyword == "":
		for file in snap_files:
			h5py_file = h5py.File(file,'r')
		 	attr = h5py_file[array_name].attrs[attribute]
			break # only extracts attribute from one file (becuase they're the same)
	else:
		for file in snap_files:
			if ((include_file_keyword in file) or (include_file_keyword[0:4]+'_noneq'+include_file_keyword[4::] in file)): # if noneq file in folder w/o
				try:
					h5py_file = h5py.File(file,'r')
				except:
					print file
					print h5py_file
					print array_name
					print attribute
					print ''
					raise ValueError('Could not find attribute in this file even though the keyword matched.')
				attr = h5py_file[array_name].attrs[attribute]
				break # only extracts once, attribute the same for all elements/files in array
	if(attr == None):
		print 'file keyword'
		print include_file_keyword
		print ''
		print 'array name is'
		print array_name
		print ''
		print 'file names'
		print snap_files
		print ''
		print 'attribute'
		print attribute
		print''
		h5py_file = h5py.File(snap_files[0],'r')
		print h5py_file[array_name]
		print h5py_file[array_name].attrs.keys()
		print ''
		raise ValueError("No particles found. Most likely no files matching file flags/included keywords")
	return attr

def read_array(snap_files,array_name,include_file_keyword="",column=None, get_scaling_conversions=True): # extracts arrays of parameters
	# file type is a keyword that is in the file name. Ex: subfind, group, ioneq. If none it searches all files in the directory
	#snap_files = glob.glob(glob_snap_directory)
	iteration = 0
	if get_scaling_conversions:
		h_scale_exponent = read_attribute(snap_files,array_name,'h-scale-exponent',include_file_keyword=include_file_keyword)
		a_exponent = read_attribute(snap_files,array_name,'aexp-scale-exponent',include_file_keyword=include_file_keyword)
		cgs_conversion = read_attribute(snap_files,array_name,'CGSConversionFactor',include_file_keyword=include_file_keyword)
		hubble_param = read_attribute(snap_files,'Header','HubbleParam',include_file_keyword=include_file_keyword)
		expansion_factor = read_attribute(snap_files,'Header','ExpansionFactor',include_file_keyword=include_file_keyword)

	if include_file_keyword == "":
		for file in snap_files: # iterates over files
			h5py_file = h5py.File(file,'r')
			if column != None:
				array = np.asarray(h5py_file[array_name])[:,column] # opens the array for one of the files
			else:
				array = np.asarray(h5py_file[array_name])

			if get_scaling_conversions:
				array = np.asarray((array)*(hubble_param**h_scale_exponent)*(expansion_factor**a_exponent)*(cgs_conversion)) # converts to mks units

			if iteration > 0: # attaches arrays from all files together
				final_array = np.concatenate((final_array,array))
			else: 
				final_array = array
			iteration += 1
	else:
		for file in snap_files: # iterates over files
			if ((include_file_keyword in file) or (include_file_keyword[0:4]+'_noneq'+include_file_keyword[4::] in file)): # if noneq file in folder w/o
				h5py_file = h5py.File(file,'r')
				if column != None:
					array = np.asarray(h5py_file[array_name])[:,column] # opens the array for one of the files
				else:
					array = np.asarray(h5py_file[array_name])

				if get_scaling_conversions:
					array = np.asarray((array)*(hubble_param**h_scale_exponent)*(expansion_factor**a_exponent)*(cgs_conversion)) # converts to mks units
				if iteration > 0: # attaches arrays from all files together
					final_array = np.concatenate((final_array,array))
				else: 
					final_array = array
				iteration += 1

	try:
		return final_array
	except:
		raise ValueError("no files found matching keyword in read array")


def gal_centered_coords(coords,gal_coords,box_size,expansion_factor,hubble_param): # centers coordinates on galaxy
	coords = coords-gal_coords # subtracts gal center from coordinates
	# Takes into accoiunt periodicity of box. Nothing can be more than half a box away
	coords = np.where(coords > (0.5*box_size*expansion_factor)/hubble_param, coords - (0.5*box_size*expansion_factor)/hubble_param, coords)
	coords = np.where(coords < (-0.5*box_size*expansion_factor)/hubble_param, coords + (0.5*box_size*expansion_factor)/hubble_param, coords)
	return coords

def particles_within_R(object_property, gal_centered_object_property_coords,radius):
	distance = np.sqrt(gal_centered_object_property_coords[:,0]**2. + gal_centered_object_property_coords[:,1]**2. + gal_centered_object_property_coords[:,2]**2.)
	object_property = object_property[distance <= radius]
	return object_property

def particles_outside_R(object_property, gal_centered_object_property_coords,radius):
	distance = np.sqrt(gal_centered_object_property_coords[:,0]**2. + gal_centered_object_property_coords[:,1]**2. + gal_centered_object_property_coords[:,2]**2.)
	object_property = object_property[distance >= radius]
	return object_property

def particles_btwn_radii(object_property, gal_centered_object_property_coords,radius1,radius2):
        distance = np.sqrt(gal_centered_object_property_coords[:,0]**2. + gal_centered_object_property_coords[:,1]**2. + gal_centered_object_property_coords[:,2]**2.)
        temp_object_property = object_property[(distance >= radius1) & (distance < radius2)]
        if (np.size(temp_object_property) == 0):
                warnings.warn('No objects between radii')
                if object_property.ndim == 1:
                        return np.zeros(1)
                if object_property.ndim == 2:
                        return np.zeros((1,np.shape(object_property)[1]))
                if object_property.ndim > 2:
                        raise ValueError("Array passed to particles_btwn_radii is not 1 or 2 dim")
        else:
                return temp_object_property

def get_indices_of_particles_between_radii(distance, gal_centered_coords):
	return np.argwhere(((distance >= radius1) & (distance < radius2)))[:,0]


def cartesian_to_spherical_coordinate(cartesian_property):
	r = np.sqrt(cartesian_property[:,0]**2.+cartesian_property[:,1]**2.+cartesian_property[:,2]**2.)
	theta = np.arccos(cartesian_property[:,2]/r) # angle from z axis
	phi = np.arctan2(cartesian_property[:,1],cartesian_property[:,0]) # angle from x axis

	r.shape = (np.size(r),1)
	theta.shape= (np.size(theta),1)
	phi.shape = (np.size(phi),1)
	return np.concatenate([r,theta,phi],axis=1)

def cartesian_to_spherical_velocity(cartesian_velocity,cartesian_coordinate):
	r = np.sqrt(cartesian_coordinate[:,0]**2. + cartesian_coordinate[:,1]**2. + cartesian_coordinate[:,2]**2.)
	r_plane = np.sqrt(cartesian_coordinate[:,0]**2. + cartesian_coordinate[:,1]**2.)
	theta = np.arccos(cartesian_coordinate[:,2]/r)

	r_dot = (cartesian_coordinate[:,0]*cartesian_velocity[:,0]+cartesian_coordinate[:,1]*cartesian_velocity[:,1]+cartesian_coordinate[:,2]*cartesian_velocity[:,2])/r
	theta_dot = (cartesian_velocity[:,0]*cartesian_coordinate[:,1]-cartesian_coordinate[:,0]*cartesian_velocity[:,1])/r_plane**2.0
	phi_dot = (cartesian_coordinate[:,2]*(cartesian_coordinate[:,0]*cartesian_velocity[:,0]+cartesian_coordinate[:,1]*cartesian_velocity[:,1])-r_plane**2.0*cartesian_velocity[:,2])/(r**2.0*r_plane)

	v_r = r_dot
	v_theta = r*theta_dot
	v_phi = r*phi_dot*np.sin(theta)

	v_r.shape = (np.size(v_r),1)
	v_theta.shape = (np.size(v_theta),1)
	v_phi.shape = (np.size(v_phi),1)
	return np.concatenate([v_r,v_theta,v_phi],axis=1)

def remove_outliers(property_array,distance_array,cut_amount): 
	# remove values below 0.5 percentile and above 99.5 percentie to look at braod properties with
	# 2d hist without having to stretch axis or use large number of bins
	distance_array = distance_array[(property_array < np.percentile(property_array,100.-cut_amount)) & (property_array > np.percentile(property_array,cut_amount))]
	property_array = property_array[(property_array < np.percentile(property_array,100.-cut_amount)) & (property_array > np.percentile(property_array,cut_amount))]
	return [property_array,distance_array]

def get_betas(gas_velocity_sph_in_R, gas_coords_in_R, DM_velocity_sph_in_R, DM_coords_in_R, radii, output_all = False): 
	### Return beta for gas and DM (optionally dispersions of gas/DM velocities in all directions)
	### beta = 1-sigma_tan/(2*sigma_r) where sigma_tan is the disperion of the theta and phi components (summed in quadrature), sigma_r = radial dispersion
	if np.size(radii) > 1:
		gas_sigma_radial = np.zeros(np.size(radii)-1)
		gas_sigma_theta = np.zeros(np.size(radii)-1)
		gas_sigma_phi = np.zeros(np.size(radii)-1)
		gas_sigma_tan = np.zeros(np.size(radii)-1)
		gas_beta = np.zeros(np.size(radii)-1)
		DM_sigma_radial = np.zeros(np.size(radii)-1)
		DM_sigma_theta = np.zeros(np.size(radii)-1)
		DM_sigma_phi = np.zeros(np.size(radii)-1)
		DM_sigma_tan = np.zeros(np.size(radii)-1)
		DM_beta = np.zeros(np.size(radii)-1)

		for i in range(0,np.size(radii)-1):
			gas_velocity_sph = particles_btwn_radii(gas_velocity_sph_in_R, gas_coords_in_R, radii[i], radii[i+1])
			DM_velocity_sph = particles_btwn_radii(DM_velocity_sph_in_R, DM_coords_in_R, radii[i], radii[i+1])
			gas_sigma_radial[i] = np.sqrt(np.mean(np.power(gas_velocity_sph[:,0],2)))
			gas_sigma_theta[i] = np.sqrt(np.mean(np.power(gas_velocity_sph[:,1],2)))
			gas_sigma_phi[i] = np.sqrt(np.mean(np.power(gas_velocity_sph[:,2],2)))
			gas_sigma_tan[i] = np.sqrt(gas_sigma_phi[i]**2.+gas_sigma_theta[i]**2.)
			gas_beta[i] = 1.0-(gas_sigma_tan[i]**2./(2.0*gas_sigma_radial[i]**2.))

			DM_sigma_radial[i] = np.sqrt(np.mean(np.power(DM_velocity_sph[:,0],2)))
			DM_sigma_theta[i] = np.sqrt(np.mean(np.power(DM_velocity_sph[:,1],2)))
			DM_sigma_phi[i] = np.sqrt(np.mean(np.power(DM_velocity_sph[:,2],2)))
			DM_sigma_tan[i] = np.sqrt(DM_sigma_phi[i]**2.+DM_sigma_theta[i]**2.)
			DM_beta[i] = 1.0-(DM_sigma_tan[i]**2./(2.0*DM_sigma_radial[i]**2.))
	else:
		gas_velocity_sph = particles_outside_R(gas_velocity_sph_in_R, gas_coords_in_R, inner_radius)
		DM_velocity_sph = particles_outside_R(DM_velocity_sph_in_R, DM_coords_in_R, inner_radius)
		
		gas_sigma_radial = np.sqrt(np.mean(np.power(gas_velocity_sph[:,0],2)))
		gas_sigma_theta = np.sqrt(np.mean(np.power(gas_velocity_sph[:,1],2)))
		gas_sigma_phi = np.sqrt(np.mean(np.power(gas_velocity_sph[:,2],2)))
		gas_sigma_tan = np.sqrt(gas_sigma_phi**2.+gas_sigma_theta**2.)
		gas_beta = 1.0-(gas_sigma_tan**2./(2.0*gas_sigma_radial**2.))

		DM_sigma_radial = np.sqrt(np.mean(np.power(DM_velocity_sph[:,0],2)))
		DM_sigma_theta = np.sqrt(np.mean(np.power(DM_velocity_sph[:,1],2)))
		DM_sigma_phi = np.sqrt(np.mean(np.power(DM_velocity_sph[:,2],2)))
		DM_sigma_tan = np.sqrt(DM_sigma_phi**2.+DM_sigma_theta**2.)
		DM_beta = 1.0-(DM_sigma_tan**2./(2.0*DM_sigma_radial**2.))

	if (output_all == True):
		return gas_sigma_radial, gas_sigma_theta, gas_sigma_phi, gas_sigma_tan, gas_beta, \
		       DM_sigma_radial, DM_sigma_theta, DM_sigma_phi, DM_sigma_tan, DM_beta
	else:
		return gas_beta, DM_beta

def get_virial_ratios_at_radii(radii, gas_mass_in_R, DM_mass_in_R, star_mass_in_R, gas_speed_in_R, DM_speed_in_R, star_speed_in_R, gas_coords_in_R, 
							   DM_coords_in_R, star_coords_in_R, gas_T_in_R, mu_in_R):
	if np.size(radii) > 1:
		mass_within_radii = np.zeros(np.size(radii)-1)
		for i in range(0,np.size(radii)-1):
			gas_within_radii = np.sum(particles_within_R(gas_mass_in_R,gas_coords_in_R,radii[i+1]))
			DM_within_radii = np.sum(particles_within_R(DM_mass_in_R,DM_coords_in_R,radii[i+1]))
			star_within_radii = np.sum(particles_within_R(star_mass_in_R,star_coords_in_R,radii[i+1]))
			mass_within_radii[i] = gas_within_radii+DM_within_radii+star_within_radii

		gas_KE_in_radii = np.zeros(np.size(radii)-1) # really will be average KE per particle, per unit mass (used for virial)
		DM_KE_in_radii = np.zeros(np.size(radii)-1)
		star_KE_in_radii = np.zeros(np.size(radii)-1)
		gas_T_KE = np.zeros(np.size(radii)-1)

		for i in range(0,np.size(gas_KE_in_radii)):
			gas_KE_in_radii[i] = 0.5*np.mean(np.power(particles_btwn_radii(gas_speed_in_R,gas_coords_in_R,radii[i], radii[i+1]),2))
			### Note: below KE is per unit mass so the units are the same as potential and bulk KE. 
			gas_T_KE[i] = 1.5*kB*np.mean(particles_btwn_radii(gas_T_in_R, gas_coords_in_R, radii[i],radii[i+1]))*(1.0/np.mean(particles_btwn_radii(mu_in_R,gas_coords_in_R,radii[i],radii[i+1])))
			DM_KE_in_radii[i] = 0.5*np.mean(np.power(particles_btwn_radii(DM_speed_in_R,DM_coords_in_R,radii[i],radii[i+1]),2))
			star_KE_in_radii[i] = 0.5*np.mean(np.power(particles_btwn_radii(star_speed_in_R,star_coords_in_R,radii[i],radii[i+1]),2))

		gas_virial_LHS = (3.0*G*mass_within_radii)/(5.0*radii[1::]) # gravitational force an sph particle would feel at that ring per unit mass
		gas_virial_RHS = gas_KE_in_radii + gas_T_KE # average KE of an sph particle in that ring per unit mass
		gas_virial_ratio = 2.0*gas_virial_RHS/gas_virial_LHS

		DM_virial_LHS = (3.0*G*mass_within_radii)/(5.0*radii[1::])
		DM_virial_RHS = DM_KE_in_radii
		DM_virial_ratio = 2.0*DM_virial_RHS/DM_virial_LHS

		star_virial_LHS = (3.0*G*mass_within_radii)/(5.0*radii[1::])
		star_virial_RHS = star_KE_in_radii
		star_virial_ratio = 2.0*star_virial_RHS/star_virial_LHS

	else: 
		gas_within_radii = np.sum(particles_within_R(gas_mass_in_R,gas_coords_in_R,radius))
		DM_within_radii = np.sum(particles_within_R(DM_mass_in_R,DM_coords_in_R,radius))
		mass_within_radii = gas_within_radii+DM_within_radii

		gas_KE_in_radii = 0.5*np.mean(np.power(particles_within_R(gas_speed_in_R,gas_coords_in_R,radius),2))
		### Note: below KE is per unit mass so the units are the same as potential and bulk KE. 
		gas_T_KE = 1.5*kB*np.mean(particles_within_R(gas_T_in_R, gas_coords_in_R, radius))*(1.0/np.mean(mu_in_R))
		DM_KE_in_radii = 0.5*np.mean(np.power(particles_within_R(DM_speed_in_R,DM_coords_in_R,radius),2))

		gas_virial_LHS = (3.0*G*mass_within_radii)/(5.0*radius) # gravitational force an sph particle would feel at that ring per unit mass
		gas_virial_RHS = gas_KE_in_radii + gas_T_KE # average KE of an sph particle in that ring per unit mass
		gas_virial_ratio = 2.0*gas_virial_RHS/gas_virial_LHS

		DM_virial_LHS = (3.0*G*mass_within_radii)/(2.0*radius)
		DM_virial_RHS = DM_KE_in_radii
		DM_virial_ratio = 2.0*DM_virial_RHS/DM_virial_LHS

	return gas_virial_LHS, gas_KE_in_radii, gas_T_KE, gas_virial_ratio, DM_virial_LHS, DM_KE_in_radii, DM_virial_ratio

def get_basic_props(snap_directory, R_in_vir, group_number, particles_included_keyword, group_included_keyword,subfind_included_keyword):
	# create array of files
	snap_files = get_snap_files(snap_directory, particles_included_keyword)

	# pull out simulation parameters that are constant for all snapshots
	box_size = read_attribute(snap_files,'Header','BoxSize',include_file_keyword=particles_included_keyword)
	expansion_factor = read_attribute(snap_files,'Header','ExpansionFactor',include_file_keyword=particles_included_keyword)
	hubble_param = read_attribute(snap_files,'Header','HubbleParam',include_file_keyword=particles_included_keyword)

	# pull out properties of central galaxy
	# Ben's code I'm trying
	arr_start = time.time()
	gal_M200s = read_array(snap_files, "Subhalo/Mass", include_file_keyword=subfind_included_keyword)
	gal_R200s = read_array(snap_files,"FOF/Group_R_Crit200",include_file_keyword=subfind_included_keyword)
	gal_coords = read_array(snap_files, "Subhalo/CentreOfPotential", include_file_keyword=subfind_included_keyword)
	gal_velocities = read_array(snap_files, "Subhalo/Velocity", include_file_keyword = subfind_included_keyword)
	GrpIDs = read_array(snap_files,"Subhalo/GroupNumber", include_file_keyword=subfind_included_keyword)
	SubIDs = read_array(snap_files, "Subhalo/SubGroupNumber", include_file_keyword=subfind_included_keyword)
	first_subhalo_IDs = read_array(snap_files,"FOF/FirstSubhaloID", include_file_keyword = subfind_included_keyword)
	# below: ben has a [:,4] after, look in to that...
	M_stellar_30kpc = read_array(snap_files, "Subhalo/ApertureMeasurements/Mass/030kpc", include_file_keyword=subfind_included_keyword,column=4)
	SFR_30kpc = read_array(snap_files, "Subhalo/ApertureMeasurements/SFR/030kpc", include_file_keyword=subfind_included_keyword)

	# can have multiple subhalos with same ID (if their virial radii overlap, we want the most massive one)
	index = np.where((GrpIDs == float(group_number)) & (SubIDs == 0))[0] # picks out most massive galaxy in the designated group
	first_subhalo_ID = np.size(GrpIDs[np.where(GrpIDs < group_number)])

	# get the properties of our galaxy
	gal_M200 = gal_M200s[index]
	gal_R200 = gal_R200s[np.where(first_subhalo_IDs == first_subhalo_ID)[0]]
	gal_coords = gal_coords[index][0]
	gal_velocity = np.reshape(gal_velocities[index,:], 3)
	gal_stellar_mass = M_stellar_30kpc[index][0]
	gal_SFR = SFR_30kpc[index][0]
	gal_speed = np.sqrt(np.sum(np.power(gal_velocity,2)))
	gal_speed = np.sqrt(gal_velocity[0]**2.0 + gal_velocity[1]**2.0)

	radius = R_in_vir*gal_R200

	return box_size, expansion_factor, hubble_param, gal_coords, gal_velocity, gal_M200, gal_R200, radius, gal_speed, gal_stellar_mass, gal_SFR

def get_gas_props(snap_directory, radius, group_number, particles_included_keyword, group_included_keyword,subfind_included_keyword,
	              box_size, expansion_factor, hubble_param, gal_coords, gal_velocity):
	# create array of files
	snap_files = get_snap_files(snap_directory, particles_included_keyword)

	# get necessary gas properties
	gas_coords = read_array(snap_files,'PartType0/Coordinates',include_file_keyword=particles_included_keyword)
	gas_coords = gal_centered_coords(gas_coords,gal_coords,box_size,expansion_factor,hubble_param)
	gas_distance = np.sqrt(gas_coords[:,0]**2.0+gas_coords[:,1]**2.0+gas_coords[:,2]**2.0)

	gas_mass = read_array(snap_files,'PartType0/Mass',include_file_keyword=particles_included_keyword)
	gas_density = read_array(snap_files,'/PartType0/Density',include_file_keyword=particles_included_keyword)

	gas_velocity = read_array(snap_files,"/PartType0/Velocity",include_file_keyword=particles_included_keyword)
	gas_velocity = gas_velocity-gal_velocity
	gas_speed = np.sqrt(gas_velocity[:,0]**2.0+gas_velocity[:,1]**2.0+gas_velocity[:,2]**2.0)

	gas_T = read_array(snap_files,'PartType0/Temperature',include_file_keyword=particles_included_keyword)

	# H_mass_frac = read_array(snap_files,'PartType0/ElementAbundance/Hydrogen',include_file_keyword=particles_included_keyword)
	# He_mass_frac = read_array(snap_files,'PartType0/ElementAbundance/Helium',include_file_keyword=particles_included_keyword)

	# C_mass_frac = read_array(snap_files,'PartType0/ElementAbundance/Carbon',include_file_keyword=particles_included_keyword)
	# Fe_mass_frac = read_array(snap_files,'PartType0/ElementAbundance/Iron',include_file_keyword=particles_included_keyword)
	# Mg_mass_frac = read_array(snap_files,'PartType0/ElementAbundance/Magnesium',include_file_keyword=particles_included_keyword)
	# Ne_mass_frac = read_array(snap_files,'PartType0/ElementAbundance/Neon',include_file_keyword=particles_included_keyword)
	# N_mass_frac = read_array(snap_files,'PartType0/ElementAbundance/Nitrogen',include_file_keyword=particles_included_keyword)
	# O_mass_frac = read_array(snap_files,'PartType0/ElementAbundance/Oxygen',include_file_keyword=particles_included_keyword)
	# Si_mass_frac = read_array(snap_files,'PartType0/ElementAbundance/Silicon',include_file_keyword=particles_included_keyword)

	# chem_abund_start = time.time()
	# e_mass_frac = read_array(snap_files,'PartType0/ChemistryAbundances',include_file_keyword=particles_included_keyword,column=0)
	# print 'reading chem abund array takes'
	# print time.time() -chem_abund_start

	# mass_frac_array = np.asarray([H_mass_frac,He_mass_frac,C_mass_frac,Fe_mass_frac,Mg_mass_frac,Ne_mass_frac,N_mass_frac,O_mass_frac,Si_mass_frac,e_mass_frac])

	# mu = np.dot(atom_mass_array[None,:],mass_frac_array).ravel()/np.sum(mass_frac_array,axis=0)*atomic_mass_unit
	# print np.shape(mu)
	# print np.shape(H_mass_frac)

	# X = H_mass_frac*H_atom_mass
	# Y = He_mass_frac*He_atom_mass
	# Z = np.sum([C_mass_frac*C_atom_mass, Fe_mass_frac*Fe_atom_mass, Mg_mass_frac*Mg_atom_mass, Ne_mass_frac*Ne_atom_mass, N_mass_frac*N_atom_mass,
	# 			O_mass_frac*O_atom_mass, Si_mass_frac*Si_atom_mass])
	# mu = (2.0*X+0.75*Y+0.5*Z)**(-1.0)*atomic_mass_unit

	mu = 0.69*atomic_mass_unit + np.zeros(np.size(gas_T))

	gas_volume = gas_mass/gas_density
	gas_number_of_particles = gas_mass/mu
	gas_pressure = (gas_number_of_particles*kB*gas_T)/(gas_volume)

	# get properties of gas particles that fall within given radius of central galaxy
	gas_coords_in_R = particles_within_R(gas_coords, gas_coords, radius)
	gas_distance_in_R = particles_within_R(gas_distance,gas_coords,radius)
	gas_mass_in_R = particles_within_R(gas_mass, gas_coords, radius)
	gas_density_in_R = particles_within_R(gas_density, gas_coords, radius)
	gas_velocity_in_R = particles_within_R(gas_velocity, gas_coords, radius)
	gas_speed_in_R = particles_within_R(gas_speed,gas_coords,radius)
	gas_T_in_R = particles_within_R(gas_T, gas_coords, radius)
	mu_in_R = particles_within_R(mu, gas_coords, radius)
	gas_volume_in_R = particles_within_R(gas_volume, gas_coords, radius)
	gas_number_of_particles_in_R = particles_within_R(gas_number_of_particles, gas_coords, radius)
	gas_pressure_in_R = particles_within_R(gas_pressure, gas_coords, radius)

	return gas_coords_in_R, gas_distance_in_R, gas_mass_in_R, mu_in_R, gas_density_in_R, gas_velocity_in_R, gas_speed_in_R, \
	       gas_T_in_R, gas_volume_in_R, gas_number_of_particles_in_R, gas_pressure_in_R

def get_gas_props_for_spectra_runs(snap_directory, radius, group_number, particles_included_keyword, group_included_keyword,subfind_included_keyword,
	              box_size, expansion_factor, hubble_param, gal_coords, gal_velocity):
	# create array of files
	snap_files = get_snap_files(snap_directory, particles_included_keyword)

	# get necessary gas properties
	gas_coords = read_array(snap_files,'PartType0/Coordinates',include_file_keyword=particles_included_keyword)
	gas_coords = gal_centered_coords(gas_coords,gal_coords,box_size,expansion_factor,hubble_param)
	gas_distance = np.sqrt(gas_coords[:,0]**2.0+gas_coords[:,1]**2.0+gas_coords[:,2]**2.0)

	gas_mass = read_array(snap_files,'PartType0/Mass',include_file_keyword=particles_included_keyword)

	gas_velocity = read_array(snap_files,"/PartType0/Velocity",include_file_keyword=particles_included_keyword)
	gas_velocity = gas_velocity-gal_velocity
	gas_speed = np.sqrt(gas_velocity[:,0]**2.0+gas_velocity[:,1]**2.0+gas_velocity[:,2]**2.0)

	# get properties of gas particles that fall within given radius of central galaxy
	gas_coords_in_R = particles_within_R(gas_coords, gas_coords, radius)
	gas_distance_in_R = particles_within_R(gas_distance,gas_coords,radius)
	gas_mass_in_R = particles_within_R(gas_mass, gas_coords, radius)
	gas_velocity_in_R = particles_within_R(gas_velocity, gas_coords, radius)
	gas_speed_in_R = particles_within_R(gas_speed,gas_coords,radius)

	return gas_coords_in_R, gas_distance_in_R, gas_mass_in_R, gas_velocity_in_R, gas_speed_in_R

def get_DM_props(snap_directory, radius, group_number, particles_included_keyword, group_included_keyword,subfind_included_keyword,
	              box_size, expansion_factor, hubble_param, gal_coords, gal_velocity):

	# create array of files
	snap_files = get_snap_files(snap_directory, particles_included_keyword)

	# get necessary DM properties
	DM_coords = read_array(snap_files,'PartType1/Coordinates',include_file_keyword=particles_included_keyword)
	DM_coords = gal_centered_coords(DM_coords,gal_coords,box_size,expansion_factor,hubble_param)
	DM_distance = np.sqrt(DM_coords[:,0]**2.0+DM_coords[:,1]**2.0+DM_coords[:,2]**2.0)

	DM_mass = read_attribute(snap_files,'Header','MassTable',include_file_keyword=particles_included_keyword)[1]*(1.e10/hubble_param)*M_sol
	DM_mass = np.zeros(np.size(DM_coords))+DM_mass

	DM_velocity = read_array(snap_files,'PartType1/Velocity',include_file_keyword=particles_included_keyword)
	DM_velocity = DM_velocity-gal_velocity
	DM_speed = np.sqrt(DM_velocity[:,0]**2.0+DM_velocity[:,1]**2.0+DM_velocity[:,2]**2.0)

	# get properties of DM particles that fall within given radius of central galaxy
	DM_coords_in_R = particles_within_R(DM_coords, DM_coords, radius)
	DM_distance_in_R = particles_within_R(DM_distance,DM_coords,radius)
	DM_mass_in_R = particles_within_R(DM_mass, DM_coords, radius)
	DM_velocity_in_R = particles_within_R(DM_velocity, DM_coords, radius)
	DM_speed_in_R = particles_within_R(DM_speed,DM_coords,radius)

	return DM_coords_in_R, DM_distance_in_R, DM_mass_in_R, DM_velocity_in_R, DM_speed_in_R

def get_star_props(snap_directory, radius, group_number, particles_included_keyword, group_included_keyword,subfind_included_keyword,
	              box_size, expansion_factor, hubble_param, gal_coords, gal_velocity):

	# create array of files
	snap_files = get_snap_files(snap_directory, particles_included_keyword)

	# get necessary DM properties
	star_coords = read_array(snap_files,'PartType4/Coordinates',include_file_keyword=particles_included_keyword)
	star_coords = gal_centered_coords(star_coords,gal_coords,box_size,expansion_factor,hubble_param)
	star_distance = np.sqrt(star_coords[:,0]**2.0+star_coords[:,1]**2.0+star_coords[:,2]**2.0)

	star_mass = read_array(snap_files, 'PartType4/Mass', include_file_keyword=particles_included_keyword)

	star_velocity = read_array(snap_files,'PartType4/Velocity',include_file_keyword=particles_included_keyword)
	star_velocity = star_velocity-gal_velocity
	star_speed = np.sqrt(star_velocity[:,0]**2.0+star_velocity[:,1]**2.0+star_velocity[:,2]**2.0)

	# get properties of DM particles that fall within given radius of central galaxy
	star_coords_in_R = particles_within_R(star_coords, star_coords, radius)
	star_distance_in_R = particles_within_R(star_distance,star_coords,radius)
	star_mass_in_R = particles_within_R(star_mass, star_coords, radius)
	star_velocity_in_R = particles_within_R(star_velocity, star_coords, radius)
	star_speed_in_R = particles_within_R(star_speed,star_coords,radius)

	return star_coords_in_R, star_distance_in_R, star_mass_in_R, star_velocity_in_R, star_speed_in_R

def get_props_for_coldens(species, snap_directory, group_number, particles_included_keyword, group_included_keyword, subfind_included_keyword, groups_dir=None, all_directories = False):
	# create array of files
	print "in get props"
	snap_files = get_snap_files(snap_directory, particles_included_keyword, all_directories=all_directories)
	# because particle included keyword is different in rotated snapshots have to grab the groups_ files seperately
	if groups_dir == None:
		snap_files = np.concatenate((snap_files, get_snap_files(snap_directory, group_included_keyword)))
	else:
		snap_files = np.concatenate((snap_files, get_snap_files(groups_dir, group_included_keyword)))
	print snap_files

	p = pool(12)
	print "pool opened"

	gas_coords = p.apply_async(read_array, [snap_files, 'PartType0/Coordinates'], {'include_file_keyword':particles_included_keyword})
	# gas_coords_x_result = p.apply_async(read_array, [snap_files, 'PartType0/Coordinates'], {'include_file_keyword':particles_included_keyword, 'column':0})
	# gas_coords_y_result = p.apply_async(read_array, [snap_files, 'PartType0/Coordinates'], {'include_file_keyword':particles_included_keyword, 'column':1})
	# gas_coords_z_result = p.apply_async(read_array, [snap_files, 'PartType0/Coordinates'], {'include_file_keyword':particles_included_keyword, 'column':2})
	print "gas coords initialized"
	smoothing_length_result = p.apply_async(read_array, [snap_files, 'PartType0/SmoothingLength'], {'include_file_keyword':particles_included_keyword})
	particle_mass_result = p.apply_async(read_array, [snap_files, 'PartType0/Mass'], {'include_file_keyword':particles_included_keyword})
	density_result = p.apply_async(read_array, [snap_files, 'PartType0/Density'], {'include_file_keyword':particles_included_keyword})
	GrpIDs_result = p.apply_async(read_array, [snap_files, "Subhalo/GroupNumber"], {'include_file_keyword':subfind_included_keyword})
	SubIDs_result = p.apply_async(read_array, [snap_files, "Subhalo/SubGroupNumber"], {'include_file_keyword':subfind_included_keyword})
	gal_coords_result = p.apply_async(read_array, [snap_files, "Subhalo/CentreOfPotential"], {'include_file_keyword':subfind_included_keyword}) # map center
	print "10 initiated"
	if species.lower() == 'h1':
		element_mass_frac_result = p.apply_async(read_array, [snap_files, 'PartType0/ElementAbundance/Hydrogen'], {'include_file_keyword':particles_included_keyword})
		ion_fracs_result = p.apply_async(read_array, [snap_files, 'PartType0/ChemicalAbundances/HydrogenI'], {'include_file_keyword':particles_included_keyword, 'get_scaling_conversions':False})
	else:
		raise ValueError('We are only dealing with HI right now, species should be h1')

	p.close()
	print "pool closed"
	gas_coords = gas_coords.get()
	# gas_coords_x = gas_coords_x_result.get()
	# gas_coords_y = gas_coords_y_result.get()
	# gas_coords_z = gas_coords_z_result.get()
	# gas_coords = np.column_stack((gas_coords_x, gas_coords_y, gas_coords_z))
	print "got gas coords"

	smoothing_length = smoothing_length_result.get()
	particle_mass = particle_mass_result.get()
	element_mass_frac = element_mass_frac_result.get()
	# ion_fracs = ion_fracs_result.get() # really the number density of the ion relative to nH fixed in if statement

	if species.lower() == 'h1':
		density = density_result.get()
		nH = density*element_mass_frac/atomic_mass_unit # number density of all hydrogen (HI and HII)
		# ion_fracs = (ion_fracs*nH*atomic_mass_unit)/(element_mass_frac*density) # now it's the fraction of the element that is the ion (N_ion/N_element)
		# ion_fracs[np.where(element_mass_frac == 0.0)] = 0 # makes sure the ion mass is zero wherever there is none of the element. Otherwise get div by zero
	else:
		raise ValueError('We are only dealing with HI right now, species should be h1')

	element_mass = particle_mass*element_mass_frac
	# ion_mass = particle_mass*element_mass_frac*ion_fracs
	GrpIDs = GrpIDs_result.get()
	SubIDs = SubIDs_result.get()
	index = np.where((GrpIDs == float(group_number)) & (SubIDs == 0))[0] # picks out most massive galaxy in the designated group
	if np.size(index) > 1:
		index = index[0]
	gal_coords = gal_coords_result.get()[index] # map center
	print gal_coords
	print ''

	box_size = read_attribute(snap_files,'Header','BoxSize',include_file_keyword=particles_included_keyword)

	# TODO move distances to Mpcs and masses to solar masses. Also check that all outputs are there.
	gas_coords/=(parsec_in_cm*1.e6) # Mpcs!!!
	smoothing_length/=(parsec_in_cm*1.e6) # Mpcs!!!
	gal_coords/=(parsec_in_cm*1.e6) # Mpcs!!!
	box_size/=1. # box_size is an attribute and so it is already in Mpcs!
	element_mass/=(M_sol) # solar masses
	# ion_mass/=(M_sol) # solar masses


	return gas_coords, smoothing_length, element_mass, None, gal_coords, box_size # None should be ion mass




def plot_velocity_dispersions(gas_speed_in_R, gas_distance_in_R, DM_speed_in_R, DM_distance_in_R, radius, gal_R200, group_number):
	# set plotting limits so they're the same for both plots (for comparison purposes)
	ymax = np.max([np.max(DM_speed_in_R),np.max(gas_speed_in_R)])
	ymin = np.min([np.min(DM_speed_in_R),np.min(gas_speed_in_R)])

	xmax = np.max([np.max(DM_distance_in_R),np.max(gas_distance_in_R)])/(1000.*parsec_in_cm)
	xmin = np.min([np.min(DM_distance_in_R),np.min(gas_distance_in_R)])/(1000.*parsec_in_cm)

	# make DM dispersion plot
	plt.hist2d(DM_distance_in_R/(1000.*parsec_in_cm),DM_speed_in_R,bins=125, norm = matplotlib.colors.LogNorm()) # dark matter
	plt.colorbar()
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("DM Vel Dispersion within %.1e R_vir (%.0f)" % ((radius/gal_R200), group_number))
	plt.xlabel("Distance from Galaxy Center (kpc)")
	plt.ylabel("Speed (cm/s)")
	title = "DM Vel Dispersion %.1e virial radii (%.0f).png" % (radius/gal_R200, group_number)
	plt.savefig(title, format='png')
	plt.close()

	# make gas dispersion plot
	plt.hist2d(gas_distance_in_R/(1000.*parsec_in_cm),gas_speed_in_R,bins=125, norm = matplotlib.colors.LogNorm()) # dark matter
	plt.colorbar()
	plt.ylim ([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("Gas Vel Dispersion within %.1e R_vir (%.0f)" % ((radius/gal_R200), group_number))
	plt.xlabel("Distance from Galaxy Center (kpc)")
	plt.ylabel("Speed (cm/s)")
	title = "Gas Velocity Dispersion %.1e virial radii (%.0f).png" % (radius/gal_R200, group_number)
	plt.savefig(title, format='png')
	plt.close()

def spherical_velocity_plots(gas_velocity_sph_in_R, gas_distance_in_R, DM_velocity_sph_in_R, DM_distance_in_R, radius, gal_R200, 
							 group_number, percentile_cut=0.0,):
	# pull out all components and how many particles fall within percentile cuts (if any is made)
	gas_velocity_radial,gas_distance_radial = remove_outliers(gas_velocity_sph_in_R[:,0],gas_distance_in_R,percentile_cut)
	gas_velocity_theta,gas_distance_theta = remove_outliers(gas_velocity_sph_in_R[:,1],gas_distance_in_R,percentile_cut)
	gas_velocity_phi,gas_distance_phi = remove_outliers(gas_velocity_sph_in_R[:,2],gas_distance_in_R,percentile_cut)

	DM_velocity_radial,DM_distance_radial = remove_outliers(DM_velocity_sph_in_R[:,0],DM_distance_in_R,percentile_cut)
	DM_velocity_theta,DM_distance_theta = remove_outliers(DM_velocity_sph_in_R[:,1],DM_distance_in_R,percentile_cut)
	DM_velocity_phi,DM_distance_phi = remove_outliers(DM_velocity_sph_in_R[:,2],DM_distance_in_R,percentile_cut)

	# set limits for plots so they are all the same (for comparison)
	ymax = np.max([np.max(gas_velocity_radial),np.max(gas_velocity_theta),np.max(gas_velocity_phi),
	np.max(DM_velocity_radial),np.max(DM_velocity_theta),np.max(DM_velocity_phi)])

	ymin = np.min([np.min(gas_velocity_radial),np.min(gas_velocity_theta),np.min(gas_velocity_phi),
	np.min(DM_velocity_radial),np.min(DM_velocity_theta),np.min(DM_velocity_phi)])

	xmax = np.max([np.max(gas_distance_in_R),np.max(DM_distance_in_R)])
	xmin = np.min([np.min(gas_distance_in_R),np.min(DM_distance_in_R)])

	# Gas Plots
	plt.hist2d(gas_distance_radial,gas_velocity_radial,bins=125, norm = matplotlib.colors.LogNorm())
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("Gas v_rad to %.1e kpc (%.0f)" % (radius/(1000.*parsec_in_cm), group_number))
	plt.xlabel("Distance from Center of Potential (cm)")
	plt.ylabel("Speed (cm/s)")
	title = "Gas Radial Velocity within %.1e virial radii (%.0f).png" % (radius/gal_R200, group_number)
	plt.savefig(title, format='png')
	plt.close()

	plt.hist2d(gas_distance_theta,gas_velocity_theta,bins=125, norm = matplotlib.colors.LogNorm())
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("Gas v_theta to %.1e kpc (%.0f)" % (radius/(1000.*parsec_in_cm), group_number))
	plt.xlabel("Distance from Center of Potential (cm)")
	plt.ylabel("Speed (cm/s)")
	title = "Gas Theta Velocity within %.1e virial radii (%.0f).png" % (radius/gal_R200, group_number)
	plt.savefig(title, format='png')
	plt.close()

	plt.hist2d(gas_distance_phi,gas_velocity_phi,bins=125, norm = matplotlib.colors.LogNorm())
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("Gas v_phi to %.1e kpc (%.0f)" % (radius/(1000.*parsec_in_cm), group_number))
	plt.xlabel("Distance from Center of Potential (cm)")
	plt.ylabel("Speed (cm/s)")
	title = "Gas Phi Velocity within %.1e virial radii (%.0f).png" % (radius/gal_R200, group_number)
	plt.savefig(title, format='png')
	plt.close()

	# DM Plots
	plt.hist2d(DM_distance_radial,DM_velocity_radial,bins=125, norm = matplotlib.colors.LogNorm())
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("DM v_rad to %.1e kpc (%.0f)" % (radius/(1000.*parsec_in_cm), group_number))
	plt.xlabel("Distance from Center of Potential (cm)")
	plt.ylabel("Speed (cm/s)")
	title = "DM Radial Velocity within %.1e virial radii (%.0f).png" % (radius/gal_R200, group_number)
	plt.savefig(title, format='png')
	plt.close()

	plt.hist2d(DM_distance_theta,DM_velocity_theta,bins=125, norm = matplotlib.colors.LogNorm())
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("DM v_theta to %.1e kpc (%.0f)" % (radius/(1000.*parsec_in_cm), group_number))
	plt.xlabel("Distance from Center of Potential (cm)")
	plt.ylabel("Speed (cm/s)")
	title = "DM Theta Velocity within %.1e virial radii (%.0f).png" % (radius/gal_R200, group_number)
	plt.savefig(title, format='png')
	plt.close()

	plt.hist2d(DM_distance_phi,DM_velocity_phi,bins=125, norm = matplotlib.colors.LogNorm())
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("DM v_phi to %.1e kpc (%.0f)" % (radius/(1000.*parsec_in_cm), group_number))
	plt.xlabel("Distance from Center of Potential (cm)")
	plt.ylabel("Speed (cm/s)")
	title = "DM Phi Velocity within %.1e virial radii (%.0f).png" % (radius/gal_R200, group_number)
	plt.savefig(title, format='png')
	plt.close()

def force_balance_plots(gas_mass_in_R, gas_coords_in_R, gas_speed_in_R, gas_T_in_R, mu_in_R, DM_mass_in_R, DM_coords_in_R,
						DM_speed_in_R, radius, gal_R200, group_number):
	### Find mass contained within certain radii

	radii_pts = 1.e2 # number of pts at which forces are evaluated
	radii = np.linspace(0.0,radius,radii_pts)
	mass_within_radii = np.zeros(np.size(radii)-1)

	for i in range(0,np.size(radii)-1):
		gas_within_radii = np.sum(particles_within_R(gas_mass_in_R,gas_coords_in_R,radii[i+1]))
		DM_within_radii = np.sum(particles_within_R(DM_mass_in_R,DM_coords_in_R,radii[i+1]))
		mass_within_radii[i] = gas_within_radii+DM_within_radii

	### Virial terms

	gas_KE_in_radii = np.zeros(np.size(radii)-1) # really will be average KE per particle, per unit mass (used for virial)
	DM_KE_in_radii = np.zeros(np.size(radii)-1)
	gas_T_KE = np.zeros(np.size(radii)-1)

	for i in range(0,np.size(gas_KE_in_radii)):
		gas_KE_in_radii[i] = 0.5*np.mean(np.power(particles_btwn_radii(gas_speed_in_R,gas_coords_in_R,radii[i],radii[i+1]),2))
		### Note: below KE is per unit mass so the units are the same as potential and bulk KE. 
		gas_T_KE[i] = 1.5*kB*np.mean(particles_btwn_radii(gas_T_in_R, gas_coords_in_R, radii[i], radii[i+1]))*(1.0/np.mean(mu_in_R))
		DM_KE_in_radii[i] = 0.5*np.mean(np.power(particles_btwn_radii(DM_speed_in_R,DM_coords_in_R,radii[i],radii[i+1]),2))

	gas_LHS = (3.0*G*mass_within_radii)/(5.0*radii[1::]) # gravitational force an sph particle would feel at that ring per unit mass
	gas_virial_RHS = gas_KE_in_radii + gas_T_KE # average KE of an sph particle in that ring per unit mass
	gas_virial_ratio = 2.0*gas_virial_RHS/gas_virial_LHS

	DM_virial_LHS = (3.0*G*mass_within_radii)/(2.0*radii[1::])
	DM_virial_RHS = DM_KE_in_radii
	DM_virial_ratio = 2.0*DM_virial_RHS/DM_virial_LHS

	# ratio of virial terms (gas and DM)
	plt.semilogy(radii[1::],gas_virial_ratio,'b.', label = 'Gas')
	plt.semilogy(radii[1::],DM_virial_ratio, 'r.',label = 'DM')
	plt.title('2 KE/ PE (virial) in %.1e R_vir (%.0f)' % ((radius/gal_R200),group_number))
	plt.xlabel('Distance from Center (cm)')
	plt.ylabel('Ratio (grav/kinetic)')
	plt.legend()
	title = "Virial Ratio (2 KE over PE) within %.1e R_vir (%.0f).png" % (radius/gal_R200, group_number)
	plt.savefig(title, format = 'png')
	plt.close()

	# compare sources of gravitational and kinetic energy
	plt.semilogy(radii[1::], gas_virial_LHS, '.', label = "Grav")
	plt.semilogy(radii[1::], gas_KE_in_radii, '.', label = "KE of SPH particles")
	plt.semilogy(radii[1::], gas_T_KE, '.', label = "KE from Temp of particles")
	plt.title("Gas Virial Sources Comparison")
	plt.legend(bbox_to_anchor=(1,0.25))
	title = "Gas Virial Sources Comparison to %.2e virial radii(%.0f).png" % (radius/gal_R200, group_number)
	plt.savefig(title, format = 'png')
	plt.close()

	plt.semilogy(radii[1::], DM_virial_LHS, '.', label = "Grav")
	plt.semilogy(radii[1::], DM_KE_in_radii, '.', label = "KE of SPH particles")
	plt.title("DM Virial Sources Comparison")
	plt.legend()
	title = "DM Virial Sources Comparison to %.2e virial radii(%.0f).png" % (radius/gal_R200, group_number)
	plt.savefig(title, format = 'png')
	plt.close()

def create_gal_data(file_name, arr_size, snap_directory, file_keyword, group_number, radius_of_output,
				 gal_mass, gal_stellar_mass, gal_SFR, virial_radius, box_size, hubble_param, expansion_factor, gal_coords, gal_velocity, mu = None,
				 gas_sigma_radial=None, gas_sigma_theta=None, gas_sigma_phi=None, gas_beta=None, DM_sigma_radial=None, DM_sigma_theta=None,
				 DM_sigma_phi=None, DM_beta=None, gas_virial_LHS=None, gas_KE_in_radii=None, gas_T_KE=None, gas_virial_ratio=None, 
	             DM_virial_LHS=None, DM_KE_in_radii=None, DM_virial_ratio=None):

	# convert to units of hdf5 file
	radius_of_output /= 1.e3*parsec_in_cm
	gal_mass /= M_sol
	virial_radius /= 1.e3*parsec_in_cm
	gal_stellar_mass /= M_sol
	box_size *= (1.e3)/hubble_param*expansion_factor # have to adjust becuase it's drawn from header so not put in physical by CGSConversion
	gal_coords /= 1.e3*parsec_in_cm
	gal_velocity /= 1.e5
	if gas_sigma_radial != None:
		gas_sigma_radial /= 1.e5
		gas_sigma_theta /= 1.e5
		gas_sigma_phi /= 1.e5
	if DM_sigma_radial != None:
		DM_sigma_radial /= 1.e5
		DM_sigma_theta /= 1.e5
		DM_sigma_phi /= 1.e5

	with h5py.File(file_name,'w') as hf:
		GalaxyProperties = hf.create_group('GalaxyProperties')

		snap_directory = GalaxyProperties.create_dataset('snap_directory', (1,), maxshape= (None,), data = snap_directory)

		file_keyword = GalaxyProperties.create_dataset('file_keyword', (1,), maxshape= (None,), data = file_keyword)

		group_number = GalaxyProperties.create_dataset('group_number', (1,), maxshape= (None,), data = group_number)

		radius_of_output = GalaxyProperties.create_dataset('radius_of_output', (arr_size,), maxshape= (None,), data = radius_of_output)
		radius_of_output.attrs['units'] = 'kiloparsecs'

		gal_mass = GalaxyProperties.create_dataset('gal_mass', (1,), maxshape= (None,), data = gal_mass)
		gal_mass.attrs['units'] = 'solar masses'

		gal_R200 = GalaxyProperties.create_dataset('gal_R200', (1,), maxshape= (None,), data = virial_radius)
		gal_R200.attrs['units'] = 'kiloparsecs'

		gal_stellar_mass_hdf5 = GalaxyProperties.create_dataset('gal_stellar_mass', (1,), maxshape=(None,), data = gal_stellar_mass)
		gal_stellar_mass_hdf5.attrs['units'] = 'solar masses'

		log10_smass = GalaxyProperties.create_dataset('log10_smass', (1,), maxshape=(None,), data = np.log10(gal_stellar_mass))

		gal_SFR_hdf5 = GalaxyProperties.create_dataset('gal_SFR', (1,), maxshape=(None,), data = gal_SFR*(secs_per_year/M_sol))
		gal_SFR_hdf5.attrs['units'] = 'M_sol/yr'

		log10_sSFR = GalaxyProperties.create_dataset('log10_sSFR', (1,), maxshape=(None,), data = np.log10((gal_SFR*(secs_per_year/M_sol))/gal_stellar_mass))

		box_size = GalaxyProperties.create_dataset('box_size', (1,), maxshape= (None,), data = box_size)
		box_size.attrs['units'] = 'kiloparsecs'

		gal_coords = GalaxyProperties.create_dataset('gal_coords', (1,3), maxshape= (None,3), data = gal_coords)
		gal_coords.attrs['units'] = 'kiloparsecs'

		gal_velocity = GalaxyProperties.create_dataset('gal_velocity', (1,3), maxshape = (None,3), data = gal_velocity)
		gal_velocity.attrs['units'] = 'km/s'

		if mu != None:
			mu = GalaxyProperties.create_dataset('mu', (1,), maxshape = (None,), data = mu)
			mu.attrs['formula'] = '(2X+3/4*Y+0.5*Z)^(-1) (Z from C, Fe, Mg, Ne, N, O, Si)'

		if gas_sigma_radial != None:
			gas_sigma_radial = GalaxyProperties.create_dataset('gas_sigma_radial', (arr_size,), maxshape= (None,), data = gas_sigma_radial)
			gas_sigma_radial.attrs['units'] = 'km/s'

		if gas_sigma_theta != None:
			gas_sigma_theta = GalaxyProperties.create_dataset('gas_sigma_theta', (arr_size,), maxshape= (None,), data = gas_sigma_theta)
			gas_sigma_theta.attrs['units'] = 'km/s'

		if gas_sigma_phi != None:
			gas_sigma_phi = GalaxyProperties.create_dataset('gas_sigma_phi', (arr_size,), maxshape= (None,), data = gas_sigma_phi)
			gas_sigma_phi.attrs['units'] = 'km/s'

		if gas_beta != None:
			gas_beta = GalaxyProperties.create_dataset('gas_beta', (arr_size,), maxshape= (None,), data = gas_beta)
			gas_beta.attrs['units'] = 'dimensionless'
			gas_beta.attrs['formula'] = '1-(gas_sigma_tan^2)/(2*gas_sigma_radial^2)'

		if DM_sigma_radial != None:
			DM_sigma_radial = GalaxyProperties.create_dataset('DM_sigma_radial', (arr_size,), maxshape= (None,), data = DM_sigma_radial)
			DM_sigma_radial.attrs['units'] = 'km/s'

		if DM_sigma_theta != None:
			DM_sigma_theta = GalaxyProperties.create_dataset('DM_sigma_theta', (arr_size,), maxshape= (None,), data = DM_sigma_theta)
			DM_sigma_theta.attrs['units'] = 'km/s'

		if DM_sigma_phi != None:
			DM_sigma_phi = GalaxyProperties.create_dataset('DM_sigma_phi', (arr_size,), maxshape= (None,), data = DM_sigma_phi)
			DM_sigma_phi.attrs['units'] = 'km/s'

		if DM_beta != None:
			DM_beta = GalaxyProperties.create_dataset('DM_beta', (arr_size,), maxshape= (None,), data = DM_beta)
			gas_beta.attrs['units'] = 'dimensionless'
			gas_beta.attrs['formula'] = '1-(DM_sigma_tan^2)/(2*DM_sigma_radial^2)'

		if gas_virial_LHS != None:
			gas_virial_LHS = GalaxyProperties.create_dataset('gas_virial_LHS', (arr_size,), maxshape= (None,), data = gas_virial_LHS)
			gas_virial_LHS.attrs['units'] = 'J/kg'
			gas_virial_LHS.attrs['formula'] = '(3/5)GM/r (M is gas and DM mass)'

		if gas_KE_in_radii != None:
			gas_KE_in_radii = GalaxyProperties.create_dataset('gas_KE_in_radii', (arr_size,), maxshape= (None,), data = gas_KE_in_radii)
			gas_KE_in_radii.attrs['units'] = 'J/kg'
			gas_KE_in_radii.attrs['formula'] = '0.5*v_rms^2'

		if gas_T_KE != None:
			gas_T_KE = GalaxyProperties.create_dataset('gas_T_KE', (arr_size,), maxshape= (None,), data = gas_T_KE)
			gas_T_KE.attrs['units'] = 'J/kg'
			gas_T_KE.attrs['formula'] = '1.5*k_B*T_ave/mu_ave'

		if gas_virial_ratio != None:
			gas_virial_ratio = GalaxyProperties.create_dataset('gas_virial_ratio', (arr_size,), maxshape= (None,), data = gas_virial_ratio)

		if DM_virial_LHS != None:
			DM_virial_LHS = GalaxyProperties.create_dataset('DM_virial_LHS', (arr_size,), maxshape= (None,), data = DM_virial_LHS)
			DM_virial_LHS.attrs['units'] = 'J/kg'
			DM_virial_LHS.attrs['formula'] = '(3/5)GM/r (M is gas and DM mass)'

		if DM_KE_in_radii != None:
			DM_KE_in_radii = GalaxyProperties.create_dataset('DM_KE_in_radii', (arr_size,), maxshape= (None,), data = DM_KE_in_radii)
			DM_KE_in_radii.attrs['units'] = 'J/kg'
			DM_KE_in_radii.attrs['formula'] = '0.5*v_rms^2'

		if DM_virial_ratio != None:
			DM_virial_ratio = GalaxyProperties.create_dataset('DM_virial_ratio', (arr_size,), maxshape= (None,), data = DM_virial_ratio)

def add_gal_data_to_file(file_name, new_dataset_bool, data_name, data, units = None, formula = None):

	with h5py.File(file_name,'r+') as hf:
		GalaxyProperties = hf.get('GalaxyProperties')

		if new_dataset_bool:
			new_data = GalaxyProperties.create_dataset(data_name, (np.size(data),) , maxshape=(None,), data = data)
			if units != None:
				new_data.attrs['units'] = units
			if formula != None:
				new_data.attrs['formula'] = formula

		else: # Note: only use this for adding one data point to an existing array!!!
			new_data = GalaxyProperties.get(data_name)
			new_data.resize(new_data.shape[0]+1, axis=0)
			new_data[-1] = data

def hdf5_remove_last_line(file_name):
	with h5py.File(file_name,'r+') as hf:
		GalaxyProperties = hf.get('GalaxyProperties')

		temp_snap_directory = GalaxyProperties.get('snap_directory')
		temp_snap_directory.resize(temp_snap_directory.shape[0]-1, axis = 0)

		temp_file_keyword = GalaxyProperties.get('file_keyword')
		temp_file_keyword.resize(temp_file_keyword.shape[0]-1, axis = 0)

		temp_group_number = GalaxyProperties.get('group_number')
		temp_group_number.resize(temp_group_number.shape[0]-1, axis = 0)

		temp_radius_of_output = GalaxyProperties.get('radius_of_output')
		temp_radius_of_output.resize(temp_radius_of_output.shape[0]-1, axis = 0)

		temp_gal_mass = GalaxyProperties.get('gal_mass')
		temp_gal_mass.resize(temp_gal_mass.shape[0]-1, axis = 0)

		temp_gal_R200 = GalaxyProperties.get('gal_R200')
		temp_gal_R200.resize(temp_gal_R200.shape[0]-1, axis = 0)

		temp_box_size = GalaxyProperties.get('box_size')
		temp_box_size.resize(temp_box_size.shape[0]-1, axis = 0)

		temp_gal_coords = GalaxyProperties.get('gal_coords')
		temp_gal_coords.resize(temp_gal_coords.shape[0]-1, axis = 0)

		temp_gal_velocity = GalaxyProperties.get('gal_velocity')
		temp_gal_velocity.resize(temp_gal_velocity.shape[0]-1, axis = 0)

		temp_gas_sigma_radial = GalaxyProperties.get('gas_sigma_radial')
		temp_gas_sigma_radial.resize(temp_gas_sigma_radial.shape[0]-1, axis = 0)

		temp_gas_sigma_theta = GalaxyProperties.get('gas_sigma_theta')
		temp_gas_sigma_theta.resize(temp_gas_sigma_theta.shape[0]-1, axis = 0)

		temp_gas_sigma_phi = GalaxyProperties.get('gas_sigma_phi')
		temp_gas_sigma_phi.resize(temp_gas_sigma_phi.shape[0]-1, axis = 0)

		temp_gas_beta = GalaxyProperties.get('gas_beta')
		temp_gas_beta.resize(temp_gas_beta.shape[0]-1, axis = 0)

		temp_DM_sigma_radial = GalaxyProperties.get('DM_sigma_radial')
		temp_DM_sigma_radial.resize(temp_DM_sigma_radial.shape[0]-1, axis = 0)

		temp_DM_sigma_theta = GalaxyProperties.get('DM_sigma_theta')
		temp_DM_sigma_theta.resize(temp_DM_sigma_theta.shape[0]-1, axis = 0)

		temp_DM_sigma_phi = GalaxyProperties.get('DM_sigma_phi')
		temp_DM_sigma_phi.resize(temp_DM_sigma_phi.shape[0]-1, axis = 0)

		temp_DM_beta = GalaxyProperties.get('DM_beta')
		temp_DM_beta.resize(temp_DM_beta.shape[0]-1, axis = 0)

		temp_gas_virial_LHS = GalaxyProperties.get('gas_virial_LHS')
		temp_gas_virial_LHS.resize(temp_gas_virial_LHS.shape[0]-1, axis = 0)

		temp_gas_KE_in_radii = GalaxyProperties.get('gas_KE_in_radii')
		temp_gas_KE_in_radii.resize(temp_gas_KE_in_radii.shape[0]-1, axis = 0)

		temp_gas_T_KE = GalaxyProperties.get('gas_T_KE')
		temp_gas_T_KE.resize(temp_gas_T_KE.shape[0]-1, axis = 0)

		temp_gas_virial_ratio = GalaxyProperties.get('gas_virial_ratio')
		temp_gas_virial_ratio.resize(temp_gas_virial_ratio.shape[0]-1, axis = 0)

		temp_DM_virial_LHS = GalaxyProperties.get('DM_virial_LHS')
		temp_DM_virial_LHS.resize(temp_DM_virial_LHS.shape[0]-1, axis = 0)

		temp_DM_KE_in_radii = GalaxyProperties.get('DM_KE_in_radii')
		temp_DM_KE_in_radii.resize(temp_DM_KE_in_radii.shape[0]-1, axis = 0)

		temp_DM_virial_ratio = GalaxyProperties.get('DM_virial_ratio')
		temp_DM_virial_ratio.resize(temp_DM_virial_ratio.shape[0]-1, axis = 0)

def plot_param_vs_radii(file_name, param_name): # takes name of parameter in hdf5 output files and plots it vs radius
	if np.size(np.asarray(param_name)) == 1:
		with h5py.File(file_name,'r') as hf:
			GalaxyProperties = hf.get("GalaxyProperties")
			radii = np.array(GalaxyProperties.get("radius_of_output"))
			if isinstance(param_name, basestring):
				current_param = np.array(GalaxyProperties.get(param_name)) # units: kpc
				plt.plot(radii, current_param,'.', label = str(param_name))
			else:
				raise ValueError("param_name must be string (or array/list of strings)")

	else:

		with h5py.File(file_name,'r') as hf:
			GalaxyProperties = hf.get("GalaxyProperties")
			radii = np.array(GalaxyProperties.get("radius_of_output"))
			for name in param_name:
				if isinstance(name, basestring):
					current_param = np.array(GalaxyProperties.get(name)) # units: kpc
					plt.plot(radii, current_param,'.', label = str(name))
				else:
					raise ValueError("param_name must be string (or array/list of strings)")

	plt.legend()
	plt.savefig('Test HDF5 output.png')
	plt.close()

def add_dataset_to_hdf5(file_name, data_set_name, new_data, columns = 1, units = 'None'):
	with h5py.File(file_name,'r+') as hf:
		GalaxyProperties = hf.get("GalaxyProperties")
		new_data = GalaxyProperties.create_dataset(data_set_name, (np.shape(new_data)[0],columns), maxshape = (None, columns), data=new_data)
		if units != 'None':
			new_data.attrs['units'] = units


def plot_virial_for_multiple_gals_by_mass(data_directory, mass_bins, mass_colors, virial_bool, stellar_mass_bool):
	data_files = get_output_files(data_directory)
	plotted_once = False

	for j in range(0,np.size(mass_bins)-1):
		num_in_bin = 0
		final_gas_virial_ratio = np.array([])
		final_DM_virial_ratio = np.array([])
		final_radii = np.array([])
		for file in data_files:
			try:
				with h5py.File(file) as hf:
					GalaxyProperties = hf.get('GalaxyProperties')
					radii = np.array(GalaxyProperties.get('radius_of_output'))
					gas_virial_ratio = np.array(GalaxyProperties.get('gas_virial_ratio'))
					DM_virial_ratio = np.array(GalaxyProperties.get('DM_virial_ratio'))
					if stellar_mass_bool:
						gal_mass = np.log10(np.array(GalaxyProperties.get('gal_stellar_mass')))
					else:
						gal_mass = np.log10(np.array(GalaxyProperties.get('gal_mass')))
					gal_R200 = np.array(GalaxyProperties.get('gal_R200'))
					if (gal_mass < mass_bins[j+1]) & (gal_mass > mass_bins[j]):
						for i in range(0,np.size(radii)):
							if np.size(final_gas_virial_ratio > 0):
								num_in_bin += 1
								final_gas_virial_ratio = np.append(final_gas_virial_ratio,gas_virial_ratio[i])
								final_DM_virial_ratio = np.append(final_DM_virial_ratio,DM_virial_ratio[i])
								if virial_bool:
									final_radii = np.append(final_radii,radii[i]/gal_R200)
								else:
									final_radii = np.append(final_radii,radii[i])
							else: 
								num_in_bin += 1
								final_gas_virial_ratio = np.array(gas_virial_ratio[i])
								final_DM_virial_ratio = np.array(DM_virial_ratio[i])
								if virial_bool:
									final_radii = np.array(radii[i]/gal_R200)
								else:
									final_radii = np.array(radii[i])
			except:
				print 'no virial data'
		
		radii_bins = np.linspace(np.min(final_radii), np.max(final_radii), 50.0)
		plot_radii = np.zeros(np.size(radii_bins)-1)
		plot_gas_virial_ratio = np.zeros(np.size(radii_bins)-1)
		plot_DM_virial_ratio = np.zeros(np.size(radii_bins)-1)
		for i in range(0,np.size(radii_bins)-1):
			plot_gas_virial_ratio[i] = np.mean(final_gas_virial_ratio[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_DM_virial_ratio[i] = np.mean(final_DM_virial_ratio[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_radii[i] = np.mean([radii_bins[i],radii_bins[i+1]])
		if plotted_once == False:
			plt.plot(plot_radii, plot_gas_virial_ratio, mass_colors[j]+'-', label = 'gas')
			plt.plot(plot_radii, plot_DM_virial_ratio, mass_colors[j]+'-.', label = 'DM')
			plt.plot(plot_radii, plot_gas_virial_ratio, mass_colors[j]+'-', label = '%.1f-%.1fkc:(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
			plotted_once = True
		else:
			plt.plot(plot_radii, plot_DM_virial_ratio, mass_colors[j]+'-.')#, label = 'DM %d-%dkpc:(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
			plt.plot(plot_radii, plot_gas_virial_ratio, mass_colors[j]+'-', label = '%.1f-%.1fkc:(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
	plt.legend(loc='upper right')
	plt.title('Virial Ratios vs Radius')
	plt.ylabel('Virial Ratio')
	if virial_bool:
		plt.xlabel('radius (virial radii)')
		plt.savefig('multi_gal_virial_test_by_mass_R_vir.png')
	else:
		plt.xlabel('radius (kpc)')
		plt.savefig('multi_gal_virial_test_by_mass_kpc.png')
	plt.close()


def get_output_files(data_directory):
	data_directory = str(data_directory)
	if data_directory[-1] != '/':
		data_directory += '/'

	data_files = glob.glob(data_directory + '*/output_*.hdf5')
	return data_files

def plot_energy_sources_for_multiple_gals_by_mass_gas_ratios(data_directory, mass_bins, mass_colors, virial_bool, stellar_mass_bool, energy_values_bool):
	data_files = get_output_files(data_directory)
	plotted_once = False

	for j in range(0,np.size(mass_bins)-1):
		num_in_bin = 0
		final_gas_T_KE = np.array([])
		final_gas_bulk_KE = np.array([])
		final_DM_bulk_KE = np.array([])
		final_radii = np.array([])
		for file in data_files:
			try:
				with h5py.File(file) as hf:
					GalaxyProperties = hf.get('GalaxyProperties')
					radii = np.array(GalaxyProperties.get('radius_of_output'))
					gas_T_KE = np.array(GalaxyProperties.get('gas_T_KE'))
					gas_bulk_KE = np.array(GalaxyProperties.get('gas_KE_in_radii'))
					DM_bulk_KE = np.array(GalaxyProperties.get('DM_KE_in_radii'))
					gal_R200 = np.array(GalaxyProperties.get('gal_R200'))
					if stellar_mass_bool:
						gal_mass = np.log10(np.array(GalaxyProperties.get('gal_stellar_mass')))
					else:
						gal_mass = np.log10(np.array(GalaxyProperties.get('gal_mass')))

					if (gal_mass < mass_bins[j+1]) & (gal_mass > mass_bins[j]):
						for i in range(0,np.size(radii)):
							if np.size(final_gas_T_KE > 0):
								num_in_bin += 1
								final_gas_T_KE = np.append(final_gas_T_KE,gas_T_KE[i])
								final_gas_bulk_KE = np.append(final_gas_bulk_KE,gas_bulk_KE[i])
								final_DM_bulk_KE = np.append(final_DM_bulk_KE,DM_bulk_KE[i])
								if virial_bool:
									final_radii = np.append(final_radii,radii[i]/gal_R200)
								else:
									final_radii = np.append(final_radii,radii[i])
							else: 
								num_in_bin += 1
								final_gas_T_KE = np.array(gas_T_KE[i])
								final_gas_bulk_KE = np.array(gas_bulk_KE[i])
								final_DM_bulk_KE = np.array(DM_bulk_KE[i])
								if virial_bool:
									final_radii = np.array(radii[i]/gal_R200)
								else:
									final_radii = np.array(radii[i])
			except:
				print 'no virial data'
		
		radii_bins = np.linspace(np.min(final_radii), np.max(final_radii), 50.0)
		plot_radii = np.zeros(np.size(radii_bins)-1)
		plot_gas_T_KE = np.zeros(np.size(radii_bins)-1)
		plot_gas_bulk_KE = np.zeros(np.size(radii_bins)-1)
		plot_DM_bulk_KE = np.zeros(np.size(radii_bins)-1)
		for i in range(0,np.size(radii_bins)-1):
			plot_gas_T_KE[i] = np.mean(final_gas_T_KE[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_gas_bulk_KE[i] = np.mean(final_gas_bulk_KE[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_DM_bulk_KE[i] = np.mean(final_DM_bulk_KE[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_radii[i] = np.mean([radii_bins[i],radii_bins[i+1]])
		if energy_values_bool:
			if plotted_once==False:
				plt.semilogy(plot_radii, plot_gas_T_KE, mass_colors[j]+'-', label = 'gas KE (T)')
				plt.semilogy(plot_radii, plot_gas_bulk_KE, mass_colors[j]+'+', label = 'gas KE (bulk)')
				plt.semilogy(plot_radii, plot_DM_bulk_KE, mass_colors[j]+'-.', label = 'DM KE (bulk)')
				plt.semilogy(plot_radii, plot_gas_T_KE, mass_colors[j]+'-', label = '%.1f-%.1f log10(M/M_sol)' %(mass_bins[j], mass_bins[j+1]))
				plotted_once = True
			else:
				plt.semilogy(plot_radii, plot_gas_T_KE, mass_colors[j]+'-', label = '%.1f-%.1fk log10(M/M_sol)' %(mass_bins[j], mass_bins[j+1]))
				plt.semilogy(plot_radii, plot_gas_bulk_KE, mass_colors[j]+'+')#, label = 'DM %d-%dkpc:(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
				plt.semilogy(plot_radii, plot_DM_bulk_KE, mass_colors[j]+'-.')#, label = 'DM %d-%dkpc:(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
		else:
			plt.plot(plot_radii, plot_gas_T_KE/plot_gas_bulk_KE, mass_colors[j]+'-', label = '%.1f-%.1f log10(M/M_sol)' %(mass_bins[j], mass_bins[j+1]))
	plt.legend(loc='upper center')
	plt.ylabel('Ratio of Energies in Gas Particles (T/Bulk KE)')
	
	if energy_values_bool:
		plt.title('KE Energy Sources vs Radius')
		plt.ylabel('KE per unit mass')
		plt.ylim(ymin=10.0**12.8,ymax=1.e16)
		if virial_bool:
			plt.xlabel('radius (virial radii)')
			plt.savefig('multi_gal_energy_R_vir.png')
		else:
			plt.xlabel('radius (kpc)')
			plt.savefig('multi_gal_energy_kpc.png')
	else:
		plt.title('gas KE ratio (T/bulk) vs Radius')
		plt.ylabel('gas T KE/gas bulk KE')
		if virial_bool:
			plt.xlabel('radius (virial radii)')
			plt.savefig('multi_gal_gas_energy_ratio_R_vir.png')
		else:
			plt.xlabel('radius (kpc)')
			plt.savefig('multi_gal_gas_energy_ratio_kpc.png')
	plt.close()

def plot_energy_sources_for_multiple_gals_by_mass_bulk_ratios(data_directory, mass_bins, mass_colors, virial_bool, stellar_mass_bool, energy_values_bool):
	data_files = get_output_files(data_directory)
	plotted_once = False

	for j in range(0,np.size(mass_bins)-1):
		num_in_bin = 0
		final_gas_T_KE = np.array([])
		final_gas_bulk_KE = np.array([])
		final_DM_bulk_KE = np.array([])
		final_radii = np.array([])
		for file in data_files:
			try:
				with h5py.File(file) as hf:
					GalaxyProperties = hf.get('GalaxyProperties')
					radii = np.array(GalaxyProperties.get('radius_of_output'))
					gas_T_KE = np.array(GalaxyProperties.get('gas_T_KE'))
					gas_bulk_KE = np.array(GalaxyProperties.get('gas_KE_in_radii'))
					DM_bulk_KE = np.array(GalaxyProperties.get('DM_KE_in_radii'))
					gal_R200 = np.array(GalaxyProperties.get('gal_R200'))
					if stellar_mass_bool:
						gal_mass = np.log10(np.array(GalaxyProperties.get('gal_stellar_mass')))
					else:
						gal_mass = np.log10(np.array(GalaxyProperties.get('gal_mass')))

					if (gal_mass < mass_bins[j+1]) & (gal_mass > mass_bins[j]):
						for i in range(0,np.size(radii)):
							if np.size(final_gas_T_KE > 0):
								num_in_bin += 1
								final_gas_T_KE = np.append(final_gas_T_KE,gas_T_KE[i])
								final_gas_bulk_KE = np.append(final_gas_bulk_KE,gas_bulk_KE[i])
								final_DM_bulk_KE = np.append(final_DM_bulk_KE,DM_bulk_KE[i])
								if virial_bool:
									final_radii = np.append(final_radii,radii[i]/gal_R200)
								else:
									final_radii = np.append(final_radii,radii[i])
							else: 
								num_in_bin += 1
								final_gas_T_KE = np.array(gas_T_KE[i])
								final_gas_bulk_KE = np.array(gas_bulk_KE[i])
								final_DM_bulk_KE = np.array(DM_bulk_KE[i])
								if virial_bool:
									final_radii = np.array(radii[i]/gal_R200)
								else:
									final_radii = np.array(radii[i])
			except:
				print 'no virial data'
		
		radii_bins = np.linspace(np.min(final_radii), np.max(final_radii), 50.0)
		plot_radii = np.zeros(np.size(radii_bins)-1)
		plot_gas_T_KE = np.zeros(np.size(radii_bins)-1)
		plot_gas_bulk_KE = np.zeros(np.size(radii_bins)-1)
		plot_DM_bulk_KE = np.zeros(np.size(radii_bins)-1)
		for i in range(0,np.size(radii_bins)-1):
			plot_gas_T_KE[i] = np.mean(final_gas_T_KE[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_gas_bulk_KE[i] = np.mean(final_gas_bulk_KE[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_DM_bulk_KE[i] = np.mean(final_DM_bulk_KE[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_radii[i] = np.mean([radii_bins[i],radii_bins[i+1]])
		if energy_values_bool:
			if plotted_once==False:
				plt.semilogy(plot_radii, plot_gas_T_KE, mass_colors[j]+'-', label = 'gas KE (T)')
				plt.semilogy(plot_radii, plot_gas_bulk_KE, mass_colors[j]+'+', label = 'gas KE (bulk)')
				plt.semilogy(plot_radii, plot_DM_bulk_KE, mass_colors[j]+'-.', label = 'DM KE (bulk)')
				plt.semilogy(plot_radii, plot_gas_T_KE, mass_colors[j]+'-', label = '%.1f-%.1f log10(M/M_sol)' %(mass_bins[j], mass_bins[j+1]))
				plotted_once = True
			else:
				plt.semilogy(plot_radii, plot_gas_T_KE, mass_colors[j]+'-', label = '%.1f-%.1f log10(M/M_sol)' %(mass_bins[j], mass_bins[j+1]))
				plt.semilogy(plot_radii, plot_gas_bulk_KE, mass_colors[j]+'+')#, label = 'DM %d-%dkpc:(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
				plt.semilogy(plot_radii, plot_DM_bulk_KE, mass_colors[j]+'-.')#, label = 'DM %d-%dkpc:(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
		else:
			plt.plot(plot_radii, plot_gas_bulk_KE/plot_DM_bulk_KE, mass_colors[j]+'-', label = '%.1f-%.1f log10(M/M_sol)' %(mass_bins[j], mass_bins[j+1]))
	plt.legend(loc='upper center')
	plt.ylabel('Ratio of Bulk KE (gas/DM)')
	
	if energy_values_bool:
		plt.title('KE Energy Sources vs Radius')
		plt.ylabel('KE per unit mass')
		plt.ylim(ymin=10.0**12.8,ymax=1.e16)
		if virial_bool:
			plt.xlabel('radius (virial radii)')
			plt.savefig('multi_gal_energy_R_vir.png')
		else:
			plt.xlabel('radius (kpc)')
			plt.savefig('multi_gal_energy_kpc.png')
	else:
		plt.title('gas/DM bulk KE ratio vs Radius')
		plt.ylabel('gas T KE/gas bulk KE')
		if virial_bool:
			plt.xlabel('radius (virial radii)')
			plt.savefig('multi_gal_gas_DM_bulk_energy_ratio_R_vir.png')
		else:
			plt.xlabel('radius (kpc)')
			plt.savefig('multi_gal_gas_DM_bulk_energy_ratio_kpc.png')
	plt.close()


def plot_energy_sources_for_multiple_gals_by_mass_gas_to_DM(data_directory, mass_bins, mass_colors, virial_bool, stellar_mass_bool, energy_values_bool):
	data_files = get_output_files(data_directory)
	plotted_once = False

	for j in range(0,np.size(mass_bins)-1):
		num_in_bin = 0
		final_gas_T_KE = np.array([])
		final_gas_bulk_KE = np.array([])
		final_DM_bulk_KE = np.array([])
		final_radii = np.array([])
		for file in data_files:
			try:
				with h5py.File(file) as hf:
					GalaxyProperties = hf.get('GalaxyProperties')
					radii = np.array(GalaxyProperties.get('radius_of_output'))
					gas_T_KE = np.array(GalaxyProperties.get('gas_T_KE'))
					gas_bulk_KE = np.array(GalaxyProperties.get('gas_KE_in_radii'))
					DM_bulk_KE = np.array(GalaxyProperties.get('DM_KE_in_radii'))
					gal_R200 = np.array(GalaxyProperties.get('gal_R200'))
					if stellar_mass_bool:
						gal_mass = np.log10(np.array(GalaxyProperties.get('gal_stellar_mass')))
					else:
						gal_mass = np.log10(np.array(GalaxyProperties.get('gal_mass')))

					if (gal_mass < mass_bins[j+1]) & (gal_mass > mass_bins[j]):
						for i in range(0,np.size(radii)):
							if np.size(final_gas_T_KE > 0):
								num_in_bin += 1
								final_gas_T_KE = np.append(final_gas_T_KE,gas_T_KE[i])
								final_gas_bulk_KE = np.append(final_gas_bulk_KE,gas_bulk_KE[i])
								final_DM_bulk_KE = np.append(final_DM_bulk_KE,DM_bulk_KE[i])
								if virial_bool:
									final_radii = np.append(final_radii,radii[i]/gal_R200)
								else:
									final_radii = np.append(final_radii,radii[i])
							else: 
								num_in_bin += 1
								final_gas_T_KE = np.array(gas_T_KE[i])
								final_gas_bulk_KE = np.array(gas_bulk_KE[i])
								final_DM_bulk_KE = np.array(DM_bulk_KE[i])
								if virial_bool:
									final_radii = np.array(radii[i]/gal_R200)
								else:
									final_radii = np.array(radii[i])
			except:
				print 'no virial data'
		
		radii_bins = np.linspace(np.min(final_radii), np.max(final_radii), 50.0)
		plot_radii = np.zeros(np.size(radii_bins)-1)
		plot_gas_T_KE = np.zeros(np.size(radii_bins)-1)
		plot_gas_bulk_KE = np.zeros(np.size(radii_bins)-1)
		plot_DM_bulk_KE = np.zeros(np.size(radii_bins)-1)
		for i in range(0,np.size(radii_bins)-1):
			plot_gas_T_KE[i] = np.mean(final_gas_T_KE[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_gas_bulk_KE[i] = np.mean(final_gas_bulk_KE[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_DM_bulk_KE[i] = np.mean(final_DM_bulk_KE[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_radii[i] = np.mean([radii_bins[i],radii_bins[i+1]])
		if energy_values_bool:
			if plotted_once==False:
				plt.semilogy(plot_radii, plot_gas_T_KE, mass_colors[j]+'-', label = 'gas KE (T)')
				plt.semilogy(plot_radii, plot_gas_bulk_KE, mass_colors[j]+'+', label = 'gas KE (bulk)')
				plt.semilogy(plot_radii, plot_DM_bulk_KE, mass_colors[j]+'-.', label = 'DM KE (bulk)')
				plt.semilogy(plot_radii, plot_gas_T_KE, mass_colors[j]+'-', label = '%.1f-%.1f log10(M/M_sol)' %(mass_bins[j], mass_bins[j+1]))
				plotted_once = True
			else:
				plt.semilogy(plot_radii, plot_gas_T_KE, mass_colors[j]+'-', label = '%.1f-%.1f log10(M/M_sol)' %(mass_bins[j], mass_bins[j+1]))
				plt.semilogy(plot_radii, plot_gas_bulk_KE, mass_colors[j]+'+')#, label = 'DM %d-%dkpc:(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
				plt.semilogy(plot_radii, plot_DM_bulk_KE, mass_colors[j]+'-.')#, label = 'DM %d-%dkpc:(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
		else:
			plt.plot(plot_radii, (plot_gas_T_KE+plot_gas_bulk_KE)/plot_DM_bulk_KE, mass_colors[j]+'-', label = '%.1f-%.1f log10(M/M_sol)' %(mass_bins[j], mass_bins[j+1]))
	plt.legend(loc='upper center')
	plt.ylabel('Ratio of Total Energy (gas/DM)')
	
	if energy_values_bool:
		plt.title('KE Energy Sources vs Radius')
		plt.ylabel('KE per unit mass')
		plt.ylim(ymin=10.0**12.8,ymax=1.e16)
		if virial_bool:
			plt.xlabel('radius (virial radii)')
			plt.savefig('multi_gal_energy_R_vir.png')
		else:
			plt.xlabel('radius (kpc)')
			plt.savefig('multi_gal_energy_kpc.png')
	else:
		plt.title('gas/DM Energy ratio vs Radius')
		plt.ylabel('gas T KE/gas bulk KE')
		if virial_bool:
			plt.xlabel('radius (virial radii)')
			plt.savefig('multi_gal_gas_DM_energy_ratio_R_vir.png')
		else:
			plt.xlabel('radius (kpc)')
			plt.savefig('multi_gal_gas_DM_energy_ratio_kpc.png')
	plt.close()

def plot_energy_sources_for_multiple_gals_by_mass_gas_T_to_DM(data_directory, mass_bins, mass_colors, virial_bool, stellar_mass_bool, energy_values_bool):
	data_files = get_output_files(data_directory)
	plotted_once = False

	for j in range(0,np.size(mass_bins)-1):
		num_in_bin = 0
		final_gas_T_KE = np.array([])
		final_gas_bulk_KE = np.array([])
		final_DM_bulk_KE = np.array([])
		final_radii = np.array([])
		for file in data_files:
			try:
				with h5py.File(file) as hf:
					GalaxyProperties = hf.get('GalaxyProperties')
					radii = np.array(GalaxyProperties.get('radius_of_output'))
					gas_T_KE = np.array(GalaxyProperties.get('gas_T_KE'))
					gas_bulk_KE = np.array(GalaxyProperties.get('gas_KE_in_radii'))
					DM_bulk_KE = np.array(GalaxyProperties.get('DM_KE_in_radii'))
					gal_R200 = np.array(GalaxyProperties.get('gal_R200'))
					if stellar_mass_bool:
						gal_mass = np.log10(np.array(GalaxyProperties.get('gal_stellar_mass')))
					else:
						gal_mass = np.log10(np.array(GalaxyProperties.get('gal_mass')))

					if (gal_mass < mass_bins[j+1]) & (gal_mass > mass_bins[j]):
						for i in range(0,np.size(radii)):
							if np.size(final_gas_T_KE > 0):
								num_in_bin += 1
								final_gas_T_KE = np.append(final_gas_T_KE,gas_T_KE[i])
								final_gas_bulk_KE = np.append(final_gas_bulk_KE,gas_bulk_KE[i])
								final_DM_bulk_KE = np.append(final_DM_bulk_KE,DM_bulk_KE[i])
								if virial_bool:
									final_radii = np.append(final_radii,radii[i]/gal_R200)
								else:
									final_radii = np.append(final_radii,radii[i])
							else: 
								num_in_bin += 1
								final_gas_T_KE = np.array(gas_T_KE[i])
								final_gas_bulk_KE = np.array(gas_bulk_KE[i])
								final_DM_bulk_KE = np.array(DM_bulk_KE[i])
								if virial_bool:
									final_radii = np.array(radii[i]/gal_R200)
								else:
									final_radii = np.array(radii[i])
			except:
				print 'no virial data'
		
		radii_bins = np.linspace(np.min(final_radii), np.max(final_radii), 50.0)
		plot_radii = np.zeros(np.size(radii_bins)-1)
		plot_gas_T_KE = np.zeros(np.size(radii_bins)-1)
		plot_gas_bulk_KE = np.zeros(np.size(radii_bins)-1)
		plot_DM_bulk_KE = np.zeros(np.size(radii_bins)-1)
		for i in range(0,np.size(radii_bins)-1):
			plot_gas_T_KE[i] = np.mean(final_gas_T_KE[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_gas_bulk_KE[i] = np.mean(final_gas_bulk_KE[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_DM_bulk_KE[i] = np.mean(final_DM_bulk_KE[np.where((final_radii>radii_bins[i]) & (final_radii<radii_bins[i+1]))[0]])
			plot_radii[i] = np.mean([radii_bins[i],radii_bins[i+1]])
		if energy_values_bool:
			if plotted_once==False:
				plt.semilogy(plot_radii, plot_gas_T_KE, mass_colors[j]+'-', label = 'gas KE (T)')
				plt.semilogy(plot_radii, plot_gas_bulk_KE, mass_colors[j]+'+', label = 'gas KE (bulk)')
				plt.semilogy(plot_radii, plot_DM_bulk_KE, mass_colors[j]+'-.', label = 'DM KE (bulk)')
				plt.semilogy(plot_radii, plot_gas_T_KE, mass_colors[j]+'-', label = '%.1f-%.1f log10(M/M_sol)' %(mass_bins[j], mass_bins[j+1]))
				plotted_once = True
			else:
				plt.semilogy(plot_radii, plot_gas_T_KE, mass_colors[j]+'-', label = '%.1f-%.1f log10(M/M_sol)' %(mass_bins[j], mass_bins[j+1]))
				plt.semilogy(plot_radii, plot_gas_bulk_KE, mass_colors[j]+'+')#, label = 'DM %d-%dkpc:(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
				plt.semilogy(plot_radii, plot_DM_bulk_KE, mass_colors[j]+'-.')#, label = 'DM %d-%dkpc:(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
		else:
			plt.plot(plot_radii, (plot_gas_T_KE)/plot_DM_bulk_KE, mass_colors[j]+'-', label = '%.1f-%.1f log10(M/M_sol)' %(mass_bins[j], mass_bins[j+1]))
	plt.legend(loc='upper center')
	plt.ylabel('Ratio of Thermal Energy in Gas to Bulk KE in DM')
	
	if energy_values_bool:
		plt.title('KE Energy Sources vs Radius')
		plt.ylabel('KE per unit mass')
		plt.ylim(ymin=10.0**12.8,ymax=1.e16)
		if virial_bool:
			plt.xlabel('radius (virial radii)')
			plt.savefig('multi_gal_energy_R_vir.png')
		else:
			plt.xlabel('radius (kpc)')
			plt.savefig('multi_gal_energy_kpc.png')
	else:
		plt.title('gas T vs DM bulk KE Energy ratio vs Radius')
		plt.ylabel('gas T KE/gas bulk KE')
		if virial_bool:
			plt.xlabel('radius (virial radii)')
			plt.savefig('multi_gal_gas_T_DM_energy_ratio_R_vir.png')
		else:
			plt.xlabel('radius (kpc)')
			plt.savefig('multi_gal_gas_T_DM_energy_ratio_kpc.png')
	plt.close()


def edit_text(file, new_file_name, keywords, replacements):
	with open(file,'r') as input:
		with open(new_file_name, 'w') as output:
			for line in input:
				iteration = 0
				for word in keywords:
					if (word in line) and (line[0] != '%'): # looks for first call of word, but not a comment! 
						output.write(replacements[iteration] + '\n')
						break
					iteration += 1
					if iteration >= np.size(keywords):
						output.write(line)

def gas_mass_in_annular_rings(gal_directory, group_number, radii, virial_radii_bool, R200, particles_included_keyword, subfind_included_keyword):
	cool_bool = True
	mass_in_ann = np.zeros(np.size(radii)-1)
	neut_mass_in_ann = np.zeros(np.size(radii)-1)
	cum_mass = np.zeros(np.size(radii)-1)

	snap_files = get_snap_files(gal_directory, particles_included_keyword)
	GrpIDs = read_array(snap_files,"Subhalo/GroupNumber", include_file_keyword=subfind_included_keyword)
	SubIDs = read_array(snap_files, "Subhalo/SubGroupNumber", include_file_keyword=subfind_included_keyword)
	index = np.where((GrpIDs == float(group_number)) & (SubIDs == 0))[0] # picks out most massive galaxy in the designated group
	H_frac = read_array(snap_files, array_name='PartType0/ElementAbundance/Hydrogen',include_file_keyword=particles_included_keyword)
	try:
		HI = read_array(snap_files, array_name='PartType0/ChemicalAbundances/HydrogenI', include_file_keyword=particles_included_keyword, get_scaling_conversions=False)
	except:
		HI = np.zeros(np.size(H_frac))
		print 'missing ChemicalAbundances in %s with tag %s' % (gal_directory, particles_included_keyword)
	part_mass = read_array(snap_files, array_name='PartType0/Mass', include_file_keyword=particles_included_keyword)
	gas_coords = read_array(snap_files, array_name='PartType0/Coordinates',include_file_keyword=particles_included_keyword)
	gal_coords = read_array(snap_files, "Subhalo/CentreOfPotential", include_file_keyword=subfind_included_keyword)[index][0]
	temperature = read_array(snap_files,'PartType0/Temperature', include_file_keyword=particles_included_keyword, get_scaling_conversions=False)
	box_size = read_attribute(snap_files,'Header','BoxSize',include_file_keyword=particles_included_keyword)
	expansion_factor = read_attribute(snap_files,'Header','ExpansionFactor',include_file_keyword=particles_included_keyword)
	hubble_param = read_attribute(snap_files,'Header','HubbleParam',include_file_keyword=particles_included_keyword)

	gas_coords = gal_centered_coords(gas_coords,gal_coords,box_size,expansion_factor,hubble_param)
	if virial_radii_bool:
		gas_coords /= R200*1.e3*parsec_in_cm
	if cool_bool:
		temp_cut_indices = np.argwhere(temperature<=1.e5)[:,0]
		H_frac = H_frac[temp_cut_indices]
		HI = HI[temp_cut_indices]
		part_mass = part_mass[temp_cut_indices]
		gas_coords = gas_coords[temp_cut_indices, :]
	# get distances ourselves so we don't calculate it every time
	distances = np.sqrt(gas_coords[:,0]**2. + gas_coords[:,1]**2. + gas_coords[:,2]**2.)

	for i in range(np.size(mass_in_ann)):
		# get indices ourselves as well since we use the same indices 3 times
		in_btwn_indices = np.argwhere(((distances > radii[i]) & (distances <= radii[i+1])))[:,0]

		part_mass_ann = part_mass[in_btwn_indices]
		H_frac_ann = H_frac[in_btwn_indices]
		HI_ann = HI[in_btwn_indices]
		neut_mass_in_ann[i] = np.sum(part_mass_ann*H_frac_ann*HI_ann)/M_sol
		mass_in_ann[i] = np.sum(part_mass_ann*H_frac_ann)/M_sol

	return mass_in_ann, neut_mass_in_ann

def gal_centered_coords_and_vels_in_annulus(gal_directory, group_number, radii, R200, particles_included_keyword, subfind_included_keyword):
	cool_bool = False

	snap_files = get_snap_files(gal_directory, particles_included_keyword)
	GrpIDs = read_array(snap_files,"Subhalo/GroupNumber", include_file_keyword=subfind_included_keyword)
	SubIDs = read_array(snap_files, "Subhalo/SubGroupNumber", include_file_keyword=subfind_included_keyword)
	index = np.where((GrpIDs == float(group_number)) & (SubIDs == 0))[0] # picks out most massive galaxy in the designated group
	gas_coords = read_array(snap_files, array_name='PartType0/Coordinates',include_file_keyword=particles_included_keyword)
	gal_coords = read_array(snap_files, "Subhalo/CentreOfPotential", include_file_keyword=subfind_included_keyword)[index][0]
	gas_vel = read_array(snap_files, array_name="PartType0/Velocity", include_file_keyword=particles_included_keyword)
	gal_vel = read_array(snap_files, "Subhalo/Velocity", include_file_keyword=subfind_included_keyword)[index][0]
	box_size = read_attribute(snap_files,'Header','BoxSize',include_file_keyword=particles_included_keyword)
	expansion_factor = read_attribute(snap_files,'Header','ExpansionFactor',include_file_keyword=particles_included_keyword)
	hubble_param = read_attribute(snap_files,'Header','HubbleParam',include_file_keyword=particles_included_keyword)

	gas_coords = gal_centered_coords(gas_coords,gal_coords,box_size,expansion_factor,hubble_param)
	gas_vel = gas_vel - gal_vel

	coords_ann = particles_btwn_radii(gas_coords, gas_coords, radii[0]*R200, radii[1]*R200)
	vel_ann = particles_btwn_radii(gas_vel, gas_coords, radii[0]*R200, radii[1]*R200)

	return coords_ann, vel_ann

def actual_cumulative_mass_for_EAGLE_gals(directory_with_sim_gals, virial_radii_bool):

	min_mass, max_mass = 5., 15.
	if virial_radii_bool:
		radii = np.arange(0.0,1.05,0.05)
	else:
		# radii = np.arange(0.0,651.,5.)*parsec_in_cm*1.e3
		radii = np.arange(20.,170.,10.)*parsec_in_cm*1.e3
		# radii = np.arange(0.,20.,5.)*parsec_in_cm*1.e3 # inner radii check for Ben

	ann_gas_mass_all_gals = []
	neut_ann_gas_mass_all_gals = []
	halo_masses = []
	stellar_masses = []
	sSFRs = []
	R200s = []

	gal_hdf5_files = glob.glob(directory_with_sim_gals+'*/output*')
	done_one = False

	for  i, file in enumerate(gal_hdf5_files):
		if not done_one:
			print "on gal %d" % (i)
			with h5py.File(file, 'r') as hf:
				galaxy_properties = hf.get('GalaxyProperties')
				gal_mass =  np.log10(np.array(galaxy_properties.get('gal_mass')))[0]
				if ((gal_mass > min_mass) & (gal_mass <= max_mass)):
					stellar_mass = np.log10(np.array(galaxy_properties.get('gal_stellar_mass')))[0]
					sSFR = np.array(galaxy_properties.get('log10_sSFR'))[0]
					R200 = np.array(galaxy_properties.get('gal_R200'))[0] # this is in kpc!
					file_keyword = np.array(galaxy_properties.get('file_keyword'))[0][-12::]
					snap_directory = np.array(galaxy_properties.get('snap_directory'))[0]
					group_number = np.array(galaxy_properties.get('group_number'))[0]

					particles_included_keyword = 'snap_noneq_'+file_keyword
					subfind_included_keyword = 'eagle_subfind_tab_'+file_keyword
					temp_mass, temp_neut_mass = gas_mass_in_annular_rings(snap_directory, group_number, radii, virial_radii_bool, R200, particles_included_keyword, subfind_included_keyword)
					ann_gas_mass_all_gals.append(temp_mass), neut_ann_gas_mass_all_gals.append(temp_neut_mass)
					halo_masses.append(gal_mass)
					stellar_masses.append(stellar_mass)
					sSFRs.append(sSFR)
					R200s.append(R200)
					# if i == 3:
					# 	done_one = True

				else:
					continue
		else:
			continue


	np.set_printoptions(threshold=np.inf)
	arrays_to_print = [ann_gas_mass_all_gals, neut_ann_gas_mass_all_gals, radii, halo_masses, stellar_masses, sSFRs, R200s]

	var_prefaces = ["ann_masses", "neut_ann_masses", "radii", "halo_masses", "stellar_masses", "sSFRs", "R200s"]

	opening_lines = 'import numpy as np \n # All sim gals used for COS realizations. 20-170 (10) kpc cool only \n'
	filename = "/cosma/home/analyse/rhorton/snapshots/sim_ann_masses_t_cut.py"
	survey_realization_functions.print_data(filename, opening_lines, arrays_to_print, var_prefaces)
	np.set_printoptions(threshold=1000)


def radial_vels_for_all_particles_in_a_shell(directory_with_sim_gals):

	min_mass, max_mass = 5., 15.
	radii = [0.45, 0.55] # the min/max radius we will look for in virial radii
	v_dot_r_hat_list = []
	halo_masses = []
	stellar_masses = []
	sSFRs = []
	R200s = []

	gal_hdf5_files = glob.glob(directory_with_sim_gals+'*/output*')

	for  i, file in enumerate(gal_hdf5_files):
		print "on gal %d" % (i)
		with h5py.File(file, 'r') as hf:
			galaxy_properties = hf.get('GalaxyProperties')
			gal_mass =  np.log10(np.array(galaxy_properties.get('gal_mass')))[0]
			stellar_mass = np.log10(np.array(galaxy_properties.get('gal_stellar_mass')))[0]
			sSFR = np.array(galaxy_properties.get('log10_sSFR'))[0]
			R200 = np.array(galaxy_properties.get('gal_R200'))[0] # in kpc!
			file_keyword = np.array(galaxy_properties.get('file_keyword'))[0][-12::]
			snap_directory = np.array(galaxy_properties.get('snap_directory'))[0]
			group_number = np.array(galaxy_properties.get('group_number'))[0]

		if ((gal_mass > min_mass) & (gal_mass <= max_mass)):
			particles_included_keyword = 'snap_noneq_'+file_keyword
			subfind_included_keyword = 'eagle_subfind_tab_'+file_keyword
			### get coordiantes and vels of all gas particles in range. Both relative to gal
			coords, vels = gal_centered_coords_and_vels_in_annulus(snap_directory, group_number, radii, R200*parsec_in_cm*1.e3, particles_included_keyword, subfind_included_keyword)
			inv_coord_norms = 1./np.linalg.norm(coords, axis=1)
			inv_coord_norms = np.reshape(inv_coord_norms, (np.size(inv_coord_norms),1))
			unit_coords = coords*inv_coord_norms

			v_dot_r_hat = np.zeros(np.shape(coords)[0])
			for i in range(np.size(v_dot_r_hat)):
				v_dot_r_hat[i] = np.dot(unit_coords[i,:], vels[i,:])

			v_dot_r_hat_list.append(v_dot_r_hat)
			halo_masses.append(gal_mass)
			stellar_masses.append(stellar_mass)
			sSFRs.append(sSFR)
			R200s.append(R200)

	np.set_printoptions(threshold=np.inf)
	arrays_to_print = [v_dot_r_hat_list, radii, halo_masses, stellar_masses, sSFRs, R200s]

	var_prefaces = ["v_dot_r", "radii", "halo_masses", "stellar_masses", "sSFRs", "R200s"]

	opening_lines = 'import numpy as np \n # All sim gals used for COS realizations \n # v dot r in shell from inner to outer radius (given in fraction of R_200) \n'
	filename = "/cosma/home/analyse/rhorton/snapshots/sim_v_dot_r.py"
	survey_realization_functions.print_data(filename, opening_lines, arrays_to_print, var_prefaces)
	np.set_printoptions(threshold=1000)








