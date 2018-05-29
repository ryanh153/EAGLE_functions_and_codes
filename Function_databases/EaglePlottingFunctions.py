### Plotting functions for Eagle Simulations
### Functions in order of appearance
###
### velocity_dispersion: Plots velocity (velocity minus galactic velocity) vs distnace as a 2d histogram for gas and DM
###
### spherical_velocity: plots all three components of spherical velocity for gas and DM as a function of distance (2d histogram)
###
### force_balance_and_virial: Plots the forces the particles experience and the ratio of virial terms for gas and DM
### so far for gas Pressure gradient, centripetal and gravity are plotted. 
### for DM only gravity and centripetal are plotted


import EagleFunctions
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import h5py
import glob
import glob

### Constants
parsec = 3.0857e18 # cm
G = 6.674e-8
M_sol = 1.98855e33 # g
kB = 1.380648e-16

atomic_mass_unit = 1.66053904e-24
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

### Plotting Functions

def velocity_dispersions(snap_directory, # directory where folders for snap files are (in this folder should be files or 
										 # subfolders for snapshots and group/FOF info)

radius, 								 # how far from galactic center should be plotted in virial radii

gal_num,								 # The index of the galaxy you want to look at in the eagle_subfind/FOF array

particles_included_keyword, 			 # string that is included in all snapshots so the right files can be selected
										 # Ex: files are snap_030_z000p205.0.hdf5 etc. => particles_included_keyword = "snap_030_z000p205"

group_included_keyword, 				 # string that is included in all group data files so that right files can be selectted
										 # Ex: files are group_tab_030_z000p205.0.hdf5 etc. => group_included_keyword = "group_tab_030_z000p205"

subfind_included_keyword):				 # string that is included in all eagle_subfind (FOF) files 
										 # Ex: files are eagle_subfind_tab_030_z000p205.0.hdf5 etc. => subfind_included_keyword = "eagle_subfind_tab_030_z000p205"

	# create array of files
	snap_files = EagleFunctions.get_snap_files(snap_directory)

	# pull out simulation parameters that are constant for all snapshots
	box_size = EagleFunctions.read_attribute(snap_files,'Header','BoxSize',include_file_keyword=particles_included_keyword)
	expansion_factor = EagleFunctions.read_attribute(snap_files,'Header','ExpansionFactor',include_file_keyword=particles_included_keyword)
	hubble_param = EagleFunctions.read_attribute(snap_files,'Header','HubbleParam',include_file_keyword=particles_included_keyword)

	# pull out properties of central galaxy
	# can get gal coords from FOF center of mass or median position of stars
	gal_coords = EagleFunctions.read_array(snap_files,"FOF/CentreOfMass",include_file_keyword=group_included_keyword)[gal_num] # zero is to get most massives halo
	
	#star_coords = EagleFunctions.read_array(snap_files,'PartType4/Coordinates',include_file_keyword=particles_included_keyword)
	#gal_coords = np.median(star_coords,axis = 0) # can be used if gal_coords from FOF file is inaccurate
	gal_M200 = EagleFunctions.read_array(snap_files,"FOF/Group_M_Crit200",include_file_keyword=subfind_included_keyword)[gal_num]
	gal_velocity = EagleFunctions.read_array(snap_files,"FOF/Velocity",include_file_keyword=group_included_keyword)[gal_num]
	gal_speed = np.sqrt(gal_velocity[0]**2.0 + gal_velocity[1]**2.0 + gal_velocity[2]**2.0)
	gal_R200 = EagleFunctions.read_array(snap_files,"FOF/Group_R_Crit200",include_file_keyword=subfind_included_keyword)[gal_num]
	radius = radius*gal_R200

	print "-------------------------------"
	print "Index in subfind/FOF array is %.0f" % (gal_num)
	print "Mass within virial radius is %.2e solar masses" % (gal_M200/M_sol)
	print "Galaxy radius is %.2e kpc" % (gal_R200/(1.e3*parsec))
	print "Galaxy velocity is (%.2e,%.2e,%.2e) km/s speed is %.2e km/s" % (gal_velocity[0]/(1.e5),gal_velocity[1]/(1.e5),gal_velocity[2]/(1.e5),gal_speed/(1.e5))
	print "-------------------------------"

	# get necessary gas properties
	gas_coords = EagleFunctions.read_array(snap_files,'PartType0/Coordinates',include_file_keyword=particles_included_keyword)
	gas_coords = EagleFunctions.gal_centered_coords(gas_coords,gal_coords,box_size,expansion_factor,hubble_param)
	gas_distance = np.sqrt(gas_coords[:,0]**2.0+gas_coords[:,1]**2.0+gas_coords[:,2]**2.0)

	gas_velocity = EagleFunctions.read_array(snap_files,"/PartType0/Velocity",include_file_keyword=particles_included_keyword)
	gas_velocity = gas_velocity-gal_velocity
	gas_speed = np.sqrt(gas_velocity[:,0]**2.0+gas_velocity[:,1]**2.0+gas_velocity[:,2]**2.0)

	# get necessary DM properties
	DM_coords = EagleFunctions.read_array(snap_files,'PartType1/Coordinates',include_file_keyword=particles_included_keyword)
	DM_coords = EagleFunctions.gal_centered_coords(DM_coords,gal_coords,box_size,expansion_factor,hubble_param)
	DM_distance = np.sqrt(DM_coords[:,0]**2.0+DM_coords[:,1]**2.0+DM_coords[:,2]**2.0)

	DM_velocity = EagleFunctions.read_array(snap_files,'PartType1/Velocity',include_file_keyword=particles_included_keyword)
	DM_velocity = DM_velocity-gal_velocity
	DM_speed = np.sqrt(DM_velocity[:,0]**2.0+DM_velocity[:,1]**2.0+DM_velocity[:,2]**2.0)

	# get properties of gas particles that fall within given radius of central galaxy
	gas_distance_in_R = EagleFunctions.particles_within_R(gas_distance,gas_coords,radius)
	gas_speed_in_R = EagleFunctions.particles_within_R(gas_speed,gas_coords,radius)

	# get properties of DM particles that fall within given radius of central galaxy
	DM_distance_in_R = EagleFunctions.particles_within_R(DM_distance,DM_coords,radius)
	DM_speed_in_R = EagleFunctions.particles_within_R(DM_speed,DM_coords,radius)

	# box_size, expansion_factor, hubble_param, gal_coords, gal_velocity, gal_M200, gal_R200 = \
	# EagleFunctions.get_basic_props(snap_directory, radius, gal_num, particles_included_keyword, group_included_keyword, subfind_included_keyword)

	# radius = radius*gal_R200
	# gal_speed = np.sqrt(np.sum(np.power(gal_velocity,2)))

	# gas_coords_in_R, gas_distance_in_R, gas_velocity_in_R, gas_speed_in_R = \
	# EagleFunctions.get_gas_props(snap_directory, radius, gal_num, particles_included_keyword, group_included_keyword, subfind_included_keyword,
	# box_size, expansion_factor, hubble_param, gal_coords, gal_velocity)

	# DM_coords_in_R, DM_distance_in_R, DM_velocity_in_R, DM_speed_in_R = \
	# EagleFunctions.get_DM_props(snap_directory, radius, gal_num, particles_included_keyword, group_included_keyword, subfind_included_keyword,
	# box_size, expansion_factor, hubble_param, gal_coords, gal_velocity)

	# set plotting limits so they're the same for both plots (for comparison purposes)
	ymax = np.max([np.max(DM_speed_in_R),np.max(gas_speed_in_R)])
	ymin = np.min([np.min(DM_speed_in_R),np.min(gas_speed_in_R)])

	xmax = np.max([np.max(DM_distance_in_R),np.max(gas_distance_in_R)])/(1000.*parsec)
	xmin = np.min([np.min(DM_distance_in_R),np.min(gas_distance_in_R)])/(1000.*parsec)

	# make DM dispersion plot
	plt.hist2d(DM_distance_in_R/(1000.*parsec),DM_speed_in_R,bins=125, norm = matplotlib.colors.LogNorm()) # dark matter
	plt.colorbar()
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("DM Vel Dispersion within %.1e R_vir (%.0f)" % ((radius/gal_R200), gal_num))
	plt.xlabel("Distance from Galaxy Center (kpc)")
	plt.ylabel("Speed (cm/s)")
	title = "DM Vel Dispersion %.1e virial radii (%.0f).png" % (radius/gal_R200, gal_num)
	plt.savefig(title, format='png')
	plt.close()

	# make gas dispersion plot
	plt.hist2d(gas_distance_in_R/(1000.*parsec),gas_speed_in_R,bins=125, norm = matplotlib.colors.LogNorm()) # dark matter
	plt.colorbar()
	plt.ylim ([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("Gas Vel Dispersion within %.1e R_vir (%.0f)" % ((radius/gal_R200), gal_num))
	plt.xlabel("Distance from Galaxy Center (kpc)")
	plt.ylabel("Speed (cm/s)")
	title = "Gas Velocity Dispersion %.1e virial radii (%.0f).png" % (radius/gal_R200, gal_num)
	plt.savefig(title, format='png')
	plt.close()



def spherical_velocity(snap_directory, # directory where folders for snap files are (in this folder should be files or 
										 # subfolders for snapshots and group/FOF info)

radius, 								 # how far from galactic center should be plotted in virial radii

gal_num,								 # The index of the galaxy you want to look at in the eagle_subfind/FOF array

particles_included_keyword, 			 # string that is included in all snapshots so the right files can be selected
										 # Ex: files are snap_030_z000p205.0.hdf5 etc. => particles_included_keyword = "snap_030_z000p205"

group_included_keyword, 				 # string that is included in all group data files so that right files can be selectted
										 # Ex: files are group_tab_030_z000p205.0.hdf5 etc. => group_included_keyword = "group_tab_030_z000p205"

subfind_included_keyword,				 # string that is included in all eagle_subfind (FOF) files 

percentile_cut=0.0):					 # Optional: if there are extremem outliers can opt to cut particles of extreme speed
										 # Ex: if percentile_cut = 1.0 the slowest and fastest 1% of particles will not be shown in the plot 


	# create array of files
	snap_files = EagleFunctions.get_snap_files(snap_directory)

	# pull out simulation parameters that are constant for all snapshots
	box_size = EagleFunctions.read_attribute(snap_files,'Header','BoxSize',include_file_keyword=particles_included_keyword)
	expansion_factor = EagleFunctions.read_attribute(snap_files,'Header','ExpansionFactor',include_file_keyword=particles_included_keyword)
	hubble_param = EagleFunctions.read_attribute(snap_files,'Header','HubbleParam',include_file_keyword=particles_included_keyword)

	# pull out properties of central galaxy
	# can get gal coords from FOF center of mass or median position of stars
	gal_coords = EagleFunctions.read_array(snap_files,"FOF/CentreOfMass",include_file_keyword=group_included_keyword)[gal_num] # zero is to get most massives halo
	
	#star_coords = EagleFunctions.read_array(snap_files,'PartType4/Coordinates',include_file_keyword=particles_included_keyword)
	#gal_coords = np.median(star_coords,axis = 0) # can be used if gal_coords from FOF file is inaccurate
	gal_M200 = EagleFunctions.read_array(snap_files,"FOF/Group_M_Crit200",include_file_keyword=subfind_included_keyword)[gal_num]
	gal_velocity = EagleFunctions.read_array(snap_files,"FOF/Velocity",include_file_keyword=group_included_keyword)[gal_num]
	gal_speed = np.sqrt(gal_velocity[0]**2.0 + gal_velocity[1]**2.0 + gal_velocity[2]**2.0)
	gal_R200 = EagleFunctions.read_array(snap_files,"FOF/Group_R_Crit200",include_file_keyword=subfind_included_keyword)[gal_num]
	radius = radius*gal_R200

	print "-------------------------------"
	print "Index in subfind/FOF array is %.0f" % (gal_num)
	print "Mass within virial radius is %.2e solar masses" % (gal_M200/M_sol)
	print "Galaxy radius is %.2e kpc" % (gal_R200/(1.e3*parsec))
	print "Galaxy velocity is (%.2e,%.2e,%.2e) km/s speed is %.2e km/s" % (gal_velocity[0]/(1.e5),gal_velocity[1]/(1.e5),gal_velocity[2]/(1.e5),gal_speed/(1.e5))
	print "-------------------------------"

	# get necessary gas properties
	gas_coords = EagleFunctions.read_array(snap_files,'PartType0/Coordinates',include_file_keyword=particles_included_keyword)
	gas_coords = EagleFunctions.gal_centered_coords(gas_coords,gal_coords,box_size,expansion_factor,hubble_param)
	gas_distance = np.sqrt(gas_coords[:,0]**2.0+gas_coords[:,1]**2.0+gas_coords[:,2]**2.0)

	gas_velocity = EagleFunctions.read_array(snap_files,"/PartType0/Velocity",include_file_keyword=particles_included_keyword)
	gas_velocity = gas_velocity-gal_velocity
	gas_speed = np.sqrt(gas_velocity[:,0]**2.0+gas_velocity[:,1]**2.0+gas_velocity[:,2]**2.0)

	# get necessary DM properties
	DM_coords = EagleFunctions.read_array(snap_files,'PartType1/Coordinates',include_file_keyword=particles_included_keyword)
	DM_coords = EagleFunctions.gal_centered_coords(DM_coords,gal_coords,box_size,expansion_factor,hubble_param)
	DM_distance = np.sqrt(DM_coords[:,0]**2.0+DM_coords[:,1]**2.0+DM_coords[:,2]**2.0)

	DM_velocity = EagleFunctions.read_array(snap_files,'PartType1/Velocity',include_file_keyword=particles_included_keyword)
	DM_velocity = DM_velocity-gal_velocity
	DM_speed = np.sqrt(DM_velocity[:,0]**2.0+DM_velocity[:,1]**2.0+DM_velocity[:,2]**2.0)

	# get properties of gas particles that fall within given radius of central galaxy
	gas_distance_in_R = EagleFunctions.particles_within_R(gas_distance,gas_coords,radius)

	# get properties of DM particles that fall within given radius of central galaxy
	DM_distance_in_R = EagleFunctions.particles_within_R(DM_distance,DM_coords,radius)

	# Convert velocities from cartesian to spherical coordinates [radial, theta, phi]
	gas_velocity_sph = EagleFunctions.cartesian_to_spherical_velocity(gas_velocity,gas_coords)
	gas_velocity_sph_in_R = EagleFunctions.particles_within_R(gas_velocity_sph,gas_coords,radius) 

	DM_velocity_sph = EagleFunctions.cartesian_to_spherical_velocity(DM_velocity,DM_coords)
	DM_velocity_sph_in_R = EagleFunctions.particles_within_R(DM_velocity_sph,DM_coords,radius)

	# pull out all components and how many particles fall within percentile cuts (if any is made)
	gas_velocity_radial,gas_distance_radial = EagleFunctions.remove_outliers(gas_velocity_sph_in_R[:,0],gas_distance_in_R,percentile_cut)
	gas_velocity_theta,gas_distance_theta = EagleFunctions.remove_outliers(gas_velocity_sph_in_R[:,1],gas_distance_in_R,percentile_cut)
	gas_velocity_phi,gas_distance_phi = EagleFunctions.remove_outliers(gas_velocity_sph_in_R[:,2],gas_distance_in_R,percentile_cut)

	DM_velocity_radial,DM_distance_radial = EagleFunctions.remove_outliers(DM_velocity_sph_in_R[:,0],DM_distance_in_R,percentile_cut)
	DM_velocity_theta,DM_distance_theta = EagleFunctions.remove_outliers(DM_velocity_sph_in_R[:,1],DM_distance_in_R,percentile_cut)
	DM_velocity_phi,DM_distance_phi = EagleFunctions.remove_outliers(DM_velocity_sph_in_R[:,2],DM_distance_in_R,percentile_cut)

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
	plt.title("Gas v_rad to %.1e kpc (%.0f)" % (radius/(1000.*parsec), gal_num))
	plt.xlabel("Distance from Center of Potential (cm)")
	plt.ylabel("Speed (cm/s)")
	title = "Gas Radial Velocity within %.1e virial radii (%.0f).png" % (radius/gal_R200, gal_num)
	plt.savefig(title, format='png')
	plt.close()

	plt.hist2d(gas_distance_theta,gas_velocity_theta,bins=125, norm = matplotlib.colors.LogNorm())
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("Gas v_theta to %.1e kpc (%.0f)" % (radius/(1000.*parsec), gal_num))
	plt.xlabel("Distance from Center of Potential (cm)")
	plt.ylabel("Speed (cm/s)")
	title = "Gas Theta Velocity within %.1e virial radii (%.0f).png" % (radius/gal_R200, gal_num)
	plt.savefig(title, format='png')
	plt.close()

	plt.hist2d(gas_distance_phi,gas_velocity_phi,bins=125, norm = matplotlib.colors.LogNorm())
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("Gas v_phi to %.1e kpc (%.0f)" % (radius/(1000.*parsec), gal_num))
	plt.xlabel("Distance from Center of Potential (cm)")
	plt.ylabel("Speed (cm/s)")
	title = "Gas Phi Velocity within %.1e virial radii (%.0f).png" % (radius/gal_R200, gal_num)
	plt.savefig(title, format='png')
	plt.close()

	# DM Plots
	plt.hist2d(DM_distance_radial,DM_velocity_radial,bins=125, norm = matplotlib.colors.LogNorm())
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("DM v_rad to %.1e kpc (%.0f)" % (radius/(1000.*parsec), gal_num))
	plt.xlabel("Distance from Center of Potential (cm)")
	plt.ylabel("Speed (cm/s)")
	title = "DM Radial Velocity within %.1e virial radii (%.0f).png" % (radius/gal_R200, gal_num)
	plt.savefig(title, format='png')
	plt.close()

	plt.hist2d(DM_distance_theta,DM_velocity_theta,bins=125, norm = matplotlib.colors.LogNorm())
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("DM v_theta to %.1e kpc (%.0f)" % (radius/(1000.*parsec), gal_num))
	plt.xlabel("Distance from Center of Potential (cm)")
	plt.ylabel("Speed (cm/s)")
	title = "DM Theta Velocity within %.1e virial radii (%.0f).png" % (radius/gal_R200, gal_num)
	plt.savefig(title, format='png')
	plt.close()

	plt.hist2d(DM_distance_phi,DM_velocity_phi,bins=125, norm = matplotlib.colors.LogNorm())
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])
	plt.title("DM v_phi to %.1e kpc (%.0f)" % (radius/(1000.*parsec), gal_num))
	plt.xlabel("Distance from Center of Potential (cm)")
	plt.ylabel("Speed (cm/s)")
	title = "DM Phi Velocity within %.1e virial radii (%.0f).png" % (radius/gal_R200, gal_num)
	plt.savefig(title, format='png')
	plt.close()

	return np.asarray([gal_M200, gal_R200, gal_velocity[0]/(1.e5),gal_velocity[1]/(1.e5),gal_velocity[2]/(1.e5),gal_speed/(1.e5)])


def force_balance_and_virial(snap_directory, # directory where folders for snap files are (in this folder should be files or 
											 # subfolders for snapshots and group/FOF info)

radius, 									 # how far from galactic center should be plotted in virial radii

gal_num,								 # The index of the galaxy you want to look at in the eagle_subfind/FOF array

particles_included_keyword, 				 # string that is included in all snapshots so the right files can be selected
											 # Ex: files are snap_030_z000p205.0.hdf5 etc. => particles_included_keyword = "snap_030_z000p205"

group_included_keyword, 					 # string that is included in all group data files so that right files can be selectted
											 # Ex: files are group_tab_030_z000p205.0.hdf5 etc. => group_included_keyword = "group_tab_030_z000p205"

subfind_included_keyword):					 # string that is included in all eagle_subfind (FOF) files 
											 # Ex: files are eagle_subfind_tab_030_z000p205.0.hdf5 etc. => subfind_included_keyword = "eagle_subfind_tab_030_z000p205"


	# create array of files
	snap_files = EagleFunctions.get_snap_files(snap_directory)

	# pull out simulation parameters that are constant for all snapshots
	box_size = EagleFunctions.read_attribute(snap_files,'Header','BoxSize',include_file_keyword=particles_included_keyword)
	expansion_factor = EagleFunctions.read_attribute(snap_files,'Header','ExpansionFactor',include_file_keyword=particles_included_keyword)
	hubble_param = EagleFunctions.read_attribute(snap_files,'Header','HubbleParam',include_file_keyword=particles_included_keyword)

	# pull out properties of central galaxy
	# can get gal coords from FOF center of mass or median position of stars
	gal_coords = EagleFunctions.read_array(snap_files,"FOF/CentreOfMass",include_file_keyword=group_included_keyword)[gal_num] # zero is to get most massives halo
	
	#star_coords = EagleFunctions.read_array(snap_files,'PartType4/Coordinates',include_file_keyword=particles_included_keyword)
	#gal_coords = np.median(star_coords,axis = 0) # can be used if gal_coords from FOF file is inaccurate
	gal_M200 = EagleFunctions.read_array(snap_files,"FOF/Group_M_Crit200",include_file_keyword=subfind_included_keyword)[gal_num]
	gal_velocity = EagleFunctions.read_array(snap_files,"FOF/Velocity",include_file_keyword=group_included_keyword)[gal_num]
	gal_speed = np.sqrt(gal_velocity[0]**2.0 + gal_velocity[1]**2.0 + gal_velocity[2]**2.0)
	gal_R200 = EagleFunctions.read_array(snap_files,"FOF/Group_R_Crit200",include_file_keyword=subfind_included_keyword)[gal_num]
	radius = radius*gal_R200

	# get necessary gas properties
	gas_coords = EagleFunctions.read_array(snap_files,'PartType0/Coordinates',include_file_keyword=particles_included_keyword)
	gas_coords = EagleFunctions.gal_centered_coords(gas_coords,gal_coords,box_size,expansion_factor,hubble_param)

	gas_mass = EagleFunctions.read_array(snap_files,'PartType0/Mass',include_file_keyword=particles_included_keyword)
	gas_density = EagleFunctions.read_array(snap_files,'/PartType0/Density',include_file_keyword=particles_included_keyword)

	gas_T = EagleFunctions.read_array(snap_files,'PartType0/Temperature',include_file_keyword=particles_included_keyword)
	print np.mean(gas_T)

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
	e_mass_frac = EagleFunctions.read_array(snap_files,'PartType0/ChemistryAbundances',include_file_keyword=particles_included_keyword)[:,0]


	mass_frac_array = np.asarray([H_mass_frac,He_mass_frac,C_mass_frac,Fe_mass_frac,Mg_mass_frac,Ne_mass_frac,N_mass_frac,O_mass_frac,Si_mass_frac,e_mass_frac])

	mu = np.dot(atom_mass_array[None,:],mass_frac_array).ravel()/np.sum(mass_frac_array,axis=0)*atomic_mass_unit
	
	print "-------------------------------"
	print "Index in subfind/FOF array is %.0f" % (gal_num)
	print "Mass within virial radius is %.2e solar masses" % (gal_M200/M_sol)
	print "Galaxy radius is %.2e kpc" % (gal_R200/(1.e3*parsec))
	print "Galaxy velocity is (%.2e,%.2e,%.2e) km/s speed is %.2e km/s" % (gal_velocity[0]/(1.e5),gal_velocity[1]/(1.e5),gal_velocity[2]/(1.e5),gal_speed/(1.e5))
	print "The average mass per particle (mu) is %.2e g" % (np.mean(mu))
	print "-------------------------------"

	gas_volume = gas_mass/gas_density
	gas_number_of_particles = gas_mass/mu
	gas_pressure = (gas_number_of_particles*kB*gas_T)/(gas_volume)

	# get necessary DM properties
	DM_coords = EagleFunctions.read_array(snap_files,'PartType1/Coordinates',include_file_keyword=particles_included_keyword)
	DM_coords = EagleFunctions.gal_centered_coords(DM_coords,gal_coords,box_size,expansion_factor,hubble_param)

	DM_velocity = EagleFunctions.read_array(snap_files,'PartType1/Velocity',include_file_keyword=particles_included_keyword)
	DM_velocity = DM_velocity-gal_velocity
	DM_speed = np.sqrt(DM_velocity[:,0]**2.0+DM_velocity[:,1]**2.0+DM_velocity[:,2]**2.0)

	DM_mass = EagleFunctions.read_attribute(snap_files,'Header','MassTable',include_file_keyword=particles_included_keyword)[1]*(1.e10/hubble_param)*M_sol
	DM_mass = np.zeros(np.size(DM_coords))+DM_mass

	# get properties of gas particles that fall within given radius of central galaxy
	gas_speed_in_R = EagleFunctions.particles_within_R(gas_speed,gas_coords,radius)

	# get properties of DM particles that fall within given radius of central galaxy
	DM_speed_in_R = EagleFunctions.particles_within_R(DM_speed,DM_coords,radius)

	### Find mass contained within certain radii

	radii_pts = 1.e2 # number of pts at which forces are evaluated
	radii = np.linspace(0.01,radius,radii_pts)
	mass_within_radii = np.zeros(np.size(radii))

	for i in range(0,np.size(radii)):
		gas_within_radii = np.sum(EagleFunctions.particles_within_R(gas_mass,gas_coords,radii[i]))
		DM_within_radii = np.sum(EagleFunctions.particles_within_R(DM_mass,DM_coords,radii[i]))
		mass_within_radii[i] = gas_within_radii+DM_within_radii

	###Compare forces on particles

	### Gravity
	gravity_acc = G*mass_within_radii/radii**2.0

	### Pressure supporting force from gas. Need gradient so first bin by radius, take first order deriv. Then use

	P_radii = np.linspace(0,radius,radii_pts+2) # radius bins for pressure, add two when getting deriv of quantity so have same num pts
	P_at_radii = np.zeros(np.size(P_radii)-1) # will be the pressure between each two values in raidus array. 
	# the gradient (first order deriv) between each 2 pts in pressure array 
	P_grad = np.zeros(np.size(P_radii)-1) # Note: will be zero at first pt. because two pts are needed to get gradient
	mass_at_radii = np.zeros(np.size(P_radii)-1) # will be the mass in each concentric shell (same shells pressure is gotten at)
	volume_of_shells = np.zeros(np.size(P_radii)-1) # volume of each shell
	plot_P_radii = np.zeros(np.size(P_radii)-1)
	iteration = 0
	for i in range(0,np.size(P_at_radii)):
		P_at_radii[i] = np.mean(EagleFunctions.particles_btwn_radii(gas_pressure, gas_coords, P_radii[i],P_radii[i+1])) # ave pressure btwn radii
		plot_P_radii[i] = np.mean([P_radii[i],P_radii[i+1]])
		mass_at_radii[i] = np.sum(EagleFunctions.particles_btwn_radii(gas_mass, gas_coords, P_radii[i],P_radii[i+1]))
		volume_of_shells = (4.0*np.pi)/(3.0)*(P_radii[i+1]**3.0 - P_radii[i]**3.0)
		if (iteration > 0):
			P_grad[i] = (P_at_radii[i]-P_at_radii[i-1])/(P_radii[1]-P_radii[0])
		else:
			P_grad[i] = None
		iteration += 1

	P_grad_acc = -1.0*P_grad*(volume_of_shells/mass_at_radii)


	#### Gas Centripetal

	# flip to spherical coordinates to get only tangential speed for centripetal force
	gas_velocity_sph = EagleFunctions.cartesian_to_spherical_velocity(gas_velocity,gas_coords)
	gas_velocity_sph_in_R = EagleFunctions.particles_within_R(gas_velocity_sph,gas_coords,radius) 
	gas_tangential_speed = np.sqrt(gas_velocity_sph[:,1]**2.0+gas_velocity_sph[:,2]**2.0)

	centrip_radii = np.linspace(0.01,radius,radii_pts+1) # add one when looking in shell so we keep radii_pts total num of data pts
	gas_tangential_speed_at_radii = np.zeros(np.size(centrip_radii)-1)
	gas_distance_at_radii = np.zeros(np.size(centrip_radii)-1)
	for i in range(0,np.size(gas_distance_at_radii)):
		gas_tangential_speed_at_radii[i] = np.sqrt(np.mean(np.power(EagleFunctions.particles_btwn_radii(gas_tangential_speed,gas_coords,centrip_radii[i],centrip_radii[i+1]),2)))
		gas_distance_at_radii[i] = (centrip_radii[i]+centrip_radii[i+1])/2.0

	gas_centripetal_acc = gas_tangential_speed_at_radii**2.0/gas_distance_at_radii

	### DM Centripetal

	# flip to spherical coordinates to get only tangential speed for centripetal force
	DM_velocity_sph = EagleFunctions.cartesian_to_spherical_velocity(DM_velocity,DM_coords)
	DM_velocity_sph_in_R = EagleFunctions.particles_within_R(DM_velocity_sph,DM_coords,radius)
	DM_tangential_speed = np.sqrt(DM_velocity_sph[:,1]**2.0+DM_velocity_sph[:,2]**2.0)

	centrip_radii = np.linspace(0.01,radius,radii_pts+1) # add one when looking in shell so we keep radii_pts total num of data pts
	DM_tangential_speed_at_radii = np.zeros(np.size(centrip_radii)-1)
	DM_distance_at_radii = np.zeros(np.size(centrip_radii)-1)
	for i in range(0,np.size(DM_distance_at_radii)):
		DM_tangential_speed_at_radii[i] = np.sqrt(np.mean(np.power(EagleFunctions.particles_btwn_radii(DM_tangential_speed,DM_coords,centrip_radii[i],centrip_radii[i+1]),2)))
		DM_distance_at_radii[i] = (centrip_radii[i]+centrip_radii[i+1])/2.0

	DM_centripetal_acc = DM_tangential_speed_at_radii**2.0/DM_distance_at_radii

	### Virial terms

	KE_radii = np.linspace(0.01,radius,radii_pts)
	gas_KE_in_radii = np.zeros(np.size(KE_radii)) # really will be average KE per particle, per unit mass (used for virial)
	DM_KE_in_radii = np.zeros(np.size(KE_radii))
	gas_T_KE = np.zeros(np.size(KE_radii))

	for i in range(0,np.size(gas_KE_in_radii)):
		#gas_KE_in_radii[i] = 0.5*np.mean(np.power(EagleFunctions.particles_within_R(gas_speed,gas_coords,KE_radii[i]),2))
		if i == 0:
			gas_KE_in_radii[i] = 0
			gas_T_KE[i] = 0
			DM_KE_in_radii[i] = 0
		else:
			gas_KE_in_radii[i] = 0.5*np.mean(np.power(EagleFunctions.particles_btwn_radii(gas_speed,gas_coords,KE_radii[i-1],KE_radii[i]),2))
			### Note: below KE is per unit mass so the units are the same as potential and bulk KE. 
			gas_T_KE[i] = 1.5*kB*np.mean(EagleFunctions.particles_btwn_radii(gas_T, gas_coords, KE_radii[i-1], KE_radii[i]))*(1.0/np.mean(mu))
			DM_KE_in_radii[i] = 0.5*np.mean(np.power(EagleFunctions.particles_btwn_radii(DM_speed,DM_coords,KE_radii[i-1],KE_radii[i]),2))
		#DM_KE_in_radii[i] = 0.5*np.mean(np.power(EagleFunctions.particles_within_R(DM_speed,DM_coords,KE_radii[i]),2))

	gas_virial_LHS = (3.0*G*mass_within_radii)/(5.0*radii)
	gas_virial_RHS = gas_KE_in_radii + gas_T_KE
	gas_virial_ratio = 2.0*gas_virial_RHS/gas_virial_LHS

	DM_virial_LHS = (3.0*G*mass_within_radii)/(2.0*radii)
	DM_virial_RHS = DM_KE_in_radii
	DM_virial_ratio = 2.0*DM_virial_RHS/DM_virial_LHS

	### Get beta (anisotropy)
	gas_beta, DM_beta = EagleFunctions.get_betas(gas_velocity_sph_in_R, DM_velocity_sph_in_R)

	# Gas forces plot
	plt.semilogy(gas_distance_at_radii,gas_centripetal_acc, '.', label = "Centripetal")
	plt.title("Force Balance within %.1e R_vir (%.0f)" % ((radius/gal_R200), gal_num))
	plt.xlabel("Distance")
	plt.ylabel("Acceleration")
	plt.semilogy(radii,gravity_acc,'.',label = "Gravity")
	plt.semilogy(plot_P_radii,P_grad_acc, '.', label = "Pressure")
	plt.legend()
	title = "Gas Force Comparison within %.1e R_vir (%.0f).png" % (radius/gal_R200, gal_num)
	plt.savefig(title, format = 'png')
	plt.close()

	# DM forces plot
	plt.semilogy(DM_distance_at_radii,DM_centripetal_acc, '.', label = "Centripetal")
	plt.title("Force Balance within %.1e R_vir (%.0f)" % ((radius/gal_R200), gal_num))
	plt.xlabel("Distance")
	plt.ylabel("Acceleration")
	plt.semilogy(radii,gravity_acc,'.',label = "Gravity")
	plt.legend()
	title = "DM Force Comparison within %.1e R_vir (%.0f).png" % (radius/gal_R200, gal_num)
	plt.savefig(title, format = 'png')
	plt.close()

	# ratio of virial terms (gas and DM)
	plt.semilogy(radii,gas_virial_ratio,'b.', label = 'Gas')
	plt.semilogy(radii,DM_virial_ratio, 'r.',label = 'DM')
	plt.title('2 KE/ PE (virial) in %.1e R_vir (%.0f)' % ((radius/gal_R200),gal_num))
	plt.xlabel('Distance from Center (cm)')
	plt.ylabel('Ratio (grav/kinetic)')
	plt.legend()
	title = "Virial Ratio (2 KE over PE) within %.1e R_vir (%.0f).png" % (radius/gal_R200, gal_num)
	plt.savefig(title, format = 'png')
	plt.close()

	# compare sources of gravitational and kinetic energy
	plt.semilogy(KE_radii, gas_virial_LHS, '.', label = "Grav")
	plt.semilogy(KE_radii, gas_KE_in_radii, '.', label = "KE of SPH particles")
	plt.semilogy(KE_radii, gas_T_KE, '.', label = "KE from Temp of particles")
	plt.title("Gas Virial Sources Comparison")
	plt.legend()
	title = "Gas Virial Sources Comparison to %.2e virial radii.png" % (radius/gal_R200)
	plt.savefig(title, format = 'png')
	plt.close()

	plt.semilogy(KE_radii, DM_virial_LHS, '.', label = "Grav")
	plt.semilogy(KE_radii, DM_KE_in_radii, '.', label = "KE of SPH particles")
	plt.title("DM Virial Sources Comparison")
	plt.legend()
	title = "DM Virial Sources Comparison to %.2e virial radii.png" % (radius/gal_R200)
	plt.savefig(title, format = 'png')
	plt.close()

	return np.asarray([gal_M200, gal_R200, gal_velocity[0]/(1.e5),gal_velocity[1]/(1.e5),gal_velocity[2]/(1.e5),gal_speed/(1.e5), gas_beta,])

def velocity_anisotropy(snap_directory, # directory where folders for snap files are (in this folder should be files or 
											 # subfolders for snapshots and group/FOF info)

radius, 									 # how far from galactic center should be plotted in virial radii

gal_num,								 # The index of the galaxy you want to look at in the eagle_subfind/FOF array

particles_included_keyword, 				 # string that is included in all snapshots so the right files can be selected
											 # Ex: files are snap_030_z000p205.0.hdf5 etc. => particles_included_keyword = "snap_030_z000p205"

group_included_keyword, 					 # string that is included in all group data files so that right files can be selectted
											 # Ex: files are group_tab_030_z000p205.0.hdf5 etc. => group_included_keyword = "group_tab_030_z000p205"

subfind_included_keyword):					 # string that is included in all eagle_subfind (FOF) files 
											 # Ex: files are eagle_subfind_tab_030_z000p205.0.hdf5 etc. => subfind_included_keyword = "eagle_subfind_tab_030_z000p205"


	# create array of files
	snap_files = EagleFunctions.get_snap_files(snap_directory)

	# pull out simulation parameters that are constant for all snapshots
	box_size = EagleFunctions.read_attribute(snap_files,'Header','BoxSize',include_file_keyword=particles_included_keyword)
	expansion_factor = EagleFunctions.read_attribute(snap_files,'Header','ExpansionFactor',include_file_keyword=particles_included_keyword)
	hubble_param = EagleFunctions.read_attribute(snap_files,'Header','HubbleParam',include_file_keyword=particles_included_keyword)

	# pull out properties of central galaxy
	# can get gal coords from FOF center of mass or median position of stars
	gal_coords = EagleFunctions.read_array(snap_files,"FOF/CentreOfMass",include_file_keyword=group_included_keyword)[gal_num] # zero is to get most massives halo
	
	#star_coords = EagleFunctions.read_array(snap_files,'PartType4/Coordinates',include_file_keyword=particles_included_keyword)
	#gal_coords = np.median(star_coords,axis = 0) # can be used if gal_coords from FOF file is inaccurate
	gal_M200 = EagleFunctions.read_array(snap_files,"FOF/Group_M_Crit200",include_file_keyword=subfind_included_keyword)[gal_num]
	gal_velocity = EagleFunctions.read_array(snap_files,"FOF/Velocity",include_file_keyword=group_included_keyword)[gal_num]
	gal_speed = np.sqrt(gal_velocity[0]**2.0 + gal_velocity[1]**2.0 + gal_velocity[2]**2.0)
	gal_R200 = EagleFunctions.read_array(snap_files,"FOF/Group_R_Crit200",include_file_keyword=subfind_included_keyword)[gal_num]
	radius = radius*gal_R200

	# get necessary gas properties
	gas_coords = EagleFunctions.read_array(snap_files,'PartType0/Coordinates',include_file_keyword=particles_included_keyword)
	gas_coords = EagleFunctions.gal_centered_coords(gas_coords,gal_coords,box_size,expansion_factor,hubble_param)

	gas_mass = EagleFunctions.read_array(snap_files,'PartType0/Mass',include_file_keyword=particles_included_keyword)
	gas_density = EagleFunctions.read_array(snap_files,'/PartType0/Density',include_file_keyword=particles_included_keyword)

	gas_T = EagleFunctions.read_array(snap_files,'PartType0/Temperature',include_file_keyword=particles_included_keyword)

	gas_velocity = EagleFunctions.read_array(snap_files,"/PartType0/Velocity",include_file_keyword=particles_included_keyword)
	gas_velocity = gas_velocity-gal_velocity
	gas_speed = np.sqrt(gas_velocity[:,0]**2.0+gas_velocity[:,1]**2.0+gas_velocity[:,2]**2.0)

# get necessary DM properties
	DM_coords = EagleFunctions.read_array(snap_files,'PartType1/Coordinates',include_file_keyword=particles_included_keyword)
	DM_coords = EagleFunctions.gal_centered_coords(DM_coords,gal_coords,box_size,expansion_factor,hubble_param)

	DM_velocity = EagleFunctions.read_array(snap_files,'PartType1/Velocity',include_file_keyword=particles_included_keyword)
	DM_velocity = DM_velocity-gal_velocity
	DM_speed = np.sqrt(DM_velocity[:,0]**2.0+DM_velocity[:,1]**2.0+DM_velocity[:,2]**2.0)

	DM_mass = EagleFunctions.read_attribute(snap_files,'Header','MassTable',include_file_keyword=particles_included_keyword)[1]*(1.e10/hubble_param)*M_sol
	DM_mass = np.zeros(np.size(DM_coords))+DM_mass

	# get properties of gas particles that fall within given radius of central galaxy
	gas_speed_in_R = EagleFunctions.particles_within_R(gas_speed,gas_coords,radius)

	# get properties of DM particles that fall within given radius of central galaxy
	DM_speed_in_R = EagleFunctions.particles_within_R(DM_speed,DM_coords,radius)

	# flip to spherical coordinates to get only tangential speed for centripetal force
	gas_velocity_sph = EagleFunctions.cartesian_to_spherical_velocity(gas_velocity,gas_coords)
	gas_velocity_sph_in_R = EagleFunctions.particles_within_R(gas_velocity_sph,gas_coords,radius) 

	# flip to spherical coordinates to get only tangential speed for centripetal force
	DM_velocity_sph = EagleFunctions.cartesian_to_spherical_velocity(DM_velocity,DM_coords)
	DM_velocity_sph_in_R = EagleFunctions.particles_within_R(DM_velocity_sph,DM_coords,radius)

	gas_beta, DM_beta = EagleFunctions.get_betas(gas_velocity_sph_in_R, DM_velocity_sph_in_R)
	return np.asarray([gas_beta, DM_beta])

