### Make spectra surveys for galaxies selected for paper 2
### Ryan Horton 

### Imports
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool as pool
import os
import glob
import h5py

### my libraries
import EagleFunctions
import SpecwizardFunctions

### Constants 
parsec_in_cm = 3.0857e18 # cm
M_sol = 0.398855e33 # g
h = 0.6777

### Infor for galaxies: Directories, group nums, keywords, gal_coords if they are shifted and rotated...
dirs = ["/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_002_x001_eagle.NEQ.snap042_acc/"]
gal_folders = ["snapshot_rot_noneq_047_z000p000_shalo_1_12.28_10.30_1.246/"]
snap_bases = ["snap_rot_noneq_047_z000p000_shalo_1_12.28_10.30_1.246"]
designator = ["data_002_x001"]
keyword_ends = ["047_z000p000"]
group_numbers =  [1]
known_gal_coords = [[25./2., 25./2., 25./2.]] # put zeros in this array if you want to take the gal coords from subfind, otherwise insert here
particles_included_keyword = ["snap_rot_noneq_" + keyword_end for keyword_end in keyword_ends] # these rotated ones have a different naming convention. May do case by case because only a few gals for this paper
group_included_keyword = ["group_tab_" + keyword_end for keyword_end in keyword_ends] # no rotated versions of these files. Not needed? We just get gal coordinates from them... does that change coords? 
subfind_included_keyword = ["eagle_subfind_tab_" + keyword_end for keyword_end in keyword_ends] # TODO ask Ben about above

### survey properties

### Check these for errors in data size or related  issues
points_per_radius = 32
radii_start, radii_stop, radii_step = 20., 55., 5. # stop will be made inclusive
cores = 16 # max number, use this is points per radius is divisable by 16
###

radii =  np.arange(radii_start, radii_stop+radii_step, radii_step) # kpc
points_per_file = points_per_radius*np.size(radii)
axis = np.array([0.,1.,2.])
# pre-create a list of the .hdf5 files that will be made so we can access that directly
# don't want to re-run specwizard all the time

### For specwizard
run_specwizard = False
if run_specwizard:
	print "Running on %s cores. %s sightlines per core" % (str(cores), str(3*np.size(radii)))
	print ''
path_to_param_template = "/cosma/home/analyse/rhorton/Ali_Spec_src/CGM_template.par"
run_output_dir = "/cosma/home/analyse/rhorton/Ali_Spec_src/"
path_to_specwizard_executable = "/cosma/home/analyse/rhorton/Ali_Spec_src/specwizard"

### For data binning

### Check these for errors in data size or related  issues
bin_stagger = 0.25 # so that we don't count things on both sides of a bin. Ex: some radii at 30 are read as  29.997 and some at 30.012
radii_step = 10. # stops are made inclusive. Use same start/stop as above
angle_start, angle_stop, angle_step = 0., 360., 45. # stop is made inclusive
###
radii_bins_for_data = np.arange(radii_start, radii_stop+radii_step, radii_step)
angle_bins_for_data = np.arange(angle_start, angle_stop+angle_step, angle_step)
plot_radii_bins = np.arange(radii_start+radii_step/2., radii_stop+radii_step*1.5, radii_step)
plot_angule_bins = np.arange(angle_start+angle_step/2., angle_stop+angle_stop*1.5, angle_step)

### plotting params
plt.rcParams["axes.labelsize"], plt.rcParams["axes.titlesize"], plt.rcParams["legend.fontsize"], plt.rcParams["xtick.labelsize"],  plt.rcParams["ytick.labelsize"] = 14., 18., 12., 12., 12.

### Given an array of radii and a number of points per radius (and an axis) creates los in cricles around given axis at equally spaced angles
def create_mult_los_per_radius_text(filename, snap_files, gal_coords, box_size, points_per_radius, radii, axis, center=False): # for axis 0=x, 1=y, 2=z, center means put a line through the galaxy's center

	if center:
		points = points_per_radius*np.size(radii) + 1
	else:
		points = points_per_radius*np.size(radii)

	with open(filename,'w') as file:
		file.write("     " + str(points) + '\n')
		if center:
			file.write("%.6f %.6f %.6f %d\n" % (gal_coords[0]/box_size, gal_coords[1]/box_size, gal_coords[2]/box_size, int(axis)))

		for i in range(0,np.size(radii)):
			for j in range(0,points_per_radius):
				radius = radii[i]*1.e3*parsec_in_cm # move to cm
				theta = (j*2.0*np.pi)/(points_per_radius)
				# print axis
				# # print radius/(1.e3*parsec_in_cm)
				# print theta*(180/np.pi) # TODO consider what the +180 really does. Loses meaning of degrees? Get an understanding of that the angles really mean dude
				# print ''

				x,y,z = negative_coords_check(gal_coords,axis,box_size,radius,theta)
				# print ''
				check_radius, check_angle = get_radius_and_angle_of_line(np.array([x,y,z]), gal_coords/box_size, axis)
				# print check_radius*box_size/(1.e3*parsec_in_cm)
				# print check_angle
				# print "%5f %.5f %.5f" % (x, y, z)
				# print ''
				# print "we are writing"
				# print "%.6f %.6f %.6f %d\n" % (x,y,z,int(axis))
				# print ""
				file.write("%.6f %.6f %.6f %d\n" % (x,y,z,int(axis)))
		file.close()

### if a coordinate goes outside the box, wrap it around
def negative_coords_check(gal_coords,axis,box_size,radius,theta):
	if axis == 0:
		x = gal_coords[0]/box_size
		y = (gal_coords[1]+np.cos(theta)*radius)/box_size
		z = (gal_coords[2]+np.sin(theta)*radius)/box_size
		
		x= wrap_around_coords(x)
		y= wrap_around_coords(y)
		z= wrap_around_coords(z)


	if axis ==1:
		x = (gal_coords[0]+np.sin(theta)*radius)/box_size
		y = (gal_coords[1])/box_size
		z = (gal_coords[2]+np.cos(theta)*radius)/box_size
		
		x= wrap_around_coords(x)
		y= wrap_around_coords(y)
		z= wrap_around_coords(z)

	if axis == 2: # z axis, adjust x and y to get circle around z axis
		x = (gal_coords[0]+np.cos(theta)*radius)/box_size
		y = (gal_coords[1]+np.sin(theta)*radius)/box_size
		z = (gal_coords[2]/box_size)

		x= wrap_around_coords(x)
		y= wrap_around_coords(y)
		z= wrap_around_coords(z)

	return x,y,z

def wrap_around_coords(coord):
	if coord < 0:
		coord += 1.0
	if coord > 1:
		coord-= 1.0
	return coord

def get_gal_coords_and_box_size(snap_files, particles_included_keyword, subfind_included_keyword, gal_coords):
	box_size = EagleFunctions.read_attribute(snap_files, "Header", "BoxSize", include_file_keyword = particles_included_keyword)*1.e6*parsec_in_cm/h # bos size is attribute, not automatic cgs conversion. Also scaled by hubble param (hence h)

	if gal_coords == 0:
		p = pool(4)
		GrpIDs_result = p.apply_async(EagleFunctions.read_array, [snap_files, "Subhalo/GroupNumber"], {'include_file_keyword':subfind_included_keyword})
		SubIDs_result = p.apply_async(EagleFunctions.read_array, [snap_files, "Subhalo/SubGroupNumber"], {'include_file_keyword':subfind_included_keyword})
		gal_coords_result = p.apply_async(EagleFunctions.read_array, [snap_files, "Subhalo/CentreOfPotential"], {'include_file_keyword':subfind_included_keyword}) # map center
		p.close()
		GrpIDs_result = p.apply_async(read_array, [snap_files, "Subhalo/GroupNumber"], {'include_file_keyword':subfind_included_keyword})
		SubIDs_result = p.apply_async(read_array, [snap_files, "Subhalo/SubGroupNumber"], {'include_file_keyword':subfind_included_keyword})
		gal_coords = p.apply_async(read_array, [snap_files, "Subhalo/CentreOfPotential"], {'include_file_keyword':subfind_included_keyword}) # map center
	else:
		gal_coords = np.array(gal_coords)*(1.e6*parsec_in_cm) # make sure in the same units as box size (which is read out in cm)

	return gal_coords, box_size

### pass the coordinates of the line and galaxy (in coordinates where 0,0,0 is center, 1,1,1 is a vertex)
def get_radius_and_angle_of_line(line, gal, axis):
	delta_x = line[0]-gal[0]
	if ((np.abs(delta_x) > 0.5) & (delta_x < 0)):
		line[0] = line[0] + 1.0
	elif ((np.abs(delta_x) > 0.5) & (delta_x > 0)):
		line[0] = line[0] - 1.0
	delta_x = line[0]-gal[0]

	delta_y = line[1]-gal[1]
	if ((np.abs(delta_y) > 0.5) & (delta_y < 0)):
		line[1] = line[1] + 1.0
	elif ((np.abs(delta_y) > 0.5) & (delta_y > 0)):
		line[1] = line[1] - 1.0
	delta_y = line[1]-gal[1]

	delta_z = line[2]-gal[2]
	if ((np.abs(delta_z) > 0.5) & (delta_z < 0)):
		line[2] = line[2] + 1.0
	elif ((np.abs(delta_z) > 0.5) & (delta_z > 0)):
		line[2] = line[2] - 1.0
	delta_z = line[2]-gal[2]
	
	radius = np.sqrt(delta_x**2.0+delta_y**2.0+delta_z**2.0)

	if (axis == 0):
		theta = np.arctan2(delta_z, delta_y)
	elif (axis == 1):
		theta = np.arctan2(delta_x, delta_z)
	elif (axis == 2):
		theta = np.arctan2(delta_y, delta_x)
	else:
		raise ValueError("axis is not 0,1, or 2 (x,y,z)")
	if theta < 0: # returns -180 - 180. Maps it to 0-2pi like the inputs
		theta = 2.*np.pi+theta

	theta = (theta*180.)/np.pi # degrees relative to axis

	return radius, theta

### Main
if run_specwizard:
	for gal_index in range(0,np.size(dirs)):
		snap_files = EagleFunctions.get_snap_files(dirs[gal_index], particles_included_keyword[gal_index])
		# because particle included keyword is different in rotated snapshots have to grab the groups_ files seperately
		snap_files = np.concatenate((snap_files, EagleFunctions.get_snap_files(dirs[gal_index], group_included_keyword[gal_index])))

		for ax in axis:
			los_filename = run_output_dir + "los_%s_axis_%s.txt" % (designator[gal_index], str(ax))
			gal_coords, box_size = get_gal_coords_and_box_size(snap_files, particles_included_keyword[gal_index], subfind_included_keyword[gal_index], known_gal_coords[gal_index])
			create_mult_los_per_radius_text(los_filename, snap_files, gal_coords, box_size, points_per_radius, radii, axis=ax)
			# continue

			# make the parameter files
			param_keywords = ["datadir","snap_base","los_coordinates_file", "outputdir"] # replace lines in the template with these keywords 
			param_replacements = [None]*np.size(param_keywords)                          # sub in...
			param_replacements[0] = "datadir = " + dirs[gal_index] + gal_folders[gal_index] + "/"
			param_replacements[1] = "snap_base = " + snap_bases[gal_index]
			param_replacements[2] = "los_coordinates_file = "+los_filename
			param_replacements[3] = 'outputdir = %s' % (run_output_dir)

			param_filename =  run_output_dir+"curr_param_%s_axis_%s.par" % (designator[gal_index], str(ax))
			EagleFunctions.edit_text(path_to_param_template, param_filename, param_keywords, param_replacements)

			# run specwizard (in parallel)
			# can not seperatre module loads and specwizard runs. Different calls of os.system() seem to happen in different environments!
			os.system("module load intel_comp/2018-update2 intel_mpi/2018 hdf5/1.8.20 && mpirun -np %s %s %s" % (str(cores), path_to_specwizard_executable, param_filename)) # make sure all the right modules are installed
			# if 8 lines per core it's about twice as fast. 1 or 2 lines it is slower by like 10%. Not perfeclty uniform though
			# os.system("module load intel_comp/2018-update2 intel_mpi/2018 hdf5/1.8.20 && %s %s" % (path_to_specwizard_executable, param_filename)) # make sure all the right modules are installed

			# store files
			os.system("mv %sspec.%s.0.hdf5 %sspec.%s_axis_%s.hdf5" % (run_output_dir, snap_bases[gal_index], run_output_dir, designator[gal_index], str(ax)))
else:
	for gal_index in range(0,np.size(dirs)):
		snap_files = EagleFunctions.get_snap_files(dirs[gal_index], particles_included_keyword[gal_index])
		# because particle included keyword is different in rotated snapshots have to grab the groups_ files seperately
		snap_files = np.concatenate((snap_files, EagleFunctions.get_snap_files(dirs[gal_index], group_included_keyword[gal_index])))

		gal_coords, box_size = get_gal_coords_and_box_size(snap_files, particles_included_keyword[gal_index], subfind_included_keyword[gal_index], known_gal_coords[gal_index])

### TODO make some intial checks of the files. Plot radius vs column density for example

spec_files = glob.glob(run_output_dir+"spec.*")
num_files = np.size(spec_files)
col_dens_arr = np.zeros(num_files*points_per_file)
radii_arr = np.zeros(num_files*points_per_file)
angle_arr = np.zeros(num_files*points_per_file)
axis_arr = np.zeros(num_files*points_per_file)

print 'reading from the end'
for file_index, file in enumerate(spec_files):
	los_file = run_output_dir+"los_" + file[len(run_output_dir)+5:-4] + "txt"

	with h5py.File(file, 'r') as hf:
		spec_hubble_velocity = np.array(hf.get("VHubble_KMpS"))
		for spec_index in range(points_per_file):
			curr_spec = hf.get("Spectrum%s" % (str(spec_index)))
			curr_h1 = curr_spec.get("h1")
			col_dens_arr[file_index*points_per_file+spec_index] = np.array(curr_h1.get("LogTotalIonColumnDensity"))

	lines = np.genfromtxt(los_file, skip_header=1)
	for line_index, line in enumerate(lines):
		ax = line[3]
		radii_arr[file_index*points_per_file+line_index], angle_arr[file_index*points_per_file+line_index] = get_radius_and_angle_of_line(line, gal_coords/box_size, ax)
		# print ax
		# print angle_arr[file_index*points_per_file+line_index]
		# print ''
		axis_arr[file_index*points_per_file+line_index] = ax

radii_arr = (radii_arr*box_size)/(1.e3*parsec_in_cm) # radii of lines in kpc

### # this is here for when we get fluxes, this will let us get the velocity array to plot it against
# length_spectra = np.size(cal_flux)
# max_box_vel = spec_hubble_velocity[-1]*(length_spectra+1)/(length_spectra)
# spec_hubble_velocity = spec_hubble_velocity - (max_box_vel/2.0) # add in peculiar velocity 
# spec_hubble_velocity = np.where(spec_hubble_velocity > max_box_vel/2.0, spec_hubble_velocity-max_box_vel, spec_hubble_velocity)
# spec_hubble_velocity = np.where(spec_hubble_velocity < (-1.0)*max_box_vel/2.0, spec_hubble_velocity+max_box_vel, spec_hubble_velocity)


### plots, what do we want to know about these sightlines? 
# TODO col dens vs impact parameter and axis
# make the structure you'll want later here. 
# store data by: galaxy, axis, angular bin, radial bin
# want to be able to set the parameters for the later to tailor how many are in each bin. 
# right now have angular resolution of 7.5 degrees and radial resolution of 10 kpc so start with 15 and 30. 6 pts per bin. 
# TODO same for equ widths
# TODO vs centroid velocity
# TODO mass of HI as a function of r along each axis? 
# TODO do that also but only along one angle at a time to probe the cylindrical symmetry
# i.e. Along z axis at 10 degrees off of the disc if I use all those sightlines what cumulative mass do I get. At 20? 30?
# TODO Do those mass estimates again but with total hydrogen

### TODO find a good data structure to represent this. Use the fact that you have control over the sightlines so you can make everything the same length in all dimensions (numpy array!)
### TODO I think the units might not be right for everything. Print the axis, radii_arr and angle_arr
# print axis_arr
# print ''
# print radii_arr
# print ''
# print angle_arr
# print ''
expected_per_bin = (points_per_file*num_files)/(np.size(axis)*(np.size(radii_bins_for_data)-1)*(np.size(angle_bins_for_data)-1))
print expected_per_bin
print ''
data_hypercube = np.zeros((np.size(axis), np.size(radii_bins_for_data)-1, np.size(angle_bins_for_data)-1, expected_per_bin))

for axis_index in range(0,np.size(axis)):
	for radii_index in range(0,np.size(radii_bins_for_data)-1):
		for angle_index in range(0,np.size(angle_bins_for_data)-1):
			# curr_indices = np.where(((axis_arr == axis[axis_index])))
			# print axis[axis_index]
			# print np.size(curr_indices)
			# print ''
			# curr_indices = np.where(((axis_arr == axis[axis_index]) & (radii_arr > radii_bins_for_data[radii_index]-bin_stagger) & (radii_arr < radii_bins_for_data[radii_index+1]-bin_stagger)))
			# print radii_bins_for_data[radii_index]-bin_stagger
			# print radii_bins_for_data[radii_index+1]-bin_stagger
			# print np.size(curr_indices)
			# print ''
			curr_indices = np.where(((axis_arr == axis[axis_index]) & (radii_arr > radii_bins_for_data[radii_index]-bin_stagger) & (radii_arr < radii_bins_for_data[radii_index+1]-bin_stagger) & 
									(angle_arr > angle_bins_for_data[angle_index]-bin_stagger) & (angle_arr < angle_bins_for_data[angle_index+1]-bin_stagger)))
			data_hypercube[axis_index, radii_index, angle_index] = col_dens_arr[curr_indices]
			# print angle_bins_for_data[angle_index]-bin_stagger
			# print angle_bins_for_data[angle_index+1]-bin_stagger
			# print np.size(curr_indices)
			# print ''

# print data_hypercube
# print ''
# ax_indices = [np.where(axis_arr == 0), np.where(axis_arr == 1), np.where(axis_arr == 2)]
# for axis_index in range(0,np.size(axis)):

# 	fig, ax = plt.subplots(1)
# 	ax.scatter(radii_arr[ax_indices[axis_index]], col_dens_arr[ax_indices[axis_index]], s=1)
# 	fig.savefig("col_dens_radius_%s.pdf" % (axis_index))
# 	plt.close(fig)

# 	fig, ax = plt.subplots(1)
# 	ax.scatter(angle_arr[ax_indices[axis_index]], col_dens_arr[ax_indices[axis_index]], s=1)
# 	fig.savefig("col_dens_theta_%s.pdf" % (axis_index))
# 	plt.close(fig)







