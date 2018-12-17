### Make spectra surveys for galaxies selected for paper 2
### Ryan Horton 

### TODO another file. Make maps of HI and H looking outwards from the center. Want the ability to pick an origin and a direction 
### and map the gas as a function of solid angle on the sky. Coldens map with z slice picked carefully to give only above/below is a start. 
### Full options are pick a ray for specwizard to trace or do it myself in eagle. Complicated though because cone grows. Could be really simple
### and do mass contained in cone? Divide the 3d space around a point in N solid angle cones. Get mass (HI and H) in each cone out to different 
### distances

# TODO collapse number of ponts. Can angles be collapsed by facto 4 (one quadrant) or 2? Look at each ang plot
# TODO collapse radii? Try larger bins. There are just too many lines. 
# TODO virialized radii? Probably should...

### Imports
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool as pool
import os
import glob
import h5py
import gen_lsf
import scipy.signal
import bisect

### my libraries
import EagleFunctions
import SpecwizardFunctions

### Constants 
parsec_in_cm = 3.0857e18 # cm
M_sol = 1.99e33 # g
c_kms = 3.0e5 # km/s
m_e = 9.10938356e-28 # g
m_p = 1.6726219e-24 # g
m_H = m_e+m_p
h = 0.6777
lambda_h1 = 1215.67 # AA
rho_bar_norm = 1.88e-29
x_H = 0.752
omega_b = 0.04825

### Infor for galaxies: Directories, group nums, keywords, gal_coords if they are shifted and rotated...
dirs = ["/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_002_x001_eagle.NEQ.snap042_acc/"]
gal_folders = ["snapshot_rot_noneq_047_z000p000_shalo_1_12.28_10.30_1.246/"]
snap_bases = ["snap_rot_noneq_047_z000p000_shalo_1_12.28_10.30_1.246"]
designator = ["data_002_x001"]
keyword_ends = ["047_z000p000"]
group_numbers =  [1]
known_gal_coords = [[25./2., 25./2., 25./2.]] # put zeros in this array if you want to take the gal coords from subfind, otherwise insert here
particles_included_keyword = ["snap_rot_noneq_" + keyword_end for keyword_end in keyword_ends] # these rotated ones have a different naming convention. May do case by case because only a few gals for this paper
group_included_keyword = ["group_tab_" + keyword_end for keyword_end in keyword_ends] 
subfind_included_keyword = ["eagle_subfind_tab_" + keyword_end for keyword_end in keyword_ends]
redshift = 0.0 ### TODO keep an eye to make sure this is still always true!!!

### survey properties

### Check these for errors in data size or related  issues
points_per_radius = 72
radii_start, radii_stop, radii_step = 20., 260., 5 # stop not inclusive
cores = 12 # max number, use this is points per radius is divisable by 16
###

radii =  np.arange(radii_start, radii_stop, radii_step) # kpc
points_per_file = points_per_radius*np.size(radii)
axis = np.array([0.,1.,2.])
axis_letters = np.array(['x', 'y', 'z'])
angle_off = np.array(['y', 'z', 'x'])
covering_frac_vals = np.array([14., 16., 18.])

### For specwizard
run_specwizard = False
making_npz_file = False
if run_specwizard:
	print "Running on %s cores. %s sightlines per core" % (str(cores), str(3*np.size(radii)))
	print ''
path_to_param_template = "/cosma/home/analyse/rhorton/Ali_Spec_src/CGM_template.par"
run_output_dir = "/cosma/home/analyse/rhorton/Ali_Spec_src/"
path_to_specwizard_executable = "/cosma/home/analyse/rhorton/Ali_Spec_src/specwizard"
h1_lookup_file = "/cosma/home/analyse/rhorton/Ali_Spec_src/IonizationTables/HM01G+C+SSH/h1.hdf5"

### For data binning

### Check these for errors in data size or related  issues
bin_stagger = 0.25 # so that we don't count things on both sides of a bin. Ex: some radii at 30 are read as  29.997 and some at 30.012
radii_step = 40. # Use same start/stop as above
angle_start, angle_step, ang_step_2_fold, ang_step_4_fold = 0., 30., 20., 10. # stop not inclusive, end set by which arr used
###
npz_filename = "survey_results.npz"
make_realistic_bool = True
directory_with_COS_LSF = "/cosma/home/analyse/rhorton/snapshots/COS_PSF"
###

angle_stop, ang_stop_2_fold, ang_stop_4_fold = 360.+angle_step, 180.+ang_step_2_fold, 90.+ang_step_4_fold
angle_bins_for_data = np.arange(angle_start, angle_stop, angle_step)
ang_bins_2_fold_for_data = np.arange(angle_start, ang_stop_2_fold, ang_step_2_fold)
ang_bins_4_fold_for_data = np.arange(angle_start, ang_stop_4_fold, ang_step_4_fold)
plot_angles = np.arange(angle_start+angle_step*0.5, angle_stop-angle_step*0.5, angle_step)
plot_ang_2_fold = np.arange(angle_start+ang_step_2_fold*0.5, ang_stop_2_fold-ang_step_2_fold*0.5, ang_step_2_fold)
plot_ang_4_fold = np.arange(angle_start+ang_step_4_fold*0.5, ang_stop_4_fold-ang_step_4_fold*0.5, ang_step_4_fold)

radii_bins_for_data = np.arange(radii_start, radii_stop+radii_step, radii_step)
plot_radii = np.arange(radii_start+radii_step*0.5, radii_stop+radii_step*0.5, radii_step)

angle_plot_stagger = np.linspace((-1.*angle_step)/4., (angle_step)/4., np.size(plot_radii))
radii_plot_stagger = np.linspace((-1.*radii_step)/4., (radii_step)/4., np.size(plot_angles))
radii_plot_stagger = np.linspace((-1.*radii_step)/4., (radii_step)/4., np.size(plot_ang_2_fold))
radii_plot_stagger = np.linspace((-1.*radii_step)/4., (radii_step)/4., np.size(plot_ang_4_fold))

### plotting params
plt.rcParams["axes.labelsize"], plt.rcParams["axes.titlesize"], plt.rcParams["legend.fontsize"], plt.rcParams["xtick.labelsize"],  plt.rcParams["ytick.labelsize"] = 14., 18., 12., 12., 12.

### plot bools
col_rad, col_ang  = True, True
H_col_rad, H_col_ang = True, True
W_rad, W_ang = True, True
vel_rad, vel_ang = True, True
ann_mass_rad, ann_mass_ang = True, True
H_ann_mass_rad, H_ann_mass_ang = True, True
cum_mass, H_cum_mass = True, True
cover_rad, cover_ang = True, True


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

				x,y,z = negative_coords_check(gal_coords,axis,box_size,radius,theta)
				check_radius, check_angle = get_radius_and_angle_of_line(np.array([x,y,z]), gal_coords/box_size, axis)

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

def get_line_kinematics(flux, velocity, temperature, ion_densities, nH, optical_depth, make_realistic_bool, rest_wavelength=None, redshift=None, directory_with_COS_LSF='./'):
	# parameters for making the spectra realistic
	pix_per_bin = 8
	snr = 10.
	cos_lsf_bool=True
	correlated_pixels = True
	if correlated_pixels:
		snr_pow = 0.38
	else:
		snr_pow = 0.5
	real_snr = snr*pix_per_bin**snr_pow
	vel_kms=True
	std=20
	num_sigma_in_gauss=3

	# requirements for identifying a line, want it to be visible after convolved to (so, visible by COS)
	depth_tol = 3./real_snr
	prominence_tol = 3. # how far up a flux must go after min, must raise by 3*f_min/real_snr
	min_val_limit = 1.0-depth_tol
	EAGLE_delta_vel = 0.40
	COS_delta_vel = 2.5
	sim_px_per_cos_px = (COS_delta_vel*pix_per_bin)/(EAGLE_delta_vel)
	seperation_tol = 2*COS_delta_vel*pix_per_bin # km/s, makes it so they must be seperated by twice the width of a bin in COS (trough, not trough, trough. as tight as I can do)
	extra_indices_bool = False

	if make_realistic_bool:
		velocity, wavelengths, flux = gen_lsf.do_it_all(velocity, flux, rest_wavelength, redshift, pix_per_bin, snr, cos_lsf_bool=cos_lsf_bool, correlated_pixels = correlated_pixels, vel_kms=vel_kms, chan=None, std=std, num_sigma_in_gauss=num_sigma_in_gauss, directory_with_COS_LSF=directory_with_COS_LSF)
		### TODO need to resample all the quantities at the same velocities as the flux has been resampled at
		### resample the same bounds as 
		num_pts = np.size(velocity)
		temperature, ion_densities, nH, optical_depth = scipy.signal.resample(temperature, num_pts), scipy.signal.resample(ion_densities, num_pts), scipy.signal.resample(nH, num_pts), scipy.signal.resample(optical_depth, num_pts)

	fig, ax = plt.subplots(1)
	ax.plot(velocity, flux)
	fig.savefig("convolved.pdf", bbox_inches="tight")
	plt.close(fig)

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
			### edit: didn't do much, removed
			# if ((flux[i] < flux[i-1]) & (flux[i] < flux[i+1]) & (flux[i]-np.random.normal(0, flux[i]/(real_snr), 1) < min_val_limit)):
			### Now ensure that the average flux over a velocity width of one COS pixel is below the depth tol. Works better
			if ((flux[i] < flux[i-1]) & (flux[i] < flux[i+1]) & (np.mean(flux[i-int(sim_px_per_cos_px/2.):i+int(sim_px_per_cos_px/2.)]) < min_val_limit)):
				min_indices.append(i)

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
	centroid_vel = np.empty(num_minima)
	temps = np.empty(num_minima)
	line_ion_densities = np.empty(num_minima)
	line_nH = np.empty(num_minima)
	prominence_mask = np.zeros(num_minima) + 1
	FWHM = []

	for i in range(0,np.size(min_indices)):
		centroid_vel[i] = velocity[min_indices[i]]
		depth[i] = 1.-flux[min_indices[i]]

	for i in range(0,num_minima):
		index = min_indices[i]
		# how high does the flux have to climb before next min (i.e. how prominent must the trough be) based on snr at trough (2 sigma) or .1 where signal is low (still want unique trough)
		min_recovery = np.max([flux[index] + prominence_tol*(flux[index]/real_snr), flux[index] + 0.05])
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
					elif depth[i] > depth[i-1]:
						prominence_mask[i-1] = 0

					else:
						prominence_mask[i] = 0
						break

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

		try:
			masked_temperature, masked_optical_depth, masked_ion_densities, masked_nH = temperature[left_index:right_index], optical_depth[left_index:right_index], ion_densities[left_index:right_index], nH[left_index:right_index]
			temps[i] = np.sum(masked_temperature[masked_optical_depth >= 0.01]*masked_optical_depth[masked_optical_depth >= 0.01])/np.sum(masked_optical_depth[masked_optical_depth >= 0.01])
			line_ion_densities[i] = np.sum(masked_ion_densities[masked_optical_depth >= 0.01]*masked_optical_depth[masked_optical_depth >= 0.01])/np.sum(masked_optical_depth[masked_optical_depth >= 0.01])
			line_nH[i] = np.sum(masked_nH[masked_optical_depth >= 0.01]*masked_optical_depth[masked_optical_depth >= 0.01])/np.sum(masked_optical_depth[masked_optical_depth >= 0.01])
		except:
			print 'either left or right index is missing'
			print FWHM
			print ''


	min_indices = min_indices[prominence_mask == 1]
	depth = depth[prominence_mask == 1]
	centroid_vel = centroid_vel[prominence_mask == 1]
	temps = temps[prominence_mask == 1]
	line_ion_densities = line_ion_densities[prominence_mask ==1]
	line_nH = line_nH[prominence_mask == 1]
	num_minima = np.size(min_indices)

	if np.size(FWHM) != np.size(centroid_vel):
		plt.plot(velocity, flux)
		plt.savefig('cause_error.pdf')
		plt.close()
		print 'some kinematic arrays are differe sizes!'
		print centroid_vel
		print np.shape(centroid_vel)
		print ''
		print FWHM
		print np.shape(FWHM)
		print ''

	return num_minima, centroid_vel, np.array(FWHM), depth, temps, line_ion_densities, line_nH

def HI_to_H_col(lookup_file, spec_output_file, spectrum, ion, redshift): 
	tol = 0.3 # only looks at where optical_depth is greater than this value

	# read specwizard file
	with h5py.File(spec_output_file, 'r') as file: 
		curr_spec = file.get(spectrum)
		ion = curr_spec.get(ion)
		optical_depth = np.array(ion.get('OpticalDepth'))
		col_dense = np.array(ion.get('LogTotalIonColumnDensity'))
		vel_space = ion.get('RedshiftSpaceOpticalDepthWeighted')
		temp = np.log10(np.array(vel_space.get("Temperature_K")))
		overdensity = np.array(vel_space.get('OverDensity'))
		n_H = np.log10(overdensity*((rho_bar_norm*h**2.*(1.+redshift)**3.*x_H*omega_b)/(m_H)))

	# Get correction factor and change optical depth array to column density array
	total_optical_depth = np.sum(optical_depth)
	tau_to_col_factor = (10.**col_dense)/total_optical_depth
	col_dense_arr = optical_depth*tau_to_col_factor
	col_dens_arr = optical_depth*tau_to_col_factor

	# initialize arrays
	dense_cords = np.empty(np.size(n_H))
	temp_cords = np.empty(np.size(n_H))
	neutral_fractions_old = np.empty(np.size(n_H))

	# read lookup table
	with h5py.File(lookup_file, 'r') as file:
		ion_bal = np.array(file.get('ionbal'))
		log_dens = np.array(file.get('logd'))
		log_T = np.array(file.get('logt'))
		lookup_redshift = np.array(file.get('redshift'))
		file.close()

	# get nearest redshift table uses (redshift is the same for all points in spectrum so outside of for loop)
	redshift_index = bisect.bisect_left(lookup_redshift, redshift)
	redshift_cord = find_cord_for_interp(lookup_redshift, redshift_index, redshift)
	redshift_cords = np.zeros(np.size(n_H))+redshift_cord

	# get nearest number density and temperature for each point in spectra
	# use that to populate neutral fraction array

	for i in range(0,np.size(n_H)):
		if n_H[i] <= -100:
			dense_index = bisect.bisect_left(log_dens, 2.9)
			dense_cords[i] = find_cord_for_interp(log_dens, dense_index, 2.9)
		else:
			dense_index = bisect.bisect_left(log_dens, n_H[i])
			dense_cords[i] = find_cord_for_interp(log_dens, dense_index, n_H[i])

		if temp[i] <= -100:
			temp_index = bisect.bisect_left(log_T, 1.1)
			temp_cords[i] = find_cord_for_interp(log_T, temp_index, 1.1)
		else:
			temp_index = bisect.bisect_left(log_T, temp[i])
			temp_cords[i] = find_cord_for_interp(log_T, temp_index, temp[i])

		neutral_fractions_old[i] = ion_bal[dense_index][temp_index][redshift_index]

	grid_cords = (np.arange(np.size(log_dens)), np.arange(np.size(log_T)), np.arange(np.size(lookup_redshift)))
	cords_to_sample = np.column_stack((dense_cords, temp_cords, redshift_cords))
	try:
		neutral_fractions = scipy.interpolate.interpn(grid_cords, ion_bal, cords_to_sample, method='linear')
		neutral_fractions_nearest = scipy.interpolate.interpn(grid_cords, ion_bal, cords_to_sample, method='nearest')
	except:
		print np.max(cords_to_sample[:,0])
		print np.min(cords_to_sample[:,0])
		print np.max(cords_to_sample[:,1])
		print np.min(cords_to_sample[:,1])
		print np.max(cords_to_sample[:,2])
		print np.min(cords_to_sample[:,2])
		raise ValueError('interpolation failed')

	# get neutral column density array
	H_col_dens_arr = col_dens_arr/np.where(optical_depth >= tol, neutral_fractions, 1.)

	# get total neutral column 
	# Can use that part to make a temperature cut at the per pixel level of the sightlines
	if np.size(optical_depth[optical_depth >= tol]) > 0:
		H_column = np.log10(np.sum(H_col_dens_arr))
	else:
		H_column = col_dense # changing this to zero did not have an effect

	return H_column

def find_cord_for_interp(array, left_index, value):
	if left_index == 0:
		return 0
	elif left_index == np.size(array):
		return left_index-1
	else:
		delta = np.abs(array[left_index-1] - value)
		grid_spacing = np.abs(array[left_index-1]-array[left_index])
		if left_index - 1 + delta/grid_spacing > 82:
			print value
			print ''
			print array
			print ''
			print left_index
			print ''
			print delta
			print ''
			print grid_spacing
			print ''

		return left_index - 1 + delta/grid_spacing

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

if making_npz_file:
	lines_gone_through = 0
	spec_files = glob.glob(run_output_dir+"spec.*")
	num_files = np.size(spec_files)
	file_id_arr, col_dens_arr, H_col_dens_arr, W_arr = np.zeros(num_files*points_per_file), np.zeros(num_files*points_per_file), np.zeros(num_files*points_per_file),  np.zeros(num_files*points_per_file)
	radii_arr, angle_arr, axis_arr = np.zeros(num_files*points_per_file), np.zeros(num_files*points_per_file), np.zeros(num_files*points_per_file)
	line_num_minima_arr, line_centroid_vel_arr, line_FWHM_arr, line_depth_arr, line_temps_arr, line_ion_densities_arr, line_nH_arr = [], [], [], [], [], [], []
	specwizard_velocity_list = [[] for _ in xrange(num_files)]

	for file_index, file in enumerate(spec_files):
		los_file = run_output_dir+"los_" + file[len(run_output_dir)+5:-4] + "txt"
		lines = np.genfromtxt(los_file, skip_header=1)

		with h5py.File(file, 'r') as hf:

			for spec_index in range(points_per_file):
				curr_spec = hf.get("Spectrum%s" % (str(spec_index)))
				curr_h1 = curr_spec.get("h1")
				flux = np.array(curr_h1.get("Flux"))
				optical_depth = np.array(curr_h1.get('OpticalDepth'))
				optical_depth_weighted = curr_h1.get('RedshiftSpaceOpticalDepthWeighted')
				temperature = np.array(optical_depth_weighted.get('Temperature_K'))
				ion_densities = np.array(optical_depth_weighted.get('NIon_CM3'))
				overdensity = np.array(optical_depth_weighted.get('OverDensity'))
				nH = overdensity*((rho_bar_norm*h**2.*(1.+redshift)**3.*x_H*omega_b)/(m_H))

				col_dens_arr[file_index*points_per_file+spec_index] = np.array(curr_h1.get("LogTotalIonColumnDensity"))
				H_col_dens_arr[file_index*points_per_file+spec_index] = HI_to_H_col(h1_lookup_file, file, "Spectrum%s" % (str(spec_index)), "h1", redshift)

				if spec_index==0:
					spec_hubble_velocity = np.array(hf.get("VHubble_KMpS"))
					delta_v = np.abs(spec_hubble_velocity[1]-spec_hubble_velocity[0])
					length_spectra = np.size(flux)
					max_box_vel = spec_hubble_velocity[-1]*(length_spectra+1)/(length_spectra)
					specwizard_velocity = spec_hubble_velocity - (max_box_vel/2.0) # add in peculiar velocity 
					specwizard_velocity = np.where(specwizard_velocity > max_box_vel/2.0, specwizard_velocity-max_box_vel, specwizard_velocity)
					specwizard_velocity = np.where(specwizard_velocity < (-1.0)*max_box_vel/2.0, specwizard_velocity+max_box_vel, specwizard_velocity)
					specwizard_velocity_list[file_index] = specwizard_velocity


				#should we put a scrub for low optical depth? It makes a difference
				W_arr[file_index*points_per_file+spec_index] = np.sum(1.-flux)*(delta_v/c_kms)*lambda_h1

				### get full line kinematics
				fig, ax = plt.subplots(1)
				ax.plot(specwizard_velocity, flux)
				fig.savefig("what_line.pdf", bbox_inches="tight")
				plt.close(fig)
				line_num_minima, line_centroid_vel, line_FWHM, line_depth, line_temps, line_ion_densities, line_nH = get_line_kinematics(flux, specwizard_velocity, temperature, ion_densities, nH, optical_depth, make_realistic_bool, lambda_h1, redshift, directory_with_COS_LSF)
				line_num_minima_arr.append(line_num_minima); line_centroid_vel_arr.append(line_centroid_vel); line_FWHM_arr.append(line_FWHM); line_depth_arr.append(line_depth); line_temps_arr.append(line_temps); line_ion_densities_arr.append(line_ion_densities); line_nH_arr.append(line_nH)

				lines_gone_through += 1
				if lines_gone_through % 10 == 0:
					print "Gone through %d out of %d lines" % (lines_gone_through, num_files*points_per_file)
					print ''

		for line_index, line in enumerate(lines):
			ax = line[3]
			radii_arr[file_index*points_per_file+line_index], angle_arr[file_index*points_per_file+line_index] = get_radius_and_angle_of_line(line, gal_coords/box_size, ax)
			axis_arr[file_index*points_per_file+line_index], file_id_arr = ax, file_index

	radii_arr = (radii_arr*box_size)/(1.e3*parsec_in_cm) # radii of lines in kpc
	specwizrd_velocity_array = np.hstack(specwizard_velocity_list)
	expected_per_bin = (points_per_file*num_files)/(np.size(axis)*(np.size(radii_bins_for_data)-1)*(np.size(angle_bins_for_data)-1))

	np.savez(run_output_dir+npz_filename, file_id_arr, axis_arr, radii_arr, angle_arr, col_dens_arr, H_col_dens_arr, W_arr, np.hstack(line_num_minima_arr), np.hstack(line_centroid_vel_arr), np.hstack(line_FWHM_arr), np.hstack(line_depth_arr), np.hstack(line_temps_arr), np.hstack(line_ion_densities_arr), np.hstack(line_nH_arr), specwizrd_velocity_array)
else:
	npz_file = np.load(run_output_dir+npz_filename)
	file_id_arr, axis_arr, radii_arr, angle_arr, col_dens_arr, H_col_dens_arr, W_arr, line_num_minima_arr, line_centroid_vel_arr, line_FWHM_arr, line_depth_arr, line_temps_arr, line_ion_densities_arr, line_nH_arr, specwizrd_velocity_array = npz_file["arr_0"], \
	npz_file["arr_1"], npz_file["arr_2"], npz_file["arr_3"], npz_file["arr_4"], npz_file["arr_5"], npz_file["arr_6"], npz_file["arr_7"], npz_file["arr_8"], npz_file["arr_9"], npz_file["arr_10"], npz_file["arr_11"], npz_file["arr_12"], npz_file["arr_13"], npz_file["arr_14"]
	expected_per_bin = (np.size(col_dens_arr))/(np.size(axis)*(np.size(radii_bins_for_data)-1)*(np.size(angle_bins_for_data)-1))

# map angles to degrees from disk of galaxy, 2 fold if we care about which side (vel for example) 4 fold otherwise
ang_arr_2_fold = np.where(angle_arr >= 180., 180.-(angle_arr%180.), angle_arr)
ang_arr_4_fold = np.where(ang_arr_2_fold >= 90., 90.-(ang_arr_2_fold%90.), ang_arr_2_fold)

dir_size, axis_size, radii_size, angle_size, ang_2_fold_size, ang_4_fold_size = np.size(dirs), np.size(axis), np.size(radii_bins_for_data)-1, np.size(angle_bins_for_data)-1, np.size(ang_bins_2_fold_for_data)-1, np.size(ang_bins_4_fold_for_data)-1
# 4 fold stuff
col_data, col_top, col_bot = np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size))
H_col_data, H_col_top,  H_col_bot =  np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size))
W_data, W_top, W_bot = np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size))
mass_data, mass_top,  mass_bot =  np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size))
cum_mass_data, cum_mass_top,  cum_mass_bot =  np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size))
H_mass_data, H_mass_top,  H_mass_bot =  np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size))
H_cum_mass_data, H_cum_mass_top,  H_cum_mass_bot =  np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size))
# 2 fold stuff
vel_data, vel_top,  vel_bot =  np.zeros((dir_size,axis_size, radii_size, ang_2_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_2_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_2_fold_size))

for i in range(np.size(covering_frac_vals)):
	if covering_frac_vals[i] != np.array([14., 16., 18.])[i]:
		raise ValueError("you changed the covering frac vals in the headers but didn not alter the way the arrays are made in main! Sorry for this not happening automatically...")
covering_fracs = [np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size)), np.zeros((dir_size,axis_size, radii_size, ang_4_fold_size))]


for gal_index in range(0,np.size(dirs)):
	for axis_index in range(0,np.size(axis)):
		for radii_index in range(0,np.size(radii_bins_for_data)-1):
			for angle_index in range(0,np.size(angle_bins_for_data)-1):
				data_indices = (gal_index, axis_index, radii_index, angle_index)

				curr_indices = np.where(((axis_arr == axis[axis_index]) & (radii_arr > radii_bins_for_data[radii_index]-bin_stagger) & (radii_arr < radii_bins_for_data[radii_index+1]-bin_stagger) & 
										(angle_arr > angle_bins_for_data[angle_index]-bin_stagger) & (angle_arr < angle_bins_for_data[angle_index+1]-bin_stagger)))
				if angle_index < ang_2_fold_size: # put quantities here that map to two quadrants (2 fold)
					ang_2_fold_ind = np.where(((axis_arr == axis[axis_index]) & (radii_arr > radii_bins_for_data[radii_index]-bin_stagger) & (radii_arr < radii_bins_for_data[radii_index+1]-bin_stagger) & 
										(ang_arr_2_fold > ang_bins_2_fold_for_data[angle_index]-bin_stagger) & (ang_arr_2_fold < ang_bins_2_fold_for_data[angle_index+1]-bin_stagger)))
				if angle_index < ang_4_fold_size: # put quantities here that map to one quadrant (4 fold)
					ang_4_fold_ind = np.where(((axis_arr == axis[axis_index]) & (radii_arr > radii_bins_for_data[radii_index]-bin_stagger) & (radii_arr < radii_bins_for_data[radii_index+1]-bin_stagger) & 
										(ang_arr_4_fold > ang_bins_4_fold_for_data[angle_index]-bin_stagger) & (ang_arr_4_fold < ang_bins_4_fold_for_data[angle_index+1]-bin_stagger)))
				
				# shouldn't need to move in and of logspace for madian/percentiles because they don't take into account all values. Just the order, which is unchanged by log10
				if angle_index < ang_4_fold_size: # put quantities here that map to one quadrant (4 fold)
					col_data[data_indices], col_top[data_indices], col_bot[data_indices] = np.median(col_dens_arr[ang_4_fold_ind]), np.percentile(col_dens_arr[ang_4_fold_ind], 84.), np.percentile(col_dens_arr[ang_4_fold_ind], 16.)
					H_col_data[data_indices], H_col_top[data_indices], H_col_bot[data_indices] = np.median(H_col_dens_arr[ang_4_fold_ind]), np.percentile(H_col_dens_arr[ang_4_fold_ind], 84.), np.percentile(H_col_dens_arr[ang_4_fold_ind], 16.)
					W_data[data_indices], W_top[data_indices], W_bot[data_indices] = np.median(W_arr[ang_4_fold_ind]), np.percentile(W_arr[ang_4_fold_ind], 84.), np.percentile(W_arr[ang_4_fold_ind], 16.)

					area = np.pi*(radii_bins_for_data[radii_index+1]**2.-radii_bins_for_data[radii_index]**2.)*((angle_bins_for_data[angle_index+1]-angle_bins_for_data[angle_index])/360.)*(parsec_in_cm*1.e3)**2.
					num_lines, curr_cols, H_curr_cols, mass_ests, H_mass_ests = np.size(ang_4_fold_ind), col_dens_arr[ang_4_fold_ind], H_col_dens_arr[ang_4_fold_ind], np.zeros(np.size(ang_4_fold_ind)), np.zeros(np.size(ang_4_fold_ind))
					# get mass by taking into account each line. Getter value/error from remove one sampling
					for rem_line in range(num_lines): # remove one line at a time
						cols = np.power(np.zeros(num_lines-1)+10.,np.concatenate((curr_cols[0:rem_line], curr_cols[rem_line+1:])))
						H_cols = np.power(np.zeros(num_lines-1)+10., np.concatenate((H_curr_cols[0:rem_line], H_curr_cols[rem_line+1:])))
						mass_ests[rem_line] = (np.sum(cols)*area)*(m_H/M_sol)
						H_mass_ests[rem_line] = (np.sum(H_cols)*area)*(m_H/M_sol)

					mass_data[data_indices], mass_top[data_indices], mass_bot[data_indices] = np.median(mass_ests), np.percentile(mass_ests, 84.), np.percentile(mass_ests, 16.)
					H_mass_data[data_indices], H_mass_top[data_indices], H_mass_bot[data_indices] = np.median(H_mass_ests), np.percentile(H_mass_ests, 84.), np.percentile(H_mass_ests, 16.)

					cum_mass_data[data_indices] = np.sum(mass_data[gal_index, axis_index, 0:radii_index+1, angle_index])
					cum_mass_top[data_indices] = cum_mass_data[data_indices]+np.sqrt(np.sum((mass_top[gal_index, axis_index, 0:radii_index+1, angle_index]-mass_data[gal_index, axis_index, 0:radii_index+1, angle_index])**2.))
					cum_mass_bot[data_indices] = cum_mass_data[data_indices]-np.sqrt(np.sum((mass_bot[gal_index, axis_index, 0:radii_index+1, angle_index]-mass_data[gal_index, axis_index, 0:radii_index+1, angle_index])**2.))
					H_cum_mass_data[data_indices] = np.sum(H_mass_data[gal_index, axis_index, 0:radii_index+1, angle_index])
					H_cum_mass_top[data_indices] = H_cum_mass_data[data_indices]+np.sqrt(np.sum((H_mass_top[gal_index, axis_index, 0:radii_index+1, angle_index]-H_mass_data[gal_index, axis_index, 0:radii_index+1, angle_index])**2.))
					H_cum_mass_bot[data_indices] = H_cum_mass_data[data_indices]-np.sqrt(np.sum((H_mass_bot[gal_index, axis_index, 0:radii_index+1, angle_index]-H_mass_data[gal_index, axis_index, 0:radii_index+1, angle_index])**2.))

					for i in range(np.size(covering_frac_vals)):
						covering_fracs[i][data_indices] = float(np.size(np.where(col_dens_arr[ang_4_fold_ind] > covering_frac_vals[i])))/np.size(ang_4_fold_ind)

				if angle_index < ang_2_fold_size:
					vel_data[data_indices], vel_top[data_indices], vel_bot[data_indices] = np.median(line_centroid_vel_arr[ang_2_fold_ind]), np.percentile(line_centroid_vel_arr[ang_2_fold_ind], 84.), np.percentile(line_centroid_vel_arr[ang_2_fold_ind], 16.)



if np.size(col_data[np.where(col_data == 0.0)]) > 0:
	print "Some regions of col_data (4 fold) cube likely not filled."
	print "%d zero elements" % (np.size(col_data[np.where(col_data == 0.0)]))
	print ''
	print "at indices"
	print ''
	print np.where(col_data == 0.0)

### Plots ###

### col vs radii
if col_rad:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for angle_index in range(np.size(plot_ang_4_fold)):
				ax.errorbar(plot_radii+radii_plot_stagger[angle_index], col_data[gal_index, axis_index,:,angle_index], label = str(plot_ang_4_fold[angle_index]),
							yerr=[col_data[gal_index, axis_index,:,angle_index] - col_bot[gal_index, axis_index,:,angle_index], col_top[gal_index, axis_index,:,angle_index] - col_data[gal_index, axis_index,:,angle_index]])
			ax.hold(False)
			ax.legend(ncol=2, loc="upper right")
			ax.set_ylim([13.,21.])
			ax.set_xlabel("Impact Parameter (kpc)")
			ax.set_ylabel(r"${\rm log_{10}}(N_{HI})$  ${\rm cm^{-2}}$")
			ax.set_title(r"$N_{HI}$ vs b: Axis=%s, $\theta$ Relative to %s" % (axis_letters[axis_index], angle_off[axis_index]))
			fig.savefig('binned_columns_radius_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

### col vs angle
if col_ang:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for radius_index in range(np.size(plot_radii)):
				ax.errorbar(plot_ang_4_fold+angle_plot_stagger[radius_index], col_data[gal_index, axis_index, radius_index, :], label = str(plot_radii[radius_index]), 
							yerr = [col_data[gal_index, axis_index, radius_index, :] - col_bot[gal_index, axis_index, radius_index, :], col_top[gal_index, axis_index, radius_index, :] - col_data[gal_index, axis_index, radius_index, :]])
			ax.hold(False)
			ax.legend(ncol=4, loc="upper center")
			ax.set_ylim([13.,21.])
			ax.set_xlabel(r"$\theta$ Relative to %s Axis (degrees)" % (angle_off[axis_index]))
			ax.set_ylabel(r"${\rm log_{10}}(N_{HI})$  ${\rm cm^{-2}}$")
			ax.set_title(r"$N_{HI}$ vs $\theta$: Axis=%s" % (axis_letters[axis_index]))
			fig.savefig('binned_columns_angle_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

### total H col vs radii
if H_col_rad:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for angle_index in range(np.size(plot_angles)):
				ax.errorbar(plot_radii+radii_plot_stagger[angle_index], H_col_data[gal_index, axis_index,:,angle_index], label = str(plot_angles[angle_index]),
							yerr=[H_col_data[gal_index, axis_index,:,angle_index] - H_col_bot[gal_index, axis_index,:,angle_index], H_col_top[gal_index, axis_index,:,angle_index] - H_col_data[gal_index, axis_index,:,angle_index]])
			ax.hold(False)
			ax.legend(ncol=2, loc="lower left")
			ax.set_ylim([12.,21.])
			ax.set_xlabel("Impact Parameter (kpc)")
			ax.set_ylabel(r"${\rm log_{10}}(N_{H})$  ${\rm cm^{-2}}$")
			ax.set_title(r"$N_{H}$ vs b: Axis=%s, $\theta$ Relative to %s" % (axis_letters[axis_index], angle_off[axis_index]))
			fig.savefig('binned_H_columns_radius_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

### total H col vs angle
if H_col_ang:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for radius_index in range(np.size(plot_radii)):
				ax.errorbar(plot_angles+angle_plot_stagger[radius_index], H_col_data[gal_index, axis_index, radius_index, :], label = str(plot_radii[radius_index]), 
							yerr = [H_col_data[gal_index, axis_index, radius_index, :] - H_col_bot[gal_index, axis_index, radius_index, :], H_col_top[gal_index, axis_index, radius_index, :] - H_col_data[gal_index, axis_index, radius_index, :]])
			ax.hold(False)
			ax.legend(ncol=4, loc="lower center")
			ax.set_ylim([12.,21.])
			ax.set_xlabel(r"$\theta$ Relative to %s Axis (degrees)" % (angle_off[axis_index]))
			ax.set_ylabel(r"${\rm log_{10}}(N_{H})$  ${\rm cm^{-2}}$")
			ax.set_title(r"$N_{H}$ vs $\theta$: Axis=%s" % (axis_letters[axis_index]))
			fig.savefig('binned_H_columns_angle_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

### W vs radii
if W_rad:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for angle_index in range(np.size(plot_angles)):
				ax.errorbar(plot_radii+radii_plot_stagger[angle_index], W_data[gal_index, axis_index,:,angle_index], label = str(plot_angles[angle_index]),
							yerr=[W_data[gal_index, axis_index,:,angle_index] - W_bot[gal_index, axis_index,:,angle_index], W_top[gal_index, axis_index,:,angle_index] - W_data[gal_index, axis_index,:,angle_index]])
			ax.hold(False)
			ax.legend(ncol=2, loc="upper right")
			ax.set_ylim([0.,1.5])
			ax.set_xlabel("Impact Parameter (kpc)")
			ax.set_ylabel(r"${\rm log_{10}}(N_{HI})$  ${\rm cm^{-2}}$")
			ax.set_title(r"N vs b: Axis=%s, $\theta$ Relative to %s" % (axis_letters[axis_index], angle_off[axis_index]))
			fig.savefig('binned_Ws_radius_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

### W vs angle
if W_ang:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for radius_index in range(np.size(plot_radii)):
				ax.errorbar(plot_angles+angle_plot_stagger[radius_index], W_data[gal_index, axis_index, radius_index, :], label = str(plot_radii[radius_index]), 
							yerr = [W_data[gal_index, axis_index, radius_index, :] - W_bot[gal_index, axis_index, radius_index, :], W_top[gal_index, axis_index, radius_index, :] - W_data[gal_index, axis_index, radius_index, :]])
			ax.hold(False)
			ax.legend(ncol=4, loc="upper center")
			ax.set_ylim([0.,1.5])
			ax.set_xlabel(r"$\theta$ Relative to %s Axis (degrees)" % (angle_off[axis_index]))
			ax.set_ylabel(r"${\rm log_{10}}(N_{HI})$  ${\rm cm^{-2}}$")
			ax.set_title(r"N vs $\theta$: Axis=%s" % (axis_letters[axis_index]))
			fig.savefig('binned_Ws_angle_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

### vel vs radii
if vel_rad:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for angle_index in range(np.size(plot_angles)):
				ax.errorbar(plot_radii+radii_plot_stagger[angle_index], vel_data[gal_index, axis_index,:,angle_index], label = str(plot_angles[angle_index]),
							yerr=[vel_data[gal_index, axis_index,:,angle_index] - vel_bot[gal_index, axis_index,:,angle_index], vel_top[gal_index, axis_index,:,angle_index] - vel_data[gal_index, axis_index,:,angle_index]])
			ax.hold(False)
			ax.legend(ncol=4, loc="upper right")
			ax.set_ylim([-200.,300.])
			ax.set_xlabel("Impact Parameter (kpc)")
			ax.set_ylabel(r"Median $v_{centroid}$  ${\rm km/s}$")
			ax.set_title(r"$v_{centroid}$ vs b: Axis=%s, $\theta$ Relative to %s" % (axis_letters[axis_index], angle_off[axis_index]))
			fig.savefig('binned_vels_radius_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

### W vs angle
if vel_ang:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for radius_index in range(np.size(plot_radii)):
				ax.errorbar(plot_angles+angle_plot_stagger[radius_index], vel_data[gal_index, axis_index, radius_index, :], label = str(plot_radii[radius_index]), 
							yerr = [vel_data[gal_index, axis_index, radius_index, :] - vel_bot[gal_index, axis_index, radius_index, :], vel_top[gal_index, axis_index, radius_index, :] - vel_data[gal_index, axis_index, radius_index, :]])
			ax.hold(False)
			ax.legend(ncol=4, loc="upper center")
			ax.set_ylim([-200.,300.])
			ax.set_xlabel(r"$\theta$ Relative to %s Axis (degrees)" % (angle_off[axis_index]))
			ax.set_ylabel(r"Median $v_{centroid}$  ${\rm km/s}$")
			ax.set_title(r"$v_{centroid}$ vs $\theta$: Axis=%s" % (axis_letters[axis_index]))
			fig.savefig('binned_vels_angle_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

### mass vs radii
if ann_mass_rad:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for angle_index in range(np.size(plot_angles)):
				ax.errorbar(plot_radii+radii_plot_stagger[angle_index], mass_data[gal_index, axis_index,:,angle_index], label = str(plot_angles[angle_index]),
							yerr=[mass_data[gal_index, axis_index,:,angle_index] - mass_bot[gal_index, axis_index,:,angle_index], mass_top[gal_index, axis_index,:,angle_index] - mass_data[gal_index, axis_index,:,angle_index]])
			ax.hold(False)
			ax.legend(ncol=2, loc="upper right")
			ax.set_yscale("log")
			ax.set_ylim([10**3.2,10**10.1])
			ax.set_xlabel("Impact Parameter (kpc)")
			ax.set_ylabel(r"$M_{HI, ann}$  $M_{\odot}$")
			ax.set_title(r"$M_{HI, ann}$ vs b: Axis=%s, $\theta$ Relative to %s" % (axis_letters[axis_index], angle_off[axis_index]))
			fig.savefig('binned_mass_radius_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

### mass vs angle
if ann_mass_ang:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for radius_index in range(np.size(plot_radii)):
				ax.errorbar(plot_angles+angle_plot_stagger[radius_index], mass_data[gal_index, axis_index, radius_index, :], label = str(plot_radii[radius_index]), 
							yerr = [mass_data[gal_index, axis_index, radius_index, :] - mass_bot[gal_index, axis_index, radius_index, :], mass_top[gal_index, axis_index, radius_index, :] - mass_data[gal_index, axis_index, radius_index, :]])
			ax.hold(False)
			ax.legend(ncol=4, loc="upper center")
			ax.set_yscale("log")
			ax.set_ylim([10**3.2,10**10.1])
			ax.set_xlabel(r"$\theta$ Relative to %s Axis (degrees)" % (angle_off[axis_index]))
			ax.set_ylabel(r"$M_{HI, ann}$  $M_{\odot}$")
			ax.set_title(r"$M_{HI, ann}$ vs $\theta$: Axis=%s" % (axis_letters[axis_index]))
			fig.savefig('binned_mass_angle_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

### H mass vs radii
if H_ann_mass_rad:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for angle_index in range(np.size(plot_angles)):
				ax.errorbar(plot_radii+radii_plot_stagger[angle_index], H_mass_data[gal_index, axis_index,:,angle_index], label = str(plot_angles[angle_index]),
							yerr=[H_mass_data[gal_index, axis_index,:,angle_index] - H_mass_bot[gal_index, axis_index,:,angle_index], H_mass_top[gal_index, axis_index,:,angle_index] - H_mass_data[gal_index, axis_index,:,angle_index]])
			ax.hold(False)
			ax.legend(ncol=2, loc="upper right")
			ax.set_yscale("log")
			ax.set_ylim([10**8.0,10**10.1])
			ax.set_xlabel("Impact Parameter (kpc)")
			ax.set_ylabel(r"$M_{H, ann}$  $M_{\odot}$")
			ax.set_title(r"$M_{H, ann}$ vs b: Axis=%s, $\theta$ Relative to %s" % (axis_letters[axis_index], angle_off[axis_index]))
			fig.savefig('binned_H_mass_radius_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

### mass vs angle
if H_ann_mass_ang:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for radius_index in range(np.size(plot_radii)):
				ax.errorbar(plot_angles+angle_plot_stagger[radius_index], H_mass_data[gal_index, axis_index, radius_index, :], label = str(plot_radii[radius_index]), 
							yerr = [H_mass_data[gal_index, axis_index, radius_index, :] - H_mass_bot[gal_index, axis_index, radius_index, :], H_mass_top[gal_index, axis_index, radius_index, :] - H_mass_data[gal_index, axis_index, radius_index, :]])
			ax.hold(False)
			ax.legend(ncol=4, loc="upper center")
			ax.set_yscale("log")
			ax.set_ylim([10**8.0,10**10.1])
			ax.set_xlabel(r"$\theta$ Relative to %s Axis (degrees)" % (angle_off[axis_index]))
			ax.set_ylabel(r"$M_{H, ann}$  $M_{\odot}$")
			ax.set_title(r"$M_{H, ann}$ vs $\theta$: Axis=%s" % (axis_letters[axis_index]))
			fig.savefig('binned_H_mass_angle_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

### cumulative HI mass vs radii
if cum_mass:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for angle_index in range(np.size(plot_angles)):
				ax.errorbar(plot_radii+radii_plot_stagger[angle_index], cum_mass_data[gal_index, axis_index,:,angle_index], label = str(plot_angles[angle_index]),
							yerr=[cum_mass_data[gal_index, axis_index,:,angle_index] - cum_mass_bot[gal_index, axis_index,:,angle_index], cum_mass_top[gal_index, axis_index,:,angle_index] - cum_mass_data[gal_index, axis_index,:,angle_index]])
			ax.hold(False)
			ax.legend(ncol=2, loc="lower right")
			ax.set_yscale("log")
			ax.set_ylim([10**5.,10**10.2])
			ax.set_xlabel("Impact Parameter (kpc)")
			ax.set_ylabel(r"$M_{HI, cum}$  $M_{\odot}$")
			ax.set_title(r"$M_{HI, cum}$ vs b: Axis=%s, $\theta$ Relative to %s" % (axis_letters[axis_index], angle_off[axis_index]))
			fig.savefig('binned_cum_mass_radius_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

### cumulative H mass vs radii
if H_cum_mass:
	for gal_index in range(np.size(dirs)):
		for axis_index in range(np.size(axis)):
			fig, ax = plt.subplots(1)
			ax.hold(True)
			for angle_index in range(np.size(plot_angles)):
				ax.errorbar(plot_radii+radii_plot_stagger[angle_index], H_cum_mass_data[gal_index, axis_index,:,angle_index], label = str(plot_angles[angle_index]),
							yerr=[H_cum_mass_data[gal_index, axis_index,:,angle_index] - H_cum_mass_bot[gal_index, axis_index,:,angle_index], H_cum_mass_top[gal_index, axis_index,:,angle_index] - H_cum_mass_data[gal_index, axis_index,:,angle_index]])
			ax.hold(False)
			ax.legend(ncol=2, loc="lower right")
			ax.set_yscale("log")
			ax.set_ylim([10**8.8,10**11.2])
			ax.set_xlabel("Impact Parameter (kpc)")
			ax.set_ylabel(r"$M_{H, cum}$  $M_{\odot}$")
			ax.set_title(r"$M_{H, cum}$ vs b: Axis=%s, $\theta$ Relative to %s" % (axis_letters[axis_index], angle_off[axis_index]))
			fig.savefig('binned_H_cum_mass_radius_%s.pdf' % (axis_letters[axis_index]), bbox_inches="tight")
			plt.close(fig)

for i in range(np.size(covering_frac_vals)):
	if covering_frac_vals[i] != np.array([14., 16., 18.])[i]:
		raise ValueError("you changed the covering frac vals in the headers but didn not alter the way the plots are made/titled! Sorry for this not happening automatically...")

	### covering fracs vs radii
	if cover_rad:
		for gal_index in range(np.size(dirs)):
			for axis_index in range(np.size(axis)):
				fig, ax = plt.subplots(1)
				ax.hold(True)
				for angle_index in range(np.size(plot_angles)):
					ax.plot(plot_radii+radii_plot_stagger[angle_index], covering_fracs[i][gal_index, axis_index,:,angle_index], label = str(plot_angles[angle_index]))
				ax.hold(False)
				ax.legend(ncol=2, loc="lower right")
				# ax.set_yscale("log")
				ax.set_ylim([-0.5,1.05])
				ax.set_xlabel("Impact Parameter (kpc)")
				ax.set_ylabel(r"$f_{cover}$ at ${\rm log_{10}(N_{HI})}$=%d" % (covering_frac_vals[i]))
				ax.set_title(r"$f_{cover}$ at ${\rm log_{10}(N_{HI})}$=%d vs b: Axis=%s, $\theta$ Relative to %s" % (covering_frac_vals[i], axis_letters[axis_index], angle_off[axis_index]))
				fig.savefig('cover_frac_%s_radius_%s.pdf' % (str(covering_frac_vals[i]), axis_letters[axis_index]), bbox_inches="tight")
				plt.close(fig)

	### covering fracs vs angle
	if cover_ang:
		for gal_index in range(np.size(dirs)):
			for axis_index in range(np.size(axis)):
				fig, ax = plt.subplots(1)
				ax.hold(True)
				for radius_index in range(np.size(plot_radii)):
					ax.plot(plot_angles+angle_plot_stagger[radius_index], covering_fracs[i][gal_index, axis_index, radius_index, :], label = str(plot_radii[radius_index]))
				ax.hold(False)
				ax.legend(ncol=4, loc="upper center")
				# ax.set_yscale("log")
				ax.set_ylim([-0.5,1.05])
				ax.set_xlabel(r"$\theta$ Relative to %s Axis (degrees)" % (angle_off[axis_index]))
				ax.set_ylabel(r"$f_{cover}$ at ${\rm log_{10}(N_{HI})}$=%d" % (covering_frac_vals[i]))
				ax.set_title(r"$f_{cover}$ at ${\rm log_{10}(N_{HI})}$=%d vs $\theta$: Axis=%s" % (covering_frac_vals[i], axis_letters[axis_index]))
				fig.savefig('cover_frac_%s_angle_%s.pdf' % (str(covering_frac_vals[i]), axis_letters[axis_index]), bbox_inches="tight")
				plt.close(fig)


print "Done!"







