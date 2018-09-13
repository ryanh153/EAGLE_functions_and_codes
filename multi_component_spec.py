### Take multiple ions in specwizard, divide them into components and make them into one spectrum
### can also convolve with the COS LSF for realistic spectra
### Author: Ryan Horton

### imports
import sys
sys.path.append('COS_PSF/') # add the folder with the cos convolution functions
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import gen_lsf
import scipy.signal as signal

### constants
c_kms = 1.0e5
ions = np.array(['h1', 'c2', 'c3', 'c4', 'n5', 'o1', 'o6', 'o7', 'o8', 'si2', 'si3', 'si4'])
central_lambdas = np.array([1215.6701, 1334.5323, 977.020, 1548.195, 1238.821, 1302.1685, 1031.927, 21.602, 18.969, 1260.422, 1206.500, 1402.770])

### set variables
transitions_file_path = '/gpfs/data/analyse/rhorton/opp_research/snapshots/COS_LSF_files/linelist_i31.v6'
spec_output_file = '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/hubble_prop/spec.snap_hubble_0.hdf5'
los_file = '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/hubble_prop/cos_los_0.txt'
gal_output_file = '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/hubble_prop/output_x001_snap_noneq_056_z000p000_70.hdf5'
COS_LSF_dir = '/gpfs/data/analyse/rhorton/opp_research/snapshots/COS_LSF_files'
output_filename = 'mock_c4_spec.txt'
redshift = 0.000
snr=10.
pix_per_bin=3
ions_to_use = ['c4']

### functions
def get_sim_spectra(spec_output_file, spectrum, ions):
	opt_depths = [None]*np.size(ions)
	with h5py.File(spec_output_file, 'r') as spec_file:
		vel_arr = np.array(spec_file.get('VHubble_KMpS'))
		spectrum = spec_file.get(spectrum)
		for i in range(np.size(ions)):
			curr_ion = spectrum.get(ions[i])
			opt_depths[i] = np.array(curr_ion.get('OpticalDepth'))

	return vel_arr, opt_depths

def find_precision(delta_lambdas):
	delta_wavelength = np.min(np.min(delta_lambdas))
	found_decimal = False
	iterations = 0

	while found_decimal == False:
		if delta_wavelength*10**iterations > 1.:
			precision=iterations
			found_decimal = True
		if iterations > 10.:
			print 'decimal not found!'
			found_decimal = True
		iterations += 1
	return precision

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

### Get line and galaxy properties
arr = np.array([0,9,15,20,26], dtype=int)
for i in arr:
	spectrum = 'Spectrum' + str(i)
	spec_end = str(i)
	spec_line = np.genfromtxt(los_file, skip_header=1)[int(spec_end)]

	with h5py.File(gal_output_file, 'r') as hf:
		galaxy_properties = hf.get('GalaxyProperties')
		gal_coords = np.array(galaxy_properties.get('gal_coords'))[0]
		box_size = np.array(galaxy_properties.get('box_size'))[0]
		halo_mass = np.array(galaxy_properties.get('gal_mass'))[0]
		stellar_mass = np.array(galaxy_properties.get('gal_stellar_mass'))[0]

	gal = np.array([gal_coords[0], gal_coords[1], gal_coords[2]])/box_size
	radius = get_correct_radius_of_line(spec_line, gal)*box_size

	### Main Body
	kept_indices = []
	for i in range(np.size(ions)):
		if ions[i] in ions_to_use:
			kept_indices.append(i)
	ions, central_lambdas = ions[kept_indices], central_lambdas[kept_indices]

	observed_lambdas = central_lambdas*(1.+redshift)
	component_centers, component_strength = [[] for _ in xrange(np.size(ions))], [[] for _ in xrange(np.size(ions))]
	num_components = np.zeros(np.size(ions))

	with open(transitions_file_path, 'r') as transitions_file:
		for line in transitions_file:
			for i in range(np.size(ions)):
				if ions[i].upper() in line:
					line=line.split()
					component_centers[i].append(float(line[0])*(1.+redshift))
					component_strength[i].append(float(line[1]))
				num_components[i] = np.size(component_strength[i])
				
	vel_arr, opt_depths = get_sim_spectra(spec_output_file, spectrum, ions) 
	delta_v = np.abs(vel_arr[0]-vel_arr[1])

	wavelengths, delta_lambdas, temp_opt_depths = [None]*int(np.sum(num_components)), [None]*int(np.sum(num_components)), [None]*int(np.sum(num_components))
	for i in range(np.size(ions)):
		prev_comps = int(np.sum(num_components[0:i]))
		for j in range(int(num_components[i])):
			delta_lambdas[prev_comps+j] = (delta_v/c_kms)*(component_centers[i][j])
			lambda_range = np.size(vel_arr)*delta_lambdas[prev_comps+j]
			wavelengths[prev_comps+j] = np.linspace(component_centers[i][j]-lambda_range/2.0, component_centers[i][j] + lambda_range/2.0, np.size(vel_arr))
			temp_opt_depths[prev_comps+j] = opt_depths[i]*component_strength[i][j]
	opt_depths = temp_opt_depths

	# Combie ions
	precision = find_precision(delta_lambdas)

	for i in range(np.shape(wavelengths)[0]):
		opt_depths[i], wavelengths[i] = signal.resample(opt_depths[i], int((np.max(wavelengths[i]-np.min(wavelengths[i])))/(10.**(-1.*precision))), wavelengths[i])

	output_wavelengths, output_optical_depths = np.concatenate(wavelengths), np.concatenate(opt_depths)
	final_wavelengths = np.arange(np.min(output_wavelengths), np.max(output_wavelengths), 10**(-1*precision))
	final_optical_depth = np.zeros(np.size(final_wavelengths))

	for i in range(np.size(final_wavelengths)):
		indices = np.argwhere(np.abs(output_wavelengths-final_wavelengths[i])<(10**(-1.*precision)))[:,0]
		if np.size(indices) > 0:
			final_optical_depth[i] = np.sum(output_optical_depths[indices])

	final_flux = np.exp(-1.*final_optical_depth)

	convolved_vel1, convolved_wavelength, convolved_flux = gen_lsf.do_it_all(final_wavelengths, final_flux, central_lambdas[0], redshift, \
		pix_per_bin=pix_per_bin, snr=snr, cos_lsf_bool=True, directory_with_COS_LSF=COS_LSF_dir, correlated_pixels=True, vel_kms=False, long_spec=True)

	### output file 
	with open(output_filename, 'w') as file:
		file.write('# halo_mass stellar_mass impact_param\n')
		file.write('# %.3f %.3f %.3f\n' % (np.log10(halo_mass), np.log10(stellar_mass), radius))
		file.write('# simulated_wavelength simulated_flux realistic_wavelength realistic_flux realistic_snr\n')
		real_size = np.size(convolved_wavelength)
		real_snr = snr*pix_per_bin**(0.38)
		for i in range(np.size(final_wavelengths)):
			if i < real_size:
				file.write('%.4f %.4f %.4f %.4f %.4f\n' % (final_wavelengths[i], final_flux[i], convolved_wavelength[i], convolved_flux[i], real_snr))
			else:
				file.write('%.4f %.4f\n' % (final_wavelengths[i], final_flux[i]))

	### plots
	fig, ax = plt.subplots()
	ax.plot(final_wavelengths, final_flux)
	ax.set_title(r'Mock Spectra: c4, $M_{halo}$=%.2f, b=%.0f' % (np.log10(halo_mass), radius))
	ax.set_xlabel(r'Wavelength $(\AA)$')
	ax.set_ylabel('Normalized Flux')
	ax.set_ylim([-0.2,1.2])
	# ax.set_xlim([1050.,1900.])
	ax.set_xlim([1545.,1552.])
	fig.savefig('comb_flux_%s.pdf' % (spec_end))
	plt.close(fig)

	fig, ax = plt.subplots()
	ax.plot(convolved_wavelength, convolved_flux)
	ax.set_title('Realistic Spectra: c4, $M_{halo}$=%.2f, b=%.0f' % (np.log10(halo_mass), radius))
	ax.set_xlabel(r'Wavelength $(\AA)$')
	ax.set_ylabel('Normalized Flux')
	ax.set_ylim([-0.2,1.2])
	# ax.set_xlim([1050.,1900.])
	ax.set_xlim([1545.,1552.])
	fig.savefig('comb_conv_flux_%s.pdf' % (spec_end))
	plt.close(fig)





