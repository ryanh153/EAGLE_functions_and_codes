import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import scipy.signal as signal
import scipy.interpolate as interpolate

angs_to_m = 1.e10
c_kms = 3.e5

def convolve_spectra_with_COS_LSF(input_x, input_flux, rest_wavelength, redshift, vel_kms=True, chan=None, directory_with_COS_LSF = './'):
	if directory_with_COS_LSF[-1] != '/':
		directory_with_COS_LSF+= '/'

	if vel_kms:
		input_vel = input_x
	else:
		input_vel = ang_to_vel(input_x, rest_wavelength, redshift)

	observed_lambda = rest_wavelength*(1.+redshift)

	if chan == None:
		if observed_lambda > 1450.:
			chan = 'G160M'
			angstroms_per_pixel = 0.012
		else:
			chan = 'G130M'
			angstroms_per_pixel = 0.0096

	elif ((chan != 'G160M') & (chan != 'G130M')):
		raise ValueError('Improper channel (chan) passed. Acceptable inputs are None, G130M, and G160M, the latter two must be strings!')


	lsf = np.loadtxt('%sCOS_%s_LSF.dat' % (directory_with_COS_LSF, chan), delimiter=',', comments = '#')
	lsf_vels = (lsf[:,0]*angstroms_per_pixel*c_kms)/(observed_lambda)
	if chan == 'G130M':
		lsf_wavelengths = np.array([1150.,1200.,1250.,1300.,1350.,1400.,1450.])
	else:
		lsf_wavelengths = np.array([1450.,1500., 1550., 1600., 1650., 1700., 1750.])
	lsf_index = find_nearest_index(lsf_wavelengths, observed_lambda)+1 # +1 because first column is pixel number
	lsf_to_use = lsf[:,lsf_index]

	lsf_delta_vel = (np.abs(lsf[0,0]-lsf[1,0])*angstroms_per_pixel*c_kms)/(observed_lambda)
	print lsf_delta_vel
	lsf_total_velocity_range = np.max(lsf_vels)-np.min(lsf_vels)

	input_delta_vel = np.abs(input_vel[0]-input_vel[1])
	input_total_vel_range = np.max(input_vel)-np.min(input_vel)

	number_of_input_points = int(input_total_vel_range/lsf_delta_vel)
	resampled_input_flux, resampled_input_vel = signal.resample(1.-input_flux, number_of_input_points, input_vel)
	convolved_line = np.convolve(resampled_input_flux, lsf_to_use, mode='same')
	resampled_input_flux = 1.-resampled_input_flux
	convolved_line = 1.-convolved_line
	wavelengths_convolved = vel_to_ang(resampled_input_vel, rest_wavelength, redshift)

	return resampled_input_vel, wavelengths_convolved,  convolved_line

def convolve_spectra_with_gaussian(input_x, input_flux, std, rest_wavelength, redshift, vel_kms=True, num_sigma_in_gauss=3.):
	
	if vel_kms:
		input_vel = input_x
	else:
		input_vel = ang_to_vel(input_x, rest_wavelength, redshift)
		temp_std_arr = ang_to_vel(np.array([rest_wavelength,rest_wavelength+std]), rest_wavelength, redshift)
		std = np.abs(temp_std_arr[0]-temp_std_arr[1])

	observed_lambda = rest_wavelength*(1.+redshift)

	input_delta_x = np.abs(input_vel[0]-input_vel[1])
	gauss_x = np.arange(-1.*std*num_sigma_in_gauss, std*num_sigma_in_gauss, input_delta_x)
	gauss_y = np.exp((-1.*(gauss_x)**2.)/(2.*std**2.))
	gauss_y /= np.sum(gauss_y)

	if np.size(gauss_y) > np.size(input_vel):
		raise ValueError('gaussian array is larger than line passed, either reduce the standard deviation (std) or the number of sigma used for the gaussian (num_sigma_in_gauss)')

	convolved_line = np.convolve(1.-input_flux, gauss_y, mode='same')
	input_flux = 1.-input_flux
	convolved_line = 1.-convolved_line
	wavelengths_convolved = vel_to_ang(input_vel, rest_wavelength, redshift)

	return input_vel, wavelengths_convolved,  convolved_line

def find_nearest_index(array, value):
	delta = 1.e6
	index = 1.e6
	for i in range(0,np.size(array)):
		if np.abs(array[i]-value) < delta:
			delta = np.abs(array[i]-value)
			index = i 
	return index

def bin_data(input_x,flux,pix_per_bin, rest_wavelength, redshift, vel_kms=True):
	
	if vel_kms:
		vel = input_x
	else:
		vel = ang_to_vel(input_x, rest_wavelength, redshift)

	for i in range(0,pix_per_bin):
		if (np.size(vel)-i) % pix_per_bin == 0:
			num_to_cut = i 
			break
	if num_to_cut == 0:
		binned_vel = np.mean(np.reshape(vel,(-1,pix_per_bin)), axis=1)
		binned_flux = np.mean(np.reshape(flux,(-1,pix_per_bin)), axis=1)
	else:
		vel = vel[0:int(-1.*num_to_cut)]
		flux = flux[0:int(-1.*num_to_cut)]
		binned_vel = np.mean(np.reshape(vel,(-1,pix_per_bin)), axis=1)
		binned_flux = np.mean(np.reshape(flux,(-1,pix_per_bin)), axis=1)

	binned_wavelengths = vel_to_ang(binned_vel, rest_wavelength, redshift)

	return binned_vel, binned_wavelengths, binned_flux

# snr is signal to noise ratio
def add_noise(input_x, input_flux, lambda_line, redshift, snr, pix_per_bin, vel_kms=True, correlated_pixels=False):
	if vel_kms:
		input_vel = input_x
	else:
		input_vel = ang_to_vel(input_x, rest_wavelength, redshift)

	if correlated_pixels:
		snr = snr*pix_per_bin**0.38
	else:
		snr = snr*pix_per_bin**0.5
	noise_vector = np.zeros(np.size(input_flux))
	for i in range(0,np.size(input_flux)):
		noise_vector[i] = np.random.normal(0.,1./snr)

	noisy_flux = input_flux + input_flux*noise_vector
	wavelengths = vel_to_ang(input_vel, lambda_line, redshift)

	return input_vel, wavelengths, noisy_flux

def vel_to_ang(input_vel, lambda_line,redshift):
	observed_lambda = lambda_line*(1.+redshift)
	delta_v = np.abs(input_vel[0]-input_vel[1])
	delta_lambda = (delta_v/c_kms)*(observed_lambda)
	lambda_range = np.size(input_vel)*delta_lambda
	wavelengths = np.linspace(observed_lambda-lambda_range/2.0, observed_lambda + lambda_range/2.0, np.size(input_vel))
	return wavelengths

def ang_to_vel(input_ang, lambda_line, redshift):
	observed_lambda = lambda_line*(1.+redshift)
	delta_ang = np.abs(input_ang[0]-input_ang[1])
	delta_v = (delta_ang*c_kms)/observed_lambda
	vel_range = np.size(input_ang)*delta_v
	velocities = np.linspace(-1.*vel_range/2.0, vel_range/2.0, np.size(input_ang))
	return velocities

def do_it_all(simulated_x, simulated_flux, rest_wavelength, redshift, pix_per_bin, snr, cos_lsf_bool=True, directory_with_COS_LSF='./', correlated_pixels = True, vel_kms=True, chan=None, std=20, num_sigma_in_gauss=3):

	if cos_lsf_bool:
		convolved_vel, convolved_wavelengths, convolved_flux = convolve_spectra_with_COS_LSF(simulated_x, simulated_flux, rest_wavelength, redshift, vel_kms=vel_kms, chan=chan, directory_with_COS_LSF=directory_with_COS_LSF)
	else:
		convolved_vel, convolved_wavelengths, convolved_flux = convolve_spectra_with_gaussian(simulated_x, simulated_flux, std, rest_wavelength, redshift, vel_kms=vel_kms, num_sigma_in_gauss=num_sigma_in_gauss)

	
	vel_kms = True
	binned_vel, binned_wavelengths, binned_flux = bin_data(convolved_vel, convolved_flux, pix_per_bin, rest_wavelength, redshift, vel_kms=vel_kms)

	noisy_vel, noisy_ang, noisy_flux = add_noise(binned_vel, binned_flux, rest_wavelength, redshift, snr, pix_per_bin, vel_kms=vel_kms, correlated_pixels=correlated_pixels)

	return noisy_vel, noisy_ang, noisy_flux






