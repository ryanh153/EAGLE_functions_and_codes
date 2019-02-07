### Imports
import numpy as np
import gen_lsf as lsf_functions
import matplotlib.pyplot as plt
import pyfits
import sys

############################################################

### The simulated spectra you are starting with. It can be from anywhere but put it into two arrays
### They should be a 1) normalized flux and 2) in velocity units of km/s

### Reading from .fits file
# fits_file = '/Users/ryanhorton1/Desktop/spectra_3.fits'

# hdu_list = pyfits.open(fits_file)
# hdu = hdu_list[1]
# data = np.array(hdu.data)

# simulated_vel = data['velocities']
# rest_wavelength = 1216.
# redshift = 0.205
# simulated_ang = lsf_functions.vel_to_ang(simulated_vel, rest_wavelength, redshift)
# simulated_flux = data['flux']

plt.rcParams['axes.labelsize'], plt.rcParams['axes.titlesize'], plt.rcParams['legend.fontsize'], plt.rcParams['xtick.labelsize'], plt.rcParams['ytick.labelsize'] = 18., 18., 16., 16, 16

### Reading from .txt file (my own format)
eagle_input = np.loadtxt('/Users/ryanhorton1/Documents/bitbucket/opp_research/snapshots/scripts/COS_PSF/h1_Spectrum0_0.1_R_vir.txt', delimiter=' ')
indices = np.argsort(eagle_input[:,0])
simulated_vel = eagle_input[:,0][indices]
rest_wavelength = 1216.
redshift = 0.205
simulated_ang = lsf_functions.vel_to_ang(simulated_vel, rest_wavelength, redshift)
simulated_flux = eagle_input[:,1][indices]

############################################################
### Convolving input spectra with COS LSF (of gaussian LSF)

### These are the REQUIRED parameters to be passed. Make sure these are correct
rest_wavelength = 1216. # the wavelength of your line in Angstroms in the rest frame 
redshift = 0.205 # the redshift of the line (so the observed wavelength can be calculated)

### Optional parameters (can be passed in any order, but must specify variable name)
### if any of these are not passed to the funciton they have a default (listed in the comments for each one)

### What x units are we using?
vel_kms = True # default: True if false, assume the x input is in angstroms

### Pick specifics of COS LSF
directory_with_COS_LSF = '/Users/ryanhorton1/Documents/bitbucket/opp_research/snapshots/scripts/COS_PSF/'
chan = None # default: None Specifies the channel. IF None it selects based on OBSERVED wavelength (<=1450 -> G130M, else -> G160M). Other inputs will return error!

convolved_vel, convolved_wavelengths, convolved_flux=lsf_functions.convolve_spectra_with_COS_LSF(simulated_vel, simulated_flux, rest_wavelength, redshift, vel_kms=vel_kms, chan=chan, directory_with_COS_LSF=directory_with_COS_LSF)

############################################################
### Convolving input spectra with gaussian

### These are the REQUIRED parameters to be passed. Make sure these are correct
rest_wavelength = 1216. # the wavelength of your line in Angstroms in the rest frame 
redshift = 0.205 # the redshift of the line (so the observed wavelength can be calculated)
std = 0.05 # this is in km/s

### Optional parameters (can be passed in any order, but must specify variable name)
### if any of these are not passed to the funciton they have a default (listed in the comments for each one)

### What x units are we using?
vel_kms = True # default: True if false, assume the x input is in angstroms
num_sigma_in_gauss = 3. # default: 3

gauss_vel, gauss_wavelengths, gauss_flux=lsf_functions.convolve_spectra_with_gaussian(simulated_vel, simulated_flux, std, rest_wavelength, redshift, vel_kms=vel_kms, num_sigma_in_gauss=num_sigma_in_gauss)

############################################################
### Binning pixels

### These are the REQUIRED parameters to be passed. Make sure these are correct
rest_wavelength = 1216. # the wavelength of your line in Angstroms in the rest frame 
redshift = 0.205 # the redshift of the line (so the observed wavelength can be calculated)
pix_per_bin = 3 # how many pixels are binned together 

### Optional parameters (can be passed in any order, but must specify variable name)
### if any of these are not passed to the funciton they have a default (listed in the comments for each one)

### What x units are we using?
vel_kms = True # default: True if false, assume the x input is in angstroms

binned_vel, binned_wavelengths, binned_flux = lsf_functions.bin_data(convolved_vel, convolved_flux, pix_per_bin, rest_wavelength, redshift, vel_kms=vel_kms)

############################################################
### Add noise

### These are the REQUIRED parameters to be passed. Make sure these are correct
rest_wavelength = 1216. # the wavelength of your line in Angstroms in the rest frame 
redshift = 0.205 # the redshift of the line (so the observed wavelength can be calculated)
snr = 10. # the signal to noise in each pixel (the final snr will be less if you bin pixels together)
pix_per_bin = 3 # how many pixels are binned together 

### Optional parameters (can be passed in any order, but must specify variable name)
### if any of these are not passed to the funciton they have a default (listed in the comments for each one)

### What x units are we using?
vel_kms = True # default: True if false, assume the x input is in angstroms

### COS sepecific pixel correlation
correlated_pixels = True # default: False COS has correlated pixel noise and thus you don't reduce S/N as fast as you should when binning multiple 
# pixels. If true use S/N = snr*pix_per_bin**0.38, if false use S/N = snr*pix_per_bin**0.5 in the noise vector that will be added here

noisy_flux = lsf_functions.add_noise(binned_vel, binned_flux, rest_wavelength, redshift, snr, pix_per_bin, vel_kms=vel_kms, correlated_pixels=correlated_pixels)

############################################################
### All in one: If you want to just do all of these steps with one line of code I made a function for that
### Defaults are
### cos_lsf_bool = True
### correlated_pixels = True
### vel_kms = True
### chan = None
### std = 20
### num_sigma_in_gauss = 3

noisy_vel, noisy_ang, noisy_flux = lsf_functions.do_it_all(simulated_vel, simulated_flux, rest_wavelength, redshift, pix_per_bin, snr, cos_lsf_bool=True, correlated_pixels = correlated_pixels, vel_kms=True, chan=None, std=None,
															 num_sigma_in_gauss=None, directory_with_COS_LSF = '/Users/ryanhorton1/Documents/bitbucket/opp_research/snapshots/scripts/COS_PSF/')

############################################################
### Examples of plots

def make_plot(x,y, filename, title=None, color='b',step_bool=False, ylabel=None, ymin=None, ymax=None, xlabel=None, xmin=None, xmax=None, x_ticks=None, y_ticks=None, relative_labels_bool=False, aspect_ratio=None, grid_bool=True):
	if ymin == None:
		ymin = np.min(y)
	if ymax == None:
		ymax = np.max(y)
	if xmin == None:
		xmin = np.min(x)
	if xmax == None:
		xmax = np.max(x)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	if step_bool:
		ax.step(x,y)
	else:
		ax.plot(x,y)

	if np.size(x_ticks) > 0:
		ax.set_xticks(x_ticks)
	if np.size(y_ticks) > 0:
		ax.set_yticks(y_ticks)
	if aspect_ratio!= None:
		ax.set_aspect(aspect=((xmax-xmin)/(ymax-ymin))*aspect_ratio)
	if grid_bool:
		ax.grid()
	ax.yaxis.grid()

	ax.set_xlim(xmin,xmax)	
	ax.set_ylim(ymin,ymax)
	ax.ticklabel_format(useOffset=relative_labels_bool)

	if title != None:
		plt.title(title)
	if xlabel != None:
		plt.xlabel(xlabel)
	if ylabel != None:
		plt.ylabel(ylabel)

	plt.savefig(filename, bbox_inches='tight')
	plt.close()

# make_plot(simulated_vel, simulated_flux, 'simulated_line_vel.pdf', title='Simulated HI Spectra Relative to the Velocity of the Galaxy', color='b', step_bool=True, ylabel='Normalized Flux', ymin=-0.1, ymax=2.0, xlabel='Velocity (km/s)', xmin=-280, xmax=280,
# 	      x_ticks = np.arange(-200,300,100), y_ticks = np.arange(0.0,3.0,1.0), relative_labels_bool=False, aspect_ratio=6./27., grid_bool=False)

# make_plot(convolved_vel, convolved_flux, 'convolved_line_vel.pdf', color='b', step_bool=True, ymin=-0.1, ymax=2.0, xmin=-280, xmax=280,
# 	      x_ticks = np.arange(-200,300,100), y_ticks = np.arange(0.0,3.0,1.0), relative_labels_bool=False, aspect_ratio=6./27., grid_bool=False)

# make_plot(binned_vel, binned_flux, 'binned_line_vel.pdf', color='b', step_bool=True, ymin=-0.1, ymax=2.0, xmin=-280, xmax=280,
# 	      x_ticks = np.arange(-200,300,100), y_ticks = np.arange(0.0,3.0,1.0), relative_labels_bool=False, aspect_ratio=6./27., grid_bool=False)

make_plot(noisy_vel, noisy_flux, 'noisy_line_vel.pdf', title='EAGLE Mock Spectra', color='b', step_bool=True, ylabel='', ymin=-0.1, ymax=2.0, xlabel='', xmin=-280, xmax=280,
	      x_ticks = np.arange(-200,300,100), y_ticks = np.arange(0.0,3.0,1.0), relative_labels_bool=False, aspect_ratio=6./27., grid_bool=False)

# make_plot(simulated_ang, simulated_flux, 'simulated_line_ang.pdf', title='Simulated HI Spectra', color='b', step_bool=True, ylabel='Normalized Flux', ymin=-0.1, ymax=2.0, xlabel='Wavelength (Angstroms)', xmin=1463, xmax=1467,
# 	      x_ticks = np.arange(1463, 1467, 1), y_ticks = np.arange(0.0,3.0,1.0), relative_labels_bool=False, aspect_ratio=6./27., grid_bool=False)

# make_plot(convolved_wavelengths, convolved_flux, 'convolved_line_ang.pdf', title='Normalized Flux vs Wavelength', color='b', step_bool=True, ylabel='Normalized Flux', ymin=-0.1, ymax=2.0, xlabel='Wavelength (Angstroms)', xmin=1463, xmax=1467,
# 	      x_ticks = np.arange(1463, 1467, 1), y_ticks = np.arange(0.0,3.0,1.0), relative_labels_bool=False, aspect_ratio=6./27., grid_bool=False)

# make_plot(gauss_wavelengths, gauss_flux, 'gauss_line_ang.pdf', color='b', step_bool=True, ymin=-0.1, ymax=2.0, xmin=1463, xmax=1467,
# 	      x_ticks = np.arange(1463, 1468, 1), y_ticks = np.arange(0.0,3.0,1.0), relative_labels_bool=False, aspect_ratio=6./27., grid_bool=False)

# make_plot(binned_wavelengths, binned_flux, 'binned_line_ang.pdf', title='Normalized Flux vs Wavelength', color='b', step_bool=True, ylabel='Normalized Flux', ymin=-0.1, ymax=2.0, xlabel='Wavelength (Angstroms)', xmin=1463, xmax=1467,
# 	      x_ticks = np.arange(1463, 1467, 1), y_ticks = np.arange(0.0,3.0,1.0), relative_labels_bool=False, aspect_ratio=6./27., grid_bool=False)

# make_plot(noisy_ang, noisy_flux, 'noisy_line_ang.pdf', title='Convolved HI Spectra with S/N=10', color='b', step_bool=True, ylabel='Normalized Flux', ymin=-0.1, ymax=2.0, xlabel='Wavelength (Angstroms)', xmin=1463, xmax=1467,
# 	      x_ticks = np.arange(1463, 1467, 1), y_ticks = np.arange(0.0,3.0,1.0), relative_labels_bool=False, aspect_ratio=6./27., grid_bool=False)






