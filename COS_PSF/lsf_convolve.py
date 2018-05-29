### Imports
import numpy as np
import gen_lsf as lsf_functions
import matplotlib.pyplot as plt
import pyfits

############################################################

### The simulated spectra you are starting with. It can be from anywhere but put it into two arrays
### They should be a 1) normalized flux and 2) velocity in km/s or wavelength in angstroms
fits_file = '/Users/ryanhorton1/Desktop/spectra_3.fits'

hdu_list = pyfits.open(fits_file)
hdu = hdu_list[1]
data = np.array(hdu.data)

simulated_vel = data['velocities']
simulated_flux = data['flux']


# raise ValueError('done')


# eagle_input = np.loadtxt('h1_Spectrum11_0.2_R_vir.txt', delimiter=' ')
# indices = np.argsort(eagle_input[:,0])
# simulated_vel = eagle_input[:,0][indices]
# simulated_ang = lsf_functions.vel_to_ang(simulated_vel, 1216., 0.205)
# simulated_flux = eagle_input[:,1][indices]

############################################################
### All in one: If you want to just do all of these steps with one line of code I made a function for that
### These are the REQUIRED parameters to be passed. Make sure these are correct
rest_wavelength = 1216. # the wavelength of your line in Angstroms in the rest frame 
redshift = 0.205 # the redshift of the line (so the observed wavelength can be calculated)
pix_per_bin = 3 # how many pixels are binned together 
snr = 5. # the signal to noise in each pixel (the final snr will be less if you bin pixels together)

### Optional Parameters: You not pass these to the function at all. The defaults will be used (see below)
### Defaults are
# cos_lsf_bool = True
# directory_with_COS_LSF = './'
# correlated_pixels = True
# vel_kms = True
# chan = None
# std = 20
# num_sigma_in_gauss = 3

cos_lsf_bool = True
directory_with_COS_LSF = './'
correlated_pixels = False
vel_kms = True
chan = None
std = 20
num_sigma_in_gauss = 3
noisy_vel, noisy_ang, noisy_flux = lsf_functions.do_it_all(simulated_vel, simulated_flux, rest_wavelength, redshift, pix_per_bin, snr, cos_lsf_bool=cos_lsf_bool, directory_with_COS_LSF=directory_with_COS_LSF, correlated_pixels = correlated_pixels, vel_kms=vel_kms, chan=chan, std=std, num_sigma_in_gauss=num_sigma_in_gauss)

############################################################
### Examples of plots

def make_plot(x,y, filename,color='b',step_bool=False, ymin=None, ymax=None, xmin=None, xmax=None, x_ticks=None, y_ticks=None, relative_labels_bool=False, aspect_ratio=None, grid_bool=True):
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
		ax.set_aspect(aspect=1./(5.*ax.get_data_ratio()))
	if grid_bool:
		ax.grid()

	ax.set_xlim(xmin,xmax)	
	ax.set_ylim(ymin,ymax)
	ax.ticklabel_format(useOffset=relative_labels_bool)
	plt.savefig(filename)
	plt.close()

make_plot(simulated_vel, simulated_flux, 'simulated_line_vel.png', color='b', step_bool=True, ymin=-0.1, ymax=1.5, xmin=-600, xmax=600,
	      x_ticks = np.arange(-600, 600, 200), y_ticks = np.arange(0.,1.5,0.5), relative_labels_bool=False, aspect_ratio=1./5., grid_bool=True)

make_plot(noisy_vel, noisy_flux, 'noisy_line_vel.png', color='b', step_bool=True, ymin=-0.1, ymax=1.5, xmin=-600, xmax=600,
	      x_ticks = np.arange(-600, 600, 200), y_ticks = np.arange(0.,1.5,0.5), relative_labels_bool=False, aspect_ratio=1./5., grid_bool=True)

make_plot(noisy_ang, noisy_flux, 'noisy_line_ang.png', color='b', step_bool=True, ymin=-0.1, ymax=1.5, xmin=1462, xmax=1468,
	      x_ticks = np.arange(1462, 1468, 1), y_ticks = np.arange(0.,1.5,0.5), relative_labels_bool=False, aspect_ratio=1./5., grid_bool=True)
