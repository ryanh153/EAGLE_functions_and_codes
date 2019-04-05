import gen_lsf
import run_lsf
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# spec_file_base = "/cosma/home/analyse/rhorton/Ali_Spec_src/CIV_Ben_proposal/11.99_x001_data_004_sh1/spec.CIV_prop_massive_axis_"
# halo_mass_str = "11p99"
# to_plot_ax = ["y", "z", "x", "y"]
# to_plot_num = [0, 12, 13, 25]

spec_file_base = "/cosma/home/analyse/rhorton/Ali_Spec_src/CIV_Ben_proposal/10.65_dwarf/spec.CIV_prop_dwarf_axis_"
halo_mass_str = "10p65"
to_plot_ax = ["z", "z"]
to_plot_num = [9, 34]

axis_char = ["x", "y", "z"]
axis = ["0.0", "1.0", "2.0"]

rest_wavelength = 1549. # the wavelength of your line in Angstroms in the rest frame 
redshift = 0.205 # the redshift of the line (so the observed wavelength can be calculated)
pix_per_bin = 3 # how many pixels are binned together 
snr = 9.88 # the signal to noise in each pixel
snr_eff = 9.88*3**(0.38)

for ax in range(np.size(axis)):
	spec_file = spec_file_base + axis[ax] + ".hdf5"
	hf = h5py.File(spec_file, 'r')
	num_specetra = len(hf.keys()) -6 # there are six groups that are not sighlines
	vel = np.array(hf.get("VHubble_KMpS"))
	length_spectra = np.size(vel)
	max_box_vel = vel[-1]*(length_spectra+1)/(length_spectra)
	vel = vel - (max_box_vel/2.0) # add in peculiar velocity 
	vel = np.where(vel > max_box_vel/2.0, vel-max_box_vel, vel)
	vel = np.where(vel < (-1.0)*max_box_vel/2.0, vel+max_box_vel, vel)
	for i in range(36):
		do_line = False
		for j in range(np.size(to_plot_ax)):
			if (((axis_char[ax] == to_plot_ax[j]) & (i == to_plot_num[j]))):
				do_line = True
				break
		if do_line:
			spec = hf.get("Spectrum%s" % (i))
			c4 = spec.get("c4")
			flux = np.array(c4.get("Flux"))
			col = np.array(c4.get("LogTotalIonColumnDensity"))
			noisy_vel, noisy_ang, noisy_flux = gen_lsf.do_it_all(vel, flux, rest_wavelength, redshift, pix_per_bin, snr, cos_lsf_bool=True, vel_kms=True, chan=None, std=None,
																	 num_sigma_in_gauss=None, directory_with_COS_LSF = '/cosma/home/analyse/rhorton/snapshots/COS_PSF')

			run_lsf.make_plot(vel, flux, 'simulated_line_vel_%s_%s_%s.pdf' % (halo_mass_str, i, ax), title=r'Simulated CIV Spectrum: ${\rm log_{10}}(N_{CIV}) = %.1f$ ${\rm (cm^{-2})}$ ' % (col), color='b', step_bool=True, ylabel='Normalized Flux', ymin=-0.1, ymax=1.5, xlabel='Velocity (km/s)', xmin=-280, xmax=280,
			      x_ticks = np.arange(-200,300,100), y_ticks = np.arange(0.0,3.0,1.0), relative_labels_bool=False, aspect_ratio=6./27., grid_bool=False)

			run_lsf.make_plot(noisy_vel, noisy_flux, 'binned_line_vel_%s_%s_%s.pdf' % (halo_mass_str, i, ax), title='Mock COS CIV Spectrum with S/N=%.0f' % (snr_eff), color='b', step_bool=True, ylabel='Normalized Flux', ymin=-0.1, ymax=1.5, xlabel='Velocity (km/s)', xmin=-280, xmax=280,
			      x_ticks = np.arange(-200,300,100), y_ticks = np.arange(0.0,3.0,1.0), relative_labels_bool=False, aspect_ratio=6./27., grid_bool=False)



