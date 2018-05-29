import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as plt3d
import sys
import h5py
import glob
import warnings
import EagleFunctions
import os

### Constants
parsec = 3.0857e18 # cm
G = 6.674e-8
M_sol = 1.98855e33 # g
kB = 1.380648e-16

def plot_spectra_for_mult_ions_at_one_point(ions, spec_file, gal_output_file, max_abs_vel = None, file_base = '.'):
	total_num = 0
	with h5py.File(spec_file,'r') as hf:
		spec_hubble_velocity = np.array(hf.get("VHubble_KMpS"))
		points_per_radius = np.array(hf.get('points_per_radius'))[0]
		radii_arr = np.array(hf.get('radii'))

	length_spectra = np.size(spec_hubble_velocity)
	max_box_vel = spec_hubble_velocity[-1]*(length_spectra+1)/(length_spectra)

	with h5py.File(gal_output_file,'r') as hf:
		GalaxyProperties = hf.get("GalaxyProperties")
		box_size = np.array(GalaxyProperties.get("box_size"))[0]/1.e3 # again to get to Mpc
		H = max_box_vel/box_size
		gal_R200 = np.array(GalaxyProperties.get("gal_R200"))[0]
		gal_M200 = np.array(GalaxyProperties.get("gal_mass"))[0]
		snap_directory = np.array(GalaxyProperties.get("snap_directory"))[0]
		gal_vel = np.array(GalaxyProperties.get("gal_velocity"))[0]
		gal_coords = np.array(GalaxyProperties.get("gal_coords"))[0]/1.e3 # converted to Mpc for hubble constant use later
		gal_hubble_vel = gal_coords[2]*H
		gal_vel_z = gal_vel[2]

	final_vel_arr = spec_hubble_velocity - (gal_hubble_vel+gal_vel_z)
	final_vel_arr = np.where(final_vel_arr > max_box_vel/2.0, final_vel_arr-max_box_vel, final_vel_arr)
	final_vel_arr = np.where(final_vel_arr < (-1.0)*max_box_vel/2.0, final_vel_arr+max_box_vel, final_vel_arr)
	if max_abs_vel != None:
		final_vel_arr_plot = final_vel_arr[np.where(np.abs(final_vel_arr) < max_abs_vel)[0]]
	else: 
		final_vel_arr_plot = final_vel_arr
	length_plotting_arrs = np.size(final_vel_arr_plot)


	for i in range(0,np.size(radii_arr)):
		for ion in ions:
			col_dense_arr = np.array([])
			spectra_arr = np.zeros((points_per_radius,length_plotting_arrs))
			for j in range(0,points_per_radius):
				curr_spectrum = 'Spectrum' + str((i)*points_per_radius + (j+1))
				with h5py.File(spec_file,'r') as hf:
					curr_hdf5_spectrum = hf.get(curr_spectrum)
					hdf5_ion = curr_hdf5_spectrum.get(ion)
					ion_flux = np.array(hdf5_ion.get("Flux"))
					optical_depth = np.array(hdf5_ion.get("OpticalDepth"))
					log_total_column_density = np.array(hdf5_ion.get("LogTotalIonColumnDensity"))
					if np.size(col_dense_arr) > 0:
						col_dense_arr = np.append(col_dense_arr,log_total_column_density)
						total_num += 1.
					else:
						col_dense_arr = np.array(log_total_column_density)
						total_num += 1.

					if max_abs_vel != None:
						ion_flux = ion_flux[np.where(np.abs(final_vel_arr) < max_abs_vel)[0]]
						optical_depth = optical_depth[np.where(np.abs(final_vel_arr) < max_abs_vel)[0]]
					spectra_arr[j,:] = ion_flux
					# if ((i == 1) & (j==0)):
					# 	make_hdf5_output(final_vel_arr_plot, ion_flux, optical_depth, snap_directory, log_total_column_density, curr_spectrum, ion, radii_arr[i], gal_R200, gal_M200)
					# 	make_txt_output(final_vel_arr_plot, ion_flux, optical_depth, snap_directory, log_total_column_density, curr_spectrum, ion, radii_arr[i], gal_R200, gal_M200)
					# else:
					# 	make_hdf5_output(final_vel_arr_plot, ion_flux, optical_depth, snap_directory, log_total_column_density, curr_spectrum, ion, radii_arr[i], gal_R200, gal_M200)
					plt.plot(final_vel_arr_plot,ion_flux,label=str(ion)+' '+str(log_total_column_density))
					break



		plt.title('Stacked Spectra: ' + ' %.1f R_vir (%.1f kpc), M=%.1f' % (radii_arr[i]/gal_R200,radii_arr[i],np.log10(gal_M200)))
		plt.xlabel('Velocity Relative to Central Halo (km/s)')
		plt.ylabel('Flux')
		plt.ylim(ymin = 0, ymax = 1)
		plt.legend()
		plt.savefig('stacked_fig'+'_%.1f_R_vir(%.1f_kpc).png' % (radii_arr[i]/gal_R200,radii_arr[i]))
		plt.close()
		mean_gal_spectra =  np.mean(spectra_arr,axis=0)
		std_gal_spectra = np.std(spectra_arr,axis=0)

		### order arrays so velocities are sorted from lowers to highest (otherwise fill_between will show a discontinuity because of the 
		### the non-sequential values of the x-array)
		sorted_indices = np.argsort(final_vel_arr_plot)
		final_vel_arr_plot_sorted = final_vel_arr_plot[sorted_indices]
		mean_gal_spectra_sorted = mean_gal_spectra[sorted_indices]
		std_gal_spectra_sorted = std_gal_spectra[sorted_indices]


		top_err = np.where(mean_gal_spectra_sorted+std_gal_spectra_sorted>1.0, 1.0, mean_gal_spectra_sorted+std_gal_spectra_sorted)
		bot_err = np.where(mean_gal_spectra_sorted-std_gal_spectra_sorted<0.0, 0.0, mean_gal_spectra_sorted-std_gal_spectra_sorted)
		plt.fill_between(final_vel_arr_plot_sorted,bot_err, top_err, interpolate = False)
		plt.plot(final_vel_arr_plot_sorted,mean_gal_spectra_sorted,'r.')
		plt.title('Mean Spectra(1 sig): ' + str(ion) + ' %.1f R_vir (%.1f kpc) M=%.1f' % (radii_arr[i]/gal_R200,radii_arr[i],np.log10(gal_M200)))
		plt.xlabel('Velocity Relative to Central Halo (km/s)')
		plt.ylabel('Flux')
		plt.ylim(ymin = 0, ymax = 1)
		plt.savefig(str(ion)+'_%.1f_R_vir_%.1f_kpc.png' % (radii_arr[i]/gal_R200,radii_arr[i]))
		plt.close()


def plot_spectra_by_output_radii(ion, spec_file, gal_output_file, max_abs_vel = None, file_base = '.'):
	col_dense_arr = np.array([])
	total_num = 0
	with h5py.File(spec_file,'r') as hf:
		spec_hubble_velocity = np.array(hf.get("VHubble_KMpS"))
		points_per_radius = np.array(hf.get('points_per_radius'))[0]
		radii_arr = np.array(hf.get('radii'))

	length_spectra = np.size(spec_hubble_velocity)
	max_box_vel = spec_hubble_velocity[-1]*(length_spectra+1)/(length_spectra)

	with h5py.File(gal_output_file,'r') as hf:
		GalaxyProperties = hf.get("GalaxyProperties")
		box_size = np.array(GalaxyProperties.get("box_size"))[0]/1.e3 # again to get to Mpc
		H = max_box_vel/box_size
		gal_R200 = np.array(GalaxyProperties.get("gal_R200"))[0]
		gal_M200 = np.array(GalaxyProperties.get("gal_mass"))[0]
		snap_directory = np.array(GalaxyProperties.get("snap_directory"))[0]
		gal_vel = np.array(GalaxyProperties.get("gal_velocity"))[0]
		gal_coords = np.array(GalaxyProperties.get("gal_coords"))[0]/1.e3 # converted to Mpc for hubble constant use later
		gal_hubble_vel = gal_coords[2]*H
		gal_vel_z = gal_vel[2]

	final_vel_arr = spec_hubble_velocity - (gal_hubble_vel+gal_vel_z)
	final_vel_arr = np.where(final_vel_arr > max_box_vel/2.0, final_vel_arr-max_box_vel, final_vel_arr)
	final_vel_arr = np.where(final_vel_arr < (-1.0)*max_box_vel/2.0, final_vel_arr+max_box_vel, final_vel_arr)
	if max_abs_vel != None:
		final_vel_arr_plot = final_vel_arr[np.where(np.abs(final_vel_arr) < max_abs_vel)[0]]
	else: 
		final_vel_arr_plot = final_vel_arr
	length_plotting_arrs = np.size(final_vel_arr_plot)


	for i in range(0,np.size(radii_arr)):
		spectra_arr = np.zeros((points_per_radius,length_plotting_arrs))
		for j in range(0,points_per_radius):
			curr_spectrum = 'Spectrum' + str((i)*points_per_radius + (j+1))
			with h5py.File(spec_file,'r') as hf:
				curr_hdf5_spectrum = hf.get(curr_spectrum)
				hdf5_ion = curr_hdf5_spectrum.get(ion)
				ion_flux = np.array(hdf5_ion.get("Flux"))
				optical_depth = np.array(hdf5_ion.get("OpticalDepth"))
				log_total_column_density = np.array(hdf5_ion.get("LogTotalIonColumnDensity"))
				if np.size(col_dense_arr) > 0:
					col_dense_arr = np.append(col_dense_arr,log_total_column_density)
					total_num += 1.
				else:
					col_dense_arr = np.array(log_total_column_density)
					total_num += 1.

				if max_abs_vel != None:
					ion_flux = ion_flux[np.where(np.abs(final_vel_arr) < max_abs_vel)[0]]
					optical_depth = optical_depth[np.where(np.abs(final_vel_arr) < max_abs_vel)[0]]
				spectra_arr[j,:] = ion_flux
				# if ((i == 1) & (j==0)):
				# 	make_hdf5_output(final_vel_arr_plot, ion_flux, optical_depth, snap_directory, log_total_column_density, curr_spectrum, ion, radii_arr[i], gal_R200, gal_M200)
				# 	make_txt_output(final_vel_arr_plot, ion_flux, optical_depth, snap_directory, log_total_column_density, curr_spectrum, ion, radii_arr[i], gal_R200, gal_M200)
				# else:
				# 	make_hdf5_output(final_vel_arr_plot, ion_flux, optical_depth, snap_directory, log_total_column_density, curr_spectrum, ion, radii_arr[i], gal_R200, gal_M200)
				plt.plot(final_vel_arr_plot,ion_flux,'.')


		plt.title('Stacked Spectra: ' + str(ion) + ' %.1f R_vir (%.1f kpc), M=%.1f' % (radii_arr[i]/gal_R200,radii_arr[i],np.log10(gal_M200)))
		plt.xlabel('Velocity Relative to Central Halo (km/s)')
		plt.ylabel('Flux')
		plt.ylim(ymin = 0, ymax = 1)
		plt.savefig(str(ion)+'_%.1f_R_vir(%.1f_kpc).png' % (radii_arr[i]/gal_R200,radii_arr[i]))
		plt.close()
		mean_gal_spectra =  np.mean(spectra_arr,axis=0)
		std_gal_spectra = np.std(spectra_arr,axis=0)

		### order arrays so velocities are sorted from lowers to highest (otherwise fill_between will show a discontinuity because of the 
		### the non-sequential values of the x-array)
		sorted_indices = np.argsort(final_vel_arr_plot)
		final_vel_arr_plot_sorted = final_vel_arr_plot[sorted_indices]
		mean_gal_spectra_sorted = mean_gal_spectra[sorted_indices]
		std_gal_spectra_sorted = std_gal_spectra[sorted_indices]


		top_err = np.where(mean_gal_spectra_sorted+std_gal_spectra_sorted>1.0, 1.0, mean_gal_spectra_sorted+std_gal_spectra_sorted)
		bot_err = np.where(mean_gal_spectra_sorted-std_gal_spectra_sorted<0.0, 0.0, mean_gal_spectra_sorted-std_gal_spectra_sorted)
		plt.fill_between(final_vel_arr_plot_sorted,bot_err, top_err, interpolate = False)
		plt.plot(final_vel_arr_plot_sorted,mean_gal_spectra_sorted,'r.')
		plt.title('Mean Spectra(1 sig): ' + str(ion) + ' %.1f R_vir (%.1f kpc) M=%.1f' % (radii_arr[i]/gal_R200,radii_arr[i],np.log10(gal_M200)))
		plt.xlabel('Velocity Relative to Central Halo (km/s)')
		plt.ylabel('Flux')
		plt.ylim(ymin = 0, ymax = 1)
		plt.savefig(str(ion)+'_%.1f_R_vir_%.1f_kpc.png' % (radii_arr[i]/gal_R200,radii_arr[i]))
		plt.close()

def plot_spectra_binned(ion, bins, spec_file, gal_output_file, max_abs_vel, file_base = '.'):
	with h5py.File(spec_file,'r') as hf:
		spec_hubble_velocity = np.array(hf.get("VHubble_KMpS"))
		points_per_radius = np.array(hf.get('points_per_radius'))[0]
		radii_arr = np.array(hf.get('radii'))

	length_spectra = np.size(spec_hubble_velocity)
	max_box_vel = spec_hubble_velocity[-1]*(length_spectra+1)/(length_spectra)

	with h5py.File(gal_output_file,'r') as hf:
		GalaxyProperties = hf.get("GalaxyProperties")
		box_size = np.array(GalaxyProperties.get("box_size"))[0]/1.e3 # again to get to Mpc
		gal_R200 = np.array(GalaxyProperties.get("gal_R200"))[0]
		gal_M200 = np.array(GalaxyProperties.get("gal_mass"))[0]
		H = max_box_vel/box_size
		gal_vel = np.array(GalaxyProperties.get("gal_velocity"))[0]
		gal_coords = np.array(GalaxyProperties.get("gal_coords"))[0]/1.e3 # converted to Mpc for hubble constant use later
		gal_hubble_vel = gal_coords[2]*H
		gal_vel_z = gal_vel[2]
		max_box_vel = spec_hubble_velocity[-1]*(length_spectra+1)/(length_spectra)

	final_vel_arr = spec_hubble_velocity - (gal_hubble_vel+gal_vel_z)
	final_vel_arr = np.where(final_vel_arr > max_box_vel/2.0, final_vel_arr-max_box_vel, final_vel_arr)
	final_vel_arr = np.where(final_vel_arr < (-1.0)*max_box_vel/2.0, final_vel_arr+max_box_vel, final_vel_arr)
	if max_abs_vel != None:
		final_vel_arr_plot = final_vel_arr[np.where(np.abs(final_vel_arr) < max_abs_vel)[0]]
	else:
		final_vel_arr_plot =final_vel_arr
	length_plotting_arrs = np.size(final_vel_arr_plot)

	bins = bins*gal_R200

	radii_in_bin_arr = np.zeros(np.size(bins)-1)
	for i in range(0,np.size(bins)-1):
		radii_in_bin = 0

		for j in range(0,np.size(radii_arr)):
			if (radii_arr[j] > bins[i]) & (radii_arr[j] <= bins[i+1]):
				radii_in_bin += 1

			elif (radii_arr[j] > bins[i+1]):
				break

		radii_in_bin_arr[i] = radii_in_bin

	for k in range(0,np.size(bins)-1):
		spectra_arr = np.zeros((radii_in_bin_arr[k]*points_per_radius,length_plotting_arrs))
		radii_plotted = 0

		for i in range(0,np.size(radii_arr)):
			if (radii_arr[i] > bins[k]) & (radii_arr[i] <= bins[k+1]):
				for j in range(0,points_per_radius):
					curr_spectrum = 'Spectrum' + str((i)*points_per_radius + (j+1))
					with h5py.File(spec_file,'r') as hf:
						curr_hdf5_spectrum = hf.get(curr_spectrum)
						hdf5_ion = curr_hdf5_spectrum.get(ion)
						ion_flux = np.array(hdf5_ion.get("Flux"))
						if max_abs_vel != None:
							ion_flux = ion_flux[np.where(np.abs(final_vel_arr) < max_abs_vel)[0]]
						spectra_arr[radii_plotted*points_per_radius + j,:] = ion_flux
						plt.plot(final_vel_arr_plot,ion_flux,'.')
				radii_plotted += 1

			elif (radii_arr[i] > bins[k+1]):
				break

		plt.title('Stacked Spectra: ' + str(ion) + ' %.1f-%.1f R_vir (%.1f-%.1f kpc) M=%.1f' % (bins[k]/gal_R200, bins[k+1]/gal_R200,bins[k],bins[k+1],np.log10(gal_M200)))
		plt.xlabel('Velocity Relative to Central Halo (km/s)')
		plt.ylabel('Flux')
		plt.ylim(ymin = 0, ymax = 1)
		plt.savefig(file_base + '/curr_spectra/' + ion + '/stacked/binned/stacked_plot_'+str(ion)+'_%.1f_to_%.1f_R_vir_(%.1f_to_%.1f_kpc).png' % (bins[k]/gal_R200, bins[k+1]/gal_R200,bins[k],bins[k+1]))
		plt.close()
		mean_gal_spectra =  np.mean(spectra_arr,axis=0)
		std_gal_spectra = np.std(spectra_arr,axis=0)

		### order arrays so velocities are sorted from lowers to highest (otherwise fill_between will show a discontinuity because of the 
		### the non-sequential values of the x-array)
		sorted_indices = np.argsort(final_vel_arr_plot)
		final_vel_arr_plot_sorted = final_vel_arr_plot[sorted_indices]
		mean_gal_spectra_sorted = mean_gal_spectra[sorted_indices]
		std_gal_spectra_sorted = std_gal_spectra[sorted_indices]

		top_err = np.where(mean_gal_spectra_sorted+std_gal_spectra_sorted>1.0, 1.0, mean_gal_spectra_sorted+std_gal_spectra_sorted)
		bot_err = np.where(mean_gal_spectra_sorted-std_gal_spectra_sorted<0.0, 0.0, mean_gal_spectra_sorted-std_gal_spectra_sorted)
		plt.fill_between(final_vel_arr_plot_sorted,bot_err, top_err, interpolate = False)
		plt.plot(final_vel_arr_plot_sorted,mean_gal_spectra_sorted,'r.')
		plt.title('Mean Spectra(1 sig): ' + str(ion) + ' %.1f-%.1f R_vir (%.1f-%.1f kpc) M=%.1f' % (bins[k]/gal_R200, bins[k+1]/gal_R200,bins[k],bins[k+1],np.log10(gal_M200)))
		plt.xlabel('Velocity Relative to Central Halo (km/s)')
		plt.ylabel('Flux')
		plt.ylim(ymin = 0, ymax = 1)
		plt.savefig(file_base + '/curr_spectra/' + ion + '/mean/binned/mean_with_err_'+str(ion)+'_%.1f_to_%.1f_R_vir_(%.1f_to_%.1f_kpc).png' % (bins[k]/gal_R200, bins[k+1]/gal_R200,bins[k],bins[k+1]))
		plt.close()

def add_radii_to_specwizard_output(spec_file, gal_output_file,max_radius,points_per_radius):

	with h5py.File(gal_output_file,'r') as hf:
		GalaxyProperties = hf.get("GalaxyProperties")
		R200 = np.array(GalaxyProperties.get("gal_R200"))[0]

	radii = np.linspace(0.1, max_radius, int(10*max_radius))*R200

	with h5py.File(spec_file,'r+') as hf:
		size = np.size(radii)

		output_radii = hf.create_dataset('radii', (size,), maxshape = (None,), data = radii)
		output_radii.attrs['units'] = 'kpc'

		points_per_radius = hf.create_dataset('points_per_radius', (1,), maxshape=(None,), data = points_per_radius)


def plot_spectra_by_output_radii_multiple_ions(ions, spec_file, gal_output_file, max_abs_vel = None, file_base = '.'):
	for ion in ions:
		plot_spectra_by_output_radii(ion, spec_file, gal_output_file, max_abs_vel, file_base)

def plot_spectra_binned_multiple_ions(ions, bins, spec_file, gal_output_file, max_abs_vel = None, file_base = '.'):
	for ion in ions:
		plot_spectra_binned(ion, bins, spec_file, gal_output_file, max_abs_vel, file_base)

def make_txt_output(final_vel_arr_plot, ion_flux, optical_depth, snap_directory, log_total_column_density, curr_spectrum, ion, radius, gal_R200, gal_M200):
	f = open(ion+"_%s_%.1f_R_vir.txt" % (curr_spectrum, radius/gal_R200),'w')
	f.write("# snap_directory log_total_column_density(cm^-2) curr_spectrum ion  radius(kpc) gal_R200(kpc) gal_M200(log10(M/10^10*M_sol))\n")
	f.write("# %s %.3f %s %s %.3f %.3f %.3f\n" % (snap_directory, log_total_column_density, curr_spectrum, ion, radius, gal_R200, np.log10(gal_M200)))
	f.write("# velocity_array(km/s) ion_flux optical_depth\n")
	lines = np.size(final_vel_arr_plot)
	for i in range(0,lines):
		f.write("%.3e %.3e %.3e\n" % (final_vel_arr_plot[i], ion_flux[i], optical_depth[i]))
	f.close()

def make_hdf5_output(final_vel_arr_plot, ion_flux, optical_depth, snap_directory, log_total_column_density, curr_spectrum, ion, radius, gal_R200, gal_M200):
	try:
		with h5py.File('spectral_data.hdf5', 'x') as hf:
			GalaxyData = hf.create_group('GalaxyData')
			GalaxyData.create_dataset('snap_directory', data = snap_directory)
			R200_hdf5 = GalaxyData.create_dataset('gal_R200', data = gal_R200)
			R200_hdf5.attrs['units'] = 'kpc'
			M200_hdf5 = GalaxyData.create_dataset('gal_M200', data = np.log10(gal_M200))
			M200_hdf5.attrs['units'] = 'log10(M/10^10*M_sol)'

			curr_spectrum = hf.create_group("_%s_%.1f_R_vir" % (curr_spectrum, radius/gal_R200))
			curr_ion = curr_spectrum.create_group(ion)
			vel_arr_hdf5 = curr_ion.create_dataset('final_vel_arr_plot', data = final_vel_arr_plot)
			vel_arr_hdf5.attrs['units'] = 'km/s'
			curr_ion.create_dataset('ion_flux', data = ion_flux)
			curr_ion.create_dataset('optical_depth', data = optical_depth)
			log_total_column_density_hdf5 = curr_ion.create_dataset('log_total_column_density', data = log_total_column_density)
			log_total_column_density_hdf5.attrs['units'] = 'cm^-2'
			radius_hdf5 = curr_ion.create_dataset('radius', data = radius)
			radius_hdf5.attrs['units'] = 'kpc'

	except:
		with h5py.File('spectral_data.hdf5', 'r+') as hf:
			try:
				curr_spectrum = hf.create_group("_%s_%.1f_R_vir" % (curr_spectrum, radius/gal_R200))
				curr_ion = curr_spectrum.create_group(ion)
				vel_arr_hdf5 = curr_ion.create_dataset('final_vel_arr_plot', data = final_vel_arr_plot)
				vel_arr_hdf5.attrs['units'] = 'km/s'
				curr_ion.create_dataset('ion_flux', data = ion_flux)
				curr_ion.create_dataset('optical_depth', data = optical_depth)
				log_total_column_density_hdf5 = curr_ion.create_dataset('log_total_column_density', data = log_total_column_density)
				log_total_column_density_hdf5.attrs['units'] = 'cm^-2'
				radius_hdf5 = curr_ion.create_dataset('radius', data = radius)
				radius_hdf5.attrs['units'] = 'kpc'
			except:
				curr_spectrum = hf.get("_%s_%.1f_R_vir" % (curr_spectrum, radius/gal_R200))
				curr_ion = curr_spectrum.create_group(ion)
				vel_arr_hdf5 = curr_ion.create_dataset('final_vel_arr_plot', data = final_vel_arr_plot)
				vel_arr_hdf5.attrs['units'] = 'km/s'
				curr_ion.create_dataset('ion_flux', data = ion_flux)
				curr_ion.create_dataset('optical_depth', data = optical_depth)
				log_total_column_density_hdf5 = curr_ion.create_dataset('log_total_column_density', data = log_total_column_density)
				log_total_column_density_hdf5.attrs['units'] = 'cm^-2'
				radius_hdf5 = curr_ion.create_dataset('radius', data = radius)
				radius_hdf5.attrs['units'] = 'kpc'


def plot_for_multiple_gals_by_radius(data_directory, spectra_files, output_files, ion, radii_bins, virial_vel_bool, virial_radii_bool, 
	mean_spectra_bool, col_dense_bool, covering_frac_val):
	total_num = 0.
	num_covered = 0.
	col_dense_arr = np.array([])
	spectra_files = get_spectra_files(data_directory)
	output_files = get_output_files(data_directory)


	for j in range(0,np.size(radii_bins)-1):
		num_in_bin = 0
		final_ion_flux = np.array([])
		final_velocities = np.array([])
		for n in range(0,np.size(spectra_files)):
			with h5py.File(output_files[n]) as hf:
				GalaxyProperties = hf.get('GalaxyProperties')
				R_vir = np.array(GalaxyProperties.get('gal_R200'))
				gal_mass = np.array(GalaxyProperties.get('gal_mass'))
				virial_vel = (200.0)*(gal_mass/5.0e12)**(1./3.)

			with h5py.File(spectra_files[n]) as hf:
				GalaxyData = hf.get('GalaxyData')
				for name in hf:
					if name != 'GalaxyData':
						curr_spectra_folder = hf.get(name)
						curr_spectra = curr_spectra_folder.get(ion)
						if virial_radii_bool:
							radius = np.array(curr_spectra.get('radius'))/R_vir
						else:
							radius = np.array(curr_spectra.get('radius'))
						if (radius < radii_bins[j+1]) and (radius > radii_bins[j]):
							if (np.size(final_ion_flux > 0)):
								num_in_bin += 1
								if mean_spectra_bool:
									final_ion_flux = np.concatenate([final_ion_flux,np.array(curr_spectra.get('ion_flux'))])
									if virial_vel_bool:
										final_velocities = np.concatenate([final_velocities, np.array(curr_spectra.get('final_vel_arr_plot'))/virial_vel])
									else:
										final_velocities = np.concatenate([final_velocities, np.array(curr_spectra.get('final_vel_arr_plot'))])

							else:
								num_in_bin += 1
								if mean_spectra_bool:
									final_ion_flux = np.array(curr_spectra.get('ion_flux'))
									if virial_vel_bool:
										final_velocities = np.array(curr_spectra.get('final_vel_arr_plot'))/virial_vel
									else:
										final_velocities = np.array(curr_spectra.get('final_vel_arr_plot'))

							if (np.size(col_dense_arr) > 0):
								if col_dense_bool:
									col_dense_arr = np.append(col_dense_arr,np.array(curr_spectra.get('log_total_column_density')))
									total_num += 1.
									if col_dense_arr[-1] >= covering_frac_val:
										num_covered += 1.
									col_dense_radii = np.append(col_dense_radii,radius)

							else:			
								if col_dense_bool:
									col_dense_arr = np.array(curr_spectra.get('log_total_column_density'))
									col_dense_radii = radius
		
		if virial_vel_bool:
			vel_bins = np.linspace(np.min(final_velocities), np.max(final_velocities), (np.max(final_velocities)-np.min(final_velocities))/0.1)
		else:
			vel_bins = np.linspace(np.min(final_velocities), np.max(final_velocities), (np.max(final_velocities)-np.min(final_velocities))/5.0)
		plot_velocities = np.zeros(np.size(vel_bins)-1)
		plot_fluxes = np.zeros(np.size(vel_bins)-1)
		for i in range(0,np.size(vel_bins)-1):
			plot_fluxes[i] = np.mean(final_ion_flux[np.where((final_velocities>vel_bins[i]) & (final_velocities<vel_bins[i+1]))])
			plot_velocities[i] = np.mean([vel_bins[i],vel_bins[i+1]])

		plt.plot(plot_velocities, plot_fluxes, '-', label = '%.2f-%.2fkpc:(n=%d)' %(radii_bins[j], radii_bins[j+1], num_in_bin))
		plt.plot()
	plt.legend(loc='lower left')
	plt.ylim(ymin=0, ymax=1)
	plt.title('Flux vs Speed Relative to Central Galaxy for %s' % (ion))
	plt.ylabel('normalized flux')
	if virial_vel_bool:
		plt.xlim(xmin=-3, xmax = 3)
		plt.xlabel('vel (vel/virial velocity)')
	else:
		plt.xlim(xmin=-700, xmax = 700)
		plt.xlabel('vel (km/s)')
	plt.savefig('multi_gal_test_by_radius.png')
	plt.close()

	plt.plot(col_dense_radii, col_dense_arr, 'b.')
	plt.title('Column Densities (covering frac at %.1f=%.2f)' % (covering_frac_val, num_covered/total_num))
	plt.savefig('col_dense.png')
	plt.close()



def column_density_in_mass_radius_range(data_directory, ion, max_mass, min_mass, max_radii, min_radii):
	total_num = 0.
	num_covered = 0.
	col_dense_arr = np.array([])
	spectra_files = get_spectra_files(data_directory)
	output_files = get_output_files(data_directory)

	num_in_bin = 0
	final_columns = np.array([])

	for n in range(0,np.size(spectra_files)):
		with h5py.File(output_files[n]) as hf:
			GalaxyProperties = hf.get('GalaxyProperties')
			R_vir = np.array(GalaxyProperties.get('gal_R200'))
			gal_mass = np.array(GalaxyProperties.get('gal_mass'))
			if ((gal_mass > max_mass) or (gal_mass < min_mass)):
				continue
			virial_vel = (200.0)*(gal_mass/5.0e12)**(1./3.)

		###### Working here
		with h5py.File(spectra_files[n]) as hf:
			GalaxyData = hf.get('GalaxyData')
			for name in hf:
				if name != 'GalaxyData':
					curr_spectra_folder = hf.get(name)
					curr_spectra = curr_spectra_folder.get(ion)
					if virial_radii_bool:
						radius = np.array(curr_spectra.get('radius'))/R_vir
					else:
						radius = np.array(curr_spectra.get('radius'))
					if (radius < min_radius) and (radius > max_radius):
						if (np.size(final_columns > 0)):
							num_in_bin += 1
							final_columns = np.concatenate(final_columns, curr_spectra.get('log_total_column_density'))
						else:
							num_in_bin += 1
							if mean_spectra_bool:
								final_columns = np.array(curr_spectra.get('log_total_column_density'))

		
	####### To here
def plot_for_multiple_gals_by_mass(data_directory, ion, mass_bins,mass_colors,min_radius, max_radius, stellar_mass_bool, virial_vel_bool, virial_radii_bool):
	spectra_files = get_spectra_files(data_directory)
	output_files = get_output_files(data_directory)

	for j in range(0,np.size(mass_bins)-1):
		num_in_bin = 0
		final_ion_flux = np.array([])
		final_velocities = np.array([])
		for n in range(0,np.size(spectra_files)):
			with h5py.File(output_files[n]) as hf:
				GalaxyProperties = hf.get('GalaxyProperties')
				R_vir = np.array(GalaxyProperties.get('gal_R200'))
				gal_mass = np.array(GalaxyProperties.get('gal_mass'))
				virial_vel = (200.0)*(gal_mass/5.0e12)**(1./3.)
				if stellar_mass_bool:
					mass = np.log10(np.array(GalaxyProperties.get('gal_stellar_mass')))
				else:
					mass = np.log10(np.array(GalaxyProperties.get('gal_mass')))

			with h5py.File(spectra_files[n]) as hf:
				if (mass > mass_bins[j]) & (mass < mass_bins[j+1]):
					for name in hf:
						if name != 'GalaxyData':
							curr_spectra_folder = hf.get(name)
							curr_spectra = curr_spectra_folder.get(ion)
							if virial_radii_bool:
								radius = np.array(curr_spectra.get('radius'))/R_vir
							else:
								radius = np.array(curr_spectra.get('radius'))
							if (radius < max_radius) & (radius > min_radius):
								if np.size(final_ion_flux > 0):
									num_in_bin += 1
									final_ion_flux = np.concatenate([final_ion_flux,np.array(curr_spectra.get('ion_flux'))])
									if virial_vel_bool:
										final_velocities = np.concatenate([final_velocities, np.array(curr_spectra.get('final_vel_arr_plot'))/virial_vel])
									else:
										final_velocities = np.concatenate([final_velocities, np.array(curr_spectra.get('final_vel_arr_plot'))])
								else: 
									num_in_bin += 1
									final_ion_flux = np.array(curr_spectra.get('ion_flux'))
									if virial_vel_bool:
										final_velocities = np.array(curr_spectra.get('final_vel_arr_plot'))/virial_vel
									else:
										final_velocities = np.array(curr_spectra.get('final_vel_arr_plot'))
		
		if virial_vel_bool:
			vel_bins = np.linspace(np.min(final_velocities), np.max(final_velocities), (np.max(final_velocities)-np.min(final_velocities))/0.1)
		else:
			vel_bins = np.linspace(np.min(final_velocities), np.max(final_velocities), (np.max(final_velocities)-np.min(final_velocities))/5.0)
		plot_velocities = np.zeros(np.size(vel_bins)-1)
		plot_fluxes = np.zeros(np.size(vel_bins)-1)
		for i in range(0,np.size(vel_bins)-1):
			plot_fluxes[i] = np.mean(final_ion_flux[np.where((final_velocities>vel_bins[i]) & (final_velocities<vel_bins[i+1]))])
			plot_velocities[i] = np.mean([vel_bins[i],vel_bins[i+1]])
		plt.plot(plot_velocities, plot_fluxes, color=mass_colors[j], linestyle='-', label = '%.1f-%.1f:(n=%d)' %(mass_bins[j], mass_bins[j+1], num_in_bin))
	plt.legend(loc='lower left')
	plt.ylim(ymin=0.0, ymax = 1)
	if virial_radii_bool:
		plt.title('Flux vs Speed (relative to galaxy) for %s (R=%.2f-%.2fR_vir)' % (ion,min_radius,max_radius))
	else:
		plt.title('Flux vs Speed (relative to galaxy) for %s (R=%.2f-%.2fkpc)' % (ion,min_radius,max_radius))
	plt.ylabel('normalized flux')
	if virial_vel_bool:
		plt.xlim(xmin=-3, xmax = 3)
		plt.xlabel('vel (vel/virial velocity)')
	else:
		plt.xlim(xmin=-700, xmax = 700)
		plt.xlabel('vel (km/s)')
	plt.savefig('multi_gal_test_by_mass.png')
	plt.close()

def get_spectra_files(data_directory):
	data_directory = str(data_directory)
	if data_directory[-1] != '/':
		data_directory += '/'

	data_files = glob.glob(data_directory + '*/spectral_data.hdf5')
	return data_files

def get_output_files(data_directory):
	data_directory = str(data_directory)
	if data_directory[-1] != '/':
		data_directory += '/'

	data_files = glob.glob(data_directory + '*/output_*.hdf5')
	return data_files

### Get info on galaxy (box size znd coordinates of galaxy)
def get_gal_data_for_los_gen(snap_directory, group_number, particles_included_keyword, group_included_keyword, subfind_included_keyword):
	# get basic properties of simulation and galaxy
	glob_snap_directory = str(snap_directory) # makes directory a string so it can be passed to glob (python function)
	if glob_snap_directory[-1] != '/': # checks is directory passed included a '\' character and then makes sure it only looks at .hdf5 files
		glob_snap_directory += '/'

	snap_files = glob.glob(glob_snap_directory+"*hdf5") # list of snap files in the directory
	snap_files = np.concatenate([snap_files,glob.glob(glob_snap_directory+"*/*hdf5")])

	box_size = EagleFunctions.read_attribute(snap_files,'Header','BoxSize',include_file_keyword=particles_included_keyword)
	expansion_factor = EagleFunctions.read_attribute(snap_files,'Header','ExpansionFactor',include_file_keyword=particles_included_keyword)
	hubble_param = EagleFunctions.read_attribute(snap_files,'Header','HubbleParam',include_file_keyword=particles_included_keyword)

	gal_coords = EagleFunctions.read_array(snap_files, "Subhalo/CentreOfPotential", include_file_keyword=subfind_included_keyword)
	GrpIDs = EagleFunctions.read_array(snap_files,"Subhalo/GroupNumber", include_file_keyword=subfind_included_keyword)
	SubIDs = EagleFunctions.read_array(snap_files, "Subhalo/SubGroupNumber", include_file_keyword=subfind_included_keyword)
	index = np.where((GrpIDs == group_number) & (SubIDs == 0))[0] # picks out most massive galaxy in the designated group
	first_subhalo_ID = np.size(GrpIDs[np.where(GrpIDs < group_number)])
	gal_coords = gal_coords[index][0]

	box_size *= (1.e6*parsec)/hubble_param*expansion_factor # have to adjust becuase it's drawn from header so not put in physical by CGSConversion
	 # points in los file
	return gal_coords, box_size

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

### Given an array of radii and a number of points per radius (and an axis) creates los in cricles around given axis at equally spaced angles
def create_mult_los_per_radius_text(filename, snap_directory, group_number, particles_included_keyword, group_included_keyword, 
	                subfind_included_keyword, points_per_radius, radii, axis, center=False): # for axis 0=x, 1=y, 2=z

	gal_coords,box_size = get_gal_data_for_los_gen(snap_directory,group_number, particles_included_keyword,group_included_keyword,subfind_included_keyword)

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
				radius = radii[i]*kpc
				theta = (j*2.0*np.pi)/(points_per_radius)

				x,y,z = negative_coords_check(gal_coords,axis,box_size,radius,theta)
				file.write("%.6f %.6f %.6f %d\n" % (x,y,z,int(axis)))
				file.close()

### Given a max and min radius picks a random radius and angle from given axis
def create_one_los_text_random(filename, snap_directory, group_number, particles_included_keyword, group_included_keyword, 
	                subfind_included_keyword, cos_id, min_radius, max_radius, axis):

	gal_coords,box_size = get_gal_data_for_los_gen(snap_directory,group_number, particles_included_keyword,group_included_keyword,subfind_included_keyword)

	min_radius = (min_radius*1.e3*parsec)
	max_radius = (max_radius*1.e3*parsec)
	radius = (max_radius-min_radius)*np.random.random(1)+min_radius
	theta = np.random.random(1)*2.0*np.pi
	points = 1

	with open(filename,'w') as file:
		file.write("     " + str(points) + '\n')
		x,y,z = negative_coords_check(gal_coords,axis,box_size,radius,theta)
		file.write("%.6f %.6f %.6f %d %d\n" % (x,y,z,int(axis), int(cos_id)))
		file.close()

### Given an axis and a radius picks and angle and creates an los there
def create_one_los_text(filename, snap_directory, group_number, particles_included_keyword, group_included_keyword, 
	                subfind_included_keyword, radius, cos_id, semi_random_radii, axis):

	gal_coords,box_size = get_gal_data_for_los_gen(snap_directory,group_number, particles_included_keyword,group_included_keyword,subfind_included_keyword)
	if semi_random_radii:
		radius = np.random.normal(radius,10.)
		### make sure a really weird one isn't drawn...
		while ((radius < 10.) or (radius > 500)):
			radius = np.random.normal(radius,10.)

	radius = radius*1.e3*parsec
	theta = np.random.random(1)*2.0*np.pi
	points = 1
	with open(filename,'w') as file:
		file.write("     " + str(points) + '\n')
		x,y,z = negative_coords_check(gal_coords,axis,box_size,radius,theta)
		file.write("%.6f %.6f %.6f %d %d\n" % (x,y,z,int(axis), int(cos_id)))
		file.close()

	line_coords = np.array([x,y,z])*box_size


def add_los_to_text(filename, snap_directory, group_number, particles_included_keyword, group_included_keyword, 
	                subfind_included_keyword, radius, cos_id, semi_random_radii, axis):

	gal_coords,box_size = get_gal_data_for_los_gen(snap_directory,group_number, particles_included_keyword,group_included_keyword,subfind_included_keyword)

	if semi_random_radii:
		radius = np.random.normal(radius,10.)
		### make sure a really weird one isn't drawn...
		while ((radius < 10.) or (radius > 500)):
			radius = np.random.normal(radius,10.)

	radius = radius*1.e3*parsec
	theta = np.random.random(1)*2.0*np.pi
	points = 1
	with open(filename,'r+') as file:
		lines = 0
		for line in file:
			lines += 1
		file.close()

	with open(filename,'r') as file:
		first_line = True
		with open('temp_file.txt','w') as temp_file:
			for line in file:
				if first_line:
					temp_file.write("     " + str(lines) + '\n')
					first_line = False
				else:
					temp_file.write(line)
			x,y,z = negative_coords_check(gal_coords,axis,box_size,radius,theta)
			temp_file.write("%.6f %.6f %.6f %d %.d\n" % (x,y,z,int(axis), int(cos_id)))
			temp_file.close()
		file.close()

		os.rename('temp_file.txt', filename)





