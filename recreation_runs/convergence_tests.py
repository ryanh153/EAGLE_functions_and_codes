import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hires_file = "/cosma/home/analyse/rhorton/Ali_Spec_src/convergence_tests/data_001_x008/hires/survey_results.npz"
lowres_file = "/cosma/home/analyse/rhorton/Ali_Spec_src/convergence_tests/data_001_x008/lowres/survey_results.npz"

radii = np.arange(20.,180.,20.)
plot_pts = np.size(radii)
r_err = 1. # range to look for (20 +/- 1 for example) because radii saved will have some variance)
hires_col_med, hires_col_low, hires_col_hi = np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts)
lowres_col_med, lowres_col_low, lowres_col_hi = np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts)

hires_npz = np.load(hires_file)
try:
	hires_radii, hires_col, hires_W = hires_npz["radii_arr"], hires_npz["col_dens_arr"], hires_npz["W_arr"]
except: # used to not have these names. If using an old file read them out, then save them properly. 
	gal_coords, box_size, file_id_arr, axis_arr, radii_arr, angle_arr, col_dens_arr, H_col_dens_arr, W_arr, line_num_minima_arr, line_centroid_vel_arr, line_FWHM_arr, line_depth_arr, line_temps_arr, line_ion_densities_arr, line_nH_arr, specwizrd_velocity_array = hires_npz["arr_0"], \
	hires_npz["arr_1"], hires_npz["arr_2"], hires_npz["arr_3"], hires_npz["arr_4"], hires_npz["arr_5"], hires_npz["arr_6"], hires_npz["arr_7"], hires_npz["arr_8"], hires_npz["arr_9"], hires_npz["arr_10"], hires_npz["arr_11"], hires_npz["arr_12"], hires_npz["arr_13"], hires_npz["arr_14"], \
	hires_npz["arr_15"], hires_npz["arr_16"]

	np.savez(hires_file, gal_coords=gal_coords, box_size=box_size, file_id_arr=file_id_arr, axis_arr=axis_arr, radii_arr=radii_arr, angle_arr=angle_arr, col_dens_arr=col_dens_arr, H_col_dens_arr=H_col_dens_arr, W_arr=W_arr, line_num_minima_arr=np.hstack(line_num_minima_arr), line_centroid_vel_arr=np.hstack(line_centroid_vel_arr), line_FWHM_arr=np.hstack(line_FWHM_arr), line_depth_arr=np.hstack(line_depth_arr), line_temps_arr=np.hstack(line_temps_arr), line_ion_densities_arr=np.hstack(line_ion_densities_arr), line_nH_arr=np.hstack(line_nH_arr), specwizrd_velocity_array=specwizrd_velocity_array)
	hires_npz = np.load(hires_file)
	hires_radii, hires_col, hires_W = hires_npz["radii_arr"], hires_npz["col_dens_arr"], hires_npz["W_arr"]

lowres_npz = np.load(lowres_file)
try:
	lowres_radii, lowres_col, lowres_W = lowres_npz["radii_arr"], lowres_npz["col_dens_arr"], lowres_npz["W_arr"]
except: # used to not have these names. If using an old file read them out, then save them properly. 
	gal_coords, box_size, file_id_arr, axis_arr, radii_arr, angle_arr, col_dens_arr, H_col_dens_arr, W_arr, line_num_minima_arr, line_centroid_vel_arr, line_FWHM_arr, line_depth_arr, line_temps_arr, line_ion_densities_arr, line_nH_arr, specwizrd_velocity_array = lowres_npz["arr_0"], \
	lowres_npz["arr_1"], lowres_npz["arr_2"], lowres_npz["arr_3"], lowres_npz["arr_4"], lowres_npz["arr_5"], lowres_npz["arr_6"], lowres_npz["arr_7"], lowres_npz["arr_8"], lowres_npz["arr_9"], lowres_npz["arr_10"], lowres_npz["arr_11"], lowres_npz["arr_12"], lowres_npz["arr_13"], lowres_npz["arr_14"], \
	lowres_npz["arr_15"], lowres_npz["arr_16"]

	np.savez(lowres_file, gal_coords=gal_coords, box_size=box_size, file_id_arr=file_id_arr, axis_arr=axis_arr, radii_arr=radii_arr, angle_arr=angle_arr, col_dens_arr=col_dens_arr, H_col_dens_arr=H_col_dens_arr, W_arr=W_arr, line_num_minima_arr=np.hstack(line_num_minima_arr), line_centroid_vel_arr=np.hstack(line_centroid_vel_arr), line_FWHM_arr=np.hstack(line_FWHM_arr), line_depth_arr=np.hstack(line_depth_arr), line_temps_arr=np.hstack(line_temps_arr), line_ion_densities_arr=np.hstack(line_ion_densities_arr), line_nH_arr=np.hstack(line_nH_arr), specwizrd_velocity_array=specwizrd_velocity_array)
	lowres_npz = np.load(lowres_file)
	lowres_radii, lowres_col, lowres_W = lowres_npz["radii_arr"], lowres_npz["col_dens_arr"], lowres_npz["W_arr"]

print hires_radii
print ''

for i, radius in enumerate(radii):
	indices = np.argwhere(((hires_radii > radius - r_err) & (hires_radii > radius + r_err)))[:,0]
	temp_hires_cols, temp_lowres_cols = hires_col[indices], lowres_col[indices]
	hires_col_med[i], hires_col_low, hires_col_hi = np.median(temp_hires_cols), np.percentile(temp_hires_cols, 16.), np.percentile(temp_hires_cols, 84.)
	lowres_col_med[i], lowres_col_low, lowres_col_hi = np.median(temp_lowres_cols), np.percentile(temp_lowres_cols, 16.), np.percentile(temp_lowres_cols, 84.)

fig, ax = plt.subplots(1)
ax.plot(radii, hires_col_med)
ax.hold(True)
ax.plot(radii, lowres_col_med)
ax.hold(False)
fig.savefig("convergence_tests.pdf")
plt.close(fig)