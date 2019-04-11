import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

num_compares = 7
radii = np.arange(20.,180.,20.)
plot_pts = np.size(radii)
r_err = 1. # range to look for (20 +/- 1 for example) because radii saved will have some variance)

hires_files, hires_col_list, radii_list = [[] for x in range(num_compares)], [[] for x in range(num_compares)], [[] for x in range(num_compares)]
lowres_files, lowres_col_list = [[] for x in range(num_compares)], [[] for x in range(num_compares)]
hires_W_list, lowres_W_list = [[] for x in range(num_compares)], [[] for x in range(num_compares)]

hires_col_med, hires_col_1low, hires_col_1hi , hires_col_2low, hires_col_2hi = np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts) 
hires_W_med, hires_W_1low, hires_W_1hi , hires_W_2low, hires_W_2hi = np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts) 
hires_14, hires_16, hires_18 = np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts)

lowres_col_med, lowres_col_1low, lowres_col_1hi, lowres_col_2low, lowres_col_2hi = np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts)
lowres_W_med, lowres_W_1low, lowres_W_1hi, lowres_W_2low, lowres_W_2hi = np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts)
lowres_14, lowres_16, lowres_18 = np.zeros(plot_pts), np.zeros(plot_pts), np.zeros(plot_pts)

for i in range(num_compares):
	hires_files[i] = "/cosma/home/analyse/rhorton/Ali_Spec_src/convergence_tests/data_00"+str(i+1)+"_x008/survey_results.npz"
	lowres_files[i] = "/cosma/home/analyse/rhorton/Ali_Spec_src/convergence_tests/data_00"+str(i+1)+"_x001/survey_results.npz"

for i in range(num_compares):
	hires_npz = np.load(hires_files[i])
	hires_radii, hires_col, hires_W = hires_npz["radii_arr"], hires_npz["col_dens_arr"], hires_npz["W_arr"]
	hires_col_list[i], hires_W_list[i], radii_list[i] = list(hires_col), list(hires_W), list(hires_radii)

	lowres_npz = np.load(lowres_files[i])
	lowres_radii, lowres_col, lowres_W = lowres_npz["radii_arr"], lowres_npz["col_dens_arr"], lowres_npz["W_arr"]
	lowres_col_list[i], lowres_W_list[i] = list(lowres_col), list(lowres_W)

hires_col_list, lowres_col_list, radii_list = list(itertools.chain.from_iterable(hires_col_list)), list(itertools.chain.from_iterable(lowres_col_list)), list(itertools.chain.from_iterable(radii_list))
hires_W_list, lowres_W_list = list(itertools.chain.from_iterable(hires_W_list)), list(itertools.chain.from_iterable(lowres_W_list))

for j, radius in enumerate(radii):
	temp_hires_cols = [hires_col_list[i] for i in range(np.size(hires_col_list)) if ((radii_list[i] > radius - r_err) & (radii_list[i] < radius + r_err))]
	temp_lowres_cols = [lowres_col_list[i] for i in range(np.size(lowres_col_list)) if ((radii_list[i] > radius - r_err) & (radii_list[i] < radius + r_err))]

	hires_col_med[j], hires_col_1low[j], hires_col_1hi[j] = np.median(temp_hires_cols), np.percentile(temp_hires_cols, 16.), np.percentile(temp_hires_cols, 84.)
	hires_col_2low[j], hires_col_2hi[j] = np.percentile(temp_hires_cols, 5.), np.percentile(temp_hires_cols, 95.)
	lowres_col_med[j], lowres_col_1low[j], lowres_col_1hi[j] = np.median(temp_lowres_cols), np.percentile(temp_lowres_cols, 16.), np.percentile(temp_lowres_cols, 84.)
	lowres_col_2low[j], lowres_col_2hi[j] = np.percentile(temp_lowres_cols, 5.), np.percentile(temp_lowres_cols, 95.)

	num_temp_cols = float(np.size(temp_hires_cols))
	hires_14[j], hires_16[j], hires_18[j] = np.size([x for x in temp_hires_cols if x >= 14.])/num_temp_cols, np.size([x for x in temp_hires_cols if x >= 16.])/num_temp_cols, np.size([x for x in temp_hires_cols if x >= 18.])/num_temp_cols
	lowres_14[j], lowres_16[j], lowres_18[j] = np.size([x for x in temp_lowres_cols if x >= 14.])/num_temp_cols, np.size([x for x in temp_lowres_cols if x >= 16.])/num_temp_cols, np.size([x for x in temp_lowres_cols if x >= 18.])/num_temp_cols

	temp_hires_W = [hires_W_list[i] for i in range(np.size(hires_W_list)) if ((radii_list[i] > radius - r_err) & (radii_list[i] < radius + r_err))]
	temp_lowres_W = [lowres_W_list[i] for i in range(np.size(lowres_W_list)) if ((radii_list[i] > radius - r_err) & (radii_list[i] < radius + r_err))]

	hires_W_med[j], hires_W_1low[j], hires_W_1hi[j] = np.median(temp_hires_W), np.percentile(temp_hires_W, 16.), np.percentile(temp_hires_W, 84.)
	hires_W_2low[j], hires_W_2hi[j] = np.percentile(temp_hires_W, 5.), np.percentile(temp_hires_W, 95.)
	lowres_W_med[j], lowres_W_1low[j], lowres_W_1hi[j] = np.median(temp_lowres_W), np.percentile(temp_lowres_W, 16.), np.percentile(temp_lowres_W, 84.)
	lowres_W_2low[j], lowres_W_2hi[j] = np.percentile(temp_lowres_W, 5.), np.percentile(temp_lowres_W, 95.)

col_fig, col_ax = plt.subplots(1)
col_ax.plot(radii, hires_col_med, color='b', label="HiRes")
col_ax.fill_between(radii, hires_col_1low, hires_col_1hi, color='b', alpha=0.33)
col_ax.fill_between(radii, hires_col_2low, hires_col_2hi, color='b', alpha=0.15)
col_ax.plot(radii, lowres_col_med, color='k', label="Fiducial")
col_ax.fill_between(radii, lowres_col_1low, lowres_col_1hi, color='k', alpha=0.33)
col_ax.fill_between(radii, lowres_col_2low, lowres_col_2hi, color='k', alpha=0.15)
col_ax.set_xlim((20.,160.))
col_ax.set_ylim((12.,21.))
col_ax.legend(loc="upper right")
col_ax.set_xlabel("Impact Prameter (kpc)")
col_ax.set_ylabel(r"${\rm log_{10}}(N_{HI})$ ${\rm cm^{-2}}$")
col_ax.set_title("Column Densities vs Impact Parameter for Different Resolutions")
col_fig.savefig("col_conv.pdf", bbox_inches="tight")
plt.close(col_fig)

W_fig, W_ax = plt.subplots(1)
W_ax.plot(radii, hires_W_med, color='b', label="HiRes")
W_ax.fill_between(radii, hires_W_1low, hires_W_1hi, color='b', alpha=0.33)
W_ax.fill_between(radii, hires_W_2low, hires_W_2hi, color='b', alpha=0.15)
W_ax.plot(radii, lowres_W_med, color='k', label="Fiducial")
W_ax.fill_between(radii, lowres_W_1low, lowres_W_1hi, color='k', alpha=0.33)
W_ax.fill_between(radii, lowres_W_2low, lowres_W_2hi, color='k', alpha=0.15)
W_ax.set_xlim((20.,160.))
W_ax.set_ylim((0.,1.5))
W_ax.legend(loc="upper right")
W_ax.set_xlabel("Impact Prameter (kpc)")
W_ax.set_ylabel(r"$Equivalent Width$ $(\AA{})$")
W_ax.set_title("Equivalent Widths vs Impact Parameter for Different Resolutions")
W_fig.savefig("W_conv.pdf", bbox_inches="tight")
plt.close(W_fig)

cov_fig, cov_ax = plt.subplots(1)
cov_ax.plot(radii, hires_14, 'b', label="HiRes 14", linestyle="-")
cov_ax.plot(radii, lowres_14, 'k', label="Fiducial 14", linestyle="-")
cov_ax.plot(radii, hires_16, 'b', label="HiRes 16", linestyle="--")
cov_ax.plot(radii, lowres_16, 'k', label="Fiducial 16", linestyle="--")
cov_ax.plot(radii, hires_18, 'b', label="HiRes 18", linestyle=":")
cov_ax.plot(radii, lowres_18, 'k', label="Fiducial 18", linestyle=":")
cov_ax.set_xlim((20.,160.))
cov_ax.set_ylim((0.0,1.0))
cov_ax.legend(loc = "lower center", ncol=3)
cov_ax.set_xlabel("Impact Parameter (kpc)")
cov_ax.set_ylabel(r"Covering Fraction of ${\rm log_{10}}(N_{HI})$")
cov_ax.set_title("Covering Fractions for Different Resolutions")
cov_fig.savefig("cov_conv.pdf", bbox_inches="tight")
plt.close(cov_fig)



