### imports
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker
import bisect
import os
import subprocess
import itertools

do_cum_mass_plots = False
do_kin_plots = False
do_r_dot_v_plots = True

if do_cum_mass_plots:
   import spec_cum_data_semi as mock_spectra_data
if do_kin_plots:
   import survey_realization_functions
if do_r_dot_v_plots:
   import sim_v_dot_r

### Constants
omega_b = 0.04825
omega_m = 0.307
f_h = 0.752
G = 6.67e-8 # cgs
sol_mass_to_g = 1.99e33
parsec_to_cm = 3.0857e18 # cm

### Set plot parameters

plt.rcParams['axes.labelsize'], plt.rcParams['axes.titlesize'], plt.rcParams['legend.fontsize'] = 18., 20., 14.
log_fmt = ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
sf = ticker.ScalarFormatter()
sf.set_scientific(False)

### Prochaska Data
proch_radii = np.arange(25., 160., 10.)
tens = np.zeros(np.size(proch_radii)) + 10.
proch_ann_mass = np.power(tens, np.array([8.9, 9.9, 9.4, 8.2, 10.0, 7.3, 10.6, 9.2, 8.2, 10.2, 9.0, 8.6, 8.1, 8.4]))
adj_proch_ann_mass = np.power(tens, np.array([8.9, 9.9, 9.4, 8.2, 10.0, 7.3, 9.6, 9.2, 8.2, 9.2, 9.0, 8.6, 8.1, 8.4]))
proch_mass_err_logged = np.array([0.3, 0.4, 0.3, 0.3, 0.3, 0.3, 0.4, 0.2, 0.2, 0.4, 0.3, 0.3, 0.3, 0.3])
proch_mass_err = np.power(tens, np.array([8.9, 9.9, 9.4, 8.2, 10.0, 7.3, 9.6, 9.2, 8.2, 9.2, 9.0, 8.6, 8.1, 8.4]))*np.log(10.)*np.array([0.3, 0.4, 0.3, 0.3, 0.3, 0.3, 0.4, 0.2, 0.2, 0.4, 0.3, 0.3, 0.3, 0.3])


if do_cum_mass_plots:
   proch_cum_mass, adj_proch_cum_mass, proch_cum_mass_err, proch_cum_mass_err_logged = np.zeros(np.size(proch_ann_mass)), np.zeros(np.size(proch_ann_mass)), np.zeros(np.size(proch_ann_mass)), np.zeros(np.size(proch_ann_mass))
   for i in range(0,np.size(proch_ann_mass)):
      proch_cum_mass[i] = np.sum(proch_ann_mass[0:i+1])
      adj_proch_cum_mass[i] = np.sum(adj_proch_ann_mass[0:i+1])
      proch_cum_mass_err[i] = np.sqrt(np.sum(proch_mass_err[0:i+1]**2.))
      proch_cum_mass_err_logged[i] = np.sqrt(np.sum(proch_mass_err_logged[0:i+1]**2.))

   ### data analysis

   mean_bool = False
   cool_bool = False
   to_virial = False
   normalized_units = False

   upper_mass = np.array([9.7, 15., 15.])
   lower_mass = np.array([5., 9.7, 9.7])

   upper_ssfr = np.array([-5., -5., -11.])
   lower_ssfr = np.array([-15., -11., -15.])

   labels = ['Low Mass', 'Active', 'Passive']
   labels2 = ['Total H', 'Neutral H']
   temp_labels = ['Total', r'Cool (r$<10^{5}$)']
   colors = ['k', 'b', 'r']
   if to_virial:
      stagger = [0., -0.01, 0.01]
   else:
      stagger = [0.0, -0.5, 0.5]

   ### to 160 for cos or to virial for more theoretical stuff?
   if to_virial:
      if cool_bool:
         import sim_ann_mass_data_vir_t_cut as cool_sim_data
      import sim_ann_mass_data_vir as sim_data
   else:
      if cool_bool:
         import sim_annular_mass_data_t_cut as cool_sim_data
      else:
         import sim_annular_mass_data as sim_data

   ### remove wierdly low halo mass halo and halo with no chem array (so no simulated HI data)
   ann_masses, neut_ann_masses, halo_masses, stellar_masses, sSFRs, R200s = sim_data.ann_masses, sim_data.neut_ann_masses, sim_data.halo_masses, sim_data.stellar_masses, sim_data.sSFRs, sim_data.R200s
   
   mask = np.ones(np.size(halo_masses),dtype=bool)
   low_indices = np.argwhere(ann_masses[:,0]==0.0)[:,0]
   mask[low_indices] = 0
   low_indices =np.argwhere(neut_ann_masses[:,0] == 0.)[:,0]
   mask[low_indices] = 0

   ann_masses, neut_ann_masses, halo_masses, stellar_masses, sSFRs, R200s = ann_masses[mask,:], neut_ann_masses[mask,:], halo_masses[mask], stellar_masses[mask], sSFRs[mask], R200s[mask]
   if cool_bool:
      cool_ann_masses, cool_neut_ann_masses, cool_halo_masses, cool_stellar_masses, cool_sSFRs = cool_sim_data.ann_masses[mask,:], cool_sim_data.neut_ann_masses[mask,:], cool_sim_data.halo_masses[mask], cool_sim_data.stellar_masses[mask], cool_sim_data.sSFRs[mask]

   ### mock survey results
   mock_cum_mass, mock_cum_mass_top, mock_cum_mass_bot, mock_neut_cum_mass, mock_neut_cum_mass_top, mock_neut_cum_mass_bot, mock_radii = mock_spectra_data.cum_mass_jack, mock_spectra_data.cum_mass_jack_top, \
      mock_spectra_data.cum_mass_jack_bot, mock_spectra_data.neut_cum_mass_jack, mock_spectra_data.neut_cum_mass_jack_top, mock_spectra_data.neut_cum_mass_jack_bot, mock_spectra_data.radii
   mock_ann_mass, mock_ann_mass_top, mock_ann_mass_bot, mock_neut_ann_mass, mock_neut_ann_mass_top, mock_neut_ann_mass_bot = mock_spectra_data.mass_ann, mock_spectra_data.mass_ann_top, mock_spectra_data.mass_ann_bot, \
      mock_spectra_data.neut_ann_mass, mock_spectra_data.neut_ann_mass_top, mock_spectra_data.neut_ann_mass_bot
   mock_cols, mock_cols_top, mock_cols_bot, mock_neut_cols, mock_neut_cols_top, mock_neut_cols_bot = mock_spectra_data.med_cols, mock_spectra_data.med_cols_top, \
      mock_spectra_data.med_cols_bot, mock_spectra_data.neut_cols, mock_spectra_data.neut_cols_top, mock_spectra_data.neut_cols_bot 
   all_cols, all_H_cols, temperatures, ion_num_dense, n_H = mock_spectra_data.all_cols, mock_spectra_data.all_H_cols, mock_spectra_data.temperatures, mock_spectra_data.ion_num_densities, mock_spectra_data.n_H

   ### pass on only ann masses out to the virial radius for each galaxy 
   radii = np.arange(0.,650.,5.)
   if normalized_units:
            plot_radii = np.arange(0.025, 1.025, 0.05)
            unlog_halo_mass = np.power(np.zeros(np.size(halo_masses))+10., halo_masses)[:, np.newaxis]
            m_close = unlog_halo_mass*(omega_b/omega_m)*f_h
            ann_masses /= m_close
            neut_ann_masses /= m_close
   else:
      for i in range(np.size(halo_masses)):
         if to_virial:
            vir_index = bisect.bisect(radii, R200s[i])
            if vir_index == np.size(radii):
               continue
            elif np.abs(radii[vir_index-1]-R200s[i]) < np.abs(radii[vir_index]-R200s[i]):
               vir_index -= 1

            ann_masses[i,vir_index+1::] = 0
            neut_ann_masses[i,vir_index+1::] = 0
            if cool_bool:
               cool_ann_masses[i,vir_index+1::] = 0
               cool_neut_ann_masses[i,vir_index+1::] = 0

            plot_radii = np.arange(2.5,652.5,5.)
         else:
            plot_radii = np.arange(25., 160., 10.)
            # plot_radii = np.arange(2.5, 17.5, 5.)
   plot_cum_masses = np.sum(ann_masses, axis=1)
   plot_neut_cum_mases = np.sum(neut_ann_masses, axis=1)
   if cool_bool:
      cool_plot_cum_masses = np.sum(cool_ann_masses, axis=1)
      cool_plot_neut_cum_mases = np.sum(cool_neut_ann_masses, axis=1)

   ann_med_fig, ann_med_ax = plt.subplots()
   ann_med_plotsean_fig, ann_mean_ax = plt.subplots()
   cum_med_fig, cum_med_ax = plt.subplots()
   cum_mean_fig, cum_mean_ax = plt.subplots()
   mock_col_fig, mock_col_ax = plt.subplots()
   mock_ann_fig, mock_ann_ax = plt.subplots()
   mock_cum_fig, mock_cum_ax = plt.subplots()
   col_comp_temp_fig, col_comp_temp_ax = plt.subplots()
   col_comp_ion_dens_fig, col_comp_ion_dens_ax = plt.subplots()
   col_comp_H_dens_fig, col_comp_H_dens_ax = plt.subplots()

   for i in range(np.size(upper_mass)):
      curr_indices = np.where((stellar_masses > lower_mass[i]) & (stellar_masses <= upper_mass[i]) & (sSFRs > lower_ssfr[i]) & (sSFRs <= upper_ssfr[i]))[0]
      curr_ann_masses = ann_masses[curr_indices]
      neut_curr_ann_masses = neut_ann_masses[curr_indices]

      if cool_bool:
         cool_curr_indices = np.where((stellar_masses > lower_mass[i]) & (stellar_masses <= upper_mass[i]) & (sSFRs > lower_ssfr[i]) & (sSFRs <= upper_ssfr[i]))[0]
         cool_curr_ann_masses = cool_ann_masses[curr_indices]
         cool_neut_curr_ann_masses = cool_neut_ann_masses[curr_indices]

      if i == 0:
         mock_cum_ax.plot(0.,0., color=colors[i], label=labels2[0])
      mock_cum_ax.plot(mock_radii[i], mock_cum_mass[i], color=colors[i], label=labels[i], alpha=0.5)
      mock_cum_ax.fill_between(mock_radii[i], mock_cum_mass_bot[i], mock_cum_mass_top[i], color=colors[i], alpha=0.33)

      if i == 0:
         mock_cum_ax.plot(mock_radii[i], mock_neut_cum_mass[i], color=colors[i], linestyle='dotted', label=labels2[1], alpha=0.5)
      else:
         mock_cum_ax.plot(mock_radii[i], mock_neut_cum_mass[i], color=colors[i], linestyle='dotted', alpha=0.5)
      mock_cum_ax.fill_between(mock_radii[i], mock_neut_cum_mass_bot[i], mock_neut_ann_mass_top[i], color=colors[i], alpha=0.33)

      if i == 0:
         mock_ann_ax.plot(0.,0., color=colors[i], label=labels2[0])
      mock_ann_ax.plot(mock_radii[i], mock_ann_mass[i], color=colors[i], label=labels[i], alpha=0.5)
      mock_ann_ax.fill_between(mock_radii[i], mock_ann_mass_bot[i], mock_ann_mass_top[i], color=colors[i], alpha=0.33)

      if i == 0:
         mock_ann_ax.plot(mock_radii[i], mock_neut_ann_mass[i], color=colors[i], linestyle='dotted', label=labels2[1], alpha=0.5)
      else:
         mock_ann_ax.plot(mock_radii[i], mock_neut_ann_mass[i], color=colors[i], linestyle='dotted', alpha=0.5)
      mock_ann_ax.fill_between(mock_radii[i], mock_neut_ann_mass_bot[i], mock_neut_ann_mass_top[i], color=colors[i], alpha=0.33)

      if i == 0:
         mock_col_ax.plot(mock_radii[i], mock_cols[i], color=colors[i], label=labels2[0], alpha=0.5)
         mock_col_ax.plot(mock_radii[i], mock_neut_cols[i], color=colors[i], linestyle='dotted', label=labels2[1], alpha=0.5)
      mock_col_ax.plot(mock_radii[i], mock_cols[i], color=colors[i], alpha=0.5, label=labels[i])
      mock_col_ax.plot(mock_radii[i], mock_neut_cols[i], color=colors[i], linestyle='dotted', alpha=0.5)
      mock_col_ax.fill_between(mock_radii[i], mock_cols_bot[i], mock_cols_top[i], color=colors[i], alpha=0.33)
      mock_col_ax.fill_between(mock_radii[i], mock_neut_cols_bot[i], mock_neut_cols_top[i], color=colors[i], alpha=0.33)

      # if i == 0:
      #    mock_col_plots[i] = mock_col_ax.errorbar(0, 0, yerr=[[1], [1]], color=colors[i], label=labels2[0], marker='.', linestyle='None')
      # mock_col_plots[2*i+1] =mock_col_ax.errorbar(mock_radii[i]+stagger[i], mock_cols[i], yerr=[mock_cols[i]-mock_cols_bot[i], mock_cols_top[i]-mock_cols[i]], color=colors[i], label=labels[i], marker='.', markersize = 12., linestyle='None')
      # if i == 0:
      #    mock_col_plots[2*i+2] =mock_col_ax.errorbar(mock_radii[i]+stagger[i], mock_neut_cols[i], yerr=[mock_neut_cols[i]-mock_neut_cols_bot[i], mock_neut_cols_top[i]-mock_neut_cols[i]], color=colors[i], marker='*', markersize = 12., linestyle='None', label=labels2[1])
      # else:   
      #    mock_col_plots[2*i+2] =mock_col_ax.errorbar(mock_radii[i]+stagger[i], mock_neut_cols[i], yerr=[mock_neut_cols[i]-mock_neut_cols_bot[i], mock_neut_cols_top[i]-mock_neut_cols[i]], color=colors[i], marker='*', markersize = 12., linestyle='None')

      if mean_bool:
         if cool_bool == False:
            means = np.mean(curr_ann_masses, axis=0)
            errs = np.std(curr_ann_masses, axis=0)
            cum_mass_mean = np.cumsum(means)
            cum_err = np.sqrt(np.cumsum(errs**2.))
            mean_err_top, mean_err_bot, means, cum_err_top, cum_err_bot, cum_mass_mean = (means+errs), (means-errs), (means), (cum_mass_mean+cum_err), (cum_mass_mean-cum_err), (cum_mass_mean) 

            ann_mean_ax.errorbar(plot_radii+stagger[i], means, yerr=[mean_err_bot, mean_err_bot], label=labels[i], color=colors[i], marker='.')

            cum_mean_ax.plot(plot_radii, cum_mass_mean, color=colors[i], label=labels[i])
            cum_mean_ax.fill_between(plot_radii, cum_err_bot, cum_err_top, color=colors[i], alpha=0.33)

            neut_means = np.mean(neut_curr_ann_masses, axis=0)
            neut_errs = np.std(neut_curr_ann_masses, axis=0)
            neut_cum_mass_mean = np.cumsum(neut_means)
            neut_cum_err = np.sqrt(np.cumsum(neut_errs**2.))

            neut_mean_err_top, neut_mean_err_bot, neut_means, neut_cum_err_top, neut_cum_err_bot, neut_cum_mass_mean = \
            (neut_means+neut_errs), (neut_means-neut_errs), (neut_means), (neut_cum_mass_mean+neut_cum_err), (neut_cum_mass_mean-neut_cum_err), (neut_cum_mass_mean) 

            ann_mean_ax.errorbar(plot_radii+stagger[i], neut_means, yerr=[neut_mean_err_bot, neut_mean_err_bot], label=labels[i], color=colors[i], marker='*')

            cum_mean_ax.plot(plot_radii, neut_cum_mass_mean, linestyle='dotted', color=colors[i])
            cum_mean_ax.fill_between(plot_radii, neut_cum_err_bot, neut_cum_err_top, color=colors[i], label=labels[i], alpha=0.33)

         # ### Cool
         else:
        
            cool_means = np.mean(cool_curr_ann_masses, axis=0)
            cool_errs = np.std(cool_curr_ann_masses, axis=0)
            cool_cum_mass_mean = np.cumsum(cool_means)
            cool_cum_err = np.sqrt(np.cumsum(cool_errs**2.))
            cool_mean_err_top, cool_mean_err_bot, cool_means, cool_cum_err_top, cool_cum_err_bot, cool_cum_mass_mean = (cool_means+cool_errs), (cool_means-cool_errs), (cool_means), (cool_cum_mass_mean+cool_cum_err), (cool_cum_mass_mean-cool_cum_err), (cool_cum_mass_mean) 

            ann_mean_ax.plot(plot_radii, cool_means, color=colors[i])
            ann_mean_ax.fill_between(plot_radii, cool_mean_err_bot, cool_mean_err_top, color=colors[i], label=labels[i], alpha=0.33)

            cum_mean_ax.plot(plot_radii, cool_cum_mass_mean, linestyle='--', color=colors[i])
            cum_mean_ax.fill_between(plot_radii, cool_cum_err_bot, cool_cum_err_top, color=colors[i], label=labels[i], alpha=0.33)

            cool_neut_means = np.mean(cool_neut_curr_ann_masses, axis=0)
            cool_neut_errs = np.std(cool_neut_curr_ann_masses, axis=0)
            cool_neut_cum_mass_mean = np.cumsum(cool_neut_means)
            cool_neut_cum_err = np.sqrt(np.cumsum(cool_neut_errs**2.))

            cool_neut_mean_err_top, cool_neut_mean_err_bot, cool_neut_means, cool_neut_cum_err_top, cool_neut_cum_err_bot, cool_neut_cum_mass_mean = \
            np.log10(cool_neut_means+cool_neut_errs), np.log10(cool_neut_means-cool_neut_errs), np.log10(cool_neut_means), np.log10(cool_neut_cum_mass_mean+cool_neut_cum_err), np.log10(cool_neut_cum_mass_mean-cool_neut_cum_err), np.log10(cool_neut_cum_mass_mean) 

            neut_ann_mean_ax.plot(plot_radii, cool_neut_means, color=colors[i])
            neut_ann_mean_ax.fill_between(plot_radii, cool_neut_mean_err_bot, cool_neut_mean_err_top, color=colors[i], label=labels[i], alpha=0.33)
            ann_mean_ax.errorbar(plot_radii, neut_means, yerr=[neut_mean_err_bot, neut_mean_err_bot], label=labels[i], color=colors[i], marker='*')

            cum_mean_ax.plot(plot_radii, cool_neut_cum_mass_mean, color=colors[i])
            cum_mean_ax.fill_between(plot_radii, cool_neut_cum_err_bot, cool_neut_cum_err_top, color=colors[i], label=labels[i], alpha=0.33)
      else:

         medians = np.median(curr_ann_masses, axis=0)
         med_low, med_high = np.percentile(curr_ann_masses, 14., axis=0), np.percentile(curr_ann_masses, 86., axis=0)
         cum_mass_med = np.cumsum(medians)
         cum_mass_bot_err = np.cumsum(np.sqrt(np.power(medians-med_low,2.)))
         cum_mass_top_err = np.cumsum(np.sqrt(np.power(med_high-medians,2.)))
         medians, med_low, med_high, cum_mass_med, cum_mass_bot_err, cum_mass_top_err = (medians), (med_low), (med_high), (cum_mass_med), (cum_mass_med-cum_mass_bot_err), (cum_mass_top_err+cum_mass_med)

         if i == 0:
            ann_med_ax.plot(plot_radii, medians, color=colors[i], label=labels2[0])
         ann_med_ax.plot(plot_radii, medians, color=colors[i], label=labels[i])
         ann_med_ax.fill_between(plot_radii, med_low, med_high, color=colors[i],alpha=0.33)

         if i == 0:
            cum_med_ax.plot(plot_radii, cum_mass_med, color=colors[i], label=labels2[0])
         cum_med_ax.plot(plot_radii, cum_mass_med, color=colors[i], label=labels[i])
         cum_med_ax.fill_between(plot_radii, cum_mass_bot_err, cum_mass_top_err, color=colors[i],alpha=0.33)

         neut_medians = np.median(neut_curr_ann_masses, axis=0)
         neut_med_low, neut_med_high = np.percentile(neut_curr_ann_masses, 14., axis=0), np.percentile(neut_curr_ann_masses, 86., axis=0)
         neut_cum_mass_med = np.cumsum(neut_medians)
         neut_cum_mass_bot_err = neut_cum_mass_med - np.sqrt(np.cumsum(np.power(neut_medians-neut_med_low,2.)))
         neut_cum_mass_top_err = neut_cum_mass_med + np.sqrt(np.cumsum(np.power(neut_med_high-neut_medians,2.)))

         if i==0:
            ann_med_ax.plot(plot_radii, neut_medians, color=colors[i], linestyle='dotted', label=labels2[1])
         else:
            ann_med_ax.plot(plot_radii, neut_medians, color=colors[i], linestyle='dotted')
         ann_med_ax.fill_between(plot_radii, neut_med_low, neut_med_high, color=colors[i],alpha=0.33)

         if i==0:
            cum_med_ax.plot(plot_radii, neut_cum_mass_med, color=colors[i], linestyle='dotted', label=labels2[1])
         else:
            cum_med_ax.plot(plot_radii, neut_cum_mass_med, color=colors[i], linestyle='dotted')
         cum_med_ax.fill_between(plot_radii, neut_cum_mass_bot_err, neut_cum_mass_top_err, color=colors[i],alpha=0.33)

   mock_cum_ax.errorbar(proch_radii, proch_cum_mass, yerr=proch_cum_mass_err, elinewidth=2, ecolor='k', marker='*', markersize=15., color='limegreen', markeredgecolor='k', label="P")
   mock_cum_ax.plot(proch_radii, proch_cum_mass, marker='*', markersize=15., color='limegreen', markeredgecolor='k', label="Prochaska 2017") # for label. Don't know why
   mock_cum_ax.plot(145., 2.0e10, linestyle='None', marker='^', markersize=15., color='darkviolet', markeredgecolor='k', label='Werk 2014')

   cum_med_ax.errorbar(proch_radii, proch_cum_mass, yerr=proch_cum_mass_err, elinewidth=2, ecolor='k', marker='*', markersize=15., color='limegreen', markeredgecolor='k', label="P")
   cum_med_ax.plot(proch_radii, proch_cum_mass, marker='*', markersize=15., color='limegreen', markeredgecolor='k', label="Prochaska 2017") # for label. Don't know why
   cum_med_ax.plot(145., 2.0e10, linestyle='None', marker='^', markersize=15., color='darkviolet', markeredgecolor='k', label='Werk 2014')

   mock_ann_ax.errorbar(proch_radii, proch_ann_mass, yerr=proch_mass_err, linestyle='', elinewidth=2, ecolor='k', marker='*', markersize=15., color='limegreen', markeredgecolor='k')
   mock_ann_ax.plot(proch_radii, proch_ann_mass, marker='*', linestyle='', markersize=15., color='limegreen', markeredgecolor='k', label="Prochaska 2017") # for label. Don't know why

   ann_med_ax.errorbar(proch_radii, proch_ann_mass, yerr=proch_mass_err, linestyle='', elinewidth=2, ecolor='k', marker='*', markersize=15., color='limegreen', markeredgecolor='k')
   ann_med_ax.plot(proch_radii, proch_ann_mass, marker='*', linestyle='', markersize=15., color='limegreen', markeredgecolor='k', label="Prochaska 2017") # for label. Don't know why

   max_col, min_col = np.max([np.max(all_cols), np.max(all_H_cols)]), np.min([np.min(all_cols), np.min(all_H_cols)])
   one_to_one = np.linspace(min_col, max_col, 1.e2)
   col_comp_temp_cb = col_comp_temp_ax.scatter(all_cols, all_H_cols, marker='.', linewidth=0.25, s=6., c=np.log10(temperatures), cmap='jet')
   col_comp_temp_ax.plot(one_to_one, one_to_one, color='k')
   col_comp_temp_fig.colorbar(col_comp_temp_cb, label=r'$log_{10}(T)$')
   col_comp_temp_ax.set_yscale('log')
   col_comp_temp_ax.set_xscale('log')
   col_comp_temp_ax.xaxis.set_major_formatter(log_fmt)
   col_comp_temp_ax.yaxis.set_major_formatter(log_fmt)
   col_comp_temp_ax.set_xlim([10.**12,5*10.**21])
   col_comp_temp_ax.set_ylim([10.**16,5*10.**21])
   col_comp_temp_ax.set_title('HI to H Comparisons')
   col_comp_temp_fig.savefig('col_comparison_temp.pdf')
   plt.close(col_comp_temp_fig)

   col_comp_ion_dens_cb = col_comp_ion_dens_ax.scatter(all_cols, all_H_cols, marker='.', linewidth=0.25, s=6., c=np.log10(ion_num_dense), cmap='cubehelix')
   col_comp_ion_dens_ax.plot(one_to_one, one_to_one, color='k')
   col_comp_ion_dens_fig.colorbar(col_comp_ion_dens_cb,label=r'$log_{10}(n_{HI})$')
   col_comp_ion_dens_ax.set_yscale('log')
   col_comp_ion_dens_ax.set_xscale('log')
   col_comp_ion_dens_ax.xaxis.set_major_formatter(log_fmt)
   col_comp_ion_dens_ax.yaxis.set_major_formatter(log_fmt)
   col_comp_ion_dens_ax.set_xlim([10.**12,5*10.**21])
   col_comp_ion_dens_ax.set_ylim([10.**16,5*10.**21])
   col_comp_ion_dens_ax.set_title('HI to H Comparisons')
   col_comp_ion_dens_fig.savefig('col_comparison_ion_dens.pdf')
   plt.close(col_comp_ion_dens_fig)

   col_comp_H_dens_cb = col_comp_H_dens_ax.scatter(all_cols, all_H_cols, marker='.', linewidth=0.25, s=6., c=np.log10(n_H), cmap='cubehelix')
   col_comp_H_dens_ax.plot(one_to_one, one_to_one, color='k')
   col_comp_H_dens_fig.colorbar(col_comp_H_dens_cb,label=r'$log_{10}(n_{H})$')
   col_comp_H_dens_ax.set_yscale('log')
   col_comp_H_dens_ax.set_xscale('log')
   col_comp_H_dens_ax.xaxis.set_major_formatter(log_fmt)
   col_comp_H_dens_ax.yaxis.set_major_formatter(log_fmt)
   col_comp_H_dens_ax.set_xlim([10.**12,5*10.**21])
   col_comp_H_dens_ax.set_ylim([10.**16,5*10.**21])
   col_comp_H_dens_ax.set_title('HI to H Comparisons')
   col_comp_H_dens_fig.savefig('col_comparison_H_dens.pdf')
   plt.close(col_comp_H_dens_fig)


   ## Fancy legend
   lines = mock_cum_ax.get_lines()
   leg1 = plt.legend([lines[i] for i in [1,3,5]], [lines[i].get_label() for i in [1,3,5]], loc=[0.01,0.74])
   leg2 = plt.legend([lines[i] for i in [0,2]], [lines[i].get_label() for i in [0,2]], loc=[0.35,0.81])
   leg3 = plt.legend([lines[i] for i in [7,8,9]], [lines[i].get_label() for i in [7,8,9]], loc=[0.5,0.01], numpoints=1)
   mock_cum_ax.add_artist(leg1)
   mock_cum_ax.add_artist(leg2)
   mock_cum_ax.add_artist(leg3)
   mock_cum_ax.set_xlabel('Radius (kpc)')
   mock_cum_ax.set_ylabel(r'$M_{cum}$ $(M_{\odot})$')
   mock_cum_ax.set_title('Cumulative H Mass vs Radius')
   mock_cum_ax.set_yscale('log')
   mock_cum_ax.set_ylim([10**4.0,10**13.])
   mock_cum_ax.set_xlim([20.,160.])
   mock_cum_ax.yaxis.set_major_formatter(log_fmt)
   mock_cum_fig.savefig('jacknife_mass.pdf')
   plt.close(mock_cum_fig)

   # fancy legend
   lines = mock_ann_ax.get_lines()
   leg1 = plt.legend([lines[i] for i in [1,3,5]], [lines[i].get_label() for i in [1,3,5]], loc=[0.01,0.74])
   leg2 = plt.legend([lines[i] for i in [0,2]], [lines[i].get_label() for i in [0,2]], loc=[0.35,0.81])
   leg3 = plt.legend([lines[i] for i in [7,8]], [lines[i].get_label() for i in [7,8]], loc=[0.05,0.01], numpoints=1)
   mock_ann_ax.add_artist(leg1)
   mock_ann_ax.add_artist(leg2)
   mock_ann_ax.add_artist(leg3)
   mock_ann_ax.set_xlabel('Radius (kpc)')
   mock_ann_ax.set_ylabel(r'$M_{ann}$ $(M_{\odot})$')
   mock_ann_ax.set_title('Annular H Mass vs Radius')
   mock_ann_ax.set_yscale('log')
   mock_ann_ax.set_ylim([10**2.5,10**12.8])
   mock_ann_ax.set_xlim([20.,160.])
   mock_ann_ax.yaxis.set_major_formatter(log_fmt)
   mock_ann_fig.savefig('mock_ann_mass.pdf')
   plt.close(mock_ann_fig)

   lines = mock_col_ax.get_lines()
   leg1 = mock_col_ax.legend([lines[i] for i in [2,4,6]], [lines[i].get_label() for i in [2,4,6]], loc = [0.01, 0.02])
   leg2 = mock_col_ax.legend([lines[i] for i in [0,1]], [lines[i].get_label() for i in [0,1]], loc=[0.35,0.02])
   mock_col_ax.add_artist(leg1)
   mock_col_ax.add_artist(leg2)
   mock_col_ax.set_xlabel('Radius (kpc)')
   mock_col_ax.set_ylabel(r'${\rm log_{10}}(N)$ $({\rm cm^{-2}})$')
   mock_col_ax.set_title('Column Density vs Radius')
   mock_col_ax.set_yscale('log')
   mock_col_ax.set_ylim([10.**12,10.**21])
   mock_col_ax.set_xlim([20.,160.])
   mock_col_ax.yaxis.set_major_formatter(log_fmt)
   mock_col_fig.savefig('mock_cols.pdf')
   plt.close(mock_col_fig)

   if mean_bool:
      if cool_bool == False:
         ann_mean_ax.set_xlabel('Radius (kpc)')
         ann_mean_ax.set_ylabel(r'$M_{ann}$ $(M_{\odot})$')
         ann_mean_ax.set_title('H Mass in Each Annulus vs Radius')
         ann_mean_ax.set_yscale('log')
         ann_mean_ax.set_ylim([10**3.0,10**12.1])
         ann_mean_ax.set_xlim([20.,160.])
         ann_mean_ax.legend(loc='upper left')
         ann_mean_fig.savefig('ann_masses_mean.pdf')
         plt.close(ann_mean_fig)

         cum_mean_ax.set_xlabel('Radius (kpc)')
         cum_mean_ax.set_ylabel(r'$M_{cum}$ $(M_{\odot})$')
         cum_mean_ax.set_title('Cumulative H Mass vs Radius')
         cum_mean_ax.set_yscale('log')
         cum_mean_ax.set_ylim([10**4.0,10**12.1])
         cum_mean_ax.set_xlim([20.,160.])
         cum_mean_ax.legend(loc='upper left')
         cum_mean_fig.savefig('cum_masses_mean.pdf')
         plt.close(cum_mean_fig)

      ### cool
      else:
         ann_mean_ax.set_xlabel('Radius (kpc)')
         ann_mean_ax.set_ylabel(r'$M_{ann}$ $(M_{\odot})$')
         ann_mean_ax.set_title('Cool H Mass in Each Annulus vs Radius')
         ann_mean_ax.set_ylim([6.0,12.1])
         ann_meanax.set_xlim([20.,160.])
         ann_mean_ax.legend(loc='upper left')
         ann_mean_fig.savefig('ann_masses_mean_cool.pdf')
         plt.close(ann_mean_fig)

         cum_mean_ax.set_xlabel('Radius (kpc)')
         cum_mean_ax.set_ylabel(r'$M_{cum}$ $(M_{\odot})$')
         cum_mean_ax.set_title('Cool Cumulative H Mass vs Radius')
         cum_mean_ax.set_yscale('log')
         cum_mean_ax.set_ylim([10**8.0,10**12.1])
         cum_meanax.set_xlim([20.,160.])
         cum_mean_ax.legend(loc='upper left')
         cum_mean_fig.savefig('cum_masses_mean_cool.pdf')
         plt.close(cum_mean_fig)

   else:
      if cool_bool:
         ann_med_ax.set_xlabel('Radius (kpc)')
         ann_med_ax.set_ylabel(r'$M_{ann}$ $(M_{\odot})$')
         ann_med_ax.set_title('Cool H Mass in Each Annulus vs Radius')
         ann_med_ax.set_ylim([6.0,12.1])
         ann_med_ax.set_xlim([20.,160.])
         ann_med_ax.legend(loc='upper left')
         ann_med_fig.savefig('ann_masses_median_cool.pdf')
         plt.close(ann_med_fig)

         cum_med_ax.set_xlabel('Radius (kpc)')
         cum_med_ax.set_ylabel(r'$M_{cum}$ $(M_{\odot})$')
         cum_med_ax.set_title('Cool Cumulative H Mass vs Radius')
         cum_med_ax.set_ylim([6.0,12.1])
         cum_med_ax.set_xlim([20.,160.])
         cum_med_fig.savefig('cum_masses_median_cool.pdf')
         plt.close(cum_med_fig)

      else:
         # fancy legend
         lines = ann_med_ax.get_lines()
         leg1 = ann_med_ax.legend([lines[i] for i in [1,3,5]], [lines[i].get_label() for i in [1,3,5]], loc = [0.02,0.75])
         leg2 = ann_med_ax.legend([lines[i] for i in [0,2]], [lines[i].get_label() for i in [0,2]], loc = [0.37,0.82])
         leg3 = ann_med_ax.legend([lines[i] for i in [7,8]], [lines[i].get_label() for i in [7,8]], loc=[0.02,0.01], numpoints=1)
         ann_med_ax.add_artist(leg1)
         ann_med_ax.add_artist(leg2)
         ann_med_ax.add_artist(leg3)
         if normalized_units:
            ann_med_ax.set_xlabel('Radius (kpc)')
            ann_med_ax.set_ylabel(r'$M_{ann}$ $(M_{\odot})$')
            ann_med_ax.set_ylim([10**-4.,10**0.5])
            ann_med_ax.set_xlim([0.,1.])
         else:
            ann_med_ax.set_xlabel('Radius (kpc)')
            ann_med_ax.set_ylabel(r'$M_{ann}$ $(M_{\odot})$')
            ann_med_ax.set_ylim([10**2.5,10**12.8])
            ann_med_ax.set_xlim([20.,160.])

         ann_med_ax.set_title('H Mass in Each Annulus vs Radius')
         ann_med_ax.set_yscale('log')
         ann_med_ax.yaxis.set_major_formatter(log_fmt)
         ann_med_fig.savefig('ann_masses_median.pdf')
         plt.close(ann_med_fig)

         # Fancy legend
         lines = cum_med_ax.get_lines()
         leg1 = cum_med_ax.legend([lines[i] for i in [1,3,5]], [lines[i].get_label() for i in [1,3,5]], loc=[0.01,0.74])
         leg2 = cum_med_ax.legend([lines[i] for i in [0,2]], [lines[i].get_label() for i in [0,2]], loc=[0.35,0.81])
         leg3 = cum_med_ax.legend([lines[i] for i in [7,8,9]], [lines[i].get_label() for i in [7,8,9]], loc=[0.5,0.01], numpoints=1)
         cum_med_ax.add_artist(leg1)
         cum_med_ax.add_artist(leg2)
         cum_med_ax.add_artist(leg3)
         if normalized_units:
            cum_med_ax.set_xlabel(r'Radius (r/$R_{200}$)')
            cum_med_ax.set_ylabel(r'${\rm log_{10}}(M_{cum}/M_{200})$')
            cum_med_ax.set_ylim([10**-4.5,10**0.7])
            cum_med_ax.set_xlim([0.,1.])
         else:
            cum_med_ax.set_xlabel('Radius (kpc)')
            cum_med_ax.set_ylabel(r'$M_{cum}$ $(M_{\odot})$')
            cum_med_ax.set_ylim([10**4.,10**13.])
            cum_med_ax.set_xlim([20.,160.])
         cum_med_ax.set_title('Cumulative H Mass vs Radius')
         cum_med_ax.set_yscale('log')
         cum_med_ax.yaxis.set_major_formatter(log_fmt)
         cum_med_fig.savefig('cum_masses_median.pdf')
         plt.close(cum_med_fig)

   if (to_virial & cool_bool):
      fig, ax = plt.subplots()
      plot_hmass = np.power(np.ones(np.size(halo_masses))*10.,halo_masses)
      baryonically_closed = plot_hmass*(omega_b/omega_m*0.752)
      ax.scatter(plot_hmass , plot_cum_masses, color='darkgreen', marker='o', edgecolors='k', linewidth=0.75, s=30., label='Total H')
      ax.scatter(plot_hmass , cool_plot_cum_masses, color='fuchsia', marker='s', edgecolors='k', linewidth=0.75, s=30., label='Cool H')
      ax.scatter(plot_hmass , plot_neut_cum_mases, color='gold', marker='^', edgecolors='k', linewidth=0.75, s=30., label='Neutral H')
      ax.plot(plot_hmass, baryonically_closed, color='k', label=r'$M_{H}$ For Closure')

      subax = inset_axes(ax, width="30%", height="30%", loc="lower right", borderpad=1.)
      subax.scatter(plot_hmass , cool_plot_cum_masses/plot_cum_masses, color='fuchsia', marker='s', edgecolors='k', linewidth=0.75, s=15., label='Cool H')
      subax.scatter(plot_hmass , plot_neut_cum_mases/plot_cum_masses, color='gold', marker='^', edgecolors='k', linewidth=0.75, s=15., label='Neutral H')
      # subax.set_yscale('log')
      # subax.set_ylim([10**-3, 10**0.5])
      subax.set_ylim([0., 1.])
      subax.set_xscale('log')
      subax.set_xlim([10**10.5,10**14.0])
      subax.set_ylabel(r'$M_{x}/M_{H}$ $(M_{\odot})$', fontsize=12.)

      ax.legend(loc='upper left',fontsize=14., ncol=2)
      ax.set_yscale('log')
      ax.set_xscale('log')
      ax.set_ylim([10**5, 10**14.])
      ax.set_xlim([10**10.5,10**14.0])
      ax.set_xlabel(r'$M_{halo}$ $(M_{\odot})$',fontsize=18.)
      ax.set_ylabel(r'$M_{<R_{vir}}$ $(M_{\odot})$', fontsize=18.)
      ax.set_title(r'Hydrogen Masses Within $R_{vir}$', fontsize=20.)
      fig.set_tight_layout(True)
      fig.savefig('cum_H_vs_halo.pdf')
      plt.close(fig)

      fig, ax = plt.subplots()
      ax.scatter(plot_hmass , cool_plot_cum_masses/plot_cum_masses, color='fuchsia', marker='s', edgecolors='k', linewidth=0.75, s=30., label='Cool H')
      ax.scatter(plot_hmass , plot_neut_cum_mases/plot_cum_masses, color='gold', marker='^', edgecolors='k', linewidth=0.75, s=30., label='Neutral H')
      ax.legend(loc='upper left',fontsize=16., ncol=3)
      ax.set_yscale('log')
      ax.set_xscale('log')
      ax.set_ylim([10**-4.5, 10**0.])
      ax.set_xlim([10**10.5,10**14.0])
      ax.set_xlabel(r'$M_{halo}$ $(M_{\odot})$',fontsize=18.)
      ax.set_ylabel(r'$M_{x}/M_{H}$ $(M_{\odot})$', fontsize=18.)
      ax.set_title(r'$M_{x}/M_{H}$ in $R_{vir}$', fontsize=20.)
      fig.set_tight_layout(True)
      fig.savefig('frac_cum_H_vs_halo.pdf')
      plt.close(fig)



if do_kin_plots:
   bins_for_median = 10
   num_minimas, centroid_vels, depths, FWHMs, radii, temps, escape_vels, virial_radii_for_kin, halo_masses_for_kin, stellar_masses_for_kin, ssfr_for_kin = np.load('/Users/ryanhorton1/Desktop/kin_outputs.npz')

   survey_realization_functions.kinematic_plots(num_minimas, centroid_vels, depths, FWHMs, radii, temps, escape_vels, virial_radii_for_kin, halo_masses_for_kin, stellar_masses_for_kin, ssfr_for_kin, bins_for_median)

if do_r_dot_v_plots:
   ### original read out. Slow...
   # v_dot_r_arr = sim_v_dot_r.v_dot_r
   # radii = sim_v_dot_r.radii
   # halo_masses = sim_v_dot_r.halo_masses
   # stellar_masses = sim_v_dot_r.stellar_masses 
   # sSFRs = sim_v_dot_r.sSFRs
   # R200s = sim_v_dot_r.R200s

   # np.savez("vim_v_dot_r.npz", v_dot_r_arr =v_dot_r_arr, radii=radii, halo_masses= halo_masses, stellar_masses= stellar_masses, sSFRs= sSFRs, R200s= R200s)

   npzfile = np.load("vim_v_dot_r.npz")
   v_dot_r_arr = npzfile["v_dot_r_arr"]*1.e-5 # km/s. this is really an array of lists. Each is a different size...
   halo_masses = npzfile["halo_masses"] # this in logged
   stellar_masses = npzfile["stellar_masses"] # logged
   sSFRs = npzfile["sSFRs"] # logged
   R200s = npzfile["R200s"]

   low_mass_vels = []
   active_vels = []
   passive_vels = []
   virial_vels = []
   low_mass_vir = []
   active_vir = []
   passive_vir = []

   for i in range(np.size(v_dot_r_arr)):
      if (stellar_masses[i] <  8.0):
         continue
      curr_vir_vel = (np.sqrt((G*10.**halo_masses[i]*sol_mass_to_g)/(R200s[i]*parsec_to_cm*1.e3)))*1.e-5 # km/s
      if stellar_masses[i] <= 9.7:
         low_mass_vels.append(v_dot_r_arr[i])
         low_mass_vir.append(v_dot_r_arr[i]/curr_vir_vel)
      elif sSFRs[i] <= -11.:
         passive_vels.append(v_dot_r_arr[i])
         passive_vir.append(v_dot_r_arr[i]/curr_vir_vel)
      else:
         active_vels.append(v_dot_r_arr[i])
         active_vir.append(v_dot_r_arr[i]/curr_vir_vel)
      virial_vels.append(v_dot_r_arr[i]/curr_vir_vel)

   virial_vels = list(itertools.chain.from_iterable(virial_vels))
   low_mass_vir = list(itertools.chain.from_iterable(low_mass_vir))
   active_vir = list(itertools.chain.from_iterable(active_vir))
   passive_vir = list(itertools.chain.from_iterable(passive_vir))
   low_mass_vels = list(itertools.chain.from_iterable(low_mass_vels))
   active_vels = list(itertools.chain.from_iterable(active_vels))
   passive_vels = list(itertools.chain.from_iterable(passive_vels))

   print np.size(np.where(np.array(low_mass_vir, copy=False) > 1.5))/float(len(low_mass_vir))
   print np.size(np.where(np.array(active_vir, copy=False) > 1.5))/float(len(active_vir))
   print np.size(np.where(np.array(passive_vir, copy=False) > 1.5))/float(len(passive_vir))
   print ''
   
   normed = True
   bins = np.arange(-5., 5.25, 0.25)
   # only want to shows bins with more than 10 particles
   low_mask = np.zeros(np.size(low_mass_vir)) + 1
   active_mask = np.zeros(np.size(active_vir)) + 1
   passive_mask = np.zeros(np.size(passive_vir)) + 1
   for i in range(np.size(bins)-1):
      indices = np.argwhere(((bins[i] < low_mass_vir) & (low_mass_vir < bins[i+1])))[:,0]
      if np.size(indices) < 10:
         low_mask[indices] = 0

      indices = np.argwhere(((bins[i] < active_vir) & (active_vir < bins[i+1])))[:,0]
      if np.size(indices) < 10:
         active_mask[indices] = 0

      indices = np.argwhere(((bins[i] < passive_vir) & (passive_vir < bins[i+1])))[:,0]
      if np.size(indices) < 10:
         passive_mask[indices] = 0

   low_mass_vir = list(itertools.compress(low_mass_vir, low_mask))
   active_vir = list(itertools.compress(active_vir, active_mask)) 
   passive_vir = list(itertools.compress(passive_vir, passive_mask)) 
   
   fig, ax = plt.subplots(1)
   ax.hist(virial_vels, bins=bins, density=normed)
   ax.set_title("Histogram of Virial Velocities")
   ax.set_xlabel(r"$v/v_{200}$")
   ax.set_ylim(10**(-5), 10**(0.5))
   ax.set_xlim(-5.,5.)
   if normed:
      ax.set_ylabel("Probability")
   else:
      ax.set_ylabel("Number of particles")
   ax.set_yscale("log")
   if normed:
      fig.savefig("vir_hist_normed.pdf", bbox_to_inches="tight")
   else:
      fig.savefig("vir_hist.pdf", bbox_to_inches="tight")
   plt.close(fig)

   colors = ['k', 'b', 'r']
   labels = ["Low Mass", "Active", "Passive"]
   populations = [low_mass_vir, active_vir, passive_vir]

   fig, ax = plt.subplots(1)
   normed = True
   for i in range(len(colors)):
      ax.hist(populations[i], bins=bins, color=colors[i], alpha=0.33, label=labels[i], density=normed)
   ax.set_title("Histogram of Virial Velocities")
   ax.set_xlabel(r"$v/v_{200}$")
   ax.legend(loc="upper left")
   ax.set_ylim(10**(-5), 10**(0.5))
   ax.set_xlim(-5.,5.)
   if normed:
      ax.set_ylabel("Probability")
   else:
      ax.set_ylabel("Number of particles")
   ax.set_yscale("log")
   if normed:
      fig.savefig("vir_hist_pops_normed.pdf", bbox_to_inches="tight")
   else:
      fig.savefig("vir_hist_pops.pdf", bbox_to_inches="tight")
   plt.close(fig)


   bins = np.arange(-950., 1000., 50.) # max of all vels is 910, min smaller in abs val
   fig, ax = plt.subplots(1)
   ax.hist([low_mass_vels, active_vels, passive_vels], bins=bins, color=colors, label=labels, density=normed)
   ax.legend(loc= "upper left")
   ax.set_title("Histogram of Velocities")
   ax.set_xlabel(r"v (km/s)")
   if normed:
      ax.set_ylabel("Probability (Norm per Pop)")
   else:
      ax.set_ylabel("Number of particles")
   ax.set_yscale("log")
   if normed:
      fig.savefig("vel_hist_normed.pdf", bbox_to_inches="tight")
   else:
      fig.savefig("vel_hist.pdf", bbox_to_inches="tight")
   plt.close(fig)




