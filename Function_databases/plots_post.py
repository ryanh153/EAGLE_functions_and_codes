### imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import bisect
import os
import subprocess

do_cum_mass_plots = True
do_kin_plots = False

if do_cum_mass_plots:
   import spec_cum_data_semi as mock_spectra_data
if do_kin_plots:
   import survey_realization_functions

### Constants
omega_b = 0.04825
omega_m = 0.307

### Set plot parameters

plt.rcParams['axes.labelsize'], plt.rcParams['axes.titlesize'], plt.rcParams['legend.fontsize'] = 18., 20., 16.
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

   upper_mass = np.array([9.7, 15., 15.])
   lower_mass = np.array([5., 9.7, 9.7])

   upper_ssfr = np.array([-5., -5., -11.])
   lower_ssfr = np.array([-15., -11., -15.])

   labels = ['Low Mass', 'Active', 'Passive']
   labels2 = ['Total H', 'Neutral H']
   temp_labels = ['Total', r'Cool (r$<10^{5}$)']
   colors = ['k', 'b', 'r']
   stagger = [0.0, -1.5, 1.5]

   mean_bool = False
   cool_bool = True
   to_virial = True

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
   low_indices = np.array([2,27])
   mask = np.ones(np.size(sim_data.halo_masses),dtype=bool)
   mask[low_indices] = 0
   ann_masses, neut_ann_masses, halo_masses, stellar_masses, sSFRs, R200s = sim_data.ann_masses[mask,:], sim_data.neut_ann_masses[mask,:], sim_data.halo_masses[mask], sim_data.stellar_masses[mask], sim_data.sSFRs[mask], sim_data.R200s[mask]
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

   print mock_cum_mass[:,-1]
   print ''
   print mock_neut_cum_mass[:,-1]

   ### pass on only ann masses out to the virial radius for each galaxy 
   radii = np.arange(0.,650.,5.)
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
   plot_cum_masses = np.sum(ann_masses, axis=1)
   plot_neut_cum_mases = np.sum(neut_ann_masses, axis=1)
   if cool_bool:
      cool_plot_cum_masses = np.sum(cool_ann_masses, axis=1)
      cool_plot_neut_cum_mases = np.sum(cool_neut_ann_masses, axis=1)

   ann_med_fig, ann_med_ax = plt.subplots()
   ann_med_plots = [[] for _ in range(7)]
   ann_med_plotsean_fig, ann_mean_ax = plt.subplots()
   cum_med_fig, cum_med_ax = plt.subplots()
   cum_mean_fig, cum_mean_ax = plt.subplots()
   mock_col_fig, mock_col_ax = plt.subplots()
   mock_col_plots = [[] for _ in xrange(7)]
   mock_ann_fig, mock_ann_ax = plt.subplots()
   mock_ann_plots = [[] for _ in range(8)]
   mock_cum_fig, mock_cum_ax = plt.subplots()
   col_comp_temp_fig, col_comp_temp_ax = plt.subplots()
   col_comp_ion_dens_fig, col_comp_ion_dens_ax = plt.subplots()
   col_comp_H_dens_fig, col_comp_H_dens_ax = plt.subplots()

   for i in range(np.size(upper_mass)):
      print ''
      print labels[i]
      print ''
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
      mock_cum_ax.fill_between(mock_radii[i], mock_neut_cum_mass_bot[i], mock_neut_cum_mass_top[i], color=colors[i], alpha=0.33)

      if i == 0:
         mock_ann_plots[i] = mock_ann_ax.errorbar(0, 0, yerr=[[1], [1]], color=colors[i], label=labels2[0], marker='.', linestyle='None')
      mock_ann_plots[2*i+1] = mock_ann_ax.errorbar(mock_radii[i]+stagger[i], mock_ann_mass[i], yerr=[mock_ann_mass[i]-mock_ann_mass_bot[i], mock_ann_mass_top[i]-mock_ann_mass[i]], color=colors[i], label=labels[i], marker='.', markersize = 12., linestyle='None')
      if i==0:
         mock_ann_plots[2*i+2] = mock_ann_ax.errorbar(mock_radii[i]+stagger[i]+1., mock_neut_ann_mass[i], yerr=[mock_neut_ann_mass[i]-mock_neut_ann_mass_bot[i], mock_neut_ann_mass_top[i]-mock_neut_ann_mass[i]], color=colors[i], marker='*', markersize = 12., linestyle='None', label=labels2[1])
      else:
         mock_ann_plots[2*i+2] = mock_ann_ax.errorbar(mock_radii[i]+stagger[i]+1., mock_neut_ann_mass[i], yerr=[mock_neut_ann_mass[i]-mock_neut_ann_mass_bot[i], mock_neut_ann_mass_top[i]-mock_neut_ann_mass[i]], color=colors[i], marker='*', markersize = 12., linestyle='None')

      if i == 0:
         mock_col_plots[i] = mock_col_ax.errorbar(0, 0, yerr=[[1], [1]], color=colors[i], label=labels2[0], marker='.', linestyle='None')
      mock_col_plots[2*i+1] =mock_col_ax.errorbar(mock_radii[i]+stagger[i], mock_cols[i], yerr=[mock_cols[i]-mock_cols_bot[i], mock_cols_top[i]-mock_cols[i]], color=colors[i], label=labels[i], marker='.', markersize = 12., linestyle='None')
      if i == 0:
         mock_col_plots[2*i+2] =mock_col_ax.errorbar(mock_radii[i]+stagger[i], mock_neut_cols[i], yerr=[mock_neut_cols[i]-mock_neut_cols_bot[i], mock_neut_cols_top[i]-mock_neut_cols[i]], color=colors[i], marker='*', markersize = 12., linestyle='None', label=labels2[1])
      else:   
         mock_col_plots[2*i+2] =mock_col_ax.errorbar(mock_radii[i]+stagger[i], mock_neut_cols[i], yerr=[mock_neut_cols[i]-mock_neut_cols_bot[i], mock_neut_cols_top[i]-mock_neut_cols[i]], color=colors[i], marker='*', markersize = 12., linestyle='None')

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

         print cum_mass_med[-1]

         if i==0:
            ann_med_plots[i] = ann_med_ax.errorbar(plot_radii+stagger[i], medians, yerr=[med_low, med_high], label=labels2[0], color=colors[i], marker='.', markersize = 12., linestyle='None')
         ann_med_plots[2*i+1] = ann_med_ax.errorbar(plot_radii+stagger[i], medians, yerr=[med_low, med_high], label=labels[i], color=colors[i], marker='.', markersize = 12., linestyle='None')

         if i == 0:
            cum_med_ax.plot(plot_radii, cum_mass_med, color=colors[i], label=labels2[0])
         cum_med_ax.plot(plot_radii, cum_mass_med, color=colors[i], label=labels[i])
         cum_med_ax.fill_between(plot_radii, cum_mass_bot_err, cum_mass_top_err, color=colors[i],alpha=0.33)

         neut_medians = np.median(neut_curr_ann_masses, axis=0)
         neut_med_low, neut_med_high = np.percentile(neut_curr_ann_masses, 14., axis=0), np.percentile(neut_curr_ann_masses, 86., axis=0)
         neut_cum_mass_med = np.cumsum(neut_medians)
         neut_cum_mass_bot_err = np.sqrt(np.cumsum(np.power(neut_medians-neut_med_low,2.)))
         neut_cum_mass_top_err = np.sqrt(np.cumsum(np.power(neut_med_high-neut_medians,2.)))

         neut_medians, neut_med_low, neut_med_high, neut_cum_mass_med, neut_cum_mass_bot_err, neut_cum_mass_top_err = \
         (neut_medians), (neut_med_low), (neut_med_high), (neut_cum_mass_med), (neut_cum_mass_med-neut_cum_mass_bot_err), (neut_cum_mass_top_err+neut_cum_mass_med)

         print neut_cum_mass_med[-1]

         if i==0:
            ann_med_plots[2*i+2] = ann_med_ax.errorbar(plot_radii+stagger[i], neut_medians, yerr=[neut_med_low, neut_med_high], color=colors[i], marker='*', markersize = 12., linestyle='None', label=labels2[1])
         else:
            ann_med_plots[2*i+2] = ann_med_ax.errorbar(plot_radii+stagger[i], neut_medians, yerr=[neut_med_low, neut_med_high], color=colors[i], marker='*', markersize = 12., linestyle='None')

         if i==0:
            cum_med_ax.plot(plot_radii, neut_cum_mass_med, color=colors[i], linestyle='dotted', label=labels2[1])
         else:
            cum_med_ax.plot(plot_radii, neut_cum_mass_med, color=colors[i], linestyle='dotted')
         cum_med_ax.fill_between(plot_radii, neut_cum_mass_bot_err, neut_cum_mass_top_err, color=colors[i],alpha=0.33)

   mock_cum_ax.errorbar(proch_radii, proch_cum_mass, yerr=proch_cum_mass_err, elinewidth=2, ecolor='k', marker='*', markersize=15., color='limegreen', label="P")
   mock_cum_ax.plot(proch_radii, proch_cum_mass, marker='*', markersize=15., color='limegreen', label="Prochaska 2017")
   mock_cum_ax.plot(145., 2.0e10, linestyle='None', marker='^', markersize=15., color='darkviolet', markeredgecolor='k', label='Werk 2014')

   mock_ann_plots[-1] = mock_ann_ax.errorbar(proch_radii+0.75, proch_ann_mass, yerr=proch_mass_err, color='limegreen', label='Prochaska 2017', markersize = 12., linestyle='None', marker='.')

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

   leg1 = plt.legend([lines[i] for i in [1,3,5]], [lines[i].get_label() for i in [1,3,5]], loc=[0.015,0.775])
   leg2 = plt.legend([lines[i] for i in [0,2]], [lines[i].get_label() for i in [0,2]], loc=[0.35,0.843])
   leg3 = plt.legend([lines[i] for i in [9,11]], [lines[i].get_label() for i in [10,11]], loc=[0.585,0.02], numpoints=1)
   mock_cum_ax.add_artist(leg1)
   mock_cum_ax.add_artist(leg2)
   mock_cum_ax.add_artist(leg3)
   mock_cum_ax.set_xlabel('Radius (kpc)')
   mock_cum_ax.set_ylabel(r'$M_{cum}$ $(M_{\odot})$')
   mock_cum_ax.set_title('Cumulative H Mass vs Radius')
   mock_cum_ax.set_yscale('log')
   mock_cum_ax.set_ylim([10**4.0,10**12.5])
   mock_cum_ax.set_xlim([20.,160.])
   mock_cum_ax.yaxis.set_major_formatter(log_fmt)
   mock_cum_fig.savefig('jacknife_mass.pdf')
   plt.close(mock_cum_fig)

   # fancy legend
   leg1 = plt.legend([mock_ann_plots[i] for i in [1,3,5]], [mock_ann_plots[i].get_label() for i in [1,3,5]], loc=[0.02,0.775])
   leg2 = plt.legend([mock_ann_plots[i] for i in [0,2,7]], [mock_ann_plots[i].get_label() for i in [0,2,7]], loc = [0.55,0.775])
   mock_ann_ax.add_artist(leg1)
   mock_ann_ax.add_artist(leg2)
   mock_ann_ax.set_xlabel('Radius (kpc)')
   mock_ann_ax.set_ylabel(r'$M_{ann}$ $(M_{\odot})$')
   mock_ann_ax.set_title('Annular H Mass vs Radius')
   mock_ann_ax.set_yscale('log')
   mock_ann_ax.set_ylim([10**3.0,10**12.5])
   mock_ann_ax.set_xlim([20.,160.])
   mock_ann_fig.savefig('mock_ann_mass.pdf')
   plt.close(mock_ann_fig)

   leg1 = plt.legend([mock_col_plots[i] for i in [1,3,5]], [mock_col_plots[i].get_label() for i in [1,3,5]], loc = [0.02, 0.02])
   leg2 = plt.legend([mock_col_plots[i] for i in [0,2]], [mock_col_plots[i].get_label() for i in [0,2]], loc=[0.35,0.02])
   mock_col_ax.add_artist(leg1)
   mock_col_ax.add_artist(leg2)
   mock_col_ax.set_xlabel('Radius (kpc)')
   mock_col_ax.set_ylabel(r'$N$ $cm^{-2}$')
   mock_col_ax.set_title('Column Density vs Radius')
   mock_col_ax.set_yscale('log')
   mock_col_ax.set_ylim([10.**12,10.**21])
   mock_col_ax.set_xlim([20.,160.])
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
         leg1 = plt.legend([ann_med_plots[i] for i in [1,3,5]], [ann_med_plots[i].get_label() for i in [1,3,5]], loc = [0,0.68])
         leg2 = plt.legend([ann_med_plots[i] for i in [0,2]], [ann_med_plots[i].get_label() for i in [0,2]], loc = [0.3,0.74])
         ann_med_ax.add_artist(leg1)
         ann_med_ax.add_artist(leg2)
         ann_med_ax.set_xlabel('Radius (kpc)')
         ann_med_ax.set_ylabel(r'$M_{ann}$ $(M_{\odot})$')
         ann_med_ax.set_title('H Mass in Each Annulus vs Radius')
         ann_med_ax.set_yscale('log')
         ann_med_ax.set_ylim([10**3.,10**12.5])
         ann_med_ax.set_xlim([20.,160.])
         ann_med_fig.savefig('ann_masses_median.pdf')
         plt.close(ann_med_fig)

         ### Fancy legend
         lines = cum_med_ax.get_lines()
         leg1 = plt.legend([lines[i] for i in [1,3,5]], [lines[i].get_label() for i in [1,3,5]], loc = [0.015,0.68])
         leg2 = plt.legend([lines[i] for i in [0,2]], [lines[i].get_label() for i in [0,2]], loc= [0.3,0.74])
         cum_med_ax.add_artist(leg1)
         cum_med_ax.add_artist(leg2)
         cum_med_ax.set_xlabel('Radius (kpc)')
         cum_med_ax.set_ylabel(r'$M_{cum}$ $(M_{\odot})$')
         cum_med_ax.set_title('Cumulative H Mass vs Radius')
         cum_med_ax.set_yscale('log')
         cum_med_ax.set_ylim([10**4.,10**12.2])
         cum_med_ax.set_xlim([20.,160.])
         cum_med_fig.savefig('cum_masses_median.pdf')
         plt.close(cum_med_fig)

   if to_virial:
      fig, ax = plt.subplots()
      plot_hmass = np.power(np.ones(np.size(halo_masses))*10.,halo_masses)
      baryonically_closed = plot_hmass*(omega_b/omega_m*0.752)
      ax.scatter(plot_hmass , plot_cum_masses, color='darkgreen', marker='o', edgecolors='k', linewidth=0.75, s=30., label='Total H')
      ax.scatter(plot_hmass , cool_plot_cum_masses, color='fuchsia', marker='s', edgecolors='k', linewidth=0.75, s=30., label='Cool H')
      ax.scatter(plot_hmass , plot_neut_cum_mases, color='gold', marker='^', edgecolors='k', linewidth=0.75, s=30., label='Neutral H')
      ax.plot(plot_hmass, baryonically_closed, color='k', label='H Needed For Closure')
      ax.legend(loc='lower right',fontsize=16.)
      ax.set_yscale('log')
      ax.set_xscale('log')
      ax.set_ylim([10**6, 10**13.])
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
      ax.legend(loc='lower center',fontsize=16., ncol=3)
      ax.set_yscale('log')
      ax.set_xscale('log')
      ax.set_ylim([10**-4, 10**0.5])
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


