import numpy as np 
import SpecwizardFunctions

data_directory = '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.06_x001_data001_sh1'
# # by radii plots
#radii_bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
radii_bins = np.array([20.0,30,45.0,55.0,70.0,80.0,95.0,105.])
# by mass plots
# mass_bins = np.array([8.0,9.4,9.85,10.6,12.0])
# mass_colors = np.array(['b','c','g','r'])
max_radius = 1.5
min_radius = 0.0
# max_radius = 120.
# min_radius = 0.
mass_bins = np.array([10.4,11.0,11.5,12.0,12.4,12.8,13.2])
mass_colors = np.array(['m','b','c','g','r','k'])
SpecwizardFunctions.plot_for_multiple_gals_by_radius(data_directory, ion = 'h1', radii_bins = radii_bins, virial_vel_bool=False, virial_radii_bool=False,
	mean_spectra_bool = False, col_dense_bool = True, covering_frac_val = 16.0)
# SpecwizardFunctions.plot_for_multiple_gals_by_mass(data_directory, ion = 'h1', mass_bins = mass_bins, mass_colors=mass_colors, min_radius = min_radius,
# 												   max_radius=max_radius, stellar_mass_bool=False, virial_vel_bool = True, virial_radii_bool=True)