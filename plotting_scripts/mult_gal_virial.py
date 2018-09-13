import numpy as np 
import EagleFunctions

data_directory = '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals'

# by radii plots
#radii_bins = np.array([0.0,55.0,100.0,160.0])
# radii_bins = np.array([0.0,160.0])

# by mass plots
#mass_bins = np.array([10.0,11.4,11.85,12.6,14.0])
# mass_bins = np.array([10.0,14.0])
#mass_bins = np.array([8.0, 9.4, 9.85, 10.6, 12.0])
# mass_colors = np.array(['b','c','g','r'])
max_radius = 160.0
min_radius = 0.0

radii_bins = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0])
mass_bins = np.array([10.5,11.0,11.5, 12.0, 12.5, 13.0])
mass_colors = np.array(['m','b','c','g','y','r'])

EagleFunctions.plot_virial_for_multiple_gals_by_mass(data_directory, mass_bins = mass_bins, mass_colors = mass_colors, virial_bool = True, stellar_mass_bool = False)
EagleFunctions.plot_energy_sources_for_multiple_gals_by_mass_gas_ratios(data_directory, mass_bins = mass_bins, mass_colors = mass_colors, virial_bool = True, stellar_mass_bool = False, energy_values_bool = True)
EagleFunctions.plot_energy_sources_for_multiple_gals_by_mass_gas_ratios(data_directory, mass_bins = mass_bins, mass_colors = mass_colors, virial_bool = True, stellar_mass_bool = False, energy_values_bool = False)
EagleFunctions.plot_energy_sources_for_multiple_gals_by_mass_bulk_ratios(data_directory, mass_bins = mass_bins, mass_colors = mass_colors, virial_bool = True, stellar_mass_bool = False, energy_values_bool = False)
EagleFunctions.plot_energy_sources_for_multiple_gals_by_mass_gas_to_DM(data_directory, mass_bins = mass_bins, mass_colors = mass_colors, virial_bool = True, stellar_mass_bool = False, energy_values_bool = False)
EagleFunctions.plot_energy_sources_for_multiple_gals_by_mass_gas_T_to_DM(data_directory, mass_bins = mass_bins, mass_colors = mass_colors, virial_bool = True, stellar_mass_bool = False, energy_values_bool = False)