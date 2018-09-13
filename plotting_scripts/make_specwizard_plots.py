import SpecwizardFunctions
import numpy as np
import os
import sys

gal_output_file = sys.argv[1]
spec_file = sys.argv[2]
output_file_location = sys.argv[3]
max_radius = float(sys.argv[4])
points_per_radius = int(sys.argv[5])


bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
ions = np.array(['h1', 'c2', 'c3', 'c4', 'o6', 'mg2', 'si2', 'si3', 'si4', 'n5', 'ne8'])
max_abs_vel = 1600 # km/s

for ion in ions:
	os.makedirs('curr_spectra/' + ion + '/mean/binned/')
	os.makedirs('curr_spectra/' + ion + '/mean/by_radii')
	os.makedirs('curr_spectra/' + ion + '/stacked/binned/')
	os.makedirs('curr_spectra/' + ion + '/stacked/by_radii')

#SpecwizardFunctions.add_radii_to_specwizard_output(spec_file, gal_output_file, max_radius, points_per_radius) 

SpecwizardFunctions.plot_spectra_binned_multiple_ions(ions, bins, spec_file, gal_output_file, max_abs_vel, output_file_location)

SpecwizardFunctions.plot_spectra_by_output_radii_multiple_ions(ions, spec_file, gal_output_file, max_abs_vel, output_file_location)
