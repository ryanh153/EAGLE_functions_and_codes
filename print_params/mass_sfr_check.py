import EagleFunctions
import numpy as np
import sys
import h5py
import glob

M_sol = 1.99e33
secs_per_year = 3.15576e7

# Rongmon's Galaxy props
cos_smass_data = [8.18787787,   8.33354416,   8.51305444,   8.53377744,   8.67590461,
8.94278847,   8.96148255,   8.97774601,   8.98739364,   8.99886369,
9.07921225,   9.14949108,   9.22331265,   9.24755452,   9.33851642,
9.34020383,   9.38248044,   9.39390006,   9.40637154,   9.47150015,
9.48652534,   9.49378503,   9.55790648,   9.56034587,   9.58859192,
9.61710455,   9.62000868,   9.64402038,   9.68204316,   9.69883032,
9.70309828,   9.72174887,   9.72441368,   9.7354116,    9.75204885,
9.76457639,   9.81940171,   9.82857968,   9.83622828,   9.8590409,
9.87351011,   9.93012719,   9.96441475,   9.97808091,  10.01395726,
10.06606576,  10.09681876,  10.10105959,  10.13005144,  10.13629731,
0.16179202,  10.16324056,  10.18318541,  10.23714978,  10.25914865,
10.33758386,  10.34894328,  10.39582417,  10.42583152,  10.50141622,
10.5215785,   10.54915885,  10.55501198,  10.63264772,  10.63719967,
10.77362586,  10.7889097,   10.80625631,  10.81049218,  10.8148662,
10.82282876,  10.85216302,  10.86962383,  10.92197873,  10.92522041,
10.95169399,  10.97880362,  11.02322691,  11.11466983,  11.23023444,
11.2481175,   11.31422807,  11.32984311,  11.35516442,  11.45914539,
11.54703883]

cos_ssfr_data = [ -9.11845029,  -9.83023429,  -9.8953207,   -8.65189081,  -9.72905027,
-9.6007832,   -9.76946669,  -9.58956902, -10.66646065,  -9.95118767,
-10.14276674, -10.13461025,  -9.38029476, -11.00195541,  -9.65432552,
-11.67019037,  -9.72289189,  -9.87099078,  -9.98454382, -11.4838662,
-10.11668083,  -9.54500144, -12.13278292, -10.29566414,  -9.85984125,
-9.94395078,  -9.8502092,   -9.85623413,  -9.93721655,  -9.62907588,
-9.41180724, -10.04194358, -10.90106048,  -9.69133414, -10.93979062,
-9.49923417,  -9.98668313, -12.09430602,  -9.38515143,  -9.83885321,
-9.2122369,  -10.12854529,  -9.85466103, -10.33393426, -12.07792327,
-10.04058178,  -9.75525242,  -9.83786005, -10.00645109, -10.20386667,
-9.52938726, -10.10446543,  -9.54770064, -10.43092181,  -9.7768698,
-9.58297227, -10.04957505,  -9.65087977, -10.01120229, -10.36324046,
-11.03841024,  -9.92776109, -10.02087604,  -9.48275944,  -9.58176451,
-11.66020918, -10.14745653, -12.21813622, -11.01862754, -10.1595684,
-11.89096006, -11.71128861, -11.92214445, -12.15533152, -10.05510468,
-10.17062651, -12.54585476, -12.03844013,  -9.83689797, -11.7523872,
-11.78280195, -11.55649111, -12.01008797, -12.14467714, -11.9823251,
-12.18260778]

### Make File
with h5py.File('cos_data_comparisons1.hdf5', 'w') as hf:
	cos_gals = hf.create_group('cos_gals')

	cos_smass = cos_gals.create_dataset('cos_smass', (np.size(cos_smass_data),), maxshape=(None,), data = cos_smass_data)
	cos_ssfr = cos_gals.create_dataset('cos_ssfr', (np.size(cos_ssfr_data),), maxshape=(None,), data = cos_ssfr_data)

	matching_gals = cos_gals.create_group('matching_gals')
	for i in range(0,np.size(cos_smass)):
		matching_gals.create_dataset('gal_'+str(i),(0,), maxshape=(None,))

	my_gals = hf.create_group('my_gals')

	my_smass = my_gals.create_dataset('my_smass', (0,), maxshape=(None,))
	my_ssfr = my_gals.create_dataset('my_ssfr',(0,), maxshape=(None,))
	data_type = h5py.special_dtype(vlen=str)
	my_directory = my_gals.create_dataset('my_directory', (0,), maxshape=(None,), dtype = data_type)
	my_subhalo = my_gals.create_dataset('my_subhalo', (0,), maxshape=(None,))
	my_gal_id = my_gals.create_dataset('my_gal_id', (0,), maxshape=(None,))

### Get Matches
output_files = glob.glob('/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/*/output*.hdf5')
for file in output_files:
	with h5py.File(file, 'r') as hf:
		GalaxyProperties = hf.get('GalaxyProperties')
		snap_directory = np.array(GalaxyProperties.get('snap_directory'))[0]
		group_number = np.array(GalaxyProperties.get('group_number'))[0]
		gal_stellar_mass = np.array(GalaxyProperties.get('gal_stellar_mass'))[0]
		gal_SFR = np.array(GalaxyProperties.get('gal_SFR'))[0]
		gal_sSFR = gal_SFR/gal_stellar_mass


	with h5py.File('cos_data_comparisons1.hdf5', 'r+') as hf:
		cos_gals = hf.get('cos_gals')
		matching_gals = cos_gals.get('matching_gals')

		my_gals = hf.get('my_gals')
		my_directory = my_gals.get('my_directory')
		my_smass = my_gals.get('my_smass')
		my_ssfr = my_gals.get('my_ssfr')
		my_subhalo = my_gals.get('my_subhalo')
		my_gal_id = my_gals.get('my_gal_id')

		tol = 0.2 # measurements have 0.2 decs errs
		gals = np.size(cos_ssfr_data)
		for i in range(0,gals):
			if ((cos_smass_data[i] > np.log10(gal_stellar_mass)-tol) & (cos_smass_data[i] < np.log10(gal_stellar_mass)+tol)):
				if ((cos_ssfr_data[i] < np.log10(gal_sSFR)*(1.0-tol)) & (cos_ssfr_data[i] > np.log10(gal_sSFR)*(1.0+tol))):
					print 'match'
					my_smass.resize(np.size(my_smass)+1, axis=0)
					my_smass[-1] = np.log10(gal_stellar_mass)

					my_ssfr.resize(np.size(my_ssfr)+1, axis=0)
					my_ssfr[-1] = np.log10(gal_sSFR)

					my_directory.resize(np.size(my_directory)+1, axis = 0)
					my_directory[-1] = snap_directory

					my_subhalo.resize(np.size(my_subhalo)+1, axis = 0)
					my_subhalo[-1] = group_number

					my_gal_id.resize(np.size(my_gal_id)+1, axis=0)
					if np.size(my_directory) == 1:
						my_gal_id[-1] = 0
					else:
						my_gal_id[-1] = my_gal_id[-2]+1

					curr_gal = matching_gals.get('gal_'+str(i))
					curr_gal.resize(np.size(curr_gal)+1, axis=0)
					curr_gal[-1] = my_gal_id[-1]

