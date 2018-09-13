import h5py 
import numpy as np
import SpecwizardFunctions
import EagleFunctions
import sys
import os
import subprocess
import glob
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# Rongmon's Galaxy props
# cos_smass_data = np.asarray([8.18787787,   8.33354416,   8.51305444,   8.53377744,   8.67590461]) # for tests

cos_smass_data = np.asarray([8.18787787,   8.33354416,   8.51305444,   8.53377744,   8.67590461,
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
11.2481175,   11.31422807,  11.32984311,  11.35516442,  11.45914539, 11.54703883,
9.92000008,  10.61999989,  10.11999989,  10.31999969,   9.81999969,
10.31999969,  10.81999969,  10.11999989,  10.72000027,  10.42000008,
10.52000046,   9.92000008,   9.92000008,  10.42000008,  10.22000027,
10.92000008,   9.92000008,  10.92000008,  10.52000046,  10.61999989,
10.31999969,  10.42000008,  10.02000046,  10.02000046,  10.52000046,
10.31999969,  10.11999989,  10.72000027,  10.22000027,  10.52000046,
10.11999989,   9.92000008,   9.92000008,   9.92000008,  10.11999989,
10.02000046,  10.22000027,  10.02000046,  10.52000046,   9.92000008,
10.31999969,   9.92000008,  10.72000027,  10.22000027,   9.92000008,
7.96999979,   7.96999979,   7.57000017,   8.56999969,   9.02999973,
9.02999973,   9.02999973,   9.06999969,  10.26000023,   9.64999962,
9.94999981,   9.10000038,   7.36999989,   7.36999989,   7.92999983,   9.25,
10.03999996,   7.59000015,   9.81000042,  10.96000004,   9.43999958,
11.27000046,   9.90999985,  10.27999973,   9.02000046,  10.68999958,
10.26000023,   8.52999973,  10.23999977,  10.52000046,  11.36999989,
9.92000008,   9.46000004,  10.23999977,  10.88000011,  10.77000046,
9.98999977,  10.86999989,  10.55000019,  10.5       ,  10.81999969,
10.53999996,  10.85000038,  11.  ,        10.38000011 , 10.77000046,
7.3499999 ,   9.63000011])

# cos_ssfr_data = np.asarray([ -9.11845029,  -9.83023429,  -9.8953207,   -8.65189081,  -9.72905027]) # for tests

cos_ssfr_data = np.asarray([ -9.11845029,  -9.83023429,  -9.8953207,   -8.65189081,  -9.72905027,
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
-11.78280195, -11.55649111, -12.01008797, -12.14467714, -11.9823251, -12.18260778,
-9.5,        -12.,         -10.10000038, -12.30000019,  -9.60000038,
-10.69999981, -12.60000038, -11.69999981, -10.30000019, -10.39999962,
-12.69999981, -11.89999962, -10.80000019, -10.10000038, -10.        , -11.89999962,
-10.69999981, -11.        , -10.89999962, -12.10000038, -12.        , -11.       ,  -12.,
-9.69999981, -10.39999962, -10.30000019, -10.        , -10.39999962,
-10.80000019, -12.30000019, -10.39999962,  -9.80000019, -10.5      ,  -11.80000019,
-12.30000019, -11.80000019, -12.        , -11.10000038, -12.69999981,
-9.80000019, -10.19999981,  -9.60000038, -11.89999962,  -9.60000038,
-12.10000038,
-9.17000008,    -9.17000008,    -9.78999996,    -9.15999985,    -9.32999992,
-9.32999992,    -9.32999992,    -8.72999954,    -9.89999962 ,   -9.93999958,
-10.02000046,    -9.68999958,   -10.05000019,   -10.02999973 ,  -11.14999962,
-9.85000038,    -9.89000034,    -9.88000011,    -9.31000042 ,  -10.17000008,
-10.46000004,   -11.80000019,    -9.90999985,   -10.39000034 ,  -10.13000011,
-9.26000023,    -9.46000004,    -9.97999954,   -11.67000008 ,  -10.18000031,
-11.31999969,    -9.31999969,    -8.93999958,  1000.         ,   -9.56000042,
-9.53999996,   -10.36999989,   -10.59000015,    -9.96000004 ,  -10.11999989,
-10.46000004,   -10.59000015,   -10.72999954,   -11.32999992 ,  -10.         , 1000.,
-9.75      ,    -9.43999958])

# cos_radii_data = np.asarray([77.98085002,   82.85343494,   76.85737969,  112.24468121,  134.63288495]) # for tests

cos_radii_data = np.asarray([77.98085002,   82.85343494,   76.85737969,  112.24468121,  134.63288495,
100.97865962,   83.06022428,   90.99245595,  110.31462131,  120.8950971,
149.17362215,   91.81011234,   90.67314237,   60.33845501,  43.70634583,
22.95573566,  44.27568518,   52.73828267,   18.26371425,   54.7385523,
19.14568634,  154.3851633,   132.28551322,   31.60220339,   19.65594407,
92.59809421,  112.13284071,   37.24195188,   87.17373271,   35.39584813,
32.26139323,   88.28653497,   82.73143092,   38.87860762,  142.84387818,
113.11055654,   47.06561111,  102.49789974,  150.01790289,   33.81329924,
100.61456673,   96.81503374,  116.39334879,   46.17003039,   52.22940924,
37.64658353,   77.24753045,   95.01508464,  125.13355257,  119.71348736,
102.07601666,  120.43253868,   58.0714979,    87.17211188,   90.03786035,
97.76284567,  135.92796744,  116.33226036,   90.27531913,   89.32814394,
134.39174397,  135.24517493,  105.62945733,   14.39471823,  101.11046547,
101.35764251,  114.03069403,   92.35297664,   58.27564925,  126.06769878,
54.70831159,   41.53788126,  124.66651322,   66.72192782,   53.16469939,
65.75027945,   37.53981442,   47.40981787,   28.90151523,   33.34913736,
37.86916926,  103.38357412,   15.4314477,    29.11587861,   39.8609142,
55.54569059,
102.,  215.,  178.,  170.,  189.,  214.,  226.,  160.,   63.,  226.,  162.,  119.,
102.,  214.,  140.,   95.,  171.,  108.,  209.,  162.,  221.,  140.,  176.,  230.,
104.,  208.,  198.,  200.,  103.,  155.,  198.,  199.,  153.,  155.,  111.,  222.,
170.,  231.,  110.,  223.,  224.,  130.,  180.,  128.,  196.,
110.,   79.,   26.,   57.,  172.,   93.,   74.,   89.,   53.,   97.,   22.,   45.,
7.,   70.,   81.,  173.,  157.,   97.,  257.,  438. , 133.,  139.,  138.,  137.,
72.,  505.,   93.,  114.,  138.,  269.,  502.,  309.,  237.,   35.,  228.,  354.,
239.,  381.,  378.,  197.,  117.,  261.,  227.,  353. , 425.,  403.,   70.,   55.])

cos_columns = [14.24865913,  15.63188839, 14.78208637,  14.75774479,  15.33683872,
  15.45203018,  19.54999924,  19.35000038,  12.67929745,  15.42936611,
  14.8832655,   16.29372406,  16.18286705,  15.24644375,  18.,          16.58548927,
  15.44044495,  15.7924633 ,  18.61000061,  15.09130478,  15.12010098,
  15.82603836,  16.28199196,  15.56638622,  16.05509949,  15.30142689,
  14.75970173,  16.33325768,  15.87655735,  16.26819038,  12.43311596,
  15.42364216,  15.25041008,  19.79999924,  14.52999973,  15.07318592,
  14.82578754,  16.49802017,  13.85509682,  15.73176479,  13.12020779,
  14.95844746,  12.53103065,  15.99501228,  14.57126617,  14.70775032,
  14.78208828,  14.10789394,  13.77996159,  14.26246071,  13.07916832,
  13.63074875,  14.43175411,  14.32103443,  14.37844849,  14.46681976,
  13.55505848,  14.65403938,  12.70161247,  14.22205639,  12.91422749,
  13.91991234,  12.728899  ,  14.75007343,  14.31647873,  14.09795284,
  12.86601448,  13.94899273,  13.0870285 ,  12.77795982,  14.25744343,
  14.23258972,  12.58767033,  14.33285141,  13.57776546,  14.53046608,
  14.65308857,  14.42543316,  14.55914402,  13.87470436,  14.40061378,
  14.22891998,  14.83675194,  14.64227962,  14.16492081,  14.54248905]


def make_cos_data_comparison_file(cos_comparisons_file, cos_smass_data, cos_ssfr_data,):
	with h5py.File(cos_comparisons_file, 'w') as hf:
		cos_gals = hf.create_group('cos_gals')

		cos_smass = cos_gals.create_dataset('cos_smass', (np.size(cos_smass_data),), maxshape=(None,), data = cos_smass_data)
		cos_ssfr = cos_gals.create_dataset('cos_ssfr', (np.size(cos_ssfr_data),), maxshape=(None,), data = cos_ssfr_data)
		cos_radii = cos_gals.create_dataset('cos_radii', (np.size(cos_radii_data),), maxshape=(None,), data = cos_radii_data)

		matching_gals = cos_gals.create_group('matching_gals')
		for i in range(0,np.size(cos_smass)):
			matching_gals.create_dataset('gal_'+str(i),(0,), maxshape=(None,))

		my_gals = hf.create_group('my_gals')

		my_smass = my_gals.create_dataset('my_smass', (0,), maxshape=(None,))
		my_smass.attrs['units'] = 'solar masses'

		my_ssfr = my_gals.create_dataset('my_ssfr',(0,), maxshape=(None,))
		my_ssfr.attrs['units'] = 'per year (fraction of galaxy mass formed)'

		data_type = h5py.special_dtype(vlen=str)
		my_directory = my_gals.create_dataset('my_directory', (0,), maxshape=(None,), dtype = data_type)

		my_subhalo = my_gals.create_dataset('my_subhalo', (0,), maxshape=(None,))
		my_gal_id = my_gals.create_dataset('my_gal_id', (0,), maxshape=(None,))


		my_keyword = my_gals.create_dataset('my_keyword', (0,), maxshape=(None,),dtype = data_type)
	
def populate_hdf5_with_matching_gals(cos_comparison_file, directory_with_gal_folders, starting_gal_id):
	if directory_with_gal_folders[-1] != '/':
		directory_with_gal_folders += '/'
	output_files = glob.glob(directory_with_gal_folders + '*/output*.hdf5')
	for file in output_files:
		with h5py.File(file, 'r') as hf:
			GalaxyProperties = hf.get('GalaxyProperties')
			snap_directory = np.array(GalaxyProperties.get('snap_directory'))[0]
			group_number = np.array(GalaxyProperties.get('group_number'))[0]
			gal_stellar_mass = np.array(GalaxyProperties.get('gal_stellar_mass'))[0]
			gal_SFR = np.array(GalaxyProperties.get('gal_SFR'))[0]
			gal_SFR = gal_SFR
			gal_sSFR = gal_SFR/gal_stellar_mass
			keyword_end = np.array(GalaxyProperties.get('file_keyword'))[0][-12::]


		with h5py.File(cos_comparisons_file, 'r+') as hf:
			cos_gals = hf.get('cos_gals')
			matching_gals = cos_gals.get('matching_gals')

			my_gals = hf.get('my_gals')
			my_directory = my_gals.get('my_directory')
			my_smass = my_gals.get('my_smass')
			my_ssfr = my_gals.get('my_ssfr')
			my_subhalo = my_gals.get('my_subhalo')
			my_gal_id = my_gals.get('my_gal_id')
			my_keyword = my_gals.get('my_keyword')

			tol = 0.2 # measurements have 0.2 decs errs
			gals = np.size(cos_ssfr_data)
			for i in range(0,gals):
				if ((cos_smass_data[i] > np.log10(gal_stellar_mass)-tol) & (cos_smass_data[i] < np.log10(gal_stellar_mass)+tol)):
					if ((cos_ssfr_data[i] < np.log10(gal_sSFR)*(1.0-tol)) & (cos_ssfr_data[i] > np.log10(gal_sSFR)*(1.0+tol))):

						curr_directory = snap_directory
						curr_subhalo = group_number

						if (np.size(my_directory) == 0):
							my_smass.resize(np.size(my_smass)+1, axis=0)
							my_smass[-1] = np.log10(gal_stellar_mass)

							my_ssfr.resize(np.size(my_ssfr)+1, axis=0)
							my_ssfr[-1] = np.log10(gal_sSFR)

							my_directory.resize(np.size(my_directory)+1, axis = 0)
							my_directory[-1] = snap_directory

							my_subhalo.resize(np.size(my_subhalo)+1, axis = 0)
							my_subhalo[-1] = group_number

							my_keyword.resize(np.size(my_keyword)+1, axis= 0)
							my_keyword[-1] = keyword_end

							my_gal_id.resize(np.size(my_gal_id)+1, axis=0)
							my_gal_id[-1] = starting_gal_id
							curr_gal_id = my_gal_id[-1]

						else:
							if np.size(np.argwhere((curr_directory == np.array(my_directory))&(curr_subhalo == np.array(my_subhalo)))) != 0:
								index = np.argwhere((curr_directory == np.array(my_directory))&(curr_subhalo == np.array(my_subhalo)))[0][0]
								curr_gal_id = np.array(my_gal_id)[index]

							else:
								my_smass.resize(np.size(my_smass)+1, axis=0)
								my_smass[-1] = np.log10(gal_stellar_mass)

								my_ssfr.resize(np.size(my_ssfr)+1, axis=0)
								my_ssfr[-1] = np.log10(gal_sSFR)

								my_directory.resize(np.size(my_directory)+1, axis = 0)
								my_directory[-1] = snap_directory

								my_subhalo.resize(np.size(my_subhalo)+1, axis = 0)
								my_subhalo[-1] = group_number

								my_keyword.resize(np.size(my_keyword)+1, axis= 0)
								my_keyword[-1] = keyword_end

								my_gal_id.resize(np.size(my_gal_id)+1, axis=0)
								my_gal_id[-1] = my_gal_id[-2]+1
								curr_gal_id = my_gal_id[-1]

						curr_gal = matching_gals.get('gal_'+str(i))
						curr_gal.resize(np.size(curr_gal)+1, axis=0)
						curr_gal[-1] = curr_gal_id


def run_specwizard_for_matching_gals(cos_comparisons_file, directory_with_gal_folders,random_radii, min_radius=None,max_radius=None):
	realizations = 2

	if directory_with_gal_folders[-1] != '/':
		directory_with_gal_folders += '/'
	output_files = glob.glob(directory_with_gal_folders + '*/output*.hdf5')

	with h5py.File(cos_comparisons_file, 'r+') as hf:
		cos_gals = hf.get('cos_gals')
		cos_radii = np.array(cos_gals.get('cos_radii'))
		matching_gals = cos_gals.get('matching_gals')

		my_gals = hf.get('my_gals')
		my_gal_id = np.array(my_gals.get('my_gal_id'))
		my_directory = np.array(my_gals.get('my_directory'))
		my_subhalo = np.array(my_gals.get('my_subhalo'))

		calls = 0
		max_calls = 10000
		gals_used = 0
		for n in range(0,realizations):
			# if gals_used > max_calls:
			# 	break
			for i in range(0,np.size(matching_gals.keys())):
				# e
				curr_gal =  np.array(matching_gals.get('gal_'+str(i)))
				if np.size(curr_gal) > 0:
					gal_id = np.random.choice(np.array(curr_gal))
					gal_index = np.argwhere(gal_id==np.array(my_gal_id))[0][0]
					gal_directory = my_directory[gal_index]
					gal_subhalo = my_subhalo[gal_index]
					for j in range(0,np.size(output_files)):
						with h5py.File(output_files[j], 'r') as hf1:
							GalaxyProperties = hf1.get('GalaxyProperties')
							temp_snap_directory = np.array(GalaxyProperties.get('snap_directory'))[0]
							temp_subhalo_num = np.array(GalaxyProperties.get('group_number'))[0]

							if ((temp_snap_directory==gal_directory) & (temp_subhalo_num==gal_subhalo)):
								temp_smass = np.array(GalaxyProperties.get('gal_stellar_mass'))[0]
								keyword_end = np.array(GalaxyProperties.get('file_keyword'))[0][-12::]
								gal_folder = 'snapshot_'+keyword_end
								particles_included_keyword = 'snap_noneq_' + keyword_end
								snap_base = 'snap_noneq_'+keyword_end
								group_included_keyword = 'group_tab_' + keyword_end
								subfind_included_keyword = 'eagle_subfind_tab_' + keyword_end
								os.chdir("../Ali_Spec_src/")
								axis = np.random.choice(np.asarray([0,1,2]))
								filename = 'los_'+str(int(gal_id))+'.txt'
								gals_used += 1

								if os.path.isfile(filename):
									SpecwizardFunctions.add_los_to_text(filename,gal_directory+'/',gal_subhalo, particles_included_keyword,
										                                group_included_keyword, subfind_included_keyword, radius=cos_radii[i],axis=axis)
								else:
									if random_radii:
										SpecwizardFunctions.create_one_los_text_random(filename,gal_directory+'/',gal_subhalo, particles_included_keyword,
											                                group_included_keyword, subfind_included_keyword,min_radius,max_radius,axis=axis)
									else:
										SpecwizardFunctions.create_one_los_text(filename,gal_directory+'/',gal_subhalo, particles_included_keyword,
										                                group_included_keyword, subfind_included_keyword, radius=cos_radii[i] ,axis=axis)
		

		los_coord_files = glob.glob('los_*.txt')

		for i in range(0,np.size(los_coord_files)):
			os.chdir('../snapshots')
			with h5py.File(cos_comparisons_file, 'r+') as hf:
				my_gals = hf.get('my_gals')
				my_directory = my_gals.get('my_directory')
				my_gal_id = my_gals.get('my_gal_id')
				my_keyword = my_gals.get('my_keyword')
				my_group_numbers = my_gals.get('my_subhalo')

				curr_id = int(los_coord_files[i][4:-4])

				index = np.argwhere(curr_id == np.array(my_gal_id))[0][0]
				gal_directory = np.asarray(my_directory)[index]
				gal_subhalo = np.asarray(my_group_numbers)[index]
				keyword_end = np.asarray(my_keyword)[index]
				gal_folder = 'snapshot_'+str(keyword_end)
				snap_base = 'snap_noneq_' + str(keyword_end)
				gal_id = int(np.asarray(my_gal_id)[index])
				for j in range(0,np.size(output_files)):
					found_file = False
					with h5py.File(output_files[j], 'r') as hf1:
						GalaxyProperties = hf1.get('GalaxyProperties')
						temp_snap_directory = np.array(GalaxyProperties.get('snap_directory'))[0]
						temp_subhalo_num = np.array(GalaxyProperties.get('group_number'))[0]

						if ((temp_snap_directory==gal_directory) & (temp_subhalo_num==gal_subhalo)):
							found_file = True
							temp_smass = np.array(GalaxyProperties.get('gal_stellar_mass'))[0]
							subprocess.call("cp " + output_files[j] + " ../Ali_Spec_src/spec_outputs/gal_output_"+str(gal_id)+".hdf5", shell=True)
					if found_file:
						break


				os.chdir('../Ali_Spec_src')
				EagleFunctions.edit_text('mybox.par', 'curr_params.par',['datadir','snap_base','los_coordinates_file'],
									['datadir = ' + gal_directory + '/' + gal_folder + '/', 'snap_base = ' + snap_base,'los_coordinates_file = ./'+str(los_coord_files[i])])
				subprocess.call("./specwizard_Equ curr_params.par", shell=True)
				subprocess.call("mv spec.snap_noneq* spec_outputs/spec.snap_"+str(gal_id)+".hdf5", shell=True)
				subprocess.call("cp curr_params.par spec_outputs/param_"+str(gal_id)+".par", shell=True)
				subprocess.call("mv los_"+str(gal_id)+".txt spec_outputs/",shell=True)
				calls += 1
			if calls >= max_calls:
				print 'broke because of too many calls'
				break

	print 'num of calls'
	print calls
	return calls

def make_col_dense_plots(ion, calls, spec_output_directory, covering_frac_bool, covering_frac_val):
	if spec_output_directory[-1] != '/':
		spec_output_directory += '/'

	masses = np.array([])
	radii = np.array([])
	virial_radii = np.array([])
	cols = np.array([])
	covered = 0
	total = 0
	offset = 130

	los_files = glob.glob(spec_output_directory + 'los*')
	print los_files
	for i in range(0,np.size(los_files)):	

		los_file = spec_output_directory + 'los_'+str(i+offset)+'.txt'
		print los_file
		gal_output_file = spec_output_directory + 'gal_output_'+str(i+offset)+'.hdf5'
		spec_output_file = spec_output_directory + 'spec.snap_'+str(i+offset)+'.hdf5'
		if os.path.isfile(spec_output_file) == False:
			print i
			print 'no spec file for that number'
			continue

		with h5py.File(gal_output_file, 'r') as hf:
			galaxy_properties = hf.get("GalaxyProperties")
			gal_directory = np.array(galaxy_properties.get('snap_directory'))[0]
			gal_coords = np.array(galaxy_properties.get('gal_coords'))[0]
			box_size = np.array(galaxy_properties.get("box_size"))[0]
			gal_mass = np.array(galaxy_properties.get('gal_mass'))[0]
			gal_R200 = np.array(galaxy_properties.get('gal_R200'))[0]

		lines = np.genfromtxt(los_file, skip_header=1)
		gal = np.array([gal_coords[0], gal_coords[1], gal_coords[2]])/box_size
		if np.size(lines) > 4:
			spec_num = 0
			for line in lines:
				radius = np.sqrt(np.sum(np.power((line[0:3]-gal[0:3]),2)))
				if radius > 0.5:
					radius = (1.0-radius)*box_size
				else:
					radius *= box_size

				with h5py.File(spec_output_file,'r') as hf:
					spectrum = hf.get('Spectrum'+str(spec_num))
					curr_ion = spectrum.get(ion)
					col_dense = np.array(curr_ion.get('LogTotalIonColumnDensity'))
					total += 1
					if covering_frac_bool:
						if col_dense > covering_frac_val:
							covered += 1

				masses = np.concatenate((masses,np.array([gal_mass])))
				radii = np.concatenate((radii,np.array([radius])))
				virial_radii = np.concatenate((virial_radii,np.array([radius/gal_R200])))
				cols = np.concatenate((cols,np.array([col_dense])))
				spec_num += 1

		else:
			spec_num = 0
			radius = np.sqrt(np.sum(np.power((lines[0:3]-gal[0:3]),2)))
			if radius > 0.5:
				radius = (1.0-radius)*box_size
			else:
				radius *= box_size

			with h5py.File(spec_output_file,'r') as hf:
				spectrum = hf.get('Spectrum'+str(spec_num))
				curr_ion = spectrum.get(ion)
				col_dense = np.array(curr_ion.get('LogTotalIonColumnDensity'))
				total += 1
				if covering_frac_bool:
					if col_dense > covering_frac_val:
						covered += 1

				masses = np.concatenate((masses,np.array([gal_mass])))
				radii = np.concatenate((radii,np.array([radius])))
				virial_radii = np.concatenate((virial_radii,np.array([radius/gal_R200])))
				cols = np.concatenate((cols,np.array([col_dense])))
				spec_num += 1
			spec_num += 1

	masses = masses[masses!=0]
	radii = radii[radii!=0]
	print 'in col dense'
	print radii
	virial_radii = virial_radii[virial_radii!=0]
	cols = cols[cols!=0]



	plt.scatter(radii, cols, c=np.log10(masses), cmap='magma')
	try:
		plt.title('Col Density vs Impact Param(frac covered at %.1f=%.2f)' % (covering_frac_val, float(covered)/total))
	except:
		plt.title('Col Density vs Impact Param')
	plt.xlabel('Impact Parameter (kpc)')
	plt.ylabel('Column Density (log10(cm^2)')
	plt.colorbar()
	plt.grid()
	plt.savefig(ion+'_with_color.png')
	plt.close()

	plt.scatter(virial_radii, cols, c=np.log10(masses), cmap='magma')
	try:
		plt.title('Col Density vs Impact Param(frac covered at %.1f=%.2f)' % (covering_frac_val, float(covered)/total))
	except:
		plt.title('Col Density vs Impact Param')
	plt.xlabel('Impact Parameter (in virial radii)')
	plt.ylabel('Column Density (log10(cm^2)')
	plt.colorbar()
	plt.grid()
	plt.savefig(ion+'_virial_with_color.png')
	plt.close()

def make_contour_col_dense_plots(ion, calls, spec_output_directory, covering_frac_bool, covering_frac_val):
	if spec_output_directory[-1] != '/':
		spec_output_directory += '/'

	masses = np.array([])
	radii = np.array([])
	virial_radii = np.array([])
	cols = np.array([])
	covered = 0
	total = 0
	offset = 130

	los_files = glob.glob(spec_output_directory +  'los*')

	for i in range(0,np.size(los_files)):	

		los_file = spec_output_directory +  'los_'+str(i+offset)+'.txt'
		gal_output_file = spec_output_directory +  'gal_output_'+str(i+offset)+'.hdf5'
		spec_output_file = spec_output_directory +  'spec.snap_'+str(i+offset)+'.hdf5'
		if os.path.isfile(spec_output_file) == False:
			print i
			print 'no spec file for that number'
			continue

		with h5py.File(gal_output_file, 'r') as hf:
			galaxy_properties = hf.get("GalaxyProperties")
			gal_directory = np.array(galaxy_properties.get('snap_directory'))[0]
			gal_coords = np.array(galaxy_properties.get('gal_coords'))[0]
			box_size = np.array(galaxy_properties.get("box_size"))[0]
			gal_mass = np.array(galaxy_properties.get('gal_mass'))[0]
			gal_R200 = np.array(galaxy_properties.get('gal_R200'))[0]

		lines = np.genfromtxt(los_file, skip_header=1)
		gal = np.array([gal_coords[0], gal_coords[1], gal_coords[2]])/box_size
		if np.size(lines) > 4:
			spec_num = 0
			for line in lines:
				radius = np.sqrt(np.sum(np.power((line[0:3]-gal[0:3]),2)))
				if radius > 0.5:
					radius = (1.0-radius)*box_size
				else:
					radius *= box_size

				with h5py.File(spec_output_file,'r') as hf:
					spectrum = hf.get('Spectrum'+str(spec_num))
					curr_ion = spectrum.get(ion)
					col_dense = np.array(curr_ion.get('LogTotalIonColumnDensity'))
					total += 1
					if covering_frac_bool:
						if col_dense > covering_frac_val:
							covered += 1

				masses = np.concatenate((masses,np.array([gal_mass])))
				radii = np.concatenate((radii,np.array([radius])))
				virial_radii = np.concatenate((virial_radii,np.array([radius/gal_R200])))
				cols = np.concatenate((cols,np.array([col_dense])))
				spec_num += 1

		else:
			spec_num = 0
			radius = np.sqrt(np.sum(np.power((lines[0:3]-gal[0:3]),2)))
			if radius > 0.5:
				radius = (1.0-radius)*box_size
			else:
				radius *= box_size

			with h5py.File(spec_output_file,'r') as hf:
				spectrum = hf.get('Spectrum'+str(spec_num))
				curr_ion = spectrum.get(ion)
				col_dense = np.array(curr_ion.get('LogTotalIonColumnDensity'))
				total += 1
				if covering_frac_bool:
					if col_dense > covering_frac_val:
						covered += 1

				masses = np.concatenate((masses,np.array([gal_mass])))
				radii = np.concatenate((radii,np.array([radius])))
				virial_radii = np.concatenate((virial_radii,np.array([radius/gal_R200])))
				cols = np.concatenate((cols,np.array([col_dense])))
				spec_num += 1
			spec_num += 1

	masses = masses[masses!=0]
	radii = radii[radii!=0]
	virial_radii = virial_radii[virial_radii!=0]
	cols = cols[cols!=0]


	num_pts = np.size(radii)
	H, ybins, xbins, image = plt.hist2d(virial_radii,cols,bins=30)
	plt.colorbar()
	# plt.hold(True)
	# plt.plot(cos_radii_data, cos_columns)
	# plt.hold(False)
	plt.savefig('contour_test.png')
	plt.close()


	extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
	levels = (4., 8., 12.)
	contour = plt.contour(H,levels, extent = extent)
	plt.clabel(contour, inline=1)
	plt.colorbar()
	plt.savefig('contour_test2.png')
	plt.close()

	# plt.scatter(radii, cols, c=np.log10(masses), cmap='magma')
	# plt.title('Col Density vs Impact Param(frac covered at %.1f=%.2f)' % (covering_frac_val, float(covered)/total))
	# plt.xlabel('Impact Parameter (kpc)')
	# plt.ylabel('Column Density (log10(cm^2)')
	# plt.colorbar()
	# plt.grid()
	# plt.savefig(ion+'_with_color.png')
	# plt.close()

	# plt.scatter(virial_radii, cols, c=np.log10(masses), cmap='magma')
	# plt.title('Col Density vs Impact Param(frac covered at %.1f=%.2f)' % (covering_frac_val, float(covered)/total))
	# plt.xlabel('Impact Parameter (in virial radii)')
	# plt.ylabel('Column Density (log10(cm^2)')
	# plt.colorbar()
	# plt.grid()
	# plt.savefig(ion+'_virial_with_color.png')
	# plt.close()


def plot_for_multiple_gals_by_radius(spec_output_directory, calls, ion, radii_bins, max_abs_vel, virial_vel_bool, virial_radii_bool, 
	mean_spectra_bool, min_halo_mass,max_halo_mass):
	total_num = 0.
	num_covered = 0.
	col_dense_arr = np.array([])


	for j in range(0,np.size(radii_bins)-1):
		num_in_bin = 0
		final_ion_flux = np.array([])
		final_velocities = np.array([])
		offset = 130

		los_files = glob.glob(spec_output_directory + 'los*')

		for i in range(0,np.size(los_files)):	
			# if i> 5:
			# 	break

			los_file = spec_output_directory + 'los_'+str(i+offset)+'.txt'
			gal_output_file = spec_output_directory + 'gal_output_'+str(i+offset)+'.hdf5'
			spec_output_file = spec_output_directory + 'spec.snap_'+str(i+offset)+'.hdf5'
			if os.path.isfile(spec_output_file) == False:
				print i
				print 'no spec file for that number'
				continue
 
			with h5py.File(gal_output_file) as hf:
				GalaxyProperties = hf.get("GalaxyProperties")
				box_size = np.array(GalaxyProperties.get("box_size"))[0]
				gal_R200 = np.array(GalaxyProperties.get("gal_R200"))[0]
				gal_mass = np.array(GalaxyProperties.get("gal_mass"))[0]
				snap_directory = np.array(GalaxyProperties.get("snap_directory"))[0]
				gal_vel = np.array(GalaxyProperties.get("gal_velocity"))[0]
				gal_coords = np.array(GalaxyProperties.get("gal_coords"))[0]

			lines = np.genfromtxt(los_file, skip_header=1)
			gal = np.array([gal_coords[0], gal_coords[1], gal_coords[2]])/box_size
			if np.size(lines) > 4:
				spec_num = 0
				for line in lines:
					radius = np.sqrt(np.sum(np.power((line[0:3]-gal[0:3]),2)))
					if radius > 0.5:
						radius = (1.0-radius)*box_size
					else:
						radius *= box_size

					with h5py.File(spec_output_file) as hf:
						spec_hubble_velocity = np.array(hf.get('VHubble_KMpS'))
						curr_spectra_folder = hf.get('Spectrum'+str(spec_num))
						curr_spectra = curr_spectra_folder.get(ion)
						ion_flux = np.array(curr_spectra.get("Flux"))
						col_dense = np.array(curr_spectra.get("LotTotalIonColumnDensity"))
					if max_abs_vel != None:
						ion_flux = ion_flux[np.where(np.abs(final_vel_arr) < max_abs_vel)[0]]
						optical_depth = optical_depth[np.where(np.abs(final_vel_arr) < max_abs_vel)[0]]
					length_spectra = np.size(spec_hubble_velocity)
					max_box_vel = spec_hubble_velocity[-1]*(length_spectra+1)/(length_spectra)
					spec_num += 1

					H = max_box_vel/(box_size/1.e3) # stuff with H in Mpc
					gal_hubble_vel = (gal_coords[2]/1.e3)*H # switched gal coords to Mpc
					gal_vel_z = gal_vel[2]
					virial_vel = (200.0)*(gal_mass/5.0e12)**(1./3.)


					spec_hubble_velocity = spec_hubble_velocity - (max_box_vel/2.0) # add in peculiar velocity 
					#spec_hubble_velocity = spec_hubble_velocity - (gal_hubble_vel+gal_vel_z) # old. Before new spec (now it centers for you)
					spec_hubble_velocity = np.where(spec_hubble_velocity > max_box_vel/2.0, spec_hubble_velocity-max_box_vel, spec_hubble_velocity)
					spec_hubble_velocity = np.where(spec_hubble_velocity < (-1.0)*max_box_vel/2.0, spec_hubble_velocity+max_box_vel, spec_hubble_velocity)
					if max_abs_vel != None:
						spec_hubble_velocity = spec_hubble_velocity[np.where(np.abs(spec_hubble_velocity) < max_abs_vel)[0]]


					if virial_radii_bool:
						radius = radius/gal_R200

					if (radius < radii_bins[j+1]) and (radius > radii_bins[j]):
						if ((np.log10(gal_mass) > min_halo_mass) & (np.log10(gal_mass) < max_halo_mass)):
							if (np.size(final_ion_flux > 0)):
								num_in_bin += 1
								final_ion_flux = np.concatenate([final_ion_flux,ion_flux])
								final_velocities = np.concatenate([final_velocities, spec_hubble_velocity])
								if virial_vel_bool:
									final_velocities = final_velocities/virial_vel


							else:
								num_in_bin += 1
								final_ion_flux = ion_flux
								final_velocities = spec_hubble_velocity
								if virial_vel_bool:
									final_velocities = final_velocities/virial_vel

			else:
				radius = np.sqrt(np.sum(np.power((lines[0:3]*box_size-gal[0:3]),2)))
				if radius > 0.5:
					radius = (1.0-radius)*box_size
				else:
					radius *= box_size

				with h5py.File(spec_output_file) as hf:
					spec_hubble_velocity = np.array(hf.get('VHubble_KMpS'))
					curr_spectra_folder = hf.get('Spectrum0')
					curr_spectra = curr_spectra_folder.get(ion)
					ion_flux = np.array(curr_spectra.get("Flux"))
					col_dense = np.array(curr_spectra.get("LotTotalIonColumnDensity"))
				if max_abs_vel != None:
					ion_flux = ion_flux[np.where(np.abs(final_vel_arr) < max_abs_vel)[0]]
					optical_depth = optical_depth[np.where(np.abs(final_vel_arr) < max_abs_vel)[0]]
				length_spectra = np.size(spec_hubble_velocity)
				max_box_vel = spec_hubble_velocity[-1]*(length_spectra+1)/(length_spectra)

				H = max_box_vel/(box_size/1.e3) # stuff with H in Mpc
				gal_hubble_vel = (gal_coords[2]/1.e3)*H # switched gal coords to Mpc
				gal_vel_z = gal_vel[2]
				virial_vel = (200.0)*(gal_mass/5.0e12)**(1./3.)

				final_vel_arr = spec_hubble_velocity - (max_box_vel/2.0)
				#final_vel_arr = spec_hubble_velocity - (gal_hubble_vel+gal_vel_z) # old. Before new spec (now it centers for you)
				final_vel_arr = np.where(final_vel_arr > max_box_vel/2.0, final_vel_arr-max_box_vel, final_vel_arr)
				final_vel_arr = np.where(final_vel_arr < (-1.0)*max_box_vel/2.0, final_vel_arr+max_box_vel, final_vel_arr)
				if max_abs_vel != None:
					final_vel_arr = final_vel_arr[np.where(np.abs(final_vel_arr) < max_abs_vel)[0]]


				if virial_radii_bool:
					radius = radius/gal_R200

				if (radius < radii_bins[j+1]) and (radius > radii_bins[j]):
					if ((np.log10(gal_mass) > min_halo_mass) & (np.log10(gal_mass) < max_halo_mass)):
						if (np.size(final_ion_flux > 0)):
							num_in_bin += 1
							final_ion_flux = np.concatenate([final_ion_flux,ion_flux])

						else:
							num_in_bin += 1
							final_ion_flux = ion_flux
							if virial_vel_bool:
								final_velocities = final_vel_arr/virial_vel
							else:
								final_velocities = final_vel_arr		
		
			print 'in flux plots'
			print radius		
		try:
			if virial_vel_bool:
				vel_bins = np.linspace(np.min(final_velocities), np.max(final_velocities), (np.max(final_velocities)-np.min(final_velocities))/0.1)
			else:
				vel_bins = np.linspace(np.min(final_velocities), np.max(final_velocities), (np.max(final_velocities)-np.min(final_velocities))/5.0)
			plot_velocities = np.zeros(np.size(vel_bins)-1)
			plot_fluxes = np.zeros(np.size(vel_bins)-1)
			for i in range(0,np.size(vel_bins)-1):
				if mean_spectra_bool:
					plot_fluxes[i] = np.mean(final_ion_flux[np.where((final_velocities>vel_bins[i]) & (final_velocities<vel_bins[i+1]))])
				else:
					plot_fluxes[i] = np.median(final_ion_flux[np.where((final_velocities>vel_bins[i]) & (final_velocities<vel_bins[i+1]))])
				plot_velocities[i] = np.median([vel_bins[i],vel_bins[i+1]])
			if virial_radii_bool:
				plt.plot(plot_velocities, plot_fluxes, '-', label = '%.1f-%.1fR_vir:(n=%d)' %(radii_bins[j], radii_bins[j+1], num_in_bin))
			else:
				plt.plot(plot_velocities, plot_fluxes, '-', label = '%.0f-%.0fkpc:(n=%d)' %(radii_bins[j], radii_bins[j+1], num_in_bin))
		except:
			print 'plot fail for these radii. maybe none in bin? %.1f-%.1f' % (radii_bins[j], radii_bins[j+1])
	plt.legend(loc='lower left')
	plt.grid()
	plt.ylim(ymin=-0.2, ymax=1)
	plt.title('Flux vs Speed Relative to Central Galaxy for %s [%.1f-%.1f]' % (ion, min_halo_mass,max_halo_mass) )
	plt.ylabel('normalized flux')
	if virial_vel_bool:
		plt.xlim(xmin=-3, xmax = 3)
		plt.xlabel('vel (vel/virial velocity)')
	else:
		plt.xlim(xmin=-700, xmax = 700)
		plt.xlabel('vel (km/s)')
	if virial_radii_bool:
		if virial_vel_bool:
			if mean_spectra_bool:
				plt.savefig(ion +'_mean_spectra_virial_radius_virial_vel.png')
			else:
				plt.savefig(ion +'_median_spectra_virial_radius_virial_vel.png')
		else:
			if mean_spectra_bool:
				plt.savefig(ion +'_mean_spectra_virial_radius_physical_vel.png')
			else:
				plt.savefig(ion +'_median_spectra_virial_radius_physical_vel.png')
	else:
		if virial_vel_bool:
			if mean_spectra_bool:
				plt.savefig(ion +'_mean_spectra_physical_radius_virial_vel.png')
			else:
				plt.savefig(ion +'_median_spectra_physical_radius_virial_vel.png')
		else:
			if mean_spectra_bool:
				plt.savefig(ion +'new_mean_spectra_physical_radius_physical_vel.png')
			else:
				plt.savefig(ion +'_median_spectra_physical_radius_physical_vel.png')

	plt.close()

def equivalent_width(spec_output_directory,calls,ion,lambda_line): #lambda_line in angstroms
	los_files = glob.glob('/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/spec_outputs/los*')

	for i in range(0,np.size(los_files)):	
		gal_output_file = '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/spec_outputs/gal_output_'+str(i)+'.hdf5'
		spec_output_file = '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/spec_outputs/spec.snap_'+str(i)+'.hdf5'
		with h5py.File(spec_output_file,'r') as hf:
			vel = hf.get('VHubble_KMpS')
			delta_v = np.abs(vel[1]-vel[0])
			Spectrum0 = hf.get('Spectrum0')
			ion = Spectrum0.get(ion)
			flux = np.array(ion.get('Flux'))
			equi_width = np.sum(1.-flux)*(delta_v/c_kms)*lambda_line
		break
	print equi_width
	return equi_width




c_kms = 2.9979e+05 ### speed of light in km/s

cos_smass_data -= 0.2
cos_comparisons_file = 'cos_data_comparisons.hdf5'
directory_with_gal_folders = '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/'
spec_output_directory = '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/spec_output_4rel_3ax_all_surv/'
R_in_vir = 2.0
ion = 'h1'
starting_gal_id = int(130)

# make_cos_data_comparison_file(cos_comparisons_file, cos_smass_data, cos_ssfr_data)

# populate_hdf5_with_matching_gals(cos_comparisons_file, directory_with_gal_folders, starting_gal_id)

# calls = run_specwizard_for_matching_gals(cos_comparisons_file, directory_with_gal_folders, random_radii=False)

calls = 56
os.chdir("../snapshots/")

make_col_dense_plots(ion, calls, spec_output_directory, covering_frac_bool = True, covering_frac_val = 14.0)

make_contour_col_dense_plots(ion, calls, spec_output_directory, covering_frac_bool = True, covering_frac_val = 14.0)

radii_bins = [0,50,100,150,200] #kpc
# radii_bins = [0.0,0.4,0.8,1.2,1.6] # virial radii
plot_for_multiple_gals_by_radius(spec_output_directory, calls, ion, radii_bins, max_abs_vel = None,
	virial_vel_bool=True, virial_radii_bool=False, mean_spectra_bool=True, min_halo_mass=11.5., max_halo_mass=12.5)

equivalent_width(spec_output_directory,calls,ion,1215.67)





