import EagleFunctions
import numpy as np
import h5py

gal_directories = np.array([
						    '/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_007_x001_eagle.NEQ.snap023restart',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_007_x001_eagle.NEQ.snap023restart',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_008_x001_eagle.NEQ.snap023restart',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_009_x001_eagle.NEQ.snap023restart/',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_006_x001_eagle.NEQ.snap023restart/',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_000_x001_eagle.NEQ.snap023restart/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_004_x001_eagle.NEQ.snap023restart/',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_001_x001_eagle.NEQ.snap023restart',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_003_x001_eagle.NEQ.snap023restart/',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_005_x001_eagle.NEQ.snap023restart',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_005_x001_eagle.NEQ.snap023restart',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_002_x001_eagle.NEQ.snap023restart',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_000_x001_eagle.NEQ.snap023restart/',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_000_x001_eagle.NEQ.snap023restart',
							'/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_000_x001_eagle.NEQ.snap023restart',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x008/data_B001_x008_eagle.NEQ.snap025restart/',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x008/data_B002_x008_eagle.NEQ.snap025restart',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x008/data_B003_x008_eagle.NEQ.snap025restart/',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x008/data_B000_x008_eagle.NEQ.snap025restart',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x008/data_B007_x008_eagle.NEQ.snap025restart/',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x008/data_B005_x008_eagle.NEQ.snap025restart',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x008/data_B004_x008_eagle.NEQ.snap025restart',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x008/data_B006_x008_eagle.NEQ.snap025restart/',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x008/data_B008_x008_eagle.NEQ.snap025restart/',
							'/cosma5/data/dp004/dc-oppe1/data/Halo_x008/data_B009_x008_eagle.NEQ.snap025restart/',
							'/cosma5/data/dp004/dc-oppe1/data/Barnes/Barnes20_hr.NEQ.snap025restart',
							'/cosma5/data/dp004/dc-oppe1/data/Barnes/Barnes20_hr.NEQ.snap025restart/',
							'/cosma5/data/dp004/dc-oppe1/data/Barnes/Barnes27_hr.NEQ.snap023restart/'
							])

snapshot_file_ends = np.array([
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '039_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '039_z000p205',
							   '030_z000p205',
							   '039_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '039_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '039_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '030_z000p205',
							   '034_z000p149',
							   '047_z000p000',
							   '056_z000p000'
							   ])

group_numbers = np.array([
						  '76',
						  '68',
						  '55',
						  '42',
						  '38',
						  '30',
						  '25',
						  '32',
						  '29',
						  '26',
						  '28',
						  '21',
						  '19',
						  '22',
						  '17',
						  '20',
						  '24',
						  '18',
						  '12',
						  '14',
						  '16',
						  '15',
						  '13',
						  '11',
						  '9',
						  '6',
						  '8',
						  '7',
						  '3',
						  '1',
						  '1',
						  '1',
						  '1',
						  '5',
						  '4',
						  '1',
						  '5',
						  '1',
						  '2',
						  '2',
						  '3',
						  '1',
						  '1',
						  '3',
						  '2',
						  '2',
						  '1',
						  '1',
						  '1',
						  '1',
						  '1',
						  '1',
						  '1',
						  '1',
						  '1',
						  '1',
						  '1',
						  '1',
						  '1',
						  '1'
						  ])

gal_output_directories = np.array([
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/10.46_mybox_sh76/output_mybox_snap_noneq_039_z000p205_76.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/10.65_mybox_sh68/output_mybox_snap_noneq_039_z000p205_68.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/10.74_mybox_sh55/output_mybox_snap_noneq_039_z000p205_55.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/10.86_mybox_sh42/output_mybox_snap_noneq_039_z000p205_42.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/10.97_mybox_sh38/output_mybox_snap_noneq_039_z000p205_38.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.04_mybox_sh30/output_mybox_snap_noneq_039_z000p205_30.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.05_L012N0376_sh25/output_mybox_snap_noneq_039_z000p205_25.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.09_L012N0376_sh32/output_mybox_snap_noneq_039_z000p205_32.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.10_L012N0376_sh29/output_mybox_snap_noneq_039_z000p205_29.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.11_L012N0376_sh26/output_mybox_snap_noneq_039_z000p205_26.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.11_L012N0376_sh28/output_mybox_snap_noneq_039_z000p205_28.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.16_mybox_sh21/output_mybox_snap_noneq_039_z000p205_21.hdf5', 
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.29_L012N0376_sh19/output_mybox_snap_noneq_039_z000p205_19.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.29_L012N0376_sh22/output_mybox_snap_noneq_039_z000p205_22.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.30_L012N0376_sh17/output_mybox_snap_noneq_039_z000p205_17.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.31_mybox_sh20/output_mybox_snap_noneq_039_z000p205_20.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.34_L012N0376_sh24/output_mybox_snap_noneq_039_z000p205_24.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.37_L012N0376_sh18/output_mybox_snap_noneq_039_z000p205_18.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.45_mybox_sh12/output_mybox_snap_noneq_039_z000p205_12.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.48_L012N0376_sh14/output_mybox_snap_noneq_039_z000p205_14.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.48_L012N0376_sh16/output_mybox_snap_noneq_039_z000p205_16.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.50_L012N0376_sh15/output_mybox_snap_noneq_039_z000p205_15.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.57_mybox_sh13/output_mybox_snap_noneq_039_z000p205_13.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.63_L012N0376_sh11/output_mybox_snap_noneq_039_z000p205_11.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.67_L012N0376_sh9/output_mybox_snap_noneq_039_z000p205_9.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.70_mybox_sh6/output_mybox_snap_noneq_039_z000p205_6.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.72_mybox_sh8/output_mybox_snap_noneq_039_z000p205_8.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.74_L012N0376_sh7/output_mybox_snap_noneq_039_z000p205_7.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.83_x001_data007_sh3/output_mybox_snap_noneq_030_z000p205_3.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.85_x001_data007_sh1/output_mybox_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.85_x001_data008_sh1/output_mybox_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.85_x001_data009_sh1/output_x001_data009_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.92_x001_data006_sh1/output_x001_data006_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.95_x001_data000_sh5/output_x001_data000_snap_noneq_030_z000p205_5.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.97_mybox_sh4/output_mybox_snap_noneq_039_z000p205_4.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/11.99_x001_data004_sh1/output_x001_data004_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.05_L012N0376_sh5/output_mybox_snap_noneq_039_z000p205_5.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.06_x001_data001_sh1/output_mybox_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.10_x001_data003_sh2/output_x001_data003_snap_noneq_030_z000p205_2.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.15_x001_data005_sh2/output_mybox_snap_noneq_030_z000p205_2.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.20_L012N0376_sh3/output_mybox_snap_noneq_039_z000p205_3.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.22_x001_data005_sh1/output_mybox_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.27_x001_data002_sh1/output_mybox_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.42_x001_data000_sh3/output_x001_data000_snap_noneq_030_z000p205_3.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.52_x001_data000_sh2/output_mybox_snap_noneq_030_z000p205_2.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.58_L012N0376_sh2/output_mybox_snap_noneq_039_z000p205_2.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.67_x001_data000_sh1/output_mybox_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.72_x008_dataB001_sh1/output_x008_dataB001_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.73_x008_dataB002_sh1/output_mybox_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.73_x008_dataB003_sh1/output_x008_dataB003_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.76_x008_dataB000_sh1/output_mybox_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.86_x008_dataB007_sh1/output_x008_dataB007_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.88_x008_dataB005_sh1/output_mybox_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.90_x008_dataB004_sh1/output_mybox_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/12.97_x008_dataB006_sh1/output_x008_dataB006_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/13.15_x008_dataB008_sh1/output_x008_dataB008_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/13.16_x008_dataB009_sh1/output_x008_dataB009_snap_noneq_030_z000p205_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/13.20_barnes_sh1/output_mybox_snap_noneq_034_z000p149_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/13.32_Barnes20_sh1/output_Barnes_20_snap_noneq_047_z000p000_1.hdf5',
								   '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals/13.54_Barnes27_sh1/output_Barnes_27_snap_noneq_056_z000p000_1.hdf5'
								   ])



print ''
print np.size(gal_directories)
print np.size(group_numbers)
print np.size(snapshot_file_ends)
print np.size(gal_output_directories)
print ''

for i in range(0,np.size(gal_directories)):

	# with h5py.File(gal_output_directories[i]) as f:
	# 	galaxy_properties = f.get('GalaxyProperties')
	# 	sSFR = galaxy_properties.get('sSFR')
	# 	print np.array(sSFR)

	# 	if sSFR != None:
	# 		print 'found'
	# 		del f['sSFR']

	secs_per_year = 3.1536e7
	grams_per_solar_mass = 1.989e33

	print 'index'
	print i
	snap_directory = gal_directories[i] # directory of all the snap files to be analyzed
	group_number = float(group_numbers[i])
	# what galaxy, and how much of it we want to look at 
	R_in_vir = 2.0
	snapshot_file_end = snapshot_file_ends[i]

	### Keywords to get correct files
	particles_included_keyword = "snap_noneq_" + snapshot_file_end
	group_included_keyword = "group_tab_" + snapshot_file_end
	subfind_included_keyword = "eagle_subfind_tab_" + snapshot_file_end

	# get basic properties of simulation and galaxy
	print 'getting basic props\n'
	print snap_directory
	print gal_output_directories[i]
	print group_number
	print snapshot_file_end

	box_size, expansion_factor, hubble_param, gal_coords, gal_velocity, gal_M200, gal_R200, radius, gal_speed, gal_stellar_mass, gal_SFR = \
	EagleFunctions.get_basic_props(snap_directory, R_in_vir, group_number, particles_included_keyword, group_included_keyword, subfind_included_keyword)

	print gal_stellar_mass
	print gal_SFR
	sSFR = (gal_SFR/gal_stellar_mass)*secs_per_year

	print sSFR


	EagleFunctions.add_gal_data_to_file(gal_output_directories[i], True, 'log10_smass', np.log10(gal_stellar_mass/grams_per_solar_mass), units = 'solar_masses')




