import numpy as np
import matplotlib.pyplot as plt 

### constants
c_kms = 3.0e5
parsec_to_cm = 3.0857e18 # cm
G = 6.67e-8 # cgs
sol_mass_to_g = 1.99e33
m_e = 9.10938356e-28 # g
m_p = 1.6726219e-24 # g
mu = 1.3 # Hydrogen to total mass correction factor
pi = 3.1415

# ### Matching LOS

# # low_mass
# low_mass_arr = np.array([44.9, 76.2, 62.8, 44.9, 70.1, 70.1, 76.2, 76.2, 81.3, 62.8, 81.3, 76.2, 62.8, 95.4, 81.3, 44.9, 81.3, 70.1, 34.2, 62.8, 62.8, 76.2, 62.8, 9.7, 70.1])
# low_mass_cos_percentiles = [4.,94.,96.,90.,88.,78.,96.,94.,88.,98.,98.,58., 42.,80.,0., 26.,44.,68.,  100.,14.,98.,0.,40.,0., 76.,78.,34.,0.,96.,6.,88.,94.,62.,86.,50.,72., 100.,50.]
# low_mass_high_frac = 0.315789473684
# low_mass_low_frac = 0.157894736842
# eagle_slope = -0.00286173
# eagle_err = np.sqrt(1.43803332e-07)
# cos_slope = -0.00632769
# cos_err = np.sqrt(1.78523703e-06)
# delta = np.abs(eagle_slope-cos_slope)/(eagle_err+cos_err)
# print 'low mass'
# print 'eagle'
# print eagle_slope
# print eagle_err
# print 'cos'
# print cos_slope
# print cos_err
# print 'delta'
# print delta 
# print ''

# # high_mass_blue
# high_mass_blue_arr = np.array([89.4, 69.5, 92.1, 99.3, 97.1, 76.1, 99.3, 99.0, 85.9, 94.3, 98.6, 89.4, 85.9, 76.1, 95.9, 89.4, 85.9, 94.3, 99.0, 99.5, 97.1, 92.1, 89.4, 92.1, 81.5])
# high_mass_blue_cos_percentiles = [ 100., 2., 0., 100., 72., 86., 16., 62., 88., 92., 18., 32., 6., 88., 82., 98., 96., 94., 96., 70., 100., 88., 90., 100., 100., 100., 100., 82., 82., 94., 96.,  8., 80., 98.]
# high_mass_blue_high_frac = 0.470588235294
# high_mass_blue_low_frac = 0.117647058824
# eagle_slope = -0.00314004
# eagle_err = np.sqrt(1.21616478e-07)
# cos_slope = -0.00532466
# cos_err = np.sqrt(1.20611295e-06)
# delta = np.abs(eagle_slope-cos_slope)/(eagle_err+cos_err)
# print 'high mass blue'
# print 'eagle'
# print eagle_slope
# print eagle_err
# print 'cos'
# print cos_slope
# print cos_err
# print 'delta'
# print delta 
# print ''

# # high_mass_red
# high_mass_red_arr = np.array([0.0, 0.0, 0.0, 10.1, 0.0, 10.1, 10.1, 66.9, 0.0, 0.0, 0.0, 0.0, 55.9, 42.6, 55.9, 0.0, 0.0, 0.0, 0.0, 10.1, 0.0, 0.0, 0.0, 42.6, 10.1])
# high_mass_red_cos_percentiles = [54., 12., 56., 18., 80., 48., 68., 100., 62., 34., 30., 54., 87., 50., 100., 100., 22., 100., 0., 76. ]
# high_mass_red_high_frac = 0.2
# high_mass_red_low_frac = 0.1
# eagle_slope = -0.00305541
# eagle_err = np.sqrt(8.09873039e-07 )
# cos_slope = -0.00572197
# cos_err = np.sqrt(5.01529792e-06)
# delta = np.abs(eagle_slope-cos_slope)/(eagle_err+cos_err)
# print 'high mass red'
# print 'eagle'
# print eagle_slope
# print eagle_err
# print 'cos'
# print cos_slope
# print cos_err
# print 'delta'
# print delta 
# print ''

# bins = np.arange(0,110,10)
# plt.hist(high_mass_red_arr, bins)
# plt.title('Red KS Results Mean: %.1f std: %.1f' % (np.mean(high_mass_red_arr), np.std(high_mass_red_arr)))
# plt.xlabel('Rejection Confidence (%)')
# plt.ylabel('Number of Realizations')
# plt.savefig('high_mass_red_hist.png')
# plt.close()

# plt.hist(high_mass_blue_arr, bins)
# plt.xlabel('Rejection Confidence (%)')
# plt.ylabel('Number of Realizations')
# plt.title('Blue KS Results Mean: %.1f std: %.1f' % (np.mean(high_mass_blue_arr), np.std(high_mass_blue_arr)))
# plt.savefig('high_mass_blue_hist.png')
# plt.close()

# plt.hist(low_mass_arr, bins)
# plt.xlabel('Rejection Confidence (%)')
# plt.ylabel('Number of Realizations')
# plt.title('Low Mass KS Results Mean: %.1f std: %.1f' % (np.mean(low_mass_arr), np.std(low_mass_arr)))
# plt.savefig('low_mass_hist.png')
# plt.close()

### Semi_randomm LOS

#low mass
low_mass_arr = np.array([44.9, 81.3, 85.6, 85.6, 85.6, 89.0, 44.9, 62.8, 81.3, 70.1, 62.8, 54.4, 62.8, 70.1, 70.1, 54.4, 34.2, 85.6, 62.8, 96.7, 81.3, 70.1, 76.2, 89.0, 85.6])
low_mass_cos_percentiles = [[   2.,   92.,   98.,   86.,   90.,   62.,  100.,   90.,   84.,   92.,   98.,   50., 66.,   76.,    0.,   26.,   22.,   70.,   90.,   26.,   96.,    0.,   50.,    0., 88.,   78.,   42.,    0.,   94.,    2.,   90.,   82.,   60.,   76.,   52.,   84., 100.,   38.]]
low_mass_high_frac = 0.315789473684
low_mass_low_frac = 0.157894736842
eagle_slope = -0.00337388
eagle_err = np.sqrt(1.15775577e-07)
cos_slope = -0.00632769
cos_err = np.sqrt(1.78523703e-06)
delta = np.abs(eagle_slope-cos_slope)/(eagle_err+cos_err)
print 'low mass'
print 'eagle'
print eagle_slope
print eagle_err
print 'cos'
print cos_slope
print cos_err
print 'delta'
print delta 
print ''

# high mass blue
high_mass_blue_arr = np.array([89.4, 97.1, 99.0, 94.3, 85.9, 99.3, 99.0, 92.1, 95.9, 61.7, 89.4, 94.3, 81.5, 89.4, 92.1, 85.9, 76.1, 81.5, 97.9, 69.5, 99.5, 81.5, 81.5, 89.4, 97.9])
high_mass_blue_cos_percentiles = [[ 100.,    2.,    0.,   98.,   66.,   98.,   26.,   62.,   84.,   90.,   14.,   42., 4.,   86.,   74.,   92.,   84.,   96.,  100.,   64.,  100.,  100.,   96.,   98., 98.,  100.,  100.,   86.,   86.,   86.,   82.,   16.,   80.,   98.]]
high_mass_blue_high_frac = 0.441176470588
high_mass_blue_low_frac = 0.0882352941176
eagle_slope = -0.00302074
eagle_err = np.sqrt(1.06521288e-07)
cos_slope = -0.00532466
cos_err = np.sqrt(1.20611295e-06)
delta = np.abs(eagle_slope-cos_slope)/(eagle_err+cos_err)
print 'high mass blue'
print 'eagle'
print eagle_slope
print eagle_err
print 'cos'
print cos_slope
print cos_err
print 'delta'
print delta 
print ''

# high_mass_red
high_mass_red_arr = np.array([10.1, 0.0, 0.0, 27.3, 0.0, 10.1, 0.0, 10.1, 27.3, 10.1, 0.0, 0.0, 27.3, 55.9, 42.6, 91.8, 0.0, 0.0, 0.0, 10.1, 55.9, 0.0, 10.1, 27.3, 0.0])
high_mass_red_cos_percentiles = [[  30., 8., 56., 18., 82., 44., 62.,   100., 68., 42., 44., 60., 86., 0.,   100.,    98.,    22.,    96., 0.,    92. ]]
high_mass_red_high_frac = 0.25
high_mass_red_low_frac = 0.15
eagle_slope = -0.00306975
eagle_err = np.sqrt(8.43916939e-07)
cos_slope = -0.00572197
cos_err = np.sqrt(5.01529792e-06)
delta = np.abs(eagle_slope-cos_slope)/(eagle_err+cos_err)
print 'high mass red'
print 'eagle'
print eagle_slope
print eagle_err
print 'cos'
print cos_slope
print cos_err
print 'delta'
print delta 
print ''


# bins = np.arange(0,110,10)
# plt.hist(low_mass_arr, bins)
# plt.title('Low Mass KS Results Mean: %.1f std: %.1f' % (np.mean(low_mass_arr), np.std(low_mass_arr)))
# plt.savefig('low_mass_hist.png')
# plt.close()

# plt.hist(high_mass_blue_arr, bins)
# plt.title('High Mass Blue KS Results Mean: %.1f std: %.1f' % (np.mean(high_mass_blue_arr), np.std(high_mass_blue_arr)))
# plt.savefig('high_mass_blue_hist.png')
# plt.close()

# plt.hist(high_mass_red_arr, bins)
# plt.title('High Mass Red KS Results Mean: %.1f std: %.1f' % (np.mean(high_mass_red_arr), np.std(high_mass_red_arr)))
# plt.savefig('high_mass_red_hist.png')
# plt.close()




# spec_output_directory = np.array(['/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/gass/gass_1',
# 									'/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/gass/gass_2',
# 									'/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/gass/gass_3',
# 									'/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/gass/gass_4',
# 									'/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/gass/gass_5',
# 									'/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/gass/gass_6',
# 									'/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/gass/gass_7',
# 									'/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/gass/gass_8',
# 									'/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/gass/gass_9',
# 									'/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/gass/gass_10',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/halos/halos_1',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/halos/halos_2',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/halos/halos_3',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/halos/halos_5',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/halos/halos_6',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/halos/halos_7',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/halos/halos_8',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/halos/halos_9',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/halos/halos_10',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/dwarfs/dwarfs_1',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/dwarfs/dwarfs_2',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/dwarfs/dwarfs_3',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/dwarfs/dwarfs_4',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/dwarfs/dwarfs_5',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/dwarfs/dwarfs_6',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/dwarfs/dwarfs_7',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/dwarfs/dwarfs_8',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/dwarfs/dwarfs_9',
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/dwarfs/dwarfs_10'])

# spec_output_directory = np.array(['/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/halos/halos_%s' % (num),
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/dwarfs/dwarfs_%s' % (num),
# 								  '/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/gass/gass_%s' % (num)])



# spec_output_directory = np.array(['/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/gass/gass_25'])

# files = glob.glob('/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/gass/*')
# files.append(glob.glob('/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/halos/*'))
# files.append(glob.glob('/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/1rel_runs/dwarfs/*'))
# files.append(glob.glob('/gpfs/data/analyse/rhorton/opp_research/Ali_Spec_src/id_test/matched_los/multi_rel_runs/*'))
# files = np.hstack(files)
# spec_output_directory = files'

### Virial contour lines
# line fits
# low mass
# slope
# -0.308088621499
# 0.00949486882861
# intercept
# 0.751676285316
# 0.0104068184855

# line fits
# blue
# slope
# -0.555577639525
# 0.0138160282998
# intercept
# 0.981068079592
# 0.0184262854699

# line fits
# red
# slope
# -0.600370247567
# 0.0194092070495
# intercept
# 0.846147062588
# 0.0333323122062

# cumulative massees
# ['Low Mass', 'Blue', 'Red']
# 10.1115624377
# 10.85690459
# 11.3168551333
# errs
# 0.52646936916
# 0.52646936916
# 0.611763014611
# 0.611763014611
# 0.413723509808
# 0.413723509808

# cumulative massees
# ['M_halo<11.7', 'M_halo=11.7-12.5', 'M_halo>12.5']
# 9.93361048054
# 10.6155949612
# 11.4433215555
# errs
# 0.498106087671
# 0.498106087671
# 0.331167915704
# 0.331167915704
# 0.361385013864
# 0.361385013864

