import numpy as np
import SpecwizardFunctions
import EagleFunctions

snap_directory = '/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023'
data_dir = '/cosma5/data/dp004/dc-oppe1/data/L012N0376/data_L012N0376_eagle.NEQ.snap023/snapshot_noneq_056_z000p000'
path_to_param_template = './param_0.par'
los_name = 'cos_los.txt'
group_number = 70
particles_included_keyword = 'snap_noneq_056_z000p000'
group_included_keyword = 'group_tab_056_z000p000'
subfind_included_keyword = 'eagle_subfind_tab_056_z000p000'
points_per_radius = 4
radii = np.arange(20,50,10)

param_keywords = ['datadir','snap_base','los_coordinates_file']
param_replacements = [None]*np.size(param_keywords)
param_replacements[0] = 'datadir = ' + data_dir + '/'
param_replacements[1] = 'snap_base = ' + particles_included_keyword
param_replacements[2] = 'los_coordinates_file = ./'+ los_name[0:-4]+'_0'+los_name[-4::]

SpecwizardFunctions.create_mult_los_per_radius_text(los_name[0:-4]+'_0'+los_name[-4::], snap_directory, group_number, particles_included_keyword, group_included_keyword, 
	                subfind_included_keyword, points_per_radius, radii, 0) 

EagleFunctions.edit_text(path_to_param_template, 'curr_params_0.par', param_keywords, param_replacements)

SpecwizardFunctions.create_mult_los_per_radius_text(los_name[0:-4]+'_1'+los_name[-4::], snap_directory, group_number, particles_included_keyword, group_included_keyword, 
	                subfind_included_keyword, points_per_radius, radii, 1) 

EagleFunctions.edit_text(path_to_param_template, 'curr_params_1.par', param_keywords, param_replacements)

SpecwizardFunctions.create_mult_los_per_radius_text(los_name[0:-4]+'_2'+los_name[-4::], snap_directory, group_number, particles_included_keyword, group_included_keyword, 
	                subfind_included_keyword, points_per_radius, radii, 2) 

EagleFunctions.edit_text(path_to_param_template, 'curr_params_2.par', param_keywords, param_replacements)