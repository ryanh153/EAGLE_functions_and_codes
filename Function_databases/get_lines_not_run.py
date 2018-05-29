import numpy as np
import os
import sys
import subprocess

# path_to_old_run = sys.argv[1]
# if path_to_old_run[-1] != '/':
# 	path_to_old_run += '/'
# where_los_not_run_go = sys.argv[2]
# if where_los_not_run_go[-1] != '/':
# 	where_los_not_run_go += '/'

# dir_contents = os.listdir(path_to_old_run)

# all_los = []
# all_spec = []
# for file in dir_contents:
# 	if file[0:5] == 'spec.':
# 		all_spec.append(file)
# 	elif file[0:4] == 'los_':
# 		all_los.append(file)

# spec_nums = []
# for spec_file in all_spec:
# 	if spec_file[-2] == '_':
# 		spec_nums.append(spec_file[-1])
# 	elif spec_file[-3] == '_':
# 		spec_nums.append(spec_file[-2::])
# 	else:
# 		print 'spec file does not fit expected format?'

# for los_file in all_los:
# 	if los_file[-6] == '_':
# 		curr_num = los_file[-5]
# 		single = True
# 	elif los_file[-7] == '_':
# 		curr_num = los_file[-6:-4]
# 		single = False
# 	else:
# 		print 'los does not fit expected format?'

# 	if curr_num in spec_nums:
# 		continue
# 	else:
# 		subprocess.call('cp %s %s' % (path_to_old_run + los_file, where_los_not_run_go), shell=True)
# 		subprocess.call('cp %s %s' % (path_to_old_run + 'gal_output_'+str(curr_num)+'.hdf5', where_los_not_run_go), shell=True)

### This is when I fucked up the re-naming system for specwizard output files...

directory = sys.argv[1]
if directory[-1] != '/':
	directory += '/'

dir_contents = os.listdir(directory)

spec_files = []
for file in dir_contents:
	if file[0:5] == 'spec.':
		spec_files.append(file)

for file in spec_files:
	if file[9] != '_':
		subprocess.call('cp %s %s' % (directory+file, directory+file[0:9]+'_'+file[9::]), shell=True)
		subprocess.call('rm %s' % (directory+file), shell=True)
	else:
		print 'spec file does not fit expected format?'


