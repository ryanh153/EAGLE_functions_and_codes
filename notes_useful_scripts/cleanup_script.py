import os
import sys
import glob

def remove(path):
	for file in glob.glob(path):
		try:
			os.remove(file)
		except:
			print 'unable to remove file %s' % (str(file))

remove('std*')
remove('*png')
remove('../Ali_Spec_src/spec_outputs/*')
remove('../Ali_Spec_src/bonus_los_*')
remove('../Ali_Spec_src/los_*')
remove('cos_data*')
remove('particle_id_files*')

try:
	path_of_curr_output_folder = str(sys.argv[1])
	remove('%s/los_*' % (path_of_curr_output_folder))
	remove('%s/bonus_*' % (path_of_curr_output_folder))
	remove('%s/spec.snap*' % (path_of_curr_output_folder))
	remove('%s/gal_output*' % (path_of_curr_output_folder))
	remove('%s/param*' % (path_of_curr_output_folder))
	remove('%s/*.png' % (path_of_curr_output_folder))
except:
	print 'path to output folder not provided or is broken'