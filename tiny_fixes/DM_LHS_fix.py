import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as plt3d
import sys
import h5py
import glob
import warnings

data_directory = '/gpfs/data/analyse/rhorton/opp_research/data/end_summer_gals'

if data_directory[-1] != '/':
	data_directory += '/'

data_files = glob.glob(data_directory + '*/output_*.hdf5')

for n in range(0,np.size(data_files)):
	with h5py.File(data_files[n]) as hf:
		GalaxyProperties = hf.get('GalaxyProperties')
		DM_ratio = np.array(GalaxyProperties.get('DM_virial_ratio'))
		DM_ratio *= (5.0/2.0)
		GalaxyProperties.__delitem__('DM_virial_ratio')
		DM_virial_LHS = GalaxyProperties.create_dataset('DM_virial_ratio', (np.size(DM_ratio),), maxshape= (None,), data = DM_ratio)
