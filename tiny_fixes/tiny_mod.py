import h5py
import numpy as np

parsec = 3.0857e18 # cm
G = 6.674e-8
M_sol = 1.98855e33 # g

file_name = '/Users/ryanhorton1/Documents/bitbucket/opp_research/snapshots/output_mybox_snap_noneq_030_z000p205_1.hdf5'

# pull out
with h5py.File(file_name,'r+') as hf:
	GalaxyData = hf.get("GalaxyProperties")
	gal_stellar_mass = np.array(GalaxyData.get('gal_stellar_mass'))
	gal_SFR = np.array(GalaxyData.get('gal_SFR'))
	gal_R200 = np.array(GalaxyData.get('gal_R200'))

# correct
new_gal_R200 = gal_stellar_mass*M_sol
new_gal_stellar_mass = gal_SFR
new_gal_SFR = gal_R200*1.e3*parsec

# put back in
with h5py.File(file_name, 'a') as hf:
	GalaxyProperties = hf.get('GalaxyProperties')
	GalaxyProperties.__delitem__('gal_stellar_mass')
	GalaxyProperties.__delitem__('gal_SFR')
	GalaxyProperties.__delitem__('gal_R200')


	# convert to units of hdf5 file
	new_gal_R200 /= 1.e3*parsec
	new_gal_stellar_mass /= M_sol
	new_gal_SFR = (new_gal_SFR)*(3600.0*24.0*365.0)/(M_sol)

	gal_R200 = GalaxyProperties.create_dataset('gal_R200', (1,), maxshape= (None,), data = new_gal_R200)
	gal_R200.attrs['units'] = 'kiloparsecs'

	gal_stellar_mass = GalaxyProperties.create_dataset('gal_stellar_mass', (1,), maxshape=(None,), data = new_gal_stellar_mass)
	gal_stellar_mass.attrs['units'] = 'solar masses'

	gal_SFR = GalaxyProperties.create_dataset('gal_SFR', (1,), maxshape=(None,), data = new_gal_SFR)
	gal_SFR.attrs['units'] = 'M_sol/year'



