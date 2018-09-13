import h5py
import numpy as np
import sys

file_name = sys.argv[1]


with h5py.File(file_name,'r+') as hf:
	GalaxyData = hf.get("GalaxyProperties")
	gal_mass = np.array(GalaxyData.get('gal_mass'))
	stellar_mass = np.array(GalaxyData.get('gal_stellar_mass'))
	SFR = np.array(GalaxyData.get('gal_SFR'))
	gal_R200 = np.array(GalaxyData.get('gal_R200'))
	mu = np.array(GalaxyData.get('mu'))

print ''	
print 'galaxy halo mass is %.3f log10(solar masses)' % (np.log10(gal_mass))
print ''
print 'galaxy stellar mass is %.3f log10(solar masses)' % (np.log10(stellar_mass))
print ''
print 'Star formation rate is %.3e solar masses per year' % (SFR)
print ''
print 'Specific star formation rate is %.3e per year' % (SFR/stellar_mass)
print ''
print 'Virial radius is %4f kpc' % (gal_R200)
print ''
