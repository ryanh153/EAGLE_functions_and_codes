### Make coldens plots for galaxies selected for paper 2
### Ryan Horton 

### should I make this self contained (except coldens obviously)

### Imports
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

### libraries from .py files that have to be in the same locations
import EagleFunctions
import coldens

### Constants 
parsec_in_cm = 3.0857e18 # cm
M_sol = 1.99e33 # g
h = 0.6777

### Info for galaxies: Directories, group nums, keywords, gal_coords if they are shifted and rotated...
# ### initial test galaxy
# dirs = ['/cosma5/data/dp004/dc-oppe1/data/Halo_x001/data_002_x001_eagle.NEQ.snap042_acc']
# designator = ['data_002_x001']
# keyword_ends = ['047_z000p000']
# group_numbers =  [1]
# known_gal_coords = [[25./2., 25./2., 25./2.]] # put zeros in this array if you want to take the gal coords from subfind, otherwise insert here
# particles_included_keyword = ['snap_rot_noneq_' + keyword_end for keyword_end in keyword_ends] # these rotated ones have a different naming convention. May do case by case because only a few gals for this paper
# group_included_keyword = ['group_tab_' + keyword_end for keyword_end in keyword_ends] # no rotated versions of these files. Not needed? We just get gal coordinates from them... does that change coords? 
# subfind_included_keyword = ['eagle_subfind_tab_' + keyword_end for keyword_end in keyword_ends] # TODO ask Ben about above
# all_directories = False

### box ben gave me. Can pick different subhaloes
dirs = ["/cosma5/data/dp004/dc-oppe1/data/L034box/data_L034N1034/snapshot_028_z000p000/"] # this might actually be one too deep here. Check that if it fails
group_dirs = ["/cosma5/data/dp004/dc-oppe1/data/L034box/data_L034N1034/groups_028_z000p000/"] # if the directories are different
designator = ["L034N1034_sh112"]
keyword_ends = ["028_z000p000_shalo_112_"]
group_numbers =  [112] # is this the same as subhalo number? assume yes and see if it works. 
known_gal_coords = [[(25.*1034.)/(752.*2.), (25.*1034.)/(752.*2.), (25.*1034.)/(752.*2.)]] # put zeros in this array if you want to take the gal coords from subfind, otherwise insert here
particles_included_keyword = ["snap_rot_" + keyword_end for keyword_end in keyword_ends] # these rotated ones have a different naming convention. May do case by case because only a few gals for this paper
group_included_keyword = ["group_tab_" + keyword_end[0:12] for keyword_end in keyword_ends] # I don't see a groups folder for this box...?
subfind_included_keyword = ["eagle_subfind_tab_" + keyword_end[0:12] for keyword_end in keyword_ends]
all_directories = True

### plotting params
plt.rcParams['axes.labelsize'], plt.rcParams['axes.titlesize'], plt.rcParams['legend.fontsize'], plt.rcParams['xtick.labelsize'],  plt.rcParams['ytick.labelsize'] = 14., 18., 12., 12., 12.

# make a plot of the column density
def coldens_plot(fig_name, Lx, Ly, coldens_map, ion, axis, vmin, vmax):  #ion and axis are strings
	# currently  coldens.py returns the column density values in linear space. Assume this (so log the map)
	coldens_map = np.log10(coldens_map)
	coldens_map = ndimage.rotate(coldens_map, 90.) # to bring the angles in line with my angle convention

	fig, ax = plt.subplots(1)
	im = ax.imshow(coldens_map.transpose(), extent = (-0.5*Lx, 0.5*Lx, -0.5*Ly, 0.5*Ly), aspect = 'auto', cmap='inferno', vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
	ax.plot([0,0.5*Lx], [0,0], "w-", lw=2)
	ax.plot([0,(0.5*Lx)/np.sqrt(2.)], [0,(0.5*Ly)/np.sqrt(2.)], "w-", lw=2)
	cb = fig.colorbar(im)
	cb.set_clim(vmin, vmax)
	cb.set_label(r"$N_{%s}$ ${\rm cm^{-2}}$" % (ion), fontsize = plt.rcParams["axes.labelsize"])
	ax.set_xlabel('Distance from Galaxy Center (Mpcs)')
	ax.set_ylabel('Distance from Galaxy Center (Mpcs)')
	ax.set_title("Column Density map of %s (Along %s)" % (ion, axis))
	fig.savefig(fig_name, bbox_inches = "tight")
	plt.close(fig)

### For now one galaxy, in the future we can hopefully take an index for each galaxy. Maybe dict? Maybe not. 

# get gas properties from eagle
species = 'h1' # ion species, will convert to lower case. Currnelty set up only for h1 but masses will be returned for h1 and all h
Lx, Ly, Lz = 0.5, 0.5, 0.5 # sizes of the image (x,y) and depth of the image (z) in Mpcs
npix_x, npix_y = 250, 250 # the number of pixels along each side of the final image (size of output array from coldens)
vmin_HI, vmax_HI, vmin_H, vmax_H = 13., 21.5, 19., 22.
ion, element = "HI", "H"
neighbors = 58 # Ben says this (58) is the needed value but he dones't understand it and I should play with
for gal_index in range(0,np.size(dirs)):
	print "getting galaxy props"
	gas_coords, smoothing_length, element_mass, ion_mass, gal_coords, box_size = EagleFunctions.get_props_for_coldens(species,
																				dirs[gal_index], group_numbers[gal_index], particles_included_keyword[gal_index], group_included_keyword[gal_index], subfind_included_keyword[gal_index], group_dirs[gal_index], all_directories=all_directories)
	print "got galaxy props"
	if known_gal_coords[gal_index] == 0:
		coldens_map = coldens.main(gas_coords, smoothing_length, ion_mass, gal_coords, Lx, Ly, Lz, npix_x, npix_y, neighbors, box_size, phi=270., fig_name='coldens_test_x.pdf', ion=ion)
		coldens_map = coldens.main(gas_coords, smoothing_length, ion_mass, gal_coords, Lx, Ly, Lz, npix_x, npix_y, neighbors, box_size, theta=90., fig_name='coldens_test_y.pdf', ion=ion)
		coldens_map = coldens.main(gas_coords, smoothing_length, ion_mass, gal_coords, Lx, Ly, Lz, npix_x, npix_y, neighbors, box_size, fig_name='coldens_test_z.pdf', ion=ion)
	else:
		# coldens_map = coldens.main(gas_coords, smoothing_length, ion_mass, known_gal_coords[gal_index], Lx, Ly, Lz, npix_x, npix_y, neighbors, box_size, phi=270.)
		# coldens_plot('%s_%s_x.pdf' % (designator[gal_index], ion), Lx, Ly, coldens_map, ion, "x", vmin_HI, vmax_HI)
		# coldens_map = coldens.main(gas_coords, smoothing_length, ion_mass, known_gal_coords[gal_index], Lx, Ly, Lz, npix_x, npix_y, neighbors, box_size, theta=90.)
		# coldens_plot('%s_%s_y.pdf' % (designator[gal_index], ion), Lx, Ly, coldens_map, ion, "y", vmin_HI, vmax_HI)
		# coldens_map = coldens.main(gas_coords, smoothing_length, ion_mass, known_gal_coords[gal_index], Lx, Ly, Lz, npix_x, npix_y, neighbors, box_size)
		# coldens_plot('%s_%s_z.pdf' % (designator[gal_index], ion), Lx, Ly, coldens_map, ion, "z", vmin_HI, vmax_HI)

		coldens_map = coldens.main(gas_coords, smoothing_length, element_mass, known_gal_coords[gal_index], Lx, Ly, Lz, npix_x, npix_y, neighbors, box_size, phi=270.)
		coldens_plot('%s_%s_x.pdf' % (designator[gal_index], element), Lx, Ly, coldens_map, element, "x", vmin_H, vmax_H)
		coldens_map = coldens.main(gas_coords, smoothing_length, element_mass, known_gal_coords[gal_index], Lx, Ly, Lz, npix_x, npix_y, neighbors, box_size, theta=90.)
		coldens_plot('%s_%s_y.pdf' % (designator[gal_index], element), Lx, Ly, coldens_map, element, "y", vmin_H, vmax_H)
		coldens_map = coldens.main(gas_coords, smoothing_length, element_mass, known_gal_coords[gal_index], Lx, Ly, Lz, npix_x, npix_y, neighbors, box_size)
		coldens_plot('%s_%s_z.pdf' % (designator[gal_index], element), Lx, Ly, coldens_map, element, "z", vmin_H, vmax_H)






