import numpy as np
import h5py

AGN_path = '/cosma/home/dp004/dc-oppe1/zooms/data/COS-AGN/COS-AGN.hdf5'
dwarfs_path = '/gpfs/data/analyse/rhorton/opp_research/snapshots/COS-Dwarfs.hdf5'
GASS_path = '/cosma/home/dp004/dc-oppe1/zooms/data/COS-GASS/COS-GASS.hdf5'
GTO_path = '/cosma/home/dp004/dc-oppe1/zooms/data/COS-GTO/COS-GTO.hdf5'
halos_path = '/gpfs/data/analyse/rhorton/opp_research/snapshots/COS-Halos.hdf5'

num_AGN = 21
num_dwarfs = 43
num_gass = 45
num_gto = 48
num_halos = 44

agn_ids = np.arange(1,num_AGN+1,1)
dwarf_ids = np.arange(num_AGN+1,num_AGN+num_dwarfs+1,1)
gass_ids = np.arange(num_AGN+num_dwarfs+1, num_AGN+num_dwarfs+num_gass+1,1)
gto_ids = np.arange(num_AGN+num_dwarfs+num_gass+1, num_AGN+num_dwarfs+num_gass+num_gto+1,1)
halo_ids = np.arange(num_AGN+num_dwarfs+num_gass+num_gto+1, num_AGN+num_dwarfs+num_gass+num_gto+num_halos+1,1)
single_ids = np.arange(num_AGN+num_dwarfs+num_gass+num_gto+num_halos+1,num_AGN+num_dwarfs+num_gass+num_gto+num_halos+2,1)

def select_data_for_run(max_smass, min_smass, max_ssfr, min_ssfr, cos_gass, cos_gto, cos_halos, cos_dwarfs, cos_AGN, single_gal_for_tests):
  cos_smass = np.array([])
  cos_ssfr = np.array([])
  cos_radii = np.array([])

  cos_h1_cols = np.array([])
  cos_h1_cols_errs = np.array([])
  cos_h1_cols_flags = np.array([])
  cos_h1_cols_radii = np.array([])
  cos_si3_cols = np.array([])
  cos_si3_cols_radii = np.array([])
  cos_o6_cols = np.array([])
  cos_o6_cols_radii = np.array([])
  cos_c4_cols = np.array([])
  cos_c4_cols_radii = np.array([])

  cos_h1_eq_widths = np.array([])
  cos_h1_W_errs = np.array([])
  cos_h1_W_flags = np.array([])
  cos_h1_eq_widths_radii = np.array([])
  cos_si3_eq_widths = np.array([])
  cos_si3_eq_widths_radii = np.array([])
  cos_o6_eq_widths = np.array([])
  cos_o6_eq_widths_radii = np.array([])
  cos_c4_eq_widths = np.array([])
  cos_c4_eq_widths_radii = np.array([])

  cos_AGN_vals = np.array([])
  which_survey = []
  id_arr = []

  if cos_gass:
    print 'gass'
    with h5py.File(GASS_path, 'r') as hf:
      pairs = hf.get('Pairs')
      smass = np.array(pairs.get('lmstar'))
      ssfr = np.array(pairs.get('lssfr'))
      radii = np.array(pairs.get('bgal'))
      H1 = pairs.get('HI')
      h1_eq_widths = np.array(H1.get('EW_Lya'))
      h1_W_err = np.array(H1.get('EW_Lya_err'))
      h1_W_flags = np.array(H1.get('EW_Lya_tag'))
      si3 = pairs.get('SiIII')
      si3_eq_widths = np.array(si3.get('EW_SiIII'))
      hf.close()

    name = np.array(['gass']*np.size(smass))

    id_arr = np.concatenate((id_arr, gass_ids[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))

    cos_smass = np.concatenate((cos_smass,smass[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_ssfr = np.concatenate((cos_ssfr, ssfr[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_radii = np.concatenate((cos_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))

    cos_h1_eq_widths = np.concatenate((cos_h1_eq_widths, h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_W_errs = np.concatenate((cos_h1_W_errs, h1_W_err[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_W_flags = np.concatenate((cos_h1_W_flags, h1_W_flags[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_eq_widths_radii = np.concatenate((cos_h1_eq_widths_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_cols = np.concatenate((cos_h1_cols, np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_h1_cols_errs = np.concatenate((cos_h1_cols_errs, np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_h1_cols_flags = np.concatenate((cos_h1_cols_flags, np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_h1_cols_radii = np.concatenate((cos_h1_cols_radii, np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_si3_eq_widths = np.concatenate((cos_si3_eq_widths,si3_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_si3_eq_widths_radii = np.concatenate((cos_si3_eq_widths_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_si3_cols = np.concatenate((cos_si3_cols,1.+np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_si3_cols_radii = np.concatenate((cos_si3_cols_radii, 1.+np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_o6_cols = np.concatenate((cos_o6_cols, 1.+np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_o6_cols_radii = np.concatenate((cos_o6_cols_radii, 1.+np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_o6_eq_widths = np.concatenate((cos_o6_eq_widths, np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_o6_eq_widths_radii = np.concatenate((cos_o6_eq_widths_radii, np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_c4_cols = np.concatenate((cos_c4_cols, 1.+np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_c4_cols_radii = np.concatenate((cos_c4_cols_radii, 1.+np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_c4_eq_widths = np.concatenate((cos_c4_eq_widths, np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_c4_eq_widths_radii = np.concatenate((cos_c4_eq_widths_radii, np.zeros(np.size(h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    which_survey = np.concatenate((which_survey, name[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    print np.size(which_survey)

  if cos_gto:
    with h5py.File(GTO_path, 'r') as hf:
      pairs = hf.get('Pairs')
      smass = np.array(pairs.get('lmstar'))
      ssfr = np.array(pairs.get('lssfr'))
      radii = np.array(pairs.get('bgal'))
      H1 = pairs.get('HI')
      col = np.array(H1.get('N_HI'))
      hf.close()

    name = np.array(['gto']*np.size(smass))

    id_arr = np.concatenate((id_arr, gto_ids[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))

    cos_smass = np.concatenate((cos_smass,smass[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_ssfr = np.concatenate((cos_ssfr, ssfr[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_radii = np.concatenate((cos_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))

    cos_h1_cols = np.concatenate((cos_h1_cols, col[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_cols_radii = np.concatenate((cos_h1_cols_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_eq_widths = np.concatenate((cos_h1_eq_widths, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_h1_eq_widths_radii = np.concatenate((cos_h1_eq_widths_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_si3_eq_widths = np.concatenate((cos_si3_eq_widths,np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_si3_eq_widths_radii = np.concatenate((cos_si3_eq_widths_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_si3_cols = np.concatenate((cos_si3_cols,np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_si3_cols_radii = np.concatenate((cos_si3_cols_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_o6_cols = np.concatenate((cos_o6_cols, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_o6_cols_radii = np.concatenate((cos_o6_cols_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_o6_eq_widths = np.concatenate((cos_o6_eq_widths, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_o6_eq_widths_radii = np.concatenate((cos_o6_eq_widths_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_c4_cols = np.concatenate((cos_c4_cols, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_c4_cols_radii = np.concatenate((cos_c4_cols_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_c4_eq_widths = np.concatenate((cos_c4_eq_widths, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_c4_eq_widths_radii = np.concatenate((cos_c4_eq_widths_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    which_survey = np.concatenate((which_survey, name[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))

  if cos_halos:
    print 'halos'
    with h5py.File(halos_path,'r') as hf:
      pairs = hf.get('Pairs')
      smass = np.array(pairs.get('lmstar'))
      ssfr = np.array(pairs.get('lssfr'))
      radii = np.array(pairs.get('bgal'))
      H1 = pairs.get('HI')
      h1_col = np.array(H1.get('N_HI'))
      h1_cols_err = np.array(H1.get('N_HI_err'))
      h1_cols_flags = np.array(H1.get('N_HI_tag'))
      h1_eq_widths = np.array(H1.get('EW_Lya'))
      h1_W_err = np.array(H1.get('EW_Lya_err'))
      h1_W_flags = np.array(H1.get('EW_Lya_tag'))
      si3 = pairs.get('SiIII')
      si3_col = np.array(si3.get('N_SiIII'))
      o6 = pairs.get('OVI')
      o6_col = np.array(o6.get('N_OVI'))
      c4 = pairs.get('CIV')
      c4_col = np.array(c4.get('N_CIV'))
      hf.close()

    name = np.array(['halos']*np.size(smass))

    id_arr = np.concatenate((id_arr, halo_ids[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))


    cos_smass = np.concatenate((cos_smass,smass[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_ssfr = np.concatenate((cos_ssfr, ssfr[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_radii = np.concatenate((cos_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))

    cos_h1_cols = np.concatenate((cos_h1_cols, h1_col[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_cols_errs = np.concatenate((cos_h1_cols_errs, h1_cols_err[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_cols_flags = np.concatenate((cos_h1_cols_flags, h1_cols_flags[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_cols_radii = np.concatenate((cos_h1_cols_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_eq_widths = np.concatenate((cos_h1_eq_widths, h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_W_errs = np.concatenate((cos_h1_W_errs, h1_W_err[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_W_flags = np.concatenate((cos_h1_W_flags, h1_W_flags[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_eq_widths_radii = np.concatenate((cos_h1_eq_widths_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))

    cos_si3_cols = np.concatenate((cos_si3_cols, si3_col[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_si3_cols_radii = np.concatenate((cos_si3_cols_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_si3_eq_widths = np.concatenate((cos_si3_eq_widths,np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_si3_eq_widths_radii = np.concatenate((cos_si3_eq_widths_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_o6_cols = np.concatenate((cos_o6_cols, o6_col[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_o6_cols_radii = np.concatenate((cos_o6_cols_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_o6_eq_widths = np.concatenate((cos_o6_eq_widths, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_o6_eq_widths_radii = np.concatenate((cos_o6_eq_widths_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_c4_cols = np.concatenate((cos_c4_cols, c4_col[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_c4_cols_radii = np.concatenate((cos_c4_cols_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_c4_eq_widths = np.concatenate((cos_c4_eq_widths, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_c4_eq_widths_radii = np.concatenate((cos_c4_eq_widths_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    which_survey = np.concatenate((which_survey, name[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    print np.size(which_survey)

  if cos_dwarfs:
    print 'dwarfs'
    with h5py.File(dwarfs_path, 'r') as hf:
      pairs = hf.get('Pairs')
      smass = np.array(pairs.get('lmstar'))
      ssfr = np.array(pairs.get('lssfr'))
      radii = np.array(pairs.get('bgal'))
      H1 = pairs.get('HI')
      h1_eq_widths = np.array(H1.get('EW_Lya'))
      h1_W_err = np.array(H1.get('EW_Lya_err'))
      h1_W_flags = np.array(H1.get('EW_Lya_tag'))
      h1_cols = np.array(H1.get('N_HI'))
      h1_cols_err = np.array(H1.get('N_HI_err'))
      h1_cols_flags = np.array(H1.get('N_HI_tag'))
      si3 = pairs.get('SiIII')
      si3_eq_widths = np.array(si3.get('EW_SiIII'))
      c4 = pairs.get('CIV')
      c4_eq_widths = np.array(c4.get('EW_CIV'))
      hf.close()

    name = np.array(['dwarfs']*np.size(smass))

    id_arr = np.concatenate((id_arr, dwarf_ids[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))


    cos_smass = np.concatenate((cos_smass,smass[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_ssfr = np.concatenate((cos_ssfr, ssfr[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_radii = np.concatenate((cos_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))

    cos_h1_eq_widths = np.concatenate((cos_h1_eq_widths, h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_W_errs = np.concatenate((cos_h1_W_errs, h1_W_err[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_W_flags = np.concatenate((cos_h1_W_flags, h1_W_flags[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_eq_widths_radii = np.concatenate((cos_h1_eq_widths_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_cols = np.concatenate((cos_h1_cols, h1_cols[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_cols_errs = np.concatenate((cos_h1_cols_errs, h1_cols_err[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_cols_flags = np.concatenate((cos_h1_cols_flags, h1_cols_flags[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_cols_radii = np.concatenate((cos_h1_cols_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))

    cos_si3_eq_widths = np.concatenate((cos_si3_eq_widths,si3_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))/1.0e3
    cos_si3_eq_widths_radii = np.concatenate((cos_si3_eq_widths_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_si3_cols = np.concatenate((cos_si3_cols,np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_si3_cols_radii = np.concatenate((cos_si3_cols_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_o6_cols = np.concatenate((cos_o6_cols, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_o6_cols_radii = np.concatenate((cos_o6_cols_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_o6_eq_widths = np.concatenate((cos_o6_eq_widths, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_o6_eq_widths_radii = np.concatenate((cos_o6_eq_widths_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_c4_eq_widths = np.concatenate((cos_c4_eq_widths,c4_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))/1.0e3
    cos_c4_eq_widths_radii = np.concatenate((cos_c4_eq_widths_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_c4_cols = np.concatenate((cos_c4_cols,np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_c4_cols_radii = np.concatenate((cos_c4_cols_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    which_survey = np.concatenate((which_survey, name[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    print np.size(which_survey)

  if cos_AGN:
    with h5py.File(AGN_path) as hf:
      pairs = hf.get('Pairs')
      smass = np.array(pairs.get('lmstar'))
      ssfr = np.array(pairs.get('lssfr'))
      radii = np.array(pairs.get('bgal'))
      AGN = np.array(pairs.get('lAGN'))
      H1 = pairs.get('HI')
      h1_eq_widths = np.array(H1.get('EW_Lya'))
      si3 = pairs.get('SiIII')
      si3_eq_widths = np.array(si3.get('EW_SiIII'))
      hf.close()

    name = np.array(['agn']*np.size(smass))

    id_arr = np.concatenate((id_arr, agn_ids[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))

    cos_smass = np.concatenate((cos_smass,smass[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_ssfr = np.concatenate((cos_ssfr, ssfr[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_radii = np.concatenate((cos_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))

    cos_h1_eq_widths = np.concatenate((cos_h1_eq_widths, h1_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_eq_widths_radii = np.concatenate((cos_h1_eq_widths_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_h1_cols = np.concatenate((cos_h1_cols, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_h1_cols_radii = np.concatenate((cos_h1_cols_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_o6_cols = np.concatenate((cos_o6_cols, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_o6_cols_radii = np.concatenate((cos_o6_cols_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_o6_eq_widths = np.concatenate((cos_o6_eq_widths, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_o6_eq_widths_radii = np.concatenate((cos_o6_eq_widths_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_si3_eq_widths = np.concatenate((cos_si3_eq_widths,si3_eq_widths[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_si3_eq_widths_radii = np.concatenate((cos_si3_eq_widths_radii, radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
    cos_si3_cols = np.concatenate((cos_si3_cols,np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_si3_cols_radii = np.concatenate((cos_si3_cols_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_c4_cols = np.concatenate((cos_c4_cols, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_c4_cols_radii = np.concatenate((cos_c4_cols_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_c4_eq_widths = np.concatenate((cos_c4_eq_widths, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))
    cos_c4_eq_widths_radii = np.concatenate((cos_c4_eq_widths_radii, np.zeros(np.size(radii[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))))

    cos_AGN_vals = np.concatenate((cos_AGN_vals,AGN[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))

    which_survey = np.concatenate((which_survey, name[((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr))]))
 
  if single_gal_for_tests:
    with h5py.File(halos_path,'r') as hf:
      pairs = hf.get('Pairs')
      smass = np.array(pairs.get('lmstar'))[0]
      ssfr = np.array(pairs.get('lssfr'))[0]
      radii = np.array(pairs.get('bgal'))[0]
      H1 = pairs.get('HI')
      col = np.array(H1.get('N_HI'))[0]
      h1_eq_widths = np.array(H1.get('EW_Lya'))[0]
      si3 = pairs.get('SiIII')
      si3_col = np.array(si3.get('N_SiIII'))[0]
      o6 = pairs.get('OVI')
      o6_col = np.array(o6.get('N_OVI'))[0]
      hf.close()

    if ((smass < max_smass) & (smass > min_smass) & (ssfr < max_ssfr) & (ssfr > min_ssfr)) == False:
      raise ValueError('smass or ssfr not in tolerance and we have single galaxy test turned on')   

    name = np.array(['single']*1)
    id_arr = np.concatenate((id_arr, single_ids))


    cos_smass = np.concatenate((cos_smass,np.array([smass])))
    cos_ssfr = np.concatenate((cos_ssfr, np.array([ssfr])))
    cos_radii = np.concatenate((cos_radii, np.array([radii])))

    cos_h1_cols = np.concatenate((cos_h1_cols, np.array([col])))
    cos_h1_cols_radii = np.concatenate((cos_h1_cols_radii, np.array([radii])))
    cos_h1_eq_widths = np.concatenate((cos_h1_eq_widths, np.array([h1_eq_widths])))
    cos_h1_eq_widths_radii = np.concatenate((cos_h1_eq_widths_radii, np.array([radii])))

    cos_si3_cols = np.concatenate((cos_si3_cols, np.array([si3_col])))
    cos_si3_cols_radii = np.concatenate((cos_si3_cols_radii, np.array([radii])))
    cos_si3_eq_widths = np.concatenate((cos_si3_eq_widths,np.zeros(np.size(radii))))
    cos_si3_eq_widths_radii = np.concatenate((cos_si3_eq_widths_radii, np.zeros(np.size(radii))))

    cos_o6_cols = np.concatenate((cos_o6_cols, np.array([o6_col])))
    cos_o6_cols_radii = np.concatenate((cos_o6_cols_radii, np.array([radii])))
    cos_o6_eq_widths = np.concatenate((cos_o6_eq_widths, np.zeros(np.size(radii))))
    cos_o6_eq_widths_radii = np.concatenate((cos_o6_eq_widths_radii, np.zeros(np.size(radii))))

    which_survey = np.concatenate((which_survey, name))


  return cos_smass, cos_ssfr, cos_radii, cos_h1_eq_widths, cos_h1_W_errs, cos_h1_W_flags, cos_h1_eq_widths_radii, cos_h1_cols, cos_h1_cols_errs, cos_h1_cols_flags, cos_h1_cols_radii, cos_si3_eq_widths, cos_si3_eq_widths_radii, cos_si3_cols, cos_si3_cols_radii, cos_o6_cols, cos_o6_cols_radii, cos_o6_eq_widths, cos_o6_eq_widths_radii, cos_c4_cols, cos_c4_cols_radii, cos_c4_eq_widths, cos_c4_eq_widths_radii, cos_AGN_vals, np.asarray(which_survey), id_arr

def get_rid_of_bad_data(AGN_bool, cos_smass_data, cos_ssfr_data, cos_radii_data, cos_id_arr, cos_h1_equ_widths, cos_h1_equ_widths_radii, cos_h1_cols, cos_h1_cols_radii, cos_si3_equ_widths, cos_si3_equ_widths_radii, cos_si3_cols, cos_si3_cols_radii, cos_o6_cols, cos_o6_cols_radii, cos_o6_equ_widths, cos_o6_equ_widths_radii, cos_AGN):
  perma_cos_ssfr_data = cos_ssfr_data

  if AGN_bool == False:
    cos_smass_data = cos_smass_data[((cos_h1_equ_widths > 0.) & (perma_cos_ssfr_data != 1000))]
    cos_ssfr_data = cos_ssfr_data[((cos_h1_equ_widths > 0.) & (perma_cos_ssfr_data != 1000))]
    cos_radii_data = cos_radii_data[((cos_h1_equ_widths > 0.) & (perma_cos_ssfr_data != 1000))]
    cos_id_arr = cos_id_arr[((cos_h1_equ_widths > 0.) & (perma_cos_ssfr_data != 1000))]

    cos_h1_equ_widths_radii = cos_h1_equ_widths_radii[((cos_h1_equ_widths > 0.) & (perma_cos_ssfr_data != 1000))]
    cos_h1_equ_widths = cos_h1_equ_widths[((cos_h1_equ_widths > 0.) & (perma_cos_ssfr_data != 1000))]
    cos_h1_cols_radii = cos_h1_cols_radii[((cos_h1_cols > 0.) & (perma_cos_ssfr_data != 1000))]
    cos_h1_cols = cos_h1_cols[((cos_h1_cols > 0.) & (perma_cos_ssfr_data != 1000))]

    cos_si3_equ_widths_radii = cos_si3_equ_widths_radii[((cos_si3_equ_widths > 0.) & (perma_cos_ssfr_data != 1000))]
    cos_si3_equ_widths = cos_si3_equ_widths[((cos_si3_equ_widths > 0.) & (perma_cos_ssfr_data != 1000))]
    cos_si3_cols_radii = cos_si3_cols_radii[((cos_si3_cols > 0.) & (perma_cos_ssfr_data != 1000))]
    cos_si3_cols = cos_si3_cols[((cos_si3_cols > 0.) & (perma_cos_ssfr_data != 1000))]

    cos_o6_equ_widths_radii = cos_o6_equ_widths_radii[((cos_o6_equ_widths > 0.) & (perma_cos_ssfr_data != 1000))]
    cos_o6_equ_widths = cos_o6_equ_widths[((cos_o6_equ_widths > 0.) & (perma_cos_ssfr_data != 1000))]
    cos_o6_cols_radii = cos_o6_cols_radii[((cos_o6_cols > 0.) & (perma_cos_ssfr_data != 1000))]
    cos_o6_cols = cos_o6_cols[((cos_o6_cols > 0.) & (perma_cos_ssfr_data != 1000))]

  else:
    num_real_gals = int(np.size(cos_smass_data)/2.0)
    cos_smass_data = cos_smass_data[((cos_h1_equ_widths > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    cos_ssfr_data = cos_ssfr_data[((cos_h1_equ_widths > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    cos_radii_data = cos_radii_data[((cos_h1_equ_widths > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    cos_id_arr = cos_id_arr[((cos_h1_equ_widths > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    cos_AGN = cos_AGN[((cos_h1_equ_widths > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]

    cos_h1_equ_widths_radii = cos_h1_equ_widths_radii[((cos_h1_equ_widths > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    cos_h1_equ_widths = cos_h1_equ_widths[((cos_h1_equ_widths > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    cos_h1_cols_radii = cos_h1_cols_radii[((cos_h1_cols > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    cos_h1_cols = cos_h1_cols[((cos_h1_cols > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]

    cos_si3_equ_widths_radii = cos_si3_equ_widths_radii[((cos_si3_equ_widths > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    cos_si3_equ_widths = cos_si3_equ_widths[((cos_si3_equ_widths > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    cos_si3_cols_radii = cos_si3_cols_radii[((cos_si3_cols > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    cos_si3_cols = cos_si3_cols[((cos_si3_cols > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]

    cos_o6_equ_widths_radii = cos_o6_equ_widths_radii[((cos_o6_equ_widths > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    cos_o6_equ_widths = cos_o6_equ_widths[((cos_o6_equ_widths > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    cos_o6_cols_radii = cos_o6_cols_radii[((cos_o6_cols > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    cos_o6_cols = cos_o6_cols[((cos_o6_cols > 0.) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))] 

  cos_h1_equ_widths[cos_h1_equ_widths > 5.] /= 1000.
  cos_si3_equ_widths[cos_si3_equ_widths > 5.] /= 1000.
  cos_o6_equ_widths_radii[cos_o6_equ_widths > 5.] /= 1000.

  return cos_smass_data, cos_ssfr_data, cos_radii_data, cos_id_arr, cos_h1_equ_widths, cos_h1_equ_widths_radii, cos_h1_cols, cos_h1_cols_radii, cos_si3_equ_widths, cos_si3_equ_widths_radii, cos_si3_cols, cos_si3_cols_radii, cos_o6_cols, cos_o6_cols_radii, cos_o6_equ_widths, cos_o6_equ_widths_radii, cos_AGN


def cos_where_matched_in_EAGLE(AGN_bool, proch_bool, proch_ids, where_matched_bools, cos_smass_data, cos_ssfr_data, cos_radii_data, cos_id_arr, cos_h1_equ_widths, cos_h1_equ_widths_radii, cos_h1_cols, cos_h1_cols_radii, cos_si3_equ_widths, cos_si3_equ_widths_radii, cos_si3_cols, cos_si3_cols_radii, cos_o6_cols, cos_o6_cols_radii, cos_o6_equ_widths, cos_o6_equ_widths_radii, cos_c4_cols, cos_c4_cols_radii, cos_c4_equ_widths, cos_c4_equ_widths_radii, cos_AGN):
  perma_cos_ssfr_data = cos_ssfr_data

  if AGN_bool == False:
    cos_h1_cols_indices = np.argwhere(((cos_h1_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_h1_equ_widths_indices = np.argwhere(((cos_h1_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_si3_cols_indices = np.argwhere(((cos_si3_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_si3_equ_widths_indices = np.argwhere(((cos_si3_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_o6_cols_indices = np.argwhere(((cos_o6_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_o6_equ_widths_indices = np.argwhere(((cos_o6_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_c4_cols_indices = np.argwhere(((cos_c4_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_c4_equ_widths_indices = np.argwhere(((cos_c4_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]

  if proch_bool:
    cos_h1_cols_indices = np.argwhere(((cos_h1_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000) & (np.in1d(cos_id_arr,proch_ids))))[:,0]
    cos_h1_equ_widths_indices = np.argwhere(((cos_h1_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000) & (np.in1d(cos_id_arr,proch_ids))))[:,0]
    cos_si3_cols_indices = np.argwhere(((cos_si3_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000) & (np.in1d(cos_id_arr,proch_ids))))[:,0]
    cos_si3_equ_widths_indices = np.argwhere(((cos_si3_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000) & (np.in1d(cos_id_arr,proch_ids))))[:,0]
    cos_o6_cols_indices = np.argwhere(((cos_o6_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000) & (np.in1d(cos_id_arr,proch_ids))))[:,0]
    cos_o6_equ_widths_indices = np.argwhere(((cos_o6_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000) & (np.in1d(cos_id_arr,proch_ids))))[:,0]
    cos_c4_cols_indices = np.argwhere(((cos_c4_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000) & (np.in1d(cos_id_arr,proch_ids))))[:,0]
    cos_c4_equ_widths_indices = np.argwhere(((cos_c4_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000) & (np.in1d(cos_id_arr,proch_ids))))[:,0]

    # cos_smass_data = cos_smass_data[((cos_h1_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_ssfr_data = cos_ssfr_data[((cos_h1_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_radii_data = cos_radii_data[((cos_h1_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_id_arr  = cos_id_arr[((cos_h1_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]

    # cos_h1_cols_radii = cos_h1_cols_radii[((cos_h1_cols > 0.) & (cos_h1_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_h1_cols = cos_h1_cols[((cos_h1_cols > 0.) & (cos_h1_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_h1_equ_widths_radii = cos_h1_equ_widths_radii[((cos_h1_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_h1_equ_widths = cos_h1_equ_widths[((cos_h1_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]

    # cos_si3_equ_widths_radii = cos_si3_equ_widths_radii[((cos_si3_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_si3_equ_widths = cos_si3_equ_widths[((cos_si3_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_si3_cols_radii = cos_si3_cols_radii[((cos_si3_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_si3_cols = cos_si3_cols[((cos_si3_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]

    # cos_o6_equ_widths_radii = cos_o6_equ_widths_radii[((cos_o6_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_o6_equ_widths = cos_o6_equ_widths[((cos_o6_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_o6_cols_radii = cos_o6_cols_radii[((cos_o6_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_o6_cols = cos_o6_cols[((cos_o6_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]

    # cos_c4_equ_widths_radii = cos_c4_equ_widths_radii[((cos_c4_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_c4_equ_widths = cos_c4_equ_widths[((cos_c4_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_c4_cols_radii = cos_c4_cols_radii[((cos_c4_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]
    # cos_c4_cols = cos_c4_cols[((cos_c4_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000))]

  else:
    cos_h1_cols_indices = np.argwhere(((cos_h1_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_h1_equ_widths_indices = np.argwhere(((cos_h1_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_si3_cols_indices = np.argwhere(((cos_si3_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_si3_equ_widths_indices = np.argwhere(((cos_si3_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_o6_cols_indices = np.argwhere(((cos_o6_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_o6_equ_widths_indices = np.argwhere(((cos_o6_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_c4_cols_indices = np.argwhere(((cos_c4_cols > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]
    cos_c4_equ_widths_indices = np.argwhere(((cos_c4_equ_widths > 0.) & (where_matched_bools == True) & (perma_cos_ssfr_data != 1000)))[:,0]

    # num_real_gals = int(np.size(cos_smass_data)/2.0)
    # cos_smass_data = cos_smass_data[((cos_h1_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_ssfr_data = cos_ssfr_data[((cos_h1_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_radii_data = cos_radii_data[((cos_h1_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_id_arr = cos_id_arr[((cos_h1_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_AGN = cos_AGN[((cos_h1_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]

    # cos_h1_equ_widths_radii = cos_h1_equ_widths_radii[((cos_h1_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_h1_equ_widths = cos_h1_equ_widths[((cos_h1_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_h1_cols_radii = cos_h1_cols_radii[((cos_h1_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_h1_cols = cos_h1_cols[((cos_h1_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]

    # cos_si3_equ_widths_radii = cos_si3_equ_widths_radii[((cos_si3_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_si3_equ_widths = cos_si3_equ_widths[((cos_si3_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_si3_cols_radii = cos_si3_cols_radii[((cos_si3_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_si3_cols = cos_si3_cols[((cos_si3_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]

    # cos_o6_equ_widths_radii = cos_o6_equ_widths_radii[((cos_o6_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_o6_equ_widths = cos_o6_equ_widths[((cos_o6_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_o6_cols_radii = cos_o6_cols_radii[((cos_o6_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_o6_cols = cos_o6_cols[((cos_o6_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))] 

    # cos_c4_equ_widths_radii = cos_c4_equ_widths_radii[((cos_c4_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_c4_equ_widths = cos_c4_equ_widths[((cos_c4_equ_widths > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_c4_cols_radii = cos_c4_cols_radii[((cos_c4_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))]
    # cos_c4_cols = cos_c4_cols[((cos_c4_cols > 0.) & (where_matched_bools[0:num_real_gals] == True) & (perma_cos_ssfr_data[0:num_real_gals] != 1000))] 

  return cos_h1_cols_indices, cos_h1_equ_widths_indices, cos_si3_cols_indices, cos_si3_equ_widths_indices, cos_o6_cols_indices, cos_o6_equ_widths_indices, cos_c4_cols_indices, cos_c4_equ_widths_indices 
  # return cos_smass_data, cos_ssfr_data, cos_radii_data, cos_id_arr, cos_h1_equ_widths, cos_h1_equ_widths_radii, cos_h1_cols, cos_h1_cols_radii, cos_si3_equ_widths, cos_si3_equ_widths_radii, cos_si3_cols, cos_si3_cols_radii, cos_o6_cols, cos_o6_cols_radii, cos_o6_equ_widths, cos_o6_equ_widths_radii, cos_c4_cols, cos_c4_cols_radii, cos_c4_equ_widths, cos_c4_equ_widths_radii, cos_AGN

