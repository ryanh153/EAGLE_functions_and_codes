import os
import sys
import glob
import subprocess

name_of_folder = str(sys.argv[1])

subprocess.call('mkdir %s' % (name_of_folder), shell=True)
subprocess.call('cp -r IonizationTables/ specwizard.F90 specwizard_modules.F90 specwizard_numerical.F90 specwizard_subroutines.F90 %s' % (name_of_folder), shell=True)