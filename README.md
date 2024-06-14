# EPROBED_v01.00
This is the repository for E-PROBED version 1.00.

E-PROBED version 1.0

eprobed_functions.py contains all E-PROBED functions. 

All netcdf files in this package contain E-PROBED coefficients.

test.py shall be ran to test model. 3 plots of E-region Ne profiles must be produced if model runs correctly. This script also shows sample code lines to follow in order to run the model to produce individual vertical profiles.

test_2D.py shall also be ran to test model. A single contour-filled plot of an E-region Ne latitude-local time profile at 100 km must be produced if model runs correctly. This script also shows sample code lines to follow in order to run the model to produce multi-dimensional profile (e.g. latitude-local time-altitude profile) of E-region Ne.

E-PROBED requires the following python packages:
numpy, matplotlib, scipy, shutil, netCDF4
