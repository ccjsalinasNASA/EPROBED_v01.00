# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:32:05 2024

@author: ccsalina
"""

import numpy as np
import matplotlib.pyplot as plt
import eprobed_functions as eprobed

#Calculate an E-region Ne vertical profile over longitude 0 degrees, 
#latitude 0 degrees, altitudes 90-120 km, 0 UT, day-of-year 80 and year 2007.
input_lon = 19.22
input_lat = -34.42
input_alt = np.arange(90,120,1) #90-120 km at 1 km resolution.
input_ut_hour = 15
input_doy = 15
input_year = 2012
ne_EPROBED,z,params_SZA,params_ALL = eprobed.eprobedFunc_lon_lat_ut_daily(input_lon,input_lat,input_alt,input_ut_hour,input_doy,input_year)
LT = eprobed.calculate_local_time(input_lon, input_ut_hour)

plt.figure(1)
plt.plot(ne_EPROBED,input_alt)

#Calculate an E-region Ne vertical profile over latitude 0 degrees, 12 LT,
#altitudes 90-120 km, 0 UT, day-of-year 80 and year 2007.
#INPUTS THAT CAN BE MODIFIED:
input_lat = 0
input_lst = 12
input_alt = np.arange(90,120,1) #90-120 km at 1 km resolution.
input_ut_hour = 0
input_doy = 80
input_year = 2008

#RUNS MODEL:
ne_EPROBED,z,params_SZA,params_ALL = eprobed.eprobedFunc_lat_lst_daily(input_lat,input_alt,input_lst,input_doy,input_year)

plt.figure(2)
plt.plot(ne_EPROBED,input_alt)
plt.xlabel('Ne (#/cc)')
plt.ylabel('altitude (km)')
#Calculate an E-region Ne vertical profile over latitude 0 degrees, 12 LT,
#altitudes 90-120 km, 0 UT, day-of-year 80 and year 2007. This uses 
#monthly-mean F10.7 index. For day-of-year 80 in year 2007, this will use 
#average of F10.7 index values for March 2007 but the solar zenith angle for
#the exact day-of-year 80 is used.
#INPUTS THAT CAN BE MODIFIED:
input_lat = 0
input_lst = 12
input_alt = np.arange(90,120,1) #90-120 km at 1 km resolution.
input_ut_hour = 0
input_month = 3
input_doy = 80
input_year = 2008

#RUNS MODEL:
ne_EPROBED,z,params_SZA,params_ALL = eprobed.eprobedFunc_lat_lst_monthly(input_lat,input_alt,input_lst,input_month,input_doy,input_year)

plt.figure(3)
plt.plot(ne_EPROBED,input_alt)
plt.xlabel('Ne (#/cc)')
plt.ylabel('altitude (km)')
