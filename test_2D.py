# -*- coding: utf-8 -*-
"""
Created on Fri May 17 22:00:30 2024

@author: ccsalina
"""

import numpy as np
import matplotlib.pyplot as plt
# import netCDF4 as nc
# from netCDF4 import Dataset
import eprobed_functions as eprobed
# import shutil
# import os
# import scipy.io

#Open g-function file.
fname_gfunc = 'func_gps.nc'
C0_GPS = eprobed.ncread(fname_gfunc,'C0_GPS')
C_GPS = eprobed.ncread(fname_gfunc,'C_GPS')

#INPUTS.
iyear = 2007
idoy = 167
lat = np.arange(-87.5,88.5,5)
lst = np.arange(1,25,1)
z = np.arange(90,121,1)

#Determine bin sizes.
nlat = np.size(lat)
nlst = np.size(lst)
nalt = np.size(z)

#Calculate time-related parameters using input year and day-of-year.
imonth = eprobed.get_month(idoy, iyear)
itime = imonth + ((iyear-2007)*12)

#Initialize float arrays.
model = np.zeros([nalt,nlat,nlst])
nme_2D = np.zeros([nlat,nlst])
hme_2D = np.zeros([nlat,nlst])
h_2D = np.zeros([nlat,nlst])
cosChi_2D = np.zeros([nlat,nlst])
ibin = 1
recon_nme,recon_hme,recon_h = eprobed.reconstructCoefficients(itime)
for ilat in np.arange(0,nlat):
    for ilst in np.arange(0,nlst):
        Y = iyear
        zenithAngle = eprobed.calculate_zenithAngle(lat[ilat],lst[ilst],idoy,Y)
        cosZenith = np.cos(zenithAngle*np.pi/180) #cos(zenith Angle).
        nme,hme,h = eprobed.calculate_Nme_Hme_H_efficient(itime,cosZenith,recon_nme,recon_hme,recon_h)
        
        #Calculate the g-function.
        gFunction = C0_GPS + (C_GPS[0]*cosZenith**1) + (C_GPS[1]*cosZenith**2) + (C_GPS[2]*cosZenith**3)+ (C_GPS[3]*cosZenith**4) 
        gFunction = gFunction + (C_GPS[4]*cosZenith**5) + (C_GPS[5]*cosZenith**6) + (C_GPS[6]*cosZenith**7)
        
        #Calculate and apply corrections.
        if (lat[ilat] >= -84)&(lat[ilat] <= 84)&(lst[ilst] >= 1)&(lst[ilst] <= 23):
            correct = eprobed.nme_photo_correction(itime,lst[ilst],lat[ilat])
            nme_corrected = nme + correct
            correct = eprobed.hme_photo_correction(itime,lst[ilst],lat[ilat])
            hme_corrected = hme + correct
            correct = eprobed.h_photo_correction(itime,lst[ilst],lat[ilat])
            h_corrected = h + correct

            if (nme_corrected > 0):
                nme = nme_corrected
                hme = hme_corrected
                h = h_corrected
            else:
                nme = nme
                hme = hme
                h = h
        else:
            nme = nme
            hme = hme
            h = h
        
        zprime = (z - hme)/h
        nme_2D[ilat,ilst] = nme
        hme_2D[ilat,ilst] = hme
        h_2D[ilat,ilst] = h
        cosChi_2D[ilat,ilst] = cosZenith
        model[:,ilat,ilst] = 1e11*nme*np.exp(0.5*(1-zprime-(gFunction*np.exp(-1*zprime))))
        print(str(100*ibin/(nlat*nlst))+"% finished running for DOY "+str(idoy)+" and year "+str(iyear)+".")
        ibin = ibin + 1

clevels = np.linspace(8,12,12)
fontsizeValue = 16

plt.figure(1,figsize=(10, 6), dpi=300)
altIndex = np.where(z == 100)[0]
contour = plt.contourf(lst, lat, np.log10(model[altIndex[0],:,:]), levels = clevels, cmap='jet')
colorbar = plt.colorbar()
colorbar.set_ticks(np.linspace(8,12,5)) 
colorbar.set_ticklabels(['10$^8$','10$^9$','10$^{10}$','10$^{11}$','10$^{12}$'], fontsize=fontsizeValue) 
plt.title('Ne (#/cc)', fontsize=fontsizeValue, pad = 10)
plt.xlim([1,25])
plt.ylim([-90,90])
plt.xticks(fontsize=fontsizeValue) # Set the font size of ticks
plt.yticks(fontsize=fontsizeValue) # Set the font size of ticks
plt.xlabel('Local time (hr)', fontsize=fontsizeValue) # Add labels and title
plt.ylabel('Latitude (deg)', fontsize=fontsizeValue) # Add labels and title
plt.xticks(np.arange(0,25,2),fontsize=fontsizeValue) # Set the font size of ticks
plt.yticks(np.arange(-90,91,15),fontsize=fontsizeValue) # Set the font size of ticks

