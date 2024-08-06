# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:32:25 2024

@author: ccsalina
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import shutil
from scipy.interpolate import RegularGridInterpolator

def calculate_local_time(longitude, UT):
    # Calculate the number of hours offset from Greenwich Mean Time (GMT)
    offset_hours = longitude / 15

    # Calculate the local time by adding the offset to the UT
    local_time = UT + offset_hours

    # Adjust local time if it's outside the range of 0 to 24
    local_time = local_time % 24

    return local_time

def get_month(day_of_year, year):
    dayStart = np.array([1,32,60,91,121,152,182,213,244,274,305,335,365])
    if (year%4 == 0):
        dayStart[3:11] = dayStart[3:11] + 1    
    else:
        dayStart = dayStart    
    
    month = 0
    for istart in np.arange(0,12):
        if (day_of_year >= dayStart[istart])&(day_of_year < dayStart[istart+1]):
            month = istart+1   
        else:
            nothing = 0
    
    return month
    
def ncread(fname,param):
    nc_file = nc.Dataset(fname, 'r') 
    data = nc_file.variables[param][:]
    nc_file.close()
    
    return data

def ncdisp(filename):
    with Dataset(filename, 'r') as nc_file:
        print("NetCDF dimensions:")
        for dimname, dim in nc_file.dimensions.items():
            print("\t{}: {}".format(dimname, len(dim)))

        print("\nNetCDF variables:")
        for varname, var in nc_file.variables.items():
            print("\t{}: {}".format(varname, var.shape))
            for attrname in var.ncattrs():
                print("\t\t{}: {}".format(attrname, getattr(var, attrname)))

        print("\nNetCDF global attributes:")
        for attrname in nc_file.ncattrs():
            print("\t{}: {}".format(attrname, getattr(nc_file, attrname)))
    
def calculate_zenithAngle(lat,lst,dayOfYear,Y):
    #This function calculates the solar zenith angle in vector form.
    #Inputs: 
    #   lat is latitude in degrees.
    #   lst is local-solar-time in hours.
    #   dayOfyear is day of year in days.
    #   Y is year.
    #Output:
    #   zenithAngle is solar zenith angle in degrees.
    #   0-degrees zenith angle is local noon-time.    
    
    # lat = [0,0,0]
    # lst = [12,12,12]
    # dayOfYear = [80,80,80]
    lat = np.array(lat, dtype=float)
    lst = np.array(lst, dtype=float)
    dayOfYear = np.array(dayOfYear, dtype=float)
    
    #Calculate local hour angle in radians.
    ts = lst*3600
    ts = ts - (3600*12)
    H_a = 2*np.pi*ts/86400
    #print('Ha = '+str(H_a*180/np.pi))
    
    #Calculate number of leap days since the year 2000.
    DL = (Y-2001)/4
    
    #Calculate number of days from the beginning of Julian year 2000.
    Njd = 364.5 + (Y-2001)*365 + DL + dayOfYear
    #print('Njd = '+str(Njd))
    
    #Calculate obliquity of the ecliptic - angle between the plane of the Earth's Equator and the plane of eclipic.
    eob = 23.439 - 0.0000004*Njd
    #print('eob = '+str(eob))
    
    #Calculate the mean longitude of the Sun.
    LM = 280.460 + 0.9856474*Njd
    #print('LM = '+str(LM))
    
    #Calculate mean anomaly of the Sun.
    gM = 357.528 + 0.9856003*Njd
    #print('gM = '+str(gM))
    
    #Calculate ecliptic longitude of the Sun.
    lambda_ec = LM + 1.915*np.sin(gM*np.pi/180) + 0.020*np.sin(2*gM*np.pi/180)
    #print('lambda_ec = '+str(lambda_ec))
    
    #Calculate solar declination angle.
    solDeclinationAngle = np.arcsin(np.sin(eob*np.pi/180)*np.sin(lambda_ec*np.pi/180))
    #print('solDeclinationAngle = '+str(solDeclinationAngle))
    
    #Calculate cosine of solar zenith angle.
    cos_zenith = np.sin(lat*np.pi/180)*np.sin(solDeclinationAngle) + np.cos(lat*np.pi/180)*np.cos(solDeclinationAngle)*np.cos(H_a)
    
    #Calculate zenith angle.
    zenithAngle = np.arccos(cos_zenith)
    zenithAngle = zenithAngle*180/np.pi
    #print('zenithAngle = '+str(zenithAngle))
    
    return zenithAngle

def reconstructCoefficients(x_time):
    #This script reconstructs the coefficients relating Nme,Hme,H with cos(solarZenithAngle).
    #Inputs are time in months on or after January 2007.    
    solarInd = x_time - 1

    #################################RECONSTRUCT TIME-SERIES OF COS-CHI COEFFICIENTS.#################################
    # fname_coeffs = 'coefficients_cosChiAndTimeSeries_f107.nc'
    # fname_coeffs = 'coefficients_cosChiAndTimeSeries_f107_COSMIC1.nc'
    fname_coeffs = 'coefficients_cosChiAndTimeSeries_f107_COSMIC1_inv_logDiff.nc'
    periods_timeSeries = ncread(fname_coeffs,'periods_timeSeries')
    solarIndex = ncread(fname_coeffs,'f107')
        
    coeffs_seasonal_nme = ncread(fname_coeffs,'coeffs_seasonal_nme')
    coeffs_seasonal_hme = ncread(fname_coeffs,'coeffs_seasonal_hme')
    coeffs_seasonal_h = ncread(fname_coeffs,'coeffs_seasonal_h')
    
    coeffs_solar_nme = ncread(fname_coeffs,'coeffs_solar_nme')
    coeffs_solar_hme = ncread(fname_coeffs,'coeffs_solar_hme')
    coeffs_solar_h = ncread(fname_coeffs,'coeffs_solar_h')
    
    #[ncoeffs_cosChi,ntime]= np.shape(coeffs_cosChi_nme)
    [ncoeffs_cosChi,ncoeffs_timeSeries_seasonal]= np.shape(coeffs_seasonal_nme)
    [ncoeffs_cosChi,ncoeffs_timeSeries_solar]= np.shape(coeffs_solar_nme)
    
    #Get coeffs_cosChi.
    if np.size(x_time) == 1:
        #print('Calculate coefficients for one month.')    
        #icosChi = 0
        recon_nme = np.zeros([ncoeffs_cosChi])
        recon_hme = np.zeros([ncoeffs_cosChi])
        recon_h = np.zeros([ncoeffs_cosChi])
        for icosChi in np.arange(0,ncoeffs_cosChi):
            #################RECONSTRUCT FOR NME#################TO VECTORIZE, CONVERT LOOPS INTO EXPLICIT EXPRESSION OF EQUATION.
            #Reconstruct seasonal component.
            coeffs = coeffs_seasonal_nme[icosChi,:]
            dc_comp = coeffs[0]
            nperiods = np.size(periods_timeSeries)
            iperiod = 0
            icoeff = 1
            sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
            cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
            recon = dc_comp + sin_comp + cos_comp
            icoeff = icoeff + 2
            for iperiod in range(1,nperiods):
                sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
                cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
                recon = recon + sin_comp + cos_comp    
                icoeff = icoeff + 2
            
            #Reconstruct seasonal+solar component. If I don't want to incorporate solar index dependence, 
            #I can just comment this out because the seasonal and solar coefficients were calculated separately.
            coeffs_solar = coeffs_solar_nme[icosChi,:]
            # recon_all = coeffs_solar[0] + solarIndex[solarInd]*coeffs_solar[1] + recon.reshape(-1)  
            recon_all = coeffs_solar[0] + solarIndex[solarInd]*coeffs_solar[1] + recon  
            recon_nme[icosChi] = recon_all
        
            #################RECONSTRUCT FOR HME#################TO VECTORIZE, CONVERT LOOPS INTO EXPLICIT EXPRESSION OF EQUATION.
            #Reconstruct seasonal component.
            coeffs = coeffs_seasonal_hme[icosChi,:]
            dc_comp = coeffs[0]
            nperiods = np.size(periods_timeSeries)
            iperiod = 0
            icoeff = 1
            sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
            cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
            recon = dc_comp + sin_comp + cos_comp
            icoeff = icoeff + 2
            for iperiod in range(1,nperiods):
                sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
                cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
                recon = recon + sin_comp + cos_comp    
                icoeff = icoeff + 2
            
            #Reconstruct seasonal+solar component. If I don't want to incorporate solar index dependence, 
            #I can just comment this out because the seasonal and solar coefficients were calculated separately.
            coeffs_solar = coeffs_solar_hme[icosChi,:]
            # recon_all = coeffs_solar[0] + solarIndex[solarInd]*coeffs_solar[1] + recon.reshape(-1)  
            recon_all = coeffs_solar[0] + solarIndex[solarInd]*coeffs_solar[1] + recon  
            recon_hme[icosChi] = recon_all
        
            #################RECONSTRUCT FOR H#################TO VECTORIZE, CONVERT LOOPS INTO EXPLICIT EXPRESSION OF EQUATION.
            #Reconstruct seasonal component.
            coeffs = coeffs_seasonal_h[icosChi,:]
            dc_comp = coeffs[0]
            nperiods = np.size(periods_timeSeries)
            iperiod = 0
            icoeff = 1
            sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
            cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
            recon = dc_comp + sin_comp + cos_comp
            icoeff = icoeff + 2
            for iperiod in range(1,nperiods):
                sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
                cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
                recon = recon + sin_comp + cos_comp    
                icoeff = icoeff + 2
            
            #Reconstruct seasonal+solar component. If I don't want to incorporate solar index dependence, 
            #I can just comment this out because the seasonal and solar coefficients were calculated separately.
            coeffs_solar = coeffs_solar_h[icosChi,:]
            # recon_all = coeffs_solar[0] + solarIndex[solarInd]*coeffs_solar[1] + recon.reshape(-1)  
            recon_all = coeffs_solar[0] + solarIndex[solarInd]*coeffs_solar[1] + recon  
            recon_h[icosChi] = recon_all
        
    else:        
        icosChi = 0
        x_time = x_time.reshape(-1,1)
        recon_nme = np.zeros([ncoeffs_cosChi,np.size(x_time)])
        recon_hme = np.zeros([ncoeffs_cosChi,np.size(x_time)])
        recon_h = np.zeros([ncoeffs_cosChi,np.size(x_time)])
        for icosChi in np.arange(0,ncoeffs_cosChi):
            #################RECONSTRUCT FOR NME#################TO VECTORIZE, CONVERT LOOPS INTO EXPLICIT EXPRESSION OF EQUATION.
            #Reconstruct seasonal component.
            coeffs = coeffs_seasonal_nme[icosChi,:]
            dc_comp = coeffs[0]
            nperiods = np.size(periods_timeSeries)
            iperiod = 0
            icoeff = 1
            sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
            cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
            recon = dc_comp + sin_comp + cos_comp
            icoeff = icoeff + 2
            for iperiod in range(1,nperiods):
                sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
                cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
                recon = recon + sin_comp + cos_comp    
                icoeff = icoeff + 2
                
            #Reconstruct seasonal+solar component. If I don't want to incorporate solar index dependence, 
            #I can just comment this out because the seasonal and solar coefficients were calculated separately.
            coeffs_solar = coeffs_solar_nme[icosChi,:]
            # recon_all = coeffs_solar[0] + solarIndex[solarInd]*coeffs_solar[1] + recon.reshape(-1)  
            recon_all = coeffs_solar[0] + solarIndex[solarInd]*coeffs_solar[1] + recon  
            recon_nme[icosChi,:] = recon_all
            
            #################RECONSTRUCT FOR HME#################TO VECTORIZE, CONVERT LOOPS INTO EXPLICIT EXPRESSION OF EQUATION.
            #Reconstruct seasonal component.
            coeffs = coeffs_seasonal_hme[icosChi,:]
            dc_comp = coeffs[0]
            nperiods = np.size(periods_timeSeries)
            iperiod = 0
            icoeff = 1
            sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
            cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
            recon = dc_comp + sin_comp + cos_comp
            icoeff = icoeff + 2
            for iperiod in range(1,nperiods):
                sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
                cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
                recon = recon + sin_comp + cos_comp    
                icoeff = icoeff + 2
                
            #Reconstruct seasonal+solar component. If I don't want to incorporate solar index dependence, 
            #I can just comment this out because the seasonal and solar coefficients were calculated separately.
            coeffs_solar = coeffs_solar_hme[icosChi,:]
            # recon_all = coeffs_solar[0] + solarIndex[solarInd]*coeffs_solar[1] + recon.reshape(-1)  
            recon_all = coeffs_solar[0] + solarIndex[solarInd]*coeffs_solar[1] + recon  
            recon_hme[icosChi,:] = recon #recon_all
    
            #################RECONSTRUCT FOR H#################TO VECTORIZE, CONVERT LOOPS INTO EXPLICIT EXPRESSION OF EQUATION.
            #Reconstruct seasonal component.
            coeffs = coeffs_seasonal_h[icosChi,:]
            dc_comp = coeffs[0]
            nperiods = np.size(periods_timeSeries)
            iperiod = 0
            icoeff = 1
            sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
            cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
            recon = dc_comp + sin_comp + cos_comp
            icoeff = icoeff + 2
            for iperiod in range(1,nperiods):
                sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
                cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
                recon = recon + sin_comp + cos_comp    
                icoeff = icoeff + 2
                
            #Reconstruct seasonal+solar component. If I don't want to incorporate solar index dependence, 
            #I can just comment this out because the seasonal and solar coefficients were calculated separately.
            coeffs_solar = coeffs_solar_h[icosChi,:]
            # recon_all = coeffs_solar[0] + solarIndex[solarInd]*coeffs_solar[1] + recon.reshape(-1)  
            recon_all = coeffs_solar[0] + solarIndex[solarInd]*coeffs_solar[1] + recon  
            recon_h[icosChi,:] = recon #recon_all
            
        #Uncomment for checking with aid of plots.
        coeffs_cosChi_nme = ncread(fname_coeffs,'coeffs_cosChi_nme')
        y = coeffs_cosChi_nme[0,x_time-1]
        plt.figure(1)
        plt.plot(x_time,y,label = 'orig')
        plt.plot(x_time,recon_nme[0,:],label = 'with solar')
        plt.legend()
    return recon_nme, recon_hme, recon_h

def reconstructCoefficients_specSolarIndex(x_time,specSolarIndex):
    #This script reconstructs the coefficients relating Nme,Hme,H with cos(solarZenithAngle).
    #Inputs are time in months on or after January 2007 and F10.7 index.
    solarInd = x_time - 1

    #################################RECONSTRUCT TIME-SERIES OF COS-CHI COEFFICIENTS.#################################
    # fname_coeffs = 'coefficients_cosChiAndTimeSeries_f107.nc'
    # fname_coeffs = 'coefficients_cosChiAndTimeSeries_f107_COSMIC1.nc'
    fname_coeffs = 'coefficients_cosChiAndTimeSeries_f107_COSMIC1_inv_logDiff.nc'
    periods_timeSeries = ncread(fname_coeffs,'periods_timeSeries')
    #solarIndex = ncread(fname_coeffs,'f107')
        
    coeffs_seasonal_nme = ncread(fname_coeffs,'coeffs_seasonal_nme')
    coeffs_seasonal_hme = ncread(fname_coeffs,'coeffs_seasonal_hme')
    coeffs_seasonal_h = ncread(fname_coeffs,'coeffs_seasonal_h')
    
    coeffs_solar_nme = ncread(fname_coeffs,'coeffs_solar_nme')
    coeffs_solar_hme = ncread(fname_coeffs,'coeffs_solar_hme')
    coeffs_solar_h = ncread(fname_coeffs,'coeffs_solar_h')
    
    #[ncoeffs_cosChi,ntime]= np.shape(coeffs_cosChi_nme)
    [ncoeffs_cosChi,ncoeffs_timeSeries_seasonal]= np.shape(coeffs_seasonal_nme)
    [ncoeffs_cosChi,ncoeffs_timeSeries_solar]= np.shape(coeffs_solar_nme)
    
    #Get coeffs_cosChi.
    if np.size(x_time) == 1:
        #print('Calculate coefficients for one month.')    
        #icosChi = 0
        recon_nme = np.zeros([ncoeffs_cosChi])
        recon_hme = np.zeros([ncoeffs_cosChi])
        recon_h = np.zeros([ncoeffs_cosChi])
        for icosChi in np.arange(0,ncoeffs_cosChi):
            #################RECONSTRUCT FOR NME#################
            #Reconstruct seasonal component.
            coeffs = coeffs_seasonal_nme[icosChi,:]
            dc_comp = coeffs[0]
            nperiods = np.size(periods_timeSeries)
            iperiod = 0
            icoeff = 1
            sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
            cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
            recon = dc_comp + sin_comp + cos_comp
            icoeff = icoeff + 2
            for iperiod in range(1,nperiods):
                sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
                cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
                recon = recon + sin_comp + cos_comp    
                icoeff = icoeff + 2
            
            #Reconstruct seasonal+solar component. If I don't want to incorporate solar index dependence, 
            #I can just comment this out because the seasonal and solar coefficients were calculated separately.
            coeffs_solar = coeffs_solar_nme[icosChi,:]
            recon_all = coeffs_solar[0] + specSolarIndex*coeffs_solar[1] + recon.reshape(-1)  
            recon_nme[icosChi] = recon_all
        
            #################RECONSTRUCT FOR HME#################
            #Reconstruct seasonal component.
            coeffs = coeffs_seasonal_hme[icosChi,:]
            dc_comp = coeffs[0]
            nperiods = np.size(periods_timeSeries)
            iperiod = 0
            icoeff = 1
            sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
            cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
            recon = dc_comp + sin_comp + cos_comp
            icoeff = icoeff + 2
            for iperiod in range(1,nperiods):
                sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
                cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
                recon = recon + sin_comp + cos_comp    
                icoeff = icoeff + 2
            
            #Reconstruct seasonal+solar component. If I don't want to incorporate solar index dependence, 
            #I can just comment this out because the seasonal and solar coefficients were calculated separately.
            coeffs_solar = coeffs_solar_hme[icosChi,:]
            recon_all = coeffs_solar[0] + specSolarIndex*coeffs_solar[1] + recon.reshape(-1)  
            recon_hme[icosChi] = recon_all
        
            #################RECONSTRUCT FOR H#################
            #Reconstruct seasonal component.
            coeffs = coeffs_seasonal_h[icosChi,:]
            dc_comp = coeffs[0]
            nperiods = np.size(periods_timeSeries)
            iperiod = 0
            icoeff = 1
            sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
            cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
            recon = dc_comp + sin_comp + cos_comp
            icoeff = icoeff + 2
            for iperiod in range(1,nperiods):
                sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
                cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
                recon = recon + sin_comp + cos_comp    
                icoeff = icoeff + 2
            
            #Reconstruct seasonal+solar component. If I don't want to incorporate solar index dependence, 
            #I can just comment this out because the seasonal and solar coefficients were calculated separately.
            coeffs_solar = coeffs_solar_h[icosChi,:]
            recon_all = coeffs_solar[0] + specSolarIndex*coeffs_solar[1] + recon.reshape(-1)  
            recon_h[icosChi] = recon_all
        
    else:        
        icosChi = 0
        x_time = x_time.reshape(-1,1)
        recon_nme = np.zeros([ncoeffs_cosChi,np.size(x_time)])
        recon_hme = np.zeros([ncoeffs_cosChi,np.size(x_time)])
        recon_h = np.zeros([ncoeffs_cosChi,np.size(x_time)])
        for icosChi in np.arange(0,ncoeffs_cosChi):
            #################RECONSTRUCT FOR NME#################
            #Reconstruct seasonal component.
            coeffs = coeffs_seasonal_nme[icosChi,:]
            dc_comp = coeffs[0]
            nperiods = np.size(periods_timeSeries)
            iperiod = 0
            icoeff = 1
            sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
            cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
            recon = dc_comp + sin_comp + cos_comp
            icoeff = icoeff + 2
            for iperiod in range(1,nperiods):
                sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
                cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
                recon = recon + sin_comp + cos_comp    
                icoeff = icoeff + 2
                
            #Reconstruct seasonal+solar component. If I don't want to incorporate solar index dependence, 
            #I can just comment this out because the seasonal and solar coefficients were calculated separately.
            coeffs_solar = coeffs_solar_nme[icosChi,:]
            recon_all = coeffs_solar[0] + specSolarIndex*coeffs_solar[1] + recon.reshape(-1)
            recon_nme[icosChi,:] = recon_all
            
            #################RECONSTRUCT FOR HME#################
            #Reconstruct seasonal component.
            coeffs = coeffs_seasonal_nme[icosChi,:]
            dc_comp = coeffs[0]
            nperiods = np.size(periods_timeSeries)
            iperiod = 0
            icoeff = 1
            sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
            cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
            recon = dc_comp + sin_comp + cos_comp
            icoeff = icoeff + 2
            for iperiod in range(1,nperiods):
                sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
                cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
                recon = recon + sin_comp + cos_comp    
                icoeff = icoeff + 2
                
            #Reconstruct seasonal+solar component. If I don't want to incorporate solar index dependence, 
            #I can just comment this out because the seasonal and solar coefficients were calculated separately.
            coeffs_solar = coeffs_solar_nme[icosChi,:]
            recon_all = coeffs_solar[0] + specSolarIndex*coeffs_solar[1] + recon.reshape(-1)
            recon_hme[icosChi,:] = recon_all
    
            #################RECONSTRUCT FOR H#################
            #Reconstruct seasonal component.
            coeffs = coeffs_seasonal_nme[icosChi,:]
            dc_comp = coeffs[0]
            nperiods = np.size(periods_timeSeries)
            iperiod = 0
            icoeff = 1
            sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
            cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
            recon = dc_comp + sin_comp + cos_comp
            icoeff = icoeff + 2
            for iperiod in range(1,nperiods):
                sin_comp = coeffs[icoeff]*np.sin(2*np.pi*x_time/periods_timeSeries[iperiod])
                cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*x_time/periods_timeSeries[iperiod])
                recon = recon + sin_comp + cos_comp    
                icoeff = icoeff + 2
                
            #Reconstruct seasonal+solar component. If I don't want to incorporate solar index dependence, 
            #I can just comment this out because the seasonal and solar coefficients were calculated separately.
            coeffs_solar = coeffs_solar_nme[icosChi,:]
            recon_all = coeffs_solar[0] + specSolarIndex*coeffs_solar[1] + recon.reshape(-1)
            recon_h[icosChi,:] = recon_all
            
        #Uncomment for checking with aid of plots.
        coeffs_cosChi_nme = ncread(fname_coeffs,'coeffs_cosChi_nme')
        y = coeffs_cosChi_nme[0,x_time-1]
        plt.figure(1)
        plt.plot(x_time,y,label = 'orig')
        plt.plot(x_time,recon_nme[0,:],label = 'with solar')
        plt.legend()
    return recon_nme, recon_hme, recon_h

def calculate_Nme_Hme_H(x_time,cosChi):
    #This function calculates Nme, Hme and H.
    #Inputs are months on or after January 2007 and solar zenith angle.
    solarInd = x_time - 1
    recon_nme, recon_hme, recon_h = reconstructCoefficients(x_time) #Change this to version with solar index specified if needed then adjust input parameters.
    
    # fname_coeffs = 'coefficients_cosChiAndTimeSeries_f107.nc'
    # fname_coeffs = 'coefficients_cosChiAndTimeSeries_f107_COSMIC1.nc'
    fname_coeffs = 'coefficients_cosChiAndTimeSeries_f107_COSMIC1_inv_logDiff.nc'
    periods_cosChi = ncread(fname_coeffs,'periods_cosChi')
    periods_timeSeries = ncread(fname_coeffs,'periods_timeSeries')
    solarIndex = ncread(fname_coeffs,'f107')
    nperiods = np.size(periods_cosChi)
    
    ###############Calculate Nme.###############TO VECTORIZE, CONVERT LOOPS INTO EXPLICIT EXPRESSION OF EQUATION.
    coeffs = recon_nme
    dc_comp = coeffs[0]
    iperiod = 0
    icoeff = 2
    sin_comp = coeffs[icoeff]*np.sin(2*np.pi*cosChi/periods_cosChi[iperiod])
    cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*cosChi/periods_cosChi[iperiod])
    recon = dc_comp + (coeffs[1]*cosChi) + sin_comp + cos_comp
    icoeff = icoeff + 2
    for iperiod in range(1,nperiods):
        sin_comp = coeffs[icoeff]*np.sin(2*np.pi*cosChi/periods_cosChi[iperiod])
        cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*cosChi/periods_cosChi[iperiod])
        recon = recon + sin_comp + cos_comp    
        icoeff = icoeff + 2
    
    recon_nme_out = recon
    
    ###############Calculate Hme.###############TO VECTORIZE, CONVERT LOOPS INTO EXPLICIT EXPRESSION OF EQUATION.
    coeffs = recon_hme
    dc_comp = coeffs[0]
    iperiod = 0
    icoeff = 2
    sin_comp = coeffs[icoeff]*np.sin(2*np.pi*cosChi/periods_cosChi[iperiod])
    cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*cosChi/periods_cosChi[iperiod])
    recon = dc_comp + (coeffs[1]*cosChi) + sin_comp + cos_comp
    icoeff = icoeff + 2
    for iperiod in range(1,nperiods):
        sin_comp = coeffs[icoeff]*np.sin(2*np.pi*cosChi/periods_cosChi[iperiod])
        cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*cosChi/periods_cosChi[iperiod])
        recon = recon + sin_comp + cos_comp    
        icoeff = icoeff + 2
    
    recon_hme_out = recon
    
    ###############Calculate H.###############TO VECTORIZE, CONVERT LOOPS INTO EXPLICIT EXPRESSION OF EQUATION.
    coeffs = recon_h
    dc_comp = coeffs[0]
    iperiod = 0
    icoeff = 2
    sin_comp = coeffs[icoeff]*np.sin(2*np.pi*cosChi/periods_cosChi[iperiod])
    cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*cosChi/periods_cosChi[iperiod])
    recon = dc_comp + (coeffs[1]*cosChi) + sin_comp + cos_comp
    icoeff = icoeff + 2
    for iperiod in range(1,nperiods):
        sin_comp = coeffs[icoeff]*np.sin(2*np.pi*cosChi/periods_cosChi[iperiod])
        cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*cosChi/periods_cosChi[iperiod])
        recon = recon + sin_comp + cos_comp    
        icoeff = icoeff + 2
    
    recon_h_out = recon
    
    # print(recon_nme)
    # print(recon_hme)
    # print(recon_h)
    return recon_nme_out,recon_hme_out,recon_h_out

def calculate_Nme_Hme_H_efficient(x_time,cosChi,recon_nme,recon_hme,recon_h):
    #This function calculates Nme, Hme and H.
    #Inputs are months on or after January 2007 and solar zenith angle.
    solarInd = x_time - 1
    # recon_nme, recon_hme, recon_h = reconstructCoefficients(x_time) #Change this to version with solar index specified if needed then adjust input parameters.
    
    # fname_coeffs = 'coefficients_cosChiAndTimeSeries_f107.nc'
    # fname_coeffs = 'coefficients_cosChiAndTimeSeries_f107_COSMIC1.nc'
    fname_coeffs = 'coefficients_cosChiAndTimeSeries_f107_COSMIC1_inv_logDiff.nc'
    periods_cosChi = ncread(fname_coeffs,'periods_cosChi')
    periods_timeSeries = ncread(fname_coeffs,'periods_timeSeries')
    solarIndex = ncread(fname_coeffs,'f107')
    nperiods = np.size(periods_cosChi)
    
    ###############Calculate Nme.###############TO VECTORIZE, CONVERT LOOPS INTO EXPLICIT EXPRESSION OF EQUATION.
    coeffs = recon_nme
    dc_comp = coeffs[0]
    iperiod = 0
    icoeff = 2
    sin_comp = coeffs[icoeff]*np.sin(2*np.pi*cosChi/periods_cosChi[iperiod])
    cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*cosChi/periods_cosChi[iperiod])
    recon = dc_comp + (coeffs[1]*cosChi) + sin_comp + cos_comp
    icoeff = icoeff + 2
    for iperiod in range(1,nperiods):
        sin_comp = coeffs[icoeff]*np.sin(2*np.pi*cosChi/periods_cosChi[iperiod])
        cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*cosChi/periods_cosChi[iperiod])
        recon = recon + sin_comp + cos_comp    
        icoeff = icoeff + 2
    
    recon_nme_out = recon
    
    ###############Calculate Hme.###############TO VECTORIZE, CONVERT LOOPS INTO EXPLICIT EXPRESSION OF EQUATION.
    coeffs = recon_hme
    dc_comp = coeffs[0]
    iperiod = 0
    icoeff = 2
    sin_comp = coeffs[icoeff]*np.sin(2*np.pi*cosChi/periods_cosChi[iperiod])
    cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*cosChi/periods_cosChi[iperiod])
    recon = dc_comp + (coeffs[1]*cosChi) + sin_comp + cos_comp
    icoeff = icoeff + 2
    for iperiod in range(1,nperiods):
        sin_comp = coeffs[icoeff]*np.sin(2*np.pi*cosChi/periods_cosChi[iperiod])
        cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*cosChi/periods_cosChi[iperiod])
        recon = recon + sin_comp + cos_comp    
        icoeff = icoeff + 2
    
    recon_hme_out = recon
    
    ###############Calculate H.###############TO VECTORIZE, CONVERT LOOPS INTO EXPLICIT EXPRESSION OF EQUATION.
    coeffs = recon_h
    dc_comp = coeffs[0]
    iperiod = 0
    icoeff = 2
    sin_comp = coeffs[icoeff]*np.sin(2*np.pi*cosChi/periods_cosChi[iperiod])
    cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*cosChi/periods_cosChi[iperiod])
    recon = dc_comp + (coeffs[1]*cosChi) + sin_comp + cos_comp
    icoeff = icoeff + 2
    for iperiod in range(1,nperiods):
        sin_comp = coeffs[icoeff]*np.sin(2*np.pi*cosChi/periods_cosChi[iperiod])
        cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*cosChi/periods_cosChi[iperiod])
        recon = recon + sin_comp + cos_comp    
        icoeff = icoeff + 2
    
    recon_h_out = recon
    
    # print(recon_nme)
    # print(recon_hme)
    # print(recon_h)
    return recon_nme_out,recon_hme_out,recon_h_out

def nme_photo_correction(x_time,lst_target,lat_target):
    #This function calculates the solar zenith angle in vector form.
    #Inputs: 
    #   time is no. of months after December 2006.
    #   lst_target is local-solar-time in hours.
    #   lat_target is latitude in degrees.
    #Output:
    #   recon_all is percent correction that needs to be applied into parameter.
    
    #Interpolate into target latitude and local-solar-time.
    t = x_time
    # lst_target = 6 
    # lat_target = 22
    
    fname = 'coefficients_correction_Nme_COSMIC1_inv_logDiff.nc'
    coeffs_seasonal_all = ncread(fname,'coeffs_seasonal_all')
    coeffs_solar_all = ncread(fname,'coeffs_solar_all')
    lat = ncread(fname,'lat')
    lst = ncread(fname,'lst')
    solarIndex = ncread(fname,'f107')
    periods = ncread(fname,'periods')
    ncoeffs_seasonal = 1 + (np.size(periods)*2)
    ncoeffs_solar = 2
    
    latPoint = lat_target
    lstPoint = lst_target
    
    coeffs_seasonal_interp = np.zeros(ncoeffs_seasonal)
    for icoeffs_seasonal in np.arange(0,ncoeffs_seasonal):
        data = coeffs_seasonal_all[icoeffs_seasonal,:,:]
        #Future improvements will involve modifying this.
        interpolator = RegularGridInterpolator((lst,lat), data)
        coeffs_seasonal_interp[icoeffs_seasonal] = interpolator([lstPoint,latPoint])    
    
    coeffs_solar_interp = np.zeros(ncoeffs_solar)
    for icoeffs_solar in np.arange(0,ncoeffs_solar):
        data = coeffs_solar_all[icoeffs_solar,:,:]
        #Future improvements will involve modifying this.
        interpolator = RegularGridInterpolator((lst,lat), data)
        coeffs_solar_interp[icoeffs_solar] = interpolator([lstPoint,latPoint])    
    
    #Reconstruct.
    nperiods = np.size(periods)
    coeffs = coeffs_seasonal_interp #coeffs_seasonal_all[:,0,22] 
    dc_comp = coeffs[0]
    # linearTrend = t.reshape(-1,1)
    iperiod = 0
    icoeff = 1
    sin_comp = coeffs[icoeff]*np.sin(2*np.pi*t/periods[iperiod])
    cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*t/periods[iperiod])
    recon = dc_comp + sin_comp + cos_comp
    icoeff = icoeff + 2
    for iperiod in range(1,nperiods):
        sin_comp = coeffs[icoeff]*np.sin(2*np.pi*t/periods[iperiod])
        cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*t/periods[iperiod])
        recon = recon + sin_comp + cos_comp    
        icoeff = icoeff + 2
    
    #recon_all = coeffs_solar_interp[0] + solarIndex.reshape(-1)*coeffs_solar_interp[1] + recon
    recon_all = coeffs_solar_interp[0] + solarIndex[t-1]*coeffs_solar_interp[1] + recon
    
    #print(str(recon_all))    
    return recon_all

def nme_photo_correction_solarIndexInput(x_time,lst_target,lat_target,solarIndexInput):
    #This function calculates the solar zenith angle in vector form.
    #Inputs: 
    #   time is no. of months after December 2006.
    #   lst_target is local-solar-time in hours.
    #   lat_target is latitude in degrees.
    #   solarIndexInput is an input f10.7 index.
    #Output:
    #   recon_all is percent correction that needs to be applied into parameter.
    
    #Interpolate into target latitude and local-solar-time.
    t = x_time
    # lst_target = 6 
    # lat_target = 22
    
    fname = 'coefficients_correction_Nme_COSMIC1_inv_logDiff.nc'
    coeffs_seasonal_all = ncread(fname,'coeffs_seasonal_all')
    coeffs_solar_all = ncread(fname,'coeffs_solar_all')
    lat = ncread(fname,'lat')
    lst = ncread(fname,'lst')
    solarIndex = ncread(fname,'f107')
    periods = ncread(fname,'periods')
    ncoeffs_seasonal = 1 + (np.size(periods)*2)
    ncoeffs_solar = 2
    
    latPoint = lat_target
    lstPoint = lst_target
    
    coeffs_seasonal_interp = np.zeros(ncoeffs_seasonal)
    for icoeffs_seasonal in np.arange(0,ncoeffs_seasonal):
        data = coeffs_seasonal_all[icoeffs_seasonal,:,:]
        #Future improvements will involve modifying this.
        interpolator = RegularGridInterpolator((lst,lat), data)
        coeffs_seasonal_interp[icoeffs_seasonal] = interpolator([lstPoint,latPoint])    
    
    coeffs_solar_interp = np.zeros(ncoeffs_solar)
    for icoeffs_solar in np.arange(0,ncoeffs_solar):
        data = coeffs_solar_all[icoeffs_solar,:,:]
        #Future improvements will involve modifying this.
        interpolator = RegularGridInterpolator((lst,lat), data)
        coeffs_solar_interp[icoeffs_solar] = interpolator([lstPoint,latPoint])    
    
    #Reconstruct.
    nperiods = np.size(periods)
    coeffs = coeffs_seasonal_interp #coeffs_seasonal_all[:,0,22] 
    dc_comp = coeffs[0]
    # linearTrend = t.reshape(-1,1)
    iperiod = 0
    icoeff = 1
    sin_comp = coeffs[icoeff]*np.sin(2*np.pi*t/periods[iperiod])
    cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*t/periods[iperiod])
    recon = dc_comp + sin_comp + cos_comp
    icoeff = icoeff + 2
    for iperiod in range(1,nperiods):
        sin_comp = coeffs[icoeff]*np.sin(2*np.pi*t/periods[iperiod])
        cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*t/periods[iperiod])
        recon = recon + sin_comp + cos_comp    
        icoeff = icoeff + 2
    
    #recon_all = coeffs_solar_interp[0] + solarIndex.reshape(-1)*coeffs_solar_interp[1] + recon
    recon_all = coeffs_solar_interp[0] + solarIndexInput*coeffs_solar_interp[1] + recon
    
    # print(str(recon_all))    
    return recon_all

def hme_photo_correction(x_time,lst_target,lat_target):
    #This function calculates the solar zenith angle in vector form.
    #Inputs: 
    #   time is no. of months after December 2006.
    #   lst_target is local-solar-time in hours.
    #   lat_target is latitude in degrees.
    #Output:
    #   recon_all is percent correction that needs to be applied into parameter.
    
    #Interpolate into target latitude and local-solar-time.
    t = x_time
    # lst_target = 6 
    # lat_target = 22
    
    fname = 'coefficients_correction_Hme_COSMIC1_inv_logDiff.nc'
    coeffs_seasonal_all = ncread(fname,'coeffs_seasonal_all')
    coeffs_solar_all = ncread(fname,'coeffs_solar_all')
    lat = ncread(fname,'lat')
    lst = ncread(fname,'lst')
    solarIndex = ncread(fname,'f107')
    periods = ncread(fname,'periods')
    ncoeffs_seasonal = 1 + (np.size(periods)*2)
    ncoeffs_solar = 2
    
    latPoint = lat_target
    lstPoint = lst_target
    
    coeffs_seasonal_interp = np.zeros(ncoeffs_seasonal)
    for icoeffs_seasonal in np.arange(0,ncoeffs_seasonal):
        data = coeffs_seasonal_all[icoeffs_seasonal,:,:]
        #Future improvements will involve modifying this.
        interpolator = RegularGridInterpolator((lst,lat), data)
        coeffs_seasonal_interp[icoeffs_seasonal] = interpolator([lstPoint,latPoint])    
    
    coeffs_solar_interp = np.zeros(ncoeffs_solar)
    for icoeffs_solar in np.arange(0,ncoeffs_solar):
        data = coeffs_solar_all[icoeffs_solar,:,:]
        #Future improvements will involve modifying this.
        interpolator = RegularGridInterpolator((lst,lat), data)
        coeffs_solar_interp[icoeffs_solar] = interpolator([lstPoint,latPoint])    
    
    #Reconstruct.
    nperiods = np.size(periods)
    coeffs = coeffs_seasonal_interp #coeffs_seasonal_all[:,0,22] 
    dc_comp = coeffs[0]
    # linearTrend = t.reshape(-1,1)
    iperiod = 0
    icoeff = 1
    sin_comp = coeffs[icoeff]*np.sin(2*np.pi*t/periods[iperiod])
    cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*t/periods[iperiod])
    recon = dc_comp + sin_comp + cos_comp
    icoeff = icoeff + 2
    for iperiod in range(1,nperiods):
        sin_comp = coeffs[icoeff]*np.sin(2*np.pi*t/periods[iperiod])
        cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*t/periods[iperiod])
        recon = recon + sin_comp + cos_comp    
        icoeff = icoeff + 2
    
    #recon_all = coeffs_solar_interp[0] + solarIndex.reshape(-1)*coeffs_solar_interp[1] + recon
    recon_all = coeffs_solar_interp[0] + solarIndex[t-1]*coeffs_solar_interp[1] + recon
    
    #print(str(recon_all))    
    return recon_all

def hme_photo_correction_solarIndexInput(x_time,lst_target,lat_target,solarIndexInput):
    #This function calculates the solar zenith angle in vector form.
    #Inputs: 
    #   time is no. of months after December 2006.
    #   lst_target is local-solar-time in hours.
    #   lat_target is latitude in degrees.
    #   solarIndexInput is an input f10.7 index.
    #Output:
    #   recon_all is percent correction that needs to be applied into parameter.
    
    #Interpolate into target latitude and local-solar-time.
    t = x_time
    # lst_target = 6 
    # lat_target = 22
    
    fname = 'coefficients_correction_Hme_COSMIC1_inv_logDiff.nc'
    coeffs_seasonal_all = ncread(fname,'coeffs_seasonal_all')
    coeffs_solar_all = ncread(fname,'coeffs_solar_all')
    lat = ncread(fname,'lat')
    lst = ncread(fname,'lst')
    solarIndex = ncread(fname,'f107')
    periods = ncread(fname,'periods')
    ncoeffs_seasonal = 1 + (np.size(periods)*2)
    ncoeffs_solar = 2
    
    latPoint = lat_target
    lstPoint = lst_target
    
    coeffs_seasonal_interp = np.zeros(ncoeffs_seasonal)
    for icoeffs_seasonal in np.arange(0,ncoeffs_seasonal):
        data = coeffs_seasonal_all[icoeffs_seasonal,:,:]
        #Future improvements will involve modifying this.
        interpolator = RegularGridInterpolator((lst,lat), data)
        coeffs_seasonal_interp[icoeffs_seasonal] = interpolator([lstPoint,latPoint])    
    
    coeffs_solar_interp = np.zeros(ncoeffs_solar)
    for icoeffs_solar in np.arange(0,ncoeffs_solar):
        data = coeffs_solar_all[icoeffs_solar,:,:]
        #Future improvements will involve modifying this.
        interpolator = RegularGridInterpolator((lst,lat), data)
        coeffs_solar_interp[icoeffs_solar] = interpolator([lstPoint,latPoint])    
    
    #Reconstruct.
    nperiods = np.size(periods)
    coeffs = coeffs_seasonal_interp #coeffs_seasonal_all[:,0,22] 
    dc_comp = coeffs[0]
    # linearTrend = t.reshape(-1,1)
    iperiod = 0
    icoeff = 1
    sin_comp = coeffs[icoeff]*np.sin(2*np.pi*t/periods[iperiod])
    cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*t/periods[iperiod])
    recon = dc_comp + sin_comp + cos_comp
    icoeff = icoeff + 2
    for iperiod in range(1,nperiods):
        sin_comp = coeffs[icoeff]*np.sin(2*np.pi*t/periods[iperiod])
        cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*t/periods[iperiod])
        recon = recon + sin_comp + cos_comp    
        icoeff = icoeff + 2
    
    #recon_all = coeffs_solar_interp[0] + solarIndex.reshape(-1)*coeffs_solar_interp[1] + recon
    recon_all = coeffs_solar_interp[0] + solarIndexInput*coeffs_solar_interp[1] + recon
    
    # print(str(recon_all))    
    return recon_all

def h_photo_correction(x_time,lst_target,lat_target):
    #This function calculates the solar zenith angle in vector form.
    #Inputs: 
    #   time is no. of months after December 2006.
    #   lst_target is local-solar-time in hours.
    #   lat_target is latitude in degrees.
    #Output:
    #   recon_all is percent correction that needs to be applied into parameter.
    
    #Interpolate into target latitude and local-solar-time.
    t = x_time
    # lst_target = 6 
    # lat_target = 22
    
    fname = 'coefficients_correction_H_COSMIC1_inv_logDiff.nc'
    coeffs_seasonal_all = ncread(fname,'coeffs_seasonal_all')
    coeffs_solar_all = ncread(fname,'coeffs_solar_all')
    lat = ncread(fname,'lat')
    lst = ncread(fname,'lst')
    solarIndex = ncread(fname,'f107')
    periods = ncread(fname,'periods')
    ncoeffs_seasonal = 1 + (np.size(periods)*2)
    ncoeffs_solar = 2
    
    latPoint = lat_target
    lstPoint = lst_target
    
    coeffs_seasonal_interp = np.zeros(ncoeffs_seasonal)
    for icoeffs_seasonal in np.arange(0,ncoeffs_seasonal):
        data = coeffs_seasonal_all[icoeffs_seasonal,:,:]
        #Future improvements will involve modifying this.
        interpolator = RegularGridInterpolator((lst,lat), data)
        coeffs_seasonal_interp[icoeffs_seasonal] = interpolator([lstPoint,latPoint])    
    
    coeffs_solar_interp = np.zeros(ncoeffs_solar)
    for icoeffs_solar in np.arange(0,ncoeffs_solar):
        data = coeffs_solar_all[icoeffs_solar,:,:]
        #Future improvements will involve modifying this.
        interpolator = RegularGridInterpolator((lst,lat), data)
        coeffs_solar_interp[icoeffs_solar] = interpolator([lstPoint,latPoint])    
    
    #Reconstruct.
    nperiods = np.size(periods)
    coeffs = coeffs_seasonal_interp #coeffs_seasonal_all[:,0,22] 
    dc_comp = coeffs[0]
    # linearTrend = t.reshape(-1,1)
    iperiod = 0
    icoeff = 1
    sin_comp = coeffs[icoeff]*np.sin(2*np.pi*t/periods[iperiod])
    cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*t/periods[iperiod])
    recon = dc_comp + sin_comp + cos_comp
    icoeff = icoeff + 2
    for iperiod in range(1,nperiods):
        sin_comp = coeffs[icoeff]*np.sin(2*np.pi*t/periods[iperiod])
        cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*t/periods[iperiod])
        recon = recon + sin_comp + cos_comp    
        icoeff = icoeff + 2
    
    #recon_all = coeffs_solar_interp[0] + solarIndex.reshape(-1)*coeffs_solar_interp[1] + recon
    recon_all = coeffs_solar_interp[0] + solarIndex[t-1]*coeffs_solar_interp[1] + recon
    
    #print(str(recon_all))    
    return recon_all

def h_photo_correction_solarIndexInput(x_time,lst_target,lat_target,solarIndexInput):
    #This function calculates the solar zenith angle in vector form.
    #Inputs: 
    #   time is no. of months after December 2006.
    #   lst_target is local-solar-time in hours.
    #   lat_target is latitude in degrees.
    #   solarIndexInput is an input f10.7 index.
    #Output:
    #   recon_all is percent correction that needs to be applied into parameter.
    
    #Interpolate into target latitude and local-solar-time.
    t = x_time
    # lst_target = 6 
    # lat_target = 22
    
    fname = 'coefficients_correction_H_COSMIC1_inv_logDiff.nc'
    coeffs_seasonal_all = ncread(fname,'coeffs_seasonal_all')
    coeffs_solar_all = ncread(fname,'coeffs_solar_all')
    lat = ncread(fname,'lat')
    lst = ncread(fname,'lst')
    solarIndex = ncread(fname,'f107')
    periods = ncread(fname,'periods')
    ncoeffs_seasonal = 1 + (np.size(periods)*2)
    ncoeffs_solar = 2
    
    latPoint = lat_target
    lstPoint = lst_target
    
    coeffs_seasonal_interp = np.zeros(ncoeffs_seasonal)
    for icoeffs_seasonal in np.arange(0,ncoeffs_seasonal):
        data = coeffs_seasonal_all[icoeffs_seasonal,:,:]
        #Future improvements will involve modifying this.
        interpolator = RegularGridInterpolator((lst,lat), data)
        coeffs_seasonal_interp[icoeffs_seasonal] = interpolator([lstPoint,latPoint])    
    
    coeffs_solar_interp = np.zeros(ncoeffs_solar)
    for icoeffs_solar in np.arange(0,ncoeffs_solar):
        data = coeffs_solar_all[icoeffs_solar,:,:]
        #Future improvements will involve modifying this.
        interpolator = RegularGridInterpolator((lst,lat), data)
        coeffs_solar_interp[icoeffs_solar] = interpolator([lstPoint,latPoint])    
    
    #Reconstruct.
    nperiods = np.size(periods)
    coeffs = coeffs_seasonal_interp #coeffs_seasonal_all[:,0,22] 
    dc_comp = coeffs[0]
    # linearTrend = t.reshape(-1,1)
    iperiod = 0
    icoeff = 1
    sin_comp = coeffs[icoeff]*np.sin(2*np.pi*t/periods[iperiod])
    cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*t/periods[iperiod])
    recon = dc_comp + sin_comp + cos_comp
    icoeff = icoeff + 2
    for iperiod in range(1,nperiods):
        sin_comp = coeffs[icoeff]*np.sin(2*np.pi*t/periods[iperiod])
        cos_comp = coeffs[icoeff+1]*np.cos(2*np.pi*t/periods[iperiod])
        recon = recon + sin_comp + cos_comp    
        icoeff = icoeff + 2
    
    #recon_all = coeffs_solar_interp[0] + solarIndex.reshape(-1)*coeffs_solar_interp[1] + recon
    recon_all = coeffs_solar_interp[0] + solarIndexInput*coeffs_solar_interp[1] + recon
    
    # print(str(recon_all))    
    return recon_all

def eprobedFunc_lon_lat_ut_daily(input_lon,input_lat,input_alt,input_ut_hour,input_doy,input_year):
    #This function calculates E-region Ne vertical profile.
    #Inputs: 
    #   input_lon is longitude in degrees.
    #   input_lat is latitude in degrees.
    #   input_alt is altitude profile in km.   
    #   input_ut_hour is universal-time in hours.
    #   input_doy is day of year.
    #   input_year is year.
    #Output: ne_EPROBED,z,params_SZA,params_ALL
    #   ne_EPROBED is E-region vertical profile in e/m3.
    #   z is altitude bin in km.
    #   params_SZA = [Nme,Hme,H] are the 3 parameters from SZA component.
    #   params_ALL = [Nme,Hme,H] are the 3 parameters from SZA+nonSZA component.
    
    # Open the NetCDF file of g-function.
    C0_GPS = ncread('func_gps.nc','C0_GPS')
    C_GPS = ncread('func_gps.nc','C_GPS')
    
    # Open solar cycle index.
    fname_indices = 'solarGeomagnetic_daily_2000to2023.nc'
    allIndex = ncread(fname_indices,'solarGeomagneticIndex')
    indices_year = allIndex[0,:]
    indices_doy = allIndex[1,:]
    indicesOnly = allIndex[3:21]
    allIndex[allIndex == 999.9] = np.nan
    indexValue = 8 #Use number in txt file.
    solarIndex = indicesOnly[indexValue-1,:]
    
    # Get month and local-time.
    input_month = get_month(input_doy, input_year)
    input_lst = calculate_local_time(input_lon, input_ut_hour)
    
    itime = input_month + ((input_year-2007)*12)
    specSolarIndex = solarIndex[(indices_year == input_year)&(indices_doy == input_doy)]
    recon_nme,recon_hme,recon_h = reconstructCoefficients_specSolarIndex(itime,specSolarIndex)
    zenithAngle = calculate_zenithAngle(input_lat,input_lst,input_doy,input_year)
    cosZenith = np.cos(zenithAngle*np.pi/180) #cos(zenith Angle).
    nme,hme,h = calculate_Nme_Hme_H_efficient(itime,cosZenith,recon_nme,recon_hme,recon_h)
    params_SZA = [nme,hme,h]

    #Calculate the g-function.
    gFunction = C0_GPS + (C_GPS[0]*cosZenith**1) + (C_GPS[1]*cosZenith**2) + (C_GPS[2]*cosZenith**3)+ (C_GPS[3]*cosZenith**4) 
    gFunction = gFunction + (C_GPS[4]*cosZenith**5) + (C_GPS[5]*cosZenith**6) + (C_GPS[6]*cosZenith**7)
    
    #Calculate and apply corrections.
    if (input_lat >= -84)&(input_lat <= 84)&(input_lst >= 1)&(input_lst <= 23):
        correct = nme_photo_correction_solarIndexInput(itime,input_lst,input_lat,specSolarIndex)
        nme_correction = correct.data[0]
        correct = hme_photo_correction_solarIndexInput(itime,input_lst,input_lat,specSolarIndex)
        hme_correction = correct.data[0]
        correct = h_photo_correction_solarIndexInput(itime,input_lst,input_lat,specSolarIndex)
        h_correction = correct.data[0]
        
        if (nme_correction > 0):
            nme = nme + nme_correction
            hme = hme + hme_correction
            h = h + h_correction
        else:
            nme = nme
            hme = hme
            h = h
    else:
        nme = nme
        hme = hme
        h = h
            
    params_ALL = [nme,hme,h]
    z = input_alt
    zprime = (z - hme)/h
    ne_EPROBED = 1e11*nme*np.exp(0.5*(1-zprime-(gFunction*np.exp(-1*zprime))))

    return ne_EPROBED,z,params_SZA,params_ALL

def eprobedFunc_lat_lst_daily(input_lat,input_alt,input_lst,input_doy,input_year):
    #This function calculates E-region Ne vertical profile.
    #Inputs: 
    #   input_lat is latitude in degrees.
    #   input_alt is altitude in km.   
    #   input_lst is local-solar-time in hours.
    #   input_doy is day of year.
    #   input_year is year.
    #Output: ne_EPROBED,z,params_SZA,params_ALL
    #   ne_EPROBED is E-region vertical profile in e/m3.
    #   z is altitude bin in km.
    #   params_SZA = [Nme,Hme,H] are the 3 parameters from SZA component.
    #   params_ALL = [Nme,Hme,H] are the 3 parameters from SZA+nonSZA component.
        
    
    # Open the NetCDF file of g-function.
    C0_GPS = ncread('func_gps.nc','C0_GPS')
    C_GPS = ncread('func_gps.nc','C_GPS')
    
    # Open solar cycle index.
    fname_indices = 'solarGeomagnetic_daily_2000to2023.nc'
    allIndex = ncread(fname_indices,'solarGeomagneticIndex')
    indices_year = allIndex[0,:]
    indices_doy = allIndex[1,:]
    indicesOnly = allIndex[3:21]
    allIndex[allIndex == 999.9] = np.nan
    indexValue = 8 #Use number in txt file.
    solarIndex = indicesOnly[indexValue-1,:]
    
    # Get month and local-time.
    input_month = get_month(input_doy, input_year)
    
    itime = input_month + ((input_year-2007)*12)
    specSolarIndex = solarIndex[(indices_year == input_year)&(indices_doy == input_doy)]
    recon_nme,recon_hme,recon_h = reconstructCoefficients_specSolarIndex(itime,specSolarIndex)
    zenithAngle = calculate_zenithAngle(input_lat,input_lst,input_doy,input_year)
    cosZenith = np.cos(zenithAngle*np.pi/180) #cos(zenith Angle).
    nme,hme,h = calculate_Nme_Hme_H_efficient(itime,cosZenith,recon_nme,recon_hme,recon_h)
    params_SZA = [nme,hme,h]

    #Calculate the g-function.
    gFunction = C0_GPS + (C_GPS[0]*cosZenith**1) + (C_GPS[1]*cosZenith**2) + (C_GPS[2]*cosZenith**3)+ (C_GPS[3]*cosZenith**4) 
    gFunction = gFunction + (C_GPS[4]*cosZenith**5) + (C_GPS[5]*cosZenith**6) + (C_GPS[6]*cosZenith**7)
    
    #Calculate and apply corrections.
    if (input_lat >= -84)&(input_lat <= 84)&(input_lst >= 1)&(input_lst <= 23):
        correct = nme_photo_correction_solarIndexInput(itime,input_lst,input_lat,specSolarIndex)
        nme_correction = correct.data[0]
        correct = hme_photo_correction_solarIndexInput(itime,input_lst,input_lat,specSolarIndex)
        hme_correction = correct.data[0]
        correct = h_photo_correction_solarIndexInput(itime,input_lst,input_lat,specSolarIndex)
        h_correction = correct.data[0]
        
        if (nme_correction > 0):
            nme = nme + nme_correction
            hme = hme + hme_correction
            h = h + h_correction
        else:
            nme = nme
            hme = hme
            h = h
    else:
        nme = nme
        hme = hme
        h = h
    
    params_ALL = [nme,hme,h]
    z = input_alt
    zprime = (z - hme)/h
    ne_EPROBED = 1e11*nme*np.exp(0.5*(1-zprime-(gFunction*np.exp(-1*zprime))))

    return ne_EPROBED,z,params_SZA,params_ALL


def eprobedFunc_lat_lst_monthly(input_lat,input_alt,input_lst,input_month,input_doy,input_year):
    #This function calculates E-region Ne vertical profile.
    #Inputs: 
    #   input_lat is latitude in degrees.
    #   input_alt is altitude in km.   
    #   input_lst is local-solar-time in hours.
    #   input_month is month.
    #   input_doy is day of year.
    #   input_year is year.
    #Output: ne_EPROBED,z,params_SZA,params_ALL
    #   ne_EPROBED is E-region vertical profile in e/m3.
    #   z is altitude bin in km.
    #   params_SZA = [Nme,Hme,H] are the 3 parameters from SZA component.
    #   params_ALL = [Nme,Hme,H] are the 3 parameters from SZA+nonSZA component.
    
    # Open the NetCDF file of g-function.
    C0_GPS = ncread('func_gps.nc','C0_GPS')
    C_GPS = ncread('func_gps.nc','C_GPS')
        
    itime = input_month + ((input_year-2007)*12)
    recon_nme,recon_hme,recon_h = reconstructCoefficients(itime)
    zenithAngle = calculate_zenithAngle(input_lat,input_lst,input_doy,input_year)
    cosZenith = np.cos(zenithAngle*np.pi/180) #cos(zenith Angle).
    nme,hme,h = calculate_Nme_Hme_H_efficient(itime,cosZenith,recon_nme,recon_hme,recon_h)
    params_SZA = [nme,hme,h]

    #Calculate the g-function.
    gFunction = C0_GPS + (C_GPS[0]*cosZenith**1) + (C_GPS[1]*cosZenith**2) + (C_GPS[2]*cosZenith**3)+ (C_GPS[3]*cosZenith**4) 
    gFunction = gFunction + (C_GPS[4]*cosZenith**5) + (C_GPS[5]*cosZenith**6) + (C_GPS[6]*cosZenith**7)
    
    #Calculate and apply corrections.
    #Calculate and apply corrections.
    if (input_lat >= -84)&(input_lat <= 84)&(input_lst >= 1)&(input_lst <= 23):
        correct = nme_photo_correction(itime,input_lst,input_lat)
        nme_correction = correct
        correct = hme_photo_correction(itime,input_lst,input_lat)
        hme_correction = correct
        correct = h_photo_correction(itime,input_lst,input_lat)
        h_correction = correct
        
        if (nme_correction > 0):
            nme = nme + nme_correction
            hme = hme + hme_correction
            h = h + h_correction
        else:
            nme = nme
            hme = hme
            h = h
    else:
        nme = nme
        hme = hme
        h = h
    
    params_ALL = [nme,hme,h]
    z = input_alt
    zprime = (z - hme)/h
    ne_EPROBED = 1e11*nme*np.exp(0.5*(1-zprime-(gFunction*np.exp(-1*zprime))))

    return ne_EPROBED,z,params_SZA,params_ALL



