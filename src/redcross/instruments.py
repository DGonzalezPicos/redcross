#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 10:37:27 2022

@author: dario
"""

from .datacube import Datacube
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
import os, copy
class HARPSN(Datacube):
    
    def __init__(self, wlt=None, flux=None, flux_err=None, files=None, **header):
        super().__init__(flux, wlt, flux, **header)
        
    
    def obs_summary(self):
        fig, ax = plt.subplots(3,figsize=(7,5), sharex=True)
    
        self.flux_frame = np.median(self.flux, axis=(0,2))
    
        labels = ['Airmass', 'Mean flux per frame', 'BERV (km/s)']
        colors = ['r','g','b']
        for i, attr in enumerate(['airmass', 'flux_frame', 'BERV']):
            ax[i].plot(getattr(self, attr), '--o', ms=1., c=colors[i])
            ax[i].set(ylabel=labels[i])
            
        ax[len(ax)-1].set_xlabel('Frame number')
        plt.show()
        return None
    
    
    
    def read(self, files, filetype='e2ds', max_files=1000, cache=False, save=True):
        
        
        data_dir = files[0].split('HARPN')[0]
        dc_file = data_dir+'datacube_raw.npy'
        if cache:
        # check if preloaded file exists on the given directory
            if os.path.exists(dc_file):
                return HARPSN().load(dc_file)
            else:
                print('No preloaded datacube file found...')

        
        catkeyword = 'OBS-TYPE'
        bervkeyword = 'HIERARCH TNG DRS BERV'

        flux, wlt = ([] for _ in range(2))
        berv, airmass, npx, mjd = (np.array([]) for _ in range(4))
        for i,f in enumerate(files[:max_files]):
            filename = f.split('/')[-1]
            print('--->', i, filename, end='\r')
            hdul = fits.open(os.path.join(f))
            data = copy.deepcopy(hdul[0].data)
            hdr = hdul[0].header
            hdul.close()
            if hdr[catkeyword] == 'SCIENCE':
                berv = np.append(berv, hdr[bervkeyword])
                mjd=np.append(mjd, hdr['MJD-OBS'])
                airmass=np.append(airmass, hdr['AIRMASS'])
                flux.append(data)
                if filetype == 'e2ds':
    #                 norders=np.append(norders,hdr['NAXIS2'])        
                    wavedata=airtovac(read_wave_from_e2ds_header(hdr,mode='HARPSN')) # Angstrom
    #                beta = (1.0-(hdr[bervkeyword]*u.km/u.s/const.c).decompose().value) #Doppler factor BERV.
    #                wlt.append(wavedata*beta)
                    wlt.append(wavedata)  # DON'T APPLY BERV here
                    
                elif filetype == 's1d':
                    
                    gamma = (1.0-(hdr[bervkeyword]*u.km/u.s/const.c).decompose().value) #Doppler factor BERV.
                    wavedata = (hdr['CDELT1']*np.arange(len(flux[-1]), dtype=float)+hdr['CRVAL1'])*gamma
                    wlt.append(wavedata)
                
        info = {'airmass':airmass, 'MJD':mjd,'BERV':berv,
               'RA_DEG':hdr['RA-DEG'], 'DEC_DEG':hdr['DEC-DEG'], 'DATE':hdr['DATE-OBS']}
        dc = HARPSN(flux=np.swapaxes(flux, 0, 1), wlt=np.swapaxes(wlt, 0, 1), **info)  

        if save: dc.save(dc_file)
        return dc
    

# =============================================================================
#                       GENERIC UTILITY FUNCTIONS
# =============================================================================
def read_wave_from_e2ds_header(h,mode='HARPS', cache=False):
    """
    This reads the wavelength solution from the HARPS header keywords that
    encode the coefficients as a 4-th order polynomial.
    """

    if mode not in ['HARPS','HARPSN','HARPS-N','UVES']:
        raise ValueError("in read_wave+from_e2ds_header: mode needs to be set to HARPS, HARPSN or UVES.")
    npx = h['NAXIS1']
    no = h['NAXIS2']
    x = np.arange(npx, dtype=float) #fun.findgen(npx)
    wave=np.zeros((npx,no))

    if mode == 'HARPS':
        coeffkeyword = 'ESO'
    if mode in ['HARPSN','HARPS-N']:
        coeffkeyword = 'TNG'
    if mode == 'UVES':
        delt = h['CDELT1']
        for i in range(no):
            keystart = h[f'WSTART{i+1}']
            # keyend = h[f'WEND{i+1}']
            # wave[:,i] = fun.findgen(npx)*(keyend-keystart)/(npx-1)+keystart
            wave[:,i] = np.arange(npx, dtype=float)*delt+keystart #fun.findgen(npx)*delt+keystart
            #These FITS headers have a start and end, but (end-start)/npx does not equal
            #the stepsize provided in CDELT (by far). Turns out that keystart+n*CDELT is the correct
            #representation of the wavelength. I do not know why WEND is provided at all and how
            #it got to be so wrong...
    else:
        key_counter = 0
        for i in range(no):
            l = x*0.0
            for j in range(4):
                l += h[coeffkeyword+' DRS CAL TH COEFF LL%s' %key_counter]*x**j
                key_counter +=1
            wave[:,i] = l
    wave = wave.T
    return(wave)

def airtovac(wlA):
    #Convert wavelengths (nm) in air to wavelengths in vaccuum (empirical).
    s = 1e4 / wlA
    n = 1 + (0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) +
    0.0001599740894897 / (38.92568793293 - s**2))
    return(wlA*n)