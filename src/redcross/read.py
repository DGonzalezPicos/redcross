__all__ = ['read_harpsn']


from .datacube import Datacube
#import datacube
import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
import os, copy

def read_giano(files):
    # read the cube's parameters from a sample file
    hdul = fits.open(os.path.join(files[0]))
    hdr = hdul[0].header
    data = hdul[1].data
    #
    nObs = len(files)
    nOrders = len(data)
    nPix = len(data[0][1])

    bervkeyword = 'HIERARCH TNG DRS BERV'    
    berv, mjd, airmass = (np.array([]) for _ in range(3))

    wlt, flux, snr = (np.zeros((nObs, nOrders, nPix)) for _ in range(3))
    for frame in range(len(files)):
        filename = files[frame].split('/')[-1]
        print('--->', frame, filename, end='\r')
        hdul = fits.open(os.path.join(files[frame]))
        hdr = hdul[0].header
        data = hdul[1].data

        berv = np.append(berv, hdr[bervkeyword])
        beta = (1.0-(hdr[bervkeyword]*u.km/u.s/const.c).decompose().value) #Doppler factor BERV.
        mjd=np.append(mjd, hdr['MJD-OBS'])
        airmass=np.append(airmass, hdr['AIRMASS'])
        for order in range(len(data)):
            _, wave1, flux1, snr1 = hdul[1].data[order]
            # wlt[frame,nOrders-1-order,:] = airtovac(wave1*10)*beta # in AA and BERV correction applied here
            # wlt[frame,nOrders-1-order,:] = wave1*10*beta # in AA and BERV correction applied here
            sort = np.argsort(wave1)
            wlt[frame,nOrders-1-order,:] = wave1[sort]*10 # in AA (DON'T APPLY BERV correction)
            # for giano, removing the tellurics is better in the OBSERVER'S frame

            flux[frame,nOrders-1-order,:] = flux1[sort]
            snr[frame,nOrders-1-order,:] = snr1[sort]
            # comment: add orders from left to right (small to big)

        info = {'airmass':airmass, 'MJD':mjd,'BERV':berv,
                'RA_DEG':hdr['RA-DEG'], 'DEC_DEG':hdr['DEC-DEG'], 'DATE':hdr['DATE-OBS']}

    dc = Datacube(flux=np.swapaxes(flux, 0, 1), wlt=np.swapaxes(wlt, 0, 1), flux_err=np.swapaxes(snr, 0, 1), **info)  
    return dc

def airtovac(wlA):
    #Convert wavelengths (nm) in air to wavelengths in vaccuum (empirical).
    s = 1e4 / wlA
    n = 1 + (0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) +
    0.0001599740894897 / (38.92568793293 - s**2))
    return(wlA*n)

def read_wave_from_e2ds_header(h,mode='HARPS'):
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


def read_harpsn(files, filetype='s1d', max_files=1000):
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
    dc = Datacube(flux=np.swapaxes(flux, 0, 1), wlt=np.swapaxes(wlt, 0, 1), **info)  

    return dc



    
    
    
    
    
    
    
    