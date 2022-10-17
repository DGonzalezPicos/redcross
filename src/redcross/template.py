#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 08:37:34 2022

@author: dario
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splrep, splev
import astropy.units as u
import astropy.constants as const
from .datacube import Datacube
from pathos.pools import ProcessPool
from scipy import ndimage


# constants in CGS
c_cgs = 29979245800.0 # cm/s
c = 2.9979e5 # km/s
h = 6.62606957e-27
kB = 1.3806488e-16 

class Template(Datacube):
    def __init__(self, wlt=None, flux=None, file=None):
        if not file is None:
            read_file = np.load(file)
            if len(read_file) > 2:
                self.wlt, self.flux, self.cont = read_file
            else:
                self.wlt, self.flux = read_file
        else:
            self.wlt = wlt
            self.flux = flux
            
        self.mode = 'linear' # by default use **linear** interpolation
    @property
    def resolution(self):
        return np.round(np.median(self.wlt) / np.mean(np.diff(self.wlt)), 0)
    
    def get_spline(self):
        self.sort()
        self.cs = splrep(self.wlt, self.flux)
        return self
    
    def blackbody(self, T, nu):
        b = 2.*h*nu**3./c_cgs**2.
        b /= (np.exp(h*nu/kB/T)-1.)
        return b
    
    def scale_model(self, T_star=8000., transit_depth=0.01):
        freq = c_cgs/(self.wlt*1e-8) # where wlt in AA
        Fs = np.pi * self.blackbody(T_star, freq)
        
        # renaming to make it more readable
        Fp, Cp = self.flux, self.cont
        D = transit_depth
        # scale and normalise the model
        template = 1 + (D * Fp/Fs)
        continuum = 1 + (D * Cp/Fs)
        template /= continuum
        self.flux = template
        return self
        
                   
            
    def plot(self, ax=None, mode='1D', fig=None, **kwargs):
        ax = ax or plt.gca()
        if mode == '1D':
            # if not self.wlt.shape==self.flux.shape:
            #     ax.plot(self.wlt, self.gflux[0], **kwargs)
            # else:
            ax.plot(self.wlt, self.flux, **kwargs)
        elif mode == '2D':
            ext = [self.wlt.min(), self.wlt.max(), self.rv.min(), self.rv.max()]
            obj = ax.imshow(self.gflux, origin='lower', aspect='auto', extent=ext, **kwargs)
            if not fig is None: fig.colorbar(obj, ax=ax, pad=0.05)
            # ax.set(ylabel='RV (km/s)')
        return ax
    
    
    def sort(self):
        '''
        Sort `wlt` and `flux` vectors by wavelength.

        Returns
        sorted Template
        -------
        '''
        sort = np.argsort(self.wlt)
        self.wlt = self.wlt[sort]
        self.flux = self.flux[sort]
        return self
    
    def interpolate(self, beta=1., new_wave=None, return_self=False):
        if new_wave is None:
            new_wave = self.new_wlt
            
        
        # if np.isnan(self.cs[1]).any():
        if self.mode == 'linear':
            gflux = self.cl(new_wave*beta)
            
        elif self.mode == 'spline':
            gflux = splev(new_wave*beta, self.cs)
            
        if return_self:
            self.wlt = new_wave
            self.flux = gflux
            return self
        else:
            return gflux
    
    def crop(self, wmin, wmax, eps=0.1):
        '''
        Crop template in wavelength. The `eps` argument extends the range to avoid
        potential interpolation issues.

        Parameters
        ----------
        wmin : float
            minimum wavelength The default is None.
        wmax : float
            maximum wavelength. The default is None.
        eps : float, optional
            Add (eps*100)% to both edges. The default is 0.1.

        Returns
        -------
        TYPE
            template.

        '''
        span = wmax - wmin
        mask = self.wlt < (wmin-(eps*span))
        mask += self.wlt > (wmax+(eps*span))
        self.wlt = self.wlt[~mask]
        self.flux = self.flux[~mask]
        return self
        
        
    def shift_2D(self, RV, wave=None, num_cpus=0, mode='linear'):
        # compute the spline coefficients once (and store them)
        self.mode = mode
        if self.mode == 'spline':
            self.get_spline()
        elif self.mode == 'linear':
            self.cl = interp1d(self.wlt, self.flux, bounds_error=False, fill_value=0.0)
            
            
        # c = const.c.to('km/s').value
        self.rv = RV
        beta = 1 - (self.rv/c)
        
        temp = self.copy()
        if not wave is None: 
            temp.crop(np.nanmin(wave), np.nanmax(wave))
            temp.new_wlt = wave
        
    
        if num_cpus>0:
            pool = ProcessPool(nodes=num_cpus)
            output = pool.amap(temp.interpolate, beta).get()
            
        else:
            output = [temp.interpolate(beta[i]) for i in range(beta.size)]
            
        temp_dc = Datacube(wlt=temp.new_wlt, flux=np.array(output) )
        temp_dc.rv = temp.rv
        return temp_dc
    
    
    def load_TP(self):
        path = 'data/PT-two_point_profile.npy'
        return np.load(path)
    
    def vactoair(self):
        """VACUUM to AIR conversion as actually implemented by wcslib.
        Input wavelength with astropy.unit
        """
        cenwave0 = np.median(self.wlt)
        wave = (self.wlt*u.AA).to(u.m).value
        n = 1.0
        for k in range(4):
            s = (n/wave)**2
            n = 2.554e8 / (0.41e14 - s)
            n += 294.981e8 / (1.46e14 - s)
            n += 1.000064328
        
        print('Vacuum to air conversion...')
        shift = shift = ((self.wlt/n) - self.wlt) / self.wlt
        rv_shift = (np.mean(shift)* const.c).to('km/s').value
        print('--> Shift = {:.2f} A = {:.2f} km/s'.format(np.mean(shift)*cenwave0, rv_shift))
        
        self.wlt = (self.wlt / n)
        return self

    def airtovac(self):
      #Convert wavelengths (AA) in air to wavelengths (AA) in vaccuum (empirical).
      s = 1e4 / self.wlt
      n = 1 + (0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) +
      0.0001599740894897 / (38.92568793293 - s**2))
      self.wlt *= n
      return self  


    def high_pass_gaussian(self, window=15., deltaV=None, mode='subtract', debug=False):   
        
        if deltaV is not None:
            self.dRV = np.mean(np.diff(self.wlt)/np.median(self.wlt)) * c
            window = deltaV / self.dRV
            print(window)
        lowpass = ndimage.gaussian_filter1d(self.flux, window)
        
        self.flux = getattr(np, mode)(self.flux, lowpass) # subtract or divide
        return self
    
    
    
    def remove_continuum(self, exclude=None, wave_units=u.AA, ax=None):
        from specutils.spectra import Spectrum1D, SpectralRegion
        from specutils.fitting import fit_generic_continuum
        from astropy import units as u
        
        
        spectrum = Spectrum1D(flux=self.flux*u.Jy, spectral_axis=self.wlt*u.AA)
        if exclude != None:
            exclude_region = SpectralRegion(exclude[0]*wave_units, exclude[1]*wave_units)
        else:
            exclude_region = None
        
        g1_fit = fit_generic_continuum(spectrum, exclude_regions=exclude_region)
        y_continuum_fitted = g1_fit(self.wlt*wave_units)
        if ax != None:
            self.plot(ax=ax, lw=0.2, label='Template')
            ax.plot(self.wlt, y_continuum_fitted, label='Fitted continuum')
            ax.legend()
            plt.show()
            
        
        self.flux /= y_continuum_fitted.value
        return self
    
    # def pyAstro_convolve(self, row=-1):
    #     from PyAstronomy import pyasl
    #     if row < 0:
    #         flux = self.flux
    #     else:
    #         flux = self.gflux[row, ~self.nans]
    #     newflux = pyasl.instrBroadGaussFast(self.wlt[~self.nans], flux,
    #                                         self.res, edgeHandling="firstlast", fullout=False, equid=True,maxsig=5.0)
        
    #     return newflux
    
    def convolve_instrument(self, res):
        from PyAstronomy import pyasl
        # dRV = np.mean(np.diff(wave)/np.median(wave)) * c
        # print(dRV)
        
        self.flux = pyasl.instrBroadGaussFast(self.wlt, self.flux,
                                            res, edgeHandling="firstlast",
                                            fullout=False, equid=True, maxsig=5.0)
        return self
    
    def boost(self, factor=10.):
        mean = np.nanmean(self.flux)
        new_flux = self.flux - mean
        new_flux *= factor
        self.flux = new_flux + mean
        return self
    
    
    def find_lines(self, n=5):
        from scipy.signal import find_peaks
        peaks, pdict = find_peaks(self.flux, height=0)
    
        sort = np.argsort(pdict['peak_heights'])[::-1]
        heights = pdict['peak_heights'][sort]
        peaks = peaks[sort]
        self.heights = heights[:n]
        self.peaks = peaks[:n]
        return self
    
    
    def save(self, outname):
        np.save(np.array([self.wlt, self.flux]))
    
    def make_cube(self, N):
        flux = np.tile(self.flux, N).reshape(N, *self.flux.shape)
        dc = Datacube(wlt=self.wlt, flux=flux)
        return dc
    
    def merge(self, templates=[], eps=0.1):
        
        wave, flux = ([] for _ in range(2))
        for t in templates:
            wave.append(t.wlt)
            flux.append(t.flux-1.) # add Fp_i/Fs (need to subtract 1 from **1+Fp_i/Fs**)
            
        wave, flux = np.array(wave), np.array(flux)
        
        # check if they have the same wavelength grid (`eps` is the threshold parameter)
        diff = np.diff(np.median(wave, axis=1))
        if np.any(diff > eps):
            print('Unable to merge templates with different wavelength grids...')
        else:
            self.wlt = np.median(wave, axis=0)
            self.flux = np.sum(flux, axis=0)
            return self
        
    @staticmethod
    def rotational_kernel(vsini, wave):
        dRV = np.mean(np.diff(wave)/np.median(wave)) * c
        nker = 401
        hnker = (nker-1)//2
        rker = np.zeros(nker)
        for ii in range(nker):
            ik = ii - hnker
            x = ik*dRV / vsini
            if np.abs(x) < 1.0:
                y = np.sqrt(1-x**2)
                rker[ii] = y
        rker /= rker.sum()
        
        return rker
    
    def rot_broaden(self, vsini, wave):
        rker = self.rotational_kernel(vsini, wave)
        mean = np.nanmean(self.flux)
        self.flux = np.convolve(self.flux-mean, rker, mode='same')
        self.flux += mean
        return self
    
        

    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        