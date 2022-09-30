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

class Template(Datacube):
    def __init__(self, wlt=None, flux=None, filepath=None):
        if not filepath is None:
            self.filepath = filepath
            self.wlt, self.flux = np.load(self.filepath)
        else:
            self.wlt = wlt
            self.flux = flux
    @property
    def resolution(self):
        return np.round(np.median(self.wlt) / np.mean(np.diff(self.wlt)), 0)
                   
            
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
        cs = splrep(self.wlt*beta, self.flux)
        if np.isnan(cs[1]).any():
                gflux = interp1d(self.wlt*beta, self.flux, bounds_error=False, fill_value=0.0)(new_wave)
        else:
            gflux = splev(new_wave, cs)
            
        if return_self:
            self.wlt = new_wave
            self.flux = gflux
            return self
        else:
            return gflux
    
    def crop(self, wave, eps=0.1):
        span = wave.max() - wave.min()
        mask = self.wlt < (wave.min()-(eps*span))
        mask += self.wlt > (wave.max()+(eps*span))
        self.wlt = self.wlt[~mask]
        self.flux = self.flux[~mask]
        return self
        
        
    def shift_2D(self, RV, wave=None, num_cpus=6):
        
        c = const.c.to('km/s').value

        self.rv = RV
        beta = 1 + (self.rv/c)
        
        temp = self.copy()
        if not wave is None: 
            temp.crop(wave)
            temp.new_wlt = wave
        
        
        # with mp.Pool(num_cpus) as p:
        #         output = p.map(temp.interpolate, beta)
        pool = ProcessPool(nodes=num_cpus)
        output = pool.amap(temp.interpolate, beta).get()
                
        temp.gflux = np.array(output)
        temp.wlt = temp.new_wlt
        temp_dc = Datacube(wlt=temp.new_wlt, flux=temp.gflux)
        temp_dc.rv = temp.rv
        return temp_dc
    
    
    def shift_2D_slow(self, RV, wave=None):
        temp = self.copy()
        c = const.c.to('km/s').value
        self.rv = RV
        beta = 1 - (self.rv/c)
        
        
        if not wave is None: 
            temp.wlt = wave
        
        temp.gflux = np.zeros((self.rv.size, temp.wlt.size))
        # cs = splrep(wave_in*beta[j], self.flux)
        for j in range(self.rv.size):
            cs = splrep(self.wlt*beta[j], temp.flux)
            if np.isnan(cs[1]).any():
                temp.gflux[j,] = interp1d(self.wlt*beta[j], temp.flux, bounds_error=False,
                                          fill_value=0.0)(temp.wlt)
            else:
                temp.gflux[j,] = splev(temp.wlt, cs)
        return temp
    
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


    def high_pass_gaussian(self, window=None, dRV=0., debug=False):
        from scipy import ndimage
        if dRV > 0.:
            pixscale = const.c.to('km/s').value * np.mean(np.diff(self.wlt)) / np.median(self.wlt)
            window = 2 * dRV * pixscale
            if debug: print('window = {:.2f} pixels'.format(window))
            
        lowpass = ndimage.gaussian_filter1d(self.flux, window)
        if hasattr(self, 'gflux'):
            nans = self.nans
            for row in range(self.gflux.shape[0]):
                self.gflux[row,~nans] /= ndimage.gaussian_filter1d(self.gflux[row,~nans], window)
        self.flux /= lowpass
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
    
    # def convolve_instrument(self, res=50e3):
    #     '''convolve to instrumental resolution with a Gaussian kernel'''
    #     from astropy.convolution import Gaussian1DKernel, convolve
        
    #     cenwave = np.median(self.wlt)
    #     # Create kernel
    #     g = Gaussian1DKernel(stddev=0.5*cenwave/res) # set sigma = FWHM / 2. = lambda / R
    #     self.flux = convolve(self.flux, g)
    #     return self
    
    def resample(self, wave, mode='2D'):
        
        if mode == '1D':
            cs = splrep(self.wlt, self.flux)
            self.flux_res = splev(wave, cs)
            
        elif mode == '2D':
            self.gflux_res = np.zeros((self.gflux.shape[0], wave.size))
            for i in range(self.gflux.shape[0]):
                cs = splrep(self.wlt, self.gflux[i,:])
                self.gflux_res[i,:] = splev(wave, cs)
        return self
    
    def pyAstro_convolve(self, row=-1):
        from PyAstronomy import pyasl
        if row < 0:
            flux = self.flux
        else:
            flux = self.gflux[row, ~self.nans]
        newflux = pyasl.instrBroadGaussFast(self.wlt[~self.nans], flux,
                                            self.res, edgeHandling="firstlast", fullout=False, equid=True,maxsig=5.0)
        
        return newflux
    def convolve_instrument(self, res=50e3, num_cpus=5):
        import multiprocessing as mp
        
        self.res = res
        if hasattr(self, 'gflux'):
            rows = np.arange(0, self.gflux.shape[0])
            with mp.Pool(num_cpus) as p:
                output = p.map(self.pyAstro_convolve, rows)
                
            self.gflux[:,~self.nans] = np.array(output)
        else:
            self.flux = self.pyAstro_convolve()
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