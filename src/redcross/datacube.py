__all__ = ['Datacube']

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splrep, splev
from copy import deepcopy
import astropy.units as u
import astropy.constants as const
from astropy.convolution import Gaussian1DKernel, convolve_fft
import warnings
warnings.simplefilter("ignore")
# np.seterr(all="ignore")
import multiprocessing as mp


class Datacube:
    '''Main object to store, plot and work with data'''
    def __init__(self, flux=None, wlt=None, flux_err=None, night=None,**header):
        self.flux = flux
        self.wlt = wlt
        self.flux_err = flux_err
        self.night = night
        for key in header:
            setattr(self, key, header[key])
        self.frame = 'telluric' # by default
            
    @property
    def nObs(self):
        if len(self.shape) < 3:
            return len(self.flux)
        else:
            return self.shape[1]
    
    @property
    def nPix(self):
        return self.shape[-1]
    
    @property
    def nOrders(self):
        return self.flux.shape[0]
    @property
    def shape(self):
        return self.flux.shape
    @property
    def nans(self):
        return np.isnan(self.wlt)
    
    @property
    def nan_frac(self):
        '''fraction of NaN pixel channels (over unity)'''
        nans = np.isnan(self.wlt)
        return np.round(nans[nans==True].size / nans.size, 4)

# =============================================================================
#                       FUNCTIONS   
# =============================================================================
    def get_header(self):
        keys_to_extract = ['airmass','MJD','BERV','RA_DEG','DEC_DEG','DATE']    
        self.header = {key: self.__dict__[key] for key in keys_to_extract} 
        return self
    
    def plot(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        if len(self.wlt.shape)>1:
            wave = np.nanmedian(self.wlt, axis=0)
        else:
            wave = self.wlt
        ax.plot(wave, np.nanmedian(self.flux, axis=0), **kwargs)
#            ax.set(xlabel='Wavelength ({:})'.format(self.wlt_unit), ylabel='Flux')
        
        return ax
        

    def imshow(self, fig=None, ax=None, s=3.,title='', vrange=None, **kwargs):
        
        # nans = np.isnan(self.wlt)
        if not vrange is None: 
            s=-1.
            vmin = vrange[0]
            vmax = vrange[-1]
        if s > 0.:
            vmin = np.nanmean(self.flux)-np.nanstd(self.flux) * s
            vmax = np.nanmean(self.flux)+np.nanstd(self.flux) * s
        if s == 0:
            vmin = np.nanmin(self.flux)
            vmax = np.nanmax(self.flux)
            
        # plot y-axis as phase (if available)
        if hasattr(self, 'phase'):
            y1, y2 = np.min(self.phase), np.max(self.phase)
        else:
            y1, y2 = 0, self.nObs-1
            
        ext = [np.nanmin(self.wlt), np.nanmax(self.wlt), y1, y2]
        ax = ax or plt.gca()
        obj = ax.imshow(self.flux,origin='lower',aspect='auto',
                        extent=ext,vmin=vmin,vmax=vmax, **kwargs)
        if not fig is None: fig.colorbar(obj, ax=ax, pad=0.05)
        
        current_cmap = plt.cm.get_cmap()
        current_cmap.set_bad(color='white')
        ax.set(title=title)

        return obj
    
    # FAST version
    def estimate_noise(self):
        '''assign a noise value to each data point by taking the mean of the stdev
        in time and pixel dimensions'''
        xStDev = np.nanstd(self.flux, axis=0)
        yStDev = np.nanstd(self.flux, axis=1)
        # my estimate method
        self.flux_err = (0.5*(yStDev[:,np.newaxis] + xStDev[np.newaxis,:]))
        # as in Nugroho+2021
        self.flux_err = np.outer(yStDev, xStDev)/ np.nanstd(self.flux)
        return self
        
    
    def save(self, outname):
        np.save(outname, self.__dict__) 
        print('{:} saved...'.format(outname))
        return None
    
    def load(self, path):
        print('Loading Datacube from...', path)
        d = np.load(path, allow_pickle=True).tolist()
        for key in d.keys():
            setattr(self, key, d[key])
        self.get_header()
        return self
    
    
    def airtovac(self,wlA):
        #Convert wavelengths (AA) in air to wavelengths (AA) in vaccuum (empirical).
        s = 1e4 / wlA
        n = 1 + (0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) +
        0.0001599740894897 / (38.92568793293 - s**2))
        return(wlA*n)
    
    def inject_signal(self, planet, template, RV=None, factor=1., ax=None):
        temp = template.copy()
        
        if factor > 0.: temp.boost(factor)
        
        # get 2D template shifted at given RV or at planet.RV if no RV vector is passed
        if RV is None:
            RVt = planet.RV
        else:
            RVt = RV*np.ones_like(planet.RV)

        temp = temp.shift_2D(RVt, self.wlt)
        
        # inject only for out-of-eclipse frames
        mask = self.mask_eclipse(planet, return_mask=True)
        self.flux[~mask,:] *= temp.flux[~mask,:]
        
        if ax != None: self.imshow(ax=ax)
        return self
    
    
    def order(self, o):
        dco = self.copy()
    
        if type(o) in [list, np.ndarray]:
                dco.wlt = dco.wlt[o[0]:o[-1]+1,]
                dco.flux = dco.flux[o[0]:o[-1]+1,]
                if not dco.flux_err is None:
                    dco.flux_err = dco.flux_err[o[0]:o[-1]+1,]
    
        else:
            if len(dco.wlt.shape)>2:
                dco.wlt = dco.wlt[o,:,:]
            else:
                dco.wlt = dco.wlt[o,:]
            dco.flux = dco.flux[o,:,:]
            # dco.order = orders
            if not dco.flux_err is None:
                dco.flux_err = dco.flux_err[o,:,:]
        dco.o = o # store order number
        return dco
 
    
    def normalise(self, ax=None):
       self.flux = (self.flux.T / np.nanmedian(self.flux, axis=1)).T
       if not self.flux_err is None:
           self.flux_err = (self.flux_err.T / np.nanmedian(self.flux, axis=1)).T
           
       if ax != None: self.imshow(ax=ax)
       return self
        
        
        
    def copy(self):
        return deepcopy(self)
    
    
    def sigma_clip(self, sigma=5., axis=0, debug=False, ax=None):
        '''For each pixel channel, replace outliers from each frame by the median column value
        outliers are points beyond nSigma from the column mean'''

        nans = np.isnan(self.wlt)
        flux = self.flux
        outliers = 0
        # for x in range(self.nPix):
        if axis==0: # over wavelengths
            for x in np.argwhere(nans==False):   
                x = int(x)
                mean, std = np.nanmean(flux[:,x]), np.nanstd(flux[:,x])
                mask = np.abs(flux[:,x]-mean) > sigma*std
                self.flux[mask,x] = mean
                # self.flux[mask, x] = np.nan
                if not self.flux_err is None:
                    self.flux_err[mask,x] = np.nanmean(self.flux_err[:,x])
                    # self.flux_err[mask,x] = np.nan
                outliers += mask[mask==True].size
            
        else: # over time
            for i in range(self.shape[0]):
                mean, std = np.nanmean(flux[i,:]), np.nanstd(flux[i,])
                mask = (np.abs(flux[i,:]-mean) / std) > sigma
                self.flux[i,mask] = np.nanmedian(flux)
                if not self.flux_err is None:
                    self.flux_err[i,mask] = np.nanmedian(self.flux_err[i,:])
                outliers += mask[mask==True].size
            
            
        if debug:
            print('outliers = {:.2e} %'.format(outliers/self.flux.size))
            
        if ax != None: self.imshow(ax=ax)   
        return self
    
    def remove_continuum(self, mode='polyfit', deg=3., ax=None):
        '''for each order, remove the continuum by dividing by the residuals 
        from the master subtraction'''
        

        if mode == 'polyfit':
            for f in range(self.nObs):
                if len(self.wlt.shape) > 1: # each frame with its wavelength grid
                    wave = self.wlt[f,:]
                else: # common wavelength grid
                    wave = self.wlt
                nans = np.isnan(wave)
                model = np.poly1d(np.polyfit(wave[~nans], self.flux[f,~nans], deg))
                continuum = model(wave[~nans])
                self.flux[f,~nans] /= continuum
                if not self.flux_err is None:
                    self.flux_err[f,~nans] /= continuum
        else:
            master = np.nanmedian(self.flux, axis=0) 
            g1d_kernel = Gaussian1DKernel(300)
            for frame in range(self.shape[0]):
                divide = convolve_fft((self.flux[frame,] / master), g1d_kernel, boundary='wrap')
                self.flux[frame,] /= divide
        if ax != None: self.imshow(ax=ax)        
        return self
    
    def airmass_detrend(self, log_space=False, ax=None):
        '''Fit a second order polynomial to each column and divide(subtract) the fit 
        in linear(log) space'''
        nans = np.isnan(self.wlt)
        
        if log_space:
            for j in np.where(nans==False)[0]:
                y = np.log(self.flux[:,j])
                fit = np.poly1d(np.polyfit(self.airmass,y,2))(self.airmass)
                self.flux[:,j] = np.exp(y - fit)
        else:
#            for j in range(self.nPix):
            # for pix in np.argwhere(nans==False):
            #     j = int(pix)
            for j in np.where(nans==False)[0]:
                y = self.flux[:,j]
                fit = np.poly1d(np.polyfit(self.airmass,y,2))(self.airmass)
                self.flux[:,j] /= fit
                if not self.flux_err is None:
                    self.flux_err[:,j] /= fit
        if ax != None: self.imshow(ax=ax)
        return self
            
    
    def mask_cols(self, sigma=3., mode='flux', metric='mean', cycles=1, debug=False, ax=None):
        '''
         
    
         Parameters
         ----------
         sigma : float
             NUmber of standard deviations. The default is 3..
         mode : str, optional
             Select data: 'flux' or 'flux_err'. The default is 'flux'.
         metric : str, optional
             Operation to perform on 'mode': 'mean' or 'std'. The default is 'mean'.
         cycles : int, optional
             Number of iterations. The default is 1.
         debug : bool, optional
             True to show what it is doing. The default is False.
         ax : plt.ax(), optional
             Show datacube after applying function. The default is None.
    
         Returns
         -------
         TYPE
             Datacube()
    
         '''
        


        k = 0
        while k < cycles:
            y = getattr(np, 'nan'+metric)(getattr(self, mode), axis=0)
            mean, std = np.nanmean(y), np.nanstd(y)
            mask = np.abs(y - mean) > (sigma * std)
        
            n = mask.size
            frac_masked = mask[mask==True].size*100/n
            
            if frac_masked < 10.:
                self.wlt[mask] = np.nan
                self.flux[:,mask] = np.nan
                if not self.flux_err is None:
                    self.flux_err[:,mask] = np.nan 
                
                k += 1 # good job! go to next iteration
            else:
                sigma *= 1.2 # increase by 20% and try again
                print('--> {:.2f} % pixels to mask...'.format(frac_masked))
                print('--> Trying again with sigma = {:.1f}...'.format(sigma))
                continue
            
        if debug:  
            n = mask.size
            print('--> {:.2f} % of pixels masked <--'.format(frac_masked))
                
        if ax != None: self.imshow(ax=ax)
        return self
#    
    def mask_sat_lines(self, sat=0.20, ax=None, debug=False):
        
        # nans = np.isnan(self.wlt)
        master = np.median(self.flux, axis=0)        
        mask = master < sat
        self.wlt[mask] = np.nan
        self.flux[:,mask] = np.nan
        if not self.flux_err is None:
            self.flux_err[:,mask] = np.nan
        
        if debug: print('Masked fraction = {:.2f} %'.format(self.nan_frac*100))
        
        if ax != None: self.imshow(ax=ax)
        return self
        
   
    

    
    
    def high_pass_gaussian(self, window=15, mode='subtract', dRV=0., ax=None):
        '''Apply a High-Pass Gaussian filter by subtracting a Low-Pass filter from the Data
        Pass the window in units of pixels. Blur only along the wavelength dimension (axis=1)'''
        from scipy import ndimage

        nans = np.isnan(self.wlt)
        if dRV > 0.:
            pixscale = const.c.to('km/s').value * np.nanmean(np.diff(self.wlt)) / np.nanmedian(self.wlt)
            window = 2 * dRV * pixscale
        
        lowpass = ndimage.gaussian_filter(self.flux[:,~nans], [0, window])
        if mode=='divide':
            self.flux[:,~nans] /= lowpass
            if not self.flux_err is None:
                self.flux_err[:,~nans] /= lowpass
        elif mode=='subtract':
            self.flux[:,~nans] -= lowpass
        
        if ax != None: self.imshow(ax=ax)
        return self
    
    def mask_eclipse(self, planet_in, return_mask=False, 
                     invert_mask=False, debug=False):
        '''given the planet PHASE and duration of eclipse `t_14` in days
        return the datacube with the frames masked'''
        dc = self.copy()
        planet = deepcopy(planet_in)
        shape_in = self.shape
        phase = planet.phase
        phase_14 = 0.5 * ((planet.T_14) % planet.P) / planet.P
        
        mask = np.abs(phase - 0.50) < phase_14 # frames IN-eclipse
        if invert_mask:
            mask = ~mask
            
        if return_mask:
            return mask
        
        else:
            if len(dc.shape) < 3:
                dc.flux = self.flux[~mask,:]
                if not dc.flux_err is None:
                   dc.flux_err = self.flux_err[~mask,:]
            else:
                dc.flux = dc.flux[:,~mask,:]
                if not dc.flux_err is None:
                    dc.flux_err = dc.flux_err[:,~mask,:]
          
            if debug:
                print('Original self.shape = {:}'.format(shape_in))
                print('After ECLIPSE masking self.shape = {:}'.format(dc.shape))
                            
            dc.flux = dc.flux
            dc.airmass = dc.airmass[~mask]
            if not dc.flux_err is None:
                dc.flux_err = dc.flux_err

            return dc
    
    def split_orders(self, debug=True):
        '''last update: ago 21 2022
        given a datacube with:
            - wlt.shape (nOrders, nPix)
            - flux.shape (nOrders, nFrames, nPix)
        return 
           - wlt.shape (2 x nOrders, nPix)
           - flux.shape (2 x nOrders, nFrames, nPix) '''
        
        dc = self.copy()
        x = int(dc.nPix/2)
    
        dc.wlt = np.concatenate([dc.wlt[:,x:], dc.wlt[:,:x]])
        sort = np.argsort(np.nanmedian(dc.wlt, axis=1))

        dc.wlt = dc.wlt[sort,:]
        
        dc.flux = np.concatenate([dc.flux[:,:,x:], dc.flux[:,:,:x]])[sort,:,:]
        if not dc.flux_err is None:
            dc.flux_err = np.concatenate([dc.flux_err[:,:,x:], dc.flux_err[:,:,:x]])[sort,:,:]
        
        if debug:
            print(self.shape)
            print(dc.shape)
        return dc
    
    def common_wave_grid(self, wavesol):
        dco = self.copy()
        nans = np.isnan(wavesol)
        
        for i in range(dco.nObs):
            cs = splrep(dco.wlt[i,~nans], dco.flux[i,~nans])
            self.flux[i,~nans] = splev(wavesol[~nans], cs)
            self.flux[i, nans] = np.nan
    #         dco.wlt[i] = dco.wlt[i,]*beta[i]
        self.wlt = wavesol
        return self
    
    
    def align(self, ax=None):
        from .align import Align
        dco = self.copy()
        self = Align(dco).apply_shifts(ax=ax).dco
        return self

    def shift(self, RV):
        '''copy of `to_stellar_frame` for any RV'''
        if isinstance(RV, np.floating):
            RV *= np.ones(self.nObs)
        self.sort_wave()
        nans = np.isnan(self.wlt)
        beta = 1.0 - (RV*u.km/u.s/const.c).decompose().value
        for f in range(self.nObs):
            cs = splrep(self.wlt[~nans], self.flux[f,~nans])
            self.flux[f,~nans] = splev(self.wlt[~nans]*beta[f], cs)

        return self
    
    # def to_stellar_frame(self, BERV):
    #     '''given a single-order datacube and a BERV vector in km/s
    #     spline interpolate on the shifted grid to go from telluric to stellar frame
    #     NOTE: must check the correct sign of BERV
    #     can also pass BERV --> BERV + Vsys'''
    #     self.sort_wave()
    #     nans = np.isnan(self.wlt)
    #     beta = 1.0 - (BERV*u.km/u.s/const.c).decompose().value
    #     for f in range(self.nObs):
    #         cs = splrep(self.wlt[~nans], self.flux[f,~nans])
    #         self.flux[f,~nans] = splev(self.wlt[~nans]*beta[f], cs)

    #     return self

        
    def sysrem(self, n=6, mode='subtract', debug=False, ax=None, outdir=None):
        '''new sysrem implementation (august 25th 2022)'''
        from .sysrem import SysRem
#        dco = self.copy()
        sys = SysRem(self).run(n, mode, debug, outdir)
        nans =  np.isnan(self.wlt)
        
        if mode == 'divide':
            self.flux[:, ~nans] /= (1. + sys.sysrem_model)
            self.sysrem_model = sys.sysrem_model
        elif mode == 'subtract':
            self.flux[:, ~nans] = sys.r_ij
       
        if ax != None: self.imshow(ax=ax)
        return self 
    
    def reduce_orders(self, function, orders, num_cpus=4):
        from p_tqdm import p_map
#        dcr = self.copy()
        
        def run(o):
            dco = function(self.order(o))
#            print(dco.shape)
            return dco
        
        
        output = p_map(run, orders, num_cpus=num_cpus)
        
        self.wlt = np.hstack([output[k].wlt for k in range(orders.size)])
        self.flux = np.hstack([output[k].flux for k in range(orders.size)])

        return self
    
    def sort_wave(self):
        sort = np.argsort(self.wlt)
        self.wlt = self.wlt[sort]
        self.flux = self.flux[:, sort]
        return self
    
    def mask_frames(self, mask, debug=False):
        '''given a datacube with (nOrders, nFrames, nPix) return
        datacube with (nOrders, nFrames - nMaskedFrames, nPix)'''
        shape_in = self.shape
        if len(self.wlt.shape) > 2:
            self.wlt = self.wlt[:,~mask,:]
        
        self.flux = self.flux[:,~mask,:]
        if not self.flux_err is None:
            self.flux_err = self.flux_err[:,~mask,:]
            
        # Update planet header with the MASKED vectors
        for key in ['MJD','BERV','airmass']:
            # self.header[item] = self.header[item][~mask]
            setattr(self, key, getattr(self, key)[~mask])
            
        if debug:
            print('Input {:}\nOutput {:}'.format(shape_in, self.shape))
        return self
    
    
    def update(self, dco, order):
        '''update data for a given order with the reduced order data 
        and the order number'''
        self.wlt[order,:] = dco.wlt
        self.flux[order,:,:] = dco.flux
        if not dco.flux_err is None:
            if self.flux_err is None:
                self.flux_err = np.ones_like(self.flux)
            self.flux_err[order,:,:] = dco.flux_err
        return self
    
    def merge_orders(self):
        '''merge data from different orders before computing CCF'''
        dc = self.copy()
        keys = ['wlt','flux']
        if not dc.flux_err is None: keys.append('flux_err')
        for key in keys:
            setattr(dc, key, np.hstack(getattr(self, key)))
        return dc
    
    def interpolate_to_planet(self, i):
        '''help function to parallelise `to_planet_frame` '''
        c = 2.998e5
        beta = 1 + (self.planet.RV[i]/c)        
        flux_i = interp1d(self.wlt, self.flux[i,], bounds_error=False, fill_value=0.)(self.wlt*beta)
        
        return flux_i
    
    def to_planet_frame(self, planet, ax=None, num_cpus=6):
        dco = self.copy()
        dco.planet = planet
        with mp.Pool(num_cpus) as p:
            flux_i = p.map(dco.interpolate_to_planet, np.arange(self.nObs))
        
        dco.flux = np.array(flux_i)
        
      
        if ax != None: dco.imshow(ax=ax)
        dco.frame = 'planet'
        return dco
    
    def crop(self, wave, eps=0.1):
        span = wave.max() - wave.min()
        mask = self.wlt < (wave.min()-(eps*span))
        mask += self.wlt > (wave.max()+(eps*span))
        self.wlt = self.wlt[~mask]
        self.flux = self.flux[:,~mask]
        return self
    
    def resample(self, wave):
        cs = splrep(self.wlt, self.flux)
        self.flux = splev(wave, cs)
        return self
#  
#            
if __name__ == '__main__':
    print('Main...')