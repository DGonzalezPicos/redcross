__all__ = ['Datacube']

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from copy import deepcopy
import astropy.units as u
import astropy.constants as const
from astropy.convolution import Gaussian1DKernel, convolve_fft
#from functions import fit_gaussian_lines

class Datacube:
    '''Main object to store, plot and work with data'''
    def __init__(self, flux=None, wlt=None, flux_err=None, night=None,**header):
        self.flux = flux
        self.wlt = wlt
        self.flux_err = flux_err
        self.night = night
        for key in header:
            setattr(self, key, header[key])
            
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

# =============================================================================
#                       UTILITY FUNCTIONS   
# =============================================================================
    def get_header(self):
        keys_to_extract = ['airmass','MJD','BERV','RA_DEG','DEC_DEG','DATE']    
        self.header = {key: self.__dict__[key] for key in keys_to_extract} 
        return self
    
    def plot(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        
        ax.plot(self.wlt, np.median(self.flux, axis=0), **kwargs)
#            ax.set(xlabel='Wavelength ({:})'.format(self.wlt_unit), ylabel='Flux')
        
        return ax
        

    def imshow(self, fig=None, ax=None, stretch=3.,**kwargs):
        
        # nans = np.isnan(self.wlt)
        if stretch > 0.:
            vmin = np.nanmean(self.flux)-np.nanstd(self.flux) * stretch
            vmax = np.nanmean(self.flux)+np.nanstd(self.flux) * stretch
        else:
            vmin = np.nanmin(self.flux)
            vmax = np.nanmax(self.flux)
            
        current_cmap = plt.cm.get_cmap().copy()
        current_cmap.set_bad(color='white')

        ext = [np.nanmin(self.wlt), np.nanmax(self.wlt),0, self.nObs-1]
        ax = ax or plt.gca()
        obj = ax.imshow(self.flux,origin='lower',aspect='auto',
                        extent=ext,vmin=vmin,vmax=vmax, **kwargs)
        if not fig is None: fig.colorbar(obj, ax=ax, pad=0.05)

        return ax
    
    # FAST version
    def estimate_noise(self):
        '''assign a noise value to each data point by taking the mean of the stdev
        in time and pixel dimensions'''
        xStDev = np.std(self.flux, axis=0)
        yStDev = np.std(self.flux, axis=1)
        self.flux_err = (0.5*(yStDev[:,np.newaxis] + xStDev[np.newaxis,:]))
        # print('FLUX_ERR shape = ', self.flux_err.shape)
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
        return self
    
    
    def airtovac(self,wlA):
        #Convert wavelengths (AA) in air to wavelengths (AA) in vaccuum (empirical).
        s = 1e4 / wlA
        n = 1 + (0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) +
        0.0001599740894897 / (38.92568793293 - s**2))
        return(wlA*n)
    
    def inject_signal(self, planet, template_in, factor=1.):
        template = deepcopy(template_in) # important
        
        beta = 1 - (planet.RV/const.c.to('km/s')).value
        wl_shift = np.outer(beta, self.wlt) # (nObs, nPix) matrix
        template.flux = (template.flux - 1.)*factor # DARIO MARCH 30TH 2022
        template.flux += 1.

        ##
        mask = self.mask_eclipse(planet, return_mask=True)
        # mask is True for in-eclipse points
        
        ##
        cs = splrep(template.wlt, template.flux) # coefficient spline
        for k in np.argwhere(mask==False):
            exp = int(k)
            
            inject_flux = splev(wl_shift[exp], cs)
            self.flux[exp] *= inject_flux

        return self
    
    
    def order(self, o):
        dco = self.copy()
    
        if len(dco.wlt.shape)>2:
            dco.wlt = dco.wlt[o,:,:]
        else:
            dco.wlt = dco.wlt[o,:]
        dco.flux = dco.flux[o,:,:]
        # dco.order = orders
        if not dco.flux_err is None:
            dco.flux_err = dco.flux_err[o,:,:]
        return dco
 
    
    def normalise(self):
       self.flux = (self.flux.T / np.median(self.flux, axis=1)).T
       if not self.flux_err is None:
           self.flux_err = (self.flux_err.T / np.median(self.flux, axis=1)).T
       return self
        
        
        
    def copy(self):
        return deepcopy(self)
    
    
    def sigma_clip(self, nSigma=5., axis=0, debug=False):
        '''For each pixel channel, replace outliers from each frame by the median column value
        outliers are points beyond nSigma from the column mean'''

        
        mean = np.nanmean(self.flux, axis=axis)
        std = np.nanstd(self.flux, axis=axis)
        outliers = 0
        # for x in range(self.nPix):
        if axis==0: # over wavelengths
            nans = np.isnan(self.wlt)
            flux = self.flux[:,~nans]
            # flux_err = self.flux[:,~nans]
            for i,x in enumerate(np.argwhere(nans==False)):   
                x = int(x)
                mask = (np.abs(flux[:,i]-mean[i]) / std[i]) > nSigma
                self.flux[mask,x] = np.nanmedian(flux)
                # self.flux[mask, x] = np.nan
                if not self.flux_err is None:
                    self.flux_err[mask,x] = np.nanmedian(self.flux_err[:,x])
                    # self.flux_err[mask,x] = np.nan
                outliers += mask[mask==True].size
            
        else: # over time
            flux = self.flux
            for i in range(self.shape[0]):
                mask = (np.abs(flux[i,:]-mean[i]) / std[i]) > nSigma
                self.flux[i,mask] = np.nanmedian(flux)
                if not self.flux_err is None:
                    self.flux_err[i,mask] = np.nanmedian(self.flux_err[i,:])
                outliers += mask[mask==True].size
            
            
        if debug:
            print('outliers = {:.2e} %'.format(outliers/self.flux.size))
        return self
    
    def remove_continuum(self, mode='polyfit', deg=3.):
        '''for each order, remove the continuum by dividing by the residuals 
        from the master subtraction'''
    
        if mode == 'polyfit':
            for f in range(self.nObs):
                if len(self.wlt.shape) > 1: # each frame with its wavelength grid
                    wave = self.wlt[f,:]
                else: # common wavelength grid
                    wave = self.wlt
                model = np.poly1d(np.polyfit(wave, self.flux[f,:], deg))
                continuum = model(wave)
                self.flux[f,:] /= continuum
                if not self.flux_err is None:
                    self.flux_err[f,:] /= continuum
        else:
            master = np.median(self.flux, axis=0) 
            g1d_kernel = Gaussian1DKernel(300)
            for frame in range(self.shape[0]):
                divide = convolve_fft((self.flux[frame,] / master), g1d_kernel, boundary='wrap')
                self.flux[frame,] /= divide
                
        return self
    
    def airmass_detrend(self, log_space=False):
        '''Fit a second order polynomial to each column and divide(subtract) the fit 
        in linear(log) space'''
        
        if log_space:
            for j in range(self.nPix):
                y = np.log(self.flux[:,j])
                fit = np.poly1d(np.polyfit(self.airmass,y,2))(self.airmass)
                self.flux[:,j] = np.exp(y - fit)
        else:
            for j in range(self.nPix):
                y = self.flux[:,j]
                fit = np.poly1d(np.polyfit(self.airmass,y,2))(self.airmass)
                self.flux[:,j] /= fit
                if not self.flux_err is None:
                    self.flux_err[:,j] /= fit
        return self
    
    def mask_cols(self, sigma=3., mode='flux', nan=False, debug=False):
        # nans = np.isnan(self.wlt)
        if mode == 'flux':
            y = np.nanstd(self.flux, axis=0)
        elif mode == 'flux_err':
            y = np.nanstd(self.flux_err, axis=0)
        mean, std = np.nanmean(y), np.nanstd(y)
        mask = np.abs(y - mean)/std > sigma
        # mask += nans
        if nan:
            self.wlt[mask] = np.nan
            self.flux[:,mask] = np.nan
            if not self.flux_err is None:
                self.flux_err[:,mask] = np.nan 
        else:
            self.wlt = self.wlt[~mask]
            self.flux = self.flux[:,~mask]
            if not self.flux_err is None:
                self.flux_err = self.flux_err[:,~mask]
    
        if debug:  
            n = mask.size
            print('--> {:.2f} % of pixels masked'.format(mask[mask==True].size*100/n))
        return self
        
   
    def snr_cutoff(self, cutoff=40., debug=False):
        '''Discard columns where the SNR is below the CUTOFF
        Apply before CCF'''
        snr = np.nanmean(self.flux, axis=0) / np.nanstd(self.flux, axis=0)
        if debug: print('Mean SNR = {:.2f} +- {:.2f}'.format(np.nanmean(snr), np.std(snr)))
        mask = snr < cutoff
        self.wlt = self.wlt[~mask]
        self.flux = self.flux[:,~mask]
        if not self.flux_err is None:
            self.flux_err = self.flux_err[:,~mask]
            
        if debug:
            print('Input shape: {:}'.format(snr.shape))
            print('Output shape: {:}'.format(self.wlt.shape))
            print('--> {:} discarded pixels'.format(snr.size-self.wlt.size))
        return self
    
    def get_mask(self, snr_min=40, debug=False):
        'Return a MASK with columns where the mean SNR is below `snr_min` '
        snr = np.nanmedian(self.flux, axis=0) / np.nanstd(self.flux, axis=0)
        if debug: print('Mean SNR = {:.2f} +- {:.2f}'.format(np.nanmean(snr), np.std(snr)))
        mask = snr < snr_min
        
        return mask
    
    def apply_mask(self, mask, mode='mask', debug=False):
        
        if mode=='mask':
            self.wlt = self.wlt[~mask]
            self.flux = self.flux[:,~mask]
            if not self.flux_err is None:
                self.flux_err = self.flux_err[:,~mask]
        elif mode=='replace':
            self.flux[:,mask] = np.nanmedian(self.flux[:,~mask])
            if not self.flux_err is None:
                self.flux_err[:,mask] = np.nanmedian(self.flux_err[:,~mask])
            
        if debug:
            n = mask.size
            print('--> {:.2f} % of pixels masked'.format(mask[mask==True].size*100/n))
            print('Input shape = ', mask.shape)
            print('Output shape = ', self.wlt.shape)
        return self
        
    
    def lowpass_filter(self, window=15):
        'input must be a single order'
        from scipy.signal import savgol_filter
        for i in range(self.nObs):
            self.flux[i,] = savgol_filter(self.flux[i,], window, 3)
        return self
            
            
    def get_master(self):
        from functions import gaussian_smooth
        self.master = gaussian_smooth(np.median(self.flux, axis=1), xstd=300.) # average over frames
        return self
    
    @property
    def SNR(self):
        if self.flux_err is None:
            med = np.nanmedian(self.flux)
            return med / np.nanstd(self.flux)
        else:
            print('Using flux_err')
            # return np.nanmean(np.nanmean(self.flux, axis=1)/ np.nanmean(self.flux_err, axis=1))
            # return np.nanmean(self.flux / self.flux_err)
            # return np.nanmedian(self.flux) / np.nanstd(self.flux)
            std = np.nanstd(self.flux, axis=0)
            return 1/np.nanmean(np.nanstd(self.flux, axis=0))
    
    @property
    def Q(self):
        # return 1/np.mean(np.std(self.flux - np.nanmedian(self.flux), axis=0))
        # return np.mean(np.mean(self.flux, axis=1)/np.std(self.flux, axis=1))
        # return np.sqrt(np.mean(np.var(self.flux, axis=0)))
        return 1/np.nanstd(self.flux)

    
    
    def high_pass_gaussian(self, window=15, mode='subtract'):
        '''Apply a High-Pass Gaussian filter by subtracting a Low-Pass filter from the Data
        Pass the window in units of pixels. Blur only along the wavelength dimension (axis=1)'''
        from scipy import ndimage

        nans = np.isnan(self.wlt)
        
        lowpass = ndimage.gaussian_filter(self.flux[:,~nans], [0, window])
        if mode=='divide':
            self.flux[:,~nans] /= lowpass
            if not self.flux_err is None:
                self.flux_err[:,~nans] /= lowpass
        elif mode=='subtract':
            self.flux[:,~nans] -= lowpass
        return self
    
    def mask_eclipse(self, planet_in, t_14=0.18, return_mask=False, debug=False):
        '''given the planet PHASE and duration of eclipse `t_14` in days
        return the datacube with the frames masked'''
        dc = self.copy()
        planet = deepcopy(planet_in)
        shape_in = self.shape
        phase = planet.phase
        phase_14 = (t_14 % planet.P) / planet.P

        mask = (phase > (0.50 - (0.50*phase_14)))*(phase < (0.50 + (0.50*phase_14)))
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
                            
            self.flux = dc.flux
            self.airmass = dc.airmass[~mask]
            if not self.flux_err is None:
                self.flux_err = dc.flux_err

            return self
        
        
    def split_detector(self):
        '''For Giano data, split WAVELENGTH dimension in two due to detector's' response
        Return a datacube with an additional dimension = (2xnOrders, nObs, nPix)'''
        
        dc = self.copy()
        x = int(dc.nPix/2)
        # if len(self.wlt.shape) > 2:
        #     axis=1
        #     self.wlt = np.concatenate([dc.wlt[:,:,x:], dc.wlt[:,:,:x]], axis=0)
        #     sort = np.argsort(np.median(self.wlt, axis=2)) # SORT by wavelength
        #     self.wlt = self.wlt[sort,:]
        if len(dc.shape)>2:
            dc.wlt = np.concatenate([dc.wlt[:,x:], dc.wlt[:,:x]], axis=0)
            sort = np.argsort(np.median(dc.wlt, axis=1)) # SORT by wavelength
            dc.wlt = dc.wlt[sort,:]
                
            dc.flux = np.concatenate([dc.flux[:,:,x:], dc.flux[:,:,:x]], axis=0)[sort,:,:]
            if not dc.flux_err is None:
                dc.flux_err = np.concatenate([dc.flux_err[:,:,x:], dc.flux_err[:,:,:x]], axis=0)[sort,:,:]
        else:
            dc.wlt = np.stack((dc.wlt[x:], dc.wlt[:x]))
            sort = np.argsort(np.median(dc.wlt, axis=1))
            dc.wlt = dc.wlt[sort,:]
            
            dc.flux = np.stack((dc.flux[:,x:], dc.flux[:,:x]))[sort,:,:]
            if not dc.flux_err is None:
                dc.flux_err = np.stack((dc.flux_err[:,x:], dc.flux_err[:,:x]))[sort,:,:]
        return dc
    
    def align(self):
        '''Compute shift between each frame and the master spectrum wavelength grid
        Apply the shift to each frame and spline interpolate the flux onto the corrected grid
        Define the common wavelength grid as the master wavelength'''
        from scipy.interpolate import splrep, splev
        dco = self.copy()
        
        master = Datacube(wlt=np.median(dco.wlt, axis=0), flux=np.median(dco.flux, axis=0))
        cenwave = np.median(master.wlt)
        delta_wlt = np.median(master.wlt - dco.wlt, axis=1)
        beta = 1 + delta_wlt/cenwave
        for i in range(dco.nObs):
            cs = splrep(dco.wlt[i,], dco.flux[i,])
            dco.flux[i,] = splev(dco.wlt[i,]*beta[i], cs)
    #         dco.wlt[i] = dco.wlt[i,]*beta[i]
        dco.wlt = master.wlt
        return dco
    
    def common_wave_grid(self):
        dco = self.copy()
        new_grid = np.linspace(dco.wlt.min(), dco.wlt.max(), dco.nPix)
        for i in range(dco.nObs):
            cs = splrep(dco.wlt[i,], dco.flux[i,])
            self.flux[i,] = splev(new_grid, cs)
    #         dco.wlt[i] = dco.wlt[i,]*beta[i]
        self.wlt = new_grid
        return self
#    
#    def align_frames(self, centroids):
#        '''given a DCO and the pixel positions of telluric lines
#        compute the relative drift with the master spectrum'''
#        
#        rdrift = np.zeros(self.nObs)
#        master = Template(wlt=np.arange(0,self.nPix), flux=np.median(self.flux, axis=0))
#        
#        cent_master = fit_gaussian_lines(master.wlt, master.flux, centroids)
#        
#        for f in range(self.nObs):
#            flux = self.flux[f]
#            cent_data = fit_gaussian_lines(master.wlt, flux, centroids)
#            rdrift[f] = np.median(cent_data - cent_master)
#            
#            # 1 pixel = 2.7 km/s for GIANO (see Brogi+2018)
#            beta = 1+2.7*rdrift[f]*u.km/u.s /const.c
#            cs = splrep(self.wlt[f,:], self.flux[f,:])
#            self.wlt[f,:] = self.wlt[f,:] * beta
#            self.flux[f,:] = splev(self.wlt[f,:], cs)
#            
#        return self
    
    def apply_wave_solution(self, wsol):
        '''for each order, pass a CALIBRATED wavelength solution and
        spline interpolate each spectrum (each frame) onto the new grid'''
        for f in range(self.nObs):
            cs = splrep(self.wlt[f,:], self.flux[f,:])
            self.flux[f,:] = splev(wsol, cs)
            
        self.wlt = wsol
        return self
    
    def to_stellar_frame(self, BERV):
        '''given a single-order datacube and a BERV vector in km/s
        spline interpolate on the shifted grid to go from telluric to stellar frame
        NOTE: must check the correct sign of BERV
        can also pass BERV --> BERV + Vsys'''
        nans = np.isnan(self.wlt)
        beta = 1.0 + (BERV*u.km/u.s/const.c).decompose().value
        for f in range(self.nObs):
            cs = splrep(self.wlt[~nans], self.flux[f,~nans])
            self.flux[f,~nans] = splev(self.wlt[~nans]*beta[f], cs)
        return self
    
    def sysrem(self, n, mode='subtract', debug=False):
        from .sysrem import SysRemRoutine
        dco = self.copy()
        self.flux = SysRemRoutine(dco, n, mode, debug)
        return self 
    
    
   
            
            

  
            
if __name__ == '__main__':
    print('Main...')