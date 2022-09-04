import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d, splrep, splev
from copy import deepcopy
import astropy.units as u
import astropy.constants as const
from .datacube import Datacube

class CCF(Datacube):
    mode = 'ccf'
    def __init__(self, rv=None, template=None, flux=None, **kwargs):
        self.rv = rv
        self.template = template
        self.flux = flux
        
    def normalise(self):
        self.flux = self.flux / np.median(self.flux, axis=0)
        return self  
    
    @property
    def wlt(self):
        return self.rv

   
    def run(self, dc, debug=False, weighted=False):
        '''The data baseline must be around 1.0 while the template baseline must be 
        around zero.'''
        
        start=time.time()
        nans = np.isnan(dc.wlt)
        self.flux = np.zeros((dc.nObs,len(self.rv)))    
        self.template_interp1d(dc.wlt[~nans])
        self.template.flux_interp1d -= np.nanmean(self.template.flux_interp1d) # Additional step 18-02-2022
        
        # divide = dc.nObs * np.nanstd(dc.flux) * np.nanstd(self.template.flux_interp1d) # NEW 23-04-2022
        std_flux = np.nanstd(dc.flux[:,~nans], axis=1)
        std_temp = np.nanstd(self.template.flux_interp1d, axis=1)
        # divide = std_flux*std_temp
#        divide = np.outer(std_flux, std_temp)
        divide = np.sum(self.template.flux)
        # print('divide.shape = ', divide.shape)
        
        
        if weighted:
            # nans = np.isnan(dc.flux_err)
            # self.flux = np.dot(((dc.flux[:,~nans]-np.nanmean(dc.flux[:,~nans]))/ np.power(dc.flux_err[:,~nans],2)),self.template.flux_interp1d.T) # TESTING
            
            data = dc.flux[:,~nans]-np.nanmean(dc.flux[:,~nans])
#            noise2 = np.power(np.nanstd(dc.flux[:,~nans], axis=0),2)
            noise2 = np.power(np.std(dc.flux[:,~nans], axis=0),2)
#            noise2 = np.power(dc.flux_err[:,~nans],2)
#            noise2 = np.outer(np.std(data, axis=1), np.std(data, axis=0))
#            noise2 = 1.+np.power(np.nanstd(dc.flux[:,~nans], axis=0),2)
#            noise2 = 1.+ np.power(np.nanmedian(dc.flux_err[:,~nans], axis=0), 2)
            self.flux = np.dot(data/noise2, self.template.flux_interp1d.T) # TESTING


        else:
            self.flux = np.dot(dc.flux[:,~nans]-np.nanmean(dc.flux[:,~nans]), self.template.flux_interp1d.T) / divide
            

        # self.flux /= self.flux.max() ######## TESTING ######    
        
        if debug:
            print('Max CCF value = {:.2f}'.format(self.flux.max()))
            print('Baseline CCF = {:.2f}'.format(np.median(self.flux)))
            print('CCF elapsed time: {:.2f} s'.format(time.time()-start))
        return self
    
    def run_slow(self, dc):
        print('Compute CCF...')
        start=time.time()
        self.template.flux -= np.mean(self.template.flux)
        dc.flux -= np.mean(dc.flux)
        self.flux = np.zeros((dc.nObs,len(self.rv)))
        
        beta = 1+self.rv*u.km/u.s /const.c
        for i in range(len(self.rv)):

            fxt_i = interp1d(self.template.wlt*beta[i],self.template.flux)(dc.wlt)
            self.flux[:,i] = np.dot(dc.flux,fxt_i)/np.sum(fxt_i)
            
        self.flux /= np.mean(self.flux,axis=0)
        print('CCF elapsed time: {:.2f} s'.format(time.time()-start))
        return self
    

    
    def template_interp1d(self, new_wlt):
        self.template.flux_interp1d = np.zeros((len(self.rv), len(new_wlt)))
        for i in range(len(self.rv)):
            beta = 1+self.rv[i]*u.km/u.s /const.c
            cs = splrep(self.template.wlt*beta,self.template.flux)
            self.template.flux_interp1d[i,:] = splev(new_wlt, cs)
        return self
    
    def mask_eclipse(self, planet, debug=False):
        '''given the planet PHASE and duration of eclipse `t_14` in days
        return the datacube with the frames masked'''
        shape_in = self.shape
        phase = planet.phase
        phase_14 = (planet.T_14 % planet.P) / planet.P
    
        mask = np.abs(phase - 0.50) < (phase_14/2.) # frames IN-eclipse
        print(phase_14)
#        mask = (phase > (0.50 - (0.50*phase_14)))*(phase < (0.50 + (0.50*phase_14)))
        self.flux = self.flux[~mask,:]
      
        if debug:
            print('Original self.shape = {:}'.format(shape_in))
            print('After ECLIPSE masking self.shape = {:}'.format(self.shape))
                        
        return self
    
    def to_planet_frame(self, planet, ax=None):
        ccf = self.copy()
        for i in range(self.nObs):
            cs = splrep(ccf.rv, ccf.flux[i,])
            ccf.flux[i,] = splev(ccf.rv+planet.RV.value[i], cs)
        mask = (np.abs(ccf.rv)<np.percentile(np.abs(ccf.rv), 50))
        ccf.rv = ccf.rv[mask]
        ccf.flux = ccf.flux[:,mask]
        if ax != None: ccf.imshow(ax=ax)
        return ccf
    

    
                
                
class KpV:
    def __init__(self, ccf=None, planet=None, kp=None, vrest=None, bkg=20):
        self.ccf = ccf.copy()
        self.planet = deepcopy(planet)
        self.kpVec = self.planet.Kp.value + np.arange(-kp[0], kp[0], kp[1])
        self.vrestVec = np.arange(-vrest[0], vrest[0]+vrest[1], vrest[1])
        self.bkg = bkg
        
    def run(self, snr=True, ignore_eclipse=False, ax=None):
        '''Generate a Kp-Vsys map
        if snr = True, the returned values are SNR (background sub and normalised)
        else = map values'''
    
             
        if ignore_eclipse:   
            self.ccf.mask_eclipse(self.planet)
            self.planet.mask_eclipse(debug=True) ## TESTING
            
#        if self.planet.RV.size != self.ccf.shape[0]:
#            print('Interpolate planet...')
#            newX = np.arange(0, self.ccf.shape[0],1)
#            self.planet = self.planet.interpolate(newX)
        
            
        snr_map = np.zeros((len(self.kpVec), len(self.vrestVec)))
        rvel = ((self.planet.v_sys*u.km/u.s)-self.planet.BERV*u.km/u.s).value 
        
        
        for ikp in range(len(self.kpVec)):
            rv_planet = rvel + (self.kpVec[ikp]*np.sin(2*np.pi*self.planet.phase))
            for iObs in range(self.planet.RV.size):
                outRV = self.vrestVec + rv_planet[iObs]
                snr_map[ikp,] += interp1d(self.ccf.rv, self.ccf.flux[iObs,])(outRV)
       
        if snr:
            noise_map = np.std(snr_map[:,np.abs(self.vrestVec)>self.bkg])
            bkg_map = np.median(snr_map[:,np.abs(self.vrestVec)>self.bkg]) # subtract the background level
            snr_map -= bkg_map        
            self.snr = snr_map / noise_map
        else:
            self.snr = snr_map # NOT ACTUAL SIGNAL-TO-NOISE ratio (useful for computing weights)
        if ax != None: self.imshow(ax=ax)
        return self
    def snr_max(self, display=False):
        # Locate the peak
        bestSNR = self.snr.max()
        ipeak = np.where(self.snr == bestSNR)
        bestVr = float(self.vrestVec[ipeak[1]])
        bestKp = float(self.kpVec[ipeak[0]])
        
        if display:
            print('Peak position in Vrest = {:3.1f} km/s'.format(bestVr))
            print('Peak position in Kp = {:6.1f} km/s'.format(bestKp))
            print('Max SNR = {:3.1f}'.format(bestSNR))
        return(bestVr, bestKp, bestSNR)
    
    def plot(self, fig=None, ax=None, snr_max=True, v_range=None):
        lims = [self.vrestVec[0],self.vrestVec[-1],self.kpVec[0],self.kpVec[-1]]
        if v_range is not None:
            vmin = v_range[0]
            vmax = v_range[1]
        else:
            vmin = self.snr.min()
            vmax = self.snr.max()
            
            
        obj = ax.imshow(self.snr,origin='lower',extent=lims,aspect='auto', cmap='inferno',vmin=vmin,vmax=vmax)
        if not fig is None: fig.colorbar(obj, ax=ax, pad=0.05)
        ax.set_xlabel('Rest-frame velocity (km/s)')
        ax.set_ylabel('Kp (km/s)')

        
        if snr_max:
            peak = self.snr_max()
        else:
            peak = [2.0, self.planet.Kp.value] # expected vrest, Kp of Planet
            
        
        indv = np.abs(self.vrestVec - peak[0]).argmin()
        indh = np.abs(self.kpVec - peak[1]).argmin()
    
        row = self.kpVec[indh]
        col = self.vrestVec[indv]
        line_args = {'ls':':', 'c':'white','alpha':0.35,'lw':'3.'}
        ax.axhline(y=row, **line_args)
        ax.axvline(x=col, **line_args)
        ax.scatter(col, row, marker='*', s=3., c='red',label='SNR = {:.2f}'.format(self.snr[indh,indv]))
    
        return obj
    
    
    def fancy_figure(self, figsize=(6,6), peak=None, v_range=None, outname=None, title=None):
        '''Plot Kp-Vsys map with horizontal and vertical slices 
        snr_max=True prints the SNR for the maximum value'''
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(6,6)
        gs.update(wspace=0.00, hspace=0.0)
        ax1 = fig.add_subplot(gs[1:5,:5])
        ax2 = fig.add_subplot(gs[:1,:5])
        ax3 = fig.add_subplot(gs[1:5,5])
        # ax2 = fig.add_subplot(gs[0,1])
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        
        if not v_range is None:
            vmin = v_range[0]
            vmax = v_range[1]
            # fix y-axis (x-axis) of secondary axes
            ax2.set(ylim=(v_range[0], v_range[1]))
            ax3.set(xlim=(v_range[0], v_range[1]))
        else:
            vmin = self.snr.min()
            vmax = self.snr.max()
            
        lims = [self.vrestVec[0],self.vrestVec[-1],self.kpVec[0],self.kpVec[-1]]

        obj = ax1.imshow(self.snr,origin='lower',extent=lims,aspect='auto', 
                         cmap='inferno', vmin=vmin, vmax=vmax)
    
        # figure settings
        ax1.set(ylabel='Kp (km/s)', xlabel='Vrest (km/s)')
        fig.colorbar(obj, ax=ax3, pad=0.05)
        if peak is None:
            peak = self.snr_max()
            
        
        indv = np.abs(self.vrestVec - peak[0]).argmin()
        self.indh = np.abs(self.kpVec - peak[1]).argmin()
        self.peak_snr = self.snr[self.indh,indv]
    
        row = self.kpVec[self.indh]
        col = self.vrestVec[indv]
        print('Horizontal slice at Kp = {:.1f} km/s'.format(row))
        print('Vertical slice at Vrest = {:.1f} km/s'.format(col))
        ax2.plot(self.vrestVec, self.snr[self.indh,:], 'gray')
        ax3.plot(self.snr[:,indv], self.kpVec,'gray')
        
        
    
        line_args = {'ls':':', 'c':'white','alpha':0.35,'lw':'3.'}
        ax1.axhline(y=row, **line_args)
        ax1.axvline(x=col, **line_args)
        ax1.scatter(col, row, marker='*', c='red',label='SNR = {:.2f}'.format(self.peak_snr))
        ax1.legend()
    
        ax1.set(xlabel='Vrest (km/s)', ylabel='Kp (km/s)')

        if title != None:
            fig.suptitle(title, x=0.45, y=0.915, fontsize=14)
    
        if outname != None:
            fig.savefig(outname, dpi=200, bbox_inches='tight', facecolor='white')
        return self
    
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)


    
        
class Template:
    def __init__(self, wlt=None, flux=None, filepath=None):
        if not filepath is None:
            self.filepath = filepath
            self.wlt, self.flux = np.load(self.filepath)
        else:
            self.wlt = wlt
            self.flux = flux
            
    def plot(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        ax.plot(self.wlt, self.flux, **kwargs)
        return ax
        
    def check_data(self):
        cenwave = np.median(self.wlt)
        unit = 'Unknown'
        if (cenwave < 1000) & (cenwave>100):
            unit = 'nm'
            print('--> Transforming from nm to A...')
            self.wlt *= 10.
            print('New central wavelength = {:.2f} A'.format(np.median(self.wlt)))
            
        elif cenwave > 1000:
            print('Wavelength in A')
            unit = 'A'
        else:
            print('Wavelength in Unknown units!!')
        return self
    
    def get_2DTemplate(self, planet, dc):
        beta = 1 - (planet.RV/const.c.to('km/s')).value
        wl_shift = np.outer(beta, dc.wlt) # (nObs, nPix) matrix
        transit_depth = np.zeros_like(dc.flux)
        for exp in range(dc.nObs):
            transit_depth[exp,:] = splev(wl_shift[exp], splrep(self.wlt, self.flux))
            # transit_depth[exp,:] = interp1d(self.wlt, self.flux, bounds_error=True)(wl_shift[exp])
        return transit_depth
    
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


    def high_pass_gaussian(self, window):
        from scipy import ndimage
        lowpass = ndimage.gaussian_filter1d(self.flux, window)
        self.flux /= lowpass
        return self
    
    def copy(self):
        return deepcopy(self)
    
    
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
    
    def convolve_instrument(self, res=50e3):
        '''convolve to instrumental resolution with a Gaussian kernel'''
        from astropy.convolution import Gaussian1DKernel, convolve
        
        cenwave = np.median(self.wlt)
        # Create kernel
        g = Gaussian1DKernel(stddev=0.5*cenwave/res) # set sigma = FWHM / 2. = lambda / R
        self.flux = convolve(self.flux, g)
        return self
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    