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
        if not self.rv is None: self.dRV = np.mean(np.diff(self.rv)) # resolution
        self.template = template
        self.flux = flux
        
    def normalise(self):
        self.flux = self.flux / np.median(self.flux, axis=0)
        return self  
    
    @property
    def wlt(self):
        return self.rv

   
    def run(self, dc, debug=False, weighted=False, ax=None):
        
        start=time.time()
        nans = np.isnan(dc.wlt)
        self.flux = np.zeros((dc.nObs,len(self.rv)))    
        # gTemp = self.template_interp1d(dc.wlt[~nans])
        gTemp = self.template.shift_2D(self.rv, dc.wlt).gflux[:,~nans] # TESTING
        gTemp -= np.nanmean(gTemp) 
        dc.estimate_noise() # testing Sept 19th 2022
        
        if weighted:
            data = dc.flux[:,~nans]-np.nanmean(dc.flux[:,~nans])
            # noise2 = np.power(np.std(dc.flux[:,~nans], axis=0),2)
            noise2 = np.var(dc.flux[:,~nans], axis=0)
            # noise2 = np.power(dc.flux_err[:,~nans], 2) # testing Sept 19th 2022
            self.flux = np.dot(data/noise2, gTemp.T) 


        else:
            divide = np.sum(self.template.flux)
            self.flux = np.dot(dc.flux[:,~nans]-np.nanmean(dc.flux[:,~nans]), gTemp.T) / divide
                    
        if debug:
            print('Max CCF value = {:.2f}'.format(self.flux.max()))
            print('Baseline CCF = {:.2f}'.format(np.median(self.flux)))
            print('CCF elapsed time: {:.2f} s'.format(time.time()-start))
            
        if ax != None: self.imshow(ax=ax)
        return self
    
#    def log_likelihood(self, dc, debug=False):
#        nans = np.isnan(dc.wlt)
#        N = dc.nObs
#        # prepare data
#        self.flux = np.zeros((N,len(self.rv)))   
#        self.log_L = np.zeros((N,len(self.rv)))  
#        # create 2D-template (shifted for all RVs)
#        self.template_interp1d(dc.wlt[~nans])
#        self.template.flux_interp1d -= np.nanmean(self.template.flux_interp1d) 
#    
#        # intermediate quantities
#          # sum over pixel channels
#        sf2 = np.sum(np.power(dc.flux,2), axis=1) / float(N)
#        sg2 = np.sum(np.power(self.template.flux_interp1d,2), axis=1) / float(N)
#        print('sf2 ', sf2[:,np.newaxis].shape)
#        print('sg2 ', sg2[np.newaxis,:].shape)
#        R = np.dot(dc.flux, self.template.flux_interp1d.T) / float(N)
#        print('R', R.shape)
#        # output
#        self.flux = R / np.sqrt(np.outer(sf2,sg2)) # CCF-map
#        self.log_L = -0.5*float(N) * np.log(sf2[:,np.newaxis] + sg2[np.newaxis,:] - (2*R))
#        return self
    
    def log_likelihood(self, dc):
        start=time.time()
        nans = np.isnan(dc.wlt)
        self.flux = np.zeros((dc.nObs,len(self.rv)))    
        self.template_interp1d(dc.wlt[~nans])
        self.template.flux_interp1d -= np.nanmean(self.template.flux_interp1d)
        Npix = dc.nPix
        
        self.flux = np.zeros((dc.nObs,len(self.rv)))   # CCF values
        self.log_L = np.zeros((dc.nObs,len(self.rv)))  # Log-L values
        
        for i in range(dc.nObs):
            sf2 = np.dot(dc.flux[i,:], dc.flux[i,:]) / float(Npix)
            sg2 = np.dot(self.template.flux_interp1d[i,:], self.template.flux_interp1d[i,:]) / float(Npix)
            R = dc.flux[i,:].dot(self.template.flux_interp1d[i,:].T)/ float(Npix)
            
            print('sf2 ', sf2.shape)
            print('sg2 ', sg2.shape)
            print('R', R.shape)
            self.log_L[i,:] += (-0.5*Npix * np.log(sf2+sg2-2.0*R)) #the equation of logL
            
        print('Elapsed time: {:.2f} s'.format(time.time()-start))
        return self
    
    def xcorr(self, f,g):
        nx = len(f)
#        I = np.ones(nx)
#        f -= np.dot(f,I)/nx
#        g -= np.dot(g,I)/nx
        f -= np.mean(f)
        g -= np.mean(g)
        R = np.dot(f,g)/nx
        varf = np.dot(f,f)/nx
        varg = np.dot(g,g)/nx

        CC = R / np.sqrt(varf*varg)
        log_L = -0.5*nx * np.log(varf + varg - (2*R))
        print(varf, varg, R)
        return CC, log_L
    
    def get_cc_grid(self, dco):
        nans = np.isnan(dco.wlt)
        self.flux, self.log_L = (np.zeros((dco.nObs,len(self.rv))) for _ in range(2))
        self.template.flux -= np.median(self.template.flux)
        
        coef_spline = splrep(self.template.wlt, self.template.flux, s=0.0)
        beta = self.rv / 2.998E5
        for irv in range(self.rv.size):
            # Shifting data wlen instead by swapping sign of RV
            wShift = dco.wlt[~nans] * np.sqrt( (1-beta[irv]) / (1+beta[irv]) )  # A (nPix) vector
            intMod = splev(wShift,coef_spline,der=0) # A (nPix) vector
            for iObs in range(dco.nObs):
                CC, log_L = self.xcorr(dco.flux[iObs,~nans], intMod)
                self.flux[iObs,irv] = CC
                self.log_L[iObs,irv] = log_L
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
        gTemp = np.zeros((len(self.rv), len(new_wlt)))
        for i in range(len(self.rv)):
            beta = 1+self.rv[i]*u.km/u.s /const.c
            cs = splrep(self.template.wlt*beta,self.template.flux)
            gTemp[i,:] = splev(new_wlt, cs)
        return gTemp
    
    def mask_eclipse(self, planet, debug=False):
        '''given the planet PHASE and duration of eclipse `t_14` in days
        return the datacube with the frames masked'''
        shape_in = self.shape
        phase = planet.phase
        phase_14 = (planet.T_14 % planet.P) / planet.P
    
        mask = np.abs(phase - 0.50) < (phase_14/2.) # frames IN-eclipse
#        print(phase_14)
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
    def __init__(self, ccf=None, planet=None, deltaRV=None, 
                 kp_radius=50., vrest_max=80., bkg=20):
        if not ccf is None:
            self.ccf = ccf.copy()
            self.planet = deepcopy(planet)
    #        self.kpVec = self.planet.Kp.value + np.arange(-kp[0], kp[0], kp[1])
    #        self.vrestVec = np.arange(-vrest[0], vrest[0]+vrest[1], vrest[1])
            self.dRV = deltaRV or ccf.dRV
    
            self.kpVec = self.planet.Kp.value + np.arange(-kp_radius, kp_radius, self.dRV)
            self.vrestVec = np.arange(-vrest_max, vrest_max, self.dRV)
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
    
    
    def xcorr(self, f,g):
        nx = len(f)
    #        I = np.ones(nx)
    #        f -= np.dot(f,I)/nx
    #        g -= np.dot(g,I)/nx
        f -= np.mean(f)
        g -= np.mean(g)
        R = np.dot(f,g)/nx
        varf = np.dot(f,f)/nx
        varg = np.dot(g,g)/nx
    
#        CC = R / np.sqrt(varf*varg)
        log_L = -0.5*nx * np.log(varf + varg - (2*R))
        print(varf, varg, R)
        return log_L
    
    
    
    def logL_map(self, dco, template):
        
        f = dco.flux - np.mean(dco.flux)
                
        nx = dco.nPix    
        cs = splrep(template.wlt, template.flux)
        self.log_L = np.zeros((self.kpVec.size, self.vrestVec.size, dco.nObs))  
        
        rvel = ((self.planet.v_sys*u.km/u.s)-self.planet.BERV*u.km/u.s).value
        for i,kp in enumerate(self.kpVec):
            rv_planet = rvel + (self.kpVec[i]*np.sin(2*np.pi*self.planet.phase))
            for frame in range(dco.nObs):
                f = dco.flux[frame,:]
                varf = np.dot(f,f)/nx
                for j,vsys in enumerate(self.vrestVec):
                    outRV = vsys + rv_planet[frame]
                    wshift = outRV / 2.998e5
                    g = splev(template.wlt*wshift, cs)
                    varg = np.dot(g,g)/nx
                    
                    R = np.dot(f,g)/nx
                    self.log_L[i,j,frame] += -0.5*nx * np.log(varf + varg - (2*R))
        
        
        
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
    
    def plot(self, fig=None, ax=None, peak=None, v_range=None, label=''):
        lims = [self.vrestVec[0],self.vrestVec[-1],self.kpVec[0],self.kpVec[-1]]
        if v_range is not None:
            vmin = v_range[0]
            vmax = v_range[1]
        else:
            vmin = self.snr.min()
            vmax = self.snr.max()
            
        ax = ax or plt.gca()
        obj = ax.imshow(self.snr,origin='lower',extent=lims,aspect='auto', 
                        cmap='inferno',vmin=vmin,vmax=vmax, label=label)
        if not fig is None: fig.colorbar(obj, ax=ax, pad=0.05)
        ax.set_xlabel('Rest-frame velocity (km/s)')
        ax.set_ylabel('Kp (km/s)')

        
        
        peak = peak or self.snr_max()
        
        indv = np.abs(self.vrestVec - peak[0]).argmin()
        indh = np.abs(self.kpVec - peak[1]).argmin()
    
        row = self.kpVec[indh]
        col = self.vrestVec[indv]
        line_args = {'ls':':', 'c':'white','alpha':0.35,'lw':'3.'}
        ax.axhline(y=row, **line_args)
        ax.axvline(x=col, **line_args)
        ax.scatter(col, row, marker='*', s=3., c='green',alpha=0.7,label='SNR = {:.2f}'.format(self.snr[indh,indv]))

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
    
    def save(self, outname):
        delattr(self,'planet')
        np.save(outname, self.__dict__) 
        print('{:} saved...'.format(outname))
        return None
    
    def load(self, path):
        print('Loading Datacube from...', path)
        d = np.load(path, allow_pickle=True).tolist()
        for key in d.keys():
            setattr(self, key, d[key])
        return self 


    
        
class Template(Datacube):
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
        
#    def check_data(self):
#        cenwave = np.median(self.wlt)
#        unit = 'Unknown'
#        if (cenwave < 1000) & (cenwave>100):
#            unit = 'nm'
#            print('--> Transforming from nm to A...')
#            self.wlt *= 10.
#            print('New central wavelength = {:.2f} A'.format(np.median(self.wlt)))
#            
#        elif cenwave > 1000:
#            print('Wavelength in A')
#            unit = 'A'
#        else:
#            print('Wavelength in Unknown units!!')
#        return self
    
    def shift_2D(self, RV):
        c = const.c.to('km/s').value
        beta = 1 + (RV/c)
        self.gflux = np.zeros((RV.size, self.wlt.size))
        cs = splrep(self.wlt, self.flux)
        for j in range(RV.size):
            self.gflux[j,] = splev(self.wlt*beta[j], cs)
        return self
    
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
            window = dRV * pixscale
            if debug: print('window = {:.2f} pixels'.format(window))
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
    
    def pyAstro_convolve(self, row):
        from PyAstronomy import pyasl
        newflux = pyasl.instrBroadGaussFast(self.wlt, self.gflux[row,:],
                                            self.res, edgeHandling="firstlast", fullout=False, equid=True,maxsig=5.0)
        
        return newflux
    def convolve_instrument(self, res=50e3, num_cpus=5):
        import multiprocessing as mp
        self.res = res
        rows = np.arange(0, self.gflux.shape[0])
        with mp.Pool(num_cpus) as p:
            output = p.map(self.pyAstro_convolve, rows)
            
        self.gflux = np.array(output)
        return output
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    