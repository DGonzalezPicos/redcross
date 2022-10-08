import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d, splrep, splev
from copy import deepcopy
import astropy.units as u
from .datacube import Datacube
from pathos.pools import ProcessPool

c = 2.998e5 # km/s

class CCF(Datacube):
    mode = 'ccf'
    def __init__(self, rv=None, template=None, flux=None, **kwargs):
        self.rv = rv
        if not self.rv is None: self.dRV = np.mean(np.diff(self.rv)) # resolution
        self.template = template
        self.flux = flux
        
        
        self.window = 0. # high pass gaussian to apply to the template before CCF.run()
        self.num_cpus = 6 # by default
    def normalise(self):
        self.flux = self.flux / np.median(self.flux, axis=0)
        return self  
    
    @property
    def wlt(self):
        return self.rv
    
    @property
    def snr(self):
        rv_abs = np.abs(self.rv)
        p40 = np.percentile(rv_abs, 40)
        bkg = self.flux[:,rv_abs > p40]
        ccf_1d = np.median(self.flux, axis=0)
        ccf_1d -= np.median(bkg)
        return ccf_1d / np.std(bkg)
    
    def __prepare_template(self, wave):
        # start = time.time()
        # print('Computing 2D template...')
        temp2D = self.template.shift_2D(self.rv, wave)
        if self.window > 0.:
            temp2D.high_pass_gaussian(window=self.window)
        self.gTemp = temp2D.flux
        self.gTemp -= np.nanmean(self.gTemp, axis=0) 
        return self
    
    def cross_correlation(self, o):
        # if order is not None:
        #     wave = self.dc.wlt[order,:]
        #     flux = self.dc.flux[order,:,:]
        # else:
        #     wave = self.dc.wlt
        #     flux = self.dc.flux
        
        dco = self.dc.order(o)
        nans = np.isnan(dco.wlt)
        wave, flux = dco.wlt[~nans], dco.flux[:,~nans]
            
        
        # self.__prepare_template(wave[~nans]) # OLD WAY OF DOING CCF (actually slower)
        
        
        # the data is weighted by the variance of each pixel channel (sigma^2)
        data = flux - np.nanmean(flux, axis=0)
        f = data / np.var(data, axis=0)
        
        temp = self.template.copy().crop(np.min(wave), np.max(wave), eps=0.30).sort()
        temp.flux -= np.mean(temp.flux)
        # cs = splrep(temp.wlt, temp.flux)
        # print(cs)
        
        # shifts
        beta = 1 - (self.rv/c)
        # build 2D template (for every RV-shift)
        # g = np.array([splev(wave*beta[j], cs) for j in range(beta.size)])
        g = np.array([interp1d(temp.wlt, temp.flux)(wave*beta[j]) for j in range(beta.size)])
            
        ccf_i = np.dot(f, g.T)
            
        return ccf_i
    
    def run(self, dc):
        self.frame = dc.frame
        start=time.time()
        
        self.dc = dc.copy()
        self.flux = np.zeros((dc.nObs, self.rv.size))
        
        
        if len(dc.shape) > 2:
            orders = np.arange(0, dc.nOrders)

            if self.num_cpus > 0:
                pool = ProcessPool(nodes=self.num_cpus)
                output = np.array(pool.amap(self.cross_correlation, orders).get())
                self.flux = np.sum(output, axis=0)
                
            else:
                self.flux = sum([self.cross_correlation(dc.order(o)) for o in orders])
                
            
        else:
            self.flux = self.cross_correlation(dc)
            
        delattr(self, 'dc') # avoid overloading memory
        print('CCF elapsed time: {:.2f} s'.format(time.time()-start))
        return self
   
    # def run(self, dc, debug=False, ax=None):
    #     self.frame = dc.frame
    #     start=time.time()
    #     self.dc = dc.copy()        
    #     # check if data has a HPG filter applied
    #     # if True, apply the same filter to the template after resampling (window in pixels)
    #     if hasattr(dc, 'reduction'):
    #         if 'high_pass_gaussian' in dc.reduction:
    #             self.window = dc.reduction['high_pass_gaussian']['window']
                
    #     if len(dc.shape) > 2:
    #         # from p_tqdm import p_map
    #         orders = np.arange(0, dc.nOrders)
    #         pool = ProcessPool(nodes=self.num_cpus)
    #         ccf_i = np.array(pool.amap(self.cross_correlation, orders).get())
    #         # ccf_i = np.array(p_map(self.cross_correlation, orders, num_cpus=6))
    #         self.flux = np.sum(ccf_i, axis=0)
    #     else:
    #         # check whether the 2D template needs to be computed
    #         # this avoids recomputing when changing the data input for the same template
    #         nans = np.isnan(dc.wlt)
    #         if not hasattr(self, 'gTemp'):
    #             self.__prepare_template(dc.wlt[~nans])
    
    #         self.flux = self.cross_correlation()
                    
    #     if debug:
    #         print('CCF elapsed time: {:.2f} s'.format(time.time()-start))
    #         print('CCF shape = {:}'.format(self.shape))
            
    #     if ax != None: self.imshow(ax=ax)
    #     return self
    
    
    def interpolate_to_planet(self, i):
        '''help function to parallelise `to_planet_frame` '''
        inter = interp1d(self.rv, self.flux[i,], bounds_error=False, fill_value=0.0)
        flux_i = inter(self.rv+self.planet.RV[i])
        return flux_i
    
    def to_planet_frame(self, planet, ax=None, num_cpus=6, return_self=False):
        ccf = self.copy()
        ccf.planet = planet
            
        pool = ProcessPool(nodes=num_cpus)
        flux_i = pool.amap(ccf.interpolate_to_planet, np.arange(self.nObs)).get()
        
        ccf.flux = np.array(flux_i)
        
        mask = (np.abs(ccf.rv)<np.percentile(np.abs(ccf.rv), 50))
        ccf.rv = ccf.rv[mask]
        ccf.flux = ccf.flux[:,mask]
        if ax != None: ccf.imshow(ax=ax)
        ccf.frame = 'planet'
        if return_self:
            self.rv_planet = ccf.rv
            self.flux_planet = ccf.flux
            return self
        return ccf
    
    def eclipse_label(self, planet, ax, x_rv=None, c='w'):
        x_rv = x_rv or np.percentile(self.rv, 20) # x-position of text
        self.planet = planet.copy()
        # print(self.planet.frame)

        
        phase_14 = 0.5 * ((planet.T_14) % planet.P) / planet.P
        y = [0.50 - (i*phase_14) for i in (-1,1)]
        [ax.axhline(y=y[i], ls='--', c=c) for i in range(2)]
        ax.text(s='eclipse', x=x_rv+5, y=0.50, c=c, fontsize=12)#, transform=ax.transAxes)
        ax.annotate('', xy=(x_rv, y[1]), xytext=(x_rv, y[0]), c=c, arrowprops=dict(arrowstyle='<->', ec=c))
        
        # planet trail
        mask = np.abs(self.planet.phase - 0.50) < (phase_14)
        self.planet.frame = self.frame # get the correct RV
        ax.plot(self.planet.RV[mask], self.planet.phase[mask], '--r')
        ax.plot(self.planet.RV[-10:], self.planet.phase[-10:], '--r')
        return ax
    
    
    
    def autoccf(self):
        self.flux = np.zeros_like(self.rv)
        wave, flux = self.template.wlt, self.template.flux

        beta = 1 + (self.rv/c)
        for i in range(self.rv.size):
            fxt_i = interp1d(wave*beta[i], flux, fill_value="extrapolate")(wave)
            self.flux[i,] = np.dot(flux, fxt_i) / np.sum(fxt_i)
        return self

                
                
class KpV:
    def __init__(self, ccf=None, planet=None, deltaRV=None, 
                 kp_radius=50., vrest_max=80., bkg=20):
        if not ccf is None:
            self.ccf = ccf.copy()
            self.planet = deepcopy(planet)
    #        self.kpVec = self.planet.Kp.value + np.arange(-kp[0], kp[0], kp[1])
    #        self.vrestVec = np.arange(-vrest[0], vrest[0]+vrest[1], vrest[1])
            self.dRV = deltaRV or ccf.dRV
    
            self.kpVec = self.planet.Kp + np.arange(-kp_radius, kp_radius, self.dRV)
            self.vrestVec = np.arange(-vrest_max, vrest_max, self.dRV)
            self.bkg = bkg
            
            
            self.num_cpus = 6 # for the functions that allow parallelisation
            
    def shift_vsys(self, iObs):
        print(iObs)
        outRV = self.vrestVec + self.rv_planet[iObs]
        return interp1d(self.ccf.rv, self.ccf.flux[iObs,])(outRV)    
        
    def run(self, snr=True, ignore_eclipse=True, ax=None):
        '''Generate a Kp-Vsys map
        if snr = True, the returned values are SNR (background sub and normalised)
        else = map values'''
    
        ecl = False * np.ones_like(self.planet.RV)         
        if ignore_eclipse:   
            ecl = self.planet.mask_eclipse(return_mask=True)
            
        snr_map = np.zeros((len(self.kpVec), len(self.vrestVec)))
        rvel = ((self.planet.v_sys*u.km/u.s)-self.planet.BERV*u.km/u.s).value 
        
        
        for ikp in range(len(self.kpVec)):
            self.rv_planet = rvel + (self.kpVec[ikp]*np.sin(2*np.pi*self.planet.phase))
            
            # for iObs in range(self.planet.RV.size):
            for iObs in np.where(ecl==False)[0]:
                outRV = self.vrestVec + self.rv_planet[iObs]
                snr_map[ikp,] += interp1d(self.ccf.rv, self.ccf.flux[iObs,])(outRV) 
            
        if snr:
            noise_map = np.std(snr_map[:,np.abs(self.vrestVec)>self.bkg])
            bkg_map = np.median(snr_map[:,np.abs(self.vrestVec)>self.bkg]) # subtract the background level
            snr_map -= bkg_map        
            self.snr = snr_map / noise_map
        else:
            self.snr = snr_map # NOT ACTUAL SIGNAL-TO-NOISE ratio (useful for computing weights)
            
        self.bestSNR = self.snr.max() # store info as variable
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
        self.bestSNR = self.snr.max()
        ipeak = np.where(self.snr == self.bestSNR)
        bestVr = float(self.vrestVec[ipeak[1]])
        bestKp = float(self.kpVec[ipeak[0]])
        
        if display:
            print('Peak position in Vrest = {:3.1f} km/s'.format(bestVr))
            print('Peak position in Kp = {:6.1f} km/s'.format(bestKp))
            print('Max SNR = {:3.1f}'.format(self.bestSNR))
        return(bestVr, bestKp, self.bestSNR)
    
    def plot(self, fig=None, ax=None, peak=None, vmin=None, vmax=None, label=''):
        lims = [self.vrestVec[0],self.vrestVec[-1],self.kpVec[0],self.kpVec[-1]]
        vmin = vmin or self.snr.min()
        vmax = vmax or self.snr.max()

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
    
    def snr_at_peak(self, peak):
        self.indv = np.abs(self.vrestVec - peak[0]).argmin()
        self.indh = np.abs(self.kpVec - peak[1]).argmin()
        self.peak_snr = self.snr[self.indh,self.indv]
        return self
        
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
       # get the values     
        self.snr_at_peak(peak)
        
        # indv = np.abs(self.vrestVec - peak[0]).argmin()
        # self.indh = np.abs(self.kpVec - peak[1]).argmin()
        # self.peak_snr = self.snr[self.indh,indv]
    
        row = self.kpVec[self.indh]
        col = self.vrestVec[self.indv]
        print('Horizontal slice at Kp = {:.1f} km/s'.format(row))
        print('Vertical slice at Vrest = {:.1f} km/s'.format(col))
        ax2.plot(self.vrestVec, self.snr[self.indh,:], 'gray')
        ax3.plot(self.snr[:,self.indv], self.kpVec,'gray')
        
        
    
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
    
    def plot_1D(self, peak, ax, vmin=None, vmax=None, label=None, return_data=False, **kwargs):
        
        vmin = vmin or self.snr.min()
        vmax = vmax or self.snr.max()
        ind_kp0 = np.abs(self.kpVec - peak[1]).argmin()
        # print('Best Kp = {:.1f} km/s'.format(kpv_12.kpVec[ind_kp0]))
        label = label or 'Kp = {:.1f} km/s'.format(self.kpVec[ind_kp0])
        ax.plot(self.vrestVec, self.snr[ind_kp0,:], label=label, **kwargs)
        ax.set(xlabel='$\Delta v}$ (km/s)', ylabel='SNR', xlim=(self.vrestVec.min(), self.vrestVec.max()))
        ax.set_ylim((vmin, vmax))
        ax.legend(frameon=False, loc='upper right')
        if return_data:
            return self.snr[ind_kp0,:]
        else:
            return ax
    
    def merge_kpvs(self, kpv_list):
        new_kpv = kpv_list[0].copy()
        # add signal
        new_kpv.snr = np.sum([kpv_list[i].snr for i in range(len(kpv_list))], axis=0)
        
        # convert into actual SNR 
        noise_map = np.std(new_kpv.snr[:,np.abs(new_kpv.vrestVec)>new_kpv.bkg])
        bkg_map = np.median(new_kpv.snr[:,np.abs(new_kpv.vrestVec)>new_kpv.bkg]) # subtract the background level
        new_kpv.snr -= bkg_map   
        new_kpv.snr /= noise_map
        return new_kpv


    
        

    
    
    
    
    
    
    
    
    
    
    
    
    