import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d, splrep, splev
from copy import deepcopy
import astropy.units as u
from .datacube import Datacube
from pathos.pools import ProcessPool
from joblib import Parallel, delayed

c = 2.998e5 # km/s

class CCF(Datacube):
    mode = 'ccf'
    def __init__(self, rv=None, template=None, flux=None, **kwargs):
        self.rv = rv
        if not self.rv is None: self.dRV = np.mean(np.diff(self.rv)) # resolution
        self.template = template.copy()
        self.flux = flux
        
        
        self.window = 0. # high pass gaussian to apply to the template before CCF.run()
        self.n_jobs = 6 # by default
        self.spline = False # use linear interpolation unless this is True
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
    
    def cross_correlation(self, dco, noise='var'):
        '''Basic cross-correlation between a single-order datacube `dco` 
        and a 1D template'''
        # manage NaNs
        nans = np.isnan(dco.wlt)
        wave, flux = dco.wlt[~nans], dco.flux[:,~nans]
        f = flux - np.mean(flux)
        
        temp = self.template.copy().crop(np.min(wave), np.max(wave), eps=0.40)
        temp.flux -= np.mean(temp.flux)
    
        
        # shifts
        beta = 1 - (self.rv/c)
        # build 2D template (for every RV-shift)
        if self.spline:
            # For templates at very high resolution (~ 1e6) the spline decomposition fails
            # because the points are **too close** together (oversampled)
            cs = splrep(temp.wlt, temp.flux)
            g = np.array([splev(wave*b, cs) for b in beta])
        else:
            # when spline fails... linear interpolation doesn't
            # for very high-res templates there's no difference in the results
            _inter = interp1d(temp.wlt, temp.flux)
            g = np.array([_inter(wave*b) for b in beta])
            

        # compute the CCF-map in one step, `_i` refers to the given order 
        if noise == 'flux_err':
            noise2 = dco.flux_err[:, ~nans]**2
        elif noise == 'var':
            noise2 = np.var(f, axis=0)
            
        # The CCF-map in one step
        return np.dot(f/noise2, g.T)
    
    def run(self, dc, apply_filter=False, noise='var', ax=None):
        self.frame = dc.frame
        start=time.time()
        
        if apply_filter:
            if hasattr(dc, 'reduction'):
                if 'high_pass_gaussian' in dc.reduction:
                    window = dc.reduction['high_pass_gaussian']['window']
                    print('Applying filter of window = {:} pixels'.format(window))
                    self.template.high_pass_gaussian(window, mode='divide')
    
        if len(dc.shape) > 2:
            # Iterate over orders and sum each CCF_i
            orders = np.arange(0, dc.nOrders, dtype=int)
            
            # output = Parallel(n_jobs=self.n_jobs)(delayed(self.cross_correlation)(dc.order(o), noise) for o in orders)
            output = [self.cross_correlation(dc.order(o), noise) for o in orders]
            self.flux = np.sum(np.array(output), axis=0)
           
        else: # single order CCF (or merged datacube)
            self.flux = self.cross_correlation(dc, noise)
            
        print('CCF elapsed time: {:.2f} s'.format(time.time()-start))
        # print('mean {:.4e} -- std {:.4f}'.format(np.mean(self.flux), np.std(self.flux)))
        if ax != None: self.imshow(ax=ax)
        return self
    
    
    def interpolate_to_planet(self, i):
        '''help function to parallelise `to_planet_frame` '''
        inter = interp1d(self.rv, self.flux[i,], bounds_error=False, fill_value=0.0)
        flux_i = inter(self.rv+self.planet.RV[i])
        return flux_i
    
    def to_planet_frame(self, planet, ax=None, n_jobs=6, return_self=False):
        ccf = self.copy()
        ccf.planet = planet
            
        pool = ProcessPool(nodes=n_jobs)
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
            
            try:
                self.planet.frame = self.ccf.frame
                print(self.planet.frame)
            except:
                print('Define data rest frame...')
            self.n_jobs = 6 # for the functions that allow parallelisation
            
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
        
        for ikp in range(len(self.kpVec)):            
            self.planet.Kp = self.kpVec[ikp]
            pRV = self.planet.RV
            for iObs in np.where(ecl==False)[0]:
                outRV = self.vrestVec + pRV[iObs]
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
    
    def plot(self, fig=None, ax=None, peak=None, vmin=None, vmax=None, label='',
             plot_peak=True):
        lims = [self.vrestVec[0],self.vrestVec[-1],self.kpVec[0],self.kpVec[-1]]
        vmin = vmin or self.snr.min()
        vmax = vmax or self.snr.max()

        ax = ax or plt.gca()
        obj = ax.imshow(self.snr,origin='lower',extent=lims,aspect='auto', 
                        cmap='inferno',vmin=vmin,vmax=vmax, label=label)
        if not fig is None: fig.colorbar(obj, ax=ax, pad=0.05)
        ax.set_xlabel('$\Delta$v (km/s)')
        ax.set_ylabel('K$_p$ (km/s)')

        if plot_peak:
            peak = peak or self.snr_max()
            
            indv = np.abs(self.vrestVec - peak[0]).argmin()
            indh = np.abs(self.kpVec - peak[1]).argmin()
        
            row = self.kpVec[indh]
            col = self.vrestVec[indv]
            line_args ={'ls':':', 'c':'white','alpha':0.35,'lw':'3.'}
            ax.axhline(y=row, **line_args)
            ax.axvline(x=col, **line_args)
            ax.scatter(col, row, marker='*', s=3., c='green',alpha=0.7,label='SNR = {:.2f}'.format(self.snr[indh,indv]))

        return obj
    
    def snr_at_peak(self, peak):
        self.indv = np.abs(self.vrestVec - peak[0]).argmin()
        self.indh = np.abs(self.kpVec - peak[1]).argmin()
        self.peak_snr = self.snr[self.indh,self.indv]
        return self
        
    def fancy_figure(self, figsize=(6,6), peak=None, vmin=None, vmax=None,
                     outname=None, title=None, **kwargs):
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
        ax3.xaxis.tick_top()
        
        eps = 0.1 * (self.snr.max()-self.snr.max())
        vmin = vmin or self.snr.min() - eps
        vmax = vmax or self.snr.max() + eps
        
        ax2.set_ylim(vmin, vmax)
        ax3.set_xlim(vmin, vmax)
            
        lims = [self.vrestVec[0],self.vrestVec[-1],self.kpVec[0],self.kpVec[-1]]

        obj = ax1.imshow(self.snr,origin='lower',extent=lims,aspect='auto', 
                         cmap='inferno', vmin=vmin, vmax=vmax)
    
        # figure settings
        ax1.set(ylabel='$K_p$ (km/s)', xlabel='$\Delta v$ (km/s)', **kwargs)
        
        # colorbar
        cax = fig.add_axes([ax3.get_position().x1+0.01,ax3.get_position().y0,
                            0.035,ax3.get_position().height])

        fig.colorbar(obj, cax=cax)
        
        if peak is None:
            peak = self.snr_max()
       # get the values     
        self.snr_at_peak(peak)
    
        row = self.kpVec[self.indh]
        col = self.vrestVec[self.indv]
        print('Horizontal slice at Kp = {:.1f} km/s'.format(row))
        print('Vertical slice at Vrest = {:.1f} km/s'.format(col))
        ax2.plot(self.vrestVec, self.snr[self.indh,:], 'gray')
        ax3.plot(self.snr[:,self.indv], self.kpVec,'gray')
        
        
    
        line_args = {'ls':':', 'c':'white','alpha':0.35,'lw':'3.', 'dashes':(0.7, 1.)}
        ax1.axhline(y=row, **line_args)
        ax1.axvline(x=col, **line_args)
        ax1.scatter(col, row, marker='*', c='red',label='SNR = {:.2f}'.format(self.peak_snr), s=6.)
        ax1.legend(handlelength=0.75)
    
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
        print('Loading KpV object from...', path)
        d = np.load(path, allow_pickle=True).tolist()
        for key in d.keys():
            setattr(self, key, d[key])
        return self 
    
    @staticmethod
    def gaussian(x, a, x0, sigma, y0):
        return y0 + a*np.exp(-(x-x0)**2/(2*sigma**2))
    
    def __fit_slice(self, x,y, label):
        from scipy.optimize import curve_fit
        b = int(len(x) / 3) # ignore the first and last third of the data (consider only central region)
        popt, pcov = curve_fit(self.gaussian, x[b:-b], y[b:-b],
                               bounds=([0., x.min(), 0., -10.],
                                       [np.inf, x.max(), np.inf, np.inf]))
        # perr = np.sqrt(np.diag(pcov)) # uncertainty on the popt parameters
        print('{:} = {:.2f} km/s'.format(label, popt[1]))
        print('FWHM = {:.2f} km/s'.format(popt[2]))
        return popt
    
    def get_slice(self, axis=0, peak=None, vmin=None, vmax=None, fit=False,
                  ax=None, **kwargs):
        peak = peak or self.snr_max()[:2]
        peak = peak[::-1] # invert the peak x,y
        vmin = vmin or self.snr.min()
        vmax = vmax or self.snr.max()
        
        
        
        x_label = [r'$K_p$', r'$\Delta v$']
        x = np.array([self.kpVec, self.vrestVec])
        
        ind = [np.abs(x[i] - peak[i]).argmin() for i in [0,1]][axis]
        x = x[::-1][axis] # get the correct x...
        
        y = np.take(self.snr, ind, axis) # equivalent to self.snr[ind,:] for axis=0
        
        if fit:
            popt = self.__fit_slice(x,y, x_label[::-1][axis])
        
        if ax != None:
            label = '{:}\n{:.1f} km/s'.format(x_label[axis], peak[axis])
            label = ''
            ax.plot(x, y, label=label, **kwargs)
            ax.set(ylabel='SNR', xlabel=x_label[::-1][axis]+' (km/s)', 
                   xlim=(x.min(), x.max()), ylim=(vmin, vmax))
            ax.set_title('CCF at {:} = {:.1f} km/s'.format(x_label[axis], peak[axis]))
            ax.axvline(x=peak[::-1][axis], ls='--',c='k', alpha=0.4)
            if fit:
                ax.plot(x, self.gaussian(x, *popt), ls='--', alpha=0.9, 
                        label='Gaussian fit', c='darkgreen')
            ax.legend(handlelength=0.55)
            
        if fit:
            return (y, popt)
        return y
        

        
        
    
    def plot_slice(self, mode='kp', peak=None, ax=None, vmin=None, vmax=None, 
                   label=None, return_data=False, **kwargs):
        ax = ax or plt.gca()
        peak = peak or self.snr_max()[:2]
        vmin = vmin or self.snr.min()
        vmax = vmax or self.snr.max()
        
        
        if mode == 'kp':
            ind_kp0 = np.abs(self.kpVec - peak[1]).argmin()
            # print('Best Kp = {:.1f} km/s'.format(kpv_12.kpVec[ind_kp0]))
            y = self.snr[ind_kp0,:]
            label = label or 'Kp = {:.1f} km/s'.format(self.kpVec[ind_kp0])
            ax.plot(self.vrestVec, y, '-', label=label, **kwargs)
            
            ax.set(xlabel='$\Delta v$ (km/s)', ylabel='SNR', xlim=(self.vrestVec.min(), self.vrestVec.max()))
            
        elif mode == 'dv':
            ind_dv0 = np.abs(self.vrestVec - peak[0]).argmin()
            # print('Best Kp = {:.1f} km/s'.format(kpv_12.kpVec[ind_kp0]))
            y = self.snr[:,ind_dv0] # magnitude to plot / return
            label = label or '$\Delta$v = {:.1f} km/s'.format(self.vrestVec[ind_dv0])
            ax.plot(self.kpVec, y, '-', label=label, **kwargs)
            ax.set(xlabel='$K_p$ (km/s)', ylabel='SNR', xlim=(self.kpVec.min(), self.kpVec.max()))
            
        ax.set_ylim((vmin, vmax))
        ax.legend(frameon=False, loc='upper right')
            
        if return_data:
            return y
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
    
    def fit_peak(self):
        self.fit = []
        for i in range(2):
            _, pfit = self.get_slice(i, fit=True)
            # print(pfit)
            self.fit.append(pfit[1:3])
        return self


    
        

    
    
    
    
    
    
    
    
    
    
    
    
    