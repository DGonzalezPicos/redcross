import numpy as np
#from .datacube import Datacube


class SysRem:
    '''SysRem implementation adapted from PyAstronomy: 
        https://github.com/sczesla/PyAstronomy/blob/03720761a3ad8fd8f59b8bb5798a1b1cd5218f71/src/pyasl/asl/aslExt_1/sysrem.py'''
    
    def __init__(self, datacube, a_j=None):
        
        self.nans = np.isnan(datacube.wlt)
        self.r_ij  = datacube.flux[:,~self.nans]
        self.r_ij = (self.r_ij.T - np.nanmedian(self.r_ij, axis=1)).T
        
        if datacube.flux_err is None: # IMPORTANT STEP
            datacube.estimate_noise()
            
#        datacube.estimate_noise()   
            
        self.sigma_ij = datacube.flux_err[:,~self.nans]
        self.err2 = self.sigma_ij**2
        
        if a_j is None:
            self.a_j = np.ones_like(self.r_ij.shape[0])
            
        # parameters for convergence
        self.max_iter = 1000  # maximum iterations for each sysrem component
        self.atol = 1e-3 # absolute tolerance
        self.rtol = 0 # relative tolerance
        

    def compute_c(self, a=None):   
        if a is None:
            a = self.a_j
        c = np.nansum( ((self.r_ij/self.err2).T * a).T, axis=0) / np.nansum( (a**2 * (1/self.err2).T).T, axis=0 )
        return c  

    def compute_a(self, c):
        if c is None:
            c = self.compute_c()
        return np.nansum(self.r_ij / self.err2 * c, axis=1) / np.nansum( c**2 * (1/self.err2), axis=1 )
    
    def run(self, n=6, mode='subtract', debug=False):
        ''''Run SysRem for `n` cycles
        r_ij values are updated inside the function'''
        for i in range(n):
            self.iterate_ac(mode, debug)
        return self

    
        
    def iterate_ac(self, mode='subtract', debug=False):
        a = self.a_j
        # First ac iteration
        c = self.compute_c()
        a = self.compute_a(c)
        m = np.outer(a,c)
        
        converge = False
        for i in range(self.max_iter):
            m0 = m.copy()
            c = self.compute_c(a) # now pass the computed `a`
            a = self.compute_a(c) # recompute `a` with the new `c`
            m = np.outer(a,c) # correction matrix
            
            dm = (m-m0)/self.sigma_ij # fractional change
            
            if np.allclose(dm, 0, rtol=self.rtol, atol=self.atol):
                converge = True
                break
            
        self.last_ac_iteration = i
       
        if not converge:
            print('WARNING: Convergence not reached after {:} iterations...'.format(self.max_iter))
            
        # Store values in class instance
        self.a_j = a
        self.c_i = c
        if mode == 'divide':
            self.r_ij /= m
            self.sigma_ij /= m
        if mode == 'subtract':
            self.r_ij -= m
            
        if debug: 
            std = np.nanmean(np.nanstd(self.r_ij, axis=1))
            print('Convergence at iteration {:3} --- StDev = {:.4f}'.format(self.last_ac_iteration, std))
    
            
        return self
    
    
    
# my old SysRem...
#def SysRem1(dc, a_j=None, mode='subtract', max_iter=1000, debug=False):
#    nans = np.isnan(dc.wlt)
#    
#    if dc.flux_err is None:
#        dc.estimate_noise()
#    r_ij, sigma_ij = dc.flux[:,~nans], dc.flux_err[:,~nans]
#    r_ij = (r_ij.T - np.nanmedian(r_ij, axis=1)).T
#    err2 = sigma_ij**2
#    if a_j is None:
#        a_j = np.ones_like(r_ij.shape[0])
#    
#    correction = np.zeros_like(r_ij)
#    
#
#    for j in range(max_iter):
#        correction0 = correction
#        # Sysrem Algorithm 
#        c_i = np.nansum((r_ij/err2).T*a_j, axis=1) / np.nansum((a_j**2/err2.T), axis=1)
#        if np.isnan(np.nansum(c_i)):
#            print('NaN in c_i...')
#            return (dc, a_j)
#        a_j = np.nansum(r_ij*c_i/err2, axis=1) / np.nansum(c_i**2/err2, axis=1)
#
#        
#        correction = np.outer(a_j, c_i)
#        fractional_dcorr = np.nansum(np.abs(correction-correction0))/(np.nansum(np.abs(correction0))+1e-5)
#        # if (j%50)==0:
#        #     print(j, fractional_dcorr)
#        if j>1 and fractional_dcorr< 1e-2:
#            if debug: print('Convergence reached at {:} iterations'.format(j))
#            break
#        
#        
#    if mode=='subtract':
#        r_ij -= correction
#
#    elif mode=='divide':
#        r_ij /= correction
#        sigma_ij /= correction
#        err2 = sigma_ij**2
#        
#    dc.flux[:,~nans] =  r_ij 
#    dc.flux_err[:,~nans] = sigma_ij
#    return (dc, a_j)
#
#
#def SysRemRoutine(dc, N, mode='subtract', debug=False):
#   '''Perform `N` sysrem iterations on Datacube `dc`
#   The gibson mode requires proper implementation'''
##   dc = copy.deepcopy(dc_in)
#   
#   nans = np.isnan(dc.wlt)
#   # dc.flux[:,~nans] = (dc.flux[:,~nans].T - np.nanmedian(dc.flux[:,~nans], axis=1)).T ## TESTING MAY 28th
#   
#   a_j = np.ones_like(dc.flux.shape[0])
#   
#   for i in range(1,N+1):
#       dc_sys, a_j = SysRem1(dc, a_j, mode=mode, debug=debug)
#       if debug:
#           Q = 1/np.nanmean(np.nanstd(dc_sys.flux[:,~nans], axis=1))
#           print('Sysrem {:}/{:} --- Q = {:.3f}'.format(i,N, Q))
#           
#
#    # NOTE (ago 22 2022): careful with applying the weights TWICE (here and CCF)
#    # Weight columns by noise (see Spring+2022 section 4.4.2)
##   mean_std = np.nanstd(dc_sys.flux[:,~nans])
##
##   dc_sys.flux[:,~nans] /= np.nanstd(dc_sys.flux[:,~nans], axis=0) # NEW 24 April 2022
##   dc_sys.flux_err[:,~nans] /= np.nanstd(dc_sys.flux[:,~nans], axis=0)
##       
##   dc_sys.flux[:,~nans] *= mean_std
##   dc_sys.flux_err[:,~nans] *= mean_std
#
#   return dc_sys.flux  


