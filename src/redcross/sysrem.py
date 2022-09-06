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

