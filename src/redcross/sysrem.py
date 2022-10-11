import numpy as np
#from .dco import dco


class SysRem:
    '''SysRem implementation adapted from PyAstronomy: 
        https://github.com/sczesla/PyAstronomy/blob/03720761a3ad8fd8f59b8bb5798a1b1cd5218f71/src/pyasl/asl/aslExt_1/sysrem.py'''
    
    def __init__(self, dco, a_j=None):
        # Get input from a single-order `Datacube`
        self.dco = dco
        self.nans = np.isnan(dco.wlt)
        
        # Define residual matrix (mean subtracted, masked data)
        self.r_ij  = np.array(dco.flux[:,~self.nans], dtype=np.float32)
        self.r_ij = (self.r_ij.T - np.nanmedian(self.r_ij, axis=1)).T
        
        # Store the cumulative SysRem model by adding the fitted residuals
        self.sysrem_model = np.zeros_like(self.r_ij) # store the cumulative model
        self.o = dco.o # save the order number for book-keeping
        
        # SysRem needs a noise estimate for every pixel (sigma^2)
        # If not provided, call estimate_noise()
        if not hasattr(dco, 'flux_err'):
            dco.estimate_noise()
            
        # Prepare the noise matrix   
        self.sigma_ij = np.array(dco.flux_err[:,~self.nans], dtype=np.float32)
        # Replace values that are exactly zero (not usually needed)
        zero_mask = self.sigma_ij==0.
        self.sigma_ij[zero_mask] = np.median(self.sigma_ij[~zero_mask])
        self.err2 = np.power(self.sigma_ij, 2, dtype=np.float32)
        
        # Initialise the amplitude-vector 
        if a_j is None:
            self.a_j = np.ones_like(self.r_ij.shape[0], dtype=np.float32)
            
        # parameters for convergence
        self.max_iter = 1000  # maximum iterations for each sysrem component
        self.atol = 1e-5 # absolute tolerance
        self.rtol = 0.0 # relative tolerance
        

    def compute_c(self, a=None):   
        if a is None:
            a = self.a_j # use a new passed `a` or the current stored value
        return (np.nansum( ((self.r_ij / self.err2).T * a).T, axis=0) / np.nansum( (a**2 * (1/self.err2).T).T, axis=0 ))


    def compute_a(self, c=None):
        if c is None:   
            c = self.compute_c()
        return np.nansum(c * self.r_ij / self.err2 , axis=1) / np.nansum( c**2/self.err2, axis=1 )
    
    def run(self, n=6, mode='subtract', debug=False, outdir=None):
        ''''Run SysRem for `n` cycles
        r_ij values are updated inside the function'''
        for i in range(n):
#            print('Order {:} -- iteration {:}'.format(self.o, i))
            if outdir != None:
                self.dco.flux[:,~self.nans] = self.r_ij
                np.save('{:}/order{:}/sysrem{:}.npy'.format(outdir,self.o, i), {'wlt':self.dco.wlt, 'flux':self.dco.flux})
            # Keep calling this function to build `self.sysrem_model`    
            self.iterate_ac(mode, debug)

        return self

    
        
    def iterate_ac(self, mode='subtract', debug=False):
        a = self.a_j
        # First ac iteration
        c = self.compute_c()
        a = self.compute_a(c) 
        a /= np.max(a) # avoid numerical problems (`a` gets large and  `c` gets veery small)
        # Fix `a` amplitude and adjust the `c` coefficients (verified)
        m = np.outer(a,c) # SysRem model
     
        converge = False
        for i in range(self.max_iter):
            m0 = m.copy()
            c = self.compute_c(a) # now pass the computed `a`
            a = self.compute_a(c) # recompute `a` with the new `c`
            m = np.outer(a,c) # correction matrix
            
            dm = (m-m0)/self.sigma_ij # fractional change
            if np.allclose(np.nanmean(dm), 0, rtol=self.rtol, atol=self.atol):
                converge = True
                break
        
        self.last_ac_iteration = i # bookkeeping when things don't converge...
        if not converge:
            print('WARNING for order {:}: Convergence not reached after {:} iterations...'.format(self.o, self.max_iter))
            
        # Update values
        self.a_j = a
        self.c_i = c
        # Subtract current SysRem model
        self.r_ij -= m
        # Store cumulative model (to be subtracted or divided later on)
        self.sysrem_model += m
            
        if debug: 
            std = np.nanmean(np.nanstd(self.r_ij, axis=1))
            print('Convergence at iteration {:3} --- StDev = {:.4f}'.format(self.last_ac_iteration, std))
        return self



    def get_vectors(self, n, debug=False):
        '''Get the explicit `a` and `c` vectors for every SysRem iteration
        The SysRem model is then np.outer(a,c)'''
        a = np.zeros((n, self.dco.nObs))
        c = np.zeros((n, self.dco.nPix))
        for i in range(n):
            self.iterate_ac(debug=debug)
            a[i,] = self.a_j
            c[i,] = self.c_i
        return a, c
