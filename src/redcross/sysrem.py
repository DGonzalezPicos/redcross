import numpy as np
#from .dco import dco


class SysRem:
    '''SysRem implementation adapted from PyAstronomy: 
        https://github.com/sczesla/PyAstronomy/blob/03720761a3ad8fd8f59b8bb5798a1b1cd5218f71/src/pyasl/asl/aslExt_1/sysrem.py'''
    
    def __init__(self, dco, a_j=None):
        
        self.dco = dco
        self.nans = np.isnan(dco.wlt)
        self.r_ij  = np.array(dco.flux[:,~self.nans], dtype=np.float32)
        self.r_ij = (self.r_ij.T - np.nanmedian(self.r_ij, axis=1)).T
        self.sysrem_model = np.zeros_like(self.r_ij) # store the cumulative model
        self.o = dco.o # save the order number for book-keeping
        
        
        
        if dco.flux_err is None: # IMPORTANT STEP
            dco.estimate_noise()
            
#        dco.estimate_noise()   
            
        self.sigma_ij = np.array(dco.flux_err[:,~self.nans], dtype=np.float32)
        zero_mask = self.sigma_ij==0.
        self.sigma_ij[zero_mask] = np.median(self.sigma_ij[~zero_mask])
        self.err2 = np.power(self.sigma_ij, 2, dtype=np.float32)
        
        if a_j is None:
            self.a_j = np.ones_like(self.r_ij.shape[0], dtype=np.float32)
            
        # parameters for convergence
        self.max_iter = 1000  # maximum iterations for each sysrem component
        self.atol = 1e-5 # absolute tolerance
        self.rtol = 0.0 # relative tolerance
        

    def compute_c(self, a=None):   
        if a is None:
            a = self.a_j
        c = np.nansum( ((self.r_ij/self.err2).T * a).T, axis=0) / np.nansum( (a**2 * (1/self.err2).T).T, axis=0 )
        return c  

    def compute_a(self, c):
        if c is None:
            c = self.compute_c()
        return np.nansum(c*self.r_ij / self.err2 , axis=1) / np.nansum( c**2/self.err2, axis=1 )
    
    def run(self, n=6, mode='subtract', debug=False, outdir=None):
        ''''Run SysRem for `n` cycles
        r_ij values are updated inside the function'''
        for i in range(n):
#            print('Order {:} -- iteration {:}'.format(self.o, i))
            if outdir != None:
                self.dco.flux[:,~self.nans] = self.r_ij
                np.save('{:}/order{:}/sysrem{:}.npy'.format(outdir,self.o, i), {'wlt':self.dco.wlt, 'flux':self.dco.flux})
                
            self.iterate_ac(mode, debug)

        return self

    
        
    def iterate_ac(self, mode='subtract', debug=False):
        a = self.a_j
        # First ac iteration
        c = self.compute_c()
        a = self.compute_a(c) 
        ## TESTING vvv##
        a /= np.max(a) # avoid numerical problems (`a` gets large and  `c` gets veery small)
#        c /= np.max(c)
        m = np.outer(a,c)
     
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
            
        self.last_ac_iteration = i
       
        if not converge:
            print('WARNING for order {:}: Convergence not reached after {:} iterations...'.format(self.o, self.max_iter))
            
        # Store values in class instance
        self.a_j = a
        self.c_i = c

        self.r_ij -= m
            
        self.sysrem_model += m
            
        if debug: 
            std = np.nanmean(np.nanstd(self.r_ij, axis=1))
            print('Convergence at iteration {:3} --- StDev = {:.4f}'.format(self.last_ac_iteration, std))
    
            
        return self



    def get_vectors(self, n, debug=False):
        a = np.zeros((n, self.dco.nObs))
        c = np.zeros((n, self.dco.nPix))
        for i in range(n):
            self.iterate_ac(debug=debug)
            a[i,] = self.a_j
            c[i,] = self.c_i
        return a, c
