import numpy as np
#from .datacube import Datacube

def SysRem1(dc, a_j=None, mode='subtract', max_iter=1000, debug=False):
    nans = np.isnan(dc.wlt)
    
    if dc.flux_err is None:
        dc.estimate_noise()
    r_ij, sigma_ij = dc.flux[:,~nans], dc.flux_err[:,~nans]
    r_ij = (r_ij.T - np.nanmedian(r_ij, axis=1)).T
    err2 = sigma_ij**2
    if a_j is None:
        a_j = np.ones_like(r_ij.shape[0])
    
    correction = np.zeros_like(r_ij)
    

    for j in range(max_iter):
        correction0 = correction
        # Sysrem Algorithm 
        c_i = np.nansum((r_ij/err2).T*a_j, axis=1) / np.nansum((a_j**2/err2.T), axis=1)
        if np.isnan(np.sum(c_i)):
            print('NaN in c_i...')
            return (dc, a_j)
        a_j = np.nansum(r_ij*c_i/err2, axis=1) / np.nansum(c_i**2/err2, axis=1)

        
        correction = np.outer(a_j, c_i)
        fractional_dcorr = np.sum(np.abs(correction-correction0))/(np.sum(np.abs(correction0))+1e-5)
        # if (j%50)==0:
        #     print(j, fractional_dcorr)
        if j>1 and fractional_dcorr< 1e-2:
            if debug: print('Convergence reached at {:} iterations'.format(j))
            break
        
        
    if mode=='subtract':
        r_ij -= correction

    elif mode=='divide':
        r_ij /= correction
        sigma_ij /= correction
        err2 = sigma_ij**2
        
    dc.flux[:,~nans] =  r_ij 
    dc.flux_err[:,~nans] = sigma_ij
    return (dc, a_j)


def SysRemRoutine(dc_in, N, mode='subtract', debug=False):
   '''Perform `N` sysrem iterations on Datacube `dc`
   The gibson mode requires proper implementation'''
   import copy
   dc = copy.deepcopy(dc_in)
   
   nans = np.isnan(dc.wlt)
   # dc.flux[:,~nans] = (dc.flux[:,~nans].T - np.nanmedian(dc.flux[:,~nans], axis=1)).T ## TESTING MAY 28th
   
   a_j = np.ones_like(dc.flux.shape[0])
   
   for i in range(1,N+1):
       dc_sys, a_j = SysRem1(dc, a_j, mode=mode, debug=debug)
       if debug:
           Q = 1/np.nanmean(np.std(dc_sys.flux[:,~nans], axis=1))
           print('Sysrem {:}/{:} --- Q = {:.3f}'.format(i,N, Q))
           

    # Weight columns by noise (see Spring+2022 section 4.4.2)
   mean_std = np.nanstd(dc_sys.flux[:,~nans])

   dc_sys.flux[:,~nans] /= np.nanstd(dc_sys.flux[:,~nans], axis=0) # NEW 24 April 2022
   dc_sys.flux_err[:,~nans] /= np.nanstd(dc_sys.flux[:,~nans], axis=0)
       
   dc_sys.flux[:,~nans] *= mean_std
   dc_sys.flux_err[:,~nans] *= mean_std

   return dc_sys.flux  


