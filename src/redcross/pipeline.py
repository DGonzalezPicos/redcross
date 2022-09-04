"""
Created on Thu Aug 25 13:06:26 2022

@author: dario
"""
import numpy as np

class Pipeline:
    '''class to manage the reduction steps and apply them to a single-order datacube
    steps:: list of functions from `datacube` '''
    def __init__(self, steps=[]):
        self.steps = steps
        self.args = ([None for _ in range(len(steps))])
        
    def add(self, step, args=None):
        self.steps.append(step)
        self.args.append(args)
        
    def reduce(self, dco, ax=None):
#        print('Reducing order...')
        if not ax is None: dco.imshow(ax=ax[0])
        
        for i, fun in enumerate(self.steps):
            if not ax is None: dco.imshow(ax=ax[i])
            
#            print('{:}. {:10}'.format(i+1, fun))
            if self.args[i] != None:
                dco = getattr(dco, fun)(**self.args[i])
            else:
                dco = getattr(dco, fun)()
                
            if not ax is None: 
                dco.imshow(ax=ax[i+1])
                
                props = dict(boxstyle='round', facecolor='black', alpha=0.65)
#                s = np.round(np.nanmean(np.nanstd(dco.flux, axis=0)), 4)
                s = '$\sigma$ = {:.4f}'.format(np.nanstd(dco.flux))
                x, y = 0.05, 0.70
                ax[i+1].text(s=s, x=x, y=y, transform=ax[i+1].transAxes, c='white',
                        fontsize=9, alpha=0.9,bbox=props)
            
        return dco
    
    def set_sysrem(self, n):
        '''change the number of sysrem iterations after defining it'''
        sys_ind = int(np.argwhere(np.array(self.steps)=='sysrem'))
        self.args[sys_ind]['n'] = n
        return self

