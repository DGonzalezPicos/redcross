"""
Created on Thu Aug 25 13:06:26 2022

@author: dario
"""
import numpy as np

class Pipeline:
    '''class to manage the reduction steps and apply them to a single-order datacube
    steps:: list of functions from `datacube` '''
    def __init__(self, steps=None):
        self.steps = steps or []
        self.args = ([None for _ in range(len(self.steps))]) or []
        
        
        self.num_cpus = 6 # by default
        
    def add(self, step, args=None):
        self.steps.append(step)
        self.args.append(args)
        return self
    
    @property
    def info(self):
        d = {k:v for k,v in zip(self.steps, self.args)}
        return d
        
    def reduce(self, order, dc=None, ax=None):
#        print('Reducing order...')
        dc = dc or self.dc
        dco = dc.order(order)
        if not ax is None: dco.imshow(ax=ax[0])
        
        for i, fun in enumerate(self.steps):            
            if self.args[i] != None:
                dco = getattr(dco, fun)(**self.args[i])
            else:
                dco = getattr(dco, fun)()
                
            if not ax is None: 
                dco.imshow(ax=ax[i+1])
                
                props = dict(boxstyle='round', facecolor='black', alpha=0.65)
#                s = np.round(np.nanmean(np.nanstd(dco.flux, axis=0)), 4)
                
                sigma = '$\sigma$ = {:.4f}'.format(np.nanstd(dco.flux))
                x = [0.02, 0.90]
                y = 0.70
                for j,s in enumerate([fun, sigma]):
                    ax[i+1].text(s=s, x=x[j], y=y, transform=ax[i+1].transAxes, c='white',
                            fontsize=12, alpha=0.8,bbox=props)
            
        return dco
    
    def reduce_orders(self, dc, num_cpus=None):
        from p_tqdm import p_map
        
        num_cpus = num_cpus or self.num_cpus

        orders = np.arange(0, dc.nOrders)
        self.dc = dc.copy() # make copy
        
        # run parallel function over all orders
        output = p_map(self.reduce, orders, num_cpus=num_cpus)
        
        # save results in the same shape as input datacube
        [self.dc.update(output[k], k) for k in range(len(output))]
        self.dc.reduction = self.info
        return self.dc
    
    def set_sysrem(self, n):
        '''change the number of sysrem iterations after defining it'''
        sys_ind = int(np.argwhere(np.array(self.steps)=='sysrem'))
        self.args[sys_ind]['n'] = n
        return self
    @property
    def nSysRem(self):
        sys_ind = int(np.argwhere(np.array(self.steps)=='sysrem'))
        return int(self.args[sys_ind]['n'])


    
    
    
    
    
    
    