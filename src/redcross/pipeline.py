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
        
    def add(self, step, args=None):
        self.steps.append(step)
        self.args.append(args)
        
    def reduce(self, order, dc=None, ax=None):
#        print('Reducing order...')
        dc = dc or self.dc
        dco = dc.order(order)
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
                
                sigma = '$\sigma$ = {:.4f}'.format(np.nanstd(dco.flux))
                x = [0.02, 0.90]
                y = 0.70
                for j,s in enumerate([fun, sigma]):
                    ax[i+1].text(s=s, x=x[j], y=y, transform=ax[i+1].transAxes, c='white',
                            fontsize=12, alpha=0.8,bbox=props)
            
        return dco
    
    def reduce_orders(self, dc, num_cpus=4):
        import multiprocessing as mp
        import tqdm
#        from p_tqdm import p_map
#        from pathos.pools import ProcessPool
#        from joblib import Parallel, delayed
        orders = np.arange(0, dc.nOrders)
        self.dc = dc.copy() # make copy
#        output = p_map(self.reduce, orders, num_cpus=num_cpus)
#        pool = ProcessPool(nodes=4)
#        output = pool.map(self.reduce, orders, num_cpus=num_cpus)
        
        pool = mp.Pool(processes=4)
        output = []
        for result in tqdm.tqdm(pool.imap_unordered(self.reduce, orders), total=len(orders)):
            output.append(result)
        # return output
        
#         print(output[0].shape)
# #        output = Parallel(n_jobs=4, verbose=1)(delayed(self.reduce)(orders))        
#         # self.dc.wlt = np.hstack([output[k].wlt for k in range(orders.size)])
#         # self.dc.flux = np.hstack([output[k].flux for k in range(orders.size)])
#         # self.dc.flux_err = np.hstack([output[k].flux_err for k in range(orders.size)])
        [self.dc.update(output[k], k) for k in range(len(output))]
#         print(self.dc.wlt.shape)
#         self.dc.wlt = np.median(self.dc.wlt, axis=1)
#         self.dc.sort_wave()
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

