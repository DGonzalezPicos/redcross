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
        from pathos.pools import ProcessPool, ThreadPool
        from joblib import Parallel, delayed

        orders = np.arange(0, dc.nOrders)
        self.dc = dc.copy() # make copy
        
        # pool = mp.Pool(processes=num_cpus)
        # output = []
        # for result in tqdm.tqdm(pool.imap_unordered(self.reduce, orders), total=len(orders)):
        #     output.append(result)
        # amap = ProcessPool(nodes=num_cpus).amap
        # tmap = ThreadPool().map
        # pool = ProcessPool(nodes=num_cpus)
        # output = pool.amap(self.reduce, orders).get()
        
        output = Parallel(n_jobs=num_cpus)(delayed(self.reduce)(j) for j in orders)
       
        [self.dc.update(output[k], k) for k in range(len(output))]
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

