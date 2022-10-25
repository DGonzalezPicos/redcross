#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:53:25 2022

@author: dario
"""
import numpy as np
import matplotlib.pyplot as plt

class Plot:
    
    def __init__(self, ax=None, rows=1, cols=1, figsize=(12,3), outname=None):
        self.ax = ax
        self.rows = rows
        self.cols = cols
        self.figsize = figsize
        self.outname = outname
        
        # if self.ax is None:
        #     self.fig, self.ax = plt.subplots(self.rows, self.cols, figsize=self.figsize)

# =============================================================================
#                            DATA(cubes)
# =============================================================================
    
    def master_spectrum(self, dc, ax=None):
        wave = np.median(dc.wlt, axis=0)
        master = np.median(dc.flux, axis=0)
        obj = self.ax.plot(wave, master)
        return obj
        
    
    def data_orders(self, dc, ax, orders=[]):
        
        cmap = plt.cm.get_cmap('viridis', dc.nOrders)
        colors = np.array([cmap(x) for x in range(dc.nOrders)])
        
        dc.flux_err[dc.flux_err==0.0] = np.nan
    
        snr_order = np.nanpercentile(dc.flux / dc.flux_err, 90, axis=(1,2))
        cenwave = np.median(dc.wlt, axis=1)
        widths = np.array([dc.wlt[x,:].max()-dc.wlt[x,:].min() for x in range(dc.wlt.shape[0])])
        
        self.ax[1].bar(cenwave, snr_order, color=colors, width=widths, edgecolor='k', alpha=0.3)
        
        if len(orders) > 1:
            self.ax[1].bar(cenwave[orders], snr_order[orders], color=colors[orders], width=widths[orders], edgecolor='k', alpha=1.)
            [self.ax[1].text(s=str(i), x=cenwave[i]-200, y=snr_order[i]+2, fontsize=7) for i in orders]
        
        for o in range(0,dc.nOrders):
            dco = dc.order(o)
            dco.plot(ax=self.ax[0], c=colors[o], lw=0.25)
            
        # self.ax[0].set(ylabel='Flux', title='GIANO night {:} position {:}'.format(night, pos))
        self.ax[1].set(ylabel='SNR')
        
        return None
    
    
# =============================================================================
                    # Kp-DeltaV maps #
# =============================================================================
    
    def __plot_night(self, kpv_list, ax, title=None):
        '''plot a single night (as a column) of GIANO kpv maps (for pos A and B)'''
        
     # plot each night
        [kpv_list[i].plot(ax=ax[i], **self.args) for i in range(2)]
        
        # plot merged
        kpv_12 = self.kpv.merge_kpvs(kpv_list)
        obj = kpv_12.plot(ax=ax[2], label='Both nights', **self.args) 
        print('---Peak at ({:.1f}, {:.1f}) km/s with SNR = {:.1f}---'.format(*kpv_12.snr_max()))
        kpv_12.plot_slice(ax=ax[3], **self.args)
        for k in range(3):
            ax[k].text(s=self.labels[k], x=0.05, y=0.85, color='white', fontsize=19, transform=ax[k].transAxes)
            ax[k].set(xticks=[], xlabel='')
            ax[k].legend()
        if title is not None: ax[0].set_title(title)
            
        return obj
    
    def kpv_maps(self, kpv_list, instrument='HARPSN', vmin=None, vmax=None,
                 peak=None, title=None):
        
        self.kpv = kpv_list[0]
        self.kpv_list = kpv_list
        
        vmin = vmin or -4
        vmax = vmax or 8.
        # peak = peak or (0.8, 194.7)
        
        self.args = {'vmin':vmin, 'vmax':vmax, 'peak':peak}
    
        if instrument == 'GIANO':
            
            cols = int(len(kpv_list) / 2) + 1
            fig, ax = plt.subplots(4, cols,figsize=(4*cols,12))
            
            # plot each individual Kp-map
            col = [0,0,1,1]
            for i in range(4):
                row = i % 2
                kpv_list[i].plot(ax=ax[row,col[i]], **self.args)
            
            # merge positions (for both nights)
            kpv_A = self.kpv.merge_kpvs([kpv_list[0], kpv_list[2]])
            kpv_B = self.kpv.merge_kpvs([kpv_list[1], kpv_list[3]])
            [k.plot(ax=ax[i,2], **self.args) for i,k in enumerate([kpv_A, kpv_B])]
            
            # merge nights (for each position)
            kpv_1 = self.kpv.merge_kpvs(kpv_list[:2])
            kpv_2 = self.kpv.merge_kpvs(kpv_list[2:])
            [k.plot(ax=ax[2,i], **self.args) for i,k in enumerate([kpv_1, kpv_2])]
            
            # merge all
            kpv_12 = self.kpv.merge_kpvs(kpv_list) 
            obj = kpv_12.plot(ax=ax[2,2], **self.args)
            
            nights = ['1','2','1+2']
            labels = ['A','B','AB']
            for i,k in enumerate([kpv_1, kpv_2, kpv_12]):
                ax[0,i].set_title('Night {:}'.format(nights[i]), fontsize=14)
                k.get_slice(ax=ax[3,i], **self.args)
                for j in range(3):
                    ax[j,i].text(s=labels[j], x=0.05, y=0.85, color='white', fontsize=19, transform=ax[j,i].transAxes)
                    ax[j,i].set(xticks=[], xlabel='', ylabel='')
                    
            [axx.set(yticklabels='') for axx in ax[:,1:].flatten()]
            [axx.legend() for axx in ax.flatten()]

            
            # fig.subplots_adjust(hspace=0.15)
            # self.labels = ['1','2','1+2']
            # self.labels = ['A','B','AB']
            # titles = ['Night 1', 'Night 2', 'Night 1+2']
            

            # remove labels from yaxis
            # [axe.set(xticks=[], xlabel='', ylabel='') for axe in ax[:,1:].flatten()]
            cbar_ax = fig.add_axes([0.81, 0.40, 0.020, 0.4])

        
        
        elif instrument == 'HARPSN':
             # fig, ax = plt.subplots(4,figsize=(5,12))
             fig, ax = plt.subplots(2,3,figsize=(14,6))
             labels = ['1','2','1+2']
            
             args = {'vmin':vmin, 'vmax':vmax, 'peak':peak}

             kpv_12 = self.kpv.merge_kpvs(kpv_list)
             for i,k in enumerate([*self.kpv_list, kpv_12]):
                 obj = k.plot(ax=ax[0,i], **args)
                 k.get_slice(axis=0, ax=ax[1,i], **args)
                 
             
             for k in range(3):
                ax[0,k].text(s=labels[k], x=0.05, y=0.85, color='white', fontsize=19, transform=ax[0,k].transAxes)
                ax[0,k].set(xticks=[], xlabel='')
                [ax[i,k].legend(handlelength=0.75, fontsize=11) for i in range(2)]
             if title is not None: ax[0,0].set_title(title)
             cbar_ax = fig.add_axes([0.81, 0.52, 0.02, 0.35])

             [ax[j,k].set(yticks=[], ylabel='') for j in range(0,2) for k in range(1,3)]
        fig.subplots_adjust(right=0.8, hspace=0.03, wspace=0.1)
        fig.colorbar(obj, cax=cbar_ax)   
       
        # plt.show()
        if self.outname is not None:
            fig.savefig(self.outname, dpi=300, bbox_inches='tight', facecolor='white')
        return None
    
    def create_gif(self, prefix):
        import os
        os.system('gifski --fps 1 --width 1080 -o {0:}.gif {0:}-*.png'.format(prefix))
        return None
        