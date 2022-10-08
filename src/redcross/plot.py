#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:53:25 2022

@author: dario
"""
import numpy as np
import matplotlib.pyplot as plt

class Plot:
    
    def __init__(self, ax=None, rows=1, cols=1, figsize=(12,3)):
        self.ax = ax
        self.rows = rows
        self.cols = cols
        self.figsize = figsize
        
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
        
    
    def data_orders(self, dc, orders=[]):
        
        cmap = plt.cm.get_cmap('viridis', dc.nOrders)
        colors = np.array([cmap(x) for x in range(dc.nOrders)])
        
        dc.flux_err[dc.flux_err==0.0] = np.nan
        snr_order = np.nanmean(dc.flux / dc.flux_err, axis=(1,2))
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
        kpv_12.plot_1D(ax=ax[3], **self.args)
        for k in range(3):
            ax[k].text(s=self.labels[k], x=0.05, y=0.85, color='white', fontsize=19, transform=ax[k].transAxes)
            ax[k].set(xticks=[], xlabel='')
            ax[k].legend()
        if title is not None: ax[0].set_title(title)
            
        return obj
    
    def kpv_maps(self, kpv_list, vmin=None, vmax=None, peak=None, outname=None, title=None):
        
        self.kpv = kpv_list[0]
        self.kpv_list = kpv_list
        cols = int(len(kpv_list) / 2) 
        if cols > 1: # GIANO mode
            kpv_A = self.kpv.merge_kpvs([kpv_list[0], kpv_list[2]])
            kpv_B = self.kpv.merge_kpvs([kpv_list[1], kpv_list[3]])
            self.kpv_list = np.append(self.kpv_list, [kpv_A, kpv_B])
            cols+=1
        
        fig, ax = plt.subplots(4, cols,figsize=(4*cols,12))
        # fig.subplots_adjust(hspace=0.15)
        # self.labels = ['1','2','1+2']
        self.labels = ['A','B','AB']
        
        vmin = vmin or -4
        vmax = vmax or 8.
        peak = peak or (0.8, 194.7)
        
        self.args = {'vmin':vmin, 'vmax':vmax, 'peak':peak}
        # _ = [axe.legend(loc='lower right', frameon=True, fontsize=9) for axe in ax.flatten()]
        # plot each night
        # [kpv_list[i].plot(ax=ax[i], v_range=[vmin, vmax], peak=self.peak) for i in range(2)]
        
        # # plot merged
        # kpv_12 = self.kpv.merge_kpvs(kpv_list)
        # obj = kpv_12.plot(ax=ax[2], v_range=[vmin, vmax], peak=self.peak, label='Both nights') 
        # print('---Peak at ({:.1f}, {:.1f}) km/s with SNR = {:.1f}---'.format(*kpv_12.snr_max()))
        
       
        # kpv_12.plot_1D(ax=ax[3], peak=self.peak, v_range=(vmin, vmax))
              
        titles = ['Night 1', 'Night 2', 'Night 1+2']
        for i,j in enumerate(range(0,cols+2,2)):
            obj = self.__plot_night(self.kpv_list[j:j+2], ax[:,i], title=titles[i])
        
        fig.subplots_adjust(right=0.8, hspace=0.03)
        # cbar_ax = fig.add_axes([0.85, 0.40, 0.04, 0.4])
        cbar_ax = fig.add_axes([0.81, 0.40, 0.020, 0.4])
        fig.colorbar(obj, cax=cbar_ax)   
        # if title is not None:
        #     ax[0].set_title(title) 
        # remove labels from yaxis
        [axe.set(xticks=[], xlabel='', ylabel='') for axe in ax[:,1:].flatten()]

        
        # plt.show()
        if outname is not None:
            fig.savefig(outname, dpi=200, bbox_inches='tight', facecolor='white')
        return None
        