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
        
        if self.ax is None:
            self.fig, self.ax = plt.subplots(self.rows, self.cols, figsize=self.figsize)

    
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