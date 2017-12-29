# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 09:44:27 2017

@author: teogo
"""

import numpy as np
import dask.dataframe as dd
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (18, 15),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

w_dir = 'D:/BIG DATA'
directory_load = 'Merged Files'
directory_save = 'Strategy Reversal'

os.chdir(w_dir)


ccys = ['EURUSD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURAUD']

df = dd.read_hdf(directory_load+'/'+ccys[1]+'-'+str(2014)+'.h5',ccys[1]+str(2014)+'05')
plt.figure()
plt.plot(df.compute())

#%%      
with open(directory_save+'/Report.txt', 'r') as fout:
    file = fout.read()

#%%
lines = file.split()

read_ccy = False
read_PL =  False

performance_usd = []
performance_chf = []
performance_gbp = []
performance_jpy = []
performance_aud = []

for elem in lines:
    if not read_ccy:
        if elem=='Ccy:':
            read_ccy = True
    else:
        ccy = elem
        read_ccy = False
        
    if not read_PL:
        if elem=='P&L:':
            read_PL = True
    else:
        PL = float(elem)
        read_PL = False
        if ccy == ccys[0]:
            performance_usd.append(PL)
        if ccy == ccys[1]:
            performance_chf.append(PL)
        if ccy == ccys[2]:
            performance_gbp.append(PL)
        if ccy == ccys[3]:
            performance_jpy.append(PL)
        if ccy == ccys[4]:
            performance_aud.append(PL)

        
#%%
plt.figure()
plt.plot(range(2004,2017),np.cumsum(np.array([performance_usd,performance_chf,performance_gbp,performance_aud]).T,axis=0))
plt.legend(['EURUSD', 'EURCHF', 'EURGBP', 'EURAUD'])

        
#%% 
directory_save = 'Strategy Reversal Testing'
     
with open(directory_save+'/Report.txt', 'r') as fout:
    file = fout.read()


lines = file.split()

read_ccy = False
read_PL =  False

performance_usd = []
performance_chf = []
performance_gbp = []
performance_jpy = []
performance_aud = []

for elem in lines:
    if not read_ccy:
        if elem=='Ccy:':
            read_ccy = True
    else:
        ccy = elem
        read_ccy = False
        
    if not read_PL:
        if elem=='P&L:':
            read_PL = True
    else:
        PL = float(elem)
        read_PL = False
        if ccy == ccys[0]:
            performance_usd.append(PL)
        if ccy == ccys[1]:
            performance_chf.append(PL)
        if ccy == ccys[2]:
            performance_gbp.append(PL)
        if ccy == ccys[3]:
            performance_jpy.append(PL)
        if ccy == ccys[4]:
            performance_aud.append(PL)

        
#%%
plt.figure()
plt.plot(range(1,12),np.cumsum(np.array([performance_usd,performance_chf,performance_gbp,performance_aud]).T,axis=0))
plt.legend(['EURUSD', 'EURCHF', 'EURGBP', 'EURAUD'])

plt.figure()
plt.plot(range(1,12),np.array([performance_usd,performance_chf,performance_gbp,performance_aud]).T)
plt.legend(['EURUSD', 'EURCHF', 'EURGBP', 'EURAUD'])

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
