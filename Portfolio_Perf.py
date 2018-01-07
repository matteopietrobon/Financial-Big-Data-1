'''

Portfolio performance of currency strategies

'''

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


dataPath = os.path.dirname(os.path.realpath(__file__)) + '\Results\\'

#Locate Backtesting files
files = [x for x in os.listdir(dataPath) if x[7:] == 'Backtest.xlsx']


df = pd.DataFrame()
#Open and merge files

for file in files:

    temp = pd.read_excel(dataPath + file)
    
    df = pd.concat([df, temp], axis = 1)
    
    
#For plotting purposes
x_ticks = df.iloc[:,0].values
                 
#Dataframe of net PnLs
take_cols = [1,4,7,10]

perf_df = df.iloc[:,take_cols]

cs = perf_df.columns
#%%
# EQUALLY WEIGHTED

EW = perf_df.mean(axis = 1)

PnL = np.sum(EW)
sharpe = np.mean(EW)/np.std(EW)

print('\n================== Equally Weighted Portfolio =========================\n')
print('PnL:', round(PnL,4), '- Sharpe Ratio:', round(sharpe,4))

plt.figure()
plt.plot(np.cumsum(EW))

plt.title('Portfolio Performances')

plt.xticks(np.arange(len(x_ticks[0::4]))*4, x_ticks[0::4], rotation = 45)
plt.show()


#%%
# Risk Parity (6Mo variance)


stds = perf_df.rolling(6).std()
      
ix = stds.dropna(how = 'all').index

#Start as EW
RP = perf_df.mean(axis = 1)


#From first to last available index
for i in range(ix[0], ix[-1]):
    
    month_var = stds.iloc[i,:].values
    
    sum_inv = 1/sum(1/month_var)
    
    w = sum_inv/month_var;
    
    RP[i] = np.dot(w, perf_df.iloc[i,:].values)
    
    
    
    
PnL = np.sum(RP)
sharpe = np.mean(RP)/np.std(RP)

print('\n================== Risk-Parity Portfolio =========================\n')
print('PnL:', round(PnL,4), '- Sharpe Ratio:', round(sharpe,4))


plt.plot(np.cumsum(RP))
plt.plot(np.zeros(len(EW)), '--r')
plt.legend(['Equally Weighted', 'Risk-Parity'])
