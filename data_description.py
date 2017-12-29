import numpy as np
import dask.dataframe as dd
import os

w_dir = 'D:/BIG DATA'
directory_load = 'Merged Files'

os.chdir(w_dir)


ccys = ['EURUSD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURAUD']


import time  
   
total_start_time = time.time()
descriptions =[]     

for ccy in ccys:
    for year in range(2003,2017): 
        
        
        start_time = time.time()
        
        df = dd.read_hdf(directory_load+'/'+ccy+'-'+str(year)+'.h5','*')
        
        with open(directory_load+'/'+'Data Description.txt', 'a') as fout:
                description = df.describe().compute()
                descriptions.append([ccy,year,description])
                fout.write('\n'.join([
                        '\n\n=======================================================',
                        'Ccy: '+ccy+'           Year: '+str(year),
                        str(description)]))
        
        elapsed = time.time()-start_time
        with open(directory_load+'/'+'Data Description.txt', 'a') as fout:
            fout.write('\nTime Elapsed: '+str(np.round(elapsed,2)))
        
total_elapsed = time.time()-total_start_time
print(total_elapsed)     
#%%

counts = [[description[2].loc['count','Bid'] for description in descriptions if description[0]==ccy] for ccy in ccys]

min_mean_max_usd = [description[2].loc[['min','mean','max'],'Bid'] for description in descriptions if description[0]==ccys[0]]
min_mean_max_chf = [description[2].loc[['min','mean','max'],'Bid'] for description in descriptions if description[0]==ccys[1]]
min_mean_max_gbp = [description[2].loc[['min','mean','max'],'Bid'] for description in descriptions if description[0]==ccys[2]]
min_mean_max_jpy = [description[2].loc[['min','mean','max'],'Bid'] for description in descriptions if description[0]==ccys[3]]
min_mean_max_aud = [description[2].loc[['min','mean','max'],'Bid'] for description in descriptions if description[0]==ccys[4]]

#%%
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

plt.figure()
plt.plot(range(2003,2017),np.array(counts).T)
#plt.title('Data Entries by Currency each Year')
plt.legend(ccys)

plt.figure()
plt.plot(range(2003,2017),np.array(min_mean_max_usd))
#plt.title('Min, Mean and Max of '+ccys[0]+' each year')
plt.legend(['Min','Mean','Max'])

plt.figure()
plt.plot(range(2003,2017),np.array(min_mean_max_chf))
#plt.title('Min, Mean and Max of '+ccys[1]+' each year')
plt.legend(['Min','Mean','Max'])

plt.figure()
plt.plot(range(2003,2017),np.array(min_mean_max_gbp))
#plt.title('Min, Mean and Max of '+ccys[2]+' each year')
plt.legend(['Min','Mean','Max'])

plt.figure()
plt.plot(range(2003,2017),np.array(min_mean_max_jpy))
#plt.title('Min, Mean and Max of '+ccys[3]+' each year')
plt.legend(['Min','Mean','Max'])

plt.figure()
plt.plot(range(2003,2017),np.array(min_mean_max_aud))
#plt.title('Min, Mean and Max of '+ccys[4]+' each year')
plt.legend(['Min','Mean','Max'])




