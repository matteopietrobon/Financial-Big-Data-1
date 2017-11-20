import pandas as pd
import numpy as np
import os


w_dir = 'D:/OneDrive/Z - Financial Big Data/project1/Financial-Big-Data-1/Toy Data'
            
directory = w_dir+'/Toy Data'

os.chdir(w_dir)


ccys = ['EURUSD', 'EURCHF']
year = '2003'
month = '1'

#store = pd.HDFStore('FX-'+year+month+'.h5')

x = ccys[0]
file_name = 'DAT_ASCII_'+ x +'_T_'+ year+month.zfill(2)+'.csv'
df = pd.read_csv(file_name, names = ['Bid-'+x, 'Ask-'+x, 'cancella']).iloc[:,:2]
df['traded-'+ x] = np.ones(df.shape[0])

for x in ccys[1:]:
    
    file_name = 'DAT_ASCII_'+ x +'_T_'+ year+month.zfill(2)+'.csv'
    df_temp = pd.read_csv(file_name, names = ['Bid-'+x, 'Ask-'+x, 'cancella']).iloc[:,:2]
    df_temp['traded-'+x] = np.ones(df_temp.shape[0])
    
    df = df.join(df_temp, rsuffix = '-'+x, how = 'outer')

del df_temp

#Forward fill and replace nans with zeros
use_columns = []
use_columns_2 = []

for x in ccys:
    use_columns.append('Bid-'+x)
    use_columns.append('Ask-'+x)
    
    use_columns_2.append('traded-'+x)
    
    
df[use_columns] = df[use_columns].ffill() 
df[use_columns_2] = df[use_columns_2].fillna(0) 

df = df.dropna(how='any')

#%%
#Change index to datetime

df['datetime'] = pd.to_datetime(df.index.astype(str)+'000',format="%Y%m%d %H%M%S%f")

df = df.sort_values('datetime')

#%%

df_final = pd.DataFrame(df.iloc[:,:-1])
df_final.index = df['datetime']



df_final.to_hdf('FX-'+year+month.zfill(2)+'.h5', 'FX'+year+month.zfill(2), mode='w')

#pd.read_hdf('foo', 'bar')

#df_final.to_csv('FX-'+year+month+'.csv')
#df_final.to_json('FX-'+year+month.zfill(2)+'.json')