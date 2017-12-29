
import pandas as pd
import os

w_dir = 'D:/BIG DATA'
directory = 'Merged Files'
os.chdir(w_dir)
if not os.path.exists(directory):
    os.makedirs(directory)

ccys = ['EURUSD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURAUD']

for ccy in ccys:
    year_int = 2017 
    year = str(year_int)
    for month_int in range(1,12): 
        
        if year_int<2009:
            cols = ['Ask','Bid','erase']
        elif year_int==2009 and month_int <6:
            cols = ['Ask','Bid','erase']
        else:
            cols = ['Bid','Ask','erase']

        month = str(month_int)
                
        file_name = 'DAT_ASCII_'+ ccy +'_T_'+ year+month.zfill(2)+'.csv'
        df = pd.read_csv(file_name, names = cols)
        if df['Ask'].values[0]<df['Bid'].values[0]:
            print('ERROOOOOOOOR')
            
        df = df.drop('erase', axis=1)
    
        df = df.dropna(how='any')
    
        #Change index to datetime
        df['datetime'] = pd.to_datetime(df.index.astype(str)+'000',format="%Y%m%d %H%M%S%f")
        
        df = df.sort_values('datetime')
        
        df.index = df['datetime']
        df = df.drop('datetime', axis=1)
    
        df.to_hdf(directory+'/'+ccy+'-'+year+'.h5', ccy+year+month.zfill(2),format='table')

        

