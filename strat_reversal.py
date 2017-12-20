import numpy as np
import dask.dataframe as dd
import os

w_dir = 'D:/BIG DATA'
directory_load = 'Merged Files'
directory_save = 'Strategy Reversal'

os.chdir(w_dir)
if not os.path.exists(directory_save):
    os.makedirs(directory_save)
    
    



ccys = ['EURUSD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURAUD']


class strat_reversal(object):
    def __init__(self, w_dir,directory_load,directory_save,ccy,year,params):
        
        self.df = dd.read_hdf(directory_load+'/'+ccy+'-'+str(year)+'.h5','*')
        self.train_df = dd.read_hdf(directory_load+'/'+ccy+'-'+str(year-1)+'.h5', ccy+str(year-1)+'12')
        self.w_dir = w_dir
        self.directory_load = directory_load
        self.directory_save = directory_save
        self.ccy = ccy
        self.year = year
        self.params = params
        self.save_CF = False
        self.prepared_df = False
        
    def prepare(self):
        
        long_mean = int(np.maximum(1,params[0]))
        short_mean = int(np.maximum(1,params[1]))
        
        self.df['spread']     = self.df['Ask']-self.df['Bid']
        self.df['long_mean']  = self.df['Bid'].rolling(long_mean, min_periods=int(long_mean/10)).mean()
        self.df['short_mean'] = self.df['Bid'].rolling(short_mean).mean()
        
        self.train_df['spread']     = self.train_df['Ask']-self.train_df['Bid']
        self.train_df['long_mean']  = self.train_df['Bid'].rolling(long_mean, min_periods=int(long_mean/10)).mean()
        self.train_df['short_mean'] = self.train_df['Bid'].rolling(short_mean).mean()
        
        
    def execute(self,df,params):
        
        hold_per = int(np.maximum(1,params[2]))
        multiplier = int(np.maximum(0,params[3]))
        
        if not self.prepared_df:
            self.prepare()
            self.prepared_df = True
        
        # compute necessary statistics: use median to reduce instability
        df['triggered_u']  = df['short_mean']-df['long_mean'] >  df['spread']*multiplier
        df['triggered_d']  = df['short_mean']-df['long_mean'] < -df['spread']*multiplier
        # take the rolling max to avoid overlapping and net out positions
        df['open_pos']   = df['triggered_d'].rolling(hold_per).max()-\
                             df['triggered_u'].rolling(hold_per).max()
        
        # see when to open/close positions
        df['change_pos']   = df['open_pos'].diff().shift(1) # shift 1 because we see opportunity and send order
        
        
        df['CF'] = (df['change_pos']<0).mul(df['Bid']) - (df['change_pos']>0).mul(df['Ask'])
        
        if self.save_CF:
            df['CF'].to_hdf(directory_save+'/Strat_Reversal'+ccy+'-'+str(year)+'-*.h5','Strat_Reversal'+ccy+str(year),format='table')#, ccy+year+month.zfill(2)
            with open(directory_save+'/Report.txt', 'a') as fout:
                fout.write('\n'.join([
                        '\n\n=======================================================',
                        '            Ccy: '+self.ccy+'           Year: '+str(self.year),
                        '      Long Mean: '+str(self.params[0])+'       Short Mean: '+str(self.params[1]),
                        '      Hold Per : '+str(self.params[2])+'       Multiplier: '+str(self.params[3]),
                        '            P&L: '+str(np.round(df['CF'].sum().compute(),3)).zfill(6)+'  Operations: '+str(df['change_pos'].abs().sum().compute())
                        ]))
    
        return df['CF'].sum().compute()
        
    def train(self,m_max):
        
        # We only tune the multiplier, otherwise it takes too long and the difference is negligible
        curr_max = -10
        curr_best = 0
        for mult in range(3,m_max+1):
            params_temp = self.params.copy()
            params_temp[3] = mult
            
            P_L = self.execute(self.train_df,params_temp)
            if P_L>=curr_max:
                curr_max = P_L
                curr_best = mult
                
        self.params[3] = curr_best
           
        

import time  
              
for ccy in ccys:
    for year in range(2004,2017): 
        # long mean, short mean, holding period, multiplier
        params = np.array([500.0,3.0,80.0,8.0])
        
        start_time = time.time()
        strat = strat_reversal(w_dir,directory_load,directory_save,ccy,year,params)
        strat.train(20)
        strat.save_CF = True
        strat.execute(strat.df,strat.params)
        elapsed = time.time()-start_time
        with open(directory_save+'/Report.txt', 'a') as fout:
            fout.write('\n   Time Elapsed: '+str(np.round(elapsed,2)))
        
        
        
