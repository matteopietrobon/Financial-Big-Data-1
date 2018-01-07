import pandas as pd
import numpy as np
import os
import math
import time



nomi = ['t_stamp', 'Bid', 'Ask', 'Group_ID']

currs = ['EURAUD']#, 'EURCHF', 'EURGBP', 'EURJPY', 'EURUSD']

years = ['2014', '2015', '2016', '2017']


months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']



#===============================================================================

# Number of minutes to group by
time_group = 30

#===============================================================================

fPath = os.path.dirname(os.path.realpath(__file__))
savePath = fPath + '\Grouped_Data'

for curr in currs:
    for year in years:
        for month in months:

            saveName = '\\' + curr + '_' + str(year) + '_' + month + '_' + str(time_group) + '.csv'
            fCall = '\DAT_ASCII_' + curr + '_T_' + str(year) + month + '.csv'
            fName = fPath + '\Data' + fCall

            try:
                df = pd.read_csv(fName, names = nomi)
            except(FileNotFoundError):

                print(fCall, 'Not Found')
                pass

            print("\nConverting", fCall[1:])

            inizio = time.time()

            start_time = int(df.iloc[0,0][9:13])

            n_group = 0

            #Create groupby label, that is day-hour-minute_group
            df['Group_ID'] = df['t_stamp'].apply(lambda x: int(x[6:8])*10000 + \
                             int(x[9:11])*100 + math.floor(int(x[11:13])/time_group))


            #Faster if only stricly needed operatins are computed
            dg_1 = df.groupby('Group_ID').agg({'Bid': 'first', 'Ask': 'first'})
            dg_2 = df.groupby('Group_ID').agg({'Bid': 'last', 'Ask': 'last'})
            dg_3 = df.groupby('Group_ID').agg({'Bid': 'count', 'Ask': 'std'})
            dg_4 = df.groupby('Group_ID').agg({'Bid': 'min', 'Ask': 'max'})

            df_final = pd.concat([dg_1,dg_2, dg_3, dg_4], axis = 1)
            df_final.columns = ['Bid_First', 'Ask_First', 'Bid_Last', \
                                'Ask_Last', 'Volume', 'Volatility', 'Min', 'Max']


            #df_final.to_excel(fPath + '\Prova a guardare.xlsx')
            df_final.to_csv(savePath + saveName)
            print('Done in', round(time.time() - inizio,4), 'secs, File Saved')
