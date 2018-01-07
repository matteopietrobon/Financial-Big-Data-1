import os
import pandas as pd
import model_class
import matplotlib.pyplot as plt
import numpy as np


# This module reads the files four by four, picks the first three as training set
# And the last one as out of sample test set.

curr = 'EURAUD'

#Start 6 months in advance to avoid trading on model_selection Data
start = 6

fPath = os.path.dirname(os.path.realpath(__file__)) + '\Grouped_Data\\'

#Select files based on currency pair
files = [x for x in os.listdir(fPath) if x[0:6] == curr]

#Names for x_axis i nfinal plot
x_ticks = [x[7:14] for x in files[start:]]

#Stores PnLs
gains = []
net_gains = []

#start trading from the third file
for i in range(start,len(files)):


    #OPEN TEST MONTH
    test = pd.read_csv(fPath + files[i])

    #OPEN AND CONCATENATE TRAINING SETS
    df1 = pd.read_csv(fPath + files[i-1])
    df2 = pd.read_csv(fPath + files[i-2])
    df3 = pd.read_csv(fPath + files[i-3])
    
    train = pd.concat([df1, df2, df3], ignore_index = True)


    #NOW CALL CLASS THAT IMPLEMENTS MODEL SELECTION

    print('\n\n======================', files[i][7:14] ,'===========================')


    predictor = model_class.model_prediction(train, test)

    [X_train, y_train] = predictor.prepare_train()
    [X_test, y_test] = predictor.prepare_test()

    [score, PnL, N, net_PnL] = predictor.evaluate()
    
    gains.append(PnL)
    net_gains.append(net_PnL)

    print('\nAccuracy:', round(score,4), '- Number of Trades:', N)
    print('Raw Pnl:', round(PnL,4), '- Net PnL:', round(net_PnL,4))

print('\n\n====================== Summary ============================')
print('\n PnL:', round(sum(gains),4), 'Sharpe ratio:', \
      round(np.mean(gains)/np.std(gains),4))


#%%
line = np.cumsum(np.array(gains))
line_2 = np.cumsum(np.array(net_gains))

#Plot Backtesting results
plt.figure()
plt.plot(line)
plt.plot(line_2)
plt.plot(np.zeros(len(line)), 'r--')
plt.title('Equity Lines ' + curr)
plt.legend(['Raw', 'Net', 'Zero'])

plt.xticks(np.arange(len(x_ticks[0::4]))*4, x_ticks[0::4], rotation = 45)
plt.show()
