# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:41:23 2017

@author: Alessandro

MODEL SELECTION CLASS

"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier,  AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import  KernelPCA, PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np



class model_prediction():

    def __init__(self, df1, df2):

        self.df = df1
        self.df_test = df2

        #Set Parameters

        #=====================================================================

        #Parameter that sets the return threshold
        self.ret_size = 1

        #Parameter that sets the number of lags to look into the past
        self.Max_lag = 20

        #Number of lags after the present
        self.Future_Lag = 0

        #Number of periods for Gauss Indicator
        self.gauss_p = 10

        #Kernel PCA n_components
        self.n_comp = 15
        
        #Random State for PCA
        self.rs = 29

        #=====================================================================

    def prepare_train(self):



        #Start Building dataframe of features

        #COMPUTE RSI
        delta = self.df['Ask_Last'] -self.df['Ask_Last'].shift(1)

        #Get only positive gains
        delta_plus = (delta*(delta > 0)).rolling(10).sum()
        #Count how many in moving window are positive

        delta_minus = -(delta*(delta < 0)).rolling(10).sum()

        average_gain = delta_plus/10
        average_loss = delta_minus/10

        RS = average_gain/average_loss

        self.df['Hour'] =self. df['Group_ID'].apply(lambda x: round(x-round(x,-4),-2)/100)

        ft = pd.DataFrame((self.df['Ask_Last'] + self.df['Bid_Last'])/2, columns = ['Level'])
        ft['Return'] = ft['Level']/ft['Level'].shift(1) - 1



        #Add Future lag
        for i in range(1,self.Future_Lag + 1):
            ft['Return-f-' + str(i)] = ft['Return'].shift(-i)


        for i in range(1,self.Max_lag):

            ft['Return-' + str(i)] = ft['Return'].shift(i)
            ft['Volume-' + str(i)] = self.df['Volume'].shift(i)
            ft['Volatility-' + str(i)] = self.df['Volatility'].shift(i)
            ft['Range-'+ str(i)] = (self.df['Max'] - self.df['Min']).shift(i)



        ft['RSI'] = 100-100/(1+RS.shift(1))

        #Add Gaussian measure shifted by one, otherwise we are cheating
        ft['Gauss'] = ((ft['Return'] - ft['Return'].rolling(self.gauss_p).mean())/ \
                          ft['Return'].rolling(self.gauss_p).std()).shift(1)
        ft['Vol-Ret'] = (ft['Return-1']*ft['Volume-1']).rolling(5).mean()


        ft['Cum-Ret'] = (1+ft['Return-1'])*(1+ft['Return-2'])*(1+ft['Return-3']) \
                            *(1+ft['Return-4'])*(1+ft['Return-5'])
        ft['Trend'] = ft['Return-1'].rolling(10).mean() \
                          - ft['Return-1'].rolling(20).mean()

        #Interaction terms
        ft['Interaction-1'] = ft['Volume-1']*ft['Volatility-1']
        ft['Interaction-2'] = ft['Volume-1']*ft['Return-1']

        ft['Hour'] = self.df['Hour']

        ft['Trend-2'] = ft['Return-1'].rolling(5).mean() \
                     - ft['Return-1'].rolling(10).mean()

        ft['Derivative'] = ft['RSI'].rolling(5).mean() \
                           - ft['RSI'].shift(1).rolling(5).mean()

        ft.dropna(how = 'any', inplace = True)



        #Average in sample returns
        self.avg_ret = np.mean(abs(ft['Return'])) 
        
        self.avg_ret_p = sum(ft['Return']*(ft['Return'] > 0))/sum(ft['Return'] > 0)
        self.avg_ret_m = - sum(ft['Return']*(ft['Return'] < 0))/sum(ft['Return'] < 0)
        
  
        
        #Separate X and y
        y = 1*(ft.iloc[:,1:self.Future_Lag + 2].sum(axis = 1) > self.avg_ret_p*self.ret_size) \
            - 1*(ft.iloc[:,1:self.Future_Lag + 2].sum(axis = 1) < -self.avg_ret_m*self.ret_size)
        
        X = ft.iloc[:,self.Future_Lag + 2:]


        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        #Initialize Kernel PCA for dim reduction
        kpca = KernelPCA(kernel="rbf", n_components = self.n_comp, fit_inverse_transform=True)
        X = pd.DataFrame(kpca.fit_transform(X))
        
        #ONLY FOR EURUSD
        #std_pca = PCA(n_components = 15, random_state = self.rs)
        #X = pd.DataFrame(std_pca.fit_transform(X))

        self.X_train = X
        self.y_train = y

        return X,y


    def prepare_test(self):


        #Start Building dataframe of features

        #COMPUTE RSI
        delta = self.df_test['Ask_Last'] -self.df_test['Ask_Last'].shift(1)

        #Get only positive gains
        delta_plus = (delta*(delta > 0)).rolling(10).sum()
        #Count how many in moving window are positive

        delta_minus = -(delta*(delta < 0)).rolling(10).sum()

        average_gain = delta_plus/10
        average_loss = delta_minus/10

        RS = average_gain/average_loss

        self.df_test['Hour'] = self.df_test['Group_ID'].apply(lambda x: \
                               round(x-round(x,-4),-2)/100)

        ft = pd.DataFrame((self.df_test['Ask_Last'] \
             + self.df_test['Bid_Last'])/2, columns = ['Level'])
        
        ft['Return'] = ft['Level']/ft['Level'].shift(1) - 1



        #Add Future lag
        for i in range(1,self.Future_Lag + 1):
            ft['Return-f-' + str(i)] = ft['Return'].shift(-i)


        for i in range(1,self.Max_lag):

            ft['Return-' + str(i)] = ft['Return'].shift(i)
            ft['Volume-' + str(i)] = self.df_test['Volume'].shift(i)
            ft['Volatility-' + str(i)] = self.df_test['Volatility'].shift(i)
            ft['Range-'+ str(i)] = (self.df_test['Max'] - self.df_test['Min']).shift(i)



        ft['RSI'] = 100-100/(1+RS.shift(1))

        #Add Gaussian measure shifted by one, otherwise we are cheating
        ft['Gauss'] = ((ft['Return'] - ft['Return'].rolling(self.gauss_p).mean())/ \
                          ft['Return'].rolling(self.gauss_p).std()).shift(1)
        
        ft['Vol-Ret'] = (ft['Return-1']*ft['Volume-1']).rolling(5).mean()


        ft['Cum-Ret'] = (1+ft['Return-1'])*(1+ft['Return-2'])*(1+ft['Return-3']) \
                            *(1+ft['Return-4'])*(1+ft['Return-5'])
        ft['Trend'] = ft['Return-1'].rolling(10).mean() \
                          - ft['Return-1'].rolling(20).mean()

        #Interaction terms
        ft['Interaction-1'] = ft['Volume-1']*ft['Volatility-1']
        ft['Interaction-2'] = ft['Volume-1']*ft['Return-1']

        ft['Hour'] = self.df_test['Hour']

        ft['Trend-2'] = ft['Return-1'].rolling(5).mean() \
                     - ft['Return-1'].rolling(10).mean()

        ft['Derivative'] = ft['RSI'].rolling(5).mean() - ft['RSI'].shift(1).rolling(5).mean()


        ft.dropna(how = 'any', inplace = True)

        #Separate train and Test
        y = 1*(ft.iloc[:,1:self.Future_Lag + 2].sum(axis = 1) > self.avg_ret_p*self.ret_size) \
            - 1*(ft.iloc[:,1:self.Future_Lag + 2].sum(axis = 1) < - self.avg_ret_m*self.ret_size)
        
        X = ft.iloc[:,self.Future_Lag + 2:]

        # Normalize before reducing dimensions for efficientcy and speed
        # Scaling factors are extracted from train set
        X = self.scaler.transform(X)

        #Initialize Kernel PCA for dim reduction
        kpca = KernelPCA(kernel="rbf", n_components = self.n_comp, \
                         fit_inverse_transform=True)
        X = pd.DataFrame(kpca.fit_transform(X))
        
        # Only for EURUSD
        #std_pca = PCA(n_components = 15, random_state = self.rs)
        #X = pd.DataFrame(std_pca.fit_transform(X))

        self.X_test = X
        self.y_test = y

        #For PnL Purposes
        self.out = ft['Return']
        
        return X,y


    def evaluate(self):


        #Rename vars 
        X_train = self.X_train
        y_train = self.y_train

        X_test = self.X_test
        y_test = self.y_test

        #Build Models

        #######################################################################
        # Model 1

        log_clf = LogisticRegression()

        parameters = {'C': np.logspace(-2,4,20)}

        #Initialize GridSearch and fit it
        opt = GridSearchCV(log_clf, parameters, cv = 5)
        opt.fit(X_train, y_train)

        #Recall with optimized params
        log_clf = LogisticRegression(C = opt.best_params_['C'])


        ######################################################################
        # Model 2

        ada_clf = AdaBoostClassifier()


        #################################################################
        # Model 3

        parameters = {'gamma': np.logspace(-2,2,20)}

        svm_clf =  SVC(kernel = 'rbf', probability = True)
        
        #Initialize GridSearch and fit it
        opt = GridSearchCV(svm_clf, parameters, cv = 3)
        opt.fit(X_train, y_train)

        #Recall with optimized params
        svm_clf =  SVC(kernel = 'rbf', probability = True, \
                       gamma = opt.best_params_['gamma'])

        #################################################################
        # Model 4

        qda_clf =  QuadraticDiscriminantAnalysis()


        #################################################################
        # Voting Classifier

        classifiers = [log_clf, ada_clf, svm_clf, qda_clf]
        names = ['Log', 'Ada', 'SVM', 'QDA']

        Voting_clf =  VotingClassifier(estimators=list(zip(names, classifiers)), voting='hard')
        Voting_clf.fit(X_train, y_train)
        
        #Compute Out Of Sample Accuracy
        test_score = Voting_clf.score(X_test, y_test)

        y_cap = Voting_clf.predict(X_test)

        #Compute simple PnL without commissions
        PnL = sum(y_cap*self.out)

        n_ops = sum(abs(y_cap))
        
        
        #Compute PnL with commissions
        
        #To be put at the beginning to avoid issues with roll
        temp = y_cap[0]
        
        pos = y_cap - np.roll(y_cap,1)
        pos[0] = temp
        
        #Make sure that all positions are closed EOM
        pos[-1] = -sum(pos[:-1]) 
            
        #Indexes used when trading    
        ix = self.out.index
        
        long_entries = -pos*(pos > 0)*self.df_test['Ask_First'][ix]
        short_entries = -pos*(pos < 0)*self.df_test['Bid_First'][ix]
        
        net_PnL = sum(long_entries + short_entries)
        
        #Save for convenience
        self.y_cap = y_cap
        self.pos = pos
        self.ix = ix
        
        return test_score, PnL, n_ops, net_PnL
    
    
    #Class used in development, useful to output stuff in case of need
    def helper(self):
        
        return 0