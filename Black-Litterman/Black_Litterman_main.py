import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rankdata
import os

def BL_parasn(momt_data,return_data,Type,value,lamda=10):
    #momt_data: Dataframe
    #return_data: Dataframe
    #Type: 1 or -1
    #value: the change of initial q
    #lamda: multiplier of Omega

    #present 
    momt = momt_data.iloc[-1,:].tolist()
    rank_momt = rankdata(momt)
    P=np.matrix(np.zeros((2,11),dtype=float))
    q=np.matrix(np.zeros(2,dtype=float)).T
    Omega=np.matrix(np.zeros((2,2),dtype=float))
    for i,elem in enumerate(rank_momt):
            if elem==1:
                P[0,i]=1
                temp1=return_data[i]
            elif elem==11:
                P[0,i]=-1
                temp11=return_data[i]
            elif elem==2:
                P[1,i]=1
                temp2=return_data[i]
            elif elem==10:
                P[1,i]=-1
                temp10=return_data[i]
    if Type==1:
        q[0]=0.06*value
        q[1]=0.03*value
    elif Type==-1:
        q[0]=-0.06*value
        q[1]=-0.03*value
    #use difference of return to calculate variance
    Omega[0,0]=(temp1-temp11).var()*lamda
    Omega[1,1]=(temp2-temp10).var()*lamda
#    print(Omega)

    
    return P,q,Omega

def BL_parasf(return_data,Type,value,lamda=10):
    #return_data: Dataframe
    #Type:1 or -1
    #value:the change of initial q
    #lamda: multiplier of Omega
    
    #future
    return_list=return_data.mean().tolist()
    rank_return=rankdata(return_list)
    P=np.matrix(np.zeros((2,11),dtype=float))
    q=np.matrix(np.zeros(2,dtype=float)).T
    Omega=np.matrix(np.zeros((2,2),dtype=float))
    for i,elem in enumerate(rank_return):
            if elem==1:
                P[0,i]=1
                rmin1=return_list[i]
                temp1=return_data.iloc[:,i]
            elif elem==11:
                P[0,i]=-1
                rmax1=return_list[i]
                temp11=return_data.iloc[:,i]
            elif elem==2:
                P[1,i]=1
                rmin2=return_list[i]
                temp2=return_data.iloc[:,i]
            elif elem==10:
                P[1,i]=-1
                rmax2=return_list[i]
                temp10=return_data.iloc[:,i]
    surpass1=rmax1-rmin1
    surpass2=rmax2-rmin2
    Omega[0,0]=(temp1-temp11).var()*lamda
    Omega[1,1]=(temp2-temp10).var()*lamda

    if Type==1:
        q[0]=surpass1+value
        q[1]=surpass2+value
    elif Type==-1:
        q[0]=-surpass1+value
        q[1]=-surpass2+value

    return P,q,Omega

def BL_backtestf(start_dates, end_dates, price, R, q_shift, Omega_shift,
                tau=0.25, xi=0.42667):
    # back_test for Black-Litterman model
    # future returns as investors' views
    BL_weights = pd.DataFrame(None, index=start_dates[1:], columns=R.columns)
    Mark_weights = pd.DataFrame(None, index=start_dates[1:], columns=R.columns)
    
    for i in range(len(start_dates)-1):
        start = start_dates[i]
        end = end_dates[i]
        next_start = start_dates[i+1]
        next_end = end_dates[i+1]
        # select training data
        mkt_cap_training = mkt_cap.loc[start:end,:]  
        R_training = R.loc[start:end,:]         
        R_test = R.loc[next_start:next_end,:]        
        C = np.matrix(R_training.cov())
        markR = np.matrix(R_test.mean())
        markC = np.matrix(R_test.cov())
        
        # set parameters (views)
        P, q, Omega = BL_parasf(R_test, 1, q_shift, Omega_shift)

        tau = 0.025
        Sigma = tau * C
        xi = 0.42667
    
        weights = pd.DataFrame(None, index=mkt_cap_training.index, columns=mkt_cap_training.columns)
        for date, row in mkt_cap_training.iterrows():
            weights.loc[date] = row/row.sum()
        weight = np.matrix(weights.iloc[-1,:]).T # market capitalization weights
        pi = 1/xi * C * weight
        
        # formulas
        Rp = pi + Sigma * P.T * (P*Sigma*P.T + Omega).I * (q-P*pi)
        Cp = C + Sigma - Sigma * P.T * (P*Sigma*P.T + Omega).I * (P*Sigma)
        opt_weight = (xi * Cp.I * Rp - 
                      np.multiply(Cp.I, (xi*np.matrix([1]*11)*Cp.I*Rp-1)/(np.matrix([1]*11)*Cp.I*np.matrix([1]*11).T)) *
                      np.matrix([1]*11).T)
        # Markowitz portfolio
        Markweight = markC.I * markR.T / (np.matrix([1]*11)*markC.I*markR.T) # weight 
        BL_weights.loc[next_start,:] = opt_weight.T
        Mark_weights.loc[next_start,:] = Markweight.T
    BL_weights.index = pd.to_datetime(BL_weights.index, format='%Y%m%d')
    Mark_weights.index = pd.to_datetime(Mark_weights.index, format='%Y%m%d')

    return BL_weights, Mark_weights

def BL_backtestn(start_dates, end_dates, price, mtm, q_shift, Omega_shift,
                tau=0.25, xi=0.42667, Value=1000000):
    # backtest 
    # momentum strategy
    Net_value = pd.DataFrame(None, index=price.index, columns=['BL','Mkt'])
    Net_value = Net_value.loc[start_dates[1]:,:]
    MarkValue = Value
    
    for i in range(len(start_dates)-1):
        start = start_dates[i]
        end = end_dates[i]
        next_start = start_dates[i+1]
        next_end = end_dates[i+1]
        # select training data
        price_training = price.loc[start:end,:]
        mkt_cap_training = mkt_cap.loc[start:end,:]
        mtm_training = mtm.loc[start:end,:]    
        R_training = R.loc[start:end,:] 
        R_mean = R_training.mean()       
        C = np.matrix(R_training.cov())

        #ann_test = R_test.sum().tolist()
        
        # set parameters (views)
        P, q, Omega = BL_parasn(mtm_training, R_mean, q_shift, Omega_shift)

        tau = 0.025
        Sigma = tau * C
        xi = 0.42667
    
        weights = pd.DataFrame(None, index=mkt_cap_training.index, columns=mkt_cap_training.columns)
        for date, row in mkt_cap_training.iterrows():
            weights.loc[date] = row/row.sum()
        weight = np.matrix(weights.iloc[-1,:]).T # market capitalization weights
        pi = 1/xi * C * weight
        
        # formulas
        Rp = pi + Sigma * P.T * (P*Sigma*P.T + Omega).I * (q-P*pi)
        Cp = C + Sigma - Sigma * P.T * (P*Sigma*P.T + Omega).I * (P*Sigma)
        opt_weight = (xi * Cp.I * Rp - 
                      np.multiply(Cp.I, (xi*np.matrix([1]*11)*Cp.I*Rp-1)/(np.matrix([1]*11)*Cp.I*np.matrix([1]*11).T)) *
                      np.matrix([1]*11).T)
        
        # backtest
        Price = np.matrix(price_training.iloc[-1,:]).T    
        test_price = price.loc[next_start:next_end, :]
        
        # BL portfolio
        Shares = np.divide(Value * opt_weight, Price)
        cleanShares = Shares - np.mod(Shares,100)
        StockValue = np.matrix(test_price) * cleanShares
        TestNetValue = StockValue/1000000
        Value = StockValue[-1, 0]
        
        # Markowitz portfolio
        Mark_weight = weight 
        MarkShares = np.divide(MarkValue * Mark_weight, Price)
        markcleanShares = MarkShares - np.mod(MarkShares,100)
        MarkStockValue = np.matrix(test_price) * markcleanShares
        MarkNetValue = MarkStockValue/1000000
        MarkValue = MarkStockValue[-1,0]
        
        # extract data
        Net_value.ix[next_start:next_end, 'BL'] = TestNetValue.reshape(1,-1).tolist()[0]
        Net_value.ix[next_start:next_end, 'Mkt'] = MarkNetValue.reshape(1,-1).tolist()[0] 
        
    Net_value.index = pd.to_datetime(Net_value.index, format='%Y%m%d')
    return Net_value
#%%
if __name__ == '__main__':
    # loading data
    # os.chdir('Users/ivy/Desktop/Black-Litterman/')
    price = pd.read_csv('./price.csv', index_col='date')
    mkt_cap = pd.read_csv('./mkt_cap.csv', index_col='date')
    mtm = pd.read_csv('./MTM120D.csv', index_col='date')
    R = pd.read_csv('./return.csv', index_col='date')
        
    start_dates = []
    end_dates = []
    
    for year in range(2004, 2019):
        for month in [1,7]:
            bg_value = year*10000 + month*100
            ed_value = year*10000 + (month+6)*100
            start_dates.append(price.index[price.index>bg_value][0])
            end_dates.append(price.index[price.index<ed_value][-1])

#%% Part I
    # Input: future returns & variance
    # Test sensitivity with respect to changes of q & Omega
    
    # Fix q shift, look into Omega-sensitivity
    q_shifts = [0.1, 0.2, 0.5, 1, 2]
    Omega_shifts = [0.05, 0.1, 1, 10, 25, 50]
    q_shift = 0
    for Omega_shift in Omega_shifts:
        BL_weights, Mark_weights = BL_backtestf(start_dates, end_dates, price, R, q_shift, Omega_shift)
        titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shift)
        plt.figure(figsize=[12,8])
        plt.plot(BL_weights)
        plt.title(titletext)
        plt.ylabel('Black Litterman weights')
        plt.legend(BL_weights.columns)
        plt.show()
        plt.figure(figsize=[12,8])
        plt.plot(Mark_weights)
        plt.title(titletext)
        plt.ylabel('Markowitz weihgts')
        plt.legend(Mark_weights.columns)
        plt.show()

#%% 
    # Fix Omega shift, look into q-sensitivity
    Omega_shift = 10
    for q_shift in q_shifts:
        BL_weights, Mark_weights = BL_backtestf(start_dates, end_dates, price, R, q_shift, Omega_shift)
        titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shift)
        plt.figure(figsize=[8,6])
        plt.plot(BL_weights)
        plt.title(titletext)
        plt.ylabel('Black Litterman weights')
        plt.legend(BL_weights.columns)
        plt.show()
        plt.figure(figsize=[8,6])
        plt.plot(Mark_weights)
        plt.title(titletext)
        plt.ylabel('Markowitz weihgts')
        plt.legend(Mark_weights.columns)
        plt.show()        
            
#%% Part II
    # Input: momentum-ranked expected returns
    # Test momentum strategy performance vs CAPM        
    # Fix q shift, look into Omega-sensitivity    
    q_shift = 0
    for Omega_shift in Omega_shifts:
        Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shift, Omega_shift)
        titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shift)
        plt.figure(figsize=[8,6])
        plt.plot(Net_value)
        plt.title(titletext)
        plt.legend(Net_value.columns)
        plt.show()

#%%
    # Fix Omega shift, look into q-sensitivity
    Omega_shift = 0
    for q_shift in q_shifts:
        Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shift, Omega_shift)
        titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shift)
        plt.figure(figsize=[8,6])
        plt.plot(Net_value)
        plt.title(titletext)
        plt.legend(Net_value.columns)
        plt.show()