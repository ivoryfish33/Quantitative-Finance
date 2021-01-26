
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rankdata
from functools import reduce
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
                temp1=return_data.iloc[:,i]
            elif elem==11:
                P[0,i]=-1
                temp11=return_data.iloc[:,i]
            elif elem==2:
                P[1,i]=1
                temp2=return_data.iloc[:,i]
            elif elem==10:
                P[1,i]=-1
                temp10=return_data.iloc[:,i]
    if Type==1:
        q[0]=0.02*value
        q[1]=0.01*value
    elif Type==-1:
        q[0]=-0.02*value
        q[1]=-0.01*value
    #use difference of return to calculate variance
    Omega[0,0]=(temp1-temp11).var()*lamda
    Omega[1,1]=(temp2-temp10).var()*lamda

    
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
        q[0]=surpass1*value
        q[1]=surpass2*value
    elif Type==-1:
        q[0]=-surpass1*value
        q[1]=-surpass2*value

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

def BL_backtestf1(start_dates, end_dates, price, R, q_shift, Omega_shift,
                tau=0.25, xi=0.42667):
    # back_test for Black-Litterman model
    # future returns as investors' views
    # ------------------------------------- change portfolio weigths every 6 months 
    BL_weights = pd.DataFrame(None, index=start_dates[1:], columns=R.columns)
    Mark_weights = pd.DataFrame(None, index=start_dates[1:], columns=R.columns)
    G = pd.DataFrame(None, index=start_dates[1:], columns=R.columns)
    w_pi = pd.DataFrame(None, index=start_dates[1:], columns=R.columns)
    for i in range(len(start_dates)-1):
        start = start_dates[i]
        end = end_dates[i]
        next_start = start_dates[i+1]
        next_end = end_dates[i+1]
        
        # --------------------------------- select training data
        mkt_cap_training = mkt_cap.loc[start:end,:]  
        R_training = R.loc[start:end,:]         
        R_test = R.loc[next_start:next_end,:]        
        C = np.matrix(R_training.cov())
        markR = np.matrix(R_test.mean())
        markC = np.matrix(R_test.cov())
        
        # set parameters (views)
        # -------------------------------------------------------- Parameters
        P, q, Omega = BL_parasf(R_test, 1, q_shift, Omega_shift)

        tau = 0.025
        Sigma = tau * C
        xi = 0.42667
    
        weights = pd.DataFrame(None, index=mkt_cap_training.index, columns=mkt_cap_training.columns)
        for date, row in mkt_cap_training.iterrows():
            weights.loc[date] = row/row.sum()
        weight = np.matrix(weights.iloc[-1,:]).T # market capitalization weights
        pi = 1/xi * C * weight
        w_pi.loc[next_start,:] = weight.T
        # formulas
        Rp = pi + Sigma * P.T * (P*Sigma*P.T + Omega).I * (q-P*pi)
        Cp = C + Sigma - Sigma * P.T * (P*Sigma*P.T + Omega).I * (P*Sigma)
        opt_weight = (xi * Cp.I * Rp - 
                      np.multiply(Cp.I, (xi*np.matrix([1]*11)*Cp.I*Rp-1)/(np.matrix([1]*11)*Cp.I*np.matrix([1]*11).T)) *
                      np.matrix([1]*11).T)
        # Markowitz portfolio
        Markweight = markC.I * markR.T / (np.matrix([1]*11)*markC.I*markR.T) # weight 
        # ------------------------------------------------------------- Global minimum portfolio
        G_tmpt = markC.I * np.matrix([1]*11).T / sum(markC.I * np.matrix([1]*11).T)
        G.loc[next_start,:] = G_tmpt.T 
        BL_weights.loc[next_start,:] = opt_weight.T
        Mark_weights.loc[next_start,:] = Markweight.T
    BL_weights.index = pd.to_datetime(BL_weights.index, format='%Y%m%d')
    Mark_weights.index = pd.to_datetime(Mark_weights.index, format='%Y%m%d')
    G.index = pd.to_datetime(G.index, format='%Y%m%d')
    w_pi.index = pd.to_datetime(G.index, format='%Y%m%d')
    return BL_weights, Mark_weights,G, w_pi
def BL_backtestn(start_dates, end_dates, price, mtm, q_shift, Omega_shift,
                tau=0.25, xi=0.42667, Value=1000000):
    # backtest 
    # momentum strategy
    Net_value = pd.DataFrame(None, index=price.index, columns=['BL','Mkt'])
    Net_value = Net_value.loc[start_dates[1]:,:]
    BLValue = Value
    MarkValue = Value
    BL_weights = pd.DataFrame(None, index=start_dates[1:], columns=R.columns)
    
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
        C = np.matrix(R_training.cov())

        #ann_test = R_test.sum().tolist()
        
        # set parameters (views)
        P, q, Omega = BL_parasn(mtm_training, R_training, -1, q_shift, Omega_shift)

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
        Shares = np.divide(BLValue * opt_weight, Price)
        cleanShares = Shares - np.mod(Shares,100)
        StockValue = np.matrix(test_price) * cleanShares
        TestNetValue = StockValue/Value
        BLValue = StockValue[-1, 0]
        
        # Markowitz portfolio
        Mark_weight = weight 
        MarkShares = np.divide(MarkValue * Mark_weight, Price)
        markcleanShares = MarkShares - np.mod(MarkShares,100)
        MarkStockValue = np.matrix(test_price) * markcleanShares
        MarkNetValue = MarkStockValue/Value
        MarkValue = MarkStockValue[-1,0]
        
        # extract data
        Net_value.ix[next_start:next_end, 'BL'] = TestNetValue.reshape(1,-1).tolist()[0]
        Net_value.ix[next_start:next_end, 'Mkt'] = MarkNetValue.reshape(1,-1).tolist()[0] 
        
        BL_weights.loc[next_start,:] = opt_weight.T
    
    BL_weights.index = pd.to_datetime(BL_weights.index, format='%Y%m%d')    
    Net_value.index = pd.to_datetime(Net_value.index, format='%Y%m%d')
    return BL_weights, Net_value
#%%
def ind(rf, dataframe, window):
    """
    return alpha, beta, sharp ratio, max drawdown.
    """
    ret_data=dataframe/dataframe.shift(1)-1
    N=len(dataframe['BL'])-window
    betas=np.array([])
    alphas=np.array([])
    
    sharp_ratios=np.array([])
    MDDs=np.array([])
    for i in range(N):
        period_ret=pd.DataFrame()
        period_ret['BL']=ret_data['BL'][1+i:window+1+i]
        period_ret['Mkt']=ret_data['Mkt'][1+i:window+1+i]
        cov=np.cov(list(period_ret['BL']),list(period_ret['Mkt']))[0][1]
        var=np.var(period_ret['Mkt'])
        beta=cov/var
        betas=np.append(betas,beta)
        rm=reduce(lambda x,y:x*y,(period_ret['Mkt']+1))-1
        r=reduce(lambda x,y:x*y,(period_ret['BL']+1))-1
        alpha=r-float(rf[i])/100*window/252-beta*(rm-float(rf[i])/100*window/252)
        alphas=np.append(alphas,alpha)
        sharp_ratio=(r-float(rf[i])/100*window/252)/np.std(period_ret['BL'])/np.sqrt(window)
        sharp_ratios=np.append(sharp_ratios,sharp_ratio)
        MDD=np.max(1-dataframe['BL'][1+i:window+i+1]/np.maximum.accumulate(dataframe['BL'][1+i:window+i+1]))
        MDDs=np.append(MDDs,MDD)
    result=pd.DataFrame(index=ret_data.index[1:-(window-1)],columns=['Alpha','Beta','Sharp Ratio','Maximum Drawdown'])
    result['Alpha']=alphas
    result['Beta']=betas
    result['Sharp Ratio']=sharp_ratios
    result['Maximum Drawdown']=MDDs
    return result

def ind1(rf, dataframe, window):
    """
    return BL alpha, BL beta, CAPM sharp ratio, CAPM max drawdown.
    """
    ret_data=dataframe/dataframe.shift(1)-1
    N=len(dataframe['BL'])-window
    betas=np.array([])
    alphas=np.array([])
    
    sharp_ratios=np.array([])
    MDDs=np.array([])
    for i in range(N):
        period_ret=pd.DataFrame()
        period_ret['BL']=ret_data['BL'][1+i:window+1+i]
        period_ret['Mkt']=ret_data['Mkt'][1+i:window+1+i]
        cov=np.cov(list(period_ret['BL']),list(period_ret['Mkt']))[0][1]
        var=np.var(period_ret['Mkt'])
        beta=cov/var
        betas=np.append(betas,beta)
        rm=reduce(lambda x,y:x*y,(period_ret['Mkt']+1))-1
        r=reduce(lambda x,y:x*y,(period_ret['Mkt']+1))-1
        alpha=r-float(rf[i])/100*window/252-beta*(rm-float(rf[i])/100*window/252)
        alphas=np.append(alphas,alpha)
        sharp_ratio=(r-float(rf[i])/100*window/252)/np.std(period_ret['Mkt'])/np.sqrt(window)
        sharp_ratios=np.append(sharp_ratios,sharp_ratio)
        MDD=np.max(1-dataframe['Mkt'][1+i:window+i+1]/np.maximum.accumulate(dataframe['Mkt'][1+i:window+i+1]))
        MDDs=np.append(MDDs,MDD)
    result=pd.DataFrame(index=ret_data.index[1:-(window-1)],columns=['Alpha','Beta','Sharp Ratio','Maximum Drawdown'])
    result['Alpha']=alphas
    result['Beta']=betas
    result['Sharp Ratio']=sharp_ratios
    result['Maximum Drawdown']=MDDs
    return result
#%%
if __name__ == '__main__':
    # loading data

    price = pd.read_csv('price.csv', index_col='date')
    mkt_cap = pd.read_csv('mkt_cap.csv', index_col='date')
    mtm = pd.read_csv('MTM120D.csv', index_col='date')
    R = pd.read_csv('return.csv', index_col='date')
    window=120
    csv_data = pd.read_csv('10y_Treasury_Constant_Maturity_Rate.csv')
    
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
    
    q_shifts = [0.1,0.2, 0.5, 1, 2, 3]
    Omega_shifts = [ 1, 5, 10, 15, 20, 10000]
    Omega_shift = 1
    q_shift=1
    plt.figure(figsize=[24,8])
    BL_weights, Mark_weights = BL_backtestf(start_dates, end_dates, price, R, q_shift, Omega_shift)
    titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shift)
    plt.subplot(1,3,1)
    plt.plot(BL_weights)
    plt.title(titletext)
    plt.ylabel('Black Litterman weights')
    plt.yticks(np.array([-50,-40,-30,-20,-10,0,10,20]),np.array([-50,-40,-30,-20,-10,0,10,20]))
    plt.legend(BL_weights.columns)
    plt.subplot(1,3,2)
    plt.plot(BL_weights)
    plt.title(titletext)
    plt.ylabel('Black Litterman weights')
    plt.legend(BL_weights.columns)
    plt.subplot(1,3,3)
    plt.plot(Mark_weights)
    plt.title('Markowitz weihgts')
    plt.ylabel('Markowitz weihgts')
    plt.legend(Mark_weights.columns)
    plt.show()
#%%
    w_BL_q3, w_M_q3, G_q3, w_pi = BL_backtestf1(start_dates, end_dates, price, R, 1, Omega_shifts[2])
    plt.figure(figsize=[30,8])
    plt.subplot(4,3,1)
    plt.plot(w_BL_q3.XLV,color='red')
    plt.plot(w_pi.XLV,color='blue') # -------- XLV
    plt.legend(['BL','CAPM'])
    plt.ylabel('XLV weights')    
    plt.subplot(4,3,2)
    plt.plot(w_BL_q3.XLF,color='blue')
    plt.plot(w_pi.XLF,color='red')  # -------- XLF
    plt.ylabel('XLF weights')
    plt.legend(['BL','CAPM'])
    plt.subplot(4,3,3)
    plt.plot(w_BL_q3.IYT,color='red')
    plt.plot(w_pi.IYT,color='blue') # -------- IYT
    plt.ylabel('IYT weights')
    plt.legend(['BL','CAPM'])
    plt.subplot(4,3,4)
    plt.plot(w_BL_q3.XLI,color='red')
    plt.plot(w_pi.XLI,color='blue') # -------- XLI
    plt.ylabel('XLI weights')
    plt.legend(['BL','CAPM'])
    plt.subplot(4,3,5)
    plt.plot(w_BL_q3.XLE,color='red')
    plt.plot(w_pi.XLE,color='blue') # -------- XLE
    plt.ylabel('XLE weights')
    plt.legend(['BL','CAPM'])
    plt.subplot(4,3,6)
    plt.plot(w_BL_q3.RWR,color='red')
    plt.plot(w_pi.RWR,color='blue') # -------- RWR
    plt.ylabel('RWR weights')
    plt.legend(['BL','CAPM'])
    plt.subplot(4,3,7)
    plt.plot(w_BL_q3.XLK,color='red')
    plt.plot(w_pi.XLK,color='blue') # -------- XLK
    plt.ylabel('XLK weights')
    plt.legend(['BL','CAPM'])
    plt.subplot(4,3,8)
    plt.plot(w_BL_q3.XLP,color='red')
    plt.plot(w_pi.XLP,color='blue') # -------- XLP
    plt.ylabel('XLP weights')
    plt.legend(['BL','CAPM'])
    plt.subplot(4,3,9)
    plt.plot(w_BL_q3.XLY,color='red')
    plt.plot(w_pi.XLY,color='blue') # -------- XLY
    plt.ylabel('XLY weights')
    plt.legend(['BL','CAPM'])
    plt.subplot(4,3,10)
    plt.plot(w_BL_q3.XLU,color='red')
    plt.plot(w_pi.XLU,color='blue') # -------- XLU
    plt.ylabel('XLU weights')
    plt.legend(['BL','CAPM'])
    plt.subplot(4,3,11)
    plt.plot(w_BL_q3.XLB,color='red')
    plt.plot(w_pi.XLB,color='blue') # -------- XLB
    plt.ylabel('XLB weights')
    plt.legend(['BL','CAPM'])            
#%% weights change wrt Omega shifts
    q_shift = 1
    x=pd.DataFrame(None)
    plt.figure(figsize=[12,8])
    for i in range(len(Omega_shifts)):
        BL_weights, Mark_weights = BL_backtestf(start_dates, end_dates, price, R, q_shift, Omega_shifts[i])
        BL_weights=BL_weights-BL_weights.shift(1)
        BL_weights=BL_weights.fillna(0)
        BL_weights['Col_sum'] = BL_weights.apply(lambda x: abs(x).sum(), axis=1)
        x['$\Omega$ shift = %d'%Omega_shifts[i]]=BL_weights['Col_sum']
    plt.plot(x)
    plt.title('BL Weights chg with different $\Omega$ shift (q shift = 1)')
    plt.legend(x.columns)
#%% weights change wrt q shifts
    Omega_shift = 1
    x=pd.DataFrame(None)
    plt.figure(figsize=[12,8])
    for i in range(len(q_shifts)):
        BL_weights, Mark_weights = BL_backtestf(start_dates, end_dates, price, R, q_shifts[i], Omega_shift)
        BL_weights=BL_weights-BL_weights.shift(1)
        BL_weights=BL_weights.fillna(0)
        BL_weights['Col_sum'] = BL_weights.apply(lambda x: abs(x).sum(), axis=1)
        x['q shift = %.1f'%q_shifts[i]]=BL_weights['Col_sum']
    plt.plot(x)
    plt.title('BL Weights chg with different q shift ($\Omega$ shift = 10)')
    plt.legend(x.columns)
%%
#%% Part II
#     Input: momentum-ranked expected returns
#     Test momentum strategy performance vs CAPM        
#     Fix q shift, look into Omega-sensitivity    
#%% CAPM Maximum Drawdown
    q_shift = 1
    Omega_shift=10
    plt.figure(figsize=[12,8])
    _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shift, Omega_shift)    
    titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shift)
    dataframe=Net_value
    date=dataframe.index
    date=list(date.strftime("%Y-%m-%d"))
    riskfree=list(csv_data.ix[csv_data['DATE'].isin(date),:]['DGS10'])[window:]
    for j in range(len(riskfree)):
        if riskfree[j]=='.':
            riskfree[j]=riskfree[j-1]
    lmm=ind1(riskfree, dataframe, window)
    plt.plot(lmm['Maximum Drawdown'])
    plt.title('Maximum Drawdown of CAPM')
#%% CAPM Sharp Ratio
    q_shift = 1
    Omega_shift=10
    plt.figure(figsize=[12,8])
    _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shift, Omega_shift)
    titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shift)
    dataframe=Net_value
    date=dataframe.index
    date=list(date.strftime("%Y-%m-%d"))
    riskfree=list(csv_data.ix[csv_data['DATE'].isin(date),:]['DGS10'])[window:]
    for j in range(len(riskfree)):
        if riskfree[j]=='.':
            riskfree[j]=riskfree[j-1]
    lmm=ind1(riskfree, dataframe, window)
    plt.plot(lmm['Sharp Ratio'])
    plt.title('Sharp Ratio of CAPM')
#%% net value comparason wrt q shifts
    Omega_shift=10
    plt.figure(figsize=[27,18])
    for i in range(len(q_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shifts[i], Omega_shift)
        titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shifts[i])
        plt.subplot(2,3,i+1)
        plt.plot(Net_value)
        plt.title(titletext)
        plt.legend(Net_value.columns)
#%% net value comparason wrt Omega shifts
    q_shift=1
    plt.figure(figsize=[27,18])
    for i in range(len(Omega_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shift, Omega_shifts[i])
        titletext = '$\Omega$ shift = ' + str(Omega_shifts[i]) + '; $q$ shift = ' + str(q_shift)
        plt.subplot(2,3,i+1)
        plt.plot(Net_value)
        plt.title(titletext)
        plt.legend(Net_value.columns)
#%% BL value sensitivity plot wrt Omega shift
    q_shift=1
    plt.figure(figsize=[12,8])
    x=pd.DataFrame(None)
    for i in range(len(Omega_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shift, Omega_shifts[i])
        titletext = '$\Omega$ shift = ' + str(Omega_shifts[i]) + '; $q$ shift = ' + str(q_shift)
        x['$\Omega$ shift =%d' % Omega_shifts[i]]=Net_value['BL']
    plt.plot(x)
    plt.title('BL value sensitivity plot wrt $\Omega$ shift')
    plt.legend(x.columns)
#%% BL value sensitivity plot wrt q shift
    Omega_shift=10
    plt.figure(figsize=[12,8])
    x=pd.DataFrame(None)
    for i in range(len(q_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shifts[i], Omega_shift)
        titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shifts[i])
        x['q shift =%.1f' % q_shifts[i]]=Net_value['BL']
    plt.plot(x)
    plt.title('BL value sensitivity plot wrt q shift')
    plt.legend(x.columns)
#%% change of BL Sharp Ratio wrt Omega shift
    q_shift=1
    plt.figure(figsize=[27,18])
    for i in range(len(Omega_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shift, Omega_shifts[i])
        titletext = '$\Omega$ shift = ' + str(Omega_shifts[i]) + '; $q$ shift = ' + str(q_shift)
        dataframe=Net_value
        date=dataframe.index
        date=list(date.strftime("%Y-%m-%d"))
        riskfree=list(csv_data.ix[csv_data['DATE'].isin(date),:]['DGS10'])[window:]
        for j in range(len(riskfree)):
            if riskfree[j]=='.':
                riskfree[j]=riskfree[j-1]
        lmm=ind(riskfree, dataframe, window)
        plt.subplot(2,3,i+1
        plt.plot(lmm['Sharp Ratio'])
        plt.title('Sharp Ratio of BL wrt q shift = 1 and $\Omega$ shift = %d'%Omega_shifts[i])
#%% change of BL Sharp Ratio wrt q shift
    Omega_shift=10
    plt.figure(figsize=[27,18])
    for i in range(len(q_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shifts[i], Omega_shift)
        titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shifts[i])
        dataframe=Net_value
        date=dataframe.index
        date=list(date.strftime("%Y-%m-%d"))
        riskfree=list(csv_data.ix[csv_data['DATE'].isin(date),:]['DGS10'])[window:]
        for j in range(len(riskfree)):
            if riskfree[j]=='.':
                riskfree[j]=riskfree[j-1]
        lmm=ind(riskfree, dataframe, window)
        plt.subplot(2,3,i+1)
        plt.plot(lmm['Sharp Ratio'])
        plt.title('Sharp Ratio of BL wrt $\Omega$ shift = 10 and q shift = %.1f'%q_shifts[i])
#%% change of excess Sharp Ratio wrt q shift
    Omega_shift=10
    plt.figure(figsize=[27,18])
    for i in range(len(q_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shifts[i], Omega_shift)
        titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shifts[i])
        dataframe=Net_value
        date=dataframe.index
        date=list(date.strftime("%Y-%m-%d"))
        riskfree=list(csv_data.ix[csv_data['DATE'].isin(date),:]['DGS10'])[window:]
        for j in range(len(riskfree)):
            if riskfree[j]=='.':
                riskfree[j]=riskfree[j-1]
        lmm=ind(riskfree, dataframe, window)
        lmm1=ind1(riskfree, dataframe, window)
        lmm['Sharp Ratio']=lmm['Sharp Ratio']-lmm1['Sharp Ratio']
        plt.subplot(2,3,i+1)
        plt.plot(lmm['Sharp Ratio'])
        plt.title('Excess Sharp Ratio wrt $\Omega$ shift = 10 and q shift = %.1f'%q_shifts[i])
#%% change of excess Sharp Ratio wrt Omega shift
    q_shift=1
    plt.figure(figsize=[27,18])
    for i in range(len(Omega_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shift, Omega_shifts[i])
        titletext = '$\Omega$ shift = ' + str(Omega_shifts[i]) + '; $q$ shift = ' + str(q_shift)
        dataframe=Net_value
        date=dataframe.index
        date=list(date.strftime("%Y-%m-%d"))
        riskfree=list(csv_data.ix[csv_data['DATE'].isin(date),:]['DGS10'])[window:]
        for j in range(len(riskfree)):
            if riskfree[j]=='.':
                riskfree[j]=riskfree[j-1]
        lmm=ind(riskfree, dataframe, window)
        lmm1=ind1(riskfree, dataframe, window)
        lmm['Sharp Ratio']=lmm['Sharp Ratio']-lmm1['Sharp Ratio']
        plt.subplot(2,3,i+1)
        plt.plot(lmm['Sharp Ratio'])
        plt.title('Excess Sharp Ratio wrt q shift = 1 and $\Omega$ shift = %d'%Omega_shifts[i])
#%% change of maximum drawdown wrt Omega shift
    q_shift=1
    plt.figure(figsize=[12,8])
    x=pd.DataFrame(None)
    for i in range(len(Omega_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shift, Omega_shifts[i])
        titletext = '$\Omega$ shift = ' + str(Omega_shifts[i]) + '; $q$ shift = ' + str(q_shift)
        dataframe=Net_value
        date=dataframe.index
        date=list(date.strftime("%Y-%m-%d"))
        riskfree=list(csv_data.ix[csv_data['DATE'].isin(date),:]['DGS10'])[window:]
        for j in range(len(riskfree)):
            if riskfree[j]=='.':
                riskfree[j]=riskfree[j-1]
        lmm=ind(riskfree, dataframe, window)
        x['$\Omega$ shift =%d' % Omega_shifts[i]]=lmm['Maximum Drawdown']
    plt.plot(x)
    plt.title('Maximum Drawdown of BL wrt q shift = 1 and $\Omega$ shifts')
    plt.legend(x.columns)
#%% change of maximum drawdown wrt q shift
    Omega_shift=10
    plt.figure(figsize=[12,8])
    for i in range(len(q_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shifts[i], Omega_shift)
        titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shifts[i])
        dataframe=Net_value
        date=dataframe.index
        date=list(date.strftime("%Y-%m-%d"))
        riskfree=list(csv_data.ix[csv_data['DATE'].isin(date),:]['DGS10'])[window:]
        for j in range(len(riskfree)):
            if riskfree[j]=='.':
                riskfree[j]=riskfree[j-1]
        lmm=ind(riskfree, dataframe, window)
        x['q shift =%.1f' % q_shifts[i]]=lmm['Maximum Drawdown']
    plt.plot(x)
    plt.title('Maximum Drawdown of BL wrt $\Omega$ shift = 10 and q shifts')
    plt.legend(x.columns)
#%% change of alpha wrt q shift
    Omega_shift=10
    plt.figure(figsize=[12,8])
    x=pd.DataFrame(None)
    for i in range(len(q_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shifts[i], Omega_shift)
        titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shifts[i])
        dataframe=Net_value
        date=dataframe.index
        date=list(date.strftime("%Y-%m-%d"))
        riskfree=list(csv_data.ix[csv_data['DATE'].isin(date),:]['DGS10'])[window:]
        for j in range(len(riskfree)):
            if riskfree[j]=='.':
                riskfree[j]=riskfree[j-1]
        lmm=ind(riskfree, dataframe, window)
        x['q shift =%.1f' % q_shifts[i]]=lmm['Alpha']
    plt.plot(x)
    plt.title('Alpha of BL wrt $\Omega$ shift = 10 and q shifts')
    plt.legend(x.columns)
#%% change of alpha wrt Omega shift
    q_shift=1
    plt.figure(figsize=[12,8]
    for i in range(len(Omega_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shift, Omega_shifts[i])
        titletext = '$\Omega$ shift = ' + str(Omega_shifts[i]) + '; $q$ shift = ' + str(q_shift)
        dataframe=Net_value
        date=dataframe.index
        date=list(date.strftime("%Y-%m-%d"))
        riskfree=list(csv_data.ix[csv_data['DATE'].isin(date),:]['DGS10'])[window:]
        for j in range(len(riskfree)):
            if riskfree[j]=='.':
                riskfree[j]=riskfree[j-1]
        lmm=ind(riskfree, dataframe, window)
        x['$\Omega$ shift =%d' % Omega_shifts[i]]=lmm['Alpha']
    plt.plot(x)
    plt.title('Alpha of BL wrt q shift = 1 and $\Omega$ shifts')
    plt.legend(x.columns)
#%% change of beta  wrt q shift
    Omega_shift=10
    plt.figure(figsize=[12,8])
    x=pd.DataFrame(None)
    for i in range(len(q_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shifts[i], Omega_shift)
        titletext = '$\Omega$ shift = ' + str(Omega_shift) + '; $q$ shift = ' + str(q_shifts[i])
        dataframe=Net_value
        date=dataframe.index
        date=list(date.strftime("%Y-%m-%d"))
        riskfree=list(csv_data.ix[csv_data['DATE'].isin(date),:]['DGS10'])[window:]
        for j in range(len(riskfree)):
            if riskfree[j]=='.':
                riskfree[j]=riskfree[j-1]
        lmm=ind(riskfree, dataframe, window)
        x['q shift =%.1f' % q_shifts[i]]=lmm['Beta']
    plt.plot(x)
    plt.title('Beta of BL wrt $\Omega$ shift = 10 and q shifts')
    plt.legend(x.columns)
#%% change of beta wrt Omega shift
    q_shift=1
    plt.figure(figsize=[12,8])
    for i in range(len(Omega_shifts)):
        _, Net_value = BL_backtestn(start_dates, end_dates, price, mtm, q_shift, Omega_shifts[i])
        titletext = '$\Omega$ shift = ' + str(Omega_shifts[i]) + '; $q$ shift = ' + str(q_shift)
        dataframe=Net_value
        date=dataframe.index
        date=list(date.strftime("%Y-%m-%d"))
        riskfree=list(csv_data.ix[csv_data['DATE'].isin(date),:]['DGS10'])[window:]
        for j in range(len(riskfree)):
            if riskfree[j]=='.':
                riskfree[j]=riskfree[j-1]
        lmm=ind(riskfree, dataframe, window)
        x['$\Omega$ shift =%d' % Omega_shifts[i]]=lmm['Beta']
    plt.plot(x)
    plt.title('Beta of BL wrt q shift = 1 and $\Omega$ shifts')
    plt.legend(x.columns)    
