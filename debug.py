# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:29:23 2017

@author: xyy
"""
import numpy as np
import pandas as pd
from datetime import datetime
from lib.gftTools import gftIO
import matplotlib.pyplot as plt

data = gftIO.zload(r'd:\Wuwei\Project\simulator\data\monthly_rebalance_data_2.pkl')
data['begin_date'] = data.pop('x0')
data['end_date'] = data.pop('x1')
data['initial_holding'] = data.pop('x2')
data['target_portfolio_weight'] = data.pop('x3')
data['market_to_market_price'] = data.pop('x4')
data['total_return_factor'] = data.pop('x5')
data['execute_price'] = data.pop('x6')
data['execute_price_return'] = data.pop('x7')
data['trade_volume'] = data.pop('x8')
data['trading_param'] = data.pop('x9')
data['additional_Ts'] = data.pop('x10')

def parse_data(input_data):
    if type(input_data) is gftIO.GftTable:
        input_data = input_data.asMatrix()
        if input_data.columns.dtype == 'datetime64[ns]':
            return input_data.T
        else:
            return input_data
    elif type(input_data) is np.ndarray:
        return input_data.astype(datetime)
    else:
        return input_data

for key, value in data.items():
    data[key] = parse_data(value)

dates = pd.date_range('1/1/2000', periods=8)
start_date = dates[0]
end_date = dates[-1]
symbols = ['A', 'B', 'C', 'D', 'E', 'Cash']
noa = len(symbols)
weights = np.random.random(noa)
weights /= np.sum(weights)

df_w_adjustFactor = pd.DataFrame(np.ones((8, noa-1)), index=dates, columns=['A', 'B', 'C', 'D', 'E'])
market_to_market_price = pd.DataFrame(np.random.randint(200,5000,size=(8, noa)), index=dates, columns=symbols)
total_return_factor = pd.DataFrame(np.ones(shape=(8, noa)), index=dates, columns=symbols)
execute_price = pd.DataFrame(np.random.randint(low=5,size=(8, noa)), index=dates, columns=symbols)
trade_volume = pd.DataFrame(np.random.randint(low=500,size=(8, noa)), index=dates, columns=symbols)
exec_price_return = pd.DataFrame(np.random.randint(low=5,size=(8, noa)), index=dates, columns=symbols)
initial_holding = pd.DataFrame([100,200,300,400,500,600], index=symbols)
initial_holding = initial_holding.T
target_portfolio_wgt = pd.DataFrame()
for i in range(8):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    target_portfolio_wgt = target_portfolio_wgt.append(pd.DataFrame(weights).T)
target_portfolio_wgt.index = dates
target_portfolio_wgt.columns = symbols

# pandas.Timestamp
dt_beginDate=pd.Timestamp('2000-01-01')
dt_endDate=pd.Timestamp('2000-01-08')
# dt_beginDate-dt_endDate
# dt_beginDate<dt_endDate

# create dict
ls_param=['debug','exeDelayPeriods','shiftBeginDateToSignal','canTradeOnSuspend','riseLimitThres','fallLimitThres','volumeLimitPct']
ls_num=[i for i in range(len(ls_param))]
zip_param = zip(ls_param, ls_num)
dict_tradingParam=dict(zip_param)


# create df
ser_dates=pd.date_range('1/1/2000',periods=8)
ls_symbols=['A','B','C','D','E']
df_w_markToMarketPrice=pd.DataFrame(np.random.randint(low=5,size=(8,len(ls_symbols))),index=ser_dates,columns=ls_symbols)
ls_tot_index=pd.date_range('1/1/2000',periods=7)
df_w_totalReturnFactor=pd.DataFrame(np.random.randint(low=5,size=(7,len(ls_symbols))),index=ls_tot_index,columns=ls_symbols)
df_w_executePrice=pd.DataFrame(np.random.randint(low=5,size=(8,len(ls_symbols))),index=ser_dates,columns=ls_symbols)
df_w_execPriceReturn=pd.DataFrame(np.random.randint(low=5,size=(8,len(ls_symbols))),index=ser_dates,columns=ls_symbols)
df_w_tradeVolume=pd.DataFrame(np.random.randint(low=500,size=(8,len(ls_symbols))),index=ser_dates,columns=ls_symbols)
initialHolding=initial_holding
initialHolding=initialHolding.rename(index=dict(zip([0],[dt_beginDate])))




df_w_targetPortfolioWgt= pd.DataFrame()
for i in range(8):
    weights = np.random.random(5)
    weights /= np.sum(weights)
    df_w_targetPortfolioWgt = df_w_targetPortfolioWgt.append(pd.DataFrame(weights).T)
df_w_targetPortfolioWgt.index = ser_dates
df_w_targetPortfolioWgt.columns = ls_symbols


ser_dates=pd.date_range('1/1/2000',periods=8)
ls_symbols=['A','B','C','D','E']
ls_new=['A','B','C','D','F']

def addMissingColumns(df_w,ls_columns):
    ls_tmp=[s for s in ls_columns if s not in df_w.columns.tolist()]
    df_w_tmp=pd.DataFrame(np.zeros((len(df_w),len(ls_tmp))),index=df_w.index.tolist(),columns=ls_tmp)
    return pd.concat([df_w_tmp,df_w],axis=1)

df_w_new=addMissingColumns(df_w_targetPortfolioWgt,ls_new)

dt_beginDate=pd.Timestamp('2000-01-01')
dt_endDate=pd.Timestamp('2000-01-08')

ls_allDates=sorted(list(set.intersection(set(df_w_markToMarketPrice.index),set(df_w_totalReturnFactor.index),set(df_w_executePrice.index),set(df_w_execPriceReturn.index),set(df_w_tradeVolume.index))))



b=[2,4,6,10]
a=[1,3,5,7,9,10]
def alignDate(a,b):
    alist=[]
    for i in b:
        for j in a:
            if i in a:
                alist.append(i)
                break
            elif i<j and i>a[a.index(j)-1]:
                alist.append(j)
                break
    return sorted(alist)
alist=alignDate(a,b)


def fillHolding(d, nextd, ls_tradeDates, df_w_holding_param, df_w_holdingCash_param, df_w_adjustFactor):
    if nextd>d:
        ls_holdingDates=[date for date in ls_tradeDates if date>=d and date<=nextd]
        
        #df_w_holding.ix[ls_holdingDates] = df_w_holding.ix[d]*
        
        ls_tmp_holding=df_w_holding_param.loc[d].tolist()
        narray_tmp_holding=np.tile(ls_tmp_holding,(len(ls_holdingDates),1))                            
        df_w_tmp_holding=pd.DataFrame(narray_tmp_holding, index=ls_holdingDates, columns=df_w_holding_param.columns.tolist())
        
        ls_tmp_holdingCash=df_w_holdingCash_param.loc[d].tolist()
        narray_tmp_holdingCash=np.tile(ls_tmp_holdingCash,(len(ls_holdingDates),1))                            
        df_w_tmp_holdingCash=pd.DataFrame(narray_tmp_holdingCash, index=ls_holdingDates, columns=df_w_holdingCash_param.columns.tolist())
        
        df_w_tmp_holding=df_w_tmp_holding*(df_w_adjustFactor.loc[ls_holdingDates]/df_w_adjustFactor.loc[d])
        df_w_holding_param.ix[ls_holdingDates]=df_w_tmp_holding
        df_w_holdingCash_param.ix[ls_holdingDates]=df_w_tmp_holdingCash
    return df_w_holding_param,df_w_holdingCash_param

if len(df_w_targetPortfolioWgt)>0:
    df_w_targetPortfolioWgt=df_w_targetPortfolioWgt.ix[ls_allDates]
    #df_w_targetPortfolioWgt=df_w_targetPortfolioWgt.drop([date for date in df_w_targetPortfolioWgt.index if date not in ls_allDates])
    df_w_targetPortfolioWgt=df_w_targetPortfolioWgt.dropna(how='all')
    ls_rebDates=df_w_targetPortfolioWgt.index.tolist()
    if len(ls_rebDates)>0: #############  and   (1)
          ls_rebDates=ls_rebDates ############# + deltaTime (2)
          
          #ls_rebDates=[i for j in ls_rebDates for i in ls_allDates if i>=j and j>ls_allDates[ls_allDates.index(i)-1]] # align(3)
          ls_rebDates=alignDate(ls_allDates,ls_rebDates)
          df_w_targetPortfolioWgt=df_w_targetPortfolioWgt.loc[[x for x in ls_rebDates if str(x)!='nan']]
          ls_rebDates=[x for x in ls_rebDates if str(x)!='nan']
          df_w_targetPortfolioWgt.index=ls_rebDates
          #df_w_targetPortfolioWgt.rename(index=dict(zip(df_w_targetPortfolioWgt.index.tolist(),ls_rebDates)))
          
'''grab ls_cashGid, find holdingSymbols, slice rebDates, tradeDates'''
    # input: initialHolding
    # returns: ls_cashGid, ls_holdingSymbols, dt_beginDate, dt_endDate, ls_tradeDates, ls_rebDates
    
ls_cashGid=[initialHolding.columns[-1]]#############################  # find which columbs is cash, only one element
#if np.shape(initialHolding)[1]>1: ############# initialHolding is a matrix
if isinstance(initialHolding,pd.DataFrame):
    dt_beginDate=initialHolding.index[-1]
    ls_holdingSymbols=sorted(list(set([i for i in initialHolding.columns if i not in ls_cashGid])))
else:
    if len(df_w_targetPortfolioWgt)>0: #dict_tradingParam['shiftBeginDateToSignal']>0 and(4)
        dt_beginDate=max(dt_beginDate, df_w_targetPortfolioWgt.index[0])
ls_tradeDates=[date for date in ls_allDates if date>=dt_beginDate and date<=dt_endDate]
dt_beginDate=ls_tradeDates[0]
dt_endDate=ls_tradeDates[-1]
ls_rebDates=[date for date in ls_rebDates if date>=dt_beginDate and date<=dt_endDate]
if (dt_beginDate>dt_endDate):
    #return "Begin date must be less than end date!"  
    print("Begin date must be less than end date!" )


'''slice '''
# input: df_w...
# return: ls_allSymbols, ls_portfolioSymbols, sliced df_w
ls_allSymbols=sorted(list(set.intersection(set(df_w_markToMarketPrice.columns),set(df_w_executePrice.columns),set(df_w_execPriceReturn.columns),set(df_w_tradeVolume.columns))))  

df_w_tmp_targetPortfolioWgt=df_w_targetPortfolioWgt.dropna(how='all',axis=1)
df_w_tmp_targetPortfolioWgt=df_w_tmp_targetPortfolioWgt.loc[:,(df_w_tmp_targetPortfolioWgt!=0).any(axis=0)]
ls_portfolioSymbols=[s for s in df_w_tmp_targetPortfolioWgt.columns.tolist() if s not in ls_cashGid]

if len([s for s in ls_holdingSymbols if s not in ls_allSymbols])>0:
#if ls_holdingSymbols not in ls_allSymbols:
    print("Initial Portfolio has non A-share stocks!")
    #return "Initial Portfolio has non A-share stocks!",[s for s in ls_allSymbols if s not in ls_holdingSymbols]
if len([s for s in ls_portfolioSymbols if s not in ls_allSymbols])>0:
#if ls_portfolioSymbols not in ls_allSymbols:
    print("Target Portfolio has non A-share stocks! ")
    #return "Target Portfolio has non A-share stocks! ",[s for s in ls_allSymbols if s not in ls_portfolioSymbols]
 
ls_allSymbols=sorted([s for s in set.intersection(set(ls_allSymbols),set(ls_holdingSymbols),set(ls_portfolioSymbols)) if s not in ls_cashGid])
ls_priceDates=[d for d in ls_allDates if d >=(dt_beginDate-pd.Timedelta('20 days')) and d<=dt_endDate]
df_w_markToMarketPrice=df_w_markToMarketPrice.loc[ls_priceDates][ls_allSymbols]
df_w_totalReturnFactor=df_w_totalReturnFactor.loc[ls_priceDates][ls_allSymbols]
df_w_executePrice=df_w_executePrice.loc[ls_priceDates][ls_allSymbols]
df_w_execPriceReturn=df_w_execPriceReturn.loc[ls_priceDates][ls_allSymbols]
df_w_tradeVolume=df_w_tradeVolume.loc[ls_priceDates][ls_allSymbols]          


''' '''
######if initialHolding is a df_w, initialHolding=initialHolding.loc[-1]
df_w_initialHolding=initialHolding
if not isinstance(initialHolding,pd.DataFrame):
#if type(initialHolding)==int or type(initialHolding)==float:
    df_w_initialHolding=pd.DataFrame(initialHolding,index=[dt_beginDate],columns=ls_cashGid)
    
df_w_initialHolding=addMissingColumns(df_w_initialHolding,ls_allSymbols)
df_w_initialHoldingCash=df_w_initialHolding.loc[df_w_initialHolding.index][ls_cashGid]
df_w_initialHolding.pop(ls_cashGid[0])
num_initialHoldingValue=float((df_w_initialHolding*df_w_markToMarketPrice.ix[df_w_initialHolding.index]).sum(1)+df_w_initialHoldingCash.values[-1][-1])
#num_initialHoldingValue=np.dot(df_w_initialHolding.values[-1],df_w_markToMarketPrice.loc[df_w_initialHolding.index].values[0])+df_w_initialHoldingCash.values[-1][-1]
df_w_targetPortfolioWgt=addMissingColumns(df_w_targetPortfolioWgt,ls_allSymbols+ls_cashGid).loc[ls_rebDates].fillna(0)
if (df_w_targetPortfolioWgt<0).any().any():
    print("Do not support stock short selling and cash borrowing")
#    #return "Do not support stock short selling and cash borrowing"
df_w_targetPortfolioCashWgt=df_w_targetPortfolioWgt[ls_cashGid]
df_w_targetPortfolioWgt.pop(ls_cashGid[0])
df_w_targetPortfolioCashWgt=1.-df_w_targetPortfolioWgt.sum(axis=1)


# filter
df_w_buyVolume=df_w_tradeVolume.copy().fillna(0)
df_w_sellVolumn=df_w_tradeVolume.copy().fillna(0)


if dict_tradingParam['canTradeOnSuspend']>0:  #np.sum((dict_tradingParam['canTradeOnSuspend'] > 0).values.ravel())
    df_w_buyVolume[df_w_buyVolume<1]=np.inf
    df_w_sellVolumn[df_w_sellVolumn<1]=np.inf
                   
if dict_tradingParam['riseLimitThres']>0:
    pass

if dict_tradingParam['fallLimitThres']>0:
    pass

if dict_tradingParam['volumeLimitPct']>0:
    df_w_buyVolume=df_w_buyVolume*dict_tradingParam['volumeLimitPct']
    df_w_sellVolumn=df_w_sellVolumn*dict_tradingParam['volumeLimitPct']

else:
    df_w_buyVolume[df_w_buyVolume>0]=np.inf
    df_w_sellVolumn[df_w_sellVolumn>0]=np.inf

### how to round (6)
num_lotSize=100
df_w_buyVolume=round(df_w_buyVolume/num_lotSize)*num_lotSize
df_w_sellVolumn=round(df_w_sellVolumn/num_lotSize)*num_lotSize

num_buyCommission=0.0
num_sellCommission=0.


# create: holding(tradeDates, allSymbols), holdingCash, portfilioValue(tradeDates, util.getGodGid())
#         weights, execution                            cumRets, singlePeriodRets, turnoverPct

df_w_holding=pd.DataFrame(0.,index=ls_tradeDates,columns=ls_allSymbols)
df_w_weights=df_w_holding.copy()
df_w_execution=df_w_holding.copy()
df_w_holdingCash=pd.DataFrame(0.,index=ls_tradeDates,columns=ls_cashGid)
# (7)
ls_getGodGid=['getGodGid']
df_w_portfolioValue=pd.DataFrame(0.,index=ls_tradeDates,columns=ls_getGodGid)
df_w_cumRets=df_w_portfolioValue.copy()
df_w_singlePeriodRets=df_w_portfolioValue.copy()
df_w_turnoverPct=df_w_portfolioValue.copy()


## trading
d=ls_tradeDates[0]
df_w_holding.ix[d]=df_w_initialHolding.loc[d].tolist()
df_w_holdingCash.ix[d]=df_w_initialHoldingCash.values[-1][-1]

if len(ls_rebDates)<1:
    nextd = ls_tradeDates[-1]
else:
    nextd=ls_rebDates[0]



if len(ls_rebDates)<1:
    nextd=ls_tradeDates[-1]
    df_w_holding,df_w_holdingCash=fillHolding(d,nextd,ls_tradeDates,df_w_holding,df_w_holdingCash,df_w_totalReturnFactor)
else:
    nextd=ls_rebDates[0]
    df_w_holding,df_w_holdingCash=fillHolding(d, nextd, ls_tradeDates, df_w_holding, df_w_holdingCash, df_w_adjustFactor)
    
    for i in range(len(ls_rebDates)):
        d=ls_rebDates[i]
        print('date=',d)
        df_w_currentHoldingValue=df_w_holding.ix[d]*df_w_executePrice.ix[d].fillna(0) # one line
        print('df_w_currentHoldingValue=',df_w_currentHoldingValue)
        num_totalValue=df_w_currentHoldingValue.sum()+df_w_holdingCash.ix[d].values[-1]
        
        df_w_currentHoldingWgt=df_w_currentHoldingValue/num_totalValue
        #print(df_w_currentHoldingValue)
        df_w_curHoldingCashWgt=1.-df_w_currentHoldingWgt.sum()
        df_w_targetHoldingWgt=df_w_targetPortfolioWgt.ix[d]
        num_targetHoldingCashWgt=1.-df_w_targetHoldingWgt.sum()
        df_w_orderWgt=df_w_targetHoldingWgt-df_w_currentHoldingWgt
        df_w_sellOrderWgt=df_w_orderWgt.copy()
        df_w_sellOrderWgt[df_w_orderWgt>0]=0
        df_w_buyOrderWgt=df_w_orderWgt.copy()
        df_w_buyOrderWgt[df_w_orderWgt<0]=0
        num_cashAvail=df_w_holdingCash.ix[d].values[-1]
        # sell
        if (df_w_sellOrderWgt<0).any():
            df_w_currentHoldingWgt_forSell=df_w_currentHoldingWgt.copy()
            df_w_currentHoldingWgt_forSell[df_w_currentHoldingWgt_forSell<=0.]=1
            df_w_sellOrder=round(df_w_sellOrderWgt/df_w_currentHoldingWgt_forSell*df_w_holding.ix[d]/100)*100 # share
            df_w_sellOrder[df_w_targetHoldingWgt<=0.] = -df_w_holding.ix[d]
            df_w_sellExecution = df_w_sellOrder.copy()
            df_w_sellExecution=-pd.concat([abs(df_w_sellExecution),df_w_sellVolumn.ix[d]],axis=1).min(axis=1)
            num_cashAvail=num_cashAvail+(abs(df_w_sellExecution)*df_w_executePrice.ix[d]).sum()*(1.-num_sellCommission)
            df_w_execution.ix[d]=df_w_execution.ix[d]+df_w_sellExecution
            df_w_holding.ix[d]=df_w_holding.ix[d]+df_w_sellExecution
        # buy
        if (df_w_buyOrderWgt>0).any():
            num_canBuyWgt=num_cashAvail/num_totalValue-num_targetHoldingCashWgt
            if num_canBuyWgt>0:
                num_pct=min(num_canBuyWgt/df_w_buyOrderWgt.sum(),1)
                df_w_buyOrder=round(num_pct*df_w_buyOrderWgt*num_totalValue/(1.+num_buyCommission)/df_w_executePrice.ix[d]/100)*100
                df_w_buyOrder=df_w_buyOrder.fillna(0)
                df_w_buyExecution=df_w_buyOrder.copy() # redundant
                df_w_buyExecution=pd.concat([df_w_buyExecution,df_w_buyVolume.ix[d]],axis=1).min(axis=1)
                num_cashAvail=num_cashAvail-(abs(df_w_buyExecution)*df_w_executePrice.ix[d]).sum()*(1.+num_buyCommission)
                df_w_execution.ix[d]=df_w_execution.ix[d]+df_w_buyExecution
                df_w_holding.ix[d]=df_w_holding.ix[d]+df_w_buyExecution
       
        df_w_holdingCash.ix[d]=num_cashAvail
        df_w_turnoverPct.ix[d]=(abs(df_w_execution.ix[d])*df_w_executePrice.ix[d]).sum()/num_totalValue  
                           
        if i<(len(ls_rebDates)-1):
            nextd=ls_rebDates[i+1]
            df_w_holding,df_w_holdingCash=fillHolding(d, nextd, ls_tradeDates, df_w_holding, df_w_holdingCash, df_w_adjustFactor)
                    
    nextd=ls_tradeDates[-1]
    df_w_holding,df_w_holdingCash=fillHolding(d, nextd, ls_tradeDates, df_w_holding, df_w_holdingCash, df_w_adjustFactor)

df_w_portfolio_stat = pd.DataFrame()
df_w_portfolio_stat['value'] = (df_w_holding*df_w_markToMarketPrice.ix[ls_tradeDates]).sum(axis=1)
df_w_portfolio_stat['cash'] = df_w_holdingCash
df_w_portfolioValue=df_w_portfolio_stat.sum(axis=1)
df_w_weights=(df_w_holding*df_w_markToMarketPrice.ix[ls_tradeDates]).div(df_w_portfolioValue,axis=0)
#df_w_cumRets=df_w_portfolioValue/num_initialHoldingValue-1.  

df_w_single_period_ret = df_w_portfolioValue/df_w_portfolioValue.shift(1)
df_w_cumRets = df_w_single_period_ret.cumprod()
df_w_portfolioValue.cumprod()
                               
          
