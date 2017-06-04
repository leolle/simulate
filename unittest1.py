# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pandas as pd
import numpy as np
from datetime import datetime
from lib.gftTools import gftIO
import matplotlib.pyplot as plt

start = datetime.now()
def parse_data(input_data):
    if type(input_data) is gftIO.GftTable:
        input_data = input_data.asMatrix()
        input_data = input_data.dropna(axis=0, how='all')
        if input_data.columns.dtype == 'datetime64[ns]':
            return input_data.T
        else:
            return input_data
    elif type(input_data) is np.ndarray:
        return input_data.astype(datetime)
    elif type(input_data) is pd.tseries.index.DatetimeIndex:
        return input_data[0]
    else:
        return input_data

data = gftIO.zload(r'd:\Wuwei\Project\simulator\data\monthly_rebalance_data_same_as_R_5.pkl')
data = gftIO.transformDict4Name(data)
for key, value in data.items():
    data[key] = parse_data(value)

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

data['trading_param']['buyCommission'] = 0
data['trading_param']['sellCommission'] = 0

def addMissingColumns(df_w, ls_columns, fill_value = 0):
    """补充到完整的symbol list
    """
    df_list = df_w.columns.tolist()
    ls_tmp = [s for s in ls_columns if s not in df_list]
    if fill_value == 0:
        df_w_tmp = pd.DataFrame(np.zeros((len(df_w),len(ls_tmp))),index=df_w.index,columns=ls_tmp)
    else:
        df_w_tmp = pd.DataFrame(np.ones((len(df_w),len(ls_tmp))),index=df_w.index,columns=ls_tmp)
    return pd.concat([df_w_tmp,df_w],axis=1)

def fill_holding(d, nextd, ls_tradeDates, df_holding, df_holdingCash, df_w_adjustFactor):
    """填充rebalance date_1 , rebalance date_2之间的持仓数据
    """
    if nextd > d:
        ls_holdingDates = [date for date in ls_tradeDates if date>=d and date<=nextd]
        ls_tmp_holding = df_holding.loc[d].values
        narray_tmp_holding = np.tile(ls_tmp_holding,(len(ls_holdingDates),1))
        df_w_tmp_holding = pd.DataFrame(narray_tmp_holding, \
                                        index=ls_holdingDates, \
                                        columns=df_holding.columns)

        ls_tmp_holdingCash = df_holdingCash.ix[d].values
        narray_tmp_holdingCash = np.tile(ls_tmp_holdingCash,(len(ls_holdingDates),1))
        df_w_tmp_holdingCash = pd.DataFrame(narray_tmp_holdingCash, \
                                            index=ls_holdingDates, \
                                            columns=df_holdingCash.columns)

        df_w_tmp_holding = df_w_tmp_holding*(df_w_adjustFactor.loc[ls_holdingDates]/\
                                             df_w_adjustFactor.loc[d])
        df_holding.ix[ls_holdingDates] = df_w_tmp_holding
        df_holdingCash.ix[ls_holdingDates] = df_w_tmp_holdingCash

    return df_holding, df_holdingCash

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

def get_param(ls_param, name, default_value):
    """ getting parameter value from the list, if not, return default value.
    """
    result = default_value
    if name in ls_param:
        result = ls_param[name]
    return result

#todo, decimal precision
def round_to_lot(size_array, lot_size):
    """round number to the nearest lot_size
    There could be a problem that 1200.8 is rounded to 1100.
    """
    if lot_size > 0:
        size_array = size_array.fillna(0)
        return (np.sign(size_array) * lot_size * np.floor(np.round(abs(size_array)) / max(1, lot_size)))
    else:
        return size_array



dict_tradingParam = data['trading_param']

df_w_markToMarketPrice = pd.read_excel(r'd:\Wuwei\Project\simulator\data\price.xlsx', index_col = 0)
df_w_markToMarketPrice.index = pd.to_datetime(df_w_markToMarketPrice.index)
df_w_totalReturnFactor = pd.read_excel(r'd:\Wuwei\Project\simulator\data\return_factor.xlsx', index_col = 0)
df_w_totalReturnFactor.index = pd.to_datetime(df_w_totalReturnFactor.index)
df_w_executePrice = pd.read_excel(r'd:\Wuwei\Project\simulator\data\execute_price.xlsx', index_col = 0)
df_w_executePrice.index = pd.to_datetime(df_w_executePrice.index)
df_w_execPriceReturn = pd.read_excel(r'd:\Wuwei\Project\simulator\data\execute_price_return.xlsx', index_col = 0)
df_w_execPriceReturn.index = pd.to_datetime(df_w_execPriceReturn.index)
df_w_tradeVolume = pd.read_excel(r'd:\Wuwei\Project\simulator\data\trade_volume.xlsx', index_col = 0)
df_w_tradeVolume.index = pd.to_datetime(df_w_tradeVolume.index)
df_w_targetPortfolioWgt = pd.read_excel(r'd:\Wuwei\Project\simulator\data\target_portfolio_weight.xlsx', index_col = 0)
df_w_targetPortfolioWgt.index = pd.to_datetime(df_w_targetPortfolioWgt.index)
#initialHolding = df_w_tradeVolume.ix[0]
#initialHolding['Cash'] =  10000000

initialHolding = 1000000.0
ls_allDates = sorted(list(set.intersection(set(df_w_markToMarketPrice.index),set(df_w_totalReturnFactor.index),set(df_w_executePrice.index),set(df_w_execPriceReturn.index),set(df_w_tradeVolume.index))))

dt_beginDate = datetime(2000, 1, 1)
dt_endDate = datetime(2000, 1, 8)
ls_cashGid = ['Cash']
if len(df_w_targetPortfolioWgt)>0:
    df_w_targetPortfolioWgt = df_w_targetPortfolioWgt.ix[ls_allDates]
    df_w_targetPortfolioWgt = df_w_targetPortfolioWgt.dropna(how='all')
    ls_rebDates=df_w_targetPortfolioWgt.index.tolist()
    if len(ls_rebDates)>0: #############  and   (1)
        ls_rebDates=ls_rebDates ############# + deltaTime (2)

        ls_rebDates=alignDate(ls_allDates,ls_rebDates)
        df_w_targetPortfolioWgt=df_w_targetPortfolioWgt.loc[[x for x in ls_rebDates if str(x)!='nan']]
        ls_rebDates=[x for x in ls_rebDates if str(x)!='nan']
        df_w_targetPortfolioWgt.index=ls_rebDates

ls_holdingSymbols = []
if isinstance(initialHolding,pd.DataFrame):
    dt_beginDate = initialHolding.index[-1]
    ls_holdingSymbols = sorted(list(set([i for i in initialHolding.columns if i not in ls_cashGid])))
else:
    if len(df_w_targetPortfolioWgt) > 0: #dict_tradingParam['shiftBeginDateToSignal']>0 and(4)
        if dt_beginDate > df_w_targetPortfolioWgt.index[0]:
            dt_beginDate = dt_beginDate
        else:
            dt_beginDate = df_w_targetPortfolioWgt.index[0]
#        dt_beginDate = max(dt_beginDate, df_w_targetPortfolioWgt.index[0])
ls_tradeDates=[date for date in ls_allDates if date>=dt_beginDate and date<=dt_endDate]
dt_beginDate=ls_tradeDates[0]
dt_endDate=ls_tradeDates[-1]
ls_rebDates=[date for date in ls_rebDates if date>=dt_beginDate and date<=dt_endDate]
if (dt_beginDate > dt_endDate):
    print("Begin date must be less than end date!" )

ls_allSymbols = sorted(list(set.intersection(set(df_w_markToMarketPrice.columns),set(df_w_executePrice.columns),set(df_w_execPriceReturn.columns),set(df_w_tradeVolume.columns))))
df_w_tmp_targetPortfolioWgt = df_w_targetPortfolioWgt.dropna(how='all',axis=1)
df_w_tmp_targetPortfolioWgt = df_w_tmp_targetPortfolioWgt.loc[:,(df_w_tmp_targetPortfolioWgt!=0).any(axis=0)]
ls_portfolioSymbols = [s for s in df_w_tmp_targetPortfolioWgt.columns.tolist() if s not in ls_cashGid]

if len([s for s in ls_holdingSymbols if s not in ls_allSymbols])>0:
#if ls_holdingSymbols not in ls_allSymbols:
    print("Initial Portfolio has non A-share stocks!")
    #return "Initial Portfolio has non A-share stocks!",[s for s in ls_allSymbols if s not in ls_holdingSymbols]
if len([s for s in ls_portfolioSymbols if s not in ls_allSymbols])>0:
#if ls_portfolioSymbols not in ls_allSymbols:
    print("Target Portfolio has non A-share stocks! ")
    #return "Target Portfolio has non A-share stocks! ",[s for s in ls_allSymbols if s not in ls_portfolioSymbols]

# todo: process holding symbols
ls_allSymbols = sorted([s for s in set.intersection(set(ls_allSymbols),set(ls_portfolioSymbols)) if s not in ls_cashGid])
ls_priceDates = [d for d in ls_allDates if d >=(dt_beginDate-pd.Timedelta('20 days')) and d<=dt_endDate]
df_w_markToMarketPrice = df_w_markToMarketPrice.loc[ls_priceDates][ls_allSymbols]
df_w_totalReturnFactor = addMissingColumns(df_w_totalReturnFactor, ls_allSymbols, 1).loc[ls_priceDates, ls_allSymbols]
df_w_executePrice = df_w_executePrice.loc[ls_priceDates][ls_allSymbols]
df_w_execPriceReturn = df_w_execPriceReturn.loc[ls_priceDates][ls_allSymbols]
df_w_tradeVolume = df_w_tradeVolume.loc[ls_priceDates][ls_allSymbols]

df_w_initialHolding=initialHolding
if not isinstance(initialHolding,pd.DataFrame):
    df_w_initialHolding=pd.DataFrame(initialHolding,index=[dt_beginDate],columns=ls_cashGid)
#if type(initialHolding)==int or type(initialHolding)==float:

df_w_initialHolding = addMissingColumns(df_w_initialHolding,ls_allSymbols)
df_w_initialHoldingCash = df_w_initialHolding.loc[df_w_initialHolding.index][ls_cashGid]
df_w_initialHolding.pop(ls_cashGid[0])
num_initialHoldingValue = float((df_w_initialHolding*df_w_markToMarketPrice.ix[df_w_initialHolding.index]).sum(1)+df_w_initialHoldingCash.values[-1][-1])
df_w_targetPortfolioWgt = addMissingColumns(df_w_targetPortfolioWgt,ls_allSymbols+ls_cashGid).loc[ls_rebDates].fillna(0)

if (df_w_targetPortfolioWgt < 0).any().any():
    print("Do not support stock short selling and cash borrowing")
#    #return "Do not support stock short selling and cash borrowing"
df_w_targetPortfolioCashWgt = df_w_targetPortfolioWgt[ls_cashGid]
df_w_targetPortfolioWgt.pop(ls_cashGid[0])
df_w_targetPortfolioCashWgt = 1.-df_w_targetPortfolioWgt.sum(axis=1)

df_w_buyVolume = df_w_tradeVolume.copy().fillna(0)
df_w_sellVolumn = df_w_tradeVolume.copy().fillna(0)

num_lotSize = get_param(data['trading_param'], 'lotSize', 0)
df_w_buyVolume = round_to_lot(df_w_buyVolume, num_lotSize)
df_w_sellVolumn = round_to_lot(df_w_sellVolumn, num_lotSize)


if data['trading_param']['canTradeOnSuspend'] > 0:  #np.sum((dict_tradingParam['canTradeOnSuspend'] > 0).values.ravel())
    df_w_buyVolume[df_w_buyVolume < 1] = np.inf
    df_w_sellVolumn[df_w_sellVolumn < 1] = np.inf

if data['trading_param']['riseLimitThres'] > 0:
    rise_limit = df_w_execPriceReturn > data['trading_param']['riseLimitThres']
    df_w_buyVolume[rise_limit] = 0
    df_w_sellVolumn[rise_limit & (df_w_sellVolumn > 0)] = np.inf

if data['trading_param']['fallLimitThres'] < 0:
    fall_limit = df_w_execPriceReturn < data['trading_param']['fallLimitThres']
    df_w_buyVolume[fall_limit & (df_w_buyVolume > 0)] = np.inf
    df_w_sellVolumn[fall_limit] = 0

if data['trading_param']['volumeLimitPct'] > 0:
    df_w_buyVolume = df_w_buyVolume * data['trading_param']['volumeLimitPct']
    df_w_sellVolumn = df_w_sellVolumn * data['trading_param']['volumeLimitPct']
else:
    df_w_buyVolume[df_w_buyVolume > 0] = np.inf
    df_w_sellVolumn[df_w_sellVolumn > 0] = np.inf

num_buyCommission = data['trading_param']['buyCommission'] if \
                        data['trading_param']['buyCommission'] else 0.0
num_sellCommission = data['trading_param']['sellCommission'] if \
                         data['trading_param']['sellCommission'] else 0.0

df_w_holding = pd.DataFrame(0.,index=ls_tradeDates,columns=ls_allSymbols)
df_w_weights = df_w_holding.copy()
df_w_execution = df_w_holding.copy()
df_w_holdingCash = pd.DataFrame(0.,index=ls_tradeDates,columns=ls_cashGid)
ls_getGodGid=['getGodGid']
df_w_portfolioValue = pd.DataFrame(0.,index=ls_tradeDates,columns=ls_getGodGid)
df_w_cumRets = df_w_portfolioValue.copy()
df_w_singlePeriodRets = df_w_portfolioValue.copy()
df_w_turnoverPct = df_w_portfolioValue.copy()

## trading
d = ls_tradeDates[0]
df_w_holding.ix[d] = df_w_initialHolding.loc[d].tolist()
df_w_holdingCash.ix[d] = df_w_initialHoldingCash.values[-1][-1]


if len(ls_rebDates)<1:
    nextd = ls_tradeDates[-1]
    df_w_holding, df_w_holdingCash = fill_holding(d, nextd, ls_tradeDates, \
                                                 df_w_holding, df_w_holdingCash, \
                                                 df_w_totalReturnFactor)
else:
    nextd = ls_rebDates[0]
    df_w_holding, df_w_holdingCash=fill_holding(d, nextd, ls_tradeDates, \
                                               df_w_holding, df_w_holdingCash, \
                                               df_w_totalReturnFactor)

    for i in range(len(ls_rebDates)):
        d = ls_rebDates[i]
        df_w_currentHoldingValue = df_w_holding.ix[d]*(df_w_executePrice.ix[d].fillna(0)) # one line
        num_totalValue = df_w_currentHoldingValue.sum() + df_w_holdingCash.ix[d].values[-1]

        df_w_currentHoldingWgt = df_w_currentHoldingValue / num_totalValue
        df_w_curHoldingCashWgt = 1. - df_w_currentHoldingWgt.sum()
        df_w_targetHoldingWgt = df_w_targetPortfolioWgt.ix[d]
        num_targetHoldingCashWgt = 1. - df_w_targetHoldingWgt.sum()
        df_w_orderWgt = df_w_targetHoldingWgt - df_w_currentHoldingWgt

        df_w_sellOrderWgt = df_w_orderWgt.copy()
        df_w_sellOrderWgt[df_w_orderWgt >= 0.0] = 0.0

        df_w_buyOrderWgt = df_w_orderWgt.copy()
        df_w_buyOrderWgt[df_w_orderWgt <= 0.0] = 0.0

        num_cashAvail = df_w_holdingCash.ix[d].values[-1]
        # sell
        if (df_w_sellOrderWgt.dropna() < 0.0).any():
            df_w_currentHoldingWgt_forSell = df_w_currentHoldingWgt.copy().fillna(0)
            df_w_currentHoldingWgt_forSell[df_w_currentHoldingWgt_forSell <= 0.0] = 1.0
            tmp_1 = df_w_sellOrderWgt / df_w_currentHoldingWgt_forSell * df_w_holding.ix[d]
            df_w_sellOrder = round_to_lot(tmp_1, num_lotSize) # share
#            df_w_sellOrder = tmp_1
            # sell all if target holding weight is equal to 0
            df_w_sellOrder[df_w_targetHoldingWgt <= 0.0] = -df_w_holding.ix[d]

            df_w_sellExecution = df_w_sellOrder.copy()
            df_w_sellExecution = -pd.concat([df_w_sellExecution.fillna(0).abs(), \
                                             df_w_sellVolumn.ix[d]], axis=1).min(axis=1)
            num_cashAvail = num_cashAvail + (df_w_sellExecution.abs() * \
                                           df_w_executePrice.ix[d]).sum() * (1. - num_sellCommission)
            df_w_execution.ix[d] = df_w_execution.ix[d] + df_w_sellExecution
            df_w_holding.ix[d] = df_w_holding.ix[d] + df_w_sellExecution
        # buy
        if (df_w_buyOrderWgt > 0.0).any():
            num_canBuyWgt = num_cashAvail / num_totalValue - num_targetHoldingCashWgt
            if num_canBuyWgt > 0:
                num_pct = min(num_canBuyWgt / df_w_buyOrderWgt.sum(), 1)
                tmp_2 = (num_pct * df_w_buyOrderWgt * num_totalValue / (1.+num_buyCommission) / df_w_executePrice.ix[d]).fillna(0)
                df_w_buyOrder = round_to_lot(tmp_2, num_lotSize)
#                df_w_buyOrder = tmp_2
                df_w_buyOrder = df_w_buyOrder.fillna(0)
                df_w_buyExecution = df_w_buyOrder.copy() # redundant
                df_w_buyExecution = pd.concat([df_w_buyExecution.fillna(0), \
                                               df_w_buyVolume.ix[d]],axis=1).min(axis=1)
                num_cashAvail = num_cashAvail - (abs(df_w_buyExecution) * \
                                                 df_w_executePrice.ix[d]).sum() * \
                                                 (1.+num_buyCommission)
                df_w_execution.ix[d] = df_w_execution.ix[d] + df_w_buyExecution
                df_w_holding.ix[d] = df_w_holding.ix[d] + df_w_buyExecution


        df_w_holdingCash.ix[d] = num_cashAvail
        df_w_turnoverPct.ix[d] = (abs(df_w_execution.ix[d])*df_w_executePrice.ix[d]).sum()/num_totalValue

        if i < (len(ls_rebDates) - 1):
            nextd = ls_rebDates[i+1]
            df_w_holding, df_w_holdingCash = fill_holding(d, nextd, ls_tradeDates, \
                                                         df_w_holding, df_w_holdingCash, \
                                                         df_w_totalReturnFactor)
    # loop to the next day.
    nextd = ls_tradeDates[-1]
    df_w_holding, df_w_holdingCash = fill_holding(d, nextd, ls_tradeDates, \
                                                 df_w_holding, df_w_holdingCash, \
                                                 df_w_totalReturnFactor)

df_w_portfolio_stat = pd.DataFrame()
df_w_portfolio_stat['value'] = (df_w_holding*df_w_markToMarketPrice.ix[ls_tradeDates]).sum(axis=1)
df_w_portfolio_stat['cash'] = df_w_holdingCash
df_w_portfolioValue = df_w_portfolio_stat.sum(axis=1)
df_w_weights = (df_w_holding*df_w_markToMarketPrice.ix[ls_tradeDates]).div(df_w_portfolioValue,axis=0)
df_w_single_period_ret = df_w_portfolioValue/df_w_portfolioValue.shift(1)
df_w_cumRets = df_w_single_period_ret.cumprod()
#df_w_portfolioValue.plot()
df_w_cumRets.plot(title='portfolio cumulative return(%)')
print(df_w_cumRets[-1])
print(datetime.now() - start)