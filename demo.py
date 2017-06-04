# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pandas as pd
import numpy as np
from datetime import datetime
from lib.gftTools import gftIO
import matplotlib.pyplot as plt
import time

def addMissingColumns(df_w, ls_columns, fill_value = 0):
    """补充到完整的symbol list
    change to reindex, fill_value=na/0
    calculate proccessing time
    """
    df_list = df_w.columns.tolist()
    ls_tmp = [s for s in ls_columns if s not in df_list]
    if fill_value == 0:
        df_w_tmp = pd.DataFrame(np.zeros((len(df_w),len(ls_tmp))),index=df_w.index,columns=ls_tmp)
    else:    
        df_w_tmp = pd.DataFrame(np.ones((len(df_w),len(ls_tmp))),index=df_w.index,columns=ls_tmp)
    return pd.concat([df_w_tmp,df_w],axis=1)
    
def fillHolding(d, nextd, ls_tradeDates, df_holding, df_holdingCash, df_w_adjustFactor):
    """填充rebalance date_1 , rebalance date_2之间的持仓数据
    resample, factor as adjust factor
    change parameter name
    """
    if nextd > d:
        # ls_trade_date[d, nextd], index array
        ls_holdingDates = [date for date in ls_tradeDates if date>=d and date<=nextd]
        ls_tmp_holding = df_holding.loc[d].tolist()
        # change to series
        narray_tmp_holding = np.tile(ls_tmp_holding,(len(ls_holdingDates),1))                            
        df_w_tmp_holding = pd.DataFrame(narray_tmp_holding, index=ls_holdingDates, columns=df_holding.columns.tolist())
        
        # no dividends paid, cash is not changed.
        # calculate daily interest based on risk free interest rate.
        ls_tmp_holdingCash = df_holdingCash.loc[d].tolist()
        narray_tmp_holdingCash = np.tile(ls_tmp_holdingCash,(len(ls_holdingDates),1))                            
        df_w_tmp_holdingCash = pd.DataFrame(narray_tmp_holdingCash, index=ls_holdingDates, columns=df_holdingCash.columns.tolist())
        
        df_w_tmp_holding = df_w_tmp_holding*(df_w_adjustFactor.loc[ls_holdingDates]/df_w_adjustFactor.loc[d])
        df_holding.ix[ls_holdingDates] = df_w_tmp_holding
        df_holdingCash.ix[ls_holdingDates] = df_w_tmp_holdingCash
                                 
    return df_holding, df_holdingCash

def alignDate(a,b):
    """
    find a function to get the next business day.
    """
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

def round_to_lot(size_array, lot_size):
    """
    if lot size = 0, 0 / int = 0
    """
    return (np.sign(size_array) * np.floor(abs(size_array) / max(1, lot_size)) * lot_size)

# remove code below
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

#dataPack = gftIO.zload("SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK.SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK.11.pkl")
dataPack = gftIO.zload(r'd:\Wuwei\Project\simulator\data\monthly_rebalance_data_same_as_R_4.pkl')
for key, value in dataPack.items():
    dataPack[key] = parse_data(value)
x0 = dataPack['x0']
x1 = dataPack['x1']
x2 = dataPack['x2']
x3 = dataPack['x3']
x4 = dataPack['x4']
x5 = dataPack['x5']
x6 = dataPack['x6']
x7 = dataPack['x7']
x8 = dataPack['x8']
x9 = dataPack['x9']
x10 = dataPack['x10']

def simple_simulate_daily_trade_chn_stk(dt_begin, dt_end, initialHolding, \
                                        df_w_targetPortfolioWgt, df_w_markToMarketPrice, \
                                        df_w_totalReturnFactor, df_w_executePrice, \
                                        df_w_execPriceReturn, df_w_tradeVolume, \
                                        dict_tradingParam, additionalTs):
    """股票模拟交易
    input:
        dt_begin, 开始交易日期，在rebalance之前，repeat initial holding,
                                    在rebalance之后，从begin date开始交易。
        dt_end, 结束日期
        initialHolding, 输入持仓，可能为一个dataframe, 也有可能是一个数量cash
        df_w_targetPortfolioWgt, 目标持仓权重，每行总和应等于1
        df_w_markToMarketPrice, 股票价格
        df_w_totalReturnFactor, 复权因子计算
        df_w_executePrice, 交易股票执行价格
        df_w_execPriceReturn, 每只股票交易日的回报
        df_w_tradeVolume, 每只股票交易日的交易量
        dict_tradingParam, 交易参数
        additionalTs，月平衡策略参数
    output:
        result, 字典
            result['HOLDING'], 所有股票持仓和现金数据
            result['PORTFOLIO_VALUE']， 组合每日总价值
            result['SINGLE_PERIOD_RETURN']， 组合每日回报
            result['WEIGHTS']，组合中各项权重
            result['CUMULATIVE_RETURN']，累积收益
            result['TURNOVER']，每日换手率

    """
    
    ls_allDates = sorted(list(set.intersection(set(df_w_markToMarketPrice.index),set(df_w_totalReturnFactor.index),set(df_w_executePrice.index),set(df_w_execPriceReturn.index),set(df_w_tradeVolume.index))))
    
    # holding['Cash'] included?
    # 
    dt_begin = pd.to_datetime(dt_begin)
    dt_end = pd.to_datetime(dt_end)
    ls_cashGid = ['Cash']
    if len(df_w_targetPortfolioWgt)>0:
        # 
        df_w_targetPortfolioWgt = df_w_targetPortfolioWgt.ix[ls_allDates]
#        df_w_targetPortfolioWgt = df_w_targetPortfolioWgt.dropna(how='all')
        ls_rebDates = df_w_targetPortfolioWgt.index.tolist()
        if len(ls_rebDates)>0: #############  and   (1)
            ls_rebDates = ls_rebDates ############# + deltaTime (2)
              
            ls_rebDates=alignDate(ls_allDates,ls_rebDates)
            df_w_targetPortfolioWgt = df_w_targetPortfolioWgt.loc[[x for x in ls_rebDates if str(x)!='nan']]
            # do not use list
            ls_rebDates = [x for x in ls_rebDates if str(x)!='nan']
            df_w_targetPortfolioWgt.index=ls_rebDates
    
    ls_holdingSymbols = []
    # assuming initial holding is a dataframe
    if isinstance(initialHolding, pd.DataFrame):
        dt_begin = initialHolding.index[-1]
        # holding symbols to index.
        ls_holdingSymbols = sorted(list(set([i for i in initialHolding.columns if i not in ls_cashGid])))
    else:
        if len(df_w_targetPortfolioWgt) > 0: #dict_tradingParam['shiftBeginDateToSignal']>0 and(4)
            if dt_begin > df_w_targetPortfolioWgt.index[0]:
                dt_begin = dt_begin
            else:
                dt_begin = df_w_targetPortfolioWgt.index[0] 
            
    # use unique function
    ls_tradeDates = [date for date in ls_allDates if date>=dt_begin and date<=dt_end]
    dt_begin = ls_tradeDates[0]
    dt_end = ls_tradeDates[-1]
    ls_rebDates = [date for date in ls_rebDates if date>=dt_begin and date<=dt_end]
    if (dt_begin > dt_end):
        raise ValueError('input error! Begin date must be less than end date!')

    ls_allSymbols = sorted(list(set.intersection(set(df_w_markToMarketPrice.columns),set(df_w_executePrice.columns),set(df_w_execPriceReturn.columns),set(df_w_tradeVolume.columns))))  
    df_w_tmp_targetPortfolioWgt = df_w_targetPortfolioWgt.dropna(how='all',axis=1)
    df_w_tmp_targetPortfolioWgt = df_w_tmp_targetPortfolioWgt.loc[:,(df_w_tmp_targetPortfolioWgt!=0).any(axis=0)]
    ls_portfolioSymbols = [s for s in df_w_tmp_targetPortfolioWgt.columns.tolist() if s not in ls_cashGid]
    
    if len([s for s in ls_holdingSymbols if s not in ls_allSymbols])>0:
    #if ls_holdingSymbols not in ls_allSymbols:
        raise ValueError('input error! Initial Portfolio has non A-share stocks!')
        #return "Initial Portfolio has non A-share stocks!",[s for s in ls_allSymbols if s not in ls_holdingSymbols]
    if len([s for s in ls_portfolioSymbols if s not in ls_allSymbols])>0:
    #if ls_portfolioSymbols not in ls_allSymbols:
        raise ValueError('input error! Target Portfolio has non A-share stocks!')
        #return "Target Portfolio has non A-share stocks! ",[s for s in ls_allSymbols if s not in ls_portfolioSymbols]
    
    # todo: process holding symbols
    ls_allSymbols = sorted([s for s in set.intersection(set(ls_allSymbols),\
                                                        set(ls_portfolioSymbols)) if s not in ls_cashGid])
    ls_priceDates = [d for d in ls_allDates if d >=(dt_begin-pd.Timedelta('20 days')) and d<=dt_end]
    df_w_markToMarketPrice = df_w_markToMarketPrice.loc[ls_priceDates][ls_allSymbols]
    df_w_totalReturnFactor = addMissingColumns(df_w_totalReturnFactor, ls_allSymbols, 1).loc[ls_priceDates, ls_allSymbols]
    df_w_executePrice = df_w_executePrice.loc[ls_priceDates][ls_allSymbols]
    df_w_execPriceReturn = df_w_execPriceReturn.loc[ls_priceDates][ls_allSymbols]
    df_w_tradeVolume = df_w_tradeVolume.loc[ls_priceDates][ls_allSymbols]          
    
    df_w_initialHolding=initialHolding
    if not isinstance(initialHolding,pd.DataFrame):
        df_w_initialHolding=pd.DataFrame(initialHolding,index=[dt_begin],columns=ls_cashGid)
    #if type(initialHolding)==int or type(initialHolding)==float:
    
    df_w_initialHolding = addMissingColumns(df_w_initialHolding,ls_allSymbols)
    df_w_initialHoldingCash = df_w_initialHolding.loc[df_w_initialHolding.index][ls_cashGid]
    df_w_initialHolding.pop(ls_cashGid[0])
    num_initialHoldingValue = float((df_w_initialHolding*df_w_markToMarketPrice.ix[df_w_initialHolding.index]).sum(1)+df_w_initialHoldingCash.values[-1][-1])
    df_w_targetPortfolioWgt = addMissingColumns(df_w_targetPortfolioWgt,ls_allSymbols+ls_cashGid).loc[ls_rebDates].fillna(0)
    
    if (df_w_targetPortfolioWgt < 0).any().any():
        raise ValueError('input error! Do not support stock short selling and cash borrowing.')
    #    #return "Do not support stock short selling and cash borrowing"
    df_w_targetPortfolioCashWgt = df_w_targetPortfolioWgt[ls_cashGid]
    df_w_targetPortfolioWgt.pop(ls_cashGid[0]) # get the first item in the list
    df_w_targetPortfolioCashWgt = 1.-df_w_targetPortfolioWgt.sum(axis=1)
    
    df_w_buyVolume = df_w_tradeVolume.copy().fillna(0)
    df_w_sellVolumn = df_w_tradeVolume.copy().fillna(0)
    
    # if there's trading parameter in the input
    # add get parameter function
    if dict_tradingParam['canTradeOnSuspend'] > 0:  #np.sum((dict_tradingParam['canTradeOnSuspend'] > 0).values.ravel())
        df_w_buyVolume[df_w_buyVolume < 1] = np.inf
        df_w_sellVolumn[df_w_sellVolumn < 1] = np.inf
                       
    if dict_tradingParam['riseLimitThres'] > 0:
        rise_limit = df_w_execPriceReturn > dict_tradingParam['riseLimitThres'] 
        df_w_buyVolume[rise_limit] = 0
        df_w_sellVolumn[rise_limit & (df_w_sellVolumn > 0)] = np.inf
        
    if dict_tradingParam['fallLimitThres'] < 0:
        fall_limit = df_w_execPriceReturn < dict_tradingParam['fallLimitThres'] 
        df_w_buyVolume[fall_limit & (df_w_buyVolume > 0)] = np.inf
        df_w_sellVolumn[fall_limit] = 0
    
    if dict_tradingParam['volumeLimitPct'] > 0:
        df_w_buyVolume = df_w_buyVolume * dict_tradingParam['volumeLimitPct']
        df_w_sellVolumn = df_w_sellVolumn * dict_tradingParam['volumeLimitPct']
    else:
        df_w_buyVolume[df_w_buyVolume > 0] = np.inf
        df_w_sellVolumn[df_w_sellVolumn > 0] = np.inf
    
    num_lotSize = dict_tradingParam['lotSize']
    df_w_buyVolume = round_to_lot(df_w_buyVolume, num_lotSize)
    df_w_sellVolumn = round_to_lot(df_w_sellVolumn, num_lotSize)
    
    num_buyCommission = dict_tradingParam['buyCommission'] if \
                            dict_tradingParam['buyCommission'] else 0.0
    num_sellCommission = dict_tradingParam['sellCommission'] if \
                             dict_tradingParam['sellCommission'] else 0.0
    
    df_w_holding = pd.DataFrame(0.,index=ls_tradeDates,columns=ls_allSymbols)
    df_w_weights = df_w_holding.copy()
    df_w_execution = df_w_holding.copy()
    df_w_holdingCash = pd.DataFrame(0.,index=ls_tradeDates,columns=ls_cashGid)
    
    # process god gid, replace in the dataframe?
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
        df_w_holding, df_w_holdingCash = fillHolding(d, nextd, ls_tradeDates, \
                                                     df_w_holding, df_w_holdingCash, \
                                                     df_w_totalReturnFactor)
    else:
        nextd = ls_rebDates[0]
        df_w_holding, df_w_holdingCash=fillHolding(d, nextd, ls_tradeDates, \
                                                   df_w_holding, df_w_holdingCash, \
                                                   df_w_totalReturnFactor)
        # not use loop, Ricequant, Zipline logitics
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
            if (df_w_sellOrderWgt < 0.0).any():
                df_w_currentHoldingWgt_forSell = df_w_currentHoldingWgt.copy().fillna(0)
                df_w_currentHoldingWgt_forSell[df_w_currentHoldingWgt_forSell <= 0.0] = 1.0
                tmp_1 = df_w_sellOrderWgt / df_w_currentHoldingWgt_forSell * df_w_holding.ix[d]
                df_w_sellOrder = round_to_lot(tmp_1, num_lotSize) # share
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
                df_w_holding, df_w_holdingCash = fillHolding(d, nextd, ls_tradeDates, \
                                                             df_w_holding, df_w_holdingCash, \
                                                             df_w_totalReturnFactor)
        # loop to the next day.                
        nextd = ls_tradeDates[-1]
        df_w_holding, df_w_holdingCash = fillHolding(d, nextd, ls_tradeDates, \
                                                     df_w_holding, df_w_holdingCash, \
                                                     df_w_totalReturnFactor)
    result = {}
    df_w_portfolio_stat = pd.DataFrame()
    df_w_portfolio_stat['value'] = (df_w_holding*df_w_markToMarketPrice.ix[ls_tradeDates]).sum(axis=1)
    df_w_portfolio_stat['cash'] = df_w_holdingCash
    df_w_portfolioValue = df_w_portfolio_stat.sum(axis=1)
    df_w_weights = (df_w_holding*df_w_markToMarketPrice.ix[ls_tradeDates]).div(df_w_portfolioValue,axis=0)
    df_w_single_period_ret = df_w_portfolioValue/df_w_portfolioValue.shift(1)
    df_w_cumRets = df_w_single_period_ret.cumprod()
    
    result['HOLDING'] = pd.concat([df_w_holding, df_w_holdingCash], axis=1)
    result['PORTFOLIO_VALUE'] = df_w_portfolioValue
    result['SINGLE_PERIOD_RETURN'] = df_w_single_period_ret
    result['WEIGHTS'] = df_w_weights
    result['CUMULATIVE_RETURN'] = df_w_cumRets
    result['TURNOVER'] = df_w_turnoverPct
          
    #df_w_portfolioValue.plot()
#    df_w_cumRets.plot(figsize=(8,12),title='portfolio cumulative return(%)')
    
    return result

if __name__ == "__main__":    
    time_begin = datetime.now()
    simple_simulate_daily_trade_chn_stk(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
    time_end = datetime.now()
    print ('time cost', time_end - time_begin)
