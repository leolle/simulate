# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
from numpy import nan as NA
from lib.gftTools import gftIO
from lib.gftTools import gsUtils
from lib.gftTools import gsConst



def fillHolding(d, nextd, tradeDates, df_holdings, df_holdingCash, df_totalReturnFactor):
    if nextd>d:
        holdingDates = tradeDates[(tradeDates>=d) & (tradeDates<=nextd)]
        df_holdings.ix[holdingDates] = np.tile(df_holdings.ix[d], (len(holdingDates), 1))
        df_holdingCash.ix[holdingDates] = np.tile(df_holdingCash.ix[d], (len(holdingDates), 1))
        df_holdings.ix[holdingDates] = df_holdings.ix[holdingDates] * (df_totalReturnFactor.ix[holdingDates] / df_totalReturnFactor.ix[d])
    result = {'holdings':df_holdings, 'holdingCash':df_holdingCash}
    return result

def SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK(beginDate, endDate, initialHolding, df_targetPortfolioWgt, 
                   df_markToMarketPrice, df_totalReturnFactor, df_executePrice, 
                   df_execPriceReturn, df_tradeVolume, dict_tradingParam, 
                   dict_additionalTs):
    beginDate = beginDate[0]
    endDate = endDate[0]
    df_targetPortfolioWgt = df_targetPortfolioWgt.asMatrix()
    df_markToMarketPrice = df_markToMarketPrice.asMatrix()
    df_totalReturnFactor = df_totalReturnFactor.asMatrix()
    df_executePrice = df_executePrice.asMatrix()
    df_execPriceReturn = df_execPriceReturn.asMatrix()
    df_tradeVolume = df_tradeVolume.asMatrix()
    cashSymbol = gsUtils.getCashGid()
    godGid = gsUtils.getGodGid()

    allDates = df_markToMarketPrice.index
    if len(allDates[(allDates>=beginDate) & (allDates<=endDate)]) < 1:
        raise ValueError('no trading date falls between begindate and enddate')
    endDate = allDates[allDates<=endDate][-1]
    if beginDate > endDate:
        raise ValueError('beginDate should be less than endDate')

    initHldIsCash = True
    if isinstance(initialHolding, gftIO.GftTable):
        df_initialHolding = initialHolding.asMatrix()
        if df_initialHolding.shape[0] < 1:
            raise ValueError('no init holding is provided')
        initHldIsCash = False
        df_initialHolding = df_initialHolding.ix[-1:]
        beginDate = gsUtils.alignDate(df_initialHolding.index[0], allDates, method='ffill')
        if pd.isnull(beginDate):
            raise ValueError('do not have close price for the date of initHld')
        df_initialHolding.index = [beginDate]
    else:
        beginDate = gsUtils.alignDate(beginDate, allDates, method='bfill')
        if pd.isnull(beginDate):
            raise ValueError('beginDate should be less than the last trading date')

    if (df_targetPortfolioWgt<0).any(axis=1).any():
        raise ValueError('Do not support stock short selling and cash borrowing')
    if (round(df_targetPortfolioWgt.sum(1), 4) > 1).any():
        raise ValueError('Total weight is greater than 1.')
    df_targetPortfolioWgt = df_targetPortfolioWgt.dropna(axis=[0,1], how='all')
    sigDates = df_targetPortfolioWgt.index
    rebDates = sigDates
    if len(sigDates) > 0:
        execDelayPeriods = gsUtils.getParm(dict_tradingParam, "execDelayPeriods", 0)
        if execDelayPeriods > 0:
            idxs = np.array(gsUtils.alignDate(sigDates, allDates, method='bfill', returnidx=True), dtype=float)
            idxs = idxs + execDelayPeriods
            idxs[idxs>len(allDates)] = NA
            idxs[idxs == np.append(idxs[1:], NA)] = NA
            idxs_nonnan_flag = np.logical_not(np.isnan(idxs))
            if sum(idxs_nonnan_flag) < 1:
                raise ValueError("no trade date after the execute delay shift")
            df_targetPortfolioWgt = df_targetPortfolioWgt.ix[idxs_nonnan_flag]
            rebDates = allDates[np.array(idxs[idxs_nonnan_flag], dtype=int)]
            df_targetPortfolioWgt.index = rebDates
        if len(rebDates) > 0:
            if initHldIsCash:
                if gsUtils.getParm(dict_tradingParam, "shiftBeginDateToSignal", False):
                    beginDate = rebDates[rebDates>=beginDate][0]
                if pd.isnull(beginDate):
                    raise ValueError('beginDate is null after shift')

    tradeDates = allDates[(allDates>=beginDate) & (allDates<=endDate)]
    beginDate = tradeDates[0]
    endDate = tradeDates[-1]
    if beginDate > endDate:
        raise ValueError("Begin date is larger than end date after the date processing!")
    rebDates = rebDates[(rebDates>=beginDate) & (rebDates<=endDate)]
    df_targetPortfolioWgt = df_targetPortfolioWgt.ix[rebDates]

    allSymbols = np.unique(df_markToMarketPrice.columns)
    portfolioSymbols = np.unique(np.setdiff1d(df_targetPortfolioWgt.columns, cashSymbol))
    holdingSymbols = np.array([])
    if not(initHldIsCash):
        holdingSymbols = np.unique(np.setdiff1d(df_initialHolding.columns, cashSymbol))
    if len(np.setdiff1d(holdingSymbols, allSymbols)) > 0:
        raise ValueError("Initial Portfolio has non A-share stocks!")
    if len(np.setdiff1d(portfolioSymbols, allSymbols)) > 0:
        raise ValueError("Target Portfolio has non A-share stocks!")

    allSymbols = np.unique(np.setdiff1d(np.intersect1d(allSymbols, np.append(holdingSymbols, portfolioSymbols)), cashSymbol))
    priceDates = allDates[(allDates>=beginDate-datetime.timedelta(days=20)) & (allDates<=endDate)]
 
    df_markToMarketPrice = df_markToMarketPrice.reindex(priceDates, allSymbols, fill_value=NA)
    df_totalReturnFactor = df_totalReturnFactor.reindex(priceDates, allSymbols, fill_value=1.).fillna(1.)
    df_executePrice = df_executePrice.reindex(priceDates, allSymbols, fill_value=NA)    
    df_execPriceReturn = df_execPriceReturn.reindex(priceDates, allSymbols, fill_value=NA)   
    df_tradeVolume = df_tradeVolume.reindex(priceDates, allSymbols, fill_value=0.)  

    if initHldIsCash:
        df_initialHolding = pd.DataFrame(initialHolding, index=[beginDate], columns=[cashSymbol])
    df_initialHolding = df_initialHolding.reindex(columns = np.append(allSymbols, cashSymbol)).fillna(0.)
    df_initialHoldingCash = df_initialHolding.ix[:, cashSymbol]    
    df_initialHolding = df_initialHolding.ix[:, allSymbols]
    initHldValue = float((df_initialHolding * df_markToMarketPrice.ix[df_initialHolding.index]).sum(axis=1)) + df_initialHoldingCash.ix[0,0]

    df_targetPortfolioWgt = df_targetPortfolioWgt.reindex(rebDates, allSymbols, fill_value=0.).fillna(0.)

    df_buyVolume = df_tradeVolume.copy().fillna(0)
    df_sellVolume = df_buyVolume.copy()
    if gsUtils.getParm(dict_tradingParam, "canTradeOnSuspend", 0) > 0:
        df_buyVolume[df_buyVolume < 1] = np.inf
        df_sellVolume[df_sellVolume < 1] = np.inf
    riseLimitThres = gsUtils.getParm(dict_tradingParam, "riseLimitThres", 0)
    if riseLimitThres > 0:
        riseLimit = df_execPriceReturn > riseLimitThres
        df_buyVolume[riseLimit] = 0
        df_sellVolume[riseLimit & (df_sellVolume>0)] = np.inf
    fallLimitThres = gsUtils.getParm(dict_tradingParam, "fallLimitThres", 0)        
    if fallLimitThres < 0: 
        fallLimit = df_execPriceReturn < fallLimitThres
        df_buyVolume[fallLimit & (df_buyVolume>0)] = np.inf
        df_sellVolume[fallLimit] = 0
    volumeLimitPct = gsUtils.getParm(dict_tradingParam, "volumeLimitPct", 0)        
    if volumeLimitPct > 0:
        df_buyVolume = df_buyVolume * volumeLimitPct
        df_sellVolume = df_sellVolume * volumeLimitPct
    else: 
        df_buyVolume[df_buyVolume>0] = np.inf
        df_sellVolume[df_sellVolume>0] = np.inf
    
    lotSize = gsUtils.getParm(dict_tradingParam, "lotSize", 0)
    df_buyVolume = gsUtils.roundToLot(df_buyVolume, lotSize)
    df_sellVolume = gsUtils.roundToLot(df_sellVolume, lotSize)
    
    buyCommission = gsUtils.getParm(dict_tradingParam, "buyCommission", 0)
    sellCommission = gsUtils.getParm(dict_tradingParam, "sellCommission", 0)

    df_holdings = pd.DataFrame(0., index = tradeDates, columns = allSymbols)
    df_weights = df_holdings.copy()
    df_execution = df_holdings.copy()
    df_holdingCash = pd.DataFrame(0., index = tradeDates, columns = cashSymbol)
    df_portfolioValue = pd.DataFrame(0., index = tradeDates, columns = godGid)
    df_cumRets = df_portfolioValue.copy()
    df_singlePeriodRets = df_portfolioValue.copy()
    df_turnoverPct = df_portfolioValue.copy()

    d = tradeDates[0]
    df_holdings.ix[d] = df_initialHolding.ix[d]
    df_holdingCash.ix[d] = df_initialHoldingCash.ix[0,0]    
    if len(rebDates) < 1:
        nextd = tradeDates[-1]
        ls_adjustedHoldings = fillHolding(d, nextd, tradeDates, df_holdings, df_holdingCash, df_totalReturnFactor)
        df_holdings = ls_adjustedHoldings['holdings']
        df_holdingCash = ls_adjustedHoldings['holdingCash']
    else:
        nextd = rebDates[0]
        ls_adjustedHoldings = fillHolding(d, nextd, tradeDates, df_holdings, df_holdingCash, df_totalReturnFactor)
        df_holdings = ls_adjustedHoldings['holdings']
        df_holdingCash = ls_adjustedHoldings['holdingCash']
        for i in range(len(rebDates)):
            d = rebDates[i]
            s_currentHoldingValue = df_holdings.ix[d]*df_executePrice.ix[d]
            totalValue = s_currentHoldingValue.sum() + df_holdingCash.ix[d, 0]
            s_currentHoldingWgt = s_currentHoldingValue / totalValue
            s_targetHoldingWgt = df_targetPortfolioWgt.ix[d]
            targetHoldingCashWgt = 1.0 - s_targetHoldingWgt.sum()
            s_orderWgt = s_targetHoldingWgt - s_currentHoldingWgt
            s_sellOrderWgt = s_orderWgt.copy()
            s_sellOrderWgt[s_sellOrderWgt>0.] = 0.
            s_buyOrderWgt = s_orderWgt.copy() 
            s_buyOrderWgt[s_buyOrderWgt<0.]=0.
            cashAvail = df_holdingCash.ix[d, 0]
            if (s_sellOrderWgt<0).any():
                s_sellOrder = gsUtils.roundToLot(s_sellOrderWgt / s_currentHoldingWgt.where(s_currentHoldingWgt>0, 1.0) * df_holdings.ix[d], lotSize)
                s_sellOrder = s_sellOrder.where(s_targetHoldingWgt>0, -df_holdings.ix[d])
                s_sellExecution = s_sellOrder.copy()
                s_sellExecution = -pd.concat([s_sellExecution.fillna(0).abs(), df_sellVolume.ix[d]], axis=1).min(axis=1)
                cashAvail = cashAvail + (s_sellExecution.abs() * df_executePrice.ix[d]).sum() * (1-sellCommission)
                df_execution.ix[d] += s_sellExecution
                df_holdings.ix[d] += s_sellExecution
            if (s_buyOrderWgt>0).any():
                canBuyWgt = cashAvail / totalValue - targetHoldingCashWgt
                if canBuyWgt > 0:
                    s_buyOrder = gsUtils.roundToLot((min(canBuyWgt/s_buyOrderWgt.sum(), 1.0)*s_buyOrderWgt*totalValue/(1+buyCommission)/df_executePrice.ix[d]).fillna(0), lotSize)
                    s_buyExecution = s_buyOrder.copy()
                    s_buyExecution = pd.concat([s_buyExecution.fillna(0), df_buyVolume.ix[d]], axis=1).min(axis=1)
                    cashAvail = cashAvail - (s_buyExecution.abs()*df_executePrice.ix[d]).sum()*(1+buyCommission)
                    df_execution.ix[d] += s_buyExecution
                    df_holdings.ix[d] += s_buyExecution                   
            df_holdingCash.ix[d] = cashAvail
            df_turnoverPct.ix[d] <- (df_execution.ix[d].abs()*df_executePrice.ix[d]).sum() / totalValue

            if i < (len(rebDates)-1):
                nextd = rebDates[i+1]
            else:
                nextd = tradeDates[-1]
            ls_adjustedHoldings = fillHolding(d, nextd, tradeDates, df_holdings, df_holdingCash, df_totalReturnFactor)
            df_holdings = ls_adjustedHoldings['holdings']
            df_holdingCash = ls_adjustedHoldings['holdingCash']
    
    df_portfolioValue.ix[:, 0] = (df_holdings * df_markToMarketPrice.ix[tradeDates]).sum(axis=1) + df_holdingCash.ix[:, 0] 
    df_weights = (df_holdings * df_markToMarketPrice.ix[tradeDates]).div(df_portfolioValue.ix[:,0], axis=0)    
    df_cumRets = df_portfolioValue / initHldValue - 1
    df_singlePeriodRets = df_portfolioValue/ df_portfolioValue.shift(1) - 1
    df_singlePeriodRets.ix[0,0] = df_portfolioValue.ix[0, 0] / initHldValue - 1

    result = {}
    result[gsConst.Const.Holding] = pd.concat([df_holdings.replace(0, NA), df_holdingCash], axis=1)
    result[gsConst.Const.PortfolioValue] = df_portfolioValue
    result[gsConst.Const.Weights] = df_weights.replace(0, NA)
    result[gsConst.Const.SinglePeriodReturn] = df_singlePeriodRets
    result[gsConst.Const.CumulativeReturn] = df_cumRets
    result[gsConst.Const.Turnover] = df_turnoverPct
    print(df_cumRets.ix[-1])      
    return result
                               
          
dataPack = gftIO.zload(r'd:\Wuwei\Project\simulator\data\monthly_rebalance_data_execute_at_next_open_1.pkl')

dataPack['begin_date'] = dataPack.pop('x0')
dataPack['end_date'] = dataPack.pop('x1')
dataPack['initial_holding'] = dataPack.pop('x2')
dataPack['target_portfolio_weight'] = dataPack.pop('x3')
dataPack['market_to_market_price'] = dataPack.pop('x4')
dataPack['total_return_factor'] = dataPack.pop('x5')
dataPack['execute_price'] = dataPack.pop('x6')
dataPack['execute_price_return'] = dataPack.pop('x7')
dataPack['trade_volume'] = dataPack.pop('x8')
dataPack['trading_param'] = dataPack.pop('x9')
dataPack['additional_Ts'] = dataPack.pop('x10')
dataPack['trading_param']['execDelayPeriods'] = 1


result = SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK(dataPack['begin_date'], \
                                    dataPack['end_date'], \
                                    dataPack['initial_holding'], \
                                    dataPack['target_portfolio_weight'], \
                                    dataPack['market_to_market_price'] , \
                                    dataPack['total_return_factor'], \
                                    dataPack['execute_price'], \
                                    dataPack['execute_price_return'], \
                                    dataPack['trade_volume'], \
                                    dataPack['trading_param'], \
                                    dataPack['additional_Ts'])
