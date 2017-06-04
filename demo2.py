# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pandas as pd
import numpy as np
from datetime import datetime
from lib.gftTools import gftIO, gsUtils
# import matplotlib.pyplot as plt


def add_missing_columns(df_w, ls_columns, fill_value = 0):
    """补充到完整的symbol list
    """
    df_list = df_w.columns.tolist()
    ls_tmp = [s for s in ls_columns if s not in df_list]
    if fill_value == 0:
        df_w_tmp = pd.DataFrame(np.zeros((len(df_w),len(ls_tmp))), \
                                index=df_w.index, columns=ls_tmp)
    else:
        df_w_tmp = pd.DataFrame(np.ones((len(df_w),len(ls_tmp))), \
                                index=df_w.index, columns=ls_tmp)
    return pd.concat([df_w_tmp,df_w],axis=1)

def fill_holding(d, nextd, ls_trade_dates, df_holding, df_holding_cash, df_w_adjustFactor):
    """填充rebalance date_1 , rebalance date_2之间的持仓数据
    """
    if nextd > d:
        ls_holdingDates = [date for date in ls_trade_dates if date>=d and date<=nextd]
        ls_tmp_holding = df_holding.ix[d].values
        narray_tmp_holding = np.tile(ls_tmp_holding,(len(ls_holdingDates),1))
        df_w_tmp_holding = pd.DataFrame(narray_tmp_holding, \
                                        index=ls_holdingDates, \
                                        columns=df_holding.columns)

        ls_tmp_holdingCash = df_holding_cash.ix[d].values
        narray_tmp_holdingCash = np.tile(ls_tmp_holdingCash,(len(ls_holdingDates),1))
        df_w_tmp_holdingCash = pd.DataFrame(narray_tmp_holdingCash, \
                                            index=ls_holdingDates, \
                                            columns=df_holding_cash.columns)

        df_w_tmp_holding = df_w_tmp_holding*(df_w_adjustFactor.ix[ls_holdingDates]/\
                                             df_w_adjustFactor.ix[ls_holdingDates[0]])
        df_holding.ix[ls_holdingDates] = df_w_tmp_holding
        df_holding_cash.ix[ls_holdingDates] = df_w_tmp_holdingCash
        df_holding = df_holding.fillna(0)

    return df_holding, df_holding_cash

def round_to_lot(size_array, lot_size):
    """round number to the nearest lot_size.
    """
    if lot_size > 0:
        size_array = size_array.fillna(0)
        return (np.sign(size_array) * lot_size * np.floor(np.round(abs(size_array)) / max(1, lot_size)))
    else:
        return size_array

def parse_data(input_data):
    if type(input_data) is gftIO.GftTable:
        input_data = input_data.asMatrix()
#        input_data = input_data.dropna(axis=0, how='all')
        if input_data.columns.dtype == 'datetime64[ns]':
            return input_data.T
        else:
            return input_data
    elif type(input_data) is np.ndarray:
        return input_data.astype(datetime)
    else:
        return input_data

def SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK(dt_begin, dt_end, initial_holding_position, \
                                        df_w_target_portfolio_weight, df_w_market_price, \
                                        df_w_total_return_factor, df_w_execute_price, \
                                        df_w_execute_price_return, df_w_trade_volume, \
                                        dict_trading_param, additionalTs):
    """股票模拟交易
    input:
        dt_begin, 开始交易日期，在rebalance之前，repeat initial holding,
                                    在rebalance之后，从begin date开始交易。
        dt_end, 结束日期
        initial_holding_position, 输入持仓，可能为一个dataframe, 也有可能是一个数量cash
        df_w_target_portfolio_weight, 目标持仓权重，每行总和应等于1
        df_w_market_price, 股票价格
        df_w_total_return_factor, 复权因子计算
        df_w_execute_price, 交易股票执行价格
        df_w_execute_price_return, 每只股票交易日的回报
        df_w_trade_volume, 每只股票交易日的交易量
        dict_trading_param, 交易参数
        additionalTs，月平衡策略参数
    output:
        result, 字典
            result['HOLDING'], 所有股票持仓和现金数据 OTV	all dates
            result['PORTFOLIO_VALUE']， 组合总价值 TV	monthly
            result['SINGLE_PERIOD_RETURN']， 组合每日回报 TV	all dates
            result['WEIGHTS']，组合中各项权重 OTV	all dates
            result['CUMULATIVE_RETURN']，累积收益 TV	all dates
            result['TURNOVER']，换手率 TV	monthly

    """
    
    df_w_target_portfolio_weight = df_w_target_portfolio_weight.asMatrix()
    df_w_market_price = df_w_market_price.asMatrix()
    df_w_total_return_factor = df_w_total_return_factor.asMatrix()
    df_w_execute_price = df_w_execute_price.asMatrix()
    df_w_execute_price_return = df_w_execute_price_return.asMatrix()
    df_w_trade_volume = df_w_trade_volume.asMatrix()

    ls_all_dates = df_w_market_price.index.intersection(df_w_total_return_factor.index.intersection(df_w_execute_price.index.intersection(df_w_execute_price_return.index.intersection(df_w_trade_volume.index))))

    ls_cashGid = gsUtils.getCashGid()

    if len(df_w_target_portfolio_weight) > 0:
        df_w_target_portfolio_weight = df_w_target_portfolio_weight.ix[ls_all_dates]
        df_w_target_portfolio_weight = df_w_target_portfolio_weight.dropna(how='all')
        ls_rebalance_dates = df_w_target_portfolio_weight.index.tolist()
        if len(ls_rebalance_dates) > 0 and gsUtils.getParm(dict_trading_param, 'execDelayPeriods', 0) > 0:
            ls_rebalance_dates = ls_rebalance_dates.shift(gsUtils.getParm(dict_trading_param, 'execDelayPeriods', 0))
            ls_rebalance_dates = gsUtils.alignDate(ls_all_dates, ls_rebalance_dates)
            df_w_target_portfolio_weight = df_w_target_portfolio_weight.ix[ls_rebalance_dates]

    ls_holding_symbols = []
    if isinstance(initial_holding_position, pd.DataFrame):
        dt_begin = initial_holding_position.index[-1]
        ls_holding_symbols = sorted(list(set([i for i in initial_holding_position.columns if i not in ls_cashGid])))
    else:
        if len(df_w_target_portfolio_weight) > 0 and gsUtils.getParm(dict_trading_param, 'shiftBeginDateToSignal', 0) > 0:
            dt_begin = max(dt_begin, df_w_target_portfolio_weight.index[0])
    ls_trade_dates = [date for date in ls_all_dates if date >= dt_begin and date <= dt_end]
    dt_begin = ls_trade_dates[0]
    dt_end = ls_trade_dates[-1]
    ls_rebalance_dates = [date for date in ls_rebalance_dates if date>=dt_begin and date<=dt_end]
    if (dt_begin > dt_end):
        raise ValueError('input error! Begin date must be less than end date!')

    ls_all_symbols = sorted(list(set.intersection(set(df_w_market_price.columns),set(df_w_execute_price.columns),set(df_w_execute_price_return.columns),set(df_w_trade_volume.columns))))
    df_w_tmp_target_portfolio_weight = df_w_target_portfolio_weight.dropna(how='all',axis=1)
    df_w_tmp_target_portfolio_weight = df_w_tmp_target_portfolio_weight.loc[:,(df_w_tmp_target_portfolio_weight!=0).any(axis=0)]
    ls_portfolio_symbols = [s for s in df_w_tmp_target_portfolio_weight.columns if s not in ls_cashGid]

    if len([s for s in ls_holding_symbols if s not in ls_all_symbols]) > 0:
        raise ValueError('input error! Initial Portfolio has non A-share stocks!')

    if len([s for s in ls_portfolio_symbols if s not in ls_all_symbols]) > 0:
        raise ValueError('input error! Target Portfolio has non A-share stocks!')


    # todo: process holding symbols
    ls_all_symbols = sorted([s for s in set.intersection(set(ls_all_symbols),set(ls_portfolio_symbols)) if s not in ls_cashGid])
    ls_price_dates = [d for d in ls_all_dates if d >= (dt_begin-pd.Timedelta('20 days')) and d <= dt_end]
    df_w_market_price = df_w_market_price.loc[ls_price_dates][ls_all_symbols]
    df_w_total_return_factor = add_missing_columns(df_w_total_return_factor, ls_all_symbols, 1).loc[ls_price_dates, ls_all_symbols]
    df_w_execute_price = df_w_execute_price.loc[ls_price_dates][ls_all_symbols]
    df_w_execute_price_return = df_w_execute_price_return.loc[ls_price_dates][ls_all_symbols]
    df_w_trade_volume = df_w_trade_volume.loc[ls_price_dates][ls_all_symbols]
    df_w_initial_holding = initial_holding_position

    if not isinstance(initial_holding_position, pd.DataFrame):
        df_w_initial_holding = pd.DataFrame(initial_holding_position, index=[dt_begin], columns=[ls_cashGid])

    # todo: if initial holding is dataframe, fill
    df_w_initial_holding = add_missing_columns(df_w_initial_holding, ls_all_symbols)
    df_w_initial_holding_cash = df_w_initial_holding.loc[df_w_initial_holding.index, ls_cashGid]
    # todo, pop godgid
    df_w_initial_holding =df_w_initial_holding.drop(ls_cashGid, axis=1)
    num_initial_holding_positionValue = float((df_w_initial_holding*df_w_market_price.ix[df_w_initial_holding.index]).sum(1)+df_w_initial_holding_cash.values[-1])
    ls_all_symbols.append(ls_cashGid)
    df_w_target_portfolio_weight_fill = add_missing_columns(df_w_target_portfolio_weight, ls_all_symbols)
    df_w_target_portfolio_weight = df_w_target_portfolio_weight_fill.ix[ls_rebalance_dates].fillna(0)
    ls_all_symbols.pop(-1)
    if (df_w_target_portfolio_weight < 0).any().any():
        raise ValueError('input error! Do not support stock short selling and cash borrowing.')

    df_w_initial_holding_cash_weight = df_w_target_portfolio_weight[ls_cashGid]
    df_w_target_portfolio_weight = df_w_target_portfolio_weight.drop(ls_cashGid, axis=1)
    df_w_initial_holding_cash_weight = 1. - df_w_target_portfolio_weight.sum(axis=1)

    df_w_buy_volume = df_w_trade_volume.copy().fillna(0)
    df_w_sell_volumn = df_w_trade_volume.copy().fillna(0)

    num_lot_size = gsUtils.getParm(dict_trading_param, 'lotSize', 0)
    df_w_buy_volume = round_to_lot(df_w_buy_volume, num_lot_size)
    df_w_sell_volumn = round_to_lot(df_w_sell_volumn, num_lot_size)

    try:
        can_trade_on_suspend = gsUtils.getParm(dict_trading_param, 'canTradeOnSuspend', 0)
        if not can_trade_on_suspend is None:
            pass
    except KeyError:
        print("no trade on suspend information")
    else:
        if can_trade_on_suspend > 0:
            df_w_buy_volume[df_w_buy_volume < 1] = np.inf
            df_w_sell_volumn[df_w_sell_volumn < 1] = np.inf

    try:
        rise_limit_thres = gsUtils.getParm(dict_trading_param, 'riseLimitThres', 0)
        if not rise_limit_thres is None:
            pass
    except KeyError:
        print("no rise limit threshold information")
    else:
        if rise_limit_thres > 0:
            rise_limit = df_w_execute_price_return > rise_limit_thres
            df_w_buy_volume[rise_limit] = 0
            df_w_sell_volumn[rise_limit & (df_w_sell_volumn > 0)] = np.inf

    try:
        fall_limit_thres = gsUtils.getParm(dict_trading_param, 'fallLimitThres', 0)
        if not fall_limit_thres is None:
            pass
    except KeyError:
        print("no fall limit threshold information")
    else:
        if fall_limit_thres < 0:
            fall_limit = df_w_execute_price_return < fall_limit_thres
            df_w_buy_volume[fall_limit & (df_w_buy_volume > 0)] = np.inf
            df_w_sell_volumn[fall_limit] = 0

    try:
        volume_limit_pct = gsUtils.getParm(dict_trading_param, 'volumeLimitPct', 0)
        if volume_limit_pct is None:
            pass
    except KeyError:
        print("no fall limit threshold information")
    else:
        if volume_limit_pct > 0:
            df_w_buy_volume = df_w_buy_volume * volume_limit_pct
            df_w_sell_volumn = df_w_sell_volumn * volume_limit_pct
        else:
            df_w_buy_volume[df_w_buy_volume > 0] = np.inf
            df_w_sell_volumn[df_w_sell_volumn > 0] = np.inf


    num_buy_commission = gsUtils.getParm(dict_trading_param, 'buyCommission', 0)
    num_sell_commission = gsUtils.getParm(dict_trading_param, 'sellCommission', 0)

    df_w_holding = pd.DataFrame(0., index=ls_trade_dates, columns=ls_all_symbols)
    df_w_weights = df_w_holding.copy()
    df_w_execution = df_w_holding.copy()
    df_w_holding_cash = pd.DataFrame(0.,index=ls_trade_dates,columns=[ls_cashGid])
    ls_getGodGid = gsUtils.getGodGid()
    df_portfolio_value = pd.DataFrame(0., index=ls_trade_dates, columns=[ls_getGodGid])

    df_w_single_period_ret = df_portfolio_value.copy()
    df_w_turnover_percent = df_portfolio_value.copy()
    
    ## trading
    d = ls_trade_dates[0]
    df_w_holding.ix[d] = df_w_initial_holding.loc[d].tolist()
    df_w_holding_cash.ix[d] = df_w_initial_holding_cash.values[-1]


    if len(ls_rebalance_dates) < 1:
        nextd = ls_trade_dates[-1]
        df_w_holding, df_w_holding_cash = fill_holding(d, nextd, ls_trade_dates, \
                                                     df_w_holding, df_w_holding_cash, \
                                                     df_w_total_return_factor)
    else:
        nextd = ls_rebalance_dates[0]
        df_w_holding, df_w_holding_cash=fill_holding(d, nextd, ls_trade_dates, \
                                                   df_w_holding, df_w_holding_cash, \
                                                   df_w_total_return_factor)

        for i in range(len(ls_rebalance_dates)):
            d = ls_rebalance_dates[i]
            df_w_current_holding_value = df_w_holding.ix[d]*(df_w_execute_price.ix[d].fillna(0)) # one line
            num_totalValue = df_w_current_holding_value.sum() + df_w_holding_cash.ix[d].values[-1]

            df_w_current_holding_weight = df_w_current_holding_value / num_totalValue
            df_w_currrent_holding_cash_weight = 1. - df_w_current_holding_weight.sum()
            df_w_target_holding_weight = df_w_target_portfolio_weight.ix[d]
            num_target_holding_cash_weight = 1. - df_w_target_holding_weight.sum()
            df_w_order_weight = df_w_target_holding_weight - df_w_current_holding_weight

            df_w_sell_order_weight = df_w_order_weight.copy()
            df_w_sell_order_weight[df_w_order_weight >= 0.0] = 0.0

            df_w_buy_order_weight = df_w_order_weight.copy()
            df_w_buy_order_weight[df_w_order_weight <= 0.0] = 0.0

            num_cash_available = df_w_holding_cash.ix[d].values[-1]
            # sell
            if (df_w_sell_order_weight.dropna() < 0.0).any():
                df_w_current_holding_weight_for_sell = df_w_current_holding_weight.copy().fillna(0)
                df_w_current_holding_weight_for_sell[df_w_current_holding_weight_for_sell <= 0.0] = 1.0
                tmp_1 = df_w_sell_order_weight / df_w_current_holding_weight_for_sell * df_w_holding.ix[d]
                df_w_sell_order = round_to_lot(tmp_1, num_lot_size) # share

                # sell all if target holding weight is equal to 0
                df_w_sell_order[df_w_target_holding_weight <= 0.0] = -df_w_holding.ix[d]

                df_w_sell_execution = df_w_sell_order.copy()
                df_w_sell_execution = -pd.concat([df_w_sell_execution.fillna(0).abs(), \
                                                 df_w_sell_volumn.ix[d]], axis=1).min(axis=1)
                num_cash_available = num_cash_available + (df_w_sell_execution.abs() * \
                                               df_w_execute_price.ix[d]).sum() * (1. - num_sell_commission)
                df_w_execution.ix[d] = df_w_execution.ix[d] + df_w_sell_execution
                df_w_holding.ix[d] = df_w_holding.ix[d] + df_w_sell_execution
            # buy
            if (df_w_buy_order_weight > 0.0).any():
                num_can_buy_weight = num_cash_available / num_totalValue - num_target_holding_cash_weight
                if num_can_buy_weight > 0:
                    num_pct = min(num_can_buy_weight / df_w_buy_order_weight.sum(), 1)
                    tmp_2 = (num_pct * df_w_buy_order_weight * num_totalValue / (1.+num_buy_commission) / df_w_execute_price.ix[d]).fillna(0)
                    df_w_buy_order = round_to_lot(tmp_2, num_lot_size)
                    df_w_buy_order = df_w_buy_order.fillna(0)
                    df_w_buy_execution = df_w_buy_order.copy() # redundant
                    df_w_buy_execution = pd.concat([df_w_buy_execution.fillna(0), \
                                                   df_w_buy_volume.ix[d]],axis=1).min(axis=1)
                    num_cash_available = num_cash_available - (abs(df_w_buy_execution) * \
                                                     df_w_execute_price.ix[d]).sum() * \
                                                     (1.+num_buy_commission)
                    df_w_execution.ix[d] = df_w_execution.ix[d] + df_w_buy_execution
                    df_w_holding.ix[d] = df_w_holding.ix[d] + df_w_buy_execution


            df_w_holding_cash.ix[d] = num_cash_available
            df_w_turnover_percent.ix[d] = (abs(df_w_execution.ix[d])*df_w_execute_price.ix[d]).sum() / num_totalValue

            if i < (len(ls_rebalance_dates) - 1):
                nextd = ls_rebalance_dates[i+1]
                df_w_holding, df_w_holding_cash = fill_holding(d, nextd, ls_trade_dates, \
                                                             df_w_holding, df_w_holding_cash, \
                                                             df_w_total_return_factor)
        # loop to the next day.
        nextd = ls_trade_dates[-1]
        df_w_holding, df_w_holding_cash = fill_holding(d, nextd, ls_trade_dates, \
                                                     df_w_holding, df_w_holding_cash, \
                                                     df_w_total_return_factor)

    df_w_portfolio_stat = pd.DataFrame(columns=[ls_getGodGid])
    df_w_portfolio_stat['value'] = (df_w_holding*df_w_market_price.ix[ls_trade_dates]).sum(axis=1)
    df_w_portfolio_stat['cash'] = df_w_holding_cash
    df_portfolio_value = df_w_portfolio_stat.sum(axis=1)
    df_w_weights = (df_w_holding*df_w_market_price.ix[ls_trade_dates]).div(df_portfolio_value,axis=0)
    df_w_single_period_ret = df_portfolio_value / df_portfolio_value.shift(1) - 1.0
    df_w_single_period_ret[0] = df_portfolio_value[0] / num_initial_holding_positionValue - 1.0
#    df_w_single_period_ret.name = ls_getGodGid
    s_cum_rets = df_portfolio_value / num_initial_holding_positionValue
    s_cum_rets[-1] = df_portfolio_value[-1] / num_initial_holding_positionValue
    print(s_cum_rets[-1])

    result = {}
    result['HOLDING'] = pd.concat([df_w_holding, df_w_holding_cash], axis=1)
    result['PORTFOLIO_VALUE'] = pd.DataFrame(df_portfolio_value, columns=[ls_getGodGid])
#    result['PORTFOLIO_VALUE'] = result['PORTFOLIO_VALUE'].reset_index()
    result['SINGLE_PERIOD_RETURN'] = pd.DataFrame(df_w_single_period_ret, columns=[ls_getGodGid])
#    result['SINGLE_PERIOD_RETURN'] = result['SINGLE_PERIOD_RETURN']
    result['WEIGHTS'] = df_w_weights
    result['CUMULATIVE_RETURN'] = pd.DataFrame(s_cum_rets, columns=[ls_getGodGid])
    result['TURNOVER'] = pd.DataFrame(df_w_turnover_percent, columns=[ls_getGodGid])
    return result

dataPack = gftIO.zload(r'd:\Wuwei\Project\simulator\data\monthly_rebalance_data_same_as_R_5.pkl')
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
result = SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)