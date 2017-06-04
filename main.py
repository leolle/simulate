# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pandas as pd
import numpy as np
from datetime import datetime
from lib.gftTools import gftIO, gsUtils, gsConst
#import matplotlib.pyplot as plt

start = datetime.now()
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

data = gftIO.zload(r'd:\Wuwei\Project\simulator\data\monthly_rebalance_data_same_as_R_5.pkl')
#data = gftIO.transformDict4Name(data)
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


def add_missing_columns(df_w, ls_columns, fill_value = 0):
    """补充到完整的symbol list
    """
    df_list = df_w.columns.values.reshape(1,-1)
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

def align_date(a, b):
    """align the dates a according to b.
    for example,
    a = 1, 3, 5, 7, 9, 10
    b = 2, 4, 7
    result = 3, 5, 7
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

def get_param(ls_param, name, default_value):
    """ getting parameter value from the list, if not, return default value.
    """
    result = default_value
    if name in ls_param:
        result = ls_param[name]
    return result

def round_to_lot(size_array, lot_size):
    """round number to the nearest lot_size.
    """
    if lot_size > 0:
        size_array = size_array.fillna(0)
        return (np.sign(size_array) * lot_size * np.floor(np.round(abs(size_array)) / max(1, lot_size)))
    else:
        return size_array

dt_begin = data['begin_date']
dt_end = data['end_date']
dict_trading_param = data['trading_param']

df_w_market_price = data['market_to_market_price']
df_w_total_return_factor = data['total_return_factor']
df_w_execute_price = data['execute_price']
df_w_execute_price_return = data['execute_price_return']
df_w_trade_volume = data['trade_volume']
initial_holding_position = data['initial_holding']
df_w_target_portfolio_weight = data['target_portfolio_weight']
ls_all_dates = df_w_market_price.index.intersection(df_w_total_return_factor.index.intersection(df_w_execute_price.index.intersection(df_w_execute_price_return.index.intersection(df_w_trade_volume.index))))

ls_cashGid = gsUtils.getCashGid()

#if dict_trading_param['debug'] is not None:
#	print("Start the SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK \n")
try:
    dict_trading_param['debug']
except KeyError:
    print("Start the SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK \n")

if len(df_w_target_portfolio_weight) > 0:
    df_w_target_portfolio_weight = df_w_target_portfolio_weight.ix[ls_all_dates]
    df_w_target_portfolio_weight = df_w_target_portfolio_weight.dropna(how='all')
    ls_rebalance_dates = df_w_target_portfolio_weight.index.tolist()
    if len(ls_rebalance_dates) > 0 and dict_trading_param['execDelayPeriods'] > 0: #############  and   (1)
        ls_rebalance_dates = ls_rebalance_dates.shift(dict_trading_param['execDelayPeriods'])
        ls_rebalance_dates = align_date(ls_all_dates, ls_rebalance_dates)
        df_w_target_portfolio_weight = df_w_target_portfolio_weight.ix[ls_rebalance_dates]
#        ls_rebalance_dates=[x for x in ls_rebalance_dates if str(x)!='nan']
#        df_w_target_portfolio_weight.index = ls_rebalance_dates

ls_holding_symbols = []
if isinstance(initial_holding_position, pd.DataFrame):
    dt_begin = initial_holding_position.index[-1]
    ls_holding_symbols = sorted(list(set([i for i in initial_holding_position.columns if i not in ls_cashGid])))
else:
    if len(df_w_target_portfolio_weight) > 0 and dict_trading_param['shiftBeginDateToSignal'] > 0:
#        if dt_begin > df_w_target_portfolio_weight.index[0]:
#            dt_begin = dt_begin
#        else:
#            dt_begin = df_w_target_portfolio_weight.index[0]
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
    df_w_initial_holding = pd.DataFrame(initial_holding_position, \
                                        index=[dt_begin], columns=[ls_cashGid])

df_w_initial_holding = add_missing_columns(df_w_initial_holding, ls_all_symbols)
df_w_initial_holdingCash = df_w_initial_holding.loc[df_w_initial_holding.index, ls_cashGid]
df_w_initial_holding = df_w_initial_holding.drop(ls_cashGid, axis=1)
num_initial_holding_positionValue = float((df_w_initial_holding*df_w_market_price.ix[df_w_initial_holding.index]).sum(1)+df_w_initial_holdingCash.values[-1])
ls_all_symbols.append(ls_cashGid)
df_w_target_portfolio_weight_fill = add_missing_columns(df_w_target_portfolio_weight, ls_all_symbols)
df_w_target_portfolio_weight = df_w_target_portfolio_weight_fill.ix[ls_rebalance_dates].fillna(0)
ls_all_symbols.pop(-1)
if (df_w_target_portfolio_weight < 0).any().any():
    raise ValueError('input error! Do not support stock short selling and cash borrowing.')

df_w_targetPortfolioCashWgt = df_w_target_portfolio_weight[ls_cashGid]
df_w_target_portfolio_weight.pop(ls_cashGid)
df_w_targetPortfolioCashWgt = 1.-df_w_target_portfolio_weight.sum(axis=1)

df_w_buy_volume = df_w_trade_volume.copy().fillna(0)
df_w_sell_volumn = df_w_trade_volume.copy().fillna(0)

num_lot_size = get_param(dict_trading_param, 'lotSize', 0)
df_w_buy_volume = round_to_lot(df_w_buy_volume, num_lot_size)
df_w_sell_volumn = round_to_lot(df_w_sell_volumn, num_lot_size)

try:
    if not dict_trading_param['canTradeOnSuspend'] is None:
        pass
except KeyError:
    print("no trade on suspend information")
else:
    if dict_trading_param['canTradeOnSuspend'] > 0:
        df_w_buy_volume[df_w_buy_volume < 1] = np.inf
        df_w_sell_volumn[df_w_sell_volumn < 1] = np.inf

try:
    if not dict_trading_param['riseLimitThres'] is None:
        pass
except KeyError:
    print("no rise limit threshold information")
else:
    if dict_trading_param['riseLimitThres'] > 0:
        rise_limit = df_w_execute_price_return > dict_trading_param['riseLimitThres']
        df_w_buy_volume[rise_limit] = 0
        df_w_sell_volumn[rise_limit & (df_w_sell_volumn > 0)] = np.inf

try:
    if not dict_trading_param['fallLimitThres'] is None:
        pass
except KeyError:
    print("no fall limit threshold information")
else:
    if dict_trading_param['fallLimitThres'] < 0:
        fall_limit = df_w_execute_price_return < dict_trading_param['fallLimitThres']
        df_w_buy_volume[fall_limit & (df_w_buy_volume > 0)] = np.inf
        df_w_sell_volumn[fall_limit] = 0

try:
    if not dict_trading_param['volumeLimitPct'] is None:
        pass
except KeyError:
    print("no fall limit threshold information")
else:
    if dict_trading_param['volumeLimitPct'] > 0:
        df_w_buy_volume = df_w_buy_volume * dict_trading_param['volumeLimitPct']
        df_w_sell_volumn = df_w_sell_volumn * dict_trading_param['volumeLimitPct']
    else:
        df_w_buy_volume[df_w_buy_volume > 0] = np.inf
        df_w_sell_volumn[df_w_sell_volumn > 0] = np.inf


num_buy_commission = get_param(dict_trading_param, 'buyCommission', 0)
num_sell_commission = get_param(dict_trading_param, 'sellCommission', 0)

df_w_holding = pd.DataFrame(0., index=ls_trade_dates, columns=ls_all_symbols)
df_w_weights = df_w_holding.copy()
df_w_execution = df_w_holding.copy()
df_w_holding_cash = pd.DataFrame(0.,index=ls_trade_dates,columns=[ls_cashGid])
ls_getGodGid = gsUtils.getGodGid()
df_portfolio_value = pd.DataFrame(0., index=ls_trade_dates, columns=[ls_getGodGid])

df_w_single_period_return = df_portfolio_value.copy()
df_w_turnover_percent = df_portfolio_value.copy()

## trading
d = ls_trade_dates[0]
df_w_holding.ix[d] = df_w_initial_holding.loc[d].tolist()
df_w_holding_cash.ix[d] = df_w_initial_holdingCash.values[-1][-1]


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
        df_w_turnover_percent.ix[d] = (abs(df_w_execution.ix[d])*df_w_execute_price.ix[d]).sum()/num_totalValue

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

df_w_portfolio_stat = pd.DataFrame()
df_w_portfolio_stat['value'] = (df_w_holding*df_w_market_price.ix[ls_trade_dates]).sum(axis=1)
df_w_portfolio_stat['cash'] = df_w_holding_cash
df_portfolio_value = df_w_portfolio_stat.sum(axis=1)
df_w_weights = (df_w_holding*df_w_market_price.ix[ls_trade_dates]).div(df_portfolio_value,axis=0)
df_w_single_period_ret = df_portfolio_value/df_portfolio_value.shift(1)
s_cum_rets = df_w_single_period_ret.cumprod()
s_cum_rets[-1] = df_portfolio_value[-1] / num_initial_holding_positionValue - 1.0
#df_portfolio_value.plot()
#s_cum_rets.plot(figsize=(8,12),title='portfolio cumulative return(%)')
print(s_cum_rets[-1])

result = {}
result['HOLDING'] = pd.concat([df_w_holding, df_w_holding_cash], axis=1)
result['PORTFOLIO_VALUE'] = df_portfolio_value
result['SINGLE_PERIOD_RETURN'] = df_w_single_period_ret
result['WEIGHTS'] = df_w_weights
result['CUMULATIVE_RETURN'] = s_cum_rets[-1]
result['TURNOVER'] = df_w_turnover_percent



print(datetime.now() - start)
