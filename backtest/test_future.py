# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from lib.gftTools import gsConst, gftIO, gsUtils
from datetime import datetime
import sys
import itertools
import quandl
import os
import matplotlib.pyplot as plt
"""
1) 策略初始化函数

可以通过给 futures_account 添加新的属性的方法，自定义各种指标变量等等；

策略回测、模拟交易中的账户，策略初始化之前，会建立一个交易账户futures_account，在这个账户会存储上述全局变量参数信息，并在整个策略 执行期间更新并维护可用现金、期货头寸、每日交易订单委托明细等。futures_account会在策略整个回测期间存续。

2). 循环futures_account.universe里面的连续合约的交易日历.

如果有交易信号发生：

如果没有相关合约头寸：

建立头寸

如果有相关合约头寸：

关闭头寸

如果连续合约背后的具体合约发生变化：

如果有头寸：

移仓
"""

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not handler:
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.debug('start')

""" 模拟输入信息 """
# dates = pd.date_range('2010-01-01', periods=6)
# contract_name = ['gold', 'gold', 'silver', 'silver', 'silver']
# contract_code = ['AU1006', 'AU1009', 'AG1006', 'AG1009', 'AG1012']
# zipped = list(zip(contract_name, contract_code))
# index = pd.MultiIndex.from_tuples(zipped)

# noa = len(contract_code)

# data = np.array([[10, 11, 12, 11, 12, 13],
#                  [np.nan, np.nan, np.nan, 13, 14, 9],
#                  [10, 10, np.nan, np.nan, np.nan, np.nan],
#                  [np.nan, np.nan, 12, 13, np.nan, np.nan],
#                  [np.nan, np.nan, np.nan, np.nan, 14, 9]])

# market_to_market_price = pd.DataFrame(data.T, index=dates, columns=index)
# rets = market_to_market_price / market_to_market_price.shift(1) - 1.0
# rets = rets.dropna(axis=0, how='all')
# multiplier_data = np.array([[10]*6,
#                             [10]*6,
#                             [15]*6,
#                             [15]*6,
#                             [15]*6])
# multiplier = pd.DataFrame(multiplier_data.T, index=dates, columns=index)
# targets = ['gold']
# position_data = np.array([[1, 1, 1, 1, 1, 1],
#                  [np.nan, np.nan, np.nan, 1, 1, 1],
#                  [1, 1, np.nan, np.nan, np.nan, np.nan],
#                  [np.nan, np.nan, 1, 1, np.nan, np.nan],
#                  [np.nan, np.nan, np.nan, np.nan, 1, 1]])

# position = pd.DataFrame(position_data.T, index=dates, columns=index)

# wti_near = pd.read_csv('CLF2014.csv', index_col='Date')
# wti_far = pd.read_csv('CLG2014.csv', index_col='Date')
# # wti_near = quandl.get("CME/CLF2014", authtoken="G-_-G7yn75tX64T3fKXc")
# wti_far_far = pd.read_csv('CLH2014.csv', index_col='Date')
# # wti_far = quandl.get("CME/CLG2014", authtoken="G-_-G7yn75tX64T3fKXc")
# wti = pd.DataFrame({'CLF2014': wti_near['Settle'],
#                     'CLG2014': wti_far['Settle']}, index=wti_far.index)

# start_date = wti_near.index[0]
# expiry_dates = pd.Series({'CLF2014': datetime(2013, 12, 19),
#                           'CLG2014': datetime(2014, 1, 21)}).sort_values()
rollover_days = 1
"""This constructs a pandas DataFrame that contains weights (between 0.0 and 1.0)
of contract positions to hold in order to carry out a rollover of rollover_days
prior to the expiration of the earliest contract. The matrix can then be
'multiplied' with another DataFrame containing the settle prices of each
contract in order to produce a continuous time series futures contract."""
# contracts = wti.columns
# Construct a sequence of dates beginning from the earliest contract start
# date to the end date of the final contract
# dates = pd.date_range(start_date, expiry_dates[-1], freq='B')
# dates = wti_near.index.append(wti_far.index.append(wti_far_far.index)).unique()

def futures_rollover_weights(start_date, dates, expiry_dates,
                             contracts, rollover_days=5):
    """This constructs a pandas DataFrame that contains weights (between 0.0 and 1.0)
    of contract positions to hold in order to carry out a rollover of rollover_days
    prior to the expiration of the earliest contract. The matrix can then be
    'multiplied' with another DataFrame containing the settle prices of each
    contract in order to produce a continuous time series futures contract."""

    # Construct a sequence of dates beginning from the earliest contract start
    # date to the end date of the final contract
  #  dates = pd.date_range(start_date, expiry_dates[-1], freq='B')

    # Create the 'roll weights' DataFrame that will store the multipliers for
    # each contract (between 0.0 and 1.0)
    roll_weights = pd.DataFrame(np.zeros((len(dates), len(contracts))),
                                index=dates, columns=contracts)
    prev_date = roll_weights.index[0]

    # Loop through each contract and create the specific weightings for
    # each contract depending upon the settlement date and rollover_days
    for i, (item, ex_date) in enumerate(expiry_dates.iteritems()):
        if i < len(expiry_dates) - 1:
            roll_weights.ix[prev_date:ex_date - pd.offsets.BDay(), item] = 1
            roll_rng = pd.date_range(end=ex_date - pd.offsets.BDay(),
                                     periods=rollover_days + 1, freq='B')

            # Create a sequence of roll weights (i.e. [0.0,0.2,...,0.8,1.0]
            # and use these to adjust the weightings of each future
            decay_weights = np.linspace(0, 1, rollover_days + 1)
            roll_weights.ix[roll_rng, item] = 1 - decay_weights
            roll_weights.ix[roll_rng, expiry_dates.index[i+1]] = decay_weights
        else:
            roll_weights.ix[prev_date:, item] = 1
        prev_date = ex_date
    return roll_weights


#weights = futures_rollover_weights(wti_near.index[0], expiry_dates, wti.columns)
# Construct the continuous future of the WTI CL contracts
#wti_cts = (wti * weights).sum(1).dropna()

path = r'/home/weiwu/projects/simulate/data/future/'
start_date = gftIO.zload(os.path.join(path, 'start_date.pkl'))
end_date = gftIO.zload(os.path.join(path, 'end_date.pkl'))
data = gftIO.zload(os.path.join(path, 'contract_data.pkl'))
target = gftIO.zload(os.path.join(path, 'target.pkl'))

if isinstance(data, gftIO.GftTable):
    data = data.asColumnTab().copy()

if isinstance(target, list):
    target = gftIO.strSet2Np(np.array(target))

name = {'INNERCODE': 'contract_code', 'OPTIONCODE': 'contract_name',
        'SETTLEMENTDATE': 'settlement_date', 'ENDDATE': 'date',
        'CLOSEPRICE': 'close_price'}
data.rename(columns=lambda x: name[x], inplace=True)

# check if the target contracts are in the dataset.
if set(target).issubset(data['contract_name']):
    target_data = data.loc[data['contract_name'].isin(target)]

roll_weights = pd.DataFrame()
for contract in target:
    contract_data = data[data['contract_name'] == contract]
    # contract_data.set_index('date', inplace=True)
    contract_expiry_dates = contract_data[['contract_code', 'settlement_date']].\
                            drop_duplicates().sort_values('settlement_date')
    contract_expiry_dates.set_index('contract_code', inplace=True)
    contract_expiry_dates = contract_expiry_dates[contract_expiry_dates.columns[0]]
    contract_data = contract_data.loc[:, ['date', 'contract_code', 'close_price']]
    contract_data = contract_data.pivot(index='date', columns='contract_code', values='close_price')
    contracts = contract_data.columns
    contract_start_date = contract_data.index[0]
    contract_dates = contract_data.index

    contract_roll_weights = pd.DataFrame(np.zeros((len(contract_dates),
                                                   len(contracts))),
                                         index=contract_dates,
                                         columns=contracts)
    prev_date = contract_roll_weights.index[0]
    # Loop through each contract and create the specific weightings for
    # each contract depending upon the settlement date and rollover_days
    for i, (item, ex_date) in enumerate(contract_expiry_dates.iteritems()):
        print(i, item, ex_date)
        if i < len(contract_expiry_dates) - 1:
            idx_pre_ex_date = contract_data.index.searchsorted(ex_date)
            pre_ex_date = contract_dates[idx_pre_ex_date - 1]
            contract_roll_weights.ix[prev_date:pre_ex_date, item] = 1
        else:
            contract_roll_weights.ix[prev_date:, item] = 1
        prev_date = ex_date
    roll_weights = pd.concat([roll_weights, contract_roll_weights], axis=1)

roll_weights = roll_weights.loc[start_date:end_date]
value = (contract_data * contract_roll_weights).sum(1)
# Construct the continuous future of the WTI CL contracts
# wti_cts = (wti * rubber_roll_weights).sum(1).dropna()

