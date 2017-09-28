# -*- coding: utf-8 -*-
from ylib import ylog
import logging
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
# pseudo code:
1. convert input contract data to dataframe, and rename the columns.
2. check if the contract data has target contract.
3. for loop in target contracts.
4. create pandas series with contract name and contract expiry dates.
5. fetch contract price from contract data.
6. loop through each contract and create the specific position for
each contract depending upon the settlement date and rollover_days
"""


def create_future_rollover_position(start_date, end_date, contract_data,
                                    target):
    ''' create roll over strategy weight.
    '''
    if isinstance(contract_data, gftIO.GftTable):
        data = contract_data.asColumnTab().copy()

    if isinstance(target, list):
        target = gftIO.strSet2Np(np.array(target))

    name = {
        'INNERCODE': 'contract_code',
        'OPTIONCODE': 'contract_name',
        'SETTLEMENTDATE': 'settlement_date',
        'ENDDATE': 'date',
        'CLOSEPRICE': 'close_price'
    }
    data.rename(columns=lambda x: name[x], inplace=True)
    # check if the target contracts are in the dataset.
    if set(target).issubset(data['contract_name']):
        target_data = data.loc[data['contract_name'].isin(target)]

    if target is None:
        target = data['contract_name'].unique()

    roll_position = pd.DataFrame()
    for contract in target:
        contract_data = data[data['contract_name'] == contract]
        # contract_data.set_index('date', inplace=True)
        contract_expiry_dates = contract_data[['contract_code', 'settlement_date']].\
            drop_duplicates().sort_values('settlement_date')
        contract_expiry_dates.set_index('contract_code', inplace=True)
        contract_expiry_dates = contract_expiry_dates[
            contract_expiry_dates.columns[0]]
        contract_data = contract_data.loc[:, [
            'date', 'contract_code', 'close_price'
        ]]
        contract_data = contract_data.pivot(
            index='date', columns='contract_code', values='close_price')
        contracts = contract_data.columns
        contract_start_date = contract_data.index[0]
        contract_dates = contract_data.index

        contract_roll_position = pd.DataFrame(
            np.zeros((len(contract_dates), len(contracts))),
            index=contract_dates,
            columns=contracts)
        prev_date = contract_roll_position.index[0]
        # Loop through each contract and create the specific position for
        # each contract depending upon the settlement date and rollover_days
        for i, (item, ex_date) in enumerate(contract_expiry_dates.iteritems()):
            # print(i, item, ex_date)
            if i < len(contract_expiry_dates) - 1:
                idx_pre_ex_date = contract_data.index.searchsorted(ex_date)
                pre_ex_date = contract_dates[idx_pre_ex_date - 1]
                contract_roll_position.loc[prev_date:pre_ex_date, item] = 1
            else:
                contract_roll_position.loc[prev_date:, item] = 1
            prev_date = ex_date
        roll_position = pd.concat(
            [roll_position, contract_roll_position], axis=1)
    return roll_position.loc[start_date:end_date]


def create_future_long_short_position(contract_data,):
    """ create long short position, rolling over at the day before expiry date.
    """
    if isinstance(contract_data, gftIO.GftTable):
        data = contract_data.asColumnTab().copy()

    name = {
        'INNERCODE': 'contract_code',
        'OPTIONCODE': 'contract_name',
        'SETTLEMENTDATE': 'settlement_date',
        'ENDDATE': 'date',
        'CLOSEPRICE': 'close_price'
    }
    data.rename(columns=lambda x: name[x], inplace=True)
    data.dropna(subset=['settlement_date'], inplace=True)

    target = data['contract_name'].unique()
    roll_position = pd.DataFrame()
    # loop each commodity
    for num_contract, contract in enumerate(target):
        ylog.info('contract name is %s', contract)
        target_data = data[data['contract_name'] == contract]
        target_expiry_dates = target_data[['contract_code', 'settlement_date']].\
            drop_duplicates().sort_values('settlement_date')
        target_expiry_dates.set_index('contract_code', inplace=True)
        target_expiry_dates = target_expiry_dates[target_expiry_dates.columns[
            0]]
        target_data = target_data.loc[:,
                                      ['date', 'contract_code', 'close_price']]
        contract_data = target_data.pivot(
            index='date', columns='contract_code', values='close_price')
        contract_dates = contract_data.index

        prev_date = contract_dates[0]
        # Loop through each contract and create the specific weightings for
        # each contract depending upon the rollover date and price adjusted method.
        # Here for backtesting, we use last trading day rollover and backward
        # ratio price adjustment.
        contract_roll_position = pd.DataFrame(
            np.zeros((len(contract_dates),
                      len(data['contract_name'].unique()))),
            index=contract_dates,
            columns=data['contract_name'].unique())

        # row_iterator = target_expiry_dates.iteritems()
        # _, last = row_iterator.next()  # take first item from row_iterator
        for i, (item, ex_date) in enumerate(target_expiry_dates.iteritems()):
            # print(i, item, ex_date)
            if i < len(target_expiry_dates) - 1:
                idx_pre_ex_date = contract_data.index.searchsorted(ex_date)
                pre_ex_date = contract_dates[idx_pre_ex_date - 1]
                contract_roll_position.loc[prev_date:pre_ex_date, item] = 1
                idx_ex_item = pd.Index(target_expiry_dates).get_loc(ex_date)
                # ylog.info(idx_ex_item)
                # ylog.info(ex_date)
                if i < (len(target_expiry_dates) - 2):
                    far_item = target_expiry_dates.index[idx_ex_item + 1]
                    contract_roll_position.loc[prev_date:pre_ex_date,
                                               far_item] = -1
                # ylog.info('far month %s', far_item)
            else:
                contract_roll_position.loc[prev_date:, item] = 1
            prev_date = ex_date
        roll_position = pd.concat(
            [roll_position, contract_roll_position], axis=1)
    return roll_position


def create_continuous_contract(start_date, end_date, contract_data, target):
    ''' parse contract data to get continuous price for each group.
    Parameters
    ----------
    start_date: datetime
    end_date: datetime
    contract_data: OOTTV
    contract name, contract code, date, settlement date, close price
    target: list or NULL
        targets to parse, NULL will parse all contracts.

    Returns
    -------
    continuous_price: DataFrame
    '''

    if isinstance(contract_data, gftIO.GftTable):
        data = contract_data.asColumnTab().copy()

    if isinstance(target, list):
        target = gftIO.strSet2Np(np.array(target))

    name = {
        'INNERCODE': 'contract_code',
        'OPTIONCODE': 'contract_name',
        'SETTLEMENTDATE': 'settlement_date',
        'ENDDATE': 'date',
        'CLOSEPRICE': 'close_price'
    }
    data.rename(columns=lambda x: name[x], inplace=True)
    data.dropna(subset=['settlement_date'], inplace=True)
    continuous_price = pd.DataFrame()

    if target is None:
        target = data['contract_name'].unique()

    for num_contract, contract in enumerate(target):
        ylog.info(num_contract)
        ylog.info(contract)
        target_data = data[data['contract_name'] == contract]
        target_expiry_dates = target_data[['contract_code', 'settlement_date']].\
            drop_duplicates().sort_values('settlement_date')
        target_expiry_dates.set_index('contract_code', inplace=True)
        target_expiry_dates = target_expiry_dates[target_expiry_dates.columns[
            0]]
        target_data = target_data.loc[:,
                                      ['date', 'contract_code', 'close_price']]
        contract_data = target_data.pivot(
            index='date', columns='contract_code', values='close_price')
        contract_dates = contract_data.index
        continuous_contract_price = pd.Series(
            np.ones(len(contract_dates)), index=contract_dates, name=contract)
        # ylog.info(contract_dates)
        prev_date = contract_dates[0]
        # Loop through each contract and create the specific weightings for
        # each contract depending upon the rollover date and price adjusted method.
        # Here for backtesting, we use last trading day rollover and backward ratio price adjustment.
        target_data_with_datetimeindex = target_data.set_index('date')
        price_adjust_ratio = pd.Series(
            np.ones(len(target_expiry_dates)),
            index=target_expiry_dates.values,
            name='ratio')
        adjusted_price = pd.Series(index=contract_dates, name=contract)
        target_data_with_datetimeindex['close_price'].replace(
            to_replace=0, method='bfill', inplace=True)
        target_data_with_datetimeindex['close_price'].replace(
            to_replace=0, method='pad', inplace=True)
        target_data_with_datetimeindex = target_data_with_datetimeindex[
            ~target_data_with_datetimeindex.index.duplicated()]
        for i, (item, ex_date) in enumerate(target_expiry_dates.iteritems()):
            # ylog.info(i)
            # ylog.info(item)
            # ylog.info(ex_date)
            if i < len(target_expiry_dates) - 1 \
               and ex_date < target_data_with_datetimeindex.index[-1]:
                idx_ex_date = target_data_with_datetimeindex.index.searchsorted(
                    ex_date)
                pre_ex_date = contract_dates[idx_ex_date - 1]
                # ex_date has no price data, move ex_date to next trading date.
                if ex_date not in target_data_with_datetimeindex.index and \
                   idx_ex_date + 1 < len(target_data_with_datetimeindex.index):
                    ex_date = contract_dates[idx_ex_date + 1]
                else:
                    continue
                price_adjust_ratio.loc[ex_date] = target_data_with_datetimeindex['close_price'].loc[ex_date] / \
                    target_data_with_datetimeindex['close_price'].loc[pre_ex_date]

        # to create adjusted_pricested price by the product of target price date and
        # adjustment ratio.
        for i, (item, ex_date) in enumerate(target_expiry_dates.iteritems()):
            #print(i, item, ex_date)
            idx_ex_date = contract_data.index.searchsorted(ex_date)
            pre_ex_date = contract_dates[idx_ex_date - 1]
            adjusted_price.ix[prev_date:pre_ex_date] = target_data_with_datetimeindex['close_price'].ix[prev_date:pre_ex_date] * \
                price_adjust_ratio.ix[ex_date:].cumprod().iloc[-1]
            prev_date = ex_date
        continuous_price = pd.concat([continuous_price, adjusted_price], axis=1)
    return continuous_price


ylog.set_level(logging.DEBUG)
ylog.console_on()
ylog.filelog_on('parse_future_contract.')
path = r'/home/weiwu/projects/simulate/data/future/'
start_date = gftIO.zload(os.path.join(path, 'start_date.pkl'))
end_date = gftIO.zload(os.path.join(path, 'end_date.pkl'))
contract_data = gftIO.zload(os.path.join(path, 'contract_data.pkl'))
target = gftIO.zload(os.path.join(path, 'target.pkl'))
df_commission = gftIO.zload(os.path.join(path, 'df_commission_fee.pkl'))
df_position = gftIO.zload(os.path.join(path, 'df_position.pkl'))
df_price = gftIO.zload(os.path.join(path, 'df_price.pkl'))
df_multiplier = gftIO.zload(os.path.join(path, 'df_multiplier.pkl'))

# df_position = create_future_rollover_position(start_date, end_date,
#                                               contract_data, target)
# df_position = create_future_long_short_position(contract_data)
# print(df_position.head())

# create_continuous_contract(start_date, end_date, contract_data, target=None)
# if isinstance(contract_data, gftIO.GftTable):
#     data = contract_data.asColumnTab().copy()

# if isinstance(target, list):
#     target = gftIO.strSet2Np(np.array(target))

# name = {'INNERCODE': 'contract_code', 'OPTIONCODE': 'contract_name',
#         'SETTLEMENTDATE': 'settlement_date', 'ENDDATE': 'date',
#         'CLOSEPRICE': 'close_price'}
# data.rename(columns=lambda x: name[x], inplace=True)
# continuous_price = pd.DataFrame()

# # if target is None:
# target = data['contract_name'].unique()[0]
# contract = data['contract_name'].unique()[28]

# ylog.info('contract name is %s', contract)
# target_data = data[data['contract_name'] == contract]
# target_expiry_dates = target_data[['contract_code', 'settlement_date']].\
#     drop_duplicates().sort_values('settlement_date')
# target_expiry_dates.set_index('contract_code', inplace=True)
# target_expiry_dates = target_expiry_dates[target_expiry_dates.columns[0]]
# target_data = target_data.loc[:, ['date', 'contract_code', 'close_price']]
# contract_data = target_data.pivot(
#     index='date', columns='contract_code', values='close_price')
# contract_dates = contract_data.index
# continuous_contract_price = pd.Series(np.ones(len(contract_dates)),
#                                       index=contract_dates,
#                                       name=contract)
# # ylog.info(contract_dates)
# prev_date = contract_dates[0]
# # Loop through each contract and create the specific weightings for
# # each contract depending upon the rollover date and price adjusted method.
# # Here for backtesting, we use last trading day rollover and backward
# # ratio price adjustment.
# target_data_with_datetimeindex = target_data.set_index('date')
# price_adjust_ratio = pd.Series(np.ones(len(target_expiry_dates)),
#                                index=target_expiry_dates.values,
#                                name='ratio')
# adjusted_price = pd.Series(index=contract_dates,
#                            name=contract)

# target_data_with_datetimeindex['close_price'].replace(to_replace=0,
#                                                       method='bfill',
#                                                       inplace=True)
# target_data_with_datetimeindex['close_price'].replace(to_replace=0,
#                                                       method='pad',
#                                                       inplace=True)
# target_data_with_datetimeindex = target_data_with_datetimeindex[~target_data_with_datetimeindex.index.duplicated(
# )]
# # to create price adjustment ratio.
# for i, (item, ex_date) in enumerate(target_expiry_dates.iteritems()):
#     ylog.info(i)
#     # ylog.info(item)
#     ylog.info(ex_date)
#     # make sure index doesn't go beyond the datetimeindex.
#     if i < len(target_expiry_dates) - 1 \
#        and ex_date < target_data_with_datetimeindex.index[-1]:
#         idx_ex_date = target_data_with_datetimeindex.index.searchsorted(
#             ex_date)
#         pre_ex_date = contract_dates[idx_ex_date - 1]
#         # ex_date has no price data, move ex_date to next trading date.
#         if ex_date not in target_data_with_datetimeindex.index and \
#            idx_ex_date + 1 < len(target_data_with_datetimeindex.index):
#             ex_date = contract_dates[idx_ex_date + 1]
#         else:
#             continue
#         price_adjust_ratio.loc[ex_date] = target_data_with_datetimeindex['close_price'].loc[ex_date] / \
#             target_data_with_datetimeindex['close_price'].loc[pre_ex_date]

# # to create adjusted price by the product of target price date and
# # adjustment ratio.
# for i, (item, ex_date) in enumerate(target_expiry_dates.iteritems()):
#     #print(i, item, ex_date)
#     idx_ex_date = contract_data.index.searchsorted(ex_date)
#     pre_ex_date = contract_dates[idx_ex_date - 1]
#     adjusted_price.loc[prev_date:pre_ex_date] = target_data_with_datetimeindex['close_price'].loc[prev_date:pre_ex_date] * \
#         price_adjust_ratio.loc[ex_date:].cumprod().iloc[-1]
#     prev_date = ex_date

# continuous_price = pd.concat([continuous_price, adjusted_price], axis=1)
# ax = contract_data.plot(legend=True)
# continuous_price.plot(legend=True, style='k--', ax=ax)
# plt.show()
if isinstance(contract_data, gftIO.GftTable):
    data = contract_data.asColumnTab().copy()

name = {
    'INNERCODE': 'contract_code',
    'OPTIONCODE': 'contract_name',
    'SETTLEMENTDATE': 'settlement_date',
    'ENDDATE': 'date',
    'CLOSEPRICE': 'close_price'
}
data.rename(columns=lambda x: name[x], inplace=True)
data.dropna(subset=['settlement_date'], inplace=True)

target = data['contract_name'].unique()
roll_position = pd.DataFrame()
# loop each commodity
for num_contract, contract in enumerate(target):
    ylog.info('contract name is %s', contract)
    target_data = data[data['contract_name'] == contract]
    target_expiry_dates = target_data[['contract_code', 'settlement_date']].\
        drop_duplicates().sort_values('settlement_date')
    target_expiry_dates.set_index('contract_code', inplace=True)
    target_expiry_dates = target_expiry_dates[target_expiry_dates.columns[0]]
    target_data = target_data.loc[:, ['date', 'contract_code', 'close_price']]
    contract_data = target_data.pivot(
        index='date', columns='contract_code', values='close_price')
    contract_dates = contract_data.index

    prev_date = contract_dates[0]
    # Loop through each contract and create the specific weightings for
    # each contract depending upon the rollover date and price adjusted method.
    # Here for backtesting, we use last trading day rollover and backward
    # ratio price adjustment.
    contract_roll_position = pd.DataFrame(
        np.zeros((len(contract_dates), len(data['contract_name'].unique()))),
        index=contract_dates,
        columns=data['contract_name'].unique())

    # row_iterator = target_expiry_dates.iteritems()
    # _, last = row_iterator.next()  # take first item from row_iterator
    for i, (item, ex_date) in enumerate(target_expiry_dates.iteritems()):
        # print(i, item, ex_date)
        if i < len(target_expiry_dates) - 1:
            idx_pre_ex_date = contract_data.index.searchsorted(ex_date)
            pre_ex_date = contract_dates[idx_pre_ex_date - 1]
            contract_roll_position.loc[prev_date:pre_ex_date, item] = 1
            idx_ex_item = pd.Index(target_expiry_dates).get_loc(ex_date)
            # ylog.info(idx_ex_item)
            # ylog.info(ex_date)
            if i < (len(target_expiry_dates) - 2):
                far_item = target_expiry_dates.index[idx_ex_item + 1]
                contract_roll_position.loc[prev_date:pre_ex_date, far_item] = -1
            # ylog.info('far month %s', far_item)
        else:
            contract_roll_position.loc[prev_date:, item] = 1
        prev_date = ex_date
    roll_position = pd.concat([roll_position, contract_roll_position], axis=1)
