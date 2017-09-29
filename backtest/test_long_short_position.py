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

from future_position import create_future_rollover_position as rp

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not handler:
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.debug('start')

path = r'/home/weiwu/projects/simulate/data/future/'
start_date = gftIO.zload(os.path.join(path, 'start_date.pkl'))
end_date = gftIO.zload(os.path.join(path, 'end_date.pkl'))
data = gftIO.zload(os.path.join(path, 'contract_data.pkl'))
target = gftIO.zload(os.path.join(path, 'target.pkl'))
df_commission = gftIO.zload(os.path.join(path, 'df_commission_fee.pkl'))
df_position = gftIO.zload(os.path.join(path, 'df_position.pkl'))
df_price = gftIO.zload(os.path.join(path, 'df_price.pkl'))
df_multiplier = gftIO.zload(os.path.join(path, 'df_multiplier.pkl'))

if isinstance(df_commission, gftIO.GftTable):
    df_commission = df_commission.asColumnTab().copy()
if isinstance(df_position, gftIO.GftTable):
    df_position = df_position.asMatrix().copy()
if isinstance(df_price, gftIO.GftTable):
    df_price = df_price.asColumnTab().copy()
if isinstance(df_multiplier, gftIO.GftTable):
    df_multiplier = df_multiplier.asColumnTab().copy()

if isinstance(data, gftIO.GftTable):
    data = data.asColumnTab().copy()

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
df_price.rename(columns=lambda x: name[x], inplace=True)

target = data['contract_name'].unique()
roll_position = pd.DataFrame()
# loop each commodity
for num_contract, contract in enumerate(target):
    # print('contract name is %s', contract)
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
        np.zeros((len(contract_dates),
                  len(target_data['contract_code'].unique()))),
        index=contract_dates,
        columns=target_data['contract_code'].unique())

    for i, (item, ex_date) in enumerate(target_expiry_dates.iteritems()):
        # ylog.info(item)
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

# process price for simulation
price = df_price.pivot(
    index='date', columns='contract_code', values='close_price')

# process multiplier
df_multiplier_name = {
    'CONTRACTINNERCODE': 'contract_code',
    'CMVALUE': 'multiplier',
    'CTIME': 'date',
    'OPTIONCODE': 'contract_name'
}

df_multiplier.rename(columns=lambda x: df_multiplier_name[x], inplace=True)

df_multiplier.dropna(subset=['multiplier'], inplace=True)
#multiplier = df_multiplier.pivot(
#   index='date', columns='contract_code', values='multiplier')
ds_multiplier = df_multiplier.set_index('contract_code')['multiplier']
df_portfolio_value = roll_position * price * ds_multiplier
value = df_portfolio_value.sum(1)
value = value.iloc[value.nonzero()]
