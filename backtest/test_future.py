# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from lib.gftTools import gsConst, gftIO, gsUtils
from datetime import datetime
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
                contract_roll_position.ix[prev_date:pre_ex_date, item] = 1
            else:
                contract_roll_position.ix[prev_date:, item] = 1
                prev_date = ex_date
                roll_position = pd.concat(
                    [roll_position, contract_roll_position], axis=1)
