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


def create_future_rollover_position(start_date, end_date, contract_data, target):
    ''' create roll over strategy weight.
    '''
    if isinstance(contract_data, gftIO.GftTable):
        data = contract_data.asColumnTab().copy()

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
    return roll_weights
