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

if isinstance(df_commission, gftIO.GftTable):
    df_commission = df_commission.asColumnTab().copy()
if isinstance(df_position, gftIO.GftTable):
    df_position = df_position.asMatrix().copy()
if isinstance(df_price, gftIO.GftTable):
    df_price = df_price.asColumnTab().copy()
if isinstance(df_multiplier, gftIO.GftTable):
    df_multiplier = df_multiplier.asColumnTab().copy()
df_price_name = {
    'INNERCODE': 'contract_code',
    'OPTIONCODE': 'contract_name',
    'SETTLEMENTDATE': 'settlement_date',
    'ENDDATE': 'date',
    'CLOSEPRICE': 'close_price'
}
df_price.rename(columns=lambda x: df_price_name[x], inplace=True)
df_price = df_price.pivot(
    index='date', columns='contract_code', values='close_price')
df_position.replace(to_replace=0, value=np.nan, inplace=True)
df_position = df_position.loc[start_date:end_date]
df_price = df_price.loc[start_date:end_date]
df_price.fillna(method='pad', inplace=True)
df_price.fillna(method='bfill', inplace=True)
# process multiplier
df_multiplier_name = {
    'CONTRACTINNERCODE': 'contract_code',
    'CMVALUE': 'multiplier',
    'CTIME': 'date',
    'OPTIONCODE': 'contract_name'
}

df_multiplier.rename(columns=lambda x: df_multiplier_name[x], inplace=True)

df_multiplier.dropna(subset=['multiplier'], inplace=True)
ds_multiplier = df_multiplier.set_index('contract_code')['multiplier']

df_position = rp(contract_data)
#df_position.plot(legend=False)
#plt.show()

# calculate value
df_portfolio_value = df_position * df_price * ds_multiplier
value = df_portfolio_value.sum(1)
value = value.iloc[value.nonzero()]
value = value.to_frame()
value.columns = gsUtils.getCashGid()
