# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
import os
import warnings
from functools import reduce
import math as mt
import statsmodels.api as sm
from lib.gftTools import gftIO
import datetime
import logging

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

time = pd.date_range('2000-01-01', freq='D', periods=432)
X = xr.DataArray(
    np.random.randn(432, 10, 3),
    [('date', time), ('symbol', list('abcdefghij')), ('factor', list('xyz'))])

y = xr.DataArray(
    np.random.randn(432, 10), [('date', time), ('symbol', list('abcdefghij'))])


def regression(X, Y):
    """ regression X by Y
    Keyword Arguments:
    X           --
    Y           --
    constraint1 --
    constraint2 --
    constraint3 --

    Return:
    model -- dictionary, key: value
        param: xarray, factor return, factors: 36, date: 432
        resid: xarray, stock residual, date: 432, symbol: 2981
    """
    pass


param = xr.DataArray(
    np.random.randn(432, 36), [('date', time), ('factors', [
        '018D158ACE908165ED4ECEA129ABF547', '90F09502030C80B91B02A2CE72A5A0EB',
        '91B8D8D34EE8AA2BD9B60F9D223C8707', '767FA9609E0C0B661EE557AB80C6F8E4',
        'A528F7BFA538A7675484FC2A72021138', '4972C5017D9B1D2AEDA81920FA21D3BF',
        '35555851DC92592BF9A5D55906D10A2B', 'D84A4B250D077E0EFC4F7FFF393FCB44',
        'D753394CA4D5A4CFC25BB5DACA7293F3', 'C585E113DD45D2344C97D3D2967D536F',
        '22895BDA3790CB3FA9FABD1E2FF883CE', '26C8DB23CDCCD1E7D858D3340449C854',
        '3995B9BA60E9943BF39F784E2E6B451D', 'EFE87574B86C774ADFD08F421AF5D11B',
        '7F216CE9D40D23FEDC54147E9D754DEE', '26BF3A3BD0BB76D58CC560B72527AB14',
        'B3E67D32A2F6428BB06D1BFCD2547798', 'BB81923B85135A411E8BDD80015E6512',
        'BED9C798EDEE97DC1712BB3E05FC8113', '265ED5755EEC25762A1293A398C40F5D',
        '324F4A27A88603BE2C53D5AFA7745584', '62C09CCF341CA27BCE7835103FE32E97',
        'BBA8CAB7DE9857EE8B8AFD9A23E3D34A', 'EB34F4D5C92E85C2307DB2C8015C94F1',
        '54899D64BCFE8C3BA11BB84429A46C52', 'A7638B5C794E97F887D2522215A3DFF8',
        '2AECD1DE9CC1EAA3982AAD231FA0AE39', '441E8F64A7582F553BCBE42A216285F1',
        '873CF94D09229206D675ACC32328DC24', 'EB8553C313C38BC180E1972D798622BA',
        'ABEC912F31E326F4C1FC507AF787C8FA', 'E8D70EA915C420F9B9005BB21540788C',
        'BE3E35A7C0CB49EB9E1CB41D566563E7', 'B9CCDA635F039E84D489F964DB08BC5C',
        'B1B02CFAB81248BAA87754E760769BD2', 'E8A54A95C9264162BEEC88B9CF65C78B'
    ])])

# resid = xr.DataArray(np.random.randn(432, 2981))
# resid.coords['dim_0'] = time

# model = {'param': param, 'resid': resid}

# coefs = stacked.groupby('allpoints').apply(xr_regression)


def xr_regression_coef(y):
    date = y.date
    model = sm.RLM(y.values, X.loc[date].values, M=sm.robust.norms.HuberT())
    results = model.fit()
    return xr.DataArray(results.params)


def xr_regression(y):
    date = y.date
    model = sm.RLM(y.values, X.loc[date].values, M=sm.robust.norms.HuberT())
    results = model.fit()
    return xr.DataArray(results.resid)


def xr_regression_resid(y):
    date = y.date
    model = sm.RLM(y.values, X.loc[date].values, M=sm.robust.norms.HuberT())
    results = model.fit()
    return xr.DataArray(results.resid)


# result = Y.groupby('date').apply(xr_regression)
# result_coef = Y.groupby('date').apply(xr_regression_coef)
#result_resid = Y.groupby('date').apply(xr_regression_resid)

# create regression result dateframe
params = pd.DataFrame(index=X.date, columns=X.factor)
residuals = pd.DataFrame(index=X.date, columns=X.symbol)
# get the datetimeindex
idx_date = y.get_index('date')
# for dt in y.date.values:
#     cur_date = pd.Timestamp(dt)
#     #print(cur_date)
#     # get the position of current date
#     dt_pos = idx_date.get_loc(cur_date)
#     if dt_pos == 0:
#         continue
#     dt_pre_pos = dt_pos - 1
#     model = sm.RLM(
#         y[dt_pos].values, X[dt_pre_pos].values, M=sm.robust.norms.HuberT())
#     results = model.fit()
#     params.loc[cur_date] = results.params
#     residuals.loc[cur_date] = results.resid
"""
------------------------------------------------------------------------------
"""
# create regression result dateframe
params = pd.DataFrame(index=X.date, columns=X.factor)
residuals = pd.DataFrame(index=X.date, columns=X.symbol)

# get the datetimeindex
idx_date = y.get_index('date')
idx_symbol = X.get_index('symbol')

# for dt in y.date.values:
#     logger.debug('regression on %s', dt)
#     cur_date = pd.Timestamp(dt)
#     # get the position of current date
#     dt_pos = idx_date.get_loc(cur_date)
#     if dt_pos == 0:
#         continue
#     dt_pre_pos = dt_pos - 1
#     # symbols having valid value(not nan)
#     s = X[:, dt_pre_pos].notnull().all(axis=0)
#     valid_x = X[:, dt_pre_pos, s].symbol.values

#     w = y.loc[cur_date].notnull()
#     valid_y = y.loc[cur_date, w].symbol.values

#     valid_symbol = np.intersect1d(valid_x, valid_y)
#     try:
#         model = sm.RLM(
#             y.loc[cur_date, valid_symbol].values,
#             X.isel(
#                 date=dt_pre_pos,
#                 symbol=idx_symbol.get_indexer(valid_symbol)).values.T,
#             M=sm.robust.norms.HuberT())
#         results = model.fit()
#     except ValueError:
#         continue
#     params.loc[cur_date] = results.params
#     residuals.loc[cur_date, valid_symbol] = results.resid


class RLMModel:
""" create RLM regression module
"""
    def __init__(self, arg):
        self.arg = arg

    def fit(self, y, X):
        model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
        return model.fit()


regression = RLMModel(0)
cur_date = pd.Timestamp(y.date.values[1])
#print(cur_date)
# get the position of current date
dt_pos = idx_date.get_loc(cur_date)
if dt_pos == 0:
    print('not enough data from X')
dt_pre_pos = dt_pos - 1
model = sm.RLM(
    y[dt_pos].values, X[dt_pre_pos].values, M=sm.robust.norms.HuberT())
results = regression.fit(y[dt_pos].values, X[dt_pre_pos].values)
params.loc[cur_date] = results.params
residuals.loc[cur_date] = results.resid
