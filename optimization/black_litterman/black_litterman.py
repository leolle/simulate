# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from copy import copy
from lib.gftTools import gftIO
from scipy import linalg
import matplotlib.pyplot as plt
from utils.winsorize_mad import winsorize_mad

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    C = gftIO.zload("/home/weiwu/share/black_litterman/C.pkl")
    ROE = gftIO.zload(
        "/home/weiwu/share/black_litterman/ROE_daily_PIT_change.pkl")
    delta = gftIO.zload("/home/weiwu/share/black_litterman/delta.pkl")
    historical_ret = gftIO.zload(
        "/home/weiwu/share/black_litterman/historical_ret.pkl")
    # Q = gftIO.zload("/home/weiwu/share/black_litterman/Q.pkl")
    tau = gftIO.zload("/home/weiwu/share/black_litterman/tau.pkl")
    weq = gftIO.zload("/home/weiwu/share/black_litterman/weq.pkl")
if isinstance(historical_ret, gftIO.GftTable):
    historical_ret = historical_ret.asMatrix().copy()
    # historical_ret.fillna(0)
    # In [139]: historical_ret.shape
    # Out[140]: (451, 3437)

if isinstance(ROE, gftIO.GftTable):
    # views on all assets are not required
    ROE = ROE.asMatrix().copy()
    ROE.fillna(method='ffill', inplace=True)
    ROE.dropna(how='all', axis=1, inplace=True)
    # In [131]: ROE.shape
    # Out[138]: (490, 7018)

if isinstance(weq, gftIO.GftTable):
    weq = weq.asMatrix().copy()
    weq.fillna(method='ffill', inplace=True)
    # In [141]: weq.shape
    # Out[143]: (451, 3551)

logger.debug('parse data finished!')

target_symbols = historical_ret.columns.intersection(
    weq.columns.intersection(ROE.columns))
target_dates = historical_ret.index.intersection(
    weq.index.intersection(ROE.index))
# get random symbols at the target position limit
# position_limit = 8
# arr = list(range(len(target_symbols)))
# np.random.shuffle(arr)
# target_symbols = target_symbols[arr[:position_limit]]
target_symbols = [
    b'\xe2\x11\xeaH\x17m2\xc1"\'i\xcb+xu\x90',
    b'\xe2\x11\xeaH\x9bW\'\xc1"\'i\xcb+xu\x90',
    b'\xe2\x11\xeaHX\x0c@\xc1"\'i\xcb+xu\x90',
    b'\xe2\x11\xeaH0\x81\x1f\xc1"\'i\xcb+xu\x90',
    b'\xe2\x11\xeaH\x10\x84\x08\xc1"\'i\xcb+xu\x90',
    b'\xe2\x11\xeaH\xfcW(\xc1"\'i\xcb+xu\x90',
    b'\xfcDIE\x9e\xc2\xf7\x05\xc4:\xc4\x06MiS\x80',
    b'\xe2\x11\xeaHDME\xc1"\'i\xcb+xu\x90'
]
target = b'\x8eK\x9c+\x8a\xc71\x00j\xfb\x07AX\x13\xc8\xac'
logger.debug('select symbols %s', target_symbols)

# Equilibrium covariance matrix
C = .8  # confidence level, simply use a real number
delta = 3.5
la_period = 90
tau = 0.025
logger.info('confidence level %s', C)
logger.info('tau %s', tau)
logger.info('delta %s', delta)

# calculate market capitalization weight for each asset
dt_target = target_dates[-99]
dt_1Y_pre = dt_target - pd.DateOffset(years=1)
dt_1Q_pre = dt_target - pd.DateOffset(days=90)
logger.debug('1 year datetime range %s:%s', dt_1Y_pre, dt_target)
market_capital = weq.loc[dt_target, target_symbols]
logger.debug('market capital %s', market_capital)
weight = market_capital / market_capital.sum()

# use mean as equilibrium return of the stocks
if len(historical_ret) < 1:
    logging.ERROR('not enough historical data')

if np.any(np.isnan(historical_ret)):
    df_single_return = historical_ret.copy()
    df_single_return[np.isnan(df_single_return)] = 0.
df_cum = (df_single_return + 1).cumprod(axis=0) - 1
df_interval_agg_ret = df_cum - df_cum.shift(la_period)
df_ret_forecast = df_interval_agg_ret.shift(-la_period)
df_equilibrium = df_ret_forecast.loc[dt_1Q_pre:dt_target, target_symbols].mean()

Sigma = historical_ret.loc[dt_1Q_pre:dt_target, target_symbols].fillna(
    0).cov().values
V = Sigma * C

# investor view of return
Q = ROE.loc[dt_target, target_symbols] / ROE.loc[dt_1Q_pre, target_symbols] - 1
# Q = df_equilibrium
# investor view of return, positive:1, negative:-1
investor_position_view = copy(Q)
investor_position_view[investor_position_view >= df_equilibrium] = 1
investor_position_view[investor_position_view < df_equilibrium] = -1
investor_position_view = pd.DataFrame(
    np.diag(investor_position_view), columns=Q.index)
P = investor_position_view.values
# Q
# logger.debug('investor position %s', investor_position_view)
logger.debug('prediction Q %s', Q)
# Coefficient of uncertainty in the prior estimate of the mean
tauV = tau * V
Omega = np.dot(np.dot(P, tauV), P.T) * np.eye(Q.shape[0])

### calculation
pi = weight.values.dot(V * delta)
logger.debug('equilibrium return %s', pi)
# We use tau * sigma many places so just compute it once
ts = tau * V
# Compute posterior estimate of the mean
# This is a simplified version of formula (8) on page 4.
middle = linalg.inv(np.dot(np.dot(P, ts), P.T) + Omega)
logger.debug('Middle %s', middle)
# print(Q - np.expand_dims(np.dot(P, pi.T), axis=1))

# posterior estimate of the returns

# er = np.expand_dims(
#     pi, axis=0).T + np.dot(
#         np.dot(np.dot(ts, P.T), middle),
#         (Q - np.expand_dims(np.dot(P, pi.T), axis=1)))
# # Compute posterior estimate of the uncertainty in the mean
# # This is a simplified and combined version of formulas (9) and (15)
# posteriorSigma = V + ts - ts.dot(P.T).dot(middle).dot(P).dot(ts)
# logger.debug('posterior Sigma %s', posteriorSigma)
# # Compute posterior weights based on uncertainty in mean
# w = er.T.dot(linalg.inv(delta * posteriorSigma)).T
# #result = black_litterman_optimization(
# #    delta=delta, weq=weight, P=P, Q=Q, tau=tau, Sigma=Sigma)
# logger.info('unconstrainted optimized weight %s', w)
# rets = pd.DataFrame(df_equilibrium, columns=['mean'])
# # implied equiblirium return
# rets['ier'] = pi
# # estimate equiblirium return
# rets['eer'] = er
# # prediction from investors views
# rets['prediction'] = Q
# rets.to_csv('returns.csv')
