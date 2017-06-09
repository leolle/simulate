# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:03:30 2017

@author: J
"""
import pandas as pd
import numpy as np
from datetime import datetime
import scipy.optimize as sco

from lib.gftTools import gftIO, gsUtils


""" 模拟输入信息 """
dates = pd.date_range('1/1/2000', periods=8)
dt_start = dates[0]
dt_end = dates[-1]
industry = ['industry', 'industry', 'utility', 'utility', 'consumer']
symbols = ['A', 'B', 'C', 'D', 'E']  # including Cash
zipped = list(zip(industry, symbols))
index = pd.MultiIndex.from_tuples(zipped)

noa = len(symbols)
weights = np.random.random(noa)
weights /= np.sum(weights)

data = np.array([[10, 9, 10, 11, 12, 13, 14, 13],
                 [11, 11, 10, 11, 11, 12, 11, 10],
                 [10, 11, 10, 11, 12, 13, 14, 13],
                 [11, 11, 10, 11, 11, 12, 11, 11],
                 [10, 11, 10, 11, 12, 13, 14, 13]])

market_to_market_price = pd.DataFrame(data.T, index=dates, columns=index)
execute_price = pd.DataFrame(index=dates, columns=symbols)
All = slice(None)
market_to_market_price.plot()
rets = market_to_market_price / market_to_market_price.shift(1) - 1.0
target_portfolio_wgt = pd.DataFrame()
for i in range(8):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    target_portfolio_wgt = target_portfolio_wgt.append(pd.DataFrame(weights).T)
target_portfolio_wgt.index = dates
target_portfolio_wgt.columns = index


def statistics(weights):
    ''' Return portfolio statistics.

    Parameters
    ----------
    weights : array-like
        weights for different securities in portfolio

    Returns
    -------
    pret : float
        expected portfolio return
    pvol : float
        expected portfolio volatility
    pret / pvol : float
        Sharpe ratio for rf=0
    '''
    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])


def min_func_sharpe(weights):
    return -statistics(weights)[2]


noa = market_to_market_price.shape[1]
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
init_guess = noa * [1. / noa, ]
opts = sco.minimize(min_func_sharpe, init_guess, method='SLSQP',
                    bounds=bnds, constraints=cons)
opts['x'].round(3)
statistics(opts['x']).round(3)


def portfolio_optimization(target_mode, position_limit, covariance_matrix,
                           asset_return, asset_weight, target_risk,
                           target_return):
    """
    optimize fund weight target on different constraints, objective, based on
    target type and mode, fund return target, fund weight, group weight， etc.

    Parameters
    ----------
    target_mode: dictionary
        target optimization type({type: mode})
        0: return,mean,
        1: risk, variance,
        2: sr, sharp ratio.

    df_position_limit: int
        group position limit.

    df_asset_return: Dataframe, OTV,
        index=date, O: asset names, O: group name, V: asset return.

    covariance_matrix: DataFrame
        Factor exposure covariance matrix, used for portfolio risk calculation.

    asset_weight: DataFrame
        weight bound of each asset.

    target_return: double
        Target return for portfolio respected to benchmark.

    target_risk: double
        Portfolio risk tolerance whose objective is maximum return.

    Returns:
    ----------
    df_result: DataFrame
        optimized value of weight and position.

    """
    pass

