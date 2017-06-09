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

expo_factor = np.ones((5,5))
factor_covariance = market_to_market_price.cov()
delta = np.diagflat([0.088024, 0.082614, 0.084237, 0.074648,
                                 0.084237])
cov_matrix = np.dot(np.dot(expo_factor, factor_covariance),
                            expo_factor.T) + delta
                    
def calculate_total_risk(weights, cov_matrix):
    port_var = np.dot(np.dot(weights.T, cov_matrix), weights)
    return port_var
    
    
def statistics(weights):
    ''' Return portfolio statistics.
    
    Parameters
    ==========
    weights : array-like
        weights for different securities in portfolio
    
    Returns
    =======
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
    prisk = calculate_total_risk(weights, cov_matrix)
    return np.array([pret, pvol, prisk, pret / pvol])


def min_func_sharpe(weights):
    return -statistics(weights)[3]

def min_func_vol(weights):
    return -statistics(weights)[1]

def min_func_std(weights):
    return statistics(weights)[2]

def max_func_mean(weights):
    return -statistics(weights)[0]

################################
noa = market_to_market_price.shape[1]
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
init_guess = noa * [1. / noa,]
opts_sharpe = sco.minimize(min_func_sharpe, init_guess, method='SLSQP',
                       bounds=bnds, constraints=cons)
opts_sharpe['x'].round(3)
statistics(opts_sharpe['x']).round(3)

# optimized return with given risk
tolerance_risk = 27
noa = market_to_market_price.shape[1]
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1},
         {'type': 'eq', 'fun': lambda x:  calculate_total_risk(x, cov_matrix) - tolerance_risk})
bnds = tuple((0, 1) for x in range(noa))
init_guess = noa * [1. / noa,]
opts_mean = sco.minimize(max_func_mean, init_guess, method='SLSQP',
                       bounds=bnds, constraints=cons)
opts_mean['x'].round(3)
statistics(opts_mean['x']).round(3)

# optimized return with minimum variance
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
opts_risk = sco.minimize(min_func_std, init_guess, method='SLSQP',
                       bounds=bnds, constraints=cons)
opts_risk['x'].round(3)
statistics(opts_risk['x']).round(3)

# optimized return with minimum total risk
target_return = 0.5
bnds = tuple((0, 1) for x in range(noa))
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1},
         {'type': 'eq', 'fun': lambda x:
                 target_return - np.sum(rets.mean() * x) * 252})
opts_ret = sco.minimize(min_func_std, init_guess, method='SLSQP',
                       bounds=bnds, constraints=cons)
opts_ret['x'].round(3)
statistics(opts_ret['x']).round(3)
# optimized return with minimum variance
bnds = tuple((0, 1) for x in range(noa))
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
opts_ret = sco.minimize(min_func_vol, init_guess, method='SLSQP',
                       bounds=bnds, constraints=cons)
opts_ret['x'].round(3)
statistics(opts_ret['x']).round(3)
weight_opts_ret = pd.DataFrame([opts_ret['x'].round(3)], columns=index)
# optimized return with minimum variance with class constraints
bnds = tuple((0, 1) for x in range(noa))
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
opts_ret = sco.minimize(min_func_vol, init_guess, method='SLSQP',
                       bounds=bnds, constraints=cons)
opts_ret['x'].round(3)
statistics(opts_ret['x']).round(3)
#weight_opts_ret = pd.DataFrame([opts_ret['x'].round(3)], columns=index)
# optimized return with minimum variance with class constraints
#weight_opts_ret.groupby(level=0, axis=1).sum()['consumer'] = 0.4
def w_opt(W):
    def filterer(x):
        v = x.range.values
        tp = v[0]
        lower, upper = tp
        return lower <= x[].sum() <= upper
    return not W.groupby(level=0, axis=0).filter(filterer).empty


intervals = [(.08,.51), (.05,.21), (.05,.41), (.05,.41), (.2,.66), (0,.16), (0,.76), (0,.11)]
names = ['equity', 'intl_equity', 'bond', 'intl_bond', 'commodity', 'pe', 'hf', 'cash']
mapper = pd.Series(list(zip(names, intervals)))
fully_mapped = mapper[init_weights.get_level_values(0)]
original_dataset['range'] = fully_mapped.values
                
                
init_guess = pd.DataFrame(noa * [1. / noa,], index=index)
bnds = tuple((0, 1) for x in range(noa))
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1},
         {'type': 'eq', 'fun': lambda x:  0.4 <= np.sum(x[2:4]) <= 0.45})
#         {'type': 'eq', 'fun': lambda x:  0.4 <= x.groupby(level=0, axis=1).sum()['consumer'] <= 0.45})
opts_ret = sco.minimize(min_func_vol, init_guess, method='SLSQP',
                       bounds=bnds, constraints=cons)
opts_ret['x'].round(3)
statistics(opts_ret['x']).round(3)
weight_opts_ret = pd.DataFrame([opts_ret['x'].round(3)], columns=index)
################################


model = pd.DataFrame(np.array([.08,.12,.05,0.2,0.0]), index=index, columns=['strategic'])
model['tactical'] = [(.08,.51), (.05,.21), (.05,.41), (.05,.41), (.05,.41)]


def portfolio_optimization(target_mode, position_limit, covariance_matrix, \
                           asset_return, asset_weight, target_risk, \
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
