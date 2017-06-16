# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
import scipy.optimize as sco

from lib.gftTools import gftIO

# fetch data
path = '~/share/optimize/'

target_mode = gftIO.zload(path + 'x0.pkl')
position_limit = gftIO.zload(path + 'x1.pkl')
covariance_matrix = gftIO.zload(path + 'x2.pkl')
asset_return = gftIO.zload(path + 'x3.pkl')
asset_weight = gftIO.zload(path + 'x4.pkl')
target_risk = gftIO.zload(path + 'x5.pkl')
target_return = gftIO.zload(path + 'x6.pkl')
risk_model = gftIO.zload(path + 'risk_model.pkl')

# assign specific date
target_date = datetime(year=2015, month=7, day=31)

# load initial portfolio weight allocation data.
data = gftIO.zload(path + 'monthly_rebalance_data_same_as_R_5.pkl')
data['target_portfolio_weight'] = data.pop('x3')
df_target_portfolio_weight = data['target_portfolio_weight'].\
                             asMatrix().dropna(axis=0, how='all')

target_risk = 0.00087447
target_return = 0.0262495
def get_factor_exposure(df_target_weight, risk_model, factor_list, date,
                        symbols):
    ''' Return factor exposure matrix(big X).

    Parameters
    ----------
    df_target_weight : array-like
        weights for different securities in portfolio

    risk_model: dictionary
        Including specific risk, different factor exposure dataframe for all
        symbols.

    factor_list: list
        Factor exposure list.
    Returns
    -------
    factor_exposure: DataFrame
        Big X on target date for input symbols.
    '''
    factor_exposure = pd.DataFrame(index=symbols)
    for factor in factor_list:
        try:
            factor_exposure[factor] = risk_model[factor].asMatrix().\
                                      loc[date, symbols]
        except KeyError:
            raise KeyError('invalid input date: %s' % date)
    factor_exposure.columns = gftIO.strSet2Np(factor_exposure.columns.values)
    factor_exposure = factor_exposure.fillna(0)

    return factor_exposure


def statistics(asset_return, weights, cov_matrix_V):
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
    prisk : float
        expected portfolio total risk
    pret / pvol : float
        Sharpe ratio for rf=0
    '''
    weights = np.array(weights)
    pret = np.sum(asset_return.ix[:target_date].mean().fillna(0) * weights)
    pvol = np.sqrt(np.dot(weights.T, np.dot(asset_return.ix[:target_date].
                                            cov().fillna(0), weights)))
    prisk = calculate_total_risk(weights, cov_matrix_V)

    return np.array([pret, pvol, prisk, pret / pvol])


def portfolio_optimization(target_mode, df_position_limit, risk_model,
                           asset_return, df_target_portfolio_weight,
                           target_risk, target_return, target_date):
    """
    optimize fund weight target on different constraints, objective, based on
    target type and mode, fund return target, fund weight, group weight， etc.

    Parameters
    ----------
    target_mode: dictionary
        target optimization type({type: mode})
        0: minimum_variance
        1: minimum_total_risk_at_target_return
        2: maximum_return_at_target_risk

    df_position_limit: int
        group position limit.

    df_asset_return: Dataframe, OTV,
        index=date, O: asset names, O: group name, V: asset return.

    risk_model: dictionary
        Including specific risk, different factor exposure dataframe for all
        symbols.

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
    df_target_portfolio_weight = df_target_portfolio_weight.asMatrix().dropna(axis=0, how='all')
    # assign all symbols and dates from specific risk column names and index
    # repsectively
    all_symbols = risk_model['specificRisk'].columns.intersection(
        df_target_portfolio_weight.columns)
    if len(all_symbols) < 1:
        print("no intersected symbols.")
    asset_weight = df_target_portfolio_weight.loc[:, all_symbols]
    try:
        init_weight = asset_weight.ix[target_date].dropna()
    except KeyError:
        raise KeyError('invalid input date: %s' % target_date)
    target_symbols = risk_model['specificRisk'].columns.intersection(init_weight.index)

    noa = len(target_symbols)

    all_dates = risk_model['specificRisk'].index.intersection(df_target_portfolio_weight.index)
    asset_return = asset_return.asMatrix().loc[all_dates, target_symbols]
    asset_weight = df_target_portfolio_weight.loc[:, target_symbols]

    # get the target date initial holding
    init_weight = asset_weight.ix[target_date].dropna()

    # make a diagnal matrix using speicific risk
    diag = risk_model['specificRisk'].loc[target_date, target_symbols]
    delta = pd.DataFrame(np.diag(diag), index=diag.index,
                         columns=diag.index).fillna(0)

    ls_factor = ['E8D70EA915C420F9B9005BB21540788C',
                 'EB8553C313C38BC180E1972D798622BA',
                 '441E8F64A7582F553BCBE42A216285F1',
                 'ABEC912F31E326F4C1FC507AF787C8FA',
                 '873CF94D09229206D675ACC32328DC24',
                 'B9CCDA635F039E84D489F964DB08BC5C',
                 'BE3E35A7C0CB49EB9E1CB41D566563E7',
                 'B1B02CFAB81248BAA87754E760769BD2']

    big_X = get_factor_exposure(df_target_portfolio_weight, risk_model,
                                ls_factor, target_date, target_symbols)

    # assign all factors from big X factor matrix
    all_factors = big_X.columns

    covariance_matrix = risk_model['ret_cov'].set_index('date')
    cov_matrix = covariance_matrix.ix[target_date]
    cov_matrix = cov_matrix.pivot(index='factorid1',
                                  columns='factorid2', values='value')
    cov_matrix = cov_matrix.reindex(all_factors, all_factors,
                                    fill_value=np.nan)

    cov_matrix_V = big_X.dot(cov_matrix).dot(big_X.T) + delta


    def calculate_total_risk(weights, cov_matrix):
        port_var = np.dot(np.dot(weights.T, cov_matrix), weights)
        return port_var
    
    
    def max_func_return(weights):
        return -np.sum(asset_return.ix[:target_date].mean().fillna(0) * weights)
    
    
    def minimum_variance(weights):
        return calculate_total_risk(weights, cov_matrix_V)
    
    
    def min_func_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(
            asset_return.ix[:target_date].cov(), weights)))
    

    optimizer = {
        0: {'fitness': minimum_variance,
            'bnds': tuple((0, 1) for x in range(noa)),
            'cons': ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})},
        1: {'fitness': minimum_variance,
            'bnds': tuple((0, 1) for x in range(noa)),
            'cons': ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1},
                     {'type': 'eq', 'fun': lambda x:
                      -target_return + np.sum(asset_return.ix
                                              [:target_date].mean().
                                              fillna(0) * x)})},
        2: {'fitness': max_func_return,
            'bnds': tuple((0, 1) for x in range(noa)),
            'cons': ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1},
                     {'type': 'eq', 'fun': lambda x:
                      target_risk - calculate_total_risk(x, cov_matrix_V)})}
        }
    init_guess = noa * [1. / noa, ]
    opts = sco.minimize(optimizer[target_mode]['fitness'], init_guess,
                        method='SLSQP', bounds=optimizer[target_mode]['bnds'],
                        constraints=optimizer[target_mode]['cons'])

    opts_weight = opts['x']

    return pd.Series(opts_weight, index=target_symbols, name=target_date)

target_mode = 2
result = portfolio_optimization(target_mode, position_limit, risk_model,
                                asset_return, data['target_portfolio_weight'],
                                target_risk, target_return, target_date)
