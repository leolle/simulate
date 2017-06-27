# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import os
import warnings
from cvxopt import matrix, solvers, spmatrix, sparse
from cvxopt.blas import dot

from lib.gftTools import gftIO

# fetch data
path = r'd:/share/optimize/'

target_mode = gftIO.zload(os.path.join(path, 'x0.pkl'))
position_limit = gftIO.zload(os.path.join(path, 'x1.pkl'))
covariance_matrix = gftIO.zload(os.path.join(path, 'fx2.pkl'))
asset_return = gftIO.zload(os.path.join(path, 'x3.pkl'))
asset_weight = gftIO.zload(os.path.join(path, 'x4.pkl'))
target_risk = gftIO.zload(os.path.join(path, 'x5.pkl'))
target_return = gftIO.zload(os.path.join(path, 'x6.pkl'))
risk_model = gftIO.zload(os.path.join(path, 'risk_model.pkl'))

# import U_PNL_FITNESS as fitness

solvers.options['show_progress'] = False


def logrels(rets):
    """Log of return relatives, ln(1+r), for a given DataFrame rets."""
    return np.log(rets + 1)


def check_boundary_constraint(df_asset_bound, df_group_bound,
                              df_exposure_bound, df_exposure):
    ''' check input boundary limit.

    Parameters
    ----------
    df_asset_bound : dataframe-like
        Input lower and upper boundary dataframe for each asset.

    df_group_bound : dataframe-like
        Input lower and upper boundary dataframe for each group.

    df_exposure_bound : dataframe-like
        Input lower and upper boundary dataframe for each factor.

    df_exposure : dataframe
        Big X.

    Returns
    -------
    True: all boundaries in condition.
    False: any boundaries out of condition.
    '''
    if ((df_asset_bound.lower) < 0).any():
        raise ValueError('short is not supported.')
    if ((df_asset_bound.upper) > 1).any():
        raise ValueError('asset upper boundary is bigger than 1.')
    if (np.sum(df_asset_bound.lower) > 1):
        raise ValueError('asset lower boundary sum is bigger than 1.')
    if (np.sum(df_asset_bound.upper) < 1):
        raise ValueError('asset upper boundary sum is smaller than 1.')
    if ((df_asset_bound.lower > df_asset_bound.upper).any()):
        raise ValueError('asset lower boundary is bigger than upper boundary')

    if ((df_group_bound.lower) < 0).any():
        raise ValueError('short is not supported.')
    if ((df_group_bound.upper) > 1).any():
        raise ValueError('group upper boundary is bigger than 1.')
    if (np.sum(df_group_bound.lower) > 1):
        raise ValueError('group lower boundary sum is bigger than 1.')
    if (np.sum(df_group_bound.upper) < 1):
        raise ValueError('group upper boundary sum is smaller than 1.')
    if ((df_group_bound.lower > df_group_bound.upper).any()):
        raise ValueError('group lower boundary is bigger than upper boundary')

    df_factor_exposure_bound_check = pd.DataFrame(index=df_exposure.T.index,
                                                  columns=[['lower', 'upper']])
    df_factor_exposure_bound_check.lower = df_exposure.T.min(axis=1)
    df_factor_exposure_bound_check.upper = df_exposure.T.max(axis=1)

    if (df_factor_exposure_bound_check.upper < df_exposure_bound.upper).any():
        raise ValueError('factor exposure upper setting error')

    if (df_factor_exposure_bound_check.lower > df_exposure_bound.lower).any():
        raise ValueError('factor exposure lower setting error')

    return True


def statistics(weights, rets, covariance):
    """Compute expected portfolio statistics from individual asset returns.

    Parameters
    ----------
    rets : DataFrame
        Individual asset returns.  Use numeral rather than decimal form
    weights : array-like
        Individual asset weights, nx1 vector.

    Returns
    -------
    list of (pret, pvol, pstd); these are *per-period* figures (not annualized)
        pret : expected portfolio return
        pvol : expected portfolio variance
        psr  : sharpe ratio

    """

    if isinstance(weights, (tuple, list)):
        weights = np.array(weights)

    if isinstance(weights, matrix):
        pret = np.sum(logrels(rets.values).mean() * weights)
        pvol = np.dot(weights.T, np.dot(covariance, weights))
    elif isinstance(weights, pd.DataFrame):
        pret = np.dot(weights.values, logrels(rets).mean())
        pvol = np.dot(weights, np.dot(covariance, weights.T))
    pstd = np.sqrt(pvol)

    return [pret, pvol, pret/pstd]


def get_factor_exposure(risk_model, factor_list, date, symbols):
    ''' Return factor exposure matrix(big X).

    Parameters
    ----------
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


def find_nearest(array, value):
    """ Find the nearest value index from an array"""
    if isinstance(array, list):
        array = np.array(array)
        idx = (np.abs(array-value)).argmin()
    return idx


def CVXOptimizerBnd(context, target_mode, position_limit, risk_model,
                    asset_return, asset_weight, target_risk,
                    target_return, target_date, asset_constraint,
                    group_constraint, exposure_constraint):
    """
    optimize fund weight target on different constraints, objective, based on
    target type and mode, fund return target, fund weight, group weight， etc.

    Parameters
    ----------
    target_date: Timestamp
        Specific date.

    target_mode: dictionary
        target optimization type({type: mode})
        0: minimum risk.
        1: minimum risk subject to target return.
        2: maximum return subject to target risk.

    asset_return: Dataframe, OTV,
        asset return for all symbols.
        index=date, O: asset names, V: asset return.

    risk model: dictionary
        Risk factor exposure: DataFrame
            所有股票在因子上暴露的值，p.s. 如有8个因子，就有8个DataFrame,
            得把所有8个因子某一天所有值先取出来得到一个n*k的矩阵.n为股票，k为因子
        Specific Risk: DataFrame
            用来组成对角矩阵Delta.

    asset_weight: Dataframe, OOTV
        T=date, O: asset names, O: group names, V: asset weight.
        weight bound of each asset. Default is equal weight.

    target_return: double
        Target return for portfolio respected to benchmark.

    target_risk: double
        Portfolio risk tolerance whose objective is maximum return.

    Returns:
    ----------
    df_result: DataFrame
        Optimized value of weight.
        Index: target date.
        Columns: assets names.

    """
    asset_return = asset_return.asMatrix()
    asset_weights = asset_weight.asColumnTab()
    target_date = pd.to_datetime(target_date)
    if asset_constraint is not None:
        asset_constraint = asset_constraint.asMatrix()
    if group_constraint is not None:
        group_constraint = group_constraint.asMatrix()
    if exposure_constraint is not None:
        exposure_constaint = exposure_constraint.asMatrix()

    # regex to search all the factors
    ls_factor = [x[:-4] for x in risk_model.keys() if re.search(".ret$", x)]
    # ls_factor = [x.split('.')[0] for x in ls_factor]

    specific_risk = risk_model['specificRisk'].pivot(
        index='date', columns='symbol', values='specificrisk')
    # target_date = pd.datetime(year=2016, month=10, day=31)
    # target_return = -0.00096377
    # target_risk = 3.16026352e-06
    #target_mode = 1
    #position_limit = 500

    # find the nearest date next to target date from specific risk
    dt_next_to_target = specific_risk.index.searchsorted(target_date)
    dt_next_to_target = specific_risk.index[dt_next_to_target]
    target_specific_risk = specific_risk.loc[dt_next_to_target, :]

    # drop duplicated rows at date
    df_industries_asset_weight = asset_weights.drop_duplicates(
        subset=['date', 'symbol'])
    try:
        df_industries_asset_init_weight = df_industries_asset_weight[
            df_industries_asset_weight['date'] == target_date].dropna()
    except KeyError:
        raise KeyError('invalid input date: %s' % target_date)

    # drop incomplete rows
    df_industries_asset_init_weight = df_industries_asset_init_weight.dropna(
        axis=0, subset=['industry', 'symbol'], how='any')

    unique_symbol = df_industries_asset_init_weight['symbol'].unique()
    target_symbols = target_specific_risk.index.intersection(unique_symbol)
    if position_limit > len(target_symbols):
        print("position limit is bigger than total symbols")
        position_limit = len(target_symbols)

    # get random symbols at the target position limit
    arr = list(range(len(target_symbols)))
    np.random.shuffle(arr)
    target_symbols = target_symbols[arr[:position_limit]]

    df_industries_asset_target_init_weight = df_industries_asset_init_weight.\
                                             loc[df_industries_asset_init_weight['symbol'].isin(target_symbols)]
    df_pivot_industries_asset_weights = pd.pivot_table(
        df_industries_asset_target_init_weight, values='value', index=['date'],
        columns=['industry', 'symbol'])
    df_pivot_industries_asset_weights = df_pivot_industries_asset_weights.fillna(0)
    
    noa = len(target_symbols)
    if noa < 1:
        raise ValueError("no intersected symbols from specific risk and initial holding.")

    # get the ordered column list
    idx_level_0_value = df_pivot_industries_asset_weights.columns.get_level_values(0)
    idx_level_1_value = df_pivot_industries_asset_weights.columns.get_level_values(1)
    asset_return = asset_return.loc[:target_date, idx_level_1_value].fillna(0)

    diag = specific_risk.loc[dt_next_to_target, idx_level_1_value]
    delta = pd.DataFrame(np.diag(diag), index=diag.index,
                         columns=diag.index).fillna(0)

    big_X = get_factor_exposure(risk_model, ls_factor, target_date,
                                idx_level_1_value)
    big_X = big_X.fillna(0)
    all_factors = big_X.columns

    covariance_matrix = risk_model['ret_cov'].set_index('date')

    cov_matrix = covariance_matrix.loc[dt_next_to_target]
    cov_matrix = cov_matrix.pivot(index='factorid1', columns='factorid2',
                                  values='value')
    cov_matrix = cov_matrix.reindex(all_factors, all_factors, fill_value=np.nan)

    cov_matrix_V = big_X.dot(cov_matrix).dot(big_X.T) + delta

    P = matrix(cov_matrix_V.values)
    q = matrix(np.zeros((noa, 1)), tc='d')

    A = matrix(1.0, (1, noa))
    b = matrix(1.0)

    # for group weight constraint
    groups = df_pivot_industries_asset_weights.groupby(
        axis=1, level=0, sort=False, group_keys=False).count().\
        iloc[-1, :].values
    num_group = len(groups)
    num_asset = np.sum(groups)


    # set boundary vector for h
    df_asset_weight = pd.DataFrame({'lower': [0.0], 'upper': [1.0]},
                                   index=idx_level_1_value)
    df_group_weight = pd.DataFrame({'lower': [0.0], 'upper': [1.0]},
                                   index=set(idx_level_0_value))
    df_factor_exposure_bound = pd.DataFrame(index=big_X.T.index, columns=[['lower', 'upper']])
    df_factor_exposure_bound.lower = (1.0/noa)*big_X.sum()*(0.999991)
    df_factor_exposure_bound.upper = (1.0/noa)*big_X.sum()*(1.000009)

    if asset_constraint is not None:
        df_asset_weight.lower.ix[asset_constraint.lower] = asset_constraint.lower
        df_asset_weight.upper.ix[asset_constraint.upper] = asset_constraint.upper
    if group_constraint is not None:
        df_group_weight.lower.ix[group_constraint.lower] = group_constraint.lower
        df_group_weight.upper.ix[group_constraint.upper] = group_constraint.upper
    if exposure_constraint is not None:
        df_factor_exposure_bound.lower.ix[exposure_constraint.lower] = exposure_constraint.lower
        df_factor_exposure_bound.upper.ix[exposure_constraint.upper] = exposure_constraint.upper

    if check_boundary_constraint(df_asset_weight, df_group_weight,
                                 df_factor_exposure_bound, big_X):
        print("boundary setting is fine")

    df_asset_bnd_matrix = matrix(np.concatenate(((df_asset_weight.upper,
                                                  df_asset_weight.lower)), 0))
    df_group_bnd_matrix = matrix(np.concatenate(((df_group_weight.upper,
                                                  df_group_weight.lower)), 0))
    df_factor_exposure_bnd_matrix = matrix(np.concatenate(((df_factor_exposure_bound.upper,
                                                            df_factor_exposure_bound.lower)), 0))

    rets_mean = logrels(asset_return).mean()
    avg_ret = matrix(rets_mean.values)
    G = matrix(-np.transpose(np.array(avg_ret)))
    h = matrix(-np.ones((1, 1))*target_return)
    G_sparse_list = []
    for i in range(num_group):
        for j in range(groups[i]):
            G_sparse_list.append(i)
    Group_sub = spmatrix(1.0, G_sparse_list, range(num_asset))

    Group_sub = matrix(sparse([Group_sub, -Group_sub]))

    asset_sub = matrix(np.eye(noa))
    asset_sub = matrix(sparse([asset_sub, -asset_sub]))
    exp_sub = matrix(np.array(big_X.T))
    exp_sub = matrix(sparse([exp_sub, - exp_sub]))

    # minimum risk
    if target_mode == 0:
        # G = matrix(-np.eye(noa), tc='d')
        # h = matrix(-np.zeros((noa, 1)), tc='d')
        if exposure_constraint is not None:
            G = matrix(sparse([asset_sub, Group_sub, exp_sub]))
            h = matrix(sparse([df_asset_bnd_matrix, df_group_bnd_matrix,
                               df_factor_exposure_bnd_matrix]))
        else:
            G = matrix(sparse([asset_sub, Group_sub]))
            h = matrix(sparse([df_asset_bnd_matrix, df_group_bnd_matrix]))

        sol = solvers.qp(P, q, G, h, A, b)
        df_opts_weight = pd.DataFrame(np.array(sol['x']).T,
                                      columns=target_symbols,
                                      index=[target_date])
    # minimum risk subject to target return, Markowitz Mean_Variance Portfolio
    elif target_mode == 1:
        if exposure_constraint is not None:
            G = matrix(sparse([G, asset_sub, Group_sub, exp_sub]))
            h = matrix(sparse([h, df_asset_bnd_matrix, df_group_bnd_matrix,
                               df_factor_exposure_bnd_matrix]))
        else:
            G = matrix(sparse([G, asset_sub, Group_sub]))
            h = matrix(sparse([h, df_asset_bnd_matrix, df_group_bnd_matrix]))
        sol = solvers.qp(P, q, G, h, A, b)
        df_opts_weight = pd.DataFrame(np.array(sol['x']).T,
                                      columns=target_symbols,
                                      index=[target_date])
    # Computes a tangency portfolio, i.e. a maximum Sharpe ratio portfolio
    elif target_mode == 2:
        # exp_rets*x >= 1
        G = matrix(-np.transpose(np.array(avg_ret)))
        h = matrix(-np.ones((1, 1)))

        if exposure_constraint is not None:
            G = matrix(sparse([G, asset_sub, Group_sub, exp_sub]))
            h = matrix(sparse([h, df_asset_bnd_matrix, df_group_bnd_matrix,
                               df_factor_exposure_bnd_matrix]))
        else:
            G = matrix(sparse([G, asset_sub, Group_sub]))
            h = matrix(sparse([h, df_asset_bnd_matrix, df_group_bnd_matrix]))
        sol = solvers.qp(P, q, G, h)
        df_opts_weight = pd.DataFrame(np.array(sol['x']).T,
                                      columns=target_symbols,
                                      index=[target_date])
        # Rescale weights, so that sum(weights) = 1
        df_opts_weight /= df_opts_weight.sum(axis=1)

    if sol['status'] == 'optimal':
        print('result is optimal')
    elif sol['status'] == 'unknown':
        warnings.warn('Convergence problem, the algorithm failed to find a solution that satisfies the specified tolerances')

    return df_opts_weight
