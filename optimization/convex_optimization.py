# coding: utf-8
from lib.gftTools import gftIO
import logging
import numpy as np
import pandas as pd
import re
import cvxpy as cvx

from cvxopt import matrix, spmatrix
from lib.gftTools import gsConst, gftIO


def log_ret(rets):
    """Log of return relatives, ln(1+r), for a given DataFrame rets."""
    return np.log(rets + 1)


def create_constraint(obj, df_limit, ls_constraint):
    if isinstance(df_limit, pd.DataFrame):
        ls_constraint.append(obj >= df_limit.loc[:, 'value1'].values)
        ls_constraint.append(obj <= df_limit.loc[:, 'value2'].values)
        return ls_constraint


def convex_optimizer(context, mode, position_limit, forecast_return,
                     original_portfolio, target_risk, target_return, X,
                     covariance_matrix, delta, constraint):
    '''
    optimize fund weight target on different constraints, objective, based on
    target type and mode, fund return target, fund weight, group weight， etc.

    Parameters
    ----------
    mode: dictionary
        target optimization type({type: mode})
        0: minimum risk.
        1: minimum risk subject to target return.
        2: maximum return subject to target risk.

    original_portfolio: OOTV
        input original waiting for optimization

    forecast_return: Dataframe, OTV,
        asset return for all symbols.
        index=date, O: asset names, V: asset return.

    target_return: double
        Target return for portfolio respected to benchmark.

    target_risk: double
        Portfolio risk tolerance whose objective is maximum return.

    cov_matrix: OOTV
        covariance matrix from risk model if holdings are stocks.

    X: pandas panel
        factor exposure

    delta: OOTV
        specific risk, diagonal matrix

    constraint: dictionaries tuples
        dictionary: OOTV, OTVV

    Returns
    -------
    df_result: DataFrame
        Optimized value of weight.
        Index: target date.
        Columns: assets names.
    '''
    # create logger
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


    # convert gft table to pandas dataframe
    if isinstance(original_portfolio, gftIO.GftTable):
        original_portfolio = original_portfolio.asColumnTab()
    if isinstance(forecast_return, gftIO.GftTable):
        forecast_return = forecast_return.asMatrix()
    if isinstance(covariance_matrix, gftIO.GftTable):
        covariance_matrix = covariance_matrix.asColumnTab()
    if isinstance(delta, gftIO.GftTable):
        delta = delta.asMatrix()

    all_factors_gid = covariance_matrix['factorid1'].unique()


    # all_factors_gid = gftIO.strSet2Np(np.array(all_factors))
    # all_style_factors_gid = gftIO.strSet2Np(np.array(ls_style_factor))
    df_industries_asset_weight = original_portfolio.drop_duplicates(
        subset=['date', 'symbol'])

    df_industries_asset_weight = df_industries_asset_weight.dropna(
        axis=0, subset=['industry', 'symbol'], how='any')

    datetime_index = pd.DatetimeIndex(df_industries_asset_weight['date'].unique())
    target_date = datetime_index[0]


    # get unique symbols from the portfolio
    unique_symbol = df_industries_asset_weight[df_industries_asset_weight['date']==target_date]['symbol'].unique()

    noa = len(unique_symbol)
    if noa <= position_limit:
        position_limit = noa


    # select the number of position limit ranked symbols by requested mode.
    if mode == 1:
        unique_symbol = forecast_return.loc[:target_date, unique_symbol].fillna(0).std().sort_values(ascending=False)[:position_limit].index
    else:
        unique_symbol = log_ret(forecast_return.loc[:target_date,unique_symbol].fillna(0)).mean().sort_values(ascending=False)[:position_limit].index


    # create dataframe for output
    df_opts_weight = pd.DataFrame(data=np.nan, columns=unique_symbol,
                                  index=datetime_index)
    dict_opts_status = {}

    for target_date in datetime_index:
        logger.debug('target date: %s', target_date)
        try:
            if np.setdiff1d(unique_symbol, delta.loc[target_date].index):
                pass
        except ValueError:
            logger.debug('some symbols are not in asset weight symbols.')


        # get a pivot table, setting industry and symbol for two levels on the column
        df_pivot_industries_asset_weights = pd.pivot_table(
            df_industries_asset_weight, values='value', index=['date'],
            columns=['industry', 'symbol'])

        # pivot the original dataframe to multi-index dataframe
        # level 0 value: industry
        # level 1 value: assets, the order of assets are changed.
        df_pivot_industries_asset_weights = df_pivot_industries_asset_weights.fillna(0)
        idx_level_0_value = df_pivot_industries_asset_weights.columns.get_level_values(0)
        idx_level_0_value = idx_level_0_value.drop_duplicates()
        idx_level_1_value = df_pivot_industries_asset_weights.columns.get_level_values(1)
        # use the mean return prior target date as the predicted return temperarily
        # will use the forecasted return as ultimate goal
        asset_expected_return = forecast_return.loc[:target_date, idx_level_1_value].fillna(0)
        rets_mean = log_ret(asset_expected_return).mean()


        # get delta on the target date, which is a diagonal matrix
        diag = delta.loc[target_date, idx_level_1_value]
        delta_on_date = pd.DataFrame(np.diag(diag), index=diag.index,
                                     columns=diag.index).fillna(0)

        covariance_matrix.set_index('date', inplace=True)


        # get covariance matrix, re-index from the list of all factors' gid
        cov_matrix = covariance_matrix.loc[target_date]
        cov_matrix = cov_matrix.pivot(index='factorid1', columns='factorid2', values='value')
        cov_matrix = cov_matrix.reindex(all_factors_gid, all_factors_gid, fill_value=np.nan)

        # # big X is sigma in the quadratic equation, size = 35 * number of assets
        big_X = X.loc[target_date]
        big_X = big_X[idx_level_1_value]
        big_X = big_X.reindex(index=all_factors_gid)
        big_X.fillna(0,inplace=True)


        # # setup the optimization parameter

        # Factor model portfolio optimization.
        # w is the solution x variable
        w = cvx.Variable(noa)
        f = big_X.values*w

        # gamma parameter, multiplier of risk
        gamma = cvx.Parameter(sign='positive')
        # Lmax is maximum leverage
        Lmax = cvx.Parameter()
        ret = w.T * rets_mean.values

        # create quadratic form of risk
        risk = cvx.quad_form(f, cov_matrix.values) + cvx.quad_form(w, delta_on_date.values)

        eq_constraint = [cvx.sum_entries(w) == 1,
                         cvx.norm(w, 1) <= Lmax]

        # setup value constraint:
        constraint_value = []
        for cst in constraint:
            try:
                multiply_matrix = cst['ts_asset_group_loading'].\
                                  loc[target_date].loc[idx_level_1_value,
                                                       idx_level_1_value]
            except KeyError:
                multiply_matrix = cst['ts_asset_group_loading'].\
                                  loc[target_date].loc[idx_level_1_value]

            df_boundary = cst['ts_group_loading_range'].asColumnTab()
            df_boundary = df_boundary.loc[(df_boundary['date'] == target_date)]
            df_boundary.drop('date', axis=1, inplace=True)
            df_boundary.set_index('target', inplace=True)
            if set(idx_level_1_value).issubset(set(df_boundary.index)):
                df_boundary = df_boundary.reindex(idx_level_1_value)
                logger.debug("df_boundary re-index")
            # df_boundary = df_boundary.reindex(idx_level_1_value)
            #logger.debug("multiply_matrix %s",multiply_matrix.shape)
            #logger.debug("df_boundary %s",df_boundary.shape)
            create_constraint(multiply_matrix.T.values*w,
                              df_boundary, constraint_value)
            # leverage level and risk adjusted parameter
            Lmax.value = 1
            gamma.value = 1
            if mode == gsConst.Const.MinimumRisk:
                # maximize negative product of gamma and risk
                prob_factor = cvx.Problem(cvx.Maximize(-gamma*risk),
                                          eq_constraint + constraint_value)
            if mode == gsConst.Const.MinimumRiskUnderReturn:
                # minimum risk subject to target return, Markowitz Mean_Variance Portfolio
                prob_factor = cvx.Problem(cvx.Maximize(-gamma*risk),
                                          [ret >= target_return]+eq_constraint+constraint_value)
            if mode == gsConst.Const.MaximumReturnUnderRisk:
                # Portfolio optimization with a leverage limit and a bound on risk
                prob_factor = cvx.Problem(cvx.Maximize(ret),
                                          [risk <= target_risk]+eq_constraint+constraint_value)
        prob_factor.solve(verbose=False)
        logger.debug(prob_factor.status)
