# coding: utf-8
from lib.gftTools import gftIO
import logging
import numpy as np
import pandas as pd
import re
import cvxpy as cvx

from lib.gftTools import gsConst, gftIO, gsUtils


def log_ret(rets):
    """Log of return relatives, ln(1+r), for a given DataFrame rets."""
    return np.log(rets + 1)


def create_constraint(obj, df_limit, ls_constraint):
    if isinstance(df_limit, pd.DataFrame):
        ls_constraint.append(obj >= df_limit.loc[:, 'value1'].values)
        ls_constraint.append(obj <= df_limit.loc[:, 'value2'].values)
        return ls_constraint


def convex_optimizer(context,mode,position_limit,forecast_return,original_portfolio,target_risk,target_return,X,covariance_matrix,delta,*constraint):
    '''
    optimize fund weight target on different constraints, objective, based on
    target type and mode, fund return target, fund weight, group weight， etc.

    Pseudo code
    -----------
    1. transfer gftTable data(original portfolio, forecast return, covariance matrix, delta) to dataframe.

    2. remove duplicated rows in the original portfolio.

    3. create empty weight and status dataframe for solution output.

    4. loop the datetimeindex in the original portfolio:

    fetch number of position limit sorted assets by return or volitility according to requested mode.

    check if these assets have data in the input delta matrix in case missing value during risk calculation.

    use the historical mean return prior target date as the predicted return temperarily

    get target date data of sigma, X, delta from input X panel, covariance matrix, delta matrix.

    setup the Factor model portfolio optimization parameter for solution.

    create weight variables, factor variable, return variable, risk variable, equality constraint.

    load inequality constraint.

    solve for optimized weight at this period according to the problem definition

    if prob_factor.status == 'infeasible':
        # relax constraint
    else:
        df_opts_weight.loc[target_date, idx_level_1_value] = np.array(w.value.astype(np.float64)).T

    save optimized solution weight into a dataframe and the solution status to a dictionary.

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

    df_industries_asset_weight = original_portfolio.drop_duplicates(
        subset=['date', 'symbol'])

    df_industries_asset_weight = df_industries_asset_weight.dropna(
        axis=0, subset=['industry', 'symbol'], how='any')

    datetime_index = pd.DatetimeIndex(df_industries_asset_weight['date'].unique())
    target_date = datetime_index[0]

    # get unique symbols from the portfolio
    unique_symbol = df_industries_asset_weight['symbol'].unique()

    noa = len(unique_symbol)
    if noa <= position_limit:
        position_limit = noa

    # create dataframe for output
    df_opts_weight = pd.DataFrame(data=np.nan, columns=unique_symbol,
                                  index=datetime_index)
    dict_opts_status = pd.DataFrame(data=np.nan, columns=gsUtils.getGodGid(),
                                  index=datetime_index)

    for target_date in datetime_index:
        logger.debug('target date: %s', target_date)
        # select the number of position limit ranked symbols by requested mode.
        if mode == gsConst.Const.MinimumRiskUnderReturn:
            target_assets = forecast_return.loc[:target_date, unique_symbol].fillna(0).std().sort_values(ascending=False)[:position_limit].index
        else:
            target_assets = log_ret(forecast_return.loc[:target_date,unique_symbol].fillna(0)).mean().sort_values(ascending=False)[:position_limit].index
        logger.debug('target assets: %s', target_assets.shape)

        try:
            if np.setdiff1d(target_assets, delta.loc[target_date].index):
                pass
        except ValueError:
            logger.debug('some assets in the original portfolio are not in input delta.')

        # use the mean return prior target date as the predicted return temperarily
        # will use the forecasted return as ultimate goal
        asset_expected_return = forecast_return.loc[:target_date, target_assets].fillna(0)
        rets_mean = log_ret(asset_expected_return).mean()

        # get delta on the target date, which is a diagonal matrix
        diag = delta.loc[target_date, target_assets]
        delta_on_date = pd.DataFrame(np.diag(diag), index=diag.index,
                                     columns=diag.index).fillna(0)

        # extra action in case the index is set as date in the function asColumnTab
        try:
            covariance_matrix.set_index('date', inplace=True)
        except:
            pass

        # get covariance matrix, re-index from the list of all factors' gid
        cov_matrix = covariance_matrix.loc[target_date]
        cov_matrix = cov_matrix.pivot(index='factorid1', columns='factorid2', values='value')
        cov_matrix = cov_matrix.reindex(all_factors_gid, all_factors_gid, fill_value=np.nan)

        # big X is sigma in the quadratic equation, size = 35 * number of assets
        big_X = X.loc[target_date]
        big_X = big_X.loc[target_assets]
        big_X = big_X.reindex(columns=all_factors_gid)
        big_X.fillna(0,inplace=True)

        # setup the Factor model portfolio optimization parameter
        # w is the solution x variable
        w = cvx.Variable(noa)
        f = big_X.T.values*w

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
        """
        # asset constraint:
        Asset ts_asset_group_loading diagonal matrix:(OOTV, Matrix M1(n * n)), 59 * 59.
        asset value range, value1, value2, 58 * 2.
        select 58 * 58 from diagonal matrix, order by idx_leve1_value.
        product: multiply_matrix.T.values * w, [58x58] * [58x1]

        # industry constraint:
        industry ts_asset_group_loading sparse matrix:(OOTV, Matrix M1(n * m)), 58 * 26.
        industry value range, value1, value2, 26 * 2.
        select 58 * 26 from sparse matrix, order by group_constraint index value.
        product: multiply_matrix.T.values * w, [26x58] * [58x1]=[26x1]

        # factor constraint:
        factor ts_asset_group_loading exposure matrix:(OOTV, Matrix M1(n * m)), 35 * 3436.
        factor exposure value range, value1, value2, 35 * 2.
        select 35 * 58 from factor exposure matrix.
        product: multiply_matrix.T.values * w, [35x58] * [58x1]=[35x1]
        """
        constraint_value = []
        for cst in constraint:
            # in order to align the production.
            df_boundary = cst['ts_group_loading_range'].asColumnTab()
            df_boundary = df_boundary.loc[(df_boundary['date'] == target_date)]
            df_boundary.drop('date', axis=1, inplace=True)
            df_boundary.set_index('target', inplace=True)
            df_boundary_idx = df_boundary.index

            multiply_matrix = cst['ts_asset_group_loading'].\
                                  loc[target_date].loc[target_assets,
                                                       df_boundary_idx].fillna(0)

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
        if prob_factor.status == 'infeasible':
            df_opts_weight.loc[target_date, target_assets] = np.nan
            df_opts_weight.fillna(method='pad', inplace=True)
            dict_opts_status.loc[target_date] = gsConst.Const.Infeasible
        else:
            df_opts_weight.loc[target_date, target_assets] = np.array(w.value.astype(np.float64)).T
            dict_opts_status.loc[target_date] = gsConst.Const.Feasible

        return {'weight':df_opts_weight, 'status':dict_opts_status}

import datetime
time_start = datetime.datetime.now()
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not handler:
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.debug('start')
#if not context:
context = gftIO.zload("/home/weiwu/share/optimize/context.pkl")
mode = gftIO.zload("/home/weiwu/share/optimize/mode.pkl")
position_limit = gftIO.zload("/home/weiwu/share/optimize/position_limit.pkl")
forecast_return = gftIO.zload("/home/weiwu/share/optimize/forecast_return.pkl")
original_portfolio = gftIO.zload("/home/weiwu/share/optimize/original_portfolio.pkl")
target_risk = gftIO.zload("/home/weiwu/share/optimize/target_risk.pkl")
target_return = gftIO.zload("/home/weiwu/share/optimize/target_return.pkl")
X = gftIO.zload("/home/weiwu/share/optimize/X.pkl")
covariance_matrix = gftIO.zload("/home/weiwu/share/optimize/covariance_matrix.pkl")
delta = gftIO.zload("/home/weiwu/share/optimize/delta.pkl")
constraint1 = gftIO.zload("/home/weiwu/share/optimize/constraint1.pkl")
constraint2 = gftIO.zload("/home/weiwu/share/optimize/constraint2.pkl")
constraint3 = gftIO.zload("/home/weiwu/share/optimize/constraint3.pkl")
logger.debug('data loaded')
time_loading = datetime.datetime.now()

convex_optimizer(context,mode,position_limit,forecast_return,original_portfolio,target_risk,target_return,X,covariance_matrix,delta,constraint1, constraint2, constraint3)

time_end = datetime.datetime.now()
print(time_loading - time_start)
print(time_end - time_loading)
