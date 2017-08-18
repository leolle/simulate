# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
import logging

import numpy as np
import pandas as pd
import re
import cvxpy as cvx

from cvxopt import matrix, solvers, spmatrix, sparse
from cvxopt.blas import dot

from lib.gftTools import gsConst, gftIO


class RiskAnlysis(object):
    """ risk data preparation and getting attribute. """
    def __init__(self, risk_model_merge):
        self.risk_model = risk_model_merge
        self.ls_factors = [x for x in risk_model_merge.keys() if
                           re.match("[A-Z0-9]{32}$", x)]
        self.ls_factors_ret = [x[:-4] for x in risk_model_merge.keys() if
                               re.search(".ret$", x)]

    def get_factor_exposure(self, factor_list, date, symbols):
        ''' Return factor exposure matrix(big X).
        Parameters
        ----------
        risk_model: dictionary
            Including specific risk, different factor exposure dataframe for
            all symbols.
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
                factor_exposure[factor] = self.risk_model[factor].asMatrix().\
                                          loc[date, symbols]
            except KeyError:
                factor_exposure[factor] = np.nan
                # raise KeyError('invalid input date: %s' % date)
        factor_exposure.columns = gftIO.strSet2Np(factor_exposure.columns.
                                                  values)
        factor_exposure = factor_exposure.fillna(0)

        return factor_exposure

    def factor_exposure(self, asset_weight, frequency, factors):
        """ Get factor exposure.
        Keyword Arguments:
        asset_weight --
        """
        if isinstance(factors, dict):
            ls_factors = factors['factors']
        if isinstance(asset_weight, gftIO.GftTable):
            asset_weight = asset_weight.asMatrix()
            # asset_weight.fillna('ffill', inplace=True)
        # resample asset weight by monthly
        if frequency == 'MONTHLY':
            asset_weight.index.name = 'index'
            m = asset_weight.index.to_period('m')
            asset_weight = asset_weight.reset_index().groupby(m).last().set_index('index')
            asset_weight.index.name = ''

        asset_factor_exposure = pd.Panel({target_date: self.get_factor_exposure(
            ls_factors, target_date, asset_weight.columns).T for target_date in asset_weight.index})

        return asset_factor_exposure

    def cal_factor_exposure(self, asset_weight, frequency, factors):
        """ calculate asset factor exposure
        F = w^T * exposure
        Keyword Arguments:
        asset_weight -- 
        frequency    -- 
        factors      -- 
        """
        pn_factor_exposure = self.factor_exposure(asset_weight, frequency, factors)
        factor_exposure = pd.DataFrame(index=asset_weight.index,
                                       columns=pn_factor_exposure.major_axis)
        for target_date in asset_weight.index:
            factor_exposure.ix[target_date] = pn_factor_exposure.ix[target_date].dot(
                asset_weight.ix[target_date].fillna(0))
        return factor_exposure.replace(0, np.nan).fillna(method='ffill')

    def specific_risk(self):
        """ get specific risk from risk model
        Keyword Arguments:
        date --
        symbols --
        """
        # find the nearest date next to target date from specific risk
        specific_risk = self.risk_model['specificRisk'].pivot(
            index='date', columns='symbol', values='specificrisk')

        return specific_risk

    def delta(self, date, symbols):
        """
        create delta matrix from specific risk
        Keyword Arguments:
        date    -- 
        symbols -- 
        """
        # diag = self.specific_risk()
        risk = self.specific_risk()
        diag = risk.loc[date, symbols]
        delta = pd.DataFrame(np.diag(diag), index=diag.index,
                             columns=diag.index).fillna(0)

        return delta

    def covariance_matrix(self, date, factors):
        """
        Keyword Arguments:
        date    -- 
        factors -- 
        """
        covariance_matrix = self.risk_model['ret_cov'].set_index['date']
        covariance_matrix = covariance_matrix.loc[date]
        covariance_matrix = covariance_matrix.pivot(index='factorid1',
                                                    columns='factorid2',
                                                    values='value')
        covariance_matrix = covariance_matrix.reindex(factors, factors,
                                                      fill_value=np.nan)

    def factor_return(self):
        """ get factor return from all the factors """
        factors_ret = pd.DataFrame(index=self.risk_model[self.ls_factors_ret[0]+'.ret'].index,
                                   columns=self.ls_factors_ret)
        for factor in self.ls_factors_ret:
            factors_ret[factor] = self.risk_model[factor]
        factors_ret.columns = gftIO.strSet2Np(factors_ret.columns.
                                              values)
        return factors_ret


def logrels(rets):
    """Log of return relatives, ln(1+r), for a given DataFrame rets."""
    return np.log(rets + 1)


def get_ret_range(rets, df_asset_bound):
    ''' Calculate theoretical minimum and maximum theoretical returns.

    Parameters
    ----------
    rets: dataframe

    df_asset_bound : dataframe-like
        Input lower and upper boundary dataframe for each asset.

    Returns
    -------
    (f_min, f_max): tuple
    '''
    from copy import deepcopy
    f_min = 0
    f_max = 0

    rets = deepcopy(rets)

#    na_expected = np.average(rets, axis=0)
    na_expected = logrels(rets).mean().values

    na_signs = np.sign(na_expected)
    indices = np.where(na_signs == 0)
    na_signs[indices] = 1
    na_signs = np.ones(len(na_signs))

    rets = na_signs*rets
    na_expected = na_signs*na_expected

    na_sort_ind = na_expected.argsort()

    # First add the lower bounds on portfolio participation
    for i, fRet in enumerate(na_expected):
        f_min = f_min + fRet*df_asset_bound.lower[i]
        f_max = f_max + fRet*df_asset_bound.lower[i]


    # Now calculate minimum returns
    # allocate the max possible in worst performing equities
    # Subtract min since we have already counted it
    na_upper_add = df_asset_bound.upper - df_asset_bound.lower
    f_total_weight = np.sum(df_asset_bound.lower)

    for i, ls_ind in enumerate(na_sort_ind):
        f_ret_add = na_upper_add[ls_ind] * na_expected[ls_ind]
        f_total_weight = f_total_weight + na_upper_add[ls_ind]
        f_min = f_min + f_ret_add
        # Check if this additional percent puts us over the limit
        if f_total_weight > 1.0:
            f_min = f_min - na_expected[ls_ind] * (f_total_weight - 1.0)
            break
    else:
        raise ValueError("sum of total asset maximum weight is less than 1 ")
    # Repeat for max, just reverse the sort, i.e. high to low
    na_upper_add = df_asset_bound.upper - df_asset_bound.lower
    f_total_weight = np.sum(df_asset_bound.lower)
    if f_total_weight > 1:
        raise ValueError("sum of total asset minimum weight is bigger than 1 ")
    for i, ls_ind in enumerate(na_sort_ind[::-1]):
        f_ret_add = na_upper_add[ls_ind] * na_expected[ls_ind]
        f_total_weight = f_total_weight + na_upper_add[ls_ind]
        f_max = f_max + f_ret_add

        # Check if this additional percent puts us over the limit
        if f_total_weight > 1.0:
            f_max = f_max - na_expected[ls_ind] * (f_total_weight - 1.0)
            break

    return (f_min, f_max)


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

    df_exposure = df_exposure.loc[:, df_exposure_bound.index]
    df_factor_exposure_bound_check = pd.DataFrame(index=df_exposure.T.index,
                                                  columns=[['lower', 'upper']])
    df_factor_exposure_bound_check.lower = df_exposure.T.min(axis=1)
    df_factor_exposure_bound_check.upper = df_exposure.T.max(axis=1)

    if (df_factor_exposure_bound_check.upper < df_exposure_bound.upper).any():
        raise ValueError('factor exposure upper setting error')

    if (df_factor_exposure_bound_check.lower > df_exposure_bound.lower).any():
        raise ValueError('factor exposure lower setting error')

    return True


class ConstraintError(Exception):
    pass


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
        pret = np.sum(logrels(rets).mean().values * weights.T)
        pvol = np.dot(weights.T, np.dot(covariance, weights))
    elif isinstance(weights, pd.DataFrame):
        pret = np.dot(weights.values, logrels(rets).mean().T)
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

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

x0 = gftIO.zload("/home/weiwu/share/optimize/x0.pkl")
x1 = gftIO.zload("/home/weiwu/share/optimize/x1.pkl")
x2 = gftIO.zload("/home/weiwu/share/optimize/x2.pkl")
x3 = gftIO.zload("/home/weiwu/share/optimize/x3.pkl")
x4 = gftIO.zload("/home/weiwu/share/optimize/x4.pkl")
x5 = gftIO.zload("/home/weiwu/share/optimize/x5.pkl")
x6 = gftIO.zload("/home/weiwu/share/optimize/x6.pkl")
x7 = gftIO.zload("/home/weiwu/share/optimize//x7.pkl")
x8 = gftIO.zload("/home/weiwu/share/optimize//x8.pkl")
x9 = gftIO.zload("/home/weiwu/share/optimize//x9.pkl")
x10 = gftIO.zload("/home/weiwu/share/optimize/exposure_constraint.pkl")

target_mode = x0
position_limit = x1
risk_model = x2
asset_return = x3
asset_weight = x4
target_risk = x5
target_return = x6
target_date = x7
asset_constraint = x8
group_constraint = x9
exposure_constraint = x10

asset_return = x3.asMatrix()
asset_weight = x4.asColumnTab()
#asset_weight.date = target_date
if isinstance(x10, gftIO.GftTable):
    exposure_constraint = x10.asColumnTab()
    exposure_constraint = exposure_constraint.pivot(index='idname',
                                                    columns='variable',
                                                    values='value')
# position_limit = 58
if isinstance(asset_constraint, gftIO.GftTable):
    asset_constraint = asset_constraint.asMatrix()
if isinstance(group_constraint, gftIO.GftTable):
    group_constraint = group_constraint.asMatrix()

logger.debug('parse data finished!')
logger.debug('asset return number: %s', asset_return.shape[1])
logger.debug('asset weight number: %s', asset_weight.shape[0])
logger.debug('parse data finished.')

RiskModel = RiskAnlysis(risk_model)

target_date = exposure_constraint.index[0]
specific_risk = RiskModel.specific_risk()
df_industries_asset_weight = asset_weight.drop_duplicates(
    subset=['date', 'symbol'])

# drop incomplete rows
df_industries_asset_weight = df_industries_asset_weight.dropna(
    axis=0, subset=['industry', 'symbol'], how='any')

unique_symbol = df_industries_asset_weight['symbol'].unique()

try:
    if np.setdiff1d(unique_symbol, specific_risk.loc[target_date].index):
        pass
except ValueError:
    logger.debug('some symbols are not in asset weight symbols.')

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

asset_expected_return = asset_return.loc[:target_date, idx_level_1_value].fillna(0)
delta = RiskModel.delta(target_date, idx_level_1_value)
all_factors = RiskModel.ls_factors
big_X = RiskModel.get_factor_exposure(all_factors, target_date, unique_symbol)






# # find the nearest date next to target date from specific risk
# dt_next_to_target = specific_risk.index.searchsorted(target_date)
# dt_next_to_target = specific_risk.index[dt_next_to_target]
# target_specific_risk = specific_risk.loc[dt_next_to_target, :]
# logger.debug('target date: %s', target_date)
# logger.debug('next to target date: %s', dt_next_to_target)
# # drop duplicated rows at date
# df_industries_asset_weight = asset_weights.drop_duplicates(
#     subset=['date', 'symbol'])
# try:
#     df_industries_asset_init_weight = df_industries_asset_weight[
#         df_industries_asset_weight['date'] == target_date].dropna()
# except KeyError:
#     raise KeyError('invalid input date: %s' % target_date)

# # drop incomplete rows
# df_industries_asset_init_weight = df_industries_asset_init_weight.dropna(
#     axis=0, subset=['industry', 'symbol'], how='any')

# unique_symbol = df_industries_asset_init_weight['symbol'].unique()
# target_symbols = target_specific_risk.index.intersection(unique_symbol)
# if position_limit > len(target_symbols):
#     logger.debug("position limit is bigger than total symbols.")
#     position_limit = len(target_symbols)

# df_industries_asset_target_init_weight = df_industries_asset_init_weight.\
#                                          loc[df_industries_asset_init_weight['symbol'].isin(target_symbols)]
# df_pivot_industries_asset_weights = pd.pivot_table(
#     df_industries_asset_target_init_weight, values='value', index=['date'],
#     columns=['industry', 'symbol'])
# df_pivot_industries_asset_weights = df_pivot_industries_asset_weights.fillna(0)
# logger.debug("set OOTV to hierachical index dataframe.")
# noa = len(target_symbols)
# if noa < 1:
#     raise ValueError("no intersected symbols from specific risk and initial holding.")
# logger.debug("number of asset: %s", noa)
# # get the ordered column list
# idx_level_0_value = df_pivot_industries_asset_weights.columns.get_level_values(0)
# idx_level_0_value = idx_level_0_value.drop_duplicates()
# idx_level_1_value = df_pivot_industries_asset_weights.columns.get_level_values(1)
# asset_return = asset_return.loc[:target_date, idx_level_1_value].fillna(0)

# diag = specific_risk.loc[dt_next_to_target, idx_level_1_value]
# delta = pd.DataFrame(np.diag(diag), index=diag.index,
#                      columns=diag.index).fillna(0)

# big_X = get_factor_exposure(risk_model, ls_factor, target_date,
#                             idx_level_1_value)
# big_X = big_X.fillna(0)
# all_factors = big_X.columns

# covariance_matrix = risk_model['ret_cov'].set_index('date')

# cov_matrix = covariance_matrix.loc[dt_next_to_target]
# cov_matrix = cov_matrix.pivot(index='factorid1', columns='factorid2',
#                               values='value')
# cov_matrix = cov_matrix.reindex(all_factors, all_factors, fill_value=np.nan)

# rets_mean = logrels(asset_return).mean()
# avg_ret = matrix(rets_mean.values)

# # set boundary vector for h
# df_asset_weight = pd.DataFrame({'lower': [0.0], 'upper': [1.0]},
#                                index=idx_level_1_value)
# df_group_weight = pd.DataFrame({'lower': [0.0], 'upper': [1.0]},
#                                index=idx_level_0_value)

# def set_upper_limit(x, multiplier=0.1):
#     if x >= 0:
#         return x*(1 + multiplier)
#     else:
#         return x*(1 - multiplier)


# def set_lower_limit(x, multiplier=0.1):
#     if x >= 0:
#         return x*(1 - multiplier)
#     else:
#         return x*(1 + multiplier)


# df_factor_exposure_bound = pd.DataFrame(index=exposure_constraint.columns, columns=[['lower', 'upper']])
# df_factor_exposure_bound.lower = exposure_constraint.ix[-1].apply(lambda x: set_lower_limit(x))
# df_factor_exposure_bound.upper = exposure_constraint.ix[-1].apply(lambda x: set_upper_limit(x))

# df_factor_exposure_lower_bnd = pd.DataFrame(data=[[big_X.values.min()]]*len(all_factors), index=big_X.columns)
# df_factor_exposure_upper_bnd = pd.DataFrame(data=[[big_X.values.max()]]*len(all_factors), index=big_X.columns)

# df_factor_exposure_lower_bnd.ix[df_factor_exposure_bound.index] = df_factor_exposure_bound.lower.values.reshape((len(df_factor_exposure_bound),1))
# df_factor_exposure_upper_bnd.ix[df_factor_exposure_bound.index] = df_factor_exposure_bound.upper.values.reshape((len(df_factor_exposure_bound),1))


# target_mode = 2
# # Factor model portfolio optimization.
# w = cvx.Variable(noa)
# f = big_X.T.values*w
# # f = big_X.loc[:,exposure_constraint.columns].T.values*w
# gamma = cvx.Parameter(sign='positive')
# Lmax = cvx.Parameter()
# ret = w.T * rets_mean.values
# # risk = cvx.quad_form(f, cov_matrix.loc[exposure_constraint.columns,exposure_constraint.columns].values) + cvx.quad_form(w, delta.values)
# risk = cvx.quad_form(f, cov_matrix.values) + cvx.quad_form(w, delta.values)
# #ret = mu.T*w
# # for group weight constraint
# groups = df_pivot_industries_asset_weights.groupby(
#     axis=1, level=0, sort=False, group_keys=False).count().\
#     iloc[-1, :].values
# num_group = len(groups)
# num_asset = np.sum(groups)
# logger.debug('number of assets in groups: %s', groups)
# logger.debug('number of groups: %s', num_group)

# G_sparse_list = []
# for i in range(num_group):
#     for j in range(groups[i]):
#         G_sparse_list.append(i)
# Group_sub = spmatrix(1.0, G_sparse_list, range(num_asset))

# G = np.array(Group_sub)
# G_sum = np.array(matrix(Group_sub))*w

# # f == exposure_constraint.ix[-1].values.reshape(-1, 1)

# eq_constraint = [cvx.sum_entries(w) == 1,
#                  cvx.norm(w, 1) <= Lmax]
# l_eq_constraint = [w >= df_asset_weight.lower.values,
#                    w <= df_asset_weight.upper.values,
#                    G_sum >= df_group_weight.lower.values,
#                    G_sum <= df_group_weight.upper.values]
# if exposure_constraint is not None:
#     l_eq_constraint.append(f >= df_factor_exposure_lower_bnd.values)
#     l_eq_constraint.append(f <= df_factor_exposure_upper_bnd.values)
# target_return = -0.0006992348944336906
# Lmax.value = 1
# gamma.value = 1
# if target_mode == gsConst.Const.MinimumRisk:
#     # Solve the factor model problem.
#     prob_factor = cvx.Problem(cvx.Maximize(-gamma*risk),
#                               l_eq_constraint)
# if target_mode == gsConst.Const.MinimumRiskUnderReturn:
#     # minimum risk subject to target return, Markowitz Mean_Variance Portfolio
#     prob_factor = cvx.Problem(cvx.Maximize(-gamma*risk),
#                               [ret >= target_return]+l_eq_constraint)
# if target_mode == gsConst.Const.MaximumReturnUnderRisk:
#     # Portfolio optimization with a leverage limit and a bound on risk
#     prob_factor = cvx.Problem(cvx.Maximize(ret),
#                               [risk <= target_risk]+l_eq_constraint)
# prob_factor.solve(verbose=True)
# df_opts_weight = pd.DataFrame(np.array(w.value).T,
#                               columns=idx_level_1_value,
#                               index=[target_date])
# logger.debug(prob_factor.status)
# logger.debug("target return: %s", target_return)
# logger.debug("all weight are bigger than 0? %s",
#              (df_opts_weight > 0).all().all())
# logger.debug("all weight are smaller than 1? %s",
#              (df_opts_weight <= 1).all().all())
# logger.debug("weight sum smaller than 0: %s",
#              df_opts_weight[df_opts_weight < 0].sum(1))
# logger.debug(df_opts_weight)
# handler.close()
# logger.removeHandler(handler)
# while logger.handlers:
#      logger.handlers.pop()
