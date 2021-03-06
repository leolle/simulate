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

from cvxopt import matrix, solvers, spmatrix, sparse
from cvxopt.blas import dot

from lib.gftTools import gsConst, gftIO


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


def check_constraint_issue(mx_P, mx_q, mx_G, mx_h, mx_A, mx_b,
                           mx_asset_sub, mx_group_sub, mx_exp_sub,
                           mx_asset_bnd, mx_group_bnd, mx_exp_bnd):
    ''' check which constraint fails.

    Parameters
    ----------
    mx_G: 1xn matrix
        matrix only includes returns.
    mx_h: 1x1 matrix
        target return matrix.
    mx_asset_sub: 2nxn subsets matrix, n=asset number
        2 nx1 identity matrices.
    mx_group_sub: 2axn subsets matrix, a=group number
        2 axn spmatrices.
    mx_exp_sub: 2bxn subsets matrix, b=exposure factors number
        2bxn identity matrices.
    mx_asset_bnd: 2nx1 matrix, n=asset number
        asset weight constraint matrix
    mx_group_bnd: 2ax1 matrix, a=group number
        group weight constraint matrix
    mx_exp_bnd: 2bx1 matrix, b=exposure factors number
        factor exposure constraint matrix

    Returns
    -------
    None
    '''
    import itertools
    G = matrix(sparse([mx_G, mx_asset_sub]))
    h = matrix(sparse([mx_h, mx_asset_bnd]))

    boundary_sub = [mx_group_sub, mx_exp_sub]
    limit = [mx_group_bnd, mx_exp_bnd]
    error = ('group ', 'exposure ')
    stuff = [1, 2]
    for L in range(0, len(stuff)+1):
        for subset in itertools.combinations(stuff, L):
            if len(subset) == 0:
                try:
                    # G_pos = matrix(sparse([G, matrix(-np.eye(n), tc='d')]))
                    # h_pos = matrix(sparse([h, matrix(np.zeros((n, 1)))]))
                    sol = solvers.qp(mx_P, mx_q, G, h, mx_A, mx_b)
                    if sol['x'] == 'unknown':
                        raise('failed to get optimal value on\
                        position limit constraint')
                except ValueError as e:
                    raise ConstraintError('ERROR on solving position limit\
                    constraint only')

            if len(subset) > 0:
                ls = [x-1 for x in list(subset)]
                g_matrix = []
                h_matrix = []
                g_matrix.append(G)
                h_matrix.append(h)
                for i in ls:
                    g_matrix.append(boundary_sub[i])
                    h_matrix.append(limit[i])

                G_val = matrix(sparse(g_matrix))
                h_val = matrix(sparse(h_matrix))

                try:
                    sol = solvers.qp(mx_P, mx_q, G_val, h_val, mx_A, mx_b)
                    if sol['x'] == 'unknown':
                        print('failed to get optimal value on %s', [error[i] for i in ls])
                except ValueError as e:
                    raise ConstraintError('ERROR on solving combination %s, %s' % ([error[i] for i in ls], e))

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


solvers.options['show_progress'] = True

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
x10 = gftIO.zload("/home/weiwu/share/optimize//x10.pkl")

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
asset_weights = x4.asColumnTab()
target_date = pd.to_datetime(target_date)
asset_weights.date = target_date

position_limit = 200
if asset_constraint is not None:
    asset_constraint = asset_constraint.asMatrix()
if group_constraint is not None:
    group_constraint = group_constraint.asMatrix()

exposure_constraint = pd.DataFrame(data=[[0.237147, 0.215739, 1.737035,
                                         -0.043088, -0.136166, 0.165721,
                                         0.080216, -0.163873]],
                                   columns=[b'\x06\x92"\tM\xf9<\x87$\xdc(#\xc3\xacu\xd6',
                                            b'U/X\xa7d\x8f\x1eD\xf1\x85b!*\xe4\xcb;',
                                            b'\x84\x9e\x03_c\xda\xcc\xb9\\\xbc\x08\xdbd\xf9\x89\xd4',
                                            b'\xbaH\x12\xb8\xfa,\xb0\xb1\xd2\x9bv`\xe7Tw\xa8',
                                            b'\xc1\x8b\xc3\x13\xc3S\x85\xeb\xba"\x86y-\x97\xe1\x80',
                                            b'\xebI\xcb\xc0\xa75>\xbe\xe7ceV\x1d\xb4\x1c\x9e',
                                            b'\xf4&\xe31/\x91\xec\xab\xfa\xc8\x87\xf7zP\xfc\xc1',
                                            b'\xf9 \xc4\x15\xa9\x0e\xd7\xe8\x8cx@\x15\xb2[\x00\xb9'])


logger.debug('parse data finished!')
logger.debug('asset return number: %s', asset_return.shape[1])
logger.debug('asset weight number: %s', asset_weights.shape[0])
logger.debug('parse data finished.')
# regex to search all the factors
ls_factor = [x[:-4] for x in risk_model.keys() if re.search(".ret$", x)]
# ls_factor = [x.split('.')[0] for x in ls_factor]

specific_risk = risk_model['specificRisk'].pivot(
    index='date', columns='symbol', values='specificrisk')
# target_date = pd.datetime(year=2016, month=10, day=31)
# target_return = -0.00096377
# target_risk = 3.16026352e-06
#target_mode = 1

# find the nearest date next to target date from specific risk
dt_next_to_target = specific_risk.index.searchsorted(target_date)
dt_next_to_target = specific_risk.index[dt_next_to_target]
target_specific_risk = specific_risk.loc[dt_next_to_target, :]
logger.debug('target date: %s', target_date)
logger.debug('next to target date: %s', dt_next_to_target)
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
    logger.debug("position limit is bigger than total symbols.")
    position_limit = len(target_symbols)

solvers.options['show_progress'] = True
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
logger.debug("set OOTV to hierachical index dataframe.")
noa = len(target_symbols)
if noa < 1:
    raise ValueError("no intersected symbols from specific risk and initial holding.")
logger.debug("number of asset: %s", noa)
# get the ordered column list
idx_level_0_value = df_pivot_industries_asset_weights.columns.get_level_values(0)
idx_level_0_value = idx_level_0_value.drop_duplicates()
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


# P = matrix(cov_matrix_V.values)
# q = matrix(np.zeros((noa, 1)), tc='d')

# A = matrix(1.0, (1, noa))
# b = matrix(1.0)

# # for group weight constraint
# groups = df_pivot_industries_asset_weights.groupby(
#     axis=1, level=0, sort=False, group_keys=False).count().\
#     iloc[-1, :].values
# num_group = len(groups)
# num_asset = np.sum(groups)

# logger.debug('number of assets in groups: %s', groups)
# logger.debug('number of groups: %s', num_group)


# # set boundary vector for h
# df_asset_weight = pd.DataFrame({'lower': [0.0], 'upper': [1.0]},
#                                index=idx_level_1_value)
# df_group_weight = pd.DataFrame({'lower': [0.0], 'upper': [1.0]},
#                                index=set(idx_level_0_value))
# df_factor_exposure_bound = pd.DataFrame(index=exposure_constraint.columns, columns=[['lower', 'upper']])

# df_factor_exposure_bound.loc[exposure_constraint.ix[-1]>0, 'lower'] = exposure_constraint.ix[-1][exposure_constraint.ix[-1]>0]*(0.9)
# df_factor_exposure_bound.loc[exposure_constraint.ix[-1]>0, 'upper'] = exposure_constraint.ix[-1][exposure_constraint.ix[-1]>0]*(1.1)
# df_factor_exposure_bound.loc[exposure_constraint.ix[-1]<0, 'lower'] = exposure_constraint.ix[-1][exposure_constraint.ix[-1]<0]*(1.1)
# df_factor_exposure_bound.loc[exposure_constraint.ix[-1]<0, 'upper'] = exposure_constraint.ix[-1][exposure_constraint.ix[-1]<0]*(0.9)
# df_factor_exposure_bound = df_factor_exposure_bound.astype(np.double)
#df_factor_exposure_bound.lower = big_X.T.loc[exposure_constraint.columns,:].min(axis=1)
#df_factor_exposure_bound.upper = big_X.T.loc[exposure_constraint.columns,:].max(axis=1)



#if check_boundary_constraint(df_asset_weight, df_group_weight,
#                             df_factor_exposure_bound, big_X):
#    logger.debug("boundary setting is fine")
#
# df_asset_bnd_matrix = matrix(np.concatenate(((df_asset_weight.upper,
#                                               df_asset_weight.lower)), 0))
# df_group_bnd_matrix = matrix(np.concatenate(((df_group_weight.upper,
#                                               df_group_weight.lower)), 0))
# #df_factor_exposure_bnd_matrix = matrix(np.concatenate(((df_factor_exposure_bound.upper,
# #                                                        df_factor_exposure_bound.lower)), 0))
# #df_factor_exposure_bnd_matrix = matrix(df_factor_exposure_bound.upper.values*100.0)
# df_factor_exposure_bnd_matrix = matrix(df_factor_exposure_bound.lower.values)
#df_factor_exposure_bnd_matrix = matrix(df_factor_exposure_bound.iloc[0,0])

# Assuming AvgReturns as the expected returns if parameter is not specified
rets_mean = logrels(asset_return).mean()
avg_ret = matrix(rets_mean.values)
# G = matrix(-np.transpose(np.array(avg_ret)))
# h = matrix(-np.ones((1, 1))*target_return)
# G_sparse_list = []
# for i in range(num_group):
#     for j in range(groups[i]):
#         G_sparse_list.append(i)
# Group_sub = spmatrix(1.0, G_sparse_list, range(num_asset))

# Group_sub = matrix(sparse([Group_sub, -Group_sub]))

# asset_sub = matrix(np.eye(noa))
# asset_sub = matrix(sparse([asset_sub, -asset_sub]))
# #exp_sub = matrix(np.array(big_X.T.loc[df_factor_exposure_bound.index, idx_level_1_value]))
# exp_sub = matrix(-np.array(big_X.T.loc[df_factor_exposure_bound.index, idx_level_1_value]))
#exp_sub = matrix(sparse([exp_sub, -exp_sub]))


# if exposure_constraint is not None:
#     G0 = matrix(sparse([ exp_sub]))
#     h0 = matrix(sparse([ df_factor_exposure_bnd_matrix]))
# else:
#     G0 = matrix(sparse([asset_sub, Group_sub]))
#     h0 = matrix(sparse([df_asset_bnd_matrix, df_group_bnd_matrix]))
# G0 = matrix(sparse([asset_sub, Group_sub]))
# h0 = matrix(sparse([df_asset_bnd_matrix, df_group_bnd_matrix]))

# sol = solvers.qp(P, q, G0, h0, A, b)
# df_opts_weight = pd.DataFrame(np.array(sol['x']).T,
#                               columns=idx_level_1_value,
#                               index=[target_date])
# logger.debug('solution is %s ', sol['status'])
# logger.debug("target return: %s", target_return)
# logger.debug("all weight are bigger than 0? %s", (df_opts_weight>0).all().all())
# logger.debug("all weight are smaller than 1? %s", (df_opts_weight<=1).all().all())
# logger.debug("weight sum smaller than 0: %s", df_opts_weight[df_opts_weight<0].sum(1))

# df_opts_weight

# Long only portfolio optimization.
import cvxpy as cvx
w = cvx.Variable(noa)
gamma = cvx.Parameter(sign='positive')
ret = w.T*rets_mean.values 
risk = cvx.quad_form(w, cov_matrix_V.values)
prob = cvx.Problem(cvx.Maximize(ret - gamma*risk), 
               [cvx.sum_entries(w) == 1,
                w >= 0])

SAMPLES = 100
risk_data = np.zeros(SAMPLES)
ret_data = np.zeros(SAMPLES)
gamma_vals = np.logspace(-2, 3, num=SAMPLES)

for i in range(SAMPLES):
    gamma.value = gamma_vals[i]
    prob.solve()
    risk_data[i] = cvx.sqrt(risk).value
    ret_data[i] = ret.value

# Plot long only trade-off curve.
import matplotlib.pyplot as plt
# #%matplotlib inline
# #%config InlineBackend.figure_format = 'svg'

markers_on = [29, 40]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(risk_data, ret_data, 'g-')
for marker in markers_on:
    plt.plot(risk_data[marker], ret_data[marker], 'bs')
    ax.annotate(r"$\gamma = %.2f$" % gamma_vals[marker], xy=(risk_data[marker]+.08, ret_data[marker]-.03))
for i in range(noa):
    plt.plot(cvx.sqrt(cov_matrix_V.iloc[i,i]).value, rets_mean[i], 'ro')
plt.xlabel('Standard deviation')
plt.ylabel('Return')
plt.show()
