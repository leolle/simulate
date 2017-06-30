# -*- coding: utf-8 -*-
### Portfolio Optiimization
# Finds an optimal allocation of stocks in a portfolio,
# satisfying a minimum expected return.
# The problem is posed as a Quadratic Program, and solved
# using the cvxopt library.
# Uses actual past stock data, obtained using the stocks module.

import sys
import itertools
from cvxopt import matrix, solvers, spmatrix, sparse
from cvxopt.blas import dot
import numpy
import pandas as pd
import numpy as np
from datetime import datetime

solvers.options['show_progress'] = False

import logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


# solves the QP, where x is the allocation of the portfolio:
# minimize   x'Px + q'x
# subject to Gx <= h
#            Ax == b
#
# Input:  n       - # of assets
#         avg_ret - nx1 matrix of average returns
#         covs    - nxn matrix of return covariance
#         r_min   - the minimum expected return that you'd
#                   like to achieve
# Output: sol - cvxopt solution object
""" 模拟输入信息 """
dates = pd.date_range('2000-01-01', periods=6)
industry = ['industry', 'industry', 'utility', 'utility', 'consumer']
symbols = ['A', 'B', 'C', 'D', 'E']
zipped = list(zip(industry, symbols))
index = pd.MultiIndex.from_tuples(zipped)

noa = len(symbols)

data = np.array([[10, 11, 12, 13, 14, 10],
                 [10, 11, 10, 13, 14, 9],
                 [10, 10, 12, 13, 9, 11],
                 [10, 11, 12, 13, 14, 8],
                 [10, 9, 12, 13, 14, 9]])

market_to_market_price = pd.DataFrame(data.T, index=dates, columns=index)
rets = market_to_market_price / market_to_market_price.shift(1) - 1.0
rets = rets.dropna(axis=0, how='all')

# covariance of asset returns
covs    = matrix(rets.cov().values)

# average yearly return for each stock
rets_mean = rets.mean()
avg_ret = matrix(rets_mean.values)
n = len(symbols)


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
    prisk : float
        expected portfolio total risk
    pret / pvol : float
        Sharpe ratio for rf=0
    '''
    weights = np.array(weights)
    pret = np.dot(weights.T, rets_mean)
    pvol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
    # prisk = calculate_total_risk(weights, cov_matrix_V)

    return np.array([pret, pvol, pret / pvol])


def minimum_risk_subject_to_target_return():

    # minimum expected return threshold
    r_min = 0.04

    P = covs
    q = matrix(numpy.zeros((n, 1)), tc='d')
    # inequality constraints Gx <= h
    # captures the constraints (avg_ret'x >= r_min) and (x >= 0)
    G = matrix(-numpy.transpose(numpy.array(avg_ret)))
    h = matrix(-numpy.ones((1,1))*r_min)

    # equality constraint Ax = b; captures the constraint sum(x) == 1
    A = matrix(1.0, (1,n))
    b = matrix(1.0)

    groups = rets.groupby(axis=1, level=0, sort=False, group_keys=False).count().ix[-1].values
    num_group = len(groups)
    num_asset = np.sum(groups)
    G_sparse_list = []
    for i in range(num_group):
        for j in range(groups[i]):
            G_sparse_list.append(i)
    Group_sub = spmatrix(1.0, G_sparse_list, range(num_asset))
    asset_sub = matrix(np.eye(n))

    # asset_sub is for asset weight limit, Group sub is for group weight
    # constraint.
    G = matrix(sparse([G, asset_sub, -asset_sub, Group_sub, -Group_sub]))

    b_asset = tuple((0.01, 1.0) for i in rets.columns)
    b_asset_upper_bound = np.array([x[-1] for x in b_asset])
    b_asset_lower_bound = np.array([x[0] for x in b_asset])
    b_asset_matrix = matrix(numpy.concatenate((b_asset_upper_bound,
                                               -b_asset_lower_bound), 0))
    b_group = [(.05,.41), (.2,.66), (0.01,.16)]
    b_group_upper_bound = np.array([x[-1] for x in b_group])
    b_group_lower_bound = np.array([x[0] for x in b_group])
    b_group_matrix = matrix(numpy.concatenate((b_group_upper_bound,
                                               -b_group_lower_bound), 0))
    h = matrix(sparse([h, b_asset_matrix, b_group_matrix]))

    # solve minimum risk for maximum return above target .
    sol = solvers.qp(P, q, G, h, A, b)

    print(minimum_risk_subject_to_target_return.__name__)
    print(sol['x'])
    print(statistics(sol['x']))


def maximum_return_subject_to_target_risk():
    N = 100
    mus = [10**(5.0*t/N-1.0) for t in range(N)]

    P = covs
    q = avg_ret
    G = matrix(-np.eye(n))
    h = matrix(-np.zeros((n, 1)))

    # equality constraint Ax = b; captures the constraint sum(x) == 1
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    xs = [solvers.qp(mu*covs, q, G, h, A, b)['x'] for mu in mus]
    returns = [dot(-q.T, x) for x in xs]
    risks = [np.sqrt(dot(x.T, covs*x)) for x in xs]
    try: import pylab
    except ImportError: pass
    else:
        pylab.figure(1, facecolor='w')
        pylab.plot(risks, returns)
        pylab.xlabel('standard deviation')
        pylab.ylabel('expected return')
        pylab.axis([0, 0.2, 0, 0.15])
        pylab.title('Risk-return trade-off curve')
        pylab.yticks([0.00, 0.05, 0.10, 0.15])
        #pylab.show()
    sol = solvers.qp(P, q, G, h, A, b)

    print(maximum_return_subject_to_target_risk.__name__)
    print(sol['x'])
    print(statistics(sol['x']))


def maximum_return():

    P = matrix(np.zeros((n, n)))
    q = -avg_ret
    G = matrix(-np.eye(n))
    h = matrix(-numpy.zeros((n, 1)))

    # equality constraint Ax = b; captures the constraint sum(x) == 1
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    sol = solvers.qp(P, q, G, h, A, b)

    print(maximum_return.__name__)
    print(sol['x'])
    print(statistics(sol['x']))


def minimum_risk():

    P = covs
    q = matrix(numpy.zeros((n, 1)), tc='d')
    G = matrix(-np.eye(n))
    h = matrix(-numpy.zeros((n, 1)))

    # equality constraint Ax = b; captures the constraint sum(x) == 1
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    sol = solvers.qp(P, q, G, h, A, b)

    print(minimum_risk.__name__)
    print(sol['x'])
    print(statistics(sol['x']))


# solve for maximum return and under control risk.



#minimum_risk_subject_to_target_return()
#maximum_return()
#minimum_risk()
#maximum_return_subject_to_target_risk()
# N = 100
# mus = [10**(5.0*t/N-1.0) for t in range(N)]

# P = covs
# q = -avg_ret
# G = matrix(-np.eye(n))
# h = matrix(-numpy.zeros((n, 1)))

# # equality constraint Ax = b; captures the constraint sum(x) == 1
A = matrix(1.0, (1, n))
b = matrix(1.0)

# xs = [solvers.qp(mu*covs, q, G, h, A, b) for mu in mus]
# returns = [dot(avg_ret.T, x['x']) for x in xs]
# risks = [np.sqrt(dot(x['x'], covs*x['x'])) for x in xs]
# try: import pylab
# except ImportError: pass
# else:
#     pylab.figure(1, facecolor='w')
#     pylab.plot(risks, returns)
#     pylab.xlabel('standard deviation')
#     pylab.ylabel('expected return')
#     pylab.axis([0, 0.2, 0, 0.15])
#     pylab.title('Risk-return trade-off curve')
#     pylab.yticks([0.00, 0.05, 0.10, 0.15])
#    # pylab.show()

P = covs
q = matrix(numpy.zeros((n, 1)), tc='d')

groups = market_to_market_price.groupby(axis=1, level=0, sort=False,
                                        group_keys=False).\
                                        count().iloc[-1,:].values
num_group = len(groups)
num_asset = np.sum(groups)

asset_sub = matrix(np.eye(n))
asset_sub = matrix(sparse([asset_sub, -asset_sub]))
target_return = 0.010049062049062037
G = matrix(-np.transpose((rets.mean())), (1, n))
h = matrix(-np.ones((1, 1))*target_return)
G_sparse_list = []
for i in range(num_group):
	for j in range(groups[i]):
		G_sparse_list.append(i)
Group_sub = spmatrix(1.0, G_sparse_list, range(num_asset))
Group_sub = matrix(sparse([Group_sub, -Group_sub]))
exp_sub = matrix(np.array(covs.T))
exp_sub = matrix(sparse([exp_sub, - exp_sub]))
#G = matrix(sparse([G, asset_sub, -asset_sub]))
#G = matrix(sparse([G, asset_sub, -asset_sub, Group_sub, -Group_sub]))
#G = matrix(sparse([G, asset_sub, -asset_sub, Group_sub, -Group_sub, exp_sub, -exp_sub]))
#G = matrix(sparse([G, exp_sub, -exp_sub]))


b_asset = [(0.001, 0.55)] * rets.shape[1]
b_group = [(0.001, 0.4)] * num_group

b_asset_upper_bound = np.array([x[-1] for x in b_asset])
b_asset_lower_bound = np.array([x[0] for x in b_asset])
b_asset_matrix = matrix(np.concatenate((b_asset_upper_bound,
                                        -b_asset_lower_bound), 0))
b_group_upper_bound = np.array([x[-1] for x in b_group])
b_group_lower_bound = np.array([x[0] for x in b_group])
b_group_matrix = matrix(np.concatenate((b_group_upper_bound,
                                        -b_group_lower_bound), 0))
b_factor_exposure = list(zip((np.array(covs*(1.0/noa))).min(axis=1)*0.1, (np.array(covs*(1.0/noa))).max(axis=1)*10))
b_factor_exposure_upper_bound = np.array([x[-1] for x in b_factor_exposure])
b_factor_exposure_lower_bound = np.array([x[0] for x in b_factor_exposure])
b_factor_exposure_matrix = matrix(np.concatenate(
    (b_factor_exposure_upper_bound, -b_factor_exposure_lower_bound), 0))

#h = matrix(sparse([h, b_asset_matrix]))
#h = matrix(sparse([h, b_asset_matrix, b_group_matrix]))
#h = matrix(sparse([h, b_asset_matrix, b_group_matrix, b_factor_exposure_matrix]))
#h = matrix(sparse([h, b_factor_exposure_matrix]))
idx_level_0_value = market_to_market_price.columns.get_level_values(0)
idx_level_1_value = market_to_market_price.columns.get_level_values(1)
df_asset_weight = pd.DataFrame({'lower': [0.01], 'upper': [0.25]},
                               index=symbols)

df_group_weight = pd.DataFrame({'lower': [0.01], 'upper': [0.4]},
                               index=set(idx_level_0_value))

b_asset_matrix = matrix(np.concatenate(((df_asset_weight.upper, df_asset_weight.lower)), 0))
b_group_matrix = matrix(np.concatenate(((df_group_weight.upper, df_group_weight.lower)), 0))
#b_factor_exposure_matrix = 

# df_group_weight['tactical'] = [(.05,.41), (.2,.66), (0,.16)]




#sol = solvers.qp(P, q, G, h, A, b)
G = matrix(-np.transpose((rets.mean())), (1, n))
h = matrix(-np.ones((1, 1))*target_return)



class ConstraintError(Exception):
    pass


# G_pos = matrix(sparse([G, matrix(-np.eye(n), tc='d')]))
# h_pos = matrix(sparse([h, matrix(0.0, (n, 1))]))

G = matrix(sparse([G, asset_sub]))
h = matrix(sparse([h, b_asset_matrix]))


boundary_sub = [Group_sub, exp_sub]
limit = [b_group_matrix, b_factor_exposure_matrix]
error = ('group ', 'exposure ')
# stuff = [1, 2]
# for L in range(0, len(stuff)+1):
#     for subset in itertools.combinations(stuff, L):
#         if len(subset) == 0:
#             try:
#                 # G_pos = matrix(sparse([G, matrix(-np.eye(n), tc='d')]))
#                 # h_pos = matrix(sparse([h, matrix(np.zeros((n, 1)))]))
#                 sol = solvers.qp(P, q, G, h, A, b)
#                 if sol['x'] == 'unknown':
#                     print('failed to get optimal value on position limit constraint')
#             except ValueError as e:
#                 raise ConstraintError('ERROR on solving position limit constraint only')

#         if len(subset) > 0:
#             ls = [x-1 for x in list(subset)]
#             g_matrix = []
#             h_matrix = []
#             g_matrix.append(G)
#             h_matrix.append(h)
#             for i in ls:
#                 g_matrix.append(boundary_sub[i])
#                 h_matrix.append(limit[i])

#             G_val = matrix(sparse(g_matrix))
#             h_val = matrix(sparse(h_matrix))

#             try:
#                 sol = solvers.qp(P, q, G_val, h_val, A, b)
#                 if sol['x'] == 'unknown':
#                     print('failed to get optimal value on %s', [error[i] for i in ls])
#             except ValueError as e:
#                 raise ConstraintError('ERROR on solving combination %s, %s' % ([error[i] for i in ls], e))
# if sol['status'] == 'optimal':
#     print(sol['x'])
# else:
#     print(sol['status'])

# ##################################################################
# # Calculate theoretical minimum and maximum theoretical returns
# from copy import deepcopy
# f_min = 0
# f_max = 0

# rets = deepcopy(rets)

# na_expected = np.average(rets, axis=0)

# na_signs = np.sign(na_expected)
# indices = np.where(na_signs == 0)
# na_signs[indices] = 1
# na_signs = np.ones(len(na_signs))

# rets = na_signs*rets
# na_expected = na_signs*na_expected

# na_sort_ind = na_expected.argsort()

# # First add the lower bounds on portfolio participation
# for i, fRet in enumerate(na_expected):
#     f_min = f_min + fRet*df_asset_weight.lower[i]
#     f_max = f_max + fRet*df_asset_weight.lower[i]


# # Now calculate minimum returns"""
# # allocate the max possible in worst performing equities """
# # Subtract min since we have already counted it """
# na_upper_add = df_asset_weight.upper - df_asset_weight.lower
# f_total_weight = np.sum(df_asset_weight.lower)

# for i, ls_ind in enumerate(na_sort_ind):
#     f_ret_add = na_upper_add[ls_ind] * na_expected[ls_ind]
#     f_total_weight = f_total_weight + na_upper_add[ls_ind]
#     f_min = f_min + f_ret_add
#     # Check if this additional percent puts us over the limit """
#     if f_total_weight > 1.0:
#         f_min = f_min - na_expected[ls_ind] * (f_total_weight - 1.0)
#         break
# else:
#     raise ValueError("sum of total asset maximum weight is less than 1 ")
# # Repeat for max, just reverse the sort, i.e. high to low """
# na_upper_add = df_asset_weight.upper - df_asset_weight.lower
# f_total_weight = np.sum(df_asset_weight.lower)
# if f_total_weight > 1:
#     raise ValueError("sum of total asset minimum weight is bigger than 1 ")
# for i, ls_ind in enumerate(na_sort_ind[::-1]):
#     f_ret_add = na_upper_add[ls_ind] * na_expected[ls_ind]
#     f_total_weight = f_total_weight + na_upper_add[ls_ind]
#     f_max = f_max + f_ret_add

#     # Check if this additional percent puts us over the limit """
#     if f_total_weight > 1.0:
#         f_max = f_max - na_expected[ls_ind] * (f_total_weight - 1.0)
#         break

# print("max: ", f_max)
# print("min: ", f_min)


####################################################################

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

    na_expected = np.average(rets, axis=0)

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


unsorted_level_0_value = idx_level_0_value.drop_duplicates()
group_rets_min = rets.mean().sort_values(ascending=True).groupby(level=0).head(1).ix[unsorted_level_0_value].index
group_rets_max = rets.mean().sort_values(ascending=False).groupby(level=0).head(1).ix[unsorted_level_0_value].index


f_g_min = get_ret_range(rets.loc[:, group_rets_min], df_group_weight)[0]
f_g_max = get_ret_range(rets.loc[:, group_rets_max], df_group_weight)[1]

G = matrix(-np.transpose((rets.mean())), (1, n))

(f_min, f_max) = get_ret_range(rets, df_asset_weight)
f_step = (f_max - f_min) / 100
ls_f_return = [f_min + x * f_step for x in range(101)]
ls_f_risk = []
ls_portfolio = []
for f_target_return in ls_f_return:
    h = matrix(-np.ones((1, 1))*f_target_return)
    G_sr = matrix(sparse([G, asset_sub, Group_sub]))
    h_sr = matrix(sparse([h, b_asset_matrix, b_group_matrix]))
    try:
        sol = solvers.qp(P, q, G_sr, h_sr, A, b)
    except:
        ls_f_risk.append(np.nan)
        ls_portfolio.append(None)
        continue
    ls_f_risk.append(statistics(sol['x'])[1])
    df_opts_weight = pd.DataFrame(np.array(sol['x']).T,
                                  columns=symbols)
    ls_portfolio.append(df_opts_weight)

def find_nearest(array, value):
    """ Find the nearest value index from an array"""
    if isinstance(array, list):
        array = np.array(array)
    idx = (np.abs(array-value)).argmin()
    return idx

#target_risk = find_nearest(ls_f_risk, 0.13492336)
# target_risk = 0.13492336

# f_return = ls_f_return[ls_f_risk.index(min(ls_f_risk))]
# ls_f_return_new = np.array(ls_f_return)
# ls_f_risk_new = np.array(ls_f_risk)
# ls_f_risk_new = ls_f_risk_new[ls_f_risk <= target_risk]
# ls_f_return_new = ls_f_return_new[ls_f_risk <= target_risk]
# na_sharpe_ratio = ls_f_return_new / ls_f_risk_new
# i_index_max_sharpe = np.where(na_sharpe_ratio == max(na_sharpe_ratio))
# i_index_max_sharpe = i_index_max_sharpe[0]
# f_target_return = ls_f_return_new[i_index_max_sharpe]
# h = matrix(-np.ones((1, 1))*f_target_return)

# G_sr = matrix(sparse([G, asset_sub, Group_sub]))
# h_sr = matrix(sparse([h, b_asset_matrix, b_group_matrix]))
# sol = solvers.qp(P, q, G_sr, h_sr, A, b)
# df_opts_weight = pd.DataFrame(np.array(sol['x']).T,
#                               columns=symbols)

logger.debug("all weight are bigger than 0? %s", (df_opts_weight>0).all().all())
logger.debug("all weight are smaller than 1? %s", (df_opts_weight<=1).all().all())
logger.debug("weight sum smaller than 0: %s", df_opts_weight[df_opts_weight<0].sum(1))

