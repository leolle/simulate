# -*- coding: utf-8 -*-
### Portfolio Optiimization
# Finds an optimal allocation of stocks in a portfolio,
# satisfying a minimum expected return.
# The problem is posed as a Quadratic Program, and solved
# using the cvxopt library.
# Uses actual past stock data, obtained using the stocks module.

from cvxopt import matrix, solvers, spmatrix, sparse
from cvxopt.blas import dot
import numpy
import pandas as pd
import numpy as np
from datetime import datetime

solvers.options['show_progress'] = False
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


def check_boundary_constraint(asset_lower_bound, asset_upper_bound,
                              group_lower_bound, group_upper_bound):
    ''' check input boundary limit.

    Parameters
    ----------
    asset_lower_bound : array-like
        Input lower boundary array for each asset.

    asset_upper_bound : array-like
        Input upper boundary array for each asset.

    group_lower_bound : array-like
        Input lower boundary array for each group.

    group_upper_bound : array-like
        Input upper boundary array for each group.

    Returns
    -------
    True: all boundaries in condition.
    False: any boundaries out of condition.
    '''
    if ((asset_lower_bound) < 0).any():
        raise ValueError('short is not supported.')
    if ((asset_upper_bound) > 1).any():
        raise ValueError('asset upper boundary is bigger than 1.')
    if (np.sum(asset_lower_bound) > 1):
        raise ValueError('asset lower boundary sum is bigger than 1.')
    if (np.sum(asset_upper_bound) < 1):
        raise ValueError('asset upper boundary sum is smaller than 1.')
    if ((asset_lower_bound > asset_upper_bound).any()):
        raise ValueError('asset lower boundary is bigger than upper boundary')

    if ((group_lower_bound) < 0).any():
        raise ValueError('short is not supported.')
    if ((group_upper_bound) > 1).any():
        raise ValueError('group upper boundary is bigger than 1.')
    if (np.sum(group_lower_bound) > 1):
        raise ValueError('group lower boundary sum is bigger than 1.')
    if (np.sum(group_upper_bound) < 1):
        raise ValueError('group upper boundary sum is smaller than 1.')
    if ((group_lower_bound > group_upper_bound).any()):
        raise ValueError('group lower boundary is bigger than upper boundary')

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

target_return = 0.010049062049062037
G = matrix(-np.transpose((rets.mean())), (1, n))
h = matrix(-np.ones((1, 1))*target_return)
G_sparse_list = []
for i in range(num_group):
	for j in range(groups[i]):
		G_sparse_list.append(i)
Group_sub = spmatrix(1.0, G_sparse_list, range(num_asset))

exp_sub = matrix(np.array(covs.T))
G = matrix(sparse([G, asset_sub, -asset_sub]))
#G = matrix(sparse([G, asset_sub, -asset_sub, Group_sub, -Group_sub]))
#G = matrix(sparse([G, asset_sub, -asset_sub, Group_sub, -Group_sub, exp_sub, -exp_sub]))
#G = matrix(sparse([G, exp_sub, -exp_sub]))


b_asset = [(0.0, .5)] * rets.shape[1]
b_group = [(0.0, 1.0)] * num_group

b_asset_upper_bound = np.array([x[-1] for x in b_asset])
b_asset_lower_bound = np.array([x[0] for x in b_asset])
b_asset_matrix = matrix(np.concatenate((b_asset_upper_bound,
                                        -b_asset_lower_bound), 0))
b_group_upper_bound = np.array([x[-1] for x in b_group])
b_group_lower_bound = np.array([x[0] for x in b_group])
b_group_matrix = matrix(np.concatenate((b_group_upper_bound,
                                        -b_group_lower_bound), 0))
b_factor_exposure = list(zip((np.array(covs*(1.0/noa))).min(axis=1)*0.9999, (np.array(covs*(1.0/noa))).max(axis=1)*1.0001))
b_factor_exposure_upper_bound = np.array([x[-1] for x in b_factor_exposure])
b_factor_exposure_lower_bound = np.array([x[0] for x in b_factor_exposure])
b_factor_exposure_matrix = matrix(np.concatenate(
    (b_factor_exposure_upper_bound, -b_factor_exposure_lower_bound), 0))

#h = matrix(sparse([h, b_asset_matrix]))
#h = matrix(sparse([h, b_asset_matrix, b_group_matrix]))
#h = matrix(sparse([h, b_asset_matrix, b_group_matrix, b_factor_exposure_matrix]))
#h = matrix(sparse([h, b_factor_exposure_matrix]))


#sol = solvers.qp(P, q, G, h, A, b)
G = matrix(-np.transpose((rets.mean())), (1, n))
h = matrix(-np.ones((1, 1))*target_return)
# print('G', G)
# print('h', h)
# if sol['status'] == 'optimal':
#     print('result is optimal')
#     print(sol['x'])
# elif sol['status'] == 'unknown':
#     print('the algorithm failed to find a solution that satisfies the specified tolerances')


class State:
    def run(self):
        assert 0, "run not implemented"

    def next(self, input):
        assert 0, "next not implemented"


class StateMachine:
    def __init__(self, initialState):
        self.currentState = initialState
        self.currentState.run()
    # Template method:

    def runAll(self, inputs):
        for i in inputs:
            print(i)
            self.currentState = self.currentState.next(i)
            self.currentState.run()


class MouseAction:
    def __init__(self, action):
        self.action = action

    def __str__(self): return self.action

    def __cmp__(self, other):
        return (self.action > other.action) - (self.action < other.action)

    # Necessary when __cmp__ or __eq__ is defined
    # in order to make this class usable as a
    # dictionary key:

    def __hash__(self):
        return hash(self.action)


class TestAsset(State):
    def run(self):
        G1 = matrix(sparse([G, asset_sub, -asset_sub]))
        h1 = matrix(sparse([h, b_asset_matrix]))
        sol = solvers.qp(P, q, G1, h1, A, b)
        if sol['x'] == 'optimal':
            print('Asset is OK')
            return MouseTrap.test_group
        else:
            print('failure because of asset constraint')

    def next(self, input):
        if input == MouseAction.test_group:
            return MouseTrap.test_group
        #return MouseTrap.waiting


class TestGroup(State):
    def run(self):
        G2 = matrix(sparse([G, Group_sub, -Group_sub]))
        h2 = matrix(sparse([h, b_group_matrix]))

        sol = solvers.qp(P, q, G2, h2, A, b)
        if sol['x'] == 'optimal':
            print('group is OK')
            return MouseTrap.test_exposure
        else:
            print('failure because of group constraint')

    def next(self, input):
        if input == MouseAction.test_exposure:
            return MouseTrap.test_exposure


class TestExposure(State):
    def run(self):
        G3 = matrix(sparse([G, exp_sub, -exp_sub]))
        h3 = matrix(sparse([h, b_factor_exposure_matrix]))

        sol = solvers.qp(P, q, G3, h3, A, b)
        if sol['x'] == 'optimal':
            print('exposure is OK')
            return MouseTrap.test_asset_group
        else:
            print('failure because of exposure constraint')

    def next(self, input):
        if input == MouseAction.test_asset_group:
            return MouseTrap.test_asset_group

class TestAssetGroup(State):
    def run(self):
        print("Holding: Mouse caught")

    def next(self, input):
        if input == MouseAction.removed:
            return MouseTrap.waiting
        return MouseTrap.holding


class TestAssetExposure(State):
    def run(self):
        print("Holding: Mouse caught")

    def next(self, input):
        if input == MouseAction.removed:
            return MouseTrap.waiting
        return MouseTrap.holding


class TestGroupExposure(State):
    def run(self):
        print("Holding: Mouse caught")

    def next(self, input):
        if input == MouseAction.removed:
            return MouseTrap.waiting
        return MouseTrap.holding


class TestAssetGroupExposure(State):
    def run(self):
        print("Holding: Mouse caught")

    def next(self, input):
        if input == MouseAction.removed:
            return MouseTrap.waiting
        return MouseTrap.holding


class MouseTrap(StateMachine):
    def __init__(self):
        # Initial state
        StateMachine.__init__(self, MouseTrap.test_asset)


MouseAction.test_asset = MouseAction("test asset")
MouseAction.test_group = MouseAction("test group")
MouseAction.test_exposure = MouseAction("test exposure")

MouseAction.test_asset_group = MouseAction("test asset group")
MouseAction.test_asset_exposure = MouseAction("test asset exposure")
MouseAction.test_group_exposure = MouseAction("test group exposure")

MouseAction.test_asset_group_exposure = MouseAction("test asset group exposure")

MouseTrap.test_asset = TestAsset()
MouseTrap.test_group = TestGroup()
MouseTrap.test_exposure = TestExposure()
MouseTrap.test_asset_group = TestAssetGroup()
MouseTrap.test_asset_exposure = TestAssetExposure()
MouseTrap.test_group_exposure = TestGroupExposure()
MouseTrap.test_asset_group_exposure = TestAssetGroupExposure()


moves = ['test asset',
         'test group',
         'test exposure',
         'test asset group',
         'test asset exposure',
         'test group exposure',
         'test asset group exposure']

MouseTrap().runAll(map(MouseAction, moves))
