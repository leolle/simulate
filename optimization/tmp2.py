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

factor_exposure = pd.DataFrame(np.ones((5,5)),
                               columns=list('ABCDE'))


P = covs
q = matrix(np.zeros((n, 1)), tc='d')
asset_sub = matrix(np.eye(n), tc='d')
asset_sub = matrix(sparse([asset_sub, -asset_sub]))
exp_sub = matrix(factor_exposure.values)
exp_sub = matrix(sparse([exp_sub, -exp_sub]))
# set boundary vector for h
df_asset_weight = pd.DataFrame({'lower': [0.0], 'upper': [1.0]},
                               index=list("ABCDE"))
df_asset_bnd_matrix = matrix(np.concatenate(((df_asset_weight.upper,
                                              df_asset_weight.lower)), 0))


df_factor_exposure_bound = pd.DataFrame(data=[[-10, 20],[-10, 20],[-10, 20],[-10, 20],[-10, 20]], columns=['lower','upper'])


df_factor_exposure_bnd_matrix = matrix(np.concatenate(((df_factor_exposure_bound.upper,
                                                        df_factor_exposure_bound.lower)), 0))


G = matrix(sparse([asset_sub, exp_sub]))
h = matrix(sparse([df_asset_bnd_matrix, df_factor_exposure_bnd_matrix]))

# equality constraint Ax = b; captures the constraint sum(x) == 1
A = matrix(1.0, (1, n))
b = matrix(1.0)
sol = solvers.qp(P, q, G, h, A, b)
#print(sol['status'])
#print(sol['x'])


mu = np.abs(np.random.randn(n, 1))
Sigma = np.random.randn(n, n)
Sigma = Sigma.T.dot(Sigma)
# Long only portfolio optimization.
from cvxpy import *
w = Variable(n)
gamma = Parameter(sign='positive')
mu = np.abs(np.random.randn(n, 1))
#ret = w.T*rets_mean.values
ret = mu.T*w
risk = quad_form(w, Sigma)
prob = Problem(Maximize(ret - gamma*risk), 
               [sum_entries(w) == 1, 
                w >= 0])


# Compute trade-off curve.
SAMPLES = 100
risk_data = np.zeros(SAMPLES)
ret_data = np.zeros(SAMPLES)
gamma_vals = np.logspace(-2, 3, num=SAMPLES)
for i in range(SAMPLES):
    gamma.value = gamma_vals[i]
    prob.solve()
    risk_data[i] = sqrt(risk).value
    ret_data[i] = ret.value

# Plot long only trade-off curve.
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'

markers_on = [29, 40]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(risk_data, ret_data, 'g-')
for marker in markers_on:
    plt.plot(risk_data[marker], ret_data[marker], 'bs')
    ax.annotate(r"$\gamma = %.2f$" % gamma_vals[marker], xy=(risk_data[marker]+.08, ret_data[marker]-.03))
for i in range(n):
    plt.plot(sqrt(Sigma[i,i]).value, mu[i], 'ro')
plt.xlabel('Standard deviation')
plt.ylabel('Return')
plt.show()
