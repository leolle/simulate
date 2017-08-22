# -*- coding: utf-8 -*-
import logging

import numpy as np
import pandas as pd
import re
import cvxpy as cvx
np.random.seed(123)


def constraint(obj, df_limit):
    if isinstance(df_limit, pd.DataFrame):
        return [obj >= df_limit.iloc[:, 0].values,
                obj <= df_limit.iloc[:, 1].values]


def logrels(rets):
    """Log of return relatives, ln(1+r), for a given DataFrame rets."""
    return np.log(rets + 1)



dates = pd.bdate_range(start='1/1/2000', end='12/31/2000')
industry = ['industry'] * 2 + ['utility'] * 2 + ['consumer']
symbols = list('ABCDE')
factors = list('ROI')
zipped = list(zip(industry, symbols))
cols = pd.MultiIndex.from_tuples(zipped)
returns = pd.DataFrame(np.random.randn(len(dates), len(cols)), index=dates, columns=cols)
returns /= 100 + 3e-3 #drift term
df_asset_weight = pd.DataFrame({'lower': [0.0], 'upper': [1.0]},
                               index=symbols)

data = np.array([[-1.48432831,  0.98625057, -0.47563856],
       [ 0.23484376, -0.6304754 ,  0.51788663],
       [ 0.95798994, -1.20962774, -0.81357143],
       [-1.66372841,  0.23340731, -1.66528042],
       [-0.36608169,  0.51961115,  2.05809293]])
F = pd.DataFrame(data, index=symbols, columns=factors)
factor_exposure = np.array([[-1.48432775],
                            [ 0.98625005],
                            [-0.37563839]])
priority = np.array(range(1,4)).reshape(-1,1)
number= np.log(np.logspace(.000001, 2))
status = []
result = {}
noa = len(cols)
N = 100
mus = [5**(-5.0*t/N) for t in range(N)]
for n in number:
    w = cvx.Variable(noa)
    f = F.T.values*w
    ret = w.T * logrels(returns).mean().values
    Lmax = cvx.Parameter()
    Lmax.value = 1
    asset_weight_constraint = constraint(w, df_asset_weight)
    eq_constraint = [cvx.sum_entries(w) == 1,
                     cvx.norm(w, 1) <= Lmax]
    factor_neutral = [f==factor_exposure]
    # factor_neutral = [f==np.zeros((3,1))]
    penalty_func = n * cvx.norm(F.T.values*w - factor_exposure, 1)

    prob_factor = cvx.Problem(cvx.Maximize(ret - penalty_func), eq_constraint+asset_weight_constraint)
    prob_factor.solve(verbose=False)
    status.append(prob_factor.status)
    result[n] = (F.T.values*w.value - factor_exposure)/factor_exposure
