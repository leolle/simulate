# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm


np.random.seed(123)
date_range = pd.date_range(start='2016-01-01', periods=10)
ls_symbols = ['a', 'b', 'c', 'd', 'e']
ls_factors = ['volume', 'benefit', 'volatility', 'POE', 'MA25']
index = pd.MultiIndex.from_product([date_range, ls_symbols],
                                   names=['date', 'symbol'])

df_factor_exposure = pd.DataFrame(np.random.randn(50, 5), index=index, columns=ls_factors)
df_symbol_return = pd.DataFrame(np.random.randn(10, 5), index=date_range, columns=ls_symbols)
df_symbol_return /= 100 + 3e-3

for date in date_range:
    Y = df_symbol_return.ix[date].values.astype(np.float64)
    X = df_factor_exposure.ix[date].values.astype(np.float64)
    model = sm.RLM(Y, X, M=sm.robust.norms.HuberT())
    result = model.fit()
    df_model_params = pd.DataFrame(result.params, columns=['params'])
    df_model_params.index = date

    df_model_residual = pd.DataFrame(result.resid, columns=['resid'])
    df_model_residual.index = date
