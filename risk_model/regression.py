# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


class RLMModel:

    def fit(self, y, X):
        model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
        # model = sm.RLM()
        return model.fit()


def regression(context, X, y, model):
    """ regression factor exposure by stock return
    Keyword Arguments:
    X     -- factor exposure(xarray)
    Y     -- return(gftTable, dateframe)
    model -- regression model

    Return:
    model -- dictionary, key: value
        param: xarray, factor return, factors: 36, date: 432
        resid: xarray, stock residual, date: 432, symbol: 2981
    """
    if model is None:
        model = RLMModel()
    y = y.asMatrix().copy()
    # get the datetimeindex
    for dim in X.dims:
        if X[dim].values.dtype == np.dtype('<M8[ns]'):
            date = dim
        elif X[dim].values.dtype == object:
            symbol = dim
        elif X[dim].values.dtype == np.dtype('U32'):
            factor = dim
    # create regression result dateframe
    params = pd.DataFrame(data=np.nan, index=X[date], columns=X[factor])
    residuals = pd.DataFrame(data=np.nan, index=X[date], columns=X[symbol])

    # get the datetimeindex
    idx_date = y.index

    for dt in X[date].values:
        logger.debug('regression on %s', dt)
        # cur_date = pd.Timestamp(dt)
        # get the position of current date
        dt_pos = idx_date.get_loc(dt)

        # symbols having valid value(not nan) in X
        s = X[:, dt_pos].notnull().all(axis=0)
        valid_x = X[:, dt_pos, s].symbol.values

        # symbols having valid value(not nan) in y
        w = y.loc[dt].notnull()
        valid_y = y.loc[dt, w].index.values

        valid_symbol = np.intersect1d(valid_x, valid_y)
        try:
            results = model.fit(y.loc[dt, valid_symbol].values,
                                X.loc[:, dt, valid_symbol].values.T)
        except ValueError:
            continue
        except ZeroDivisionError:
            logger.debug('not enough data on y')
        else:
            params.loc[dt] = results.params
            residuals.loc[dt, valid_symbol] = results.resid
    return {'param': params, 'resid': residuals}
