# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import re
import cvxpy as cvx
from copy import copy
from lib.gftTools import gftIO
from scipy import linalg
import matplotlib.pyplot as plt


def omega(P, confidence, tau):
    """
    to calculate variance matrix of view.
    Keyword Arguments:
    P          --
    confidence --
    tau        --
    """
    pass


# def black_litterman_optimization(delta, weq, historical_ret, P, Q, C, tau,
#                                  Sigma):
def black_litterman_optimization(delta, weq, P, Q, tau, Sigma):
    '''
    Calculate new vector of expected return using the Black-Litterman model
    via reverse optimize and back out the equilibrium returns,
    which combines current market equilibrium expected returns
    with investors views described in the paper by He and Litterman.

    Pseudo code
    -----------
    1. Π = δΣw
    2. Quantify their uncertainty in the prior by selecting a value for τ.
    If the covariance matrix has been generated from historical data, then τ = 1/n is a good place to start.
    3. Formulates their views, specifying P, Ω, and Q.
    Given k views and n assets, then P is a k × n matrix where each row sums to 0 (relative view) or 1 (absolute v    iew). Q is a k × 1 vector of the excess returns for each view. Ω is a diagonal k × k matrix of the variance of     the estimated view mean about the unknown view mean. As a starting point, some authors call for the diagonal     values of Ω to be set equal to pTτΣp (where p is the row from P for the specific view). This weights the views     the same as the prior estimates.
    4. Compute the posterior estimate of the returns using the following equation.
    $$\hat\Pi = \Pi + \tau\Sigma P'(P\tau\Sigma P')^{-1}(Q-P\Pi)$$
    5. Compute the posterior variance of the estimated mean about the unknown mean using the following equation.
    $$M=\tau \Sigma - \tau \Sigma P'[P\tau\Sigma P'+\Omega]^{-1}P\tau \Sigma$$
    6. Get the covariance of retujrns about the estimated mean.
    Assuming the uncertainty in the estimates is independent of the known covariance of returns about the unknown mean.
    $$\Sigma_p = \Sigma + M$$
    7. Compute the portfolio weights for the optimal portfolio on the unconstrained efficient frontier.
    $$\omega=\hat{\Pi}(\delta\Sigma_p)^{-1}$$

    Parameters
    ----------
    tau: float
        scalar. Proportional to the relative weight given to the implied
        equilibrium return vector($\Pi$), Coefficiet of uncertainty
        in the prior estimate of the mean.

    delta: float
        risk aversion coefficient, Risk tolerance from the equilibrium portfolio.

    weq: pd.DataFrame
        weights of the assets in the equilibrium portfolio

    equilibrium_return: pd.DataFrame
        implied benchmark asset capitalization weighting return, input portfolio.

    historical_ret: pd.DataFrame
        historical return of all assets.

    Sigma: pd.DataFrame
        Prior variance-covariance matrix (NxN matrix).

    P: pd.DataFrame
        matrix of the assets involved in views (KxN matrix).

    Q: pd.DataFrame
        view matrix.

    C: pd.DataFrame
        confidence level.

    Omega: pd.DataFrame
        Matrix of variance of the views (diagonal).

    Returns
    -------
    new_expected_return_vector: pd.DataFrame

    unconstrainted_optimized_weight: pd.DataFrame
    '''
    pi = weq.dot(Sigma * delta)
    print(pi)
    # We use tau * sigma many places so just compute it once
    ts = tau * Sigma
    # Compute posterior estimate of the mean
    # This is a simplified version of formula (8) on page 4.
    middle = linalg.inv(np.dot(np.dot(P, ts), P.T) + Omega)
    print(middle)
    print(Q - np.expand_dims(np.dot(P, pi.T), axis=1))
    er = np.expand_dims(
        pi, axis=0).T + np.dot(
            np.dot(np.dot(ts, P.T), middle),
            (Q - np.expand_dims(np.dot(P, pi.T), axis=1)))
    # Compute posterior estimate of the uncertainty in the mean
    # This is a simplified and combined version of formulas (9) and (15)
    posteriorSigma = Sigma + ts - ts.dot(P.T).dot(middle).dot(P).dot(ts)
    print(posteriorSigma)
    # Compute posterior weights based on uncertainty in mean
    w = er.T.dot(linalg.inv(delta * posteriorSigma)).T
    # Compute lambda value
    # We solve for lambda from formula (17) page 7, rather than formula (18)
    # just because it is less to type, and we've already computed w*.
    lmbda = np.dot(linalg.pinv(P).T, (w.T * (1 + tau) - weq).T)
    return [er, w, lmbda]


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

C = gftIO.zload("/home/weiwu/share/black_litterman/C.pkl")
ROE = gftIO.zload("/home/weiwu/share/black_litterman/ROE_cur_year.pkl")
delta = gftIO.zload("/home/weiwu/share/black_litterman/delta.pkl")
historical_ret = gftIO.zload(
    "/home/weiwu/share/black_litterman/historical_ret.pkl")
# Q = gftIO.zload("/home/weiwu/share/black_litterman/Q.pkl")
tau = gftIO.zload("/home/weiwu/share/black_litterman/tau.pkl")
weq = gftIO.zload("/home/weiwu/share/black_litterman/weq.pkl")
if isinstance(historical_ret, gftIO.GftTable):
    historical_ret = historical_ret.asMatrix().copy()
    # In [139]: historical_ret.shape
    # Out[140]: (451, 3437)

if isinstance(ROE, gftIO.GftTable):
    # views on all assets are not required
    ROE = ROE.asMatrix().copy()
    ROE.fillna(method='ffill', inplace=True)
    ROE.fillna(method='bfill', inplace=True)

    # In [131]: ROE.shape
    # Out[138]: (490, 7018)

if isinstance(weq, gftIO.GftTable):
    weq = weq.asMatrix().copy()
    weq.fillna(method='ffill', inplace=True)
    # In [141]: weq.shape
    # Out[143]: (451, 3551)

logger.debug('parse data finished!')

target_symbols = historical_ret.columns.intersection(
    weq.columns.intersection(ROE.columns))
target_dates = historical_ret.index.intersection(
    weq.index.intersection(ROE.index))
# get random symbols at the target position limit
# position_limit = 8
# arr = list(range(len(target_symbols)))
# np.random.shuffle(arr)
# target_symbols = target_symbols[arr[:position_limit]]
target_symbols = [
    b'\xe2\x11\xeaH\x17m2\xc1"\'i\xcb+xu\x90',
    b'\xe2\x11\xeaH\x9bW\'\xc1"\'i\xcb+xu\x90',
    b'\xe2\x11\xeaHX\x0c@\xc1"\'i\xcb+xu\x90',
    b'\xe2\x11\xeaH0\x81\x1f\xc1"\'i\xcb+xu\x90',
    b'\xe2\x11\xeaH\x10\x84\x08\xc1"\'i\xcb+xu\x90',
    b'\xe2\x11\xeaH\xfcW(\xc1"\'i\xcb+xu\x90',
    b'\xfcDIE\x9e\xc2\xf7\x05\xc4:\xc4\x06MiS\x80',
    b'\xe2\x11\xeaHDME\xc1"\'i\xcb+xu\x90'
]
logger.debug('select symbols %s', target_symbols)

# calculate market capitalization weight for each asset
dt_target = target_dates[-1]
dt_1Y_pre = dt_target - pd.DateOffset(years=1)
dt_1Q_pre = dt_target - pd.DateOffset(days=90)
logger.debug('1 year datetime range %s:%s', dt_1Y_pre, dt_target)
market_capital = weq.loc[dt_target, target_symbols]
logger.debug('market capital %s', market_capital)
weight = market_capital / market_capital.sum()

# use mean as equilibrium return of the stocks
df_equilibrium = historical_ret.loc[dt_1Y_pre:dt_target, target_symbols].mean()

# Equilibrium covariance matrix
C = .8  # confidence level, simply use a real number
delta = 3.5
logger.info('confidence level %s', C)
logger.info('tau %s', tau)
logger.info('delta %s', delta)
Sigma = historical_ret.loc[dt_1Y_pre:dt_target, target_symbols].fillna(
    0).cov().values
V = Sigma * C

# investor view of return
Q = ROE.loc[dt_target, target_symbols] / ROE.loc[dt_1Q_pre, target_symbols] - 1
# Q = df_equilibrium
# investor view of return, positive:1, negative:-1
investor_position_view = copy(Q)
investor_position_view[investor_position_view >= df_equilibrium] = 1
investor_position_view[investor_position_view < df_equilibrium] = -1
investor_position_view = pd.DataFrame(
    np.diag(investor_position_view), columns=Q.index)
P = investor_position_view.values
# Q
# logger.debug('investor position %s', investor_position_view)
logger.debug('prediction Q %s', Q)
# Coefficient of uncertainty in the prior estimate of the mean
tauV = tau * V
Omega = np.dot(np.dot(P, tauV), P.T) * np.eye(Q.shape[0])

### calculation
pi = weight.values.dot(V * delta)
logger.debug('equilibrium return %s', pi)
# We use tau * sigma many places so just compute it once
ts = tau * V
# Compute posterior estimate of the mean
# This is a simplified version of formula (8) on page 4.
middle = linalg.inv(np.dot(np.dot(P, ts), P.T) + Omega)
logger.debug('Middle %s', middle)
# print(Q - np.expand_dims(np.dot(P, pi.T), axis=1))
# posterior estimate of the returns
er = np.expand_dims(
    pi, axis=0).T + np.dot(
        np.dot(np.dot(ts, P.T), middle),
        (Q - np.expand_dims(np.dot(P, pi.T), axis=1)))
# Compute posterior estimate of the uncertainty in the mean
# This is a simplified and combined version of formulas (9) and (15)
posteriorSigma = V + ts - ts.dot(P.T).dot(middle).dot(P).dot(ts)
logger.debug('posterior Sigma %s', posteriorSigma)
# Compute posterior weights based on uncertainty in mean
w = er.T.dot(linalg.inv(delta * posteriorSigma)).T
#result = black_litterman_optimization(
#    delta=delta, weq=weight, P=P, Q=Q, tau=tau, Sigma=Sigma)
logger.info('unconstrainted optimized weight %s', w)
rets = pd.DataFrame(df_equilibrium, columns=['mean'])
# implied equiblirium return
rets['ier'] = pi
# estimate equiblirium return
rets['eer'] = er
# prediction from investors views
rets['prediction'] = Q
rets.to_csv('returns.csv')
