# -*- coding: utf-8 -*-
import logging

import numpy as np
import pandas as pd
import re
import cvxpy as cvx

from lib.gftTools import gsConst, gftIO
from scipy import linalg


def omega(P, confidence, tau):
    """
    to calculate variance matrix of view.
    Keyword Arguments:
    P          --
    confidence --
    tau        --
    """
    pass


def black_litterman_optimization(delta, weq, historical_ret, P, Q, C, tau):
    '''
    Calculate new vector of expected return using the Black-Litterman model
    via reverse optimize and back out the equilibrium returns,
    which combines current market equilibrium expected returns
    with investors views.

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
    pass


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

C = gftIO.zload("/home/weiwu/share/black_litterman/C.pkl")
P = gftIO.zload("/home/weiwu/share/black_litterman/P.pkl")
delta = gftIO.zload("/home/weiwu/share/black_litterman/delta.pkl")
historical_ret = gftIO.zload(
    "/home/weiwu/share/black_litterman/historical_ret.pkl")
Q = gftIO.zload("/home/weiwu/share/black_litterman/Q.pkl")
tau = gftIO.zload("/home/weiwu/share/black_litterman/tau.pkl")
weq = gftIO.zload("/home/weiwu/share/black_litterman/weq.pkl")
