# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import linalg


def omega(P, confidence, tau):
    """
    to calculate variance matrix of view.
    Keyword Arguments:
    P     --
    confidence --
    tau   --
    """


def calculate_adjusted_return(tau, lamda, weq, equilibrium_ret, historical_ret,
                              sigma, P, Q, C, risk_free_rate):
    '''
    Calculate new vector of expected return using the Black-Litterman model
    that reverse optimize and back out the equilibrium returns,
    which combines market equilibrium expected returns with investors views.

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

    lamda: float
        risk aversion coefficient, Risk tolerance from the equilibrium portfolio.

    weq: pd.DataFrame
        weights of the assets in the equilibrium portfolio

    equilibrium_return: pd.DataFrame
        implied benchmark asset capitalization weighting return.

    historical_ret: pd.DataFrame
        historical return of all assets.

    sigma: pd.DataFrame
        Prior variance-covariance matrix (NxN matrix).

    P: pd.DataFrame
        matrix of the assets involved in views (KxN matrix).

    Q: pd.DataFrame
        view matrix.

    omega: pd.DataFrame
        Matrix of variance of the views (diagonal).

    risk_free_rate: float

    Returns
    -------
    new_expected_return_vector: pd.DataFrame
    '''
    pass


def blacklitterman(delta, weq, sigma, tau, P, Q, Omega):
    # blacklitterman
    #   This function performs the Black-Litterman blending of the prior
    #   and the views into a new posterior estimate of the returns as
    #   described in the paper by He and Litterman.
    # Inputs
    #   delta  - Risk tolerance from the equilibrium portfolio
    #   weq    - Weights of the assets in the equilibrium portfolio
    #   sigma  - Prior covariance matrix
    #   tau    - Coefficiet of uncertainty in the prior estimate of the mean (pi)
    #   P      - Pick matrix for the view(s)
    #   Q      - Vector of view returns
    #   Omega  - Matrix of variance of the views (diagonal)
    # Outputs
    #   Er     - Posterior estimate of the mean returns
    #   w      - Unconstrained weights computed given the Posterior estimates
    #            of the mean and covariance of returns.
    #   lambda - A measure of the impact of each view on the posterior estimates.
    #
    # Reverse optimize and back out the equilibrium returns
    # This is formula (12) page 6.
    pi = weq.dot(sigma * delta)
    print(pi)
    # We use tau * sigma many places so just compute it once
    ts = tau * sigma
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
    posteriorSigma = sigma + ts - ts.dot(P.T).dot(middle).dot(P).dot(ts)
    print(posteriorSigma)
    # Compute posterior weights based on uncertainty in mean
    w = er.T.dot(linalg.inv(delta * posteriorSigma)).T
    # Compute lambda value
    # We solve for lambda from formula (17) page 7, rather than formula (18)
    # just because it is less to type, and we've already computed w*.
    lmbda = np.dot(linalg.pinv(P).T, (w.T * (1 + tau) - weq).T)
    return [er, w, lmbda]


# Function to display the results of a black-litterman shrinkage
# Inputs
#   title	- Displayed at top of output
#   assets	- List of assets
#   res		- List of results structures from the bl function
#
def display(title, assets, res):
    er = res[0]
    w = res[1]
    lmbda = res[2]
    print('\n' + title)
    line = 'Country\t\t'
    for p in range(len(P)):
        line = line + 'P' + str(p) + '\t'
    line = line + 'mu\tw*'
    print(line)

    i = 0
    for x in assets:
        line = '{0}\t'.format(x)
        for j in range(len(P.T[i])):
            line = line + '{0:.1f}\t'.format(100 * P.T[i][j])

        line = line + '{0:.3f}\t{1:.3f}'.format(100 * er[i][0], 100 * w[i][0])
        print(line)
        i = i + 1

    line = 'q\t\t'
    i = 0
    for q in Q:
        line = line + '{0:.2f}\t'.format(100 * q[0])
        i = i + 1
    print(line)

    line = 'omega/tau\t'
    i = 0
    for o in Omega:
        line = line + '{0:.5f}\t'.format(o[i] / tau)
        i = i + 1
    print(line)

    line = 'lambda\t\t'
    i = 0
    for l in lmbda:
        line = line + '{0:.5f}\t'.format(l[0])
        i = i + 1
    print(line)


# Take the values from He & Litterman, 1999.
weq = np.array([0.016, 0.022, 0.052, 0.055, 0.116, 0.124, 0.615])
C = np.array([[1.000, 0.488, 0.478, 0.515, 0.439, 0.512,
               0.491], [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779], [
                   0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668
               ], [0.515, 0.655, 0.861, 1.000, 0.354, 0.777,
                   0.653], [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
              [0.512, 0.608, 0.783, 0.777, 0.405, 1.000,
               0.652], [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]])
Sigma = np.array([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187])
refPi = np.array([0.039, 0.069, 0.084, 0.090, 0.043, 0.068, 0.076])
assets = {
    'Australia', 'Canada   ', 'France   ', 'Germany  ', 'Japan    ',
    'UK       ', 'USA      '
}

# Equilibrium covariance matrix
V = np.multiply(np.outer(Sigma, Sigma), C)
#print(V)

# Risk aversion of the market
delta = 2.5

# Coefficient of uncertainty in the prior estimate of the mean
# from footnote (8) on page 11
tau = 0.05
tauV = tau * V

# Define view 1
# Germany will outperform the other European markets by 5%
# Market cap weight the P matrix
# Results should match Table 4, Page 21
P1 = np.array([0, 0, -.295, 1.00, 0, -.705, 0])
Q1 = np.array([0.05])
P = np.array([P1])
Q = np.array([Q1])
Omega = np.dot(np.dot(P, tauV), P.T) * np.eye(Q.shape[0])
res = blacklitterman(delta, weq, V, tau, P, Q, Omega)
display('View 1', assets, res)

# Define view 2
# Canadian Equities will outperform US equities by 3%
# Market cap weight the P matrix
# Results should match Table 5, Page 22
P2 = np.array([0, 1.0, 0, 0, 0, 0, -1.0])
Q2 = np.array([0.03])
P = np.array([P1, P2])
Q = np.array([Q1, Q2])
Omega = np.dot(np.dot(P, tauV), P.T) * np.eye(Q.shape[0])
res = blacklitterman(delta, weq, V, tau, P, Q, Omega)
display('View 1 + 2', assets, res)
