# -*- coding: utf-8 -*-
from unittest import TestCase, main

import pandas as pd
import numpy as np
import scipy.optimize as sco

#from lib.gftTools import gftIO, gsUtils

All = slice(None)

# target_portfolio_wgt = pd.DataFrame()
# for i in range(8):
#     weights = np.random.random(noa)
#     weights /= np.sum(weights)
#     target_portfolio_wgt = target_portfolio_wgt.append(pd.DataFrame(weights).T)
#     target_portfolio_wgt.index = dates
#     target_portfolio_wgt.columns = index

class UtilsTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates = pd.date_range('1/1/2000', periods=8)
        cls.data = np.array([[10, 9, 10, 11, 12, 13, 14, 13],
                             [11, 11, 10, 11, 11, 12, 11, 10],
                             [10, 11, 10, 11, 12, 13, 14, 13],
                             [11, 11, 10, 11, 11, 12, 11, 11],
                             [10, 11, 10, 11, 12, 13, 14, 13]])
        cls.industry = ['industry', 'industry', 'utility', 'utility',
                        'consumer']
        cls.symbols = ['A', 'B', 'C', 'D', 'E']  # including Cash
        cls.zipped = list(zip(cls.industry, cls.symbols))
        cls.index = pd.MultiIndex.from_tuples(cls.zipped)
        cls.data_frame = pd.DataFrame(cls.data.T, index=cls.dates,
                                      columns=cls.index)
        cls.rets = cls.data_frame / cls.data_frame.shift(1) - 1
        cls.noa = len(cls.symbols)
        cls.weights = np.array(cls.noa * [1. / cls.noa, ])
        cls.expo_factor = np.ones((cls.noa, cls.noa))
        cls.factor_covariance = cls.data_frame.cov()
        cls.delta = np.diagflat([0.088024, 0.082614, 0.084237, 0.074648,
                                 0.084237])
        cls.cov_matrix = np.dot(np.dot(cls.expo_factor, cls.factor_covariance),
                                cls.expo_factor.T) + cls.delta

    def statistics(self, weights):
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
        pret / pvol : float
            Sharpe ratio for rf=0
        '''
        weights = np.array(weights)

        cov_matrix = np.dot(np.dot(self.expo_factor, self.factor_covariance),
                            self.expo_factor.T) + self.delta

        pret = np.sum(self.rets.mean() * weights) * 252
        prisk = self.calculate_total_risk(weights, cov_matrix)
        pvol = np.sqrt(np.dot(weights.T, np.dot(self.rets.cov() * 252, weights)))
        return np.array([pret, pvol, prisk, pret / pvol])

    def min_func_sharpe(self, weights):
        return -self.statistics(weights)[3]

    def min_func_std(self, weights):
        return self.statistics(weights)[2]

    def max_func_return(self, weights):
        return -self.statistics(weights)[0]

    def test_opts(self):
        noa = self.rets.shape[1]
        cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        bnds = tuple((0, 1) for x in range(noa))
        init_guess = noa * [1. / noa, ]
        self.opts = sco.minimize(self.min_func_sharpe, init_guess, method='SLSQP',
                                 bounds=bnds, constraints=cons)

        self.assertIsNone(np.testing.assert_array_equal(
            np.array([0.478, 0., 0.261,  0., 0.261]),
            self.opts['x'].round(3)))
        self.assertIsNone(np.testing.assert_array_equal(
            np.array([ 10.433,   1.01 ,  27.014,  10.331]),
            self.statistics(self.opts['x']).round(3)))

    def test_specific_risk(self):
        delta = np.diagflat([0.088024, 0.082614, 0.084237, 0.074648,
                             0.084237])
        specific_risk = np.dot(np.dot(self.weights.T, delta),
                               self.weights)
        self.assertEqual(0.016550400000000003, specific_risk)

    def test_cov(self):
        delta = np.diagflat([0.088024, 0.082614, 0.084237, 0.074648,
                             0.084237])
        cov_matrix = np.dot(np.dot(self.expo_factor, self.factor_covariance),
                            self.expo_factor.T) + delta
        cov_array = np.array([[27.07016686,  26.98214286,  26.98214286,  26.98214286,
                               26.98214286],
                              [26.98214286,  27.06475686,  26.98214286,  26.98214286,
                               26.98214286],
                              [26.98214286,  26.98214286,  27.06637986,  26.98214286,
                               26.98214286],
                              [26.98214286,  26.98214286,  26.98214286,  27.05679086,
                               26.98214286],
                              [26.98214286,  26.98214286,  26.98214286,  26.98214286,
                               27.06637986]])

        self.assertIsNone(np.testing.assert_array_almost_equal(cov_array,
                          cov_matrix))

    def test_port_variance(self, ):
        delta = np.diagflat([0.088024, 0.082614, 0.084237, 0.074648,
                             0.084237])
        cov_matrix = np.dot(np.dot(self.expo_factor, self.factor_covariance),
                            self.expo_factor.T) + delta
        port_var = np.dot(np.dot(self.weights.T, cov_matrix), self.weights)
        self.assertAlmostEqual(26.998693260000007, port_var)

    def calculate_covariance_matrix(self, exposure, factor_covariance, delta):
        cov_matrix = np.dot(np.dot(exposure, factor_covariance),
                            exposure.T) + delta

        return cov_matrix

    def calculate_total_risk(self, weights, cov_matrix):
        port_var = np.dot(np.dot(weights.T, cov_matrix), weights)
        return port_var

    def test_opts_weight(self):
        test_weight = np.array([0.187,  0.2,  0.196,  0.221,  0.196])
        cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        bnds = tuple((0, 1) for x in range(self.noa))
        init_guess = self.noa * [1. / self.noa, ]
        optv = sco.minimize(self.min_func_std, init_guess, method='SLSQP',
                            bounds=bnds, constraints=cons)

        self.assertIsNone(np.testing.assert_array_almost_equal(test_weight,
                                                               optv['x'], 3))

    def test_max_ret_at_tolerance_risk(self):
        test_weight = np.array([0.233,  0.116,  0.243,  0.165,  0.243])
        tolerance_risk = 27
        cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x:
                 self.calculate_total_risk(x, self.cov_matrix) -
                 tolerance_risk})

        bnds = tuple((0, 1) for x in range(self.noa))
        init_guess = self.noa * [1. / self.noa, ]
        optv = sco.minimize(self.max_func_return, init_guess, method='SLSQP',
                            bounds=bnds, constraints=cons)

        self.assertIsNone(np.testing.assert_array_almost_equal(test_weight,
                                                               optv['x'], 3))

    def test_min_risk_at_target_return(self, ):
        test_weight = np.array([ 0.045,  0.461,  0.049,  0.396,  0.049])
        target_return = 0.5
        cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x:
                 np.sum(self.rets.mean() * x) * 252 - target_return})

        bnds = tuple((0, 1) for x in range(self.noa))
        init_guess = self.noa * [1. / self.noa, ]
        optv = sco.minimize(self.min_func_std, init_guess, method='SLSQP',
                            bounds=bnds, constraints=cons)

        self.assertIsNone(np.testing.assert_array_almost_equal(test_weight,
                                                               optv['x'], 3))

    def test_min_risk(self, ):
        test_weight = np.array([ 0.045,  0.461,  0.049,  0.396,  0.049])
        cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})

        bnds = tuple((0, 1) for x in range(self.noa))
        init_guess = self.noa * [1. / self.noa, ]
        optv = sco.minimize(self.min_func_std, init_guess, method='SLSQP',
                            bounds=bnds, constraints=cons)

        self.assertIsNone(np.testing.assert_array_almost_equal(test_weight,
                                                               optv['x'], 3))
