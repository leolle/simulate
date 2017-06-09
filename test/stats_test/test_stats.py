# -*- coding: utf-8 -*-

from unittest import TestCase, main

import pandas as pd
import numpy as np

from stats.stats import cal_max_dd, cum_returns, annual_return,\
    annual_volatility, sharpe_ratio, sortino_ratio, annual_downside_risk, \
    downside_std, int_trading_days, annual_volatility, return_std


class UtilsTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates = pd.date_range('1/1/2000', periods=10)
        cls.data = [100, 101, 100, 101, 102, 103, 104, 105, 106, 107]
        cls.dataDf = pd.DataFrame(cls.data, index=cls.dates, columns=['price'])
        cls.df_single_period_return = cls.dataDf / cls.dataDf.shift(1) - 1
        cls.f_risk_free_rate = 0.015

    def test_annualized_return(self):
        self.assertEqual(4.5013743428415527,
                         annual_return(self.df_single_period_return))

    def test_annualized_volatility(self, ):
        self.assertAlmostEqual(0.10402323532228903,
                               annual_volatility(self.df_single_period_return))

    def test_annualized_downrisk_vol(self):
        self.assertAlmostEqual(0.092421812073504264,
                               annual_downside_risk(
                                   self.df_single_period_return))

    def test_cumulative_return(self):
        self.assertAlmostEqual(0.070000000000000062,
                               cum_returns(self.df_single_period_return))

    def test_downside_std(self):
        self.assertAlmostEqual(0.005822026915819979,
                               downside_std(self.df_single_period_return))

    def test_max_dd(self):
        self.assertAlmostEqual(-0.0099009900990099098,
                               cal_max_dd(self.df_single_period_return))

    def test_return_std(self):
        self.assertAlmostEqual(0.0065528478865506338,
                               return_std(self.df_single_period_return))

    def test_sharpe_ratio(self):
        self.assertAlmostEqual(43.12857919619,
                               sharpe_ratio(self.df_single_period_return,
                                            self.f_risk_free_rate))

    def test_sortino_ratio(self):
        self.assertAlmostEqual(48.5423759,
                               sortino_ratio(self.df_single_period_return,
                                             self.f_risk_free_rate))

    def test_trading_days(self):
        self.assertEqual(10, int_trading_days(self.df_single_period_return))


if __name__ == '__main__':
    main()
