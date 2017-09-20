# -*- coding: utf-8 -*-

from unittest import TestCase, main
from lib.gftTools import gftIO, gsConst
import pandas as pd
import numpy as np

from stats import cal_max_dd, cum_returns, annual_return,\
    annual_volatility, sharpe_ratio, sortino_ratio, annual_downside_risk, \
    downside_std, int_trading_days, annual_volatility, return_std,\
    max_holding_num, min_holding_num, latest_holding_num, average_holding_num_percentage,\
    latest_holding_num_percentage, excess_annual_return, aggregate_returns,\
    portfolio_market_ratio, holding_dispersion_std


class UtilsTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        path = r'/home/weiwu/projects/simulate/data/stats/'
        cls.f_risk_free_rate = 0.0
        cls.f_risk_free_rate = gftIO.zload(path + 'f_risk_free_rate.pkl')
        df_single_period_return = gftIO.zload(path + 'df_single_period_return.pkl')
        benchmark_ret = gftIO.zload(path + 'benchmark_ret.pkl')
        holding = gftIO.zload(path + 'holding.pkl')
        closing_price = gftIO.zload(path + 'closing_price.pkl')
        market_capital = gftIO.zload(path + 'market_capital.pkl')

        if isinstance(df_single_period_return, gftIO.GftTable):
            cls.df_single_period_return = df_single_period_return.asMatrix().copy()

        if isinstance(benchmark_ret, gftIO.GftTable):
            cls.benchmark_ret = benchmark_ret.asMatrix().copy()

        if isinstance(holding, gftIO.GftTable):
            cls.holding = holding.asMatrix().copy()

        if isinstance(closing_price, gftIO.GftTable):
            cls.closing_price = closing_price.asMatrix().copy()

        if isinstance(market_capital, gftIO.GftTable):
            cls.market_capital = market_capital.asMatrix().copy()

        dt_diff = cls.df_single_period_return.index.to_series().diff().mean()
        if dt_diff < pd.Timedelta('3 days'):
            cls.periods = gsConst.Const.DAILY
        elif dt_diff > pd.Timedelta('3 days') and dt_diff < pd.Timedelta('10 days'):
            cls.periods = gsConst.Const.WEEKLY
        else:
            cls.periods = gsConst.Const.MONTHLY

    def test_annualized_return(self):
        self.assertEqual(0.38809843766416452,
                         annual_return(self.df_single_period_return, self.periods))

    def test_annualized_volatility(self, ):
        self.assertAlmostEqual(0.14032998543027239,
                               annual_volatility(self.df_single_period_return,
                                                 self.periods))

    # def test_annualized_downrisk_vol(self):
    #     self.assertAlmostEqual(0.094976063279974793,
    #                            annual_downside_risk(self.df_single_period_return,
    #                                                 self.periods))

    def test_cumulative_return(self):
        self.assertAlmostEqual(1.0914409018916436,
                               cum_returns(self.df_single_period_return))

    def test_downside_std(self):
        self.assertAlmostEqual(0.027417227850632192,
                               downside_std(self.df_single_period_return))

    def test_max_dd(self):
        self.assertAlmostEqual(-0.077828826339974566,
                               cal_max_dd(self.df_single_period_return))

    def test_return_std(self):
        self.assertAlmostEqual(0.040509777431772018,
                               return_std(self.df_single_period_return))

    def test_sharpe_ratio(self):
        self.assertAlmostEqual(2.7656130403933101,
                               sharpe_ratio(self.df_single_period_return,
                                            self.f_risk_free_rate,
                                            self.periods))

    def test_sortino_ratio(self):
        self.assertAlmostEqual(0.89169859590100764,
                               sortino_ratio(self.df_single_period_return,
                                             self.f_risk_free_rate,
                                             self.periods))

    def test_trading_days(self):
        self.assertEqual(27, int_trading_days(self.df_single_period_return))


if __name__ == '__main__':
    main()
