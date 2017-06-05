# -*- coding: utf-8 -*-

# from unittest import TestCase, main

import pandas as pd
import numpy as np

from stats.stats import annual_return, annual_volatility, downside_risk


# class UtilsTestCase(TestCase):
#     def test_annualized_return(self):
#         data = [100, 101, 100, 101, 102, 103, 104, 105, 106, 107]
#         dates = pd.date_range('1/1/2000', periods=10)
#         df_price = pd.DataFrame(data, index=dates, columns=['price'])
#         df_single_period_return = df_price / df_price.shift(1) - 1
#         f_risk_free_rate = 0.015        #self.assertEqual(7.7729923, annual_return(df_single_period_return))
#         self.assertEqual(7.7729923, data[0])

def main():
    print('a')


if __name__ == '__main__':
    main()
