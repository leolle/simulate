# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
np.random.seed(1000)
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import Image, display
from scipy.optimize import minimize
import time
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

data = pd.read_csv(
    './assets.csv', index_col='Date', parse_dates=True).rename(
        columns={
            "CBA00332.CS": "Bond",
            "NH0100.NHF": "Commodity",
            "000300.SH": "Equity"
        })
data.head()
tickers = ['Bond', 'Commodity', 'Equity']

lookback = 21 * 6
corr_lookback = 21 * 24
periodicity = 252

leverage = 2
margin_rate = .02768

n_tickers = len(tickers)
N = len(data)

df_returns = data[tickers].pct_change().dropna()
logging.debug(df_returns.head())
