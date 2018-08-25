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
# simple risk parity

df_std = df_returns.rolling(window =90).std()*np.sqrt(252)
wts = (1/df_std).apply(lambda x: x/((1/df_std).sum(axis=1)))
df_ret_rp = (df_returns * wts.shift(1)).sum(axis=1)
df_ret_static = df_returns.dot(np.array([1/2, 1/2, 0]))

ax = df_returns.cumsum().plot()
df_ret_rp.cumsum().to_frame("RP").plot(ax =ax, secondary_y= True)
(df_ret_static).to_frame("EW").cumsum().plot(ax=ax, style=['--'])

ax.set_xlabel("Time")
ax.set_ylabel("Cumulative Returns")

# risk contribution
pt_risk_contribution = wts * 0
weighting_scheme = "NRP"
# weighting_scheme = "60-40"
lookback = 60
for i in np.arange(len(wts)):
    if i >= lookback:
        if weighting_scheme == "NRP":
            wts_vec = wts.iloc[i]
        elif weighting_scheme == "60-40":
            wts_vec = np.array([0.6,0.4])
            
        Sigma = df_returns.iloc[i-lookback+1:i,:].cov()
        var_p = wts_vec.dot(Sigma).dot(wts_vec)
        risk_contribution_vec = (wts_vec * (Sigma.dot(wts_vec)))/var_p
        pt_risk_contribution.iloc[i,:] = np.array(risk_contribution_vec)

ax = pt_risk_contribution.plot()
ax.set_xlabel("Time")
ax.set_ylabel("Risk Contribution")
#ax.set_title("Risk Contribution (Weighting Scheme: " + weighting_scheme + ")")
