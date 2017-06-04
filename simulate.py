# -*- coding: utf-8 -*-
"""
Spyder Editor

"""


import pandas as pd
import numpy as np
from datetime import datetime
#from lib.gftTools import gftIO
import tsutil as tsu
# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da
import QSTK.qstktools.report as report
import QSTK.qstksim as qstksim

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
#saveData=dict({'x0':'x0','x1':'x1','x2':'x2'})
#gftIO.zdump(saveData,r'd:\Wuwei\Project\simulator\data\test_data.pkl')

#data = gftIO.zload(r'd:\Wuwei\Project\simulator\data\SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK.SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK.11.pkl')


#os.chdir(r'd:\Wuwei\Project\tools\PythonScripts\lib\gftTools\convertSymbols')
#dataPack = gftIO.zload(r'd:\Wuwei\Project\simulator\data\SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK.SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK.11.pkl')
#print(dataPack)

""" 模拟输入信息 """
dates = pd.date_range('1/1/2000', periods=8)
start_date = dates[0]
end_date = dates[-1]
symbols_and_cash = ['A', 'B', 'C', 'D', 'E', 'Cash']  # including Cash
symbols = symbols_and_cash[:-1]
noa = len(symbols)
num_of_all = len(symbols_and_cash)
weights = np.random.random(noa)
weights /= np.sum(weights)

def generate_stock_price(period, S0):
    T = 2
    mu = 0.1
    sigma = 0.01
    dt = 0.01
    N = period
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = S0*np.exp(X) ### geometric brownian motion ###
    return S

market_to_market_price = pd.DataFrame(index=dates, columns=symbols)
execute_price = pd.DataFrame(index=dates, columns=symbols)
for sym in symbols:
    market_to_market_price[sym] = generate_stock_price(8, np.random.randint(high=50, low=45))
    execute_price[sym] = generate_stock_price(8, np.random.randint(high=50, low=45))
    
total_return_factor = pd.DataFrame(np.ones(shape=(8, noa)), \
                                      index=dates, columns=symbols)
trade_volume = pd.DataFrame(np.random.randint(100, 500, size=(8, noa)), \
                                      index=dates, columns=symbols)
exec_price_return = pd.DataFrame(np.random.randint(low=5,size=(8, noa)), \
                                      index=dates, columns=symbols)
initial_holding = pd.DataFrame([[100, 100, 500, 600, 800, 10000],[100, 100, 500, 600, 800, 10000]], \
                                      index=dates[:2], columns=symbols_and_cash)
#initial_holding = 1000
target_portfolio_wgt = pd.DataFrame()
for i in range(8):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    target_portfolio_wgt = target_portfolio_wgt.append(pd.DataFrame(weights).T)
target_portfolio_wgt.index = dates
target_portfolio_wgt.columns = symbols
target_portfolio_wgt['Cash'] = 0
# List of symbols
ls_symbols = ['A', 'B', 'C', 'D', 'E']

# Start and End date of the charts
dt_start = dt.datetime(2006, 1, 1)
dt_end = dt.datetime(2010, 12, 31)
df_close = market_to_market_price

(ts_funds, ts_leverage, f_commission, f_slippage, f_borrow_cost) = qstksim.tradesim(target_portfolio_wgt,
                    market_to_market_price, f_start_cash=10000.0, i_leastcount=1, b_followleastcount=True,
                    f_slippage=0.0005, f_minimumcommision=5.0, f_commision_share=0.0035,
                    i_target_leverage=1, f_rate_borrow=3.5, log="transaction.csv")
