# -*- coding: utf-8 -*-
"""
Spyder Editor

"""


import pandas as pd
import numpy as np
from datetime import datetime
from lib.gftTools import gftIO

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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

strike = np.linspace(50, 150, 5)
ttm = np.linspace(0.5, 2.5, 8)

strike, ttm = np.meshgrid(strike, ttm)
#iv = (strike - 100) ** 2 / (100 * strike) / ttm
iv = market_to_market_price.as_matrix()
fig = plt.figure(figsize=(9,6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(strike, ttm, iv, rstride=2, cstride=2,
                       cmap=plt.cm.coolwarm, linewidth=0.5,
                       antialiased=True)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("factor exposure")
ax.set_ylabel("date")
ax.set_zlabel("stock count")