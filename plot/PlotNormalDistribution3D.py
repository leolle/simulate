# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:51:12 2017

@author: J
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pandas as pd
import numpy as np
from datetime import datetime
from lib.gftTools import gftIO
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import matplotlib as mpl

#saveData=dict({'x0':'x0','x1':'x1','x2':'x2'})
#gftIO.zdump(saveData,r'd:\Wuwei\Project\simulator\data\test_data.pkl')

def parse_data(input_data):
    if type(input_data) is gftIO.GftTable:
        input_data = input_data.asMatrix()
        if input_data.columns.dtype == 'datetime64[ns]':
            return input_data.T
        else:
            return input_data
    elif type(input_data) is np.ndarray:
        return input_data.astype(datetime)
    else:
        return input_data

data = gftIO.zload(r'd:\Wuwei\Project\simulator\data\monthly_rebalance_data_2.pkl')
data['begin_date'] = data.pop('x0')
data['end_date'] = data.pop('x1')
data['initial_holding'] = data.pop('x2')
data['target_portfolio_weight'] = data.pop('x3')
data['market_to_market_price'] = data.pop('x4')
data['total_return_factor'] = data.pop('x5')
data['execute_price'] = data.pop('x6')
data['execute_price_return'] = data.pop('x7')
data['trade_volume'] = data.pop('x8')
data['trading_param'] = data.pop('x9')
data['additional_Ts'] = data.pop('x10')

for key, value in data.items():
    data[key] = parse_data(value)
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D


#Create grid and multivariate normal
x = data['market_to_market_price'].iloc[50:,1]
x = x.fillna(method = 'ffill')
x = x.fillna(method = 'backfill')
y = data['market_to_market_price'].iloc[50:,2]
y = y.fillna(method = 'ffill')
y = y.fillna(method = 'backfill')
x = x.pct_change()[1:]
y = y.pct_change()[1:]

#Parameters to set
mu_x = 0
mu_y = 0
variance_x = x.var()
variance_y = y.var()

#x = np.linspace(-10,10,500)
#y = np.linspace(-10,10,500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

#Make a 3D plot
fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()