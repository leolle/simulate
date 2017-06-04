# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:43:59 2017

@author: J
"""

import pandas as pd
import numpy as np
from datetime import datetime
from lib.gftTools import gftIO, gsUtils, gsConst
try:
    import cPickle as pickle
except:
    import pickle
import pprint

dataPack = gftIO.zload(r'd:\Wuwei\Project\simulator\data\monthly_rebalance_data_same_as_R_5.pkl')
dataPack = gftIO.transformDict4Name(dataPack)
dataPack['begin_date'] = dataPack.pop('x0')
dataPack['end_date'] = dataPack.pop('x1')
dataPack['initial_holding'] = dataPack.pop('x2')
dataPack['target_portfolio_weight'] = dataPack.pop('x3')
dataPack['market_to_market_price'] = dataPack.pop('x4')
dataPack['total_return_factor'] = dataPack.pop('x5')
dataPack['execute_price'] = dataPack.pop('x6')
dataPack['execute_price_return'] = dataPack.pop('x7')
dataPack['trade_volume'] = dataPack.pop('x8')
dataPack['trading_param'] = dataPack.pop('x9')
dataPack['additional_Ts'] = dataPack.pop('x10')

dataPack['execute_price'] = dataPack['execute_price'].asMatrix()
dataPack['execute_price_return'] = dataPack['execute_price_return'].asMatrix()
dataPack['market_to_market_price'] = dataPack['market_to_market_price'].asMatrix()
dataPack['target_portfolio_weight'] = dataPack['target_portfolio_weight'].asMatrix()
dataPack['total_return_factor'] = dataPack['total_return_factor'].asMatrix()
dataPack['trade_volume'] = dataPack['trade_volume'].asMatrix()    

#print('data', pprint.pprint(dataPack))
#with open('data.pkl', 'rb') as fo:
#        pickle.dump(dataPack, fo)
#output = open('data.pkl', 'wb')
#
## Pickle dictionary using protocol 0.
#pickle.dump(dataPack['trading_param'], output, protocol=1)
#output.close()

#ls_data_name = ['execute_price', 'execute_price_return', 'market_to_market_price',\
#                'target_portfolio_weight', 'total_return_factor', 'trade_volume']

dates = pd.to_datetime(dataPack['begin_date'], format = '%Y%m%d')
dataPack['begin_date'] = dates.strftime('%Y-%m-%d')[0]
dates = pd.to_datetime(dataPack['end_date'], format = '%Y%m%d')
dataPack['end_date'] = dates.strftime('%Y-%m-%d')[0]

for name in list(dataPack.keys()):
    output = open(name+'.pkl', 'wb')
    pickle.dump(dataPack[name], output, protocol=1)
    output.close()


#output = open('trading_param.pkl', 'wb')
#
## Pickle dictionary using protocol 0.
#pickle.dump(dataPack['trading_param'], output, protocol=1)
#output.close()
