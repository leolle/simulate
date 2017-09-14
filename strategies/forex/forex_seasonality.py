# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#import pandas.io.data as web
#import quandl
import datetime
#from urllib import urlretrieve
import matplotlib as mat
import matplotlib.pyplot as plt
from pandas_datareader import data, wb
#plt.use("gtk")
import seaborn as sns
from tqdm import tqdm
import urllib
from os import listdir, path, mkdir


def complete_dir_path(dir_path):
    if not dir_path.endswith('/'):
        return dir_path + '/'
    else:
        return dir_path


result_dir = "./result/"
if not path.exists(result_dir):
    mkdir(result_dir)

url1 = 'http://www.netfonds.no/quotes/paperhistory.php?paper=%s.%s&csv_format=csv'
#df = pd.DataFrame()

#    for i in range(1,13):
#        seasonality = df_monthly.loc[(df_monthly.index.month==i)]
#        plt.boxplot(seasonality['return'])
#seasonality[['open','high']].plot.box()

"""
seasonality of certain month:
df_monthly.apply(lambda x: x * 100.0).loc[(df_monthly.index.month==1)].plot.box()
sns.boxplot(df_monthly.apply(lambda x: x * 100.0).loc[(df_monthly.index.month==1)])



seasonality of certain pair:
df_monthly.loc[(df_monthly.index.month==1)].plot.box()
"""

#forex_list = ['usdchf', 'gbpchf', 'eurnzd', 'eurjpy', 'eurgbp', 'eurchf',\
#              'euraud', 'eurusd', 'gbpusd', 'usdjpy', 'audusd', 'nzdusd', 'usdcad']
forex_list = ['usdchf', 'eurusd', 'gbpusd', 'usdjpy', 'audusd', 'nzdusd', \
            'xauusd', 'xagusd', 'usdcad']
df_monthly = pd.DataFrame()

forex_list = ['eurusd', 'usdchf', 'usdjpy', 'usdcad', 'audusd', 'nzdusd', 'gbpusd',  \
              'eurgbp', 'eurchf', 'euraud',  'eurjpy', 'eurnzd', 'gbpchf',\
              'xauusd', 'xagusd']

source_dict = {'usdchf':'FXSB', 'gbpchf':'FXSB', 'eurnzd':'FXSB', 'eurjpy':'FXSB',\
                 'eurgbp':'FXSB', 'eurchf':'FXSB','euraud':'FXSB', 'eurusd':'FXSB'\
                 , 'gbpusd':'FXSB', 'usdjpy':'FXSB', 'audusd':'FXSB', 'nzdusd':'FXSB'\
                 , 'usdcad':'FXSB', 'xauusd':'FXSX', 'xagusd':'FXSX'}


def fetch_forex_price(symbol_list):
    url1 = 'http://www.netfonds.no/quotes/paperhistory.php?paper=%s.%s&csv_format=csv'
    forex_close_price = pd.DataFrame()
    for pair in tqdm(symbol_list):
        print("fetching %s" % pair)
        df = pd.read_csv(url1%(pair,source_dict[pair]), index_col=['quote_date'], parse_dates=True)
        dataframe = df[[u'paper', u'open', u'high', u'low', u'close']]
        dataframe = dataframe.rename(columns={'paper': 'symbol'})
        dataframe.index.name = 'date'
        dataframe = dataframe.sort_index(ascending=True)
        forex_close_price[pair] = dataframe['close']
    print("fetching data finished.")
    return forex_close_price
#    output = pair + '.xls'
#    dataframe.plot()
#    pylab.show()
#    dataframe.to_excel(output, sheet_name=pair.upper())
forex_close_price = fetch_forex_price(forex_list)

sampled_by_month = forex_close_price.resample(rule='1m', how='last')
log_returns = np.log(sampled_by_month/sampled_by_month.shift(1))
log_returns = log_returns[1:]
pct_returns = sampled_by_month.pct_change()
log_returns.drop(log_returns.index[0], inplace=True)
pct_returns.drop(pct_returns.index[0], inplace=True)
log_returns.gbpusd = log_returns.gbpusd.dropna()


""" 
plot monthly log return of each pair at month
"""
monthinteger = 9
month = datetime.date(1900, monthinteger, 1).strftime('%B')
ax, fig = plt.subplots(1,1,figsize=(8,4))
plt.title('Forex seasonality for %s'%month)
plt.ylabel('Monthly log return')
sns.boxplot(log_returns.ix[log_returns.index.month==monthinteger])
ax.autofmt_xdate()
ax.savefig(result_dir + 'Monthly log return in %s.png'%month)

""" 
boxplot monthly log return of a pair from Jan to Dec
"""
for sym in forex_list:
    ax, fig = plt.subplots(1,1,figsize=(8,4))
    sns.boxplot(data=[log_returns[sym][log_returns[sym].index.month==s].dropna().values \
                for s in range(1,13)])
    plt.title('seasonality in %s from 2005-2017'%sym)
    plt.xlabel('Months')
    ax.savefig(result_dir + ('seasonality in %s from 2005-2017.png'%sym))

""" 
barplot monthly log return of a pair from 2005 to 2017
"""
for sym in forex_list:
    ax, fig = plt.subplots(1,1,figsize=(8,4))
    plt.title('Monthly log return in %s from 2005 to 2017 in the month %s'%(sym, month))
    sns.barplot(log_returns[sym].index[log_returns[sym].index.month==monthinteger].dropna().strftime("%Y-%m"),
            log_returns[sym].ix[log_returns.index.month==monthinteger])
    ax.autofmt_xdate()
    ax.savefig(result_dir + 'Monthly log return in %s from 2005 to 2017 in the month %s.png'%(sym, month))

#sns.boxplot(pct_returns.apply(lambda x: x * 100.0).loc[(pct_returns.index.month==monthinteger)])
#plt.ylabel('Monthly percentage change (%)')
#ax.autofmt_xdate()
