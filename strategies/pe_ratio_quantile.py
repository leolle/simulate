# coding: utf-8


import DataAPI
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.stats import mstats
sns.set_style('whitegrid')


def get_factor_data(ticker, begin_date, end_date, factor_name):
    data = DataAPI.MktIdxdEvalGet(ticker=ticker, beginDate=begin_date, endDate=end_date, field=[u"ticker", u"tradeDate", factor_name], pandas="1")
    return data


def get_index_data(ticker, begin_date, end_date):
    data = DataAPI.MktIdxdGet(ticker=ticker,beginDate=begin_date, endDate=end_date,exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")
    data.to_csv(ticker+'.csv')
    return data

#factor_data = get_factor_data(u"399006.sz", u"20010101", datetime.now(), u"pe")
def plot_factor_data(factor_data):
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    factor_data.plot(ax=ax)
    x_ticks = ax.get_xticks()[:-1]
    s1 = ax.set_xticklabels([factor_data.ix[index, 'tradeDate'] for index in x_ticks])
    ax.set_title(factor_data.ix[0, 'ticker'])
data = DataAPI.MktIdxdEvalGet(beginDate=u"20080505",endDate=datetime.now(),field=u"",pandas="1")
data = data[(data['PEType']==1)]
ticker = data[['ticker', 'secShortName']].drop_duplicates()
data = data.set_index(['tradeDate'])


ticker['secName'] = ['Shangzhen', 'HS300', 'MS_indx', 'MS_composite', 'GEM_indx', 'GEM_R', 'GEM_composite']
ticker

data.tail(8)


index_data = get_index_data(u'000001', u"20080101", datetime.now())
index_data.tail()


import seaborn as sns
from scipy.stats import mstats
from scipy import stats
import pandas as pd
import numpy as np


def get_index(data, ticker, factor):
    for index, row in ticker.iterrows():
        df_factor = data[(data['ticker']==row['ticker'])][factor]
        start_date = df_factor.index[0]
        df_index = get_index_data(row['ticker'], start_date, datetime.now())
        df_index.to_csv(row['ticker']+'.csv')
        

get_index(data, ticker, 'PEValue')          


import seaborn as sns
from scipy.stats import mstats
from scipy import stats
import pandas as pd
import numpy as np


def get_index_data(data, ticker, factor):
    for index, row in ticker.iterrows():
        df_factor = data[(data['ticker']==row['ticker'])][factor]
        start_date = df_factor.index[0]
        df_index = get_index_data(row['ticker'], start_date, datetime.now())
        df_index.to_csv(row['ticker']+'.csv')
        
        
def plot_factor(data, ticker, factor):
    for index, row in ticker.iterrows():
        df_factor = data[(data['ticker']==row['ticker'])][factor]
        #print(df_factor.shape)
        df_factor.dropna(inplace=True)
        idx = df_factor.index
        df_factor = pd.Series(index=idx, data=mstats.winsorize(df_factor, limits=0.025))
        
        #df_factor['pe_ratio_adj_by_pct'] = mstats.winsorize(df_factor, limits=0.025)
        #print(df_factor.name)
        start_date = df_factor.index[0]
        #df_factor[df_factor>np.percentile(df_factor, 90)] = np.percentile(df_factor, 90)
        #df_factor = pd.DataFrame(index=idx, data=df_factor)
        #df_factor = df_factor[(np.abs(stats.zscore(df_factor)) < 3).all()]
        #df_factor = df_factor.apply(mstats.winsorize,limits=0.005)
        #df_factor = df_factor.dropna()
        df_index = get_index_data(row['ticker'], start_date, datetime.now())
        df_index = df_index.set_index(['tradeDate'])
        #df_index = df_index.ix[idx]
        fig, axes = plt.subplots(1, 1, figsize=(10, 7))
        #axes[0].boxplot(df_factor)
        #axes[0].plot(1, df_factor.ix[-1], 'r*', markersize=15.0)
        
        a = str.decode(row['secName'], 'utf-8')
        
        axes.set_title(a)
        df_factor.plot(legend=True)

        s_index = df_index['closeIndex']
        s_index.plot(secondary_y=True, style='g', legend=True)
        
        #print(df_factor.ix[-1])
        #print(s_index.shape,df_factor.shape)
        #print(df_factor)
        #print(s_index)
        g = sns.JointGrid(x=df_factor, y=s_index, size=10)  
        g.plot_joint(sns.regplot, order=2)  
        plt.axvline(x=df_factor.ix[-1])
        plt.axhline(y=s_index.ix[-1])
        g.plot_marginals(sns.distplot)  

        
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
                                    gridspec_kw={"height_ratios": (.15, .85)})
        plt.axvline(x=df_factor.ix[-1])
        sns.boxplot(df_factor, ax=ax_box)
        sns.distplot(df_factor, ax=ax_hist)

        ax_box.set(yticks=[])
        sns.despine(ax=ax_hist)
        sns.despine(ax=ax_box, left=True)

plot_factor(data, ticker, 'PEValue')        

import pandas as pd
df_index = get_index_data(u'000001', u"20080101", datetime.now())
df_index = df_index.set_index(pd.DatetimeIndex(df_index['tradeDate']))
s_index = df_index['closeIndex']

s_index_ret = s_index / s_index.shift() - 1.0
s_index_ret = s_index_ret.dropna()
s_index_ret.tail()


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(111)
signal = s_index_ret.rolling(window=50).std()**2 - s_index_ret.rolling(window=50).mean()
ax1.plot(signal.rolling(window=200).mean())
ax1.plot(signal.rolling(window=200).mean()+1*signal.rolling(window=200).std())
ax1.plot(signal.rolling(window=200).mean()-1*signal.rolling(window=200).std())
signal.plot()
s_index.plot(secondary_y=True, style='k', legend=True)
# signal 变大
# 1. 波动变大
# 2. u为负

# signal 变小
# 1. 波动变小
# 2. u小，或者猛涨

def plot_index(data, ticker, factor):
    for index, row in ticker.iterrows():
        df_factor = data[(data['ticker']==row['ticker'])][factor]
        df_factor = df_factor.dropna()
        df_index = get_index_data(row['ticker'], df_factor.index[0], datetime.now())
#        fig, axes = plt.subplots(1, 1, figsize=(10, 7))
        a = str.decode(row['secName'], 'utf-8')
#        axes.set_title(a)
        df_index = df_index.set_index(pd.DatetimeIndex(df_index['tradeDate']))
        s_index = df_index['closeIndex']
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(111)
        ax1.set_title(a)
        s_index_ret = s_index / s_index.shift() - 1.0
        s_index_ret = s_index_ret.dropna()
        signal = s_index_ret.rolling(window=50).std()**2 - s_index_ret.rolling(window=50).mean()

        ax1.plot(signal.rolling(window=200).mean())
        ax1.plot(signal.rolling(window=200).mean()+1*signal.rolling(window=200).std())
        ax1.plot(signal.rolling(window=200).mean()-1*signal.rolling(window=200).std())
        signal.plot()
        s_index.plot(secondary_y=True, style='k', legend=True)

plot_index(data, ticker, 'PEValue')     

