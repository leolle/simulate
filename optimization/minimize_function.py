# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.optimize as opt


""" 模拟输入信息 """
dates = pd.date_range('1/1/2000', periods=8)
dt_start = dates[0]
dt_end = dates[-1]
industry = ['industry', 'industry', 'utility', 'utility', 'consumer']
symbols = ['A', 'B', 'C', 'D', 'E']  # including Cash
zipped = list(zip(industry, symbols))
index = pd.MultiIndex.from_tuples(zipped)

          
noa = len(symbols)
weights = np.random.random(noa)
weights /= np.sum(weights)

data = np.array([[10, 9, 10, 11, 12, 13, 14, 13],
                 [11, 11, 10, 11, 11, 12, 11, 10],
                 [10, 11, 10, 11, 12, 13, 14, 13],
                 [11, 11, 10, 11, 11, 12, 11, 11],
                 [10, 11, 10, 11, 12, 13, 14, 13]])

market_to_market_price = pd.DataFrame(data.T, index=dates, columns=index)
execute_price = pd.DataFrame(index=dates, columns=symbols)
All = slice(None)
market_to_market_price.plot()
rets = market_to_market_price / market_to_market_price.shift(1) - 1.0
rets = rets.fillna(0)
target_portfolio_wgt = pd.DataFrame()
for i in range(8):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    target_portfolio_wgt = target_portfolio_wgt.append(pd.DataFrame(weights).T)
target_portfolio_wgt.index = dates
target_portfolio_wgt.columns = index

expo_factor = np.ones((5,5))
factor_covariance = market_to_market_price.cov()
delta = np.diagflat([0.088024, 0.082614, 0.084237, 0.074648,
                                 0.084237])
cov_matrix = np.dot(np.dot(expo_factor, factor_covariance),
                            expo_factor.T) + delta
                    
def calculate_total_risk(weights, cov_matrix):
    port_var = np.dot(np.dot(weights.T, cov_matrix), weights)
    return port_var
    


PortValue = 100000

model = pd.DataFrame(np.array([.08,.12,.05,.05,.65]), index= index, columns = ['strategic'])
model['tactical'] = [(.05,.41), (.2,.66), (0,.16), (0,.76), (0,.11)]


def fitness(W, Rt):
    port_Rt = np.dot(Rt, W)
    port_rt = np.log(1 + port_Rt)
    q95 = pd.Series(port_rt).quantile(.05)
    cVaR = (port_rt[port_rt < q95] * np.sqrt(20)).mean() * PortValue
    mean_cVaR = (PortValue * (port_rt.mean() * 20)) / cVaR
    return -1 * mean_cVaR


#def solve_weights(Rt, b_= None):
#    if b_ is None:
#       b_ = [(0.0, 1.0) for i in range(noa)]
#    W = np.ones(noa)/noa
#    c_ = ({'type':'eq', 'fun': lambda W: sum(W) - 1})
#    optimized = opt.minimize(fitness, W, args=[Rt], method='SLSQP', constraints=c_, bounds=b_)
#
#    if not optimized.success: 
#        raise ValueError(optimized.message)
#    return optimized.x  # Return optimized weights

port_rets = rets.ix[-1].copy()
risk_tgt = 0.3

def solve_weights(Rt, b_ = None):

    W = np.ones(noa) / noa
    if  b_ is None:
        b_ = [(0.01, 1.0) for i in range(noa)]
        c_ = ({'type':'eq', 'fun': lambda W: sum(W) - 1})
    else:
        covar = Rt.cov()
        c_ = ({'type':'eq', 'fun': lambda W: sum(W) - 1},
              {'type':'eq', 'fun': lambda W: np.sqrt(np.dot(W, np.dot(covar, W)) * 252) - risk_tgt})

    optimized = opt.minimize(fitness, W, args = [Rt], method='SLSQP', constraints=c_, bounds=b_)  

    if not optimized.success: 
        raise ValueError(optimized.message)

    return optimized['x']  # Return optimized weights

class_cont = rets.ix[0].copy()
class_cont.ix[:] = np.around(np.hstack(rets.groupby(axis=0, level=0).apply(solve_weights).values),3)
scalars = class_cont.groupby(level=0).sum()
scalars.ix[:] = np.around(solve_weights((class_cont * port_rets).groupby(level=0, axis=1).sum(), list(model['tactical'].values)),3)

class_cont.groupby(level=0).transform(lambda x: x * scalars[x.name])



port = np.dot(port_rets.groupby(level=0, axis=1).agg(lambda x: np.dot(x,solve_weights(x))),\
solve_weights(port_rets.groupby(level=0, axis=1).agg(lambda x: np.dot(x,solve_weights(x))), \
list(model['tactical'].values)))

pd.Series(port, name='portfolio').cumsum().plot()
