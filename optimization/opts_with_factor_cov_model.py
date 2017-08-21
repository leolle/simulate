# -*- coding: utf-8 -*-
import logging

import numpy as np
import pandas as pd
import re
import cvxpy as cvx

from cvxopt import matrix, spmatrix

from lib.gftTools import gsConst, gftIO


class RiskAnlysis(object):
    """ risk data preparation and getting attribute. """
    def __init__(self, risk_model_merge):
        self.risk_model = risk_model_merge
        self.ls_factors = [x for x in risk_model_merge.keys() if
                           re.match("[A-Z0-9]{32}$", x)]
        self.ls_factors_ret = [x[:-4] for x in risk_model_merge.keys() if
                               re.search(".ret$", x)]

    def get_factor_exposure(self, factor_list, date, symbols):
        ''' Return factor exposure matrix(big X).
        Parameters
        ----------
        risk_model: dictionary
            Including specific risk, different factor exposure dataframe for
            all symbols.
        factor_list: list
            Factor exposure list.
        Returns
        -------
        factor_exposure: DataFrame
            Big X on target date for input symbols.
        '''
        factor_exposure = pd.DataFrame(index=symbols)
        for factor in factor_list:
            try:
                factor_exposure[factor] = self.risk_model[factor].asMatrix().\
                                          loc[date, symbols]
            except KeyError:
                factor_exposure[factor] = np.nan
                # raise KeyError('invalid input date: %s' % date)
        factor_exposure.columns = gftIO.strSet2Np(factor_exposure.columns.
                                                  values)
        factor_exposure = factor_exposure.fillna(0)

        return factor_exposure

    def factor_exposure(self, asset_weight, frequency, factors):
        """ Get factor exposure.
        Keyword Arguments:
        asset_weight --
        """
        if isinstance(factors, dict):
            ls_factors = factors['factors']
        if isinstance(asset_weight, gftIO.GftTable):
            asset_weight = asset_weight.asMatrix()
            # asset_weight.fillna('ffill', inplace=True)
        # resample asset weight by monthly
        if frequency == 'MONTHLY':
            asset_weight.index.name = 'index'
            m = asset_weight.index.to_period('m')
            asset_weight = asset_weight.reset_index().groupby(m).last().set_index('index')
            asset_weight.index.name = ''

        asset_factor_exposure = pd.Panel({target_date: self.get_factor_exposure(
            ls_factors, target_date, asset_weight.columns).T for target_date in asset_weight.index})

        return asset_factor_exposure

    def cal_factor_exposure(self, asset_weight, frequency, factors):
        """ calculate asset factor exposure
        F = w^T * exposure
        Keyword Arguments:
        asset_weight -- 
        frequency    -- 
        factors      -- 
        """
        pn_factor_exposure = self.factor_exposure(asset_weight, frequency, factors)
        factor_exposure = pd.DataFrame(index=asset_weight.index,
                                       columns=pn_factor_exposure.major_axis)
        for target_date in asset_weight.index:
            factor_exposure.ix[target_date] = pn_factor_exposure.ix[target_date].dot(
                asset_weight.ix[target_date].fillna(0))
        return factor_exposure.replace(0, np.nan).fillna(method='ffill')

    def specific_risk(self):
        """ get specific risk from risk model
        Keyword Arguments:
        date --
        symbols --
        """
        # find the nearest date next to target date from specific risk
        specific_risk = self.risk_model['specificRisk'].pivot(
            index='date', columns='symbol', values='specificrisk')

        return specific_risk

    def delta(self, date, symbols):
        """
        create delta matrix from specific risk
        Keyword Arguments:
        date    -- 
        symbols -- 
        """
        # diag = self.specific_risk()
        risk = self.specific_risk()
        diag = risk.loc[date, symbols]
        delta = pd.DataFrame(np.diag(diag), index=diag.index,
                             columns=diag.index).fillna(0)

        return delta

    def covariance_matrix(self, date, factors):
        """
        Keyword Arguments:
        date    -- 
        factors -- 
        """
        factors = gftIO.strSet2Np(np.array(factors))
        cov_matrix = self.risk_model['ret_cov'].set_index('date')
        cov_matrix = cov_matrix.loc[date]
        cov_matrix = cov_matrix.pivot(index='factorid1',
                                      columns='factorid2',
                                      values='value')
        cov_matrix = cov_matrix.reindex(factors, factors,
                                        fill_value=np.nan)

        return cov_matrix

    def factor_return(self):
        """ get factor return from all the factors """
        factors_ret = pd.DataFrame(index=self.risk_model[self.ls_factors_ret[0]+'.ret'].index,
                                   columns=self.ls_factors_ret)
        for factor in self.ls_factors_ret:
            factors_ret[factor] = self.risk_model[factor]
        factors_ret.columns = gftIO.strSet2Np(factors_ret.columns.
                                              values)
        return factors_ret


def log_ret(rets):
    """Log of return relatives, ln(1+r), for a given DataFrame rets."""
    return np.log(rets + 1)


def find_nearest(array, value):
    """ Find the nearest value index from an array"""
    if isinstance(array, list):
        array = np.array(array)
    idx = (np.abs(array-value)).argmin()
    return idx


def set_upper_limit(x, multiplier=0.1):
    if x >= 0:
        return x*(1 + multiplier)
    else:
        return x*(1 - multiplier)


def set_lower_limit(x, multiplier=0.1):
    if x >= 0:
        return x*(1 - multiplier)
    else:
        return x*(1 + multiplier)

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

x0 = gftIO.zload("/home/weiwu/share/optimize/x0.pkl")
x1 = gftIO.zload("/home/weiwu/share/optimize/x1.pkl")
x2 = gftIO.zload("/home/weiwu/share/optimize/x2.pkl")
x3 = gftIO.zload("/home/weiwu/share/optimize/x3.pkl")
x4 = gftIO.zload("/home/weiwu/share/optimize/x4.pkl")
x5 = gftIO.zload("/home/weiwu/share/optimize/x5.pkl")
x6 = gftIO.zload("/home/weiwu/share/optimize/x6.pkl")
x7 = gftIO.zload("/home/weiwu/share/optimize//x7.pkl")
x8 = gftIO.zload("/home/weiwu/share/optimize//x8.pkl")
x9 = gftIO.zload("/home/weiwu/share/optimize//x9.pkl")
x10 = gftIO.zload("/home/weiwu/share/optimize/exposure_constraint.pkl")

target_mode = x0
position_limit = x1
risk_model = x2
asset_return = x3
asset_weight = x4
target_risk = x5
target_return = x6
target_date = x7
asset_constraint = x8
group_constraint = x9
exposure_constraint = x10

asset_return = x3.asMatrix()
asset_weight = x4.asColumnTab()
if isinstance(exposure_constraint, gftIO.GftTable):
    factor_exposure_constraint = exposure_constraint.asColumnTab()
    factor_exposure_constraint = factor_exposure_constraint.pivot(
        index='idname', columns='variable', values='value')
# 数据格式转换
if isinstance(asset_constraint, gftIO.GftTable):
    asset_constraint = asset_constraint.asMatrix()
if isinstance(group_constraint, gftIO.GftTable):
    group_constraint = group_constraint.asMatrix()

logger.debug('parse data finished!')
logger.debug('asset return number: %s', asset_return.shape[1])
logger.debug('asset weight number: %s', asset_weight.shape[0])
logger.debug('parse data finished.')

RiskModel = RiskAnlysis(risk_model)

target_date = factor_exposure_constraint.index[0]
specific_risk = RiskModel.specific_risk()
df_industries_asset_weight = asset_weight.drop_duplicates(
    subset=['date', 'symbol'])

# drop incomplete rows
df_industries_asset_weight = df_industries_asset_weight.dropna(
    axis=0, subset=['industry', 'symbol'], how='any')

unique_symbol = df_industries_asset_weight['symbol'].unique()

try:
    if np.setdiff1d(unique_symbol, specific_risk.loc[target_date].index):
        pass
except ValueError:
    logger.debug('some symbols are not in asset weight symbols.')

df_pivot_industries_asset_weights = pd.pivot_table(
        df_industries_asset_weight, values='value', index=['date'],
        columns=['industry', 'symbol'])


# pivot the original dataframe to multi-index dataframe
# level 0 value: industry
# level 1 value: assets, the order of assets are changed.
df_pivot_industries_asset_weights = df_pivot_industries_asset_weights.fillna(0)
idx_level_0_value = df_pivot_industries_asset_weights.columns.get_level_values(0)
idx_level_0_value = idx_level_0_value.drop_duplicates()
idx_level_1_value = df_pivot_industries_asset_weights.columns.get_level_values(1)

asset_expected_return = asset_return.loc[:target_date, idx_level_1_value].fillna(0)
rets_mean = log_ret(asset_expected_return).mean()
delta = RiskModel.delta(target_date, idx_level_1_value)

all_factors = ['018D158ACE908165ED4ECEA129ABF547',
               '22895BDA3790CB3FA9FABD1E2FF883CE',
               '265ED5755EEC25762A1293A398C40F5D',
               '26BF3A3BD0BB76D58CC560B72527AB14',
               '26C8DB23CDCCD1E7D858D3340449C854',
               '2AECD1DE9CC1EAA3982AAD231FA0AE39',
               '324F4A27A88603BE2C53D5AFA7745584',
               '35555851DC92592BF9A5D55906D10A2B',
               '3995B9BA60E9943BF39F784E2E6B451D',
               '4972C5017D9B1D2AEDA81920FA21D3BF',
               '54899D64BCFE8C3BA11BB84429A46C52',
               '62C09CCF341CA27BCE7835103FE32E97',
               '767FA9609E0C0B661EE557AB80C6F8E4',
               '7F216CE9D40D23FEDC54147E9D754DEE',
               '90F09502030C80B91B02A2CE72A5A0EB',
               '91B8D8D34EE8AA2BD9B60F9D223C8707',
               'A528F7BFA538A7675484FC2A72021138',
               'A7638B5C794E97F887D2522215A3DFF8',
               'B3E67D32A2F6428BB06D1BFCD2547798',
               'BB81923B85135A411E8BDD80015E6512',
               'BBA8CAB7DE9857EE8B8AFD9A23E3D34A',
               'BED9C798EDEE97DC1712BB3E05FC8113',
               'C585E113DD45D2344C97D3D2967D536F',
               'D753394CA4D5A4CFC25BB5DACA7293F3',
               'D84A4B250D077E0EFC4F7FFF393FCB44',
               'EB34F4D5C92E85C2307DB2C8015C94F1',
               'EFE87574B86C774ADFD08F421AF5D11B',
               '441E8F64A7582F553BCBE42A216285F1',
               '873CF94D09229206D675ACC32328DC24',
               'ABEC912F31E326F4C1FC507AF787C8FA',
               'B1B02CFAB81248BAA87754E760769BD2',
               'B9CCDA635F039E84D489F964DB08BC5C',
               'BE3E35A7C0CB49EB9E1CB41D566563E7',
               'E8D70EA915C420F9B9005BB21540788C',
               'EB8553C313C38BC180E1972D798622BA']

ls_style_factor = ['873CF94D09229206D675ACC32328DC24',
                   '441E8F64A7582F553BCBE42A216285F1',
                   'B9CCDA635F039E84D489F964DB08BC5C',
                   'B1B02CFAB81248BAA87754E760769BD2',
                   'EB8553C313C38BC180E1972D798622BA',
                   'BE3E35A7C0CB49EB9E1CB41D566563E7',
                   'ABEC912F31E326F4C1FC507AF787C8FA',
                   'E8D70EA915C420F9B9005BB21540788C']

big_X = RiskModel.get_factor_exposure(all_factors, target_date, unique_symbol)
covariance_matrix = RiskModel.covariance_matrix(date=target_date, factors=all_factors)
# set boundary vector for h
df_asset_weight = pd.DataFrame({'lower': [0.0], 'upper': [1.0]},
                               index=idx_level_1_value)
df_group_weight = pd.DataFrame({'lower': [0.0], 'upper': [1.0]},
                               index=idx_level_0_value)


df_factor_exposure_bound = pd.DataFrame(index=factor_exposure_constraint.columns, columns=[['lower', 'upper']])
df_factor_exposure_bound.lower = factor_exposure_constraint.ix[-1].apply(lambda x: set_lower_limit(x))
df_factor_exposure_bound.upper = factor_exposure_constraint.ix[-1].apply(lambda x: set_upper_limit(x))

df_factor_exposure_lower_bnd = pd.DataFrame(data=[[big_X.values.min()]]*len(all_factors), index=big_X.columns)
df_factor_exposure_upper_bnd = pd.DataFrame(data=[[big_X.values.max()]]*len(all_factors), index=big_X.columns)

df_factor_exposure_lower_bnd.ix[df_factor_exposure_bound.index] = df_factor_exposure_bound.lower.values.reshape((len(df_factor_exposure_bound),1))
df_factor_exposure_upper_bnd.ix[df_factor_exposure_bound.index] = df_factor_exposure_bound.upper.values.reshape((len(df_factor_exposure_bound),1))

df_factor_exposure_bound = pd.DataFrame(index=big_X.columns, columns=[['lower', 'upper']])
df_factor_exposure_bound.lower = [big_X.values.min()]*len(all_factors)
df_factor_exposure_bound.upper = [big_X.values.max()]*len(all_factors)
df_factor_exposure_bound.loc[factor_exposure_constraint.columns, 'lower'] = factor_exposure_constraint.ix[-1].apply(lambda x: set_lower_limit(x))
df_factor_exposure_bound.loc[factor_exposure_constraint.columns, 'upper'] = factor_exposure_constraint.ix[-1].apply(lambda x: set_upper_limit(x))


noa = len(unique_symbol)

target_mode = 0
# Factor model portfolio optimization.
w = cvx.Variable(noa)
f = big_X.T.values*w
# f = big_X.loc[:,exposure_constraint.columns].T.values*w
gamma = cvx.Parameter(sign='positive')
Lmax = cvx.Parameter()
ret = w.T * rets_mean.values
risk = cvx.quad_form(f, covariance_matrix.values) + cvx.quad_form(w, delta.values)
# for group weight constraint
groups = df_pivot_industries_asset_weights.groupby(
    axis=1, level=0, sort=False, group_keys=False).count().\
    iloc[-1, :].values
num_group = len(groups)
num_asset = np.sum(groups)
logger.debug('number of assets in groups: %s', groups)
logger.debug('number of groups: %s', num_group)

G_sparse_list = []
for i in range(num_group):
    for j in range(groups[i]):
        G_sparse_list.append(i)
Group_sub = spmatrix(1.0, G_sparse_list, range(num_asset))

G = np.array(Group_sub)
G_sum = np.array(matrix(Group_sub))*w

eq_constraint = [cvx.sum_entries(w) == 1,
                 cvx.norm(w, 1) <= Lmax]


def constraint(obj, df_limit):
    if isinstance(df_limit, pd.DataFrame):
        return [obj >= df_limit.iloc[:, 0].values,
                obj <= df_limit.iloc[:, 1].values]


asset_weight_constraint = constraint(w, df_asset_weight)
group_weight_constraint = constraint(G_sum, df_group_weight)
if exposure_constraint:
    factor_exposure_constraint = constraint(f, df_factor_exposure_bound)
else:
    factor_exposure_constraint = None

l_eq_constraint = [w >= df_asset_weight.lower.values,
                   w <= df_asset_weight.upper.values,
                   G_sum >= df_group_weight.lower.values,
                   G_sum <= df_group_weight.upper.values]
if exposure_constraint:
    l_eq_constraint.append(f >= df_factor_exposure_lower_bnd.values)
    l_eq_constraint.append(f <= df_factor_exposure_upper_bnd.values)
target_return = -0.0006992348944336906
# leverage level and risk adjusted parameter
Lmax.value = 1
gamma.value = 1
# target_mode = 0
if target_mode == gsConst.Const.MinimumRisk:
    # maximize negative product of gamma and risk
    prob_factor = cvx.Problem(cvx.Maximize(-gamma*risk),
                              eq_constraint + asset_weight_constraint +
                              group_weight_constraint +
                              factor_exposure_constraint)
if target_mode == gsConst.Const.MinimumRiskUnderReturn:
    # minimum risk subject to target return, Markowitz Mean_Variance Portfolio
    prob_factor = cvx.Problem(cvx.Maximize(-gamma*risk),
                              [ret >= target_return]+eq_constraint+l_eq_constraint)
if target_mode == gsConst.Const.MaximumReturnUnderRisk:
    # Portfolio optimization with a leverage limit and a bound on risk
    prob_factor = cvx.Problem(cvx.Maximize(ret),
                              [risk <= target_risk]+eq_constraint+l_eq_constraint)
prob_factor.solve(verbose=True)
df_opts_weight = pd.DataFrame(np.array(w.value).T,
                              columns=idx_level_1_value,
                              index=[target_date])
dict_opts_status = {}
dict_opts_status[target_date] = prob_factor.status

logger.debug(prob_factor.status)
# logger.debug("target return: %s", target_return)
# logger.debug("all weight are bigger than 0? %s",
#              (df_opts_weight > 0).all().all())
# logger.debug("all weight are smaller than 1? %s",
#              (df_opts_weight <= 1).all().all())
# logger.debug("weight sum smaller than 0: %s",
#              df_opts_weight[df_opts_weight < 0].sum(1))
# logger.debug(df_opts_weight)


# target_mode = 2
# # Factor model portfolio optimization.
# w = cvx.Variable(noa)
# f = big_X.T.values*w
# # f = big_X.loc[:,exposure_constraint.columns].T.values*w
# gamma = cvx.Parameter(sign='positive')
# Lmax = cvx.Parameter()
# ret = w.T * rets_mean.values
# # risk = cvx.quad_form(f, cov_matrix.loc[exposure_constraint.columns,exposure_constraint.columns].values) + cvx.quad_form(w, delta.values)
# risk = cvx.quad_form(f, cov_matrix.values) + cvx.quad_form(w, delta.values)
# #ret = mu.T*w
# # for group weight constraint
# groups = df_pivot_industries_asset_weights.groupby(
#     axis=1, level=0, sort=False, group_keys=False).count().\
#     iloc[-1, :].values
# num_group = len(groups)
# num_asset = np.sum(groups)
# logger.debug('number of assets in groups: %s', groups)
# logger.debug('number of groups: %s', num_group)

# G_sparse_list = []
# for i in range(num_group):
#     for j in range(groups[i]):
#         G_sparse_list.append(i)
# Group_sub = spmatrix(1.0, G_sparse_list, range(num_asset))

# G = np.array(Group_sub)
# G_sum = np.array(matrix(Group_sub))*w

# # f == exposure_constraint.ix[-1].values.reshape(-1, 1)

# eq_constraint = [cvx.sum_entries(w) == 1,
#                  cvx.norm(w, 1) <= Lmax]
# l_eq_constraint = [w >= df_asset_weight.lower.values,
#                    w <= df_asset_weight.upper.values,
#                    G_sum >= df_group_weight.lower.values,
#                    G_sum <= df_group_weight.upper.values]
# if exposure_constraint is not None:
#     l_eq_constraint.append(f >= df_factor_exposure_lower_bnd.values)
#     l_eq_constraint.append(f <= df_factor_exposure_upper_bnd.values)
# target_return = -0.0006992348944336906
# Lmax.value = 1
# gamma.value = 1
# if target_mode == gsConst.Const.MinimumRisk:
#     # Solve the factor model problem.
#     prob_factor = cvx.Problem(cvx.Maximize(-gamma*risk),
#                               l_eq_constraint)
# if target_mode == gsConst.Const.MinimumRiskUnderReturn:
#     # minimum risk subject to target return, Markowitz Mean_Variance Portfolio
#     prob_factor = cvx.Problem(cvx.Maximize(-gamma*risk),
#                               [ret >= target_return]+l_eq_constraint)
# if target_mode == gsConst.Const.MaximumReturnUnderRisk:
#     # Portfolio optimization with a leverage limit and a bound on risk
#     prob_factor = cvx.Problem(cvx.Maximize(ret),
#                               [risk <= target_risk]+l_eq_constraint)
# prob_factor.solve(verbose=True)
# df_opts_weight = pd.DataFrame(np.array(w.value).T,
#                               columns=idx_level_1_value,
#                               index=[target_date])
# logger.debug(prob_factor.status)
# logger.debug("target return: %s", target_return)
# logger.debug("all weight are bigger than 0? %s",
#              (df_opts_weight > 0).all().all())
# logger.debug("all weight are smaller than 1? %s",
#              (df_opts_weight <= 1).all().all())
# logger.debug("weight sum smaller than 0: %s",
#              df_opts_weight[df_opts_weight < 0].sum(1))
# logger.debug(df_opts_weight)
# handler.close()
logger.removeHandler(handler)
while logger.handlers:
     logger.handlers.pop()
