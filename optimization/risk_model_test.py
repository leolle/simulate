# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import pandas as pd
import logging

import re
import cvxpy as cvx

from lib.gftTools import gftIO
# from lib.gftTools.RiskModel import RiskAnlysis, FactorExposure


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not handler:
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.debug('start')
risk_model = gftIO.zload("/home/weiwu/share/optimize/x2.pkl")
asset_weight = gftIO.zload("/home/weiwu/share/optimize/asset_weights.pkl")
begin_date = gftIO.zload("/home/weiwu/share/optimize/begin_date.pkl")
end_date = gftIO.zload("/home/weiwu/share/optimize/end_date.pkl")
frequency = gftIO.zload("/home/weiwu/share/optimize/frequency.pkl")
factors = gftIO.zload("/home/weiwu/share/optimize//factors.pkl")
exposure_constraint = gftIO.zload("/home/weiwu/share/optimize/\
exposure_constraint.pkl")

logger.debug('data loaded')

if isinstance(exposure_constraint, gftIO.GftTable):
    exposure_constraint = exposure_constraint.asColumnTab()
    exposure_constraint = exposure_constraint.pivot(index='idname',
                                                    columns='variable',
                                                    values='value')


# class ExtractDictModelData(object):
#     """ model data extraction and getting attribute. """
#     def __init__(self, model):
#         self.model = model

#     def get_input_factor(self, oset_idx):
#         """ Get oset idx from risk model.
#         Keyword Arguments:
#         oset_idx: list of oset gid
#         """
#         if len(oset_idx) < 1:
#             return None

#         factor_data = {key: self.model.get(key, None).asMatrix() for key in oset_idx}

#         return factor_data

#     def get_output(self, post_fix, oset_idx=None):
#         """ get target data from model

#         Keyword Arguments:
#         oset_idx: list of oset gid
#         poset_fix: 'specificiRisk', 'ret_cov', '*.ret'
#         """
#         if oset_idx is None:
#             return self.model.get(post_fix, None)
#         else:
#             factors_output = pd.DataFrame(
#                 index=self.model[oset_idx[0]+post_fix].index, columns=oset_idx)
#             for value in oset_idx:
#                 factors_output[value] = self.model[value+post_fix]
#             factors_output.columns = gftIO.strSet2Np(
#                 factors_output.columns.values)
#             return factors_output


# class RiskAnlysis(object):
#     """ risk data preparation and getting attribute. """
#     def __init__(self, risk_model_merge):
#         self.risk_model = risk_model_merge
#         self.ls_factors = [x for x in risk_model_merge.keys() if
#                            re.match("[A-Z0-9]{32}$", x)]
#         self.ls_factors_ret = [x[:-4] for x in risk_model_merge.keys() if
#                                re.search(".ret$", x)]

#     def get_factor_exposure(self, factor_list, date, symbols):
#         ''' Return factor exposure matrix(big X).
#         Parameters
#         ----------
#         risk_model: dictionary
#             Including specific risk, different factor exposure dataframe for
#             all symbols.
#         factor_list: list
#             Factor exposure list.
#         Returns
#         -------
#         factor_exposure: DataFrame
#             Big X on target date for input symbols.
#         '''
#         factor_exposure = pd.DataFrame(index=symbols)
#         for factor in factor_list:
#             try:
#                 factor_exposure[factor] = self.risk_model[factor].asMatrix().\
#                                           loc[date, symbols]
#             except KeyError:
#                 factor_exposure[factor] = np.nan
#                 # raise KeyError('invalid input date: %s' % date)
#         factor_exposure.columns = gftIO.strSet2Np(factor_exposure.columns.
#                                                   values)
#         factor_exposure = factor_exposure.fillna(0)

#         return factor_exposure

#     def factor_exposure(self, asset_weight, frequency, factors):
#         """ Get factor exposure.
#         Keyword Arguments:
#         asset_weight --
#         """
#         if isinstance(factors, dict):
#             ls_factors = factors['factors']
#         if isinstance(asset_weight, gftIO.GftTable):
#             asset_weight = asset_weight.asMatrix()
#             # asset_weight.fillna('ffill', inplace=True)
#         # resample asset weight by monthly
#         if frequency == 'MONTHLY':
#             asset_weight.index.name = 'index'
#             m = asset_weight.index.to_period('m')
#             asset_weight = asset_weight.reset_index().groupby(m).last().set_index('index')
#             asset_weight.index.name = ''

#         asset_factor_exposure = pd.Panel({target_date: self.get_factor_exposure(
#             ls_factors, target_date, asset_weight.columns).T for target_date in asset_weight.index})

#         return asset_factor_exposure

#     def cal_factor_exposure(self, asset_weight, frequency, factors):
#         """ calculate asset factor exposure
#         F = w^T * exposure
#         Keyword Arguments:
#         asset_weight -- 
#         frequency    -- 
#         factors      -- 
#         """
#         pn_factor_exposure = self.factor_exposure(asset_weight, frequency, factors)
#         factor_exposure = pd.DataFrame(index=asset_weight.index,
#                                        columns=pn_factor_exposure.major_axis)
#         for target_date in asset_weight.index:
#             factor_exposure.ix[target_date] = pn_factor_exposure.ix[target_date].dot(
#                 asset_weight.ix[target_date].fillna(0))
#         return factor_exposure.replace(0, np.nan).fillna(method='ffill')

#     def specific_risk(self):
#         """ get specific risk from risk model
#         Keyword Arguments:
#         date --
#         symbols --
#         """
#         # find the nearest date next to target date from specific risk
#         specific_risk = self.risk_model['specificRisk'].pivot(
#             index='date', columns='symbol', values='specificrisk')

#         return specific_risk

#     def delta(self, date, symbols):
#         """
#         create delta matrix from specific risk
#         Keyword Arguments:
#         date    -- 
#         symbols -- 
#         """
#         # diag = self.specific_risk()
#         risk = self.specific_risk()
#         diag = risk.loc[date, symbols]
#         delta = pd.DataFrame(np.diag(diag), index=diag.index,
#                              columns=diag.index).fillna(0)

#         return delta

#     def covariance_matrix(self, date, factors):
#         """
#         Keyword Arguments:
#         date    -- 
#         factors -- 
#         """
#         pass

#     def factor_return(self):
#         """ get factor return from all the factors """
#         factors_ret = pd.DataFrame(index=self.risk_model[self.ls_factors_ret[0]+'.ret'].index,
#                                    columns=self.ls_factors_ret)
#         for factor in self.ls_factors_ret:
#             factors_ret[factor] = self.risk_model[factor]
#         factors_ret.columns = gftIO.strSet2Np(factors_ret.columns.
#                                               values)
#         return factors_ret


class ExtractDictModelData(object):
    """ model data extraction and getting attribute. """
    def __init__(self, model):
        self.model = model

    def get_input_factor(self, oset_idx):
        """ Get oset idx from risk model.
        Keyword Arguments:
        oset_idx: list of oset gid
        """
        if len(oset_idx) < 1:
            return None
        date_index = self.model.get(oset_idx[0], None).asMatrix().index
        ls_factor_b_char = gftIO.strSet2Np(np.array(oset_idx))
        factor_data = pd.Panel({ls_factor_b_char[key]: self.model.get(factor).asMatrix() for key, factor in enumerate(oset_idx)})
        return factor_data.transpose(1, 2 ,0)


    def get_output(self, post_fix, oset_idx=None):
        """ get target data from model

        Keyword Arguments:
        oset_idx: list of oset gid
        poset_fix: 'specificRisk', 'ret_cov', '*.ret'
        """
        if oset_idx is None:
            return self.model.get(post_fix, None)
        else:
            factors_output = pd.DataFrame(
                index=self.model[oset_idx[0]+post_fix].index, columns=oset_idx)
            for value in oset_idx:
                factors_output[value] = self.model[value+post_fix]
            factors_output.columns = gftIO.strSet2Np(
                factors_output.columns.values)
            return factors_output


ls_style_factor = ['873CF94D09229206D675ACC32328DC24',
                   '441E8F64A7582F553BCBE42A216285F1',
                   'B9CCDA635F039E84D489F964DB08BC5C',
                   'B1B02CFAB81248BAA87754E760769BD2',
                   'EB8553C313C38BC180E1972D798622BA',
                   'BE3E35A7C0CB49EB9E1CB41D566563E7',
                   'ABEC912F31E326F4C1FC507AF787C8FA',
                   'E8D70EA915C420F9B9005BB21540788C']

ModelData = ExtractDictModelData(risk_model)
test = ModelData.get_input_factor(ls_style_factor)
#RiskModel = RiskAnlysis(risk_model)
# RiskModel.factor_exposure(asset_weight, frequency, factors)
# RiskModel.factor_return()

# specific_risk = risk_model['specificRisk'].pivot(
#     index='date', columns='symbol', values='specificrisk')
# date_index = specific_risk.index
# ls_symbols = specific_risk.columns

