# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import pandas as pd
import logging

import re
import cvxpy as cvx

from lib.gftTools import gftIO
#from lib.gftTools.RiskModel import RiskAnlysis, FactorExposure


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.debug('start')
risk_model = gftIO.zload("/home/weiwu/share/optimize/x2.pkl")
asset_weight = gftIO.zload("/home/weiwu/share/optimize/asset_weights.pkl")
begin_date = gftIO.zload("/home/weiwu/share/optimize/begin_date.pkl")
end_date = gftIO.zload("/home/weiwu/share/optimize/end_date.pkl")
frequency = gftIO.zload("/home/weiwu/share/optimize/frequency.pkl")
factors = gftIO.zload("/home/weiwu/share/optimize//factors.pkl")

logger.debug('data loaded')


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

        factor_data = {key: self.model.get(key, None).asMatrix() for key in oset_idx}

        return factor_data

    def get_output(self, post_fix, oset_idx=None):
        """ get target data from model

        Keyword Arguments:
        oset_idx: list of oset gid
        poset_fix: 'specificiRisk', 'ret_cov', '*.ret'
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

RiskModel = ExtractDictModelData(risk_model)
    # def covariance_matrix(self, date, factors):
    #     """
    #     Keyword Arguments:
    #     date    -- 
    #     factors -- 
    #     """
    #     pass

    # def factor_return(self):
    #     """ get factor return from all the factors """
    #     ls_factor_ret = [x+'.ret' for x in factors]
    #     factors_ret = pd.DataFrame(index=self.risk_model[factors[0]+'.ret'].index,
    #                                columns=factors)
    #     for factor in ls_factor_ret:
    #         factors_ret[factor] = self.risk_model[factor]
    #     factors_ret.columns = gftIO.strSet2Np(factors_ret.columns.
    #                                           values)
    #     return factors_ret

# RiskModel.factor_exposure(asset_weight, frequency, factors)
# RiskModel.factor_return()

# specific_risk = risk_model['specificRisk'].pivot(
#         index='date', columns='symbol', values='specificrisk')
# date_index = specific_risk.index
# ls_symbols = specific_risk.columns

# delta = pd.Panel({date: pd.DataFrame(np.diag(specific_risk.loc[date]),
#                                      index=specific_risk.loc[date].index,
#                                      columns=specific_risk.loc[date].index).fillna(0)
#                   for date in date_index})
