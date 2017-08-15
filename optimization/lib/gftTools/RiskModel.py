# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import pandas as pd
import logging

import re
import cvxpy as cvx

from lib.gftTools import gftIO


class RiskAnlysis(object):
    """ risk data preparation and getting attribute. """
    def __init__(self, risk_model_merge):
        self.risk_model = risk_model_merge

    def get_factor_exposure(self, date, frequency, symbols, factors):
        """ Return factor exposure matrix(big X).

        Keyword Arguments:
        date       --
        frequency  --
        factors    --

        Returns
        -------
        factor_exposure: DataFrame
            Big X on target date for input symbols.
        """
        factor_exposure = pd.DataFrame(index=symbols)
        for factor in factors:
            try:
                factor_exposure[factor] = self.risk_model[factor].asMatrix().\
                                          loc[date, symbols]
            except KeyError:
                factor_exposure[factor] = np.nan

        factor_exposure.columns = gftIO.strSet2Np(factor_exposure.columns.
                                                  values)
        factor_exposure = factor_exposure.fillna(0)

        return factor_exposure

    def factor_exposure(self, asset_weight, frequency, factors):
        """ Get assets factor exposure.
        F = w^T * exposure
        Keyword Arguments:
        asset_weight -- 
        """
        if isinstance(asset_weight, gftIO.GftTable):
            asset_weight = asset_weight.asMatrix()

        # resample asset weight by monthly
        if frequency == 'MONTHLY':
            asset_weight.index.name = 'index'
            m = asset_weight.index.to_period('m')
            asset_weight = asset_weight.reset_index().groupby(m).last().set_index('index')
            asset_weight.index.name = ''

        factor_exposure = pd.Panel({target_date: self.get_factor_exposure(
            self.risk_model, factors['factors'], target_date,
            asset_weight.columns).T for target_date in asset_weight.index})
        factor_exposure = pd.DataFrame(index=asset_weight.index,
                                       columns=factor_exposure.major_axis)
        for target_date in asset_weight.index:
            factor_exposure.ix[target_date] = factor_exposure.ix[target_date].dot(asset_weight.ix[target_date].fillna(0))
        return factor_exposure.replace(0, np.nan).fillna(method='ffill')


    def specific_risk(self, date, symbols):
        """ get specific risk from risk model

        Keyword Arguments:
        date -- 
        """
        # find the nearest date next to target date from specific risk
        specific_risk = self.risk_model['specificRisk'].pivot(
            index='date', columns='symbol', values='specificrisk')
        dt_next_to_target = specific_risk.index.searchsorted(date)
        try:
            dt_next_to_target = specific_risk.index[dt_next_to_target]
        except:
            pass
        target_specific_risk = specific_risk.loc[dt_next_to_target, symbols]

        return target_specific_risk


    def delta(self, date, symbols):
        """
        create delta matrix from specific risk
        Keyword Arguments:
        date    -- 
        symbols -- 
        """
        diag = self.specific_risk(date, symbols)
        delta = pd.DataFrame(np.diag(diag), index=diag.index,
                             columns=diag.index).fillna(0)


# logger = logging.getLogger()
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.setLevel(logging.DEBUG)

# logger.debug('start')
# risk_model = gftIO.zload("/home/weiwu/share/optimize/x2.pkl")
# asset_weight = gftIO.zload("/home/weiwu/share/optimize/asset_weight.pkl")
# begin_date = gftIO.zload("/home/weiwu/share/optimize/begin_date.pkl")
# end_date = gftIO.zload("/home/weiwu/share/optimize/end_date.pkl")
# frequency = gftIO.zload("/home/weiwu/share/optimize/frequency.pkl")
# factors = gftIO.zload("/home/weiwu/share/optimize//factors.pkl")

# logger.debug('data loaded')
# RiskModel = RiskAnlysis(risk_model)
# RiskModel.factor_exposure(asset_weight, frequency, factors)

