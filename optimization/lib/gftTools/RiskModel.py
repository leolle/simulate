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
        self.ls_factors = [x for x in risk_model_merge.keys() if
                           re.match("[A-Z0-9]{32}$", x)]
        self.ls_factors_ret = [x for x in risk_model_merge.keys() if
                               re.search(".ret$", x)]

    def get_factor_exposure(self, factor_list, date, symbols):
        ''' Return factor exposure matrix(big X).

        Parameters
        ----------
        risk_model: dictionary
            Including specific risk, different factor exposure dataframe for all
            symbols.

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
        """ Get assets factor exposure.
        F = w^T * exposure
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
        factor_exposure = pd.DataFrame(index=asset_weight.index,
                                       columns=asset_factor_exposure.major_axis)
        for target_date in asset_weight.index:
            factor_exposure.ix[target_date] = asset_factor_exposure.ix[target_date].dot(
                asset_weight.ix[target_date].fillna(0))
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

        return delta

    def covariance_matrix(self, date, factors):
        """
        Keyword Arguments:
        date    -- 
        factors -- 
        """
        pass

    def factor_return(self):
        """ get factor return from all the factors """

        factors_ret = pd.DataFrame(index=self.risk_model[self.ls_factors_ret[0]].index,
                                   columns=self.ls_factors_ret)
        for factor in self.ls_factors_ret:
            factors_ret[factor] = self.risk_model[factor]

        return factors_ret
