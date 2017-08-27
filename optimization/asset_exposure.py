# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from lib.gftTools import gftIO, gsConst


def get_factor_exposure(risk_model, factor_list, date, symbols):
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
            factor_exposure[factor] = risk_model[factor].asMatrix().\
                                      loc[date, symbols]
        except KeyError:
            factor_exposure[factor] = np.nan
            #raise KeyError('invalid input date: %s' % date)
    factor_exposure.columns = gftIO.strSet2Np(factor_exposure.columns.values)
    factor_exposure = factor_exposure.fillna(0)

    return factor_exposure


def AssetExposure(context,model,target_dates,asset_weight,factors):
    """ get model factor exposure data multiplying asset weight, return asset factor exposure as result
    Parameters
    ----------
    model: risk model includes factor exposure 
    target_dates: dates to calculate on
    asset_weight: OTV, for instance, asset weights in a portfolio, benchmark weight, etc.
    factors: a list of GID

    Returns
    -------
    dataframe
    """
    if isinstance(asset_weight, gftIO.GftTable):
        asset_weight = asset_weight.asMatrix()

    if isinstance(factors, dict):
        factors = factors['factors']
        
    # get frequency of input dates
    dt_diff = target_dates.to_series().diff().mean()
    if dt_diff < pd.Timedelta('5 days'):
        frequency = gsConst.Const.DAILY
    elif dt_diff > pd.Timedelta('5 days') and dt_diff < pd.Timedelta('10 days'):
        frequency = gsConst.Const.WEEKLY
    else:
        frequency = gsConst.Const.MONTHLY
    begin_date = target_dates[0]
    end_date = target_dates[-1]
    # resample to monthly data
    if frequency == gsConst.Const.MONTHLY:
        asset_weight.index.name = 'index'
        m = asset_weight.index.to_period('m')
        asset_weight = asset_weight.reset_index().groupby(m).last().set_index('index')
        asset_weight.index.name = ''

    # need to input factors list to calculate the factors exposure, or will cause error using regular expression
    panel_model = pd.Panel({target_date: get_factor_exposure(model, factors, target_date,
                            asset_weight.columns).T for target_date in asset_weight.index})
    factor_exposure = pd.DataFrame(index=asset_weight.index, columns=panel_model.major_axis)
    for target_date in asset_weight.index:
        factor_exposure.ix[target_date] = panel_model.ix[target_date].dot(asset_weight.ix[target_date].fillna(0))
    return factor_exposure.replace(0, np.nan).fillna(method='ffill').ix[begin_date:end_date]
    
