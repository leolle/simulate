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


def CreateAssetExposurePanel(context, model, asset_weight, factors):
    """
    create a pandas panel whose item axis is DatetimeIndex,
    major axis is factors, minor axis is asset names.
    """
    if isinstance(asset_weight, gftIO.GftTable):
        asset_weight = asset_weight.asMatrix()

    datetime_index = asset_weight.index.unique()
    panel_exposure = pd.Panel({date: get_factor_exposure(model, factors, date,
                                                         asset_weight.columns).T
                               for date in datetime_index})
    
    return panel_exposure
