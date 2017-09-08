# -*- coding: utf-8 -*-
from lib.gftTools import gftIO
import numpy as np
import pandas as pd

def multiply_panel(context,multiplier_panel,multiplicand):
    """ multiply a 3-d panel with dataframe

    Parameter
    ---------
    multiplier_panel: pandas panel
        target date dataframe: symbols * assets

    multiplicand: pandas.DataFrame
        OTV: O is assets.

    Return
    ------
    product: pandas dataframe
        asset exposure on factors, or group weight.
    """
    if isinstance(multiplicand, gftIO.GftTable):
        multiplicand = multiplicand.asMatrix().copy()
    if not isinstance(multiplicand.index, pd.DatetimeIndex):
        multiplicand.set_index('idname', inplace=True)
    if 'idname' in multiplicand:
        multiplicand.drop('idname', axis=1, inplace=True)
    datetimeindex = multiplicand.index.intersection(multiplier_panel.items)

    product = pd.DataFrame(data=np.nan, index=datetimeindex, columns=multiplier_panel.minor_axis)
    for target_date in datetimeindex:
        product.ix[target_date] = multiplier_panel[target_date].loc[multiplicand.columns,:].T.fillna(0).dot(multiplicand.ix[target_date].fillna(0))

    return product


context = gftIO.zload("/home/weiwu/share/optimize/context.pkl")
multiplier_panel = gftIO.zload("/home/weiwu/share/optimize/group_sparse_panel.pkl")
multiplicand = gftIO.zload("/home/weiwu/share/optimize/hs300_weight.pkl")
multiply_panel(context, multiplier_panel, multiplicand)
# if isinstance(multiplicand, gftIO.GftTable):
#     multiplicand = multiplicand.asMatrix().copy()
# if not isinstance(multiplicand.index, pd.DatetimeIndex):
#     multiplicand.set_index('idname', inplace=True)
# if 'idname' in multiplicand:
#     multiplicand.drop('idname', axis=1, inplace=True)
# datetimeindex = multiplicand.index.intersection(multiplier_panel.items)

# product = pd.DataFrame(data=np.nan, index=datetimeindex, columns=multiplier_panel.minor_axis)
# for target_date in datetimeindex:
#     product.ix[target_date] = multiplier_panel[target_date].loc[multiplicand.columns,:].T.fillna(0).dot(multiplicand.ix[target_date].fillna(0))
