# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import copy
from lib.gftTools import gftIO


def BoundariesSetting(context,targets,default_lower_bnd,default_upper_bnd,user_input):
    """ set constraint boundaries, use user-input value to replace default lower and upper value.
    Parameters
    ----------
    targets: OTV, portfolio asset.
    default_lower_bnd: float, lower boundary value, usually 0.
    default_upper_bnd: float, upper boundary value, usually 1.
    user_input: OTVV, customized boundary setting for some assets.

    Returns
    -------
    GftTable, ColumnTable
    """
    targets = copy.deepcopy(targets)
    if default_lower_bnd > default_upper_bnd:
        raise ValueError('default lower boundary value is greater than upper boundary value.')
    if isinstance(targets, gftIO.GftTable):
        df_target_limit = targets.asColumnTab()

    df_boundary = pd.DataFrame(columns=['date', 'target', 'lower_bnd', 'upper_bnd'])
    df_boundary['date'] = df_target_limit['idname']
    df_boundary['target'] = df_target_limit['variable']
    df_boundary['lower_bnd'] = default_lower_bnd
    df_boundary['upper_bnd'] = default_upper_bnd

    if user_input is not None:
        df_user_input_limit = user_input.copy()
        df_user_input_limit = df_user_input_limit.set_index('variable')
        df_boundary = df_boundary.set_index('target')
        df_boundary.loc[df_user_input_limit.index, 'lower_bnd'] = df_user_input_limit['lower_bnd']
        df_boundary.loc[df_user_input_limit.index, 'upper_bnd'] = df_user_input_limit['upper_bnd']
        df_boundary.reset_index(inplace=True)

    return gftIO.GftTable(matrix=None, columnTab=df_boundary,
                          matrixIsFromPython=True,
                          gid='1CCFF4BA182A45618AC5AD6086B7A436',
                          columnOrders=None)
