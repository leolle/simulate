# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:01:27 2017

@author: gft
"""
#gid:4322E3D243BA4CD978EF87DAD7ACB69B

import numpy as np
import pandas as pd


def Sd(x, maxValue, rmMaxPct, rmMinPct, keepOrder, date):
    x = x[x['idname'] == date]
    ls_x_var = list(x.variable)
    x.dropna(inplace=True)  ##remove missing values
    x = x.set_index('variable').drop('idname', axis=1)
    x = x['value']
    lenx = len(x)
    ori_x = x.copy()

    if lenx > 1:
        if (rmMaxPct > 0 or rmMinPct > 0):
            rankidx = x.rank(ascending=True, method='first')
            if rmMinPct > 0:
                x = x[rankidx > round(rmMinPct * lenx)]
            if rmMaxPct > 0:
                x = x[rankidx <= round((1 - rmMaxPct) * lenx)]
    rmx = x

    if len(rmx) > 1:
        meanvalue = rmx.mean()
        sdvalue = rmx.std()
        result = ori_x - meanvalue
        if keepOrder == 1:
            max_x = result[result > maxValue * sdvalue]
            if len(max_x) > 0:
                replace_max_x = sdvalue * maxValue * (1 + max_x.rank(
                    ascending=True, method='average') / len(max_x) / 10000)
                result[result > maxValue * sdvalue] = replace_max_x

            min_x = result[result < -maxValue * sdvalue]
            if len(min_x) > 0:
                replace_min_x = -sdvalue * maxValue * (1 + (abs(min_x)).rank(
                    ascending=True, method='average') / len(min_x) / 10000)
                result[result < -maxValue * sdvalue] = replace_min_x
        else:
            absresult = abs(result)
            result = (np.sign(result)) * (absresult.where(
                absresult <= maxValue * sdvalue, maxValue * sdvalue))

        result = result + meanvalue

    result = pd.DataFrame(result).reindex(ls_x_var).assign(
        date=date).reset_index()
    return result


def WinsorizeSd(x, maxValue=5, rmMaxPct=0, rmMinPct=0, keepOrder=0):
    x = x.asColumnTab()
    ls_date = np.unique(x.idname)

    ls_sd_result = [
        Sd(x, maxValue, rmMaxPct, rmMinPct, keepOrder, date) for date in ls_date
    ]
    result = pd.concat(ls_sd_result)

    return result
