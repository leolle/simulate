# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:02:38 2017

@author: gft
"""
#gid:66716BADCD584BAB2B5A6F11D93C5821
#gidnew:1D46BC081C88053A6746D842931CCA3D
'''
x:
b'\xecC\x1e\xb1\xa5\x11\x01\x01x\xa6\xc0&z\xec\xba\xb0'       6.102530e+04
b'\x1dF"\xda&f\xee\x03RK\xf5\xe1i\xc2\x8f\x88'                1.349660e+05
b"\x08M\x0e'\x1e\xe8\xf8\x04\x89pD\x87\x80\xcf\x8d\xa1"       3.844291e+05
b'#K\xa26\xd5\x16\x1a\x057T1\xba\xb7w \xb4'                   7.891052e+04
b'\xe9C\x11\xfa0\x98h\x05\xc4~\xd9\x14\xa1\xce\xc5\x91'       8.676408e+04
b'\xfcDIE\x9e\xc2\xf7\x05\xc4:\xc4\x06MiS\x80'                8.562230e+05
b'&J\xfb\\\x85\x1bj\x06\x9a\r\x87\xf1b\x97\x9b\xbe'           1.056949e+05
b'\x00J{\xfa \xd0n\tsr\xac\xb44\xd9\x10\x9c'                  7.245449e+04
b'?K\x90\xad\xaeM\xe7\t\x1d\xc7\x8dx}\xb7\xd7\xab'            4.473092e+04
b')EC\x02&{\xe9\n.c\xf4\xf9\x84\x9d\x15\x88'                  7.827646e+04
b'#K_\x05!#\x14\x0c\x03w\x06\xc9\xbd\xc23\x81'                7.546544e+04
b'\x88H\xb8\xc4\xe4W\xab\x0c/\xb4\xf8m\xae5\x93\x91'          1.924509e+05
b'iD\xb8\xda\x9c .\rp~\xc8\xefV\xcb\x12\xae'                  1.776781e+05
b'\x97Dw\xe9\xeezZ\rRC\n\x1aF.\x85\xab'                       1.129031e+05

'''

import numpy as np
import pandas as pd
from lib.gftTools import gftIO
import matplotlib.pyplot as plt


def Mad(x, maxValue, keepOrder, date):
    raw_return = x[x['idname'] == date].rename(
        columns={'idname': 'date'}).reset_index().drop(
            'index', axis=1)
    x = x[x['idname'] == date]
    ls_x_var = list(x.variable)
    x.dropna(inplace=True)  ##remove missing values
    x = x.set_index('variable').drop('idname', axis=1)
    x = x['value']

    if len(x) > 1:
        medianvalue = x.median()
        madvalue = x.mad() * 1.4826
        result = x - medianvalue
        if keepOrder == 1:
            max_x = result[result > maxValue * madvalue]
            if len(max_x) > 0:
                replace_max_x = madvalue * maxValue * (1 + max_x.rank(
                    ascending=True, method='average') / len(max_x) / 10000)
                result[result > maxValue * madvalue] = replace_max_x

            min_x = result[result < -maxValue * madvalue]
            if len(min_x) > 0:
                replace_min_x = -madvalue * maxValue * (1 + (abs(min_x)).rank(
                    ascending=True, method='average') / len(min_x) / 10000)
                result[result < -maxValue * madvalue] = replace_min_x
        else:
            absresult = abs(result)
            result = (np.sign(result)) * (absresult.where(
                absresult <= maxValue * madvalue, maxValue * madvalue))

        result = result + medianvalue
        result = pd.DataFrame(result).reindex(ls_x_var).assign(
            date=date).reset_index()
        return result
    else:
        return raw_return


def winsorize_mad(x, maxValue=5, keepOrder=0):
    x = x.asColumnTab().copy()
    ls_date = np.unique(x.idname)

    ls_mad_result = [Mad(x, maxValue, keepOrder, date) for date in ls_date]
    result = pd.concat(ls_mad_result)
    return result.pivot(
        index='date', columns='variable', values='value').dropna(
            how='all', axis=1, inplace=True)


if __name__ == '__main__':
    ROE = gftIO.zload("/home/weiwu/share/black_litterman/ROE_cur_year.pkl")
    result = winsorize_mad(ROE)
