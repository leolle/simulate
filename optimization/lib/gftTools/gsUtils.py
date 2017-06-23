# -*- coding: utf-8 -*-
from lib.gftTools import gftIO
import numpy as np
import pandas as pd


def getCashGid():
    return gftIO.strSet2Np(np.array(['0AC062D610A1481FA5561EC286146BCC']))


def getGodGid():
    return np.chararray(1, itemsize=16, buffer='\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0')


def getParm(dict_parm, parmName, defaultValue):
    if parmName in dict_parm:
        return dict_parm[parmName]
    else:
        return defaultValue


def alignDate(sourceDates, targetDates, method='ffill', returnidx=False):
    """
    source dates: rebalance dates
    target dates: all dates
    """

    df_dateMap = pd.DataFrame({'targetDate':targetDates, 'idx':np.arange(len(targetDates))}, index=targetDates)
    if isinstance(sourceDates, pd.tslib.Timestamp):
        df_dateMap = df_dateMap.reindex([sourceDates], method=method)
    else:
        df_dateMap = df_dateMap.reindex(sourceDates, method=method)
    if returnidx:
        result = np.array(df_dateMap['idx'])
        if isinstance(sourceDates, pd.tslib.Timestamp):
            result = result[0]
    else:
        if isinstance(sourceDates, pd.tslib.Timestamp):
            result = df_dateMap['targetDate'][0]
        else:
            if isinstance(sourceDates, pd.tseries.index.DatetimeIndex):
                result = pd.DatetimeIndex(df_dateMap['targetDate'])
            else:
                result = np.array(df_dateMap['targetDate'])
    return result


def roundToLot(sizeArray, lotSize):
    if lotSize > 0:
        sizeArray = sizeArray.fillna(0)
        return np.sign(sizeArray) * lotSize * np.floor(np.round(abs(sizeArray)) / max(1., lotSize))
    else:
        return sizeArray


def cut2bin(ser_x, totalBinNum, ascending=False):
    # calculate bin size
    totalBinNum = int(totalBinNum)
    xlen = len(ser_x)
    arr_binsize = np.repeat(xlen // totalBinNum, totalBinNum)
    remaining = xlen % totalBinNum
    if remaining > 0:
        arr_binsize[:remaining] += 1
    # map each signal to its binIdx
    arr_binmap = np.repeat(np.arange(totalBinNum) + 1, arr_binsize)
    ser_xrank = ser_x.rank(method='first', ascending=ascending)
    ser_result = pd.Series(
        arr_binmap[np.array(ser_xrank.values - 1, dtype='int')])
    return ser_result


class Strategy:
    #hold the strategy result, including cumret, holding
    def __init__(self, gid, cumret, holding):
        # save matrix and column table.
        self.gid = gid		
        self.cumret = cumret
        self.holding = holding
        self.type = "Strategy"
        self.cumretGid = None
        self.holdingGid = None
