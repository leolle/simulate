# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from copy import copy
from lib.gftTools import gftIO
from scipy import linalg
import matplotlib.pyplot as plt
from utils.winsorize_mad import winsorize_mad

ROE = gftIO.zload("/home/weiwu/share/black_litterman/ROE_cur_year.pkl")
ROE_forecast = gftIO.zload("/home/weiwu/share/black_litterman/ROE_cur_year.pkl")
ROE_daily_PIT_change = gftIO.zload(
    "/home/weiwu/share/black_litterman/ROE_daily_PIT_change.pkl")
ROE_daily_PIT_change2 = gftIO.zload(
    "/home/weiwu/share/black_litterman/ROE_daily_PIT_change2.pkl")
ROE_PIT = gftIO.zload("/home/weiwu/share/black_litterman/ROE_PIT.pkl")

if isinstance(ROE, gftIO.GftTable):
    # views on all assets are not required
    ROE = ROE.asMatrix().copy()
    ROE.fillna(method='ffill', inplace=True)
    ROE_forecast = ROE_forecast.asMatrix().copy()
    ROE_forecast.fillna(method='ffill', inplace=True)
    ROE_daily_PIT_change = ROE_daily_PIT_change.asMatrix().copy()
    ROE_daily_PIT_change.fillna(method='ffill', inplace=True)
    ROE_PIT = ROE_PIT.asMatrix().copy()
    ROE_PIT.fillna(method='ffill', inplace=True)
    ROE_daily_PIT_change2 = ROE_daily_PIT_change2.asMatrix().copy()
    ROE_daily_PIT_change2.fillna(method='ffill', inplace=True)

ROE.loc[:, b'\x8eK\x9c+\x8a\xc71\x00j\xfb\x07AX\x13\xc8\xac'].plot(style='g--')
ROE_forecast.loc[:, b'\x8eK\x9c+\x8a\xc71\x00j\xfb\x07AX\x13\xc8\xac'].plot(
    style='r--')
ROE_daily_PIT_change.loc[:,
                         b'\x8eK\x9c+\x8a\xc71\x00j\xfb\x07AX\x13\xc8\xac'].plot(
                             style='r--')
ROE_daily_PIT_change2.loc[:,
                          b'\x8eK\x9c+\x8a\xc71\x00j\xfb\x07AX\x13\xc8\xac'].plot(
                              style='g--')
#ROE_PIT.loc[:, b'\x8eK\x9c+\x8a\xc71\x00j\xfb\x07AX\x13\xc8\xac'].plot()
plt.show()
