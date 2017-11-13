# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
import os
import warnings
from functools import reduce
import math as mt
import statsmodels.api as sm
from lib.gftTools import gftIO
import datetime
import logging

time = pd.date_range('2000-01-01', freq='D', periods=432)
X = xr.DataArray(
    np.random.randn(432, 10, 3), [('date', time), ('stock', list('abcdefghij')),
                                  ('factor', list('xyz'))])

Y = xr.DataArray(
    np.random.randn(432, 10), [('date', time), ('stock', list('abcdefghij'))])


def regression(X, Y, constraint1, constraint2, constraint3):
    """ regression X by Y
    Keyword Arguments:
    X           --
    Y           --
    constraint1 --
    constraint2 --
    constraint3 --

    Return:
    model -- dictionary, key: value
        param: xarray, factor return, factors: 36, date: 432
        resid: xarray, stock residual, date: 432, symbol: 2981
    """
    pass


param = xr.DataArray(
    np.random.randn(432, 36), [('date', time), ('factors', [
        '018D158ACE908165ED4ECEA129ABF547', '90F09502030C80B91B02A2CE72A5A0EB',
        '91B8D8D34EE8AA2BD9B60F9D223C8707', '767FA9609E0C0B661EE557AB80C6F8E4',
        'A528F7BFA538A7675484FC2A72021138', '4972C5017D9B1D2AEDA81920FA21D3BF',
        '35555851DC92592BF9A5D55906D10A2B', 'D84A4B250D077E0EFC4F7FFF393FCB44',
        'D753394CA4D5A4CFC25BB5DACA7293F3', 'C585E113DD45D2344C97D3D2967D536F',
        '22895BDA3790CB3FA9FABD1E2FF883CE', '26C8DB23CDCCD1E7D858D3340449C854',
        '3995B9BA60E9943BF39F784E2E6B451D', 'EFE87574B86C774ADFD08F421AF5D11B',
        '7F216CE9D40D23FEDC54147E9D754DEE', '26BF3A3BD0BB76D58CC560B72527AB14',
        'B3E67D32A2F6428BB06D1BFCD2547798', 'BB81923B85135A411E8BDD80015E6512',
        'BED9C798EDEE97DC1712BB3E05FC8113', '265ED5755EEC25762A1293A398C40F5D',
        '324F4A27A88603BE2C53D5AFA7745584', '62C09CCF341CA27BCE7835103FE32E97',
        'BBA8CAB7DE9857EE8B8AFD9A23E3D34A', 'EB34F4D5C92E85C2307DB2C8015C94F1',
        '54899D64BCFE8C3BA11BB84429A46C52', 'A7638B5C794E97F887D2522215A3DFF8',
        '2AECD1DE9CC1EAA3982AAD231FA0AE39', '441E8F64A7582F553BCBE42A216285F1',
        '873CF94D09229206D675ACC32328DC24', 'EB8553C313C38BC180E1972D798622BA',
        'ABEC912F31E326F4C1FC507AF787C8FA', 'E8D70EA915C420F9B9005BB21540788C',
        'BE3E35A7C0CB49EB9E1CB41D566563E7', 'B9CCDA635F039E84D489F964DB08BC5C',
        'B1B02CFAB81248BAA87754E760769BD2', 'E8A54A95C9264162BEEC88B9CF65C78B'
    ])])

resid = xr.DataArray(np.random.randn(432, 2981))
resid.coords['dim_0'] = time

model = {'param': param, 'resid': resid}
