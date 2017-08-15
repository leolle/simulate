# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import pandas as pd
import logging

import re
import cvxpy as cvx

from lib.gftTools import gftIO
from lib.gftTools.RiskModel import RiskAnlysis


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.debug('start')
risk_model = gftIO.zload("/home/weiwu/share/optimize/x2.pkl")
asset_weight = gftIO.zload("/home/weiwu/share/optimize/asset_weight.pkl")
begin_date = gftIO.zload("/home/weiwu/share/optimize/begin_date.pkl")
end_date = gftIO.zload("/home/weiwu/share/optimize/end_date.pkl")
frequency = gftIO.zload("/home/weiwu/share/optimize/frequency.pkl")
factors = gftIO.zload("/home/weiwu/share/optimize//factors.pkl")

logger.debug('data loaded')

RiskModel = RiskAnlysis(risk_model)
RiskModel.factor_exposure(asset_weight, frequency, factors)
