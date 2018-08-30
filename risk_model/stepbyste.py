# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 18:26:16 2017

@author: gft
"""

###remove input

###it supposed to be bussiness day

#gid:A1F361E8D7782C2D1C19AFB15696B0FB
import numpy as np
import pandas as pd
from functools import reduce
import math as mt
import statsmodels.regression.linear_model as lm
import statsmodels.api as sm
from lib.gftTools import gftIO
import datetime
import pandas as pd
import numpy as np
import re
import os
import warnings
from functools import reduce
import math as mt
import statsmodels.api as sm
from lib.gftTools import gftIO
import datetime
import logging
from ylib import ylog

ylog.set_level(logging.DEBUG)
ylog.console_on()
ylog.filelog_on("app")

risk_model_path = '/home/weiwu/share/risk_model/'
x0 = gftIO.zload(os.path.join(risk_model_path, 'stock_return.pkl'))
x1 = gftIO.zload(os.path.join(risk_model_path, 'factors.pkl')

x2 = gftIO.zload(os.path.join(risk_model_path, 'market_capital.pkl'))
x3 = 4
x4 = 5

ylog.debug('parse data')
