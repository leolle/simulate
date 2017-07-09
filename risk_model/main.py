# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import os
import warnings

from lib.gftTools import gftIO

risk_model_path = '~/share/factor_exposure.pkl'

risk_model = gftIO.zload(risk_model_path)


risk_model_path = r'd:\Wuwei\Project\simulator\data\risk_model\factor_exposure.pkl'

risk_model = gftIO.zload(risk_model_path)
ls_factor = list(risk_model.keys())
ls_factor.pop(-1)
ls_factor.pop(7)

ls_symbols = []
for factor in ls_factor:
    ls_symbols.append(risk_model[factor].asMatrix().columns.tolist())
    
flat_list = [item for sublist in ls_symbols for item in sublist]
    
ls_symbols = list(set(flat_list))    
ls_dates = risk_model[factor].asMatrix().index
                     
index = pd.MultiIndex.from_product([ls_dates, ls_symbols], names=['date', 'symbol'])

df_factor_exposure = pd.DataFrame(index=index, columns=ls_factor)
