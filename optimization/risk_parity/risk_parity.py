# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
np.random.seed(1000)
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import Image, display
from scipy.optimize import minimize
import time

data = pd.read_csv('./assets.csv', index_col='Date', parse_dates=True).rename(columns={"CBA00332.CS":"Bond","NH0100.NHF":"Commodity","000300.SH":"Equity"})
data.head()
