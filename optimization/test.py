# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import os
import warnings
from cvxopt import matrix, solvers, spmatrix, sparse
from cvxopt.blas import dot

from lib.gftTools import gftIO

# fetch data

path = "/home/weiwu/share/"
os.chdir(path)
#target_mode = gftIO.zload(os.path.join(path, 'factor_exposure.pkl'))
target_mode = gftIO.zload('factor_exposure.pkl')
