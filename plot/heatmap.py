# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
symbols = ['A', 'B', 'C', 'D']
dates = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 4), index=dates, columns=symbols)
df.index.name = 'date'
symbol = ((np.asarray(['A', 'B', 'C', 'D']))).reshape(2,2)
perchange = df.values

labels = (np.asarray(["{0} \n {1:.2f}".format(symb,value)
                      for symb, value in zip(symbols*8,
                                             perchange.flatten())])).reshape(8,4)

sns.heatmap(df.values,annot=labels,fmt="",cmap="RdYlGn")
plt.show()
