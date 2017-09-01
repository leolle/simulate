
# coding: utf-8

from lib.gftTools import gftIO

def __gsWrapper__(context, x0, x1, x2, x3):
    return Lasso(context, x0, x1, x2, x3)



import numpy as np
import pandas as pd
from sklearn import linear_model
from lib.gftTools import gftIO
from functools import reduce
from datetime import datetime

from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def Lasso(context,df_ret,df_expo,int_max_iter,int_cv):
    '''
    df_ret=x0
    df_expo=x1
    f_test_size:float,train/test split percentage
    int_max_iter:max_iter_cnt
    int_cv:get best alpha,you need k-fold,int_cv =k_fold
    '''
    ##step1:get stock return daterange,stocklist
    df_ret=df_ret.asMatrix()
    ret_daterng=list(df_ret.index)

    ##step2:get factor loading factorname,daterange,stocklist

    dict_expo={df_expo[i][0]:df_expo[i][1].asMatrix() for i in range(len(df_expo))}
    aa={i:len(j.index) for i,j in dict_expo.items()}
    float_quantile=np.percentile(a=(list(aa.values())),q=5)
    dict_expo={k: v for k, v in dict_expo.items() if len(v.index) >= float_quantile}


    allfactorname=list(dict_expo.keys())
    ls_expo_date=[i.index for i in dict_expo.values()]

    ls_alldates_fexpo=reduce(np.intersect1d,ls_expo_date)

    ##step3:get date map for regression
    #ret_daterng=list(map(lambda x:pd.DatetimeIndex([datetime(x.year,x.month,1)])[0],ret_daterng))
    ret_daterng={pd.DatetimeIndex([datetime(x.year,x.month,1)])[0]:x for x in ret_daterng}
    df_ret_daterng=pd.DataFrame(list(ret_daterng.keys()),columns=['date_ret']).sort_values("date_ret",ascending=True)

    df_alldates_fexpo=pd.DataFrame(ls_alldates_fexpo,columns=['date_fexpo']).sort_values("date_fexpo",ascending=True)

    df_date_map = pd.merge_asof(df_ret_daterng,
                              df_alldates_fexpo, left_on ="date_ret",right_on ="date_fexpo",allow_exact_matches=False)

    df_date_map=df_date_map.dropna(how='any').reset_index()
    dict_date_map={df_date_map.date_fexpo[i]:df_date_map.date_ret[i] for i in range(len(df_date_map))}

    ##step4:get allsymbol for regression
    #dict_ret_symbol={date: len(np.unique(df_ret[df_ret.index == ret_daterng[dict_date_map[date]]].T.dropna().index)) for date in dict_date_map.keys()}

    dict_symbol={date: np.unique(df_ret[df_ret.index == ret_daterng[dict_date_map[date]]].T.dropna().index) for date in dict_date_map.keys()}

    dict_expo={date:fexpomerge(df_expo,date,allfactorname,dict_symbol)  for date in dict_date_map.keys()}
    dict_ret={date:df_ret[df_ret.index == ret_daterng[dict_date_map[date]]].T.reindex(index=dict_symbol[date]) for date in dict_date_map.keys()}
    ##y is the stock return of next period

    ##step5:x for standardization

    dict_expo_std={k: (v -v.mean(axis=0))/v.std(axis=0) for k, v in dict_expo.items()}

    ##step6:对于x进行缺失值的处理,在每个截面回归上,对于缺失值>30%的因子，直接将该因子剔除，剩余的缺失值用均值补齐
    dict_expo_filter={k: missingpre(v) for k, v in dict_expo_std.items()}

    ##step7:slice the date range
    #ls_daterange=list(dict_date_map.keys())
    ls_daterange=list(dict_date_map.keys())

    lasso_result={}

    for date in ls_daterange:
        X=dict_expo_filter[date].sort_index()
        Y=dict_ret[date].sort_index()

        train_X,test_X,train_Y,test_Y=train_test_split(X,Y, test_size=0.2, random_state=42)

        #train_X=preprocessing.scale(train_X)##标准化，均值为0，方差为1
        #test_X=preprocessing.scale(test_X)

        lasso=linear_model.LassoCV(max_iter=int_max_iter,cv=int(int_cv),normalize=False)
        #lasso.fit(train_X,train_Y)
        lasso.fit(np.array(train_X),np.array(train_Y).ravel())
        #scores=np.zeros((100,1))
        #scores[:,0]=np.mean(lasso.mse_path_,axis=1)
        #scores=np.sort(scores)

        df_params =pd.DataFrame(data=lasso.coef_,index=list(train_X.columns),columns=['params'])
        df_params=df_params[df_params['params'] !=0]
        test_score=lasso.score(test_X,test_Y)
        lasso_result[date]={'params':df_params,'score':pd.DataFrame(data=[test_score],columns=['score']),
        'fitfeaturecnt':pd.DataFrame(data=[float(df_params.shape[0])],columns=['fitfeaturecnt']),'allfeaturecnt':pd.DataFrame(data=[float(len(df_expo))],columns=['allfeaturecnt']),
        'best_lam':pd.DataFrame(data=[lasso.alpha_],columns=['best_lam'])}

    dict_final_return_new={}
    for i in list(lasso_result[ls_daterange[0]].keys()):
        ls_final_return=[lasso_result[date][i].assign(date=date) for date in ls_daterange]
        dict_final_return_new[i]=pd.concat(ls_final_return,axis=0)


    return dict_final_return_new

def fexpomerge(df_expo,date,allfactorname,dict_symbol):
    ls_raw_df_fexpo=[df_expo[i][1].asMatrix().reindex(index=[date],columns=dict_symbol[date]).rename(index={date:df_expo[i][0]}) for i in range(len(allfactorname))]
    df_fexpo_onedate=pd.concat(ls_raw_df_fexpo,axis=0).T
    return df_fexpo_onedate

def missingpre(v):
    #v=dict_expo[date]
    df_expoonedate=v
    if len(df_expoonedate.columns) != len(np.unique(df_expoonedate.columns)):
        df_expoonedate=df_expoonedate.T.drop_duplicates().T
    dict_missingrate ={i:df_expoonedate[i].count()/df_expoonedate.shape[0] for i in list(df_expoonedate.columns)}
    dict_missingrate_filter={m: n for m,n in dict_missingrate.items() if n >= 0.7}
    ls_leftfactor=list(dict_missingrate_filter.keys())

    df_expoonedate_del=df_expoonedate.reindex(columns=ls_leftfactor)
    df_expoonedate_del=df_expoonedate_del.fillna(df_expoonedate_del.mean(axis=0,skipna=True))
    return df_expoonedate_del







# In[3]:






from lib.gftTools import gftIO
def debug__gsWrapper__():
    context = gftIO.zload("/home/gft/data/context.pkl")
    x0 = gftIO.zload("/home/gft/data/x0.pkl")
    x1 = gftIO.zload("/home/gft/data/x1.pkl")
    x2 = gftIO.zload("/home/gft/data/x2.pkl")
    x3 = gftIO.zload("/home/gft/data/x3.pkl")
    __gsWrapper__(context, x0, x1, x2, x3)


# In[14]:

from lib.gftTools import gftIO


# In[17]:

context = gftIO.zload("/home/gft/data/context.pkl")
x0 = gftIO.zload("/home/gft/data/x0.pkl")
x1 = gftIO.zload("/home/gft/data/x1.pkl")
x2 = gftIO.zload("/home/gft/data/x2.pkl")
x3 = gftIO.zload("/home/gft/data/x3.pkl")







# In[16]:

df_ret=x0
df_expo=x1
int_max_iter=2000
int_cv=10


# In[30]:

test=Lasso(context,df_ret,df_expo,int_max_iter,int_cv)




test['']







# In[23]:

test.keys()


# In[33]:

test['best_lam']
