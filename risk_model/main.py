# -*- coding: utf-8 -*-
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


def risk_model(df_ret, dict_risk_expo, capital, corr_half_life, var_half_life):
    """
    Regression stock return by previous factor exposure, to get
    factor return covariance and residual.

    Pseudo code:
    1. process input data, parse, drop and fill.
    2. get intersection of all factor names, all symbol names, all dates.
    3. Solve the problem of heteroskedasticity by square root the market capitalization.
    Handbook p5, p15.
    new return = square root of market capitalization * stock return,
    add a constraint column to new return.

    calculate factor return.
    calculate factor return covariance.
    calculate the residual(specific) variances of regression.
    generate final return value.

    Keyword Arguments:
    df_ret           -- pd.DataFrame, stock daily return.
    dict_risk_expo   -- dictionary, factor exposure, key=factor.
    capital          -- pd.DataFrame, stock market capital, to calculate weight.
    corr_half_life   -- int, to compare correlation half life.
    var_half_life    -- int, to compare variance half life.

    Return:
    27 industrial factors + 8 style factors return -- pd.DataFrame
    ret_cov                                        -- pd.DataFrame, return covariance
    specificRisk                                   -- pd.DataFrame, residual
    """

    # get all factor names
    logger.debug('parse data')
    ls_fexponame = list(
        map(gftIO.gidInt2Str, list(dict_risk_expo['osets'].asColumnTab()[
            'O0'])))
    ind_factor_name = sorted(
        list(
            map(gftIO.gidInt2Str,
                list(dict_risk_expo[ls_fexponame[0]].asColumnTab()['O0']))))
    sty_factor_name = sorted(
        list(
            map(gftIO.gidInt2Str,
                list(dict_risk_expo[ls_fexponame[1]].asColumnTab()['O0']))))
    allfactor = ind_factor_name + sty_factor_name

    ##stock return preprocess
    df_w_ret = df_ret.asMatrix().T.dropna(how='all', axis=1)

    ##get factor exposure date list(all snapshots)
    dict_risk_expo_new = {
        factorname: dict_risk_expo[factorname].asMatrix().dropna(how='all')
        for factorname in allfactor
    }
    ls_ls_fexpodate = list([
        dict_risk_expo_new[factorname].index.tolist()
        for factorname in dict_risk_expo_new.keys()
    ])
    ls_alldates_fexpo = reduce(np.intersect1d, ls_ls_fexpodate)

    ## get factor exposure symbol list
    ls_ls_fexposymbol = list([
        dict_risk_expo_new[factorname].columns.tolist()
        for factorname in dict_risk_expo_new.keys()
    ])
    ls_allsymbols_fexpo = reduce(np.intersect1d, ls_ls_fexposymbol)

    ##weight preprocess
    weight = capital.asMatrix().T

    ##get the date/symbol intersection of (stock return,factor exposure,capital)

    ##ls_alldates save the stock return map date

    ##get fexpo date,find the nearest business day

    fexpodate = pd.DataFrame(ls_alldates_fexpo, columns=['date_fexpo'])
    retdate = pd.DataFrame(df_w_ret.columns, columns=['date_ret'])

    retdate.sort_values("date_ret", ascending=True, inplace=True)
    fexpodate.sort_values("date_fexpo", ascending=True, inplace=True)

    df_date_map = pd.merge_asof(
        retdate,
        fexpodate,
        left_on="date_ret",
        right_on="date_fexpo",
        allow_exact_matches=False)

    df_date_map.dropna(how='any', inplace=True)
    df_date_map = df_date_map.drop_duplicates(subset='date_fexpo').reset_index()
    dict_date_map = {
        df_date_map.date_fexpo[i]: df_date_map.date_ret[i]
        for i in range(len(df_date_map))
    }

    ls_alldates = sorted(
        list(
            set(capital.columns).intersection(set(df_w_ret.columns))
            .intersection(set(dict_date_map.values()))))
    ls_alldates_ondaybefore = sorted(list(dict_date_map.keys()))
    ##get daily symbol list
    ls_allsymbols = {
        date: list(
            set(df_w_ret[[dict_date_map[date]]].dropna().index).intersection(
                set(ls_allsymbols_fexpo)).intersection(set(capital.index)))
        for date in ls_alldates_ondaybefore
    }

    ## align the stock return and factor exposure
    dict_df_capital_raw = {
        date: capital[[date]].reindex(index=ls_allsymbols[date]).fillna(0)
        for date in ls_alldates_ondaybefore
    }
    dict_df_capital = {
        date: np.sqrt(dict_df_capital_raw[date])
        for date in ls_alldates_ondaybefore
    }

    dict_df_ret = {
        dict_date_map[date]: pd.concat(
            [(df_w_ret[[dict_date_map[date]]].reindex(
                index=ls_allsymbols[date])) * (dict_df_capital[date].rename(
                    columns={date: dict_date_map[date]})),
             pd.DataFrame(
                 data=np.zeros(1),
                 index=['constrain'],
                 columns=[dict_date_map[date]])],
            axis=0)
        for date in ls_alldates_ondaybefore
    }
    dict_df_fexpo_raw = {
        date: fexpomerge(dict_risk_expo_new, date, allfactor, ls_allsymbols)
        for date in ls_alldates_ondaybefore
    }
    dict_df_fexpo = {
        date: dict_df_fexpo_raw[date].assign(countryfactor=1).multiply(
            dict_df_capital[date].squeeze(), axis='index')
        for date in ls_alldates_ondaybefore
    }

    ##calculate constraints
    dict_df_fexpo_con = {
        date: expoconstrain(dict_df_fexpo_raw, date, ind_factor_name, allfactor,
                            dict_df_capital_raw, sty_factor_name, dict_df_fexpo)
        for date in ls_alldates_ondaybefore
    }

    # for i in dict_risk_expo_new.keys():
    #     if dict_risk_expo_new[i].index.min() > df_l_ret.index.min(
    #     ) or dict_risk_expo_new[i].index.max() < df_l_ret.index.max():
    #         raise Exception

    ########################step3:calculate factor return########################

    ls_df_fitresult = {
        dict_date_map[date]:
        Regression(date, dict_df_ret, dict_df_fexpo_con, dict_df_capital,
                   dict_df_fexpo, dict_date_map)
        for date in ls_alldates_ondaybefore
    }

    ls_df_facreturn = list(ls_df_fitresult[date]['params'].rename(
        columns={'params': date}) for date in ls_alldates)
    df_model_params = reduce(
        lambda df_para1, df_para2: pd.concat([df_para1, df_para2], axis=1),
        ls_df_facreturn)

    ########################step4:calculate factor return covariance########################

    df_allfactorret = df_model_params.T
    df_allfactorret = df_allfactorret.sort_index()

    corrhalflife = int(corr_half_life)
    varhalflife = int(var_half_life)

    halflife = max(corrhalflife, varhalflife)

    if len(ls_alldates) < halflife:
        raise Exception("More data needed")
    else:
        ls_alldatesnew = ls_alldates[halflife - 1:len(ls_alldates)]
        corrwgts = list(
            map(lambda x: mt.sqrt(0.5**(x / int(corrhalflife))),
                list(range(int(corrhalflife) - 1, -1, -1))))
        varwgts = list(
            map(lambda x: mt.sqrt(0.5**(x / int(varhalflife))),
                list(range(int(varhalflife) - 1, -1, -1))))

        ls_factorretcov = list(
            calcfactorRetCov(df_allfactorret, date, corrwgts,
                             varwgts, corrhalflife, varhalflife)
            for date in ls_alldatesnew)
        df_l_factorretcov = pd.concat(
            ls_factorretcov, axis=0).rename(columns={'variable': 'factorid2'})

        ########################step5:calculate the residual(specific) variances of regression########################

        ##part1:merge factorreturn,factor exposure and stock return
        ls_specificrisk = list(ls_df_fitresult[date]['resid'].rename(
            columns={'resid': date}) for date in ls_alldates)
        df_w_specificrisk = pd.concat(ls_specificrisk, axis=1).T
        df_w_specificrisk = df_w_specificrisk.sort_index()
        specificwgts = list(
            map(lambda x: mt.sqrt(0.5**(x / int(halflife))),
                list(range(int(halflife) - 1, -1, -1))))

        ls_factorretspe = list(
            calcfactorRetSpe(df_w_specificrisk, date, specificwgts, halflife)
            for date in ls_alldatesnew)
        df_specificrisk_var = pd.concat(ls_factorretspe, axis=0)

        ########################step6:generate final return value########################
        df_allfactorret = df_allfactorret.drop('countryfactor', axis=1)
        dict_factorret = {
            key + '.ret': df_allfactorret[[key]].rename(columns={
                key:
                list(
                    gftIO.strSet2Np(
                        np.array(list(df_allfactorret[[key]].columns))))[0]
            })
            for key in df_allfactorret.columns
        }
        dictMerged = dict(dict_factorret, **{
            'ret_cov': df_l_factorretcov,
            'specificRisk': df_specificrisk_var
        })
        return dictMerged


def Regression(date, dict_df_ret, dict_df_fexpo_con, dict_df_capital,
               dict_df_fexpo, dict_date_map):
    dateadd = dict_date_map[date]

    Y = np.array(dict_df_ret[dateadd][dateadd])
    X = np.array(dict_df_fexpo_con[date])

    model = sm.RLM(Y, X, M=sm.robust.norms.HuberT())
    results = model.fit()
    df_model_params = pd.DataFrame(results.params, columns=['params'])
    df_model_params.index = dict_df_fexpo_con[date].columns

    df_model_params = df_model_params.ix[0:37,]
    df_model_resid = pd.DataFrame(results.resid, columns=['resid'])
    df_model_resid.index = dict_df_fexpo_con[date].index
    df_model_resid = df_model_resid.reindex(dict_df_fexpo[date].index)
    df_model_resid = df_model_resid.multiply(
        1 / dict_df_capital[date].squeeze(), axis='index')

    return {'params': df_model_params, 'resid': df_model_resid}


def calcfactorRetCov(df_allfactorret, date, corrwgts, varwgts, corrhalflife,
                     varhalflife):
    ##calculate corr
    df_factorretcorr = df_allfactorret[df_allfactorret.index <=
                                       date][-corrhalflife:]
    df_retcorr = df_factorretcorr.apply(
        lambda x: np.array(x) * np.array(corrwgts)).corr()
    ##calculate standard deviation
    df_factorretstd = df_allfactorret[df_allfactorret.index <=
                                      date][-varhalflife:]
    df_retstd = df_factorretstd.apply(
        lambda x: np.array(x) * np.array(varwgts)).std()
    ##calculate covariance
    df_retcov = df_retcorr.apply(
        lambda x: np.array(x) * np.array(df_retstd)).T.apply(
            lambda x: np.array(x) * np.array(df_retstd))

    df_retcov['factorid1'] = df_retcov.index
    df_l_factorretcov = pd.melt(df_retcov, id_vars=['factorid1'])
    df_l_factorretcov['date'] = date

    ssb_map = pd.DataFrame(
        data=list(
            set(df_l_factorretcov['factorid1'][df_l_factorretcov['factorid1'] !=
                                               'countryfactor'])),
        columns=['oriname'])
    ssb_map = ssb_map[ssb_map['oriname'] != 'countryfactor']
    ssb_map['sname'] = list(gftIO.strSet2Np(np.array(ssb_map['oriname'])))
    dict_ssb_map = {
        key: list(ssb_map['sname'][ssb_map['oriname'] == key])[0]
        for key in ssb_map['oriname']
    }
    dict_ssb_map['countryfactor'] = 'countryfactor'

    df_l_factorretcov['factorid1'] = df_l_factorretcov['factorid1'].apply(
        lambda x: dict_ssb_map[x])
    df_l_factorretcov['variable'] = df_l_factorretcov['variable'].apply(
        lambda x: dict_ssb_map[x])
    df_l_factorretcov = df_l_factorretcov[
        df_l_factorretcov['factorid1'] !=
        'countryfactor'][df_l_factorretcov['variable'] != 'countryfactor']

    return df_l_factorretcov


def calcfactorRetSpe(df_w_specificrisk, date, specificwgts, halflife):
    df_residualspe = df_w_specificrisk[df_w_specificrisk.index <=
                                       date][-halflife:]
    df_retspe = df_residualspe.apply(
        lambda x: np.array(x) * np.array(specificwgts)).var()
    df_retspenew = pd.DataFrame(df_retspe, columns=['specificrisk'])
    df_retspenew['symbol'] = df_retspenew.index
    df_retspenew = df_retspenew.reset_index().assign(date=date).drop(
        'index', axis=1)

    return df_retspenew


def fexpomerge(dict_risk_expo_new, date, allfactor, ls_allsymbols):
    ls_raw_df_fexpo = [
        dict_risk_expo_new[factorname].reindex(
            index=[date], columns=ls_allsymbols[date]).rename(
                index={date: factorname}) for factorname in allfactor
    ]
    df_fexpo_onedate = pd.concat(ls_raw_df_fexpo, axis=0).T.fillna(0)
    return df_fexpo_onedate


def expoconstrain(dict_df_fexpo_raw, date, ind_factor_name, allfactor,
                  dict_df_capital_raw, sty_factor_name, dict_df_fexpo):
    df_fexpo_date = dict_df_fexpo_raw[date].reindex(
        columns=ind_factor_name).multiply(
            dict_df_capital_raw[date].squeeze(), axis='index')
    df_wgt_con = pd.DataFrame(df_fexpo_date.sum(axis=0)).T.rename(
        index={0: 'constrain'})

    df_con_add = pd.DataFrame(
        (np.zeros([1, len(allfactor) - len(ind_factor_name)])),
        index=['constrain'],
        columns=sty_factor_name)
    df_wgt_con_fnl = pd.concat([df_wgt_con, df_con_add], axis=1)
    return pd.concat(
        [dict_df_fexpo[date],
         df_wgt_con_fnl.assign(countryfactor=0)], axis=0)


"""----------------------------------------------------------------------"""
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    risk_model_path = '/home/weiwu/share/risk_model/'

    # keep from double loading
    stock_return = gftIO.zload(
        os.path.join(risk_model_path, 'stock_return.pkl'))
    factors = gftIO.zload(os.path.join(risk_model_path, 'factors.pkl'))
    market_capital = gftIO.zload(
        os.path.join(risk_model_path, 'market_capital.pkl'))
    corr_half_life = gftIO.zload(
        os.path.join(risk_model_path, 'corr_half_life.pkl'))
    var_half_life = gftIO.zload(
        os.path.join(risk_model_path, 'var_half_life.pkl'))

# model = risk_model(stock_return, factors, market_capital, corr_half_life,
#                   var_half_life)
logger.debug('parse data')

# get all factor names
ls_fexponame = factors['osets'].asColumnTab()['O0'].apply(
    gftIO.gidInt2Str).tolist()
ind_factor_name = factors[ls_fexponame[0]].asColumnTab()['O0'].apply(
    gftIO.gidInt2Str).tolist()
style_factor_name = factors[ls_fexponame[1]].asColumnTab()['O0'].apply(
    gftIO.gidInt2Str).tolist()

allfactor = ind_factor_name + style_factor_name

##stock return preprocess
if isinstance(stock_return, gftIO.GftTable):
    # df_w_ret = stock_return.asMatrix().T.dropna(how='all', axis=1)
    df_stock_return = stock_return.asMatrix().dropna(axis=1, how='all')
##stock market capital preprocess
if isinstance(market_capital, gftIO.GftTable):
    # df_w_ret = stock_return.asMatrix().T.dropna(how='all', axis=1)
    market_capital = market_capital.asMatrix().dropna(axis=1, how='all').copy()
    market_capital.fillna(method='ffill', inplace=True)
# get factor exposure date list(all snapshots)
# logger.debug('pack factors to dictionary')
# dict_risk_expo_new = {
#     factorname: factors[factorname].asMatrix().dropna(how='all')
#     for factorname in allfactor
# }

# parse all factors to a list
df_sqrt_cap = market_capital.apply(np.sqrt)
ls_pd_ind_factors = [
    factors[fac].asMatrix().dropna(how='all') * df_sqrt_cap
    for fac in ind_factor_name
]
ls_pd_style_factors = [
    factors[fac].asMatrix().dropna(how='all') for fac in style_factor_name
]
for num, fac in enumerate(ls_pd_ind_factors):
    ls_pd_ind_factors[num]['constraint'] = ls_pd_ind_factors[num].sum(axis=0)
for num, fac in enumerate(ls_pd_style_factors):
    ls_pd_style_factors[num]['constraint'] = 0
ls_pd_all_factors = ls_pd_style_factors + ls_pd_style_factors

# get list of all intersected stock symbols
ls_all_stocks = reduce(pd.Index.intersection,
                       [factors[fac].asMatrix().columns for fac in allfactor])
# ls_all_stocks = [factors[fac].asMatrix().columns for fac in allfactor]

# get list of all intersected dates
ls_all_dates = reduce(pd.Index.intersection,
                      [factors[fac].asMatrix().index for fac in allfactor])

# create country factor
df_country_factor = pd.DataFrame(0.0, index=ls_all_dates, columns=ls_all_stocks)
ls_pd_all_factors.append(df_country_factor)

# reset index of each factor dataframe
for num, fac in enumerate(ls_pd_all_factors):
    ls_pd_all_factors[num] = fac.reindex(
        index=ls_all_dates, columns=ls_all_stocks)
allfactor.append('country')

# create 3d panel
logger.debug('convert all factors&stocks to pandas panel')
pd_panel_factor = pd.Panel(
    {allfactor[key]: factor
     for key, factor in enumerate(ls_pd_all_factors)}).transpose(1, 2, 0)

# Solve the problem of heteroskedasticity by square root the market capitalization.
df_sqrt_return = market_capital.apply(np.sqrt) * df_stock_return
df_sqrt_cap_return = df_sqrt_return.reindex(
    index=ls_all_dates, columns=ls_all_stocks)
# df_sqrt_cap_return['constraint'] = 0

# add constraint to all factors
logger.debug('add constraint to all factors')
