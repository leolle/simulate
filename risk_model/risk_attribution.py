# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:21:31 2017

@author: gft
"""

##change information
##当前要计算的日期范围和 传入的risk model计算的日期范围不一致，取最新一期的权重

from lib.gftTools import gftIO
x0 = gftIO.zload("x0.pkl")
x1 = gftIO.zload("x1.pkl")
x2 = gftIO.zload("x2.pkl")
x3 = gftIO.zload("x3.pkl")
x4 = gftIO.zload("x4.pkl")
'''
49EFD5C6530545618490610BE4103358 merge_dicts by smp
0D9E9789D2EFE27F77B8AE71F7519EF0 riskmodel_merge by wjj



newest gid:2F8CFDC0CC008CF1F219537670A4C57C
'''
'''

from lib.gftTools import gftIO
x0 = gftIO.zload("x0.pkl")
x1 = gftIO.zload("x1.pkl")
x2 = gftIO.zload("x2.pkl")
x3 = gftIO.zload("x3.pkl")
x4 = gftIO.zload("x4.pkl")

x3
x4

    datestrs=['2016-06-01']
    x3=pd.to_datetime(datestrs)[0]

    datestrs=['2016-06-30']
    x4=pd.to_datetime(datestrs)[0]

#'gid:319275B73EA39BDB1D5C516328985787'
'''

from lib.gftTools import gftIO
from lib.gftTools import gsConst
import numpy as np
import pandas as pd
from functools import reduce
import math as mt


class Riskreport(object):

    def __init__(self, df_expo, df_w, df_covm, df_specrisk, ls_industryfactor,
                 ls_stylefactor):
        self.df_expo = df_expo
        self.df_w = df_w
        self.df_covm = df_covm
        self.df_specrisk = df_specrisk
        self.ls_industryfactor = ls_industryfactor
        self.ls_stylefactor = ls_stylefactor

    def calcX(self):  ##portfolio exposure((X.T*W))
        return (self.df_expo.T).dot(self.df_w)

    def calcPortCommonRisk(self):  ###((X.T*W).T)*F*((X.T*W))
        self.calcX()[np.isnan(self.calcX())] = 0
        return float(
            (((self.calcX().T).dot(self.df_covm)).dot(self.calcX())).ix[0, 0])

    def calcPortSpecificRisk(
            self):  ###W.T * DELTA(residual return variance matrix) *W
        return float(
            (((self.df_w.T).dot(self.df_specrisk)).dot(self.df_w)).ix[0, 0])

    def calcTotalRisk(self):
        return (self.calcPortCommonRisk() + self.calcPortSpecificRisk())

    def calcSigmaRisk(self):
        return (mt.sqrt(self.calcTotalRisk()))

    def calcTotalRiskall(self, f_varp, columnname):
        f_var = self.calcTotalRisk()
        f_sigma = self.calcSigmaRisk()
        try:
            f_percent = f_var / f_varp
        except:
            f_percent = np.nan
        df_riskall = pd.DataFrame(
            data=([[f_var, f_sigma, f_percent]]), columns=columnname)
        return df_riskall

    def calcFMCR(self):
        try:
            xpa = self.calcX()
            xpa[np.isnan(xpa)] = 0
            df_allcovm = self.df_covm
            ls_allfactor = self.ls_industryfactor + self.ls_stylefactor

            dict_factorcontrib = [
                pd.DataFrame(
                    data=float(
                        (((xpa.T[[i]]).dot(df_allcovm[df_allcovm.index == i])
                         ).dot(xpa)).ix[0, 0]),
                    index=[i],
                    columns=['value']) for i in ls_allfactor
            ]
            ###calculate single factor contribution
            ##()

            df_fmcar = pd.concat(dict_factorcontrib, axis=0)

            df_sigmafmcar = pd.DataFrame(df_fmcar['value'].apply(
                lambda x: np.sqrt(x) if x >= 0 else np.nan))

            df_fmcar_raw = self.df_covm.dot(self.calcX()) / self.calcSigmaRisk()
            df_percenfmcar = self.calcX() * df_fmcar_raw / self.calcSigmaRisk()

            df_fmcar.rename(columns={'value': 'FMCAR'}, inplace=True)
            df_sigmafmcar.rename(columns={'value': 'sigmaFMCAR'}, inplace=True)
            df_percenfmcar.rename(columns={'Wa': 'percenFMCAR'}, inplace=True)
            df_fmcarall = pd.concat(
                [df_fmcar, df_sigmafmcar, df_percenfmcar], axis=1)

            return df_fmcarall
        except:
            return np.nan

    def calcIndustryFMCAR(self):
        return (self.calcFMCR().reindex(self.ls_industryfactor))

    def calcstyleFMCAR(self):
        return (self.calcFMCR().reindex(self.ls_stylefactor))

    def calcPortCommonRiskall(self, f_vara, indexname):
        f_varfactor = self.calcPortCommonRisk()
        f_sigmafactor = mt.sqrt(f_varfactor)
        try:
            f_percenfactor = f_varfactor / f_vara
        except:
            f_percenfactor = np.nan

        df_factor = pd.DataFrame(
            data=[[f_varfactor, f_sigmafactor, f_percenfactor]],
            columns=['VarFactor', 'sigmaFactor', 'percenFactor'])
        return df_factor

    def calcPortSpecRiskall(self, f_vara, indexname):
        f_varss = self.calcPortSpecificRisk()
        f_sigmass = mt.sqrt(f_varss)
        try:
            f_percenss = f_varss / f_vara
        except:
            f_percenss = np.nan

        df_ss = pd.DataFrame(
            data=[[f_varss, f_sigmass, f_percenss]],
            columns=['VarSS', 'sigmaSS', 'percenSS'])
        return df_ss


class Riskmodel(object):

    def __init__(self, data, dt_selecteddate, allsymbol):
        self.data = data
        self.selectedate = dt_selecteddate
        self.allsymbol = allsymbol

    def getfactorlist(self, i):
        return (self.data[gftIO.gidInt2Str(self.data['osets'].asColumnTab().ix[
            i, 0])].asColumnTab()['O0']
               ).apply(lambda x: gftIO.gidInt2Str(x)).tolist()

    def factorcnt(self):
        return len(self.data['osets'].asColumnTab())

    def getallFactor(self):
        allfactor = []
        for i in range(self.factorcnt()):
            allfactor.extend(self.getfactorlist(i))
        return allfactor

    def selectData(self):
        dict_fexpo = dict([(factorname,
                            self.data[factorname].asMatrix().reindex(
                                columns=self.allsymbol))
                           for factorname in self.getallFactor()])
        return dict_fexpo

    def Fexpomerge(self):
        dt_latest = self.selectedate
        ls_raw_df_fexpo = [
            self.selectData()[factorname].reindex(index=[dt_latest]).rename(
                index={dt_latest: factorname})
            for factorname in self.getallFactor()
        ]
        df_fexpo_onedate = pd.concat(ls_raw_df_fexpo, axis=0).fillna(0)
        df_fexpo_onedate.index = list(
            gftIO.strSet2Np(np.array(df_fexpo_onedate.index)))
        return df_fexpo_onedate.T


def calcRiskAttribution(dict_riskmodel, df_portwgt, df_benchwgt, dt_startdate,
                        dt_enddate):
    ########################step1:parameter description########################

    #dict_riskmodel:type:dict:dict_riskmodel=x0
    #df_portwgt:type:dataframe df_portwgt=x1
    #df_benchwgt:type:dataframe  df_benchwgt=x2
    #dt_startdate:type:timestamp:dt_startdate=x3
    #dt_enddate type:dt_enddate =x4

    ########################step2:portfolio,benchmark,portfolio active data prepare########################

    ##part1:get portfolio,benchmark,portfolio active weight
    df_wp = df_portwgt.asColumnTab().rename(columns={'value': 'Wp'})
    df_wb = df_benchwgt.asColumnTab().rename(columns={'value': 'Wb'})
    df_wa = pd.merge(df_wp, df_wb, on=['idname', 'variable'], how='outer')
    df_wa.fillna(0, inplace=True)
    df_wa.set_index('variable', inplace=True)
    df_wa['Wa'] = df_wa['Wp'] - df_wa['Wb']
    ##part2:get portfolio,benchmark,portfolio active stock pool
    ls_aname = list(
        set(df_portwgt.asMatrix().columns).union(
            set(df_benchwgt.asMatrix().columns)))
    ls_aname.remove('idname')

    ########################step2:data preprocessing########################
    ##part1:date preprocessing
    ##get factor name
    rmodelobj = Riskmodel(dict_riskmodel, dt_startdate, ls_aname)

    if rmodelobj.factorcnt() < 2:
        ls_stylefactor = list(
            gftIO.strSet2Np(np.array(rmodelobj.getfactorlist(0))))
        ls_industryfactor = []
    else:
        ls_industryfactor = list(
            gftIO.strSet2Np(np.array(rmodelobj.getfactorlist(0))))
        ls_stylefactor = list(
            gftIO.strSet2Np(np.array(rmodelobj.getfactorlist(1))))

    ls_allfactor = list(gftIO.strSet2Np(np.array(rmodelobj.getallFactor())))

    ##part2:factor loading preprocessing
    dict_risk_expo_new = {
        factorname: dict_riskmodel[factorname].asMatrix().dropna(how='all')
        for factorname in list(np.array(rmodelobj.getallFactor()))
    }
    ls_ls_fexpodate = list([
        dict_risk_expo_new[factorname].index.tolist()
        for factorname in dict_risk_expo_new.keys()
    ])
    ls_alldates_fexpo = reduce(np.intersect1d, ls_ls_fexpodate)

    ##part3:covM preprocessing
    df_covm = dict_riskmodel['ret_cov']  ##factor covariance matrix
    df_covm = df_covm.reset_index().reindex(
        columns=['factorid1', 'factorid2', 'value', 'date'])

    df_covm = df_covm[df_covm['factorid1'].isin(ls_allfactor)][df_covm[
        'factorid2'].isin(ls_allfactor)]

    ##part4:specRisk preprocessing
    df_specrisk_raw = dict_riskmodel['specificRisk']

    ## make sure all the data source have the same date range
    #ls_date_range_new=list(pd.period_range(dt_startdate, dt_enddate, freq='D').to_timestamp()) ##for date range check
    #df_date_range=pd.DataFrame(data=ls_date_range_new,columns=['needdate'])

    ls_port_wgt = df_portwgt.asMatrix().index

    sourceDates = sorted(
        list(ls_port_wgt[(ls_port_wgt <= dt_enddate) & (ls_port_wgt >=
                                                        dt_startdate)]))

    ###align daterange
    ##covm specrisk align
    targetDates = sorted(list(np.unique(df_covm['date'])))
    df_date_map_covm = pd.DataFrame(
        {
            'targetDate': targetDates
        }, index=targetDates)
    df_date_map_covm = df_date_map_covm.reindex(
        sourceDates, method='ffill').dropna(how='any')
    dict_date_map_covm = {
        df_date_map_covm.index[i]: df_date_map_covm.targetDate[i]
        for i in range(len(df_date_map_covm))
    }

    ##factor align
    targetDates = sorted(list(ls_alldates_fexpo))
    df_date_map_expo = pd.DataFrame(
        {
            'targetDate': targetDates
        }, index=targetDates)
    df_date_map_expo = df_date_map_expo.reindex(
        sourceDates, method='ffill').dropna(how='any')
    dict_date_map_expo = {
        df_date_map_expo.index[i]: df_date_map_expo.targetDate[i]
        for i in range(len(df_date_map_expo))
    }

    ls_date_range = list(
        set(dict_date_map_covm.keys()).intersection(
            set(dict_date_map_expo.keys())))

    if len(ls_date_range) == 0:
        raise Exception(
            "date length is null,risk model and risk decom didn't match")

    ##covm
    dict_df_covm = dict([(dt_selecteddate, df_covm[df_covm[
        'date'] == dict_date_map_covm[dt_selecteddate]].pivot_table(
            values='value', index='factorid1', columns='factorid2'))
                         for dt_selecteddate in ls_date_range])

    ##specrisk
    df_specrisk_raw = df_specrisk_raw.reset_index().pivot(
        'date', 'symbol', 'specificrisk')
    df_specrisk_raw1 = df_specrisk_raw.reindex(columns=ls_aname).T
    df_specrisk = df_specrisk_raw1.fillna(
        df_specrisk_raw1.mean(axis=0, skipna=True))
    dict_df_specrisk = {
        date: pd.DataFrame(
            np.diag(df_specrisk[dict_date_map_covm[date]].tolist()),
            index=ls_aname,
            columns=ls_aname)
        for date in ls_date_range
    }

    ##weight
    dict_df_wa = {
        date: df_wa[df_wa.idname == date].reindex(ls_aname).fillna(0)
        for date in ls_date_range
    }

    ##factor exposure
    dict_rmodelobj = dict([(dt_selecteddate, Riskmodel(
        dict_riskmodel, dict_date_map_expo[dt_selecteddate], ls_aname))
                           for dt_selecteddate in ls_date_range])
    dict_df_fexpo = dict([(dt_selecteddate,
                           dict_rmodelobj[dt_selecteddate].Fexpomerge())
                          for dt_selecteddate in dict_rmodelobj.keys()])
    dict_final_return = {}

    ##create date-strdate dictionary
    for date in ls_date_range:
        ##part4:slice portfolio active的 specific risk
        df_specriska_singledate = dict_df_specrisk[date].fillna(0)
        df_wa_singledate = dict_df_wa[date]
        df_covm_singledate = dict_df_covm[date]
        dict_df_fexpo_singledate = dict_df_fexpo[date]
        ##part5:calculate portfolio,benchmark,portfolio active risk

        p_riskreport = Riskreport(dict_df_fexpo_singledate, df_wa_singledate[[
            'Wp'
        ]], df_covm_singledate, df_specriska_singledate, ls_industryfactor,
                                  ls_stylefactor)
        b_riskreport = Riskreport(dict_df_fexpo_singledate, df_wa_singledate[[
            'Wb'
        ]], df_covm_singledate, df_specriska_singledate, ls_industryfactor,
                                  ls_stylefactor)
        a_riskreport = Riskreport(dict_df_fexpo_singledate, df_wa_singledate[[
            'Wa'
        ]], df_covm_singledate, df_specriska_singledate, ls_industryfactor,
                                  ls_stylefactor)

        ##portfolio total risk
        f_varp = p_riskreport.calcTotalRisk()
        f_vara = a_riskreport.calcTotalRisk()
        ##interaction risk
        f_varinter = f_varp - b_riskreport.calcTotalRisk() - f_vara
        if f_varinter < 0:
            f_sigmainter = np.nan
        else:
            f_sigmainter = mt.sqrt(f_varinter)

        try:
            f_perceninter = f_varinter / f_varp
        except:
            f_perceninter = np.nan

        df_inter = pd.DataFrame(
            data=([[f_varinter, f_sigmainter, f_perceninter]]),
            columns=['VarInter', 'sigmaInter', 'percenInter'])

        xpa = a_riskreport.calcX()
        xpa[np.isnan(xpa)] = 0
        df_allcovm = a_riskreport.df_covm
        dict_factorcontrib = [
            pd.DataFrame(
                data=float((((xpa.T[[
                    i
                ]]).dot(df_allcovm[df_allcovm.index == i])).dot(xpa)).ix[0, 0]),
                index=[i],
                columns=['value']) for i in ls_allfactor
        ]
        df_l_factorcontrib = pd.concat(dict_factorcontrib, axis=0)

        if len(ls_industryfactor) > 0:
            df_TotIndustryFMCAR = pd.DataFrame(
                data=df_l_factorcontrib.reindex(index=ls_industryfactor).sum(),
                columns=['FMCAR'])
            df_TotIndustryFMCAR['percenFMCAR'] = df_TotIndustryFMCAR[
                'FMCAR'] / a_riskreport.calcTotalRisk()
            if df_TotIndustryFMCAR['FMCAR'][0] >= 0:
                df_TotIndustryFMCAR['sigmaFMCAR'] = np.sqrt(
                    df_TotIndustryFMCAR['FMCAR'][0])
            else:
                df_TotIndustryFMCAR['sigmaFMCAR'] = np.nan

            df_TotStyleFMCAR = pd.DataFrame(
                data=df_l_factorcontrib.reindex(index=ls_stylefactor).sum(),
                columns=['FMCAR'])
            df_TotStyleFMCAR['percenFMCAR'] = df_TotStyleFMCAR[
                'FMCAR'] / a_riskreport.calcTotalRisk()
            if df_TotStyleFMCAR['FMCAR'][0] >= 0:
                df_TotStyleFMCAR['sigmaFMCAR'] = np.sqrt(
                    df_TotStyleFMCAR['FMCAR'][0])
            else:
                df_TotStyleFMCAR['sigmaFMCAR'] = np.nan

            dict_final_return[date] = {
                gsConst.Const.PortRisk:
                p_riskreport.calcTotalRiskall(f_varp,
                                              ['VarP', 'sigmaP', 'percentP']),
                gsConst.Const.BenchmarkRisk:
                b_riskreport.calcTotalRiskall(f_varp,
                                              ['VarB', 'sigmaB', 'percentB']),
                gsConst.Const.PortActiveRisk:
                a_riskreport.calcTotalRiskall(f_varp,
                                              ['VarA', 'sigmaA', 'percentA']),
                gsConst.Const.InteractionRisk:
                df_inter,
                gsConst.Const.FactorRisk:
                a_riskreport.calcPortCommonRiskall(
                    f_vara, ['VarFactor', 'sigmaFactor', 'percenFactor']),
                gsConst.Const.SpecificRisk:
                a_riskreport.calcPortSpecRiskall(
                    f_vara, ['VarSS', 'sigmaSS', 'percenSS']),
                gsConst.Const.IndustryFMCAR:
                a_riskreport.calcIndustryFMCAR().reset_index(),
                gsConst.Const.StyleFMCAR:
                a_riskreport.calcstyleFMCAR().reset_index(),
                gsConst.Const.IndStyleFMCAR:
                a_riskreport.calcFMCR().reset_index(),
                gsConst.Const.PortExpo:
                p_riskreport.calcX().reset_index(),
                gsConst.Const.BenchmarkExpo:
                b_riskreport.calcX().reset_index(),
                gsConst.Const.PortExpoInd:
                p_riskreport.calcX().reindex(
                    index=ls_industryfactor).reset_index(),
                gsConst.Const.PortExpoSty:
                p_riskreport.calcX().reindex(
                    index=ls_stylefactor).reset_index(),
                gsConst.Const.BenchmarkExpoInd:
                b_riskreport.calcX().reindex(
                    index=ls_industryfactor).reset_index(),
                gsConst.Const.BenchmarkExpoSty:
                b_riskreport.calcX().reindex(
                    index=ls_stylefactor).reset_index(),
                gsConst.Const.TotIndustryFMCAR:
                df_TotIndustryFMCAR,
                gsConst.Const.TotStyleFMCAR:
                df_TotStyleFMCAR
            }
        else:

            df_TotStyleFMCAR = pd.DataFrame(
                data=df_l_factorcontrib.reindex(index=ls_stylefactor).sum(),
                columns=['FMCAR'])
            df_TotStyleFMCAR['percenFMCAR'] = df_TotStyleFMCAR[
                'FMCAR'] / a_riskreport.calcTotalRisk()
            if df_TotStyleFMCAR['FMCAR'][0] >= 0:
                df_TotStyleFMCAR['sigmaFMCAR'] = np.sqrt(
                    df_TotStyleFMCAR['FMCAR'][0])
            else:
                df_TotStyleFMCAR['sigmaFMCAR'] = np.nan

            dict_final_return[date] = {
                gsConst.Const.PortRisk:
                p_riskreport.calcTotalRiskall(f_varp,
                                              ['VarP', 'sigmaP', 'percentP']),
                gsConst.Const.BenchmarkRisk:
                b_riskreport.calcTotalRiskall(f_varp,
                                              ['VarB', 'sigmaB', 'percentB']),
                gsConst.Const.PortActiveRisk:
                a_riskreport.calcTotalRiskall(f_varp,
                                              ['VarA', 'sigmaA', 'percentA']),
                gsConst.Const.InteractionRisk:
                df_inter,
                gsConst.Const.FactorRisk:
                a_riskreport.calcPortCommonRiskall(
                    f_vara, ['VarFactor', 'sigmaFactor', 'percenFactor']),
                gsConst.Const.SpecificRisk:
                a_riskreport.calcPortSpecRiskall(
                    f_vara, ['VarSS', 'sigmaSS', 'percenSS']),
                gsConst.Const.StyleFMCAR:
                a_riskreport.calcstyleFMCAR().reset_index(),
                gsConst.Const.IndStyleFMCAR:
                a_riskreport.calcFMCR().reset_index(),
                gsConst.Const.PortExpo:
                p_riskreport.calcX().reset_index(),
                gsConst.Const.BenchmarkExpo:
                b_riskreport.calcX().reset_index(),
                gsConst.Const.TotStyleFMCAR:
                df_TotStyleFMCAR
            }

    dict_final_return_new = {}
    for i in list(dict_final_return[ls_date_range[0]].keys()):
        ls_final_return = [
            dict_final_return[date][i].assign(date=date)
            for date in ls_date_range
        ]
        dict_final_return_new[i] = pd.concat(ls_final_return, axis=0)

    return dict_final_return_new
    '''

    dict_riskmodel=x0
    df_portwgt=x1
    df_benchwgt=x2
    dt_startdate=x3
    dt_enddate =x4


    datestrs=['2016-06-01']
    x3=pd.to_datetime(datestrs)[0]

    datestrs=['2016-06-30']
    x4=pd.to_datetime(datestrs)[0]

    '''


dict_riskmodel = x0
df_portwgt = x1
df_benchwgt = x2
dt_startdate = x3
dt_enddate = x4

test = calcRiskAttribution(dict_riskmodel, df_portwgt, df_benchwgt,
                           dt_startdate, dt_enddate)
