# -*- coding: utf-8 -*-
from lib.gftTools import gftIO
import numpy as np
import pandas as pd

def getCashGid():
    return gftIO.strSet2Np(np.array(['0AC062D610A1481FA5561EC286146BCC']))


def getGodGid():
    return np.chararray(1, itemsize=16, buffer='\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0')


def getSingleGodGid():
    iv = 0
    return iv.to_bytes(16,byteorder='little')

def getParm(dict_parm, parmName, defaultValue):
    if parmName in dict_parm:
        return dict_parm[parmName]
    else:
        return defaultValue


def alignDate(sourceDates, targetDates, method='ffill', returnidx=False):
    """
    source dates: rebalance dates
    target dates: all dates
    """

    df_dateMap = pd.DataFrame({'targetDate':targetDates, 'idx':np.arange(len(targetDates))}, index=targetDates)
    if isinstance(sourceDates, pd.tslib.Timestamp):
        df_dateMap = df_dateMap.reindex([sourceDates], method=method)
    else:
        df_dateMap = df_dateMap.reindex(sourceDates, method=method)
    if returnidx:
        result = np.array(df_dateMap['idx'])
        if isinstance(sourceDates, pd.tslib.Timestamp):
            result = result[0]
    else:
        if isinstance(sourceDates, pd.tslib.Timestamp):
            result = df_dateMap['targetDate'][0]
        else:
            if isinstance(sourceDates, pd.core.indexes.datetimes.DatetimeIndex):
                result = pd.DatetimeIndex(df_dateMap['targetDate'])
            else:
                result = np.array(df_dateMap['targetDate'])
    return result


def roundToLot(sizeArray, lotSize):
    if lotSize > 0:
        sizeArray = sizeArray.fillna(0)
        return np.sign(sizeArray) * lotSize * np.floor(np.round(abs(sizeArray)) / max(1., lotSize))
    else:
        return sizeArray


def cut2bin(ser_x, totalBinNum, ascending=False):
    # calculate bin size
    totalBinNum = int(totalBinNum)
    xlen = len(ser_x)
    arr_binsize = np.repeat(xlen // totalBinNum, totalBinNum)
    remaining = xlen % totalBinNum
    if remaining > 0:
        arr_binsize[:remaining] += 1
    # map each signal to its binIdx
    arr_binmap = np.repeat(np.arange(totalBinNum) + 1, arr_binsize)
    ser_xrank = ser_x.rank(method='first', ascending=ascending)
    ser_result = pd.Series(
        arr_binmap[np.array(ser_xrank.values - 1, dtype='int')])
    return ser_result


class Strategy:
    #hold the strategy result, including cumret, holding
    def __init__(self, gid, cumret, holding):
        # save matrix and column table.
        self.gid = gid		
        self.cumret = cumret
        self.holding = holding
        self.type = "Strategy"
        self.cumretGid = None
        self.holdingGid = None




def isGodColumns(col):
    if col.size == 1:
        return col[0].__len__() == 0
    return False

class InputOperation:
    def __init__(self, just_do, left_value, right_value, order_changed):
        self.just_do = just_do
        self.left_value = left_value.input
        self.right_value = right_value.input
        self.order_changed = order_changed


class matrixCommonInfo:
    def __init__(self, rawInput):
        if isinstance(rawInput, gftIO.GftTable):
            rawInput = rawInput.asMatrix()
            self.is_df = True
        elif isinstance(rawInput, gftIO.GftTable):
            self.is_df = True
        else:
            self.is_df = False

        if self.is_df:
            self.is_nonsymbol = isGodColumns(rawInput.columns)
            if self.is_nonsymbol:
                rawInput = rawInput[rawInput.columns[0]]
        else:
            self.is_nonsymbol = False
        self.input = rawInput

    def align_matrix(self, common_index, common_columns):
        if self.is_df:
            if common_index is not None:
                self.input = self.input.reindex(common_index)
            if not self.is_nonsymbol and common_columns is not None:
                if common_columns is not None:
                    self.input = self.input[common_columns]

    def get_operation(self, another):
        if self.is_nonsymbol:
            if (another.is_df and not another.is_nonsymbol):
                return InputOperation(False, another, self, True)
        elif self.is_df:
            if another.is_nonsymbol:
                return InputOperation(False, self, another, False)

        return InputOperation(True, self, another, False)


def merge_info_inplace(info_list):
    common_index = None
    common_columns = None
    for info in info_list:
        if info.is_df:
            if common_index is None:
                common_index = info.input.index
            else:
                common_index = np.intersect1d(common_index, info.input.index)
            if not info.is_nonsymbol:
                if common_columns is None:
                    common_columns = info.input.columns
                else:
                    common_columns = np.intersect1d(common_columns, info.input.columns)

    if (common_index is not None) or (common_columns is not None):
        for info in info_list:
            info.align_matrix(common_index, common_columns)

    return info_list


def align_input(*inputs):
    input_list = []
    for input in inputs:
        input_list.append(matrixCommonInfo(input))
    return merge_info_inplace(input_list)

def classify(context, df_x,df_y,winSize,winStep, clf):
    # bunch of scoresssss
    from lib.gftTools import gsUtils
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    '''
    -------- parameter----------
    DataFrame:{columns=["date","x1","x2", ..., "xn"]}
    DataFrame:{columns=["date","y"]}
    winSize: float
    winSteop: float
    
    ---------return----------
    DataFrame:{columns=["date","y"]}
    
    assumption:
    1. 'xi' has been sorted by 'date'
    2. 'y' cloumn in 'X0' has been shifted
        
    '''

    if isinstance(df_x, gftIO.GftTable):
        df_x = df_x.asColumnTab()
    if isinstance(df_y,dict):
        df_y = df_y["y"]
    if isinstance(df_y, gftIO.GftTable):
        df_y = df_y.asColumnTab()
    
    # convert parameter type
    winSize = int(winSize)
    winStep = int(winStep) 
    
    # NOTICE: integer will be regraged as O by GS, but classifier need int
    value_column = gsUtils._findValueColumn(df_y.columns)  # value_column: value  
                                                   # df_y.columns:Index(['date', 'value'], dtype='object')
    df_y.rename(columns={value_column:"y"},inplace=True)
    df_y.y=pd.factorize(df_y.y)[0]
    # change column name
    for col_name in df_y.columns:
        if isinstance(df_y.ix[0,col_name],pd.Timestamp):
            df_y.rename(columns={col_name:"date"},inplace=True)
            break
    # remove meanless columns
    df_y=df_y[["date","y"]]
    
    # merge data
    df_x = df_x.sort_values("date",ascending=True)
    df_y = df_y.sort_values("date",ascending=True)
    df_y = df_y.set_index(np.arange(len(df_y))) # indentify index: start from 0

    # frequency error: if y_freq > x_freq, meanless data
    ls_missing_date=[d for d in list(df_y["date"]) if d not in list(df_x["date"])]
    if len(ls_missing_date)>0:
        raise ValueError("y_freq > x_freq, meanless data")
    
    # slice data: remove redundant x
    if len(df_x)!=len(df_y):
        ls_slice_data=[d for d in list(df_x["date"]) if d not in list(df_y["date"])]
        df_tmp_x=df_x.set_index(["date"])
        df_tmp_x=df_tmp_x.drop(ls_slice_data)
        df_x=df_tmp_x.reset_index(np.arange(len(df_tmp_x)),drop=False)
    
    # identify index: start from 0
    df_x = df_x.set_index(np.arange(len(df_x)))
    df_y = df_y.set_index(np.arange(len(df_y))) 

    # data to be trained
    df_data=pd.merge_ordered(df_x,df_y,on="date",how="outer") 

    # value check
    if len(df_data.index) < winSize + 1:
        raise ValueError("the number of input data is not enough")
    
    # rooling
    ls_predicted=[]
    for i in range(len(df_data.index)):
        if i<winSize:
            ls_predicted+=[np.nan]
        else:
            start_index=i-winSize
            # fit
            n_x_train= df_data.iloc[start_index:i,1:-1].values
            n_y_train= df_data.iloc[start_index:i,-1].values
            clf.fit(n_x_train, n_y_train)
            # predict
            n_x_test = df_data.iloc[[i],1:-1]
            y_test = clf.predict(n_x_test)[0]
            ls_predicted += [y_test]
    
    df_data["predicted"]=ls_predicted
    #print(ls_predicted)
    
    # drop na
    df_data=df_data.dropna()
    #print(df_data)
    
    # scoressssssss
    y_true=pd.factorize(df_data["y"])[0]
    y_pred=pd.factorize(df_data["predicted"])[0]
    num_accuracy_score=accuracy_score(y_true,y_pred)
    #print("accuracy_score:",num_accuracy_score)
    num_f1_score=f1_score(y_true,y_pred,average='macro') # micor, weighted, None
    #print("f1_score:",num_f1_score)
    num_precision_score=precision_score(y_true, y_pred, average='macro') # micor, weighted, None
    #print("precision_score:",num_precision_score)
    num_recall_score=recall_score(y_true, y_pred, average='macro') # micor, weighted, None
    #print("recall_score:",num_recall_score)
    dict_score={"accuracy_score":num_accuracy_score, "f1_score":num_f1_score,"precision_score":num_precision_score, "recall_score":num_recall_score}
    
    # score
    y_test = df_data["predicted"].values
    X_test = df_data.iloc[:,1:-2].values
    num_mean_accuracy=clf.score(X_test , y_test)
    #print(num_score)    
    
    '''
    # feature_importances
    ls_fitness=list(zip(df_data.iloc[:,1:-1],clf.feature_importances_))
    n_fitness=np.array(list(map(list,ls_fitness)))
    df_fitness=pd.DataFrame({"feature":n_fitness[:,0],"importance":n_fitness[:,1]})
    #print(df_fitness)    
    '''
    
    # result
    df_data=df_data[["date","predicted"]]
    #print(df_data)
    
    dict_result = {"result":df_data,"mean_accuracy":num_mean_accuracy, "scores":dict_score} #,"fitness":df_fitness}
    #print(dict_result)
    return dict_result


def _findValueColumn(ls_columns):
    for acolumn in ls_columns:
        if acolumn.upper() in ["VALUE","VAL","V"]:
            return acolumn
    raise ValueError("Value Column isnot found in {}!".format(ls_columns))
