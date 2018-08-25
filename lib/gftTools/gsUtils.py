# -*- coding: utf-8 -*-
from . import gftIO
import numpy as np
import pandas as pd
import xarray

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
    if isinstance(sourceDates, pd.Timestamp):
        df_dateMap = df_dateMap.reindex([sourceDates], method=method)
    else:
        df_dateMap = df_dateMap.reindex(sourceDates, method=method)
    if returnidx:
        result = np.array(df_dateMap['idx'])
        if isinstance(sourceDates, pd.Timestamp):
            result = result[0]
    else:
        if isinstance(sourceDates, pd.Timestamp):
            result = df_dateMap['targetDate'][0]
        else:
            if isinstance(sourceDates, pd.DatetimeIndex):
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
        elif isinstance(rawInput, pd.DataFrame):
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



def _findValueColumn(ls_columns):
    raise ValueError("Do not use this def, use find_value_column(data) instead.")

def find_value_column(data):
    for col_name in data.columns:
        if gftIO.get_column_type(data, col_name) == gftIO.PARAMETER_TYPE_NUMBER_NUMRIC:
            return col_name
    raise ValueError("Value Column is not found in {}!".format(data.columns.tolist()))

def find_date_column(data):
    for col_name in data.columns:
        if gftIO.get_column_type(data, col_name) == gftIO.PARAMETER_TYPE_TIMESTAMP:
            return col_name
    raise ValueError("Date Column isnot found in {}!".format(data.columns.tolist()))

class ExtractDictModelData(object):
    """ model data extraction and getting attribute. """
    def __init__(self, model):
        self.model = model

    def get_input_factor(self, oset_idx):
        """ Get oset idx from risk model.
        Keyword Arguments:
        oset_idx: list of oset gid
        """
        if len(oset_idx) < 1:
            return None
        date_index = self.model.get(oset_idx[0], None).asMatrix().index
        ls_factor_b_char = gftIO.strSet2Np(np.array(oset_idx))
        factor_data = pd.Panel({ls_factor_b_char[key]: self.model.get(factor).asMatrix() for key, factor in enumerate(oset_idx)})

        return factor_data.transpose(1, 2, 0)

    def get_output(self, post_fix, oset_idx=None):
        """ get target data from model

        Keyword Arguments:
        oset_idx: list of oset gid
        poset_fix: 'specificRisk', 'ret_cov', '*.ret'
        """
        if oset_idx is None:
            return self.model.get(post_fix, None)
        else:import pickle, zlib
import numpy as np
import pandas as pd
import binascii
import time
import random
import copy
from lib.gftTools import gsMeta


def zdump(value, filename):
    dps = pickle.dumps(value, -1)
    zDps = zlib.compress(dps, 9)
    with open(filename, "wb", -1) as fpz:
        fpz.write(zDps)


def getCacheHeader():
    # type of python pickle
    header = int(7)
    header <<= 32
    # current version
    header += 1225
    return header


cacheHeader = getCacheHeader()


def dump4CacheSever(value, timestamp, filename):
    dps = pickle.dumps(value, -1)
    zDps = zlib.compress(dps, 9)
    with open(filename, "wb", -1) as os:
        # use little endian
        binHeader = cacheHeader.to_bytes(8, byteorder='little')
        timeBytes = timestamp.to_bytes(8, byteorder='little')
        os.write(binHeader)
        os.write(timeBytes)
        os.write(zDps)
        return zDps.__len__()


def getDataAndType(value):
    data = pickle.loads(value)
    type = checkDataType(data)
    return type, data


def zdump4CacheSever(value, timestamp, filename):
    dps = pickle.dumps(value, -1)
    zDps = zlib.compress(dps, 9)
    with open(filename, "wb", -1) as os:
        # use little endian
        binHeader = cacheHeader.to_bytes(8, byteorder='little')
        timeBytes = timestamp.to_bytes(8, byteorder='little')
        os.write(binHeader)
        os.write(timeBytes)
        os.write(zDps)
        return 16 + zDps.__len__()


def zload(filename):
    with open(filename, "rb") as fpz:
        value = fpz.read()
        try:
            return pickle.loads(zlib.decompress(value), encoding="latin1")
        except:
            return pickle.loads(value)


inst_gid = None


def get_inst_gid():
    return inst_gid


# melt wide format -> long format
def convertMatrix2ColumnTab(matrix):
    if matrix.columns.size == 1:
        mat_copy = matrix.copy()
        col = oSet2Hex(mat_copy.columns.values)
        mat_copy.columns = col
        v_col_name = col[0]
        data_dic = {'T0':mat_copy.index, v_col_name:mat_copy[v_col_name].values}
        return pd.DataFrame(data_dic)
    matrix['O0'] = matrix.index
    return pd.melt(matrix, id_vars=['O0'])

def convertMatrix2ColumnTab(matrix : pd.DataFrame, o_col_name = 'O0'):
    if matrix.columns.size == 1:
        mat_copy = matrix.copy()
        col = oSet2Hex(mat_copy.columns.values)
        mat_copy.columns = col
        v_col_name = col[0]
        data_dic = {'T0': mat_copy.index, v_col_name: mat_copy[v_col_name].values}
        return pd.DataFrame(data_dic)
    matrix[o_col_name] = matrix.index
    return pd.melt(matrix, id_vars=[o_col_name])

# pivot long format -> wide format
def convertColumnTabl2Matrix(columnTab):
    tName = None
    vName = None
    oName = None
    for colName in columnTab.columns:
        if (columnTab[colName].dtype == np.float64):
            if vName is None:
                vName = colName
        if istimestamparray(np.array(columnTab[colName])):
            if tName is None:
                tName = colName
        if columnTab[colName].dtype == np.object:
            if oName is None:
                oName = colName

    if vName is None:
        raise Exception("v Name is None")

    if tName is None:
        raise Exception("tName is None")

    if oName is None:
        try:
            unique = columnTab.set_index(keys=tName, verify_integrity=True)
            unique.columns = np.chararray(1, itemsize=16, buffer='\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0')
            return unique
        except Exception:
            raise Exception("Can not transform TV to matrix, T is not unique")

    return columnTab.pivot_table(values=vName, index=tName, columns=oName)  ##将cov变成wide format格式


def isNonSymbol(df):
    if isinstance(df, pd.DataFrame):
        if ismatrix(df):
            return df.columns[0].__len__() == 0
    return False


# class saving data for both matrix(wide format) and columnTable(long format)
class GftTable:
    # about the added parameter matrixIsFromPython: in case matrixIsFromPython is not given, it will be None so the matrix will be transposed as before
    def __init__(self, matrix, columnTab, matrixIsFromPython, gid, columnOrders):
        # save matrix and column table.
        if (matrix is None):
            self.matrix = None
        elif matrixIsFromPython:
            self.matrix = matrix
        else:
            self.matrix = matrix.transpose()
        self.columnTab = columnTab
        self.type = "GftTable"
        self.gid = gid
        self.columnOrders = columnOrders

    def isNonSymbol(self):
        if self.matrix is not None:
            return self.matrix.columns[0].__len__() == 0
        return False

    @classmethod
    def fromCppMatrix(cls, matrix):
        return cls(matrix, None, False, None, None)

    @classmethod
    def fromCppMatrixWithGid(cls, matrix, gid):
        return cls(matrix, None, False, gid, None)

    @classmethod
    def fromPythonMatrix(cls, matrix):
        return cls(matrix, None, True, None, None)

    @classmethod
    def fromPythonMatrixWithGid(cls, matrix, gid):
        return cls(matrix, None, True, gid, None)

    @classmethod
    def fromColumnTable(cls, columnTable, columnOrders):
        return cls(None, columnTable, False, None, columnOrders)

    @classmethod
    def fromColumnTableWithGid(cls, columnTable, gid, columnOrders):
        return cls(None, columnTable, False, gid, columnOrders)

    @classmethod
    def fromBoth(cls, matrix, columnTable, matrixIsFromPython, columnOrders):
        return cls(matrix, columnTable, matrixIsFromPython, None, columnOrders)

    @classmethod
    def fromBothWithGid(cls, matrix, columnTable, matrixIsFromPython, gid, columnOrders):
        return cls(matrix, columnTable, matrixIsFromPython, gid, columnOrders)

    def asMatrix(self):
        if self.matrix is None:
            self.matrix = convertColumnTabl2Matrix(self.columnTab)
        return self.matrix

    def asColumnTab(self):
        if self.columnTab is None:
            self.columnTab = convertMatrix2ColumnTab(self.matrix)
        return self.columnTab

    def asColumnTabWithName(self, id_name):
        if self.columnTab is None:
            self.columnTab = convertMatrix2ColumnTab(self.matrix, id_name)
        return self.columnTab


class gs_parameter:
    def __init__(self, gid):
        self.gid = gid

    @classmethod
    def gen_rand_para(cls):
        return gs_parameter(random.getrandbits(128).to_bytes(16, byteorder='little', signed=False))


class gs_context:
    def __init__(self, myself, input_gid_list, begin_time, end_time, lookback):
        self.myself = myself
        self.input_list = input_gid_list
        self.begin_time = begin_time
        self.end_time = end_time
        self.lookback = lookback


    @classmethod
    def gen_rand_context(cls, input_len):
        gid = gs_parameter.gen_rand_para()
        input_list = [None] * input_len
        for i in range(input_len):
            input_list[i] = gs_parameter.gen_rand_para()
        return gs_context(gid, input_list, None, None, None)


def gen_gs_context_from_gid(my_gid, input_gid_list):
    my_para = gs_parameter(my_gid)
    input_para_list = []
    for gid in input_gid_list:
        input_para_list.append(gs_parameter(gid))
    return gs_context(my_para, input_para_list, None, None, None)


def transformO4Input(ndInput):
    return np.chararray(shape=(ndInput.size), itemsize=16, buffer=ndInput.data)


# def transformO4Input(ndInput):
#     twoCol = np.ndarray(shape=(ndInput.size,2),dtype=np.uint64,buffer=ndInput.data)
#     return twoCol[:,0],twoCol[:,1]
oDataType = [('v1', np.uint64), ('v2', np.uint64)]


def transformOFromPandas2Output(ndOutput):
    ret = ndOutput.astype(oDataType)
    ret.dtype = np.complex128
    return ret


def transform2DataFrame(tableInput):
    oSet = transformO4Input(tableInput[0])
    tSet = transformTime4Input(tableInput[1])
    return pd.DataFrame(tableInput[2], index=oSet, columns=tSet)


# 62091 is the difference between 1970-01-01 and 1800-01-01
max = 0xFFFFFFFF00000000 + 62091


def transformTime4Input(tArr):
    tArr.dtype = np.uint64
    tT = tArr - max
    tT.dtype = 'datetime64[D]'
    return tT


def transformTime4Output(tArr):
    tDay = tArr.astype('datetime64[D]')
    tDay.dtype = np.uint64
    tDay = tDay + max
    return tDay


def transformOTInColumnTable(colTable):
    varDict = dict()
    for col in colTable:
        if col[1].dtype == np.complex128:
            varDict.__setitem__(col[0], transformO4Input(col[1]))
        elif col[1].dtype == np.int64 or col[1].dtype == np.uint64:
            varDict.__setitem__(col[0], transformTime4Input(col[1]))
        else:
            varDict.__setitem__(col[0], col[1])
    return varDict


# Martin: the correct name called from C++ code is transformInput, transformData is the old name for testing which will be obsoleted
PARAMETER_TYPE_ERROR = int(0)
PARAMETER_TYPE_STRING = int(1)
PARAMETER_TYPE_PANEL = int(3)
PARAMETER_TYPE_OBJECT_SET = int(6)
PARAMETER_TYPE_FREQUENCY = int(7)
PARAMETER_TYPE_PARAMETERDICT = int(16)
PARAMETER_TYPE_NUMBER_INT = int(22)
PARAMETER_TYPE_NUMBER_NUMRIC = int(23)
PARAMETER_TYPE_OnTnVn_TABLE = int(24)
PARAMETER_TYPE_PYTHON_ONLY = int(27)
PARAMETER_TYPE_TIMESTAMP = int(2)
PARAMETER_TYPE_UUID = int(4)


def get_column_type(col_table: pd.DataFrame, name: str):
    dName = str(col_table[name].dtype)
    if 'datetime64' in dName:
        return PARAMETER_TYPE_TIMESTAMP
    elif 'float' in dName:
        return PARAMETER_TYPE_NUMBER_NUMRIC
    else:
        return PARAMETER_TYPE_UUID


def get_columns_type_dict(col_table: pd.DataFrame):
    ret = dict()
    for name in col_table.columns:
        ret[name] = get_column_type(col_table, name)
    return ret


def istimestamparray(array):
    return 'datetime64' in str(array.dtype)


def ismatrix(df: pd.DataFrame):
    return istimestamparray(df.index)


def checkDataType(obj):
    if isinstance(obj, str):
        return PARAMETER_TYPE_STRING
    elif isinstance(obj, int):
        return PARAMETER_TYPE_NUMBER_INT
    elif isinstance(obj, float):
        return PARAMETER_TYPE_NUMBER_NUMRIC
    elif isinstance(obj, dict):
        return PARAMETER_TYPE_PARAMETERDICT
    elif isinstance(obj, pd.DataFrame):
        return getDataFrameType(obj)
    elif isinstance(obj, GftTable):
        if obj.matrix is not None:
            return PARAMETER_TYPE_PANEL
        elif obj.columnTab is not None:
            return PARAMETER_TYPE_OnTnVn_TABLE
        else:
            return PARAMETER_TYPE_ERROR
    elif isinstance(obj, pd.tslib.Timestamp):
        return PARAMETER_TYPE_TIMESTAMP
    elif np.isscalar(obj):
        return PARAMETER_TYPE_NUMBER_NUMRIC
    else:
        return PARAMETER_TYPE_PYTHON_ONLY


def getDataFrameType(obj: pd.DataFrame):
    # TODO: this standard needs to be upgraded
    if istimestamparray(obj.index):
        return PARAMETER_TYPE_PANEL
    else:
        return PARAMETER_TYPE_OnTnVn_TABLE


def transformCppInput(data, parType):
    if parType == PARAMETER_TYPE_PANEL:
        data = transform2DataFrame(data)
        return GftTable.fromCppMatrix(data.copy())
    if parType == PARAMETER_TYPE_OnTnVn_TABLE:
        columnOrders = [data[i][0] for i in range(len(data))]
        dataDict = transformOTInColumnTable(data)
        data = pd.DataFrame(dataDict)
        return GftTable.fromColumnTable(data, columnOrders)
    if parType == PARAMETER_TYPE_FREQUENCY:
        tArr = transformTime4Input(data)
        return pd.DatetimeIndex(data=tArr, copy=False)
    if parType == PARAMETER_TYPE_OBJECT_SET:
        return transformO4Input(data)
    if parType == PARAMETER_TYPE_TIMESTAMP:
        # 62091 is the difference between 1970-01-01 and 1800-01-01
        data = data - 62091
        return pd.to_datetime(data, unit='D')
    return data


def test_coord(l_val, r_val):
    return l_val.equals(r_val)


def test_dataframe(l_val: pd.DataFrame, r_val: pd.DataFrame):
    return l_val.equals(r_val)


def test_gft_table(l_val, r_val):
    if (l_val.matrix is not None) and (r_val.matrix is not None):
        if not test_dataframe(l_val.matrix, r_val.matrix):
            return False

    if (l_val.columnTab is not None) and (r_val.columnTab is not None):
        if not test_dataframe(l_val.columnTab, r_val.columnTab):
            return False
    return True


def test_np_darray(l_val, r_val):
    if l_val.dtype == r_val.dtype and l_val.shape == r_val.shape:
        return (np.nan_to_num(l_val) == np.nan_to_num(r_val)).all();
    return False


def test_dict(l_val, r_val, recursive):
    if l_val.keys() == r_val.keys():
        for key in l_val.keys():
            if not test_equality(l_val[key], r_val[key], recursive):
                return False
        return True
    return False


def copyAndSave(key, value, old_dict, new_dict):
    old_dict[key] = value
    dataCp = copy.deepcopy(value)
    new_dict[key] = dataCp
    return dataCp


def testTwoDic(old_dict: dict, new_dict: dict, function_name: str):
    for name in old_dict.keys():
        if not test_equality(old_dict[name], new_dict[name], True):
            raise Exception(("input[" + name + "] is modified after call function:[$gid[" + function_name + "]]"))

def test_equality(l_val, r_val, recursive):
    if l_val.__class__ == r_val.__class__:
        if isinstance(l_val, pd.DataFrame):
            return test_dataframe(l_val, r_val)
        elif isinstance(l_val, GftTable):
            return test_gft_table(l_val, r_val)
        elif isinstance(l_val, np.ndarray):
            return test_np_darray(l_val, r_val)
        elif isinstance(l_val, np.chararray):
            return test_np_darray(l_val, r_val)
        elif isinstance(l_val, pd.core.indexes.datetimes.DatetimeIndex):
            return test_coord(l_val, r_val)
        elif isinstance(l_val, pd.core.indexes.base.Index):
            return test_coord(l_val, r_val)
        elif isinstance(l_val, dict):
            return test_dict(l_val, r_val, recursive)
        elif l_val.__dir__().__contains__("__dict__"):
            if recursive:
                return test_dict(l_val.__dict__,r_val.__dict__, False)
            elif l_val.__dir__().__contains__("__eq__"):
                return l_val.__eq__(r_val)
            elif l_val.__dir__().__contains__("equals"):
                return l_val.equals(r_val)
            else:
                #blindly return true.
                return True
        else:
            return l_val == r_val
    return False


def transformInput(data):
    if isinstance(data, int):
        # 62091 is the difference between 1970-01-01 and 1800-01-01
        data = data - 62091
        return pd.to_datetime(data, unit='D')
    if isinstance(data, tuple):
        data = transform2DataFrame(data)
        data = GftTable.fromCppMatrix(data)
        return data
    elif isinstance(data, list):
        columnOrders = [data[i][0] for i in range(len(data))]
        dataDict = transformOTInColumnTable(data)
        data = pd.DataFrame(dataDict)
        data = GftTable.fromColumnTable(data, columnOrders)
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype == np.complex128:
            oArr = transformO4Input(data)
            return pd.Index(data=oArr)
        elif data.dtype == np.int64 or data.dtype == np.uint64:
            tArr = transformTime4Input(data)
            return pd.DatetimeIndex(data=tArr, copy=False)
    elif isinstance(data, dict):
        return transformDict(data)
    return data


def transformDict(dic):
    for key, value in dic.items():
        if isinstance(value, dict):
            dic[key] = transformDict(value)
        else:
            dic[key] = transformInput(value)
    return dic


def gidStrHex2IntPairs(gid_str):
    if gid_str[0].__len__() == 32:
        v1 = int.from_bytes(binascii.a2b_hex(gid_str[0][0:16]), byteorder='big', signed=True)
        v2 = int.from_bytes(binascii.a2b_hex(gid_str[0][16:32]), byteorder='big', signed=True)
        return v1, v2
    elif gid_str == "GFT":
        return int(0), int(0)
    else:
        raise Exception("String [" + gid_str + "] len is not 32")


def printStr(gid_str):
    print(gid_str)
    print(type(gid_str))
    print(gid_str.__len__())


def gidStrArray2CharArray(gid_arr: np.array):
    gid_arr.shape = (gid_arr.shape[0], 1)
    d2 = np.apply_along_axis(gidStrHex2IntPairs, 1, gid_arr)
    # transform int pair two bytes
    return np.chararray(shape=(d2.shape[0]), itemsize=16, buffer=d2.data)

def intTuple2Str(tup2Int):
    str1 = int.to_bytes(int(tup2Int[0][0]),8,"big").hex()
    str2 = int.to_bytes(int(tup2Int[0][1]), 8, "big").hex()
    return str1+str2

def strSet2Np(gid_arr: np.array):
    return gidStrArray2CharArray(gid_arr)


def testGidStr2Np():
    strArr = np.array(
        ["A6D3CA2532F94917BC5C1DB2F43F5A41", "BDF62F36F82A41A2ADF2C228DB903365", "C6149560360A0DE607CD12AFD003B082"])
    strSet2Np(strArr)


# use this to transform charArray to string
def gidInt2Str(gid):
    if gid.__len__ == 0:
        return "GFT"
    if (gid.__len__() == 16):
        firstO = bytearray(gid[:8])
        firstO.reverse()
        secondO = bytearray(gid[8:])
        secondO.reverse()
        full = firstO.hex() + secondO.hex()
        return full.upper()
    return gid.hex()


def fillNan(row):
    if (np.isnan(row['val'])):
        return gidInt2Str(row['key'])
    else:
        return row['val']


def transformName4Array(arr, nameMap):
    pdArr = pd.DataFrame({'key': arr})
    pdMapped = pdArr.merge(right=nameMap, how='left', left_on='key', right_index=True)
    nullRows = pdMapped.val.isnull()
    if (pdMapped[nullRows].size > 0):
        gidName = pdMapped[nullRows].key.apply(gidInt2Str)
        pdMapped.val[gidName.index] = gidName
    return pdMapped.val


def mergeName(ootv, namemap):
    ts = time.time()
    ret = ootv.merge(right=namemap, how='left', left_on=['O_v1', 'O_v2'], right_index=True)
    elapsed = time.time() - ts
    print(elapsed)
    return ret


def transformName4Dataframe(df, nameMap):
    if isinstance(df.columns[0], str):
        # this is column table
        if df.size == 0:
            return None
        rtnDict = dict()
        for col in df.columns:
            if df[col].dtype == np.object:
                rtnDict.__setitem__(col, transformName4Array(df[col].values, nameMap))
            else:
                rtnDict.__setitem__(col, df[col].values)
        return pd.DataFrame(rtnDict)
    else:
        return pd.DataFrame(data=df.values, index=df.index, columns=transformName4Array(df.columns.values, nameMap))


def transformData4Name(data, namemap):
    if (isinstance(data, GftTable)):
        if (data.matrix is None):
            colTable = transformName4Dataframe(data.columnTab, namemap)
            return GftTable.fromColumnTable(colTable, data.columnOrders)
        else:
            mat = transformName4Dataframe(data.matrix, namemap)
            return GftTable.fromCppMatrix(mat)
    elif isinstance(data, pd.DataFrame):
        return transformName4Dataframe(data, namemap)
    elif isinstance(data, dict):
        return transformDictWithMap(data, namemap)
    return data


def transformDictWithMap(dic, namemap):
    resultDict = dict()
    for key, value in dic.items():
        if isinstance(value, dict):
            resultDict[key] = transformDictWithMap(value, namemap)
        else:
            try:
                resultDict[key] = transformData4Name(value, namemap)
            except Exception:
                print("transform " + key + ' failed')
                return
    return resultDict


def transformDict4Name(dic):
    with open("lib/gftTools/symbols.pkl", "rb") as fpz:
        value = fpz.read()
        namemap = pickle.loads(value)
    return transformData4Name(dic, namemap)


def transformOutputWithTypeGiven(obj, par_type):
    if par_type == PARAMETER_TYPE_NUMBER_NUMRIC:
        return float(obj)
    return transformOutput(obj)


def transformMatrixWithGid(panel, gid):
    data = transform2DataFrame(panel)
    data = GftTable.fromCppMatrixWithGid(data, gid)
    return data


def transformMatrix(panel):
    data = transform2DataFrame(panel)
    data = GftTable.fromCppMatrix(data)
    return data


def transformColumnTableWithGid(columnTable, gid):
    columnOrders = [columnTable[i][0] for i in range(len(columnTable))]
    dataDict = transformOTInColumnTable(columnTable)
    data = pd.DataFrame(dataDict)
    data = GftTable.fromColumnTableWithGid(data, gid, columnOrders)
    return data


def transformColumnTable(columnTable):
    columnOrders = [columnTable[i][0] for i in range(len(columnTable))]
    dataDict = transformOTInColumnTable(columnTable)
    data = pd.DataFrame(dataDict)
    data = GftTable.fromColumnTable(data, columnOrders)
    return data


def transformOutput(obj):
    if isinstance(obj, pd.DataFrame):
        return _frameToList(obj)
    elif isinstance(obj, GftTable):
        if obj.matrix is not None:
            return _wideFrameToTuple(obj.matrix)
        elif obj.columnTab is not None:
            return _longFrameToList(obj.columnTab)
        else:
            return None
    elif isinstance(obj, pd.tslib.Timestamp):
        # 62091 is the difference between 1970-01-01 and 1800-01-01
        return int(obj.tz_localize(tz="UTC").timestamp() / 86400) + 62091
    else:
        return obj


def is_serializable_2_cpp(obj):
    invert_op = getattr(obj, "__serialize__", None)
    return callable(invert_op)


def slice_data_inplace(data, begin_time: pd.Timestamp, end_time: pd.Timestamp):
    if isinstance(data, GftTable):
        slice_table_inplace(data, begin_time, end_time)
        return data
    elif isinstance(data, pd.DataFrame):
        if ismatrix(data):
            return slice_matrix(data, begin_time, end_time)
        else:
            return slice_column_table(data, begin_time, end_time)
    elif isinstance(data, pd.core.indexes.datetimes.DatetimeIndex):  #
        return data[(data > begin_time) & (data <= end_time)]
    elif is_serializable_2_cpp(data):
        all_data = data.__serialize__()

    return data


def slice_column_table(column_table: pd.DataFrame, begin_time: pd.Timestamp, end_time: pd.Timestamp):
    for name in column_table.columns:
        if 'datetime64' in str(column_table[name].dtype):
            pt_name = name
    if pt_name is not None:
        return column_table.loc[(column_table[pt_name] > begin_time) & (column_table[pt_name] <= end_time)]


def slice_matrix(matrix: pd.DataFrame, begin_time: pd.Timestamp, end_time: pd.Timestamp):
    return matrix.loc[(matrix.index > begin_time) & (matrix.index <= end_time)]


def slice_table_inplace(gft_table: GftTable, begin_time: pd.Timestamp, end_time: pd.Timestamp):
    if gft_table.columnTab is not None:
        pt_name = None
        column_table = gft_table.columnTab
        if hasattr(gft_table, "primary_t"):
            pt_name = gft_table.primary_t
        else:
            if gft_table.columnOrders is not None:
                col_names = gft_table.columnOrders
            else:
                col_names = column_table.columns
            for name in col_names:
                if 'datetime64' in str(column_table[name].dtype):
                    pt_name = name
        if pt_name is not None:
            # slice with it.
            sliced_rows = column_table.loc[(column_table[pt_name] > begin_time) & (column_table[pt_name] <= end_time)]
            gft_table.columnTab = sliced_rows
    if gft_table.matrix is not None:
        sliced_matrix = gft_table.matrix.loc[
            (gft_table.matrix.index > begin_time) & (gft_table.matrix.index <= end_time)]
        gft_table.matrix = sliced_matrix

def wrap_gfttable_dataframe_clean_gid(obj):
    global inst_gid
    ret = wrap_gfttable_dataframe_with_gid(obj, inst_gid)
    inst_gid = None
    return ret


def parseMetaInfo(data):
    with open("mtproto.bin", "wb", -1) as fpz:
        fpz.write(data)

def testLocal(num):
    abc = num
    return abc


def wrap_gfttable_dataframe_with_gid(obj, gid):
    if isinstance(obj, pd.DataFrame):
        if PARAMETER_TYPE_PANEL == getDataFrameType(obj):
            return GftTable.fromPythonMatrixWithGid(obj, gid)
        else:
            return GftTable.fromColumnTableWithGid(obj, gid, None)
    elif isinstance(obj, GftTable):
        if obj.gid is not None:
            obj.gid = gid
    return obj


def _frameToList(obj: pd.DataFrame):
    if PARAMETER_TYPE_PANEL == getDataFrameType(obj):
        return _wideFrameToTuple(obj)
    else:
        return _longFrameToList(obj)


def _longFrameToList(obj):
    alist = []
    for columnname in obj.columns:
        column_data = obj.loc[:, columnname].values
        if 0 == len(column_data):
            return None
        elif isinstance(column_data[0], np.datetime64):
            column_data = transformTime4Output(column_data)
        elif not isinstance(column_data[0], float):  # O array
            column_data = transformOFromPandas2Output(column_data)

        acolumn = (columnname, column_data)
        alist += [acolumn]
    return alist


#use this to transfor oSet -> readable hex string array
def oSet2HexStr(oSet):
    ret = oSet.astype(oDataType)
    return np.apply_along_axis(intTuple2Str, 1, ret)


def _wideFrameToTuple(obj):
    objT = obj.transpose()
    ls_index = transformOFromPandas2Output(objT.index.values)
    ls_column = transformTime4Output(objT.columns.values)
    arr_values = objT.values
    if (0 == len(ls_index) or 0 == len(ls_column) or 0 == len(arr_values)):
        return None

    return (ls_index, ls_column, arr_values)


class LongTable2Readable:  # long table means OnTnVn, or OOTV
    # Martin: the code below will be executed when load this py file. Need to comment it out
    # uuid2Readable = Uuid2Readable()

    def convert(self, ootv):
        result = []
        for column in ootv:
            arr = column[1]
            name = column[0]
            if len(arr) > 0:
                if (type(arr[0]) == np.complex128):  # O array
                    result.append((name, self.uuid2Readable.oArr2Readable(arr)))
                elif (type(arr[0]) == np.int64):  # T array
                    # please change the code to convert t array
                    result.append(column)
                else:
                    result.append(column)
            else:
                result.append(column)
        return result


def intTuple2Str(tup2Int):
    """
    convert tuple to 32bit hex
    """
    str1 = int.to_bytes(int(tup2Int[0][0]), 8, "big").hex()
    str2 = int.to_bytes(int(tup2Int[0][1]), 8, "big").hex()
    return str1+str2


def oSet2Hex(oSet):
    """
    convert array to 32 bit hex array
    """
    ret = oSet.astype(oDataType)
    return np.apply_along_axis(intTuple2Str, 1, ret.reshape(-1, 1))

            factors_output = pd.DataFrame(
                index=self.model[oset_idx[0]+post_fix].index, columns=oset_idx)
            for value in oset_idx:
                factors_output[value] = self.model[value+post_fix]
            factors_output.columns = gftIO.strSet2Np(
                factors_output.columns.values)
            return factors_output



def merge_matrix(old_data, new_data, old_desc, new_desc):
    if new_desc.required_begin_t <= old_desc.required_begin_t:
        return new_data
    if new_desc.required_begin_t <= old_desc.required_end_t:
        # so slice old data
        old_data = old_data[old_data.index < new_desc.required_begin_t]
    #concat old data with new data
    # concat along index, and use outer join for columns.
    return pd.concat(objs=[old_data, new_data],axis=0,join='outer')




def merge_col_tab(old_data, new_data, old_desc, new_desc):
    print ("merge coltabs, size:{0} and {1}".format(str(old_data.shape), str(new_data.shape)))
    if new_desc.required_begin_t <= old_desc.required_begin_t:
        return new_data

    sorted_old_cols = old_data.columns.sort_values()
    sorted_new_cols = new_data.columns.sort_values()
    if not sorted_old_cols.equals(sorted_new_cols):
        raise Exception("New data's columns{0} is not the same as old data's columns{1}".format(str(sorted_new_cols), str(sorted_old_cols)))

    #bcs pt_name may not right from desc
    pt_name = gftIO.get_pt_name(old_data, old_desc.get_pt_name())
    if pt_name is None:
        return new_data

    if new_desc.required_begin_t <= old_desc.required_end_t:
        # so slice old data
        old_data = old_data[old_data[pt_name] < new_desc.required_begin_t]

    # concat old data with new data
    ret = pd.concat(objs=[old_data, new_data],axis=0,join='outer')
    # print("Concated table size:{0}".format(str(ret.shape)))
    return ret


def merge_xarray(old_data, new_data, old_desc, new_desc):
    raise Exception("Not supported yet.")


def merge_data(old_data, new_data, old_desc, new_desc):
    if type(old_data) != type(new_data):
        raise Exception("Can not merge data of differnt types")
    if isinstance(new_data, dict):
        ret = dict()
        for key, val in dict.items():
            old_value = old_data.get(key)
            if old_value is not None:
                ret[key] = merge_data(old_value, val, old_desc, new_desc)
            else:
                ret[key] = val
    if isinstance(new_data, gftIO.GftTable):
        if (new_data.matrix is not None) and (old_data.matrix is not None):
            merged_mtx = merge_matrix(old_data.matrix, new_data.matrix, old_desc, new_desc)
            new_data.matrix = merged_mtx
            new_data.columnTab = None
            return new_data
        if (new_data.columnTab is not None) and (old_data.columnTab is not None):
            merged_col_tab = merge_col_tab(old_data.columnTab, new_data.columnTab, old_desc, new_desc)
            new_data.columnTab = merged_col_tab
            new_data.matrix = None
            return new_data
        raise Exception("Merge GftTable of different type")
    if isinstance(new_data, pd.DataFrame):
        is_new_data_matrix = gftIO.ismatrix(new_data)
        if is_new_data_matrix != gftIO.ismatrix(old_data):
            raise Exception("Merge dataframe of different shape")
        if is_new_data_matrix:
            return merge_matrix(old_data, new_data, old_desc, new_desc)
        else:
            return merge_col_tab(old_data, new_data, old_desc, new_desc)
    if isinstance(new_data, xarray):
        return merge_xarray(old_data, new_data, old_desc, new_desc)
    return new_data



# all caches would be in this struct. so i can get both data and meta.
class CacheData:
    def __init__(self, type, meta, data):
        self.data = data
        self.meta = meta
        self.type = type

    def copy_and_slice_data_with_begin_date(self, begin_date):
        return self

    def copy_and_slice_data_with_end_date(self, end_data):
        return self

    def copy_and_slice_data_with_begin_end_date(self, begin_date, end_date):
        return self


def dumpAll4Cache(type, meta, data, timestamp, filename):
    cache = CacheData(type, meta, data)
    return gftIO.zdump4CacheSever(cache, timestamp, filename)


import pickle
import copy

class Keys:
    def __init__(self):
        pass
    def has_next(self):
        return False
    def next(self):
        return None
    def copy(self):
        return copy.copy(self)

    def reset(self):
        pass
    def size(self):
        return -1

    def key_value(self):
        return None

class Values:
    def __init__(self, values):
        self.values = values
        self.list_size = 1

    def get(self, next_key):
        return None

    def copy(self):
        return copy.copy(self)

    def get_list_size(self):
        return 1


class DataIterator:
    def __init__(self, as_list, is_const, keys: Keys, values: Values):
        self.as_list = as_list
        self.keys = keys
        self.values = values
        self.cur_key = None
        self.list_size = 1
        self.is_const = is_const

    def get_list_size(self):
        if self.as_list:
            return self.values.get_list_size()
        return 1

    def set_as_list(self):
        self.as_list = True
        while self.has_next():
            data = self.next()
            if isinstance(data, list) or isinstance(data, tuple):
                self.list_size = len(data)
                return

    def copy(self):
        return DataIterator(self.as_list, self.is_const, self.keys.copy(), self.values.copy())

    def has_next(self):
        return self.keys.has_next()

    def next(self):
        return self.values.get(self.keys.next())

    def key(self):
        return self.keys.key_value()

    def get_keys(self):
        return self.keys.keys()

    def get_columns(self):
        return None

    def get_values(self):
        return self.values

    def reset(self):
        # print("{0} reseted)".format(str(type(self))))
        self.keys.reset()

    def size(self):
        return self.keys.size()

class IndexKeys(Keys):
    def __init__(self, index):
        self.pos = -1
        self.index = index
        self.last_index = self.index.size - 1

    def has_next(self):
        return self.last_index > self.pos

    def next(self):
        self.pos += 1
        return self.pos

    def key_value(self):
        return self.index[self.pos]

    def reset(self):
        self.pos = -1

    def size(self):
        return self.last_index + 1

class IndexValues(Values):
    def __init__(self, source_data):
        self.source_data = source_data
        self.list_size = 1
        if isinstance(source_data, gftIO.GftTable):
            self.matrix = source_data.asMatrix()
        elif isinstance(source_data, pd.DataFrame):
            self.matrix = source_data
        else:
            raise Exception("Not supported type:" + str(type(source_data)))

    def get(self, next_key):
        return self.matrix.iloc[[next_key]].transpose()



def create_index_iterator(source_data):
    values = IndexValues(source_data)
    keys = IndexKeys(values.matrix.index)
    return DataIterator(False, False, keys, values)


class ColKeys(Keys):
    def __init__(self, columns):
        self.pos = -1
        self.last_index = columns.size

    def has_next(self):
        return self.last_index > self.pos

    def next(self):
        self.pos += 1
        return self.pos

    def key_value(self):
        return self.columns[self.pos]

    def reset(self):
        self.pos = -1

    def size(self):
        return self.last_index + 1

class ColValues(Values):
    def __init__(self, source_data):
        self.source_data = source_data
        self.list_size = 1
        if isinstance(source_data, gftIO.GftTable):
            self.matrix = source_data.asMatrix()
        elif isinstance(source_data, pd.DataFramea):
            self.matrix = source_data

    def get(self, next_key):
        return self.matrix[[self.matrix.columns[next_key]]]


def create_column_iterator(source_data):
    values = ColValues(source_data)
    keys = ColKeys(values.matrix.columns)
    return DataIterator(False, False, keys, values)


def int_to_name(name):
    if name == '0' or name == '0.0':
        return 'date'
    if name == '1' or name == '1.0':
        return 'symbol'
    if name == '2' or name == '2.0':
        return 'factor'
    return name

class ItContinue:
    def __init__(self, skip_reason):
        self.reason = skip_reason

class XarrayKeys(Keys):
    def __init__(self, source_data, axis_name='date'):
        if axis_name is None:
            # find axis with timestamp
            self.axis_name = 'date'
        else:
            # confirm axis existed.
            self.axis_name = axis_name
        self.last_index = source_data[self.axis_name].size - 1
        self.pos = -1
        self.axis_idx = source_data[self.axis_name]

    def has_next(self):
        return self.last_index > self.pos

    def next(self):
        self.pos += 1
        return self.axis_idx[self.pos]

    def key_value(self):
        return self.axis_idx[self.pos].values

    def reset(self):
        self.pos = -1

    def size(self):
        return self.last_index + 1


class XarrayValues(Values):
    def __init__(self, source_data, axis_name='date', index_name='symbol', column_name='factor'):
        self.as_list = False
        if axis_name is None:
            # find axis with timestamp
            self.axis_name = 'date'
        else:
            # confirm axis existed.
            self.axis_name = int_to_name(axis_name)

        if index_name is None:
            self.index_name = 'symbol'
        else:
            self.index_name = int_to_name(index_name)
        if column_name is None:
            self.column_name = 'factor'
        else:
            self.column_name = int_to_name(column_name)
        self.source_data = source_data.transpose(self.axis_name, self.index_name, self.column_name)
        self.list_size = 1

    def get(self, next_key):
        return self.source_data.loc[next_key,:,:].to_pandas()


def create_xarray_iterator(source_data, axis_name = 'date', index_name = 'symbol', column_name = 'factor'):
    values = XarrayValues(source_data, axis_name, index_name, column_name)
    keys = XarrayKeys(source_data, values.axis_name)
    return DataIterator(False, False, keys, values)





class DictKeys(Keys):
    def __init__(self, dic_data):
        self.it = dic_data.items().__iter__()
        self.count = dic_data.__len__()
        self._key = None
        self.source_data = dic_data


    def has_next(self):
        return self.count > 0

    def next(self):
        self.count -= 1
        key_value = self.it.__next__()
        self._key = key_value[0]
        return key_value[1]

    def copy(self):
        return DictKeys(self.source_data)

    def reset(self):
        self.count = self.source_data.__len__()
        self.it = self.source_data.items().__iter__()

    def size(self):
        return self.source_data.__len__()

    def key_value(self):
        return self._key

    def keys(self):
        return self.source_data.keys()


class DictValues(Values):
    def __init__(self, dic_data: dict):
        self.source_data = dic_data
        self.list_size = 1
        for key, value in dic_data.items():
            if isinstance(value, list) or isinstance(value, tuple):
                self.list_size = len(value)
                return


    def get(self, next_key):
        return next_key


def create_dict_iterator(dic_data):
    values = DictValues(dic_data)
    keys = DictKeys(dic_data)
    return DataIterator(False, False, keys, values)

def prepare_iterator(*iterators):
    available_it = None
    const_iterators = list()
    ret_list = list()

    for it in iterators:
        if isinstance(it, DataIterator):
            it_copy = it.copy()
        else:
            it_copy = create_const_copy_iterator(it, None)
        if it_copy.is_const:
            const_iterators.append(it_copy)
        else:
            available_it = it_copy
        ret_list.append(it_copy)

    if available_it is None:
        return None
    for const_it in const_iterators:
        const_it.keys.reset_other_keys(available_it.keys)
    return ret_list

class ListValues(Values):
    def __init__(self, *iterators):
        self.iter_list = list()
        for it in iterators:
            if isinstance(it.values, ListValues):
                for it in it.values.iter_list:
                    self.iter_list.append(it.copy())
            else:
                self.iter_list.append(it.copy())
        self.data_list = list()
        self.list_range = range(len(iterators))
        total_size = 0
        for it in self.iter_list:
            total_size += it.get_list_size()
        self.list_size = total_size

    def get_underlaying_it(self, index):
        for it in self.iter_list:
            it_list_size = it.get_list_size()
            if it_list_size > index:
                if isinstance(it.values, ListValues):
                    return it.values.get_underlaying_it(index)
                return index, it
            else:
                index -= it_list_size
        return index, None

    def copy(self):
        return ListValues(*self.iter_list)

    def reset(self):
        # print("ListValue reseted")
        for it in self.iter_list:
            it.reset()


    def get(self, next_key):
        self.data_list.clear()
        skip_row = None
        for it in self.iter_list:
            val = it.next()
            if isinstance(val, ItContinue):
                skip_row = val
                continue
            if it.as_list:
                self.data_list.extend(val)
            else:
                self.data_list.append(val)
        if skip_row is not None:
            return skip_row
        return self.data_list


class ReferenceKeys(Keys):
    def __init__(self, first_availkeys):
        if isinstance(first_availkeys, ReferenceKeys):
            first_availkeys = first_availkeys.keys
        self.keys = first_availkeys.copy()

    def has_next(self):
        return self.keys.has_next()


    def next(self):
        return self.keys.next()


    def copy(self):
        return ReferenceKeys(self.keys.copy())

    def reset(self):
        self.keys.reset()


    def size(self):
        return self.keys.size()

    def key_value(self):
        return self.keys.key_value()




class ExtractValues(Values):
    def __init__(self, iter: DataIterator, index):
        index = int(index)
        if iter.list_size < index:
            raise Exception("list({0}) not long enough, required{1}".format(str(iter.list_size), str(index)))
        if isinstance(iter, ListIterator):
            index, iter = iter.values.get_underlaying_it(index)
        self.iter = iter.copy()
        self.values = iter.values
        self.index = index
        self.list_size = self.iter.list_size

    def get(self, next_key):
        if self.iter.as_list:
            val = self.values.get(next_key)
            if isinstance(val, ItContinue):
                return val
            return val[self.index]
        else:
            return self.values.get(next_key)

class ExtractListValues(Values):
    def __init__(self, iter: DataIterator, indexes: str):
        splited = indexes.split(',')
        int_indexes = list()
        for idx in splited:
            int_indexes.append(int(idx))
        if isinstance(iter.values, ExtractListValues):
            sub_idx = list()
            for i in int_indexes:
                sub_idx.append(iter.values.int_indexes[i])
            int_indexes = sub_idx
            iter = iter.values.source_data
        self.iter = iter
        self.values = iter.values
        self.int_indexes = int_indexes
        self.data_list = list()

    def get(self, next_key):
        self.data_list.clear()
        vals = self.values.get(next_key)
        if isinstance(vals, ItContinue):
            return vals
        for i in self.int_indexes:
            self.data_list.append(vals[i])
        return self.data_list


def create_extract_values(it, index):
    values = ExtractValues(it, index)
    if values.index == 0 and values.iter.as_list == False:
        return values.iter.copy()
    keys = ReferenceKeys(values.iter.keys)
    return DataIterator(False, False, keys, values)

def create_extract_multi_values(it, indexes_str):
    values = ExtractListValues(it, indexes_str)
    keys = ReferenceKeys(values.iter.keys)
    return DataIterator(True, False, keys, values)


class ListIterator(DataIterator):
    def __init__(self, keys, values: ListValues):
        DataIterator.__init__(self, True, False, keys, values)

    def reset(self):
        # print("ListIterator reseted.")
        self.keys.reset()
        self.values.reset()

def create_list_iterator(*iterators):
    keys = ReferenceKeys(iterators[0].keys)
    values = ListValues(*iterators)
    return ListIterator(keys, values)

class ConstIteratorKeys(Keys):
    def __init__(self, other_keys):
        if other_keys is not None:
            other_keys = other_keys.copy()
        self.other_keys = other_keys

    def reset_other_keys(self, other_key):
        self.other_keys = other_key.copy()


    def has_next(self):
        return self.other_keys.has_next()

    def next(self):
        return self.other_keys.next()


    def copy(self):
        return ConstIteratorKeys(self.other_keys)

    def reset(self):
        self.other_keys.reset()

    def size(self):
        return self.other_keys.size()

    def key_value(self):
        return self.other_keys.key_value()

class ConstIteratorValues(Values):
    def __init__(self, const_val):
        self.const_val = const_val

    def get(self, next_key):
        return self.const_val

def create_const_iterator(const_val, othter_keys):
    keys = ConstIteratorKeys(othter_keys)
    values = ConstIteratorValues(const_val)
    return DataIterator(False, True, keys, values)

class ConstCopyIteratorValues(ConstIteratorValues):
    def __init__(self, const_val):
        self.const_val = const_val
    def get(self, next_key):
        return copy.deepcopy(self.const_val)

def create_const_copy_iterator(const_val, other_iterator_keys):
    keys = ConstIteratorKeys(other_iterator_keys)
    values = ConstCopyIteratorValues(const_val)
    return DataIterator(False, True, keys, values)


def getCacheData(value):
    cache = pickle.loads(value)
    if isinstance(cache, CacheData):
        return cache.type, cache.meta, cache.data
    raise Exception("Cache type is not gsMeta.CacheData")


def slice_redundant_result_and_wrap_gft_table_is_necessary(obj, meta):
    if (meta.required_begin_t > meta.input_begin_t) or (meta.required_end_t < meta.input_end_t):
        # may have redundant date in result.
        # print("Slice redundant data in result.")
        obj = gftIO.slice_data_inplace_and_ret(obj, gftIO.get_pt_name(obj, meta.get_pt_name()), meta.required_begin_t, meta.required_end_t)

    return gftIO.wrap_gfttable_dataframe_clean_gid(obj)

import xarray as xr

def create_xarray_from_dic(data_dict, date_name = 'date', symbol_name='symbol', factor_name='factor'):
    variables = dict()
    for key, value in data_dict.items():
        if isinstance(value, gftIO.GftTable):
            variables[key] = xr.DataArray(value.asMatrix(), dims=[date_name, symbol_name])
        elif isinstance(value, pd.DataFrame):
            variables[key] = xr.DataArray(value, dims=[date_name, symbol_name])
        else:
            raise Exception("Value[{0}] of type:{1} is not acceptable".format(str(key), str(type(value))))

    combined = xr.Dataset(variables).to_array(dim=factor_name)
    return combined
