import pickle, zlib
import numpy as np
import pandas as pd
import binascii
import time
import random
import copy


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


def oSet2Hex(oSet):
    ret = oSet.astype(oDataType)
    return np.apply_along_axis(intTuple2Str, 1, ret)


def intTuple2Str(tup2Int):
    str1 = int.to_bytes(int(tup2Int[0][0]), 8, "big").hex()
    str2 = int.to_bytes(int(tup2Int[0][1]), 8, "big").hex()
    return str1+str2


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
    matrix['idname'] = matrix.index
    return pd.melt(matrix, id_vars=['idname'])


# pivot long format -> wide format
def convertColumnTabl2Matrix(columnTab):
    tName = None
    vName = None
    oName = None
    for colName in columnTab.columns:
        if (columnTab[colName].dtype == np.float64):
            if vName is None:
                vName = colName
        if (columnTab[colName].dtype == np.object):
            if oName is None:
                oName = colName
        else:
            if tName is None:
                tName = colName

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
    if parType == PARAMETER_TYPE_OBJECT_SET or parType == PARAMETER_TYPE_FREQUENCY:
        tArr = transformTime4Input(data)
        return pd.DatetimeIndex(data=tArr, copy=False)
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
                return test_dict(l_val.__dict__, False)
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


def strSet2Np(gid_arr: np.array):
    return gidStrArray2CharArray(gid_arr)


def testGidStr2Np():
    strArr = np.array(
        ["A6D3CA2532F94917BC5C1DB2F43F5A41", "BDF62F36F82A41A2ADF2C228DB903365", "C6149560360A0DE607CD12AFD003B082"])
    strSet2Np(strArr)


# use this to string charArray to string
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


