'''
convert most python object to basic python object, so it can be easily processed
by c++ program
'''
import pandas as pd

def transformOuput(obj):
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, float):
        return obj
    elif isinstance(obj, dict):
        obj_Copy = obj.copy()
        for k,v in obj_Copy.items():
            obj_Copy[k] = transformOuput(v)
        return obj_Copy
    elif isinstance(obj, pd.DataFrame):
        return _frameToList(obj)
    else:
        return obj

def _frameToList(obj):
    if isinstance(obj.index, pd.RangeIndex) :
        return _longFrameToList(obj)
    else:
        return _wideFrameToList(obj)
    
def _longFrameToList(obj):
    alist = []
    for columnname in obj.columns:
        column_data = obj.loc[:,columnname].tolist()
        acolumn = [columnname,column_data]
        alist += [acolumn]
    return alist
 
def _wideFrameToList(obj):
    ls_index = obj.index.values
    ls_column = obj.columns.values
    arr_values = obj.values
    return [ls_index, ls_column, arr_values]


#def test_scalar():
#    print(transformOuput(1))
#    print(transformOuput(1.0))
#    print(transformOuput("abc"))
#    
#def test_convert_dict():
#    adict = {"A":[1,2,3,4],"B":[2,3,4,5]}
#    result = transformOuput(adict)
#    print(result)
#    
#def test_convert_frame():
#    import pandas as pd
#    aframe = pd.DataFrame({"A":[1,2,3,4],"B":[2,3,4,5]})
#    result = transformOuput(aframe)
#    print(result)
#
#def test_convert_dict1():
#    aframe = pd.DataFrame({"A":[1,2,3,4],"B":[2,3,4,5]})
#    adict = {"A":[1,2,3,4],"B":aframe}
#    result = transformOuput(adict)
#    print(result)
#    
#if __name__ == "__main__":
#    test_scalar()
#    test_convert_dict()
#    test_convert_frame()
#    test_convert_dict1()

#from . import gftIO
#
#
##dataPack = gftIO.zload("testSave_0.pkl")
#dataPack = gftIO.zload("rawData.pkl")
#result = transformOuput(dataPack)
#print(result)