#!/usr/bin/python3

# convert symbol in sqlite db to python persistence format

import os
import sys
import sqlite3

# gftIO locates in the upper level folder
scriptFolder = os.path.dirname(__file__)
sys.path.insert(0, os.path.normpath(os.path.join(scriptFolder, "..")))
import gftIO 

testOArrFile = os.path.join(scriptFolder, 'testSave_0.pkl')

loadedVal = gftIO.zload(testOArrFile)
print(loadedVal.keys())
x0 = loadedVal['x0']
oArr = x0[0]
tArr = x0[1]
vArr = x0[2][0]

print(type(oArr))
print(type(oArr[0]))

print(type(tArr))
print(type(tArr[0]))

print(type(vArr))
print(type(vArr[0]))

ootvIn = [('o1', oArr), ('t1', tArr), ('v1', vArr)]
longTable2Readable = gftIO.LongTable2Readable()
ootvOut = longTable2Readable.convert(ootvIn)
print(ootvOut)
