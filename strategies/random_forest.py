# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:18:01 2017

@author: gft
"""

##gid:380DC47C21BA25F6574F694DC440833E
from lib.gftTools import gftIO
from lib.gftTools import gsUtils
import pandas as pd
import numpy as np
import random
import datetime

x0 = gftIO.zload(
    "C:/wangqiujun/_43_Zimmerman/random_forest/randomforest/dataprepare.pkl")

n_estimators = 200  #paper:200 :n_estimators
criterion = 'mse'  ##paper:'mse'
max_depth = -999  ##paper:对深度没有要求:-999表示对深度没有要求
min_samples_split = 100  ##每个节点至少有100个观测才能继续往下分:min_samples_split
sample_size = 1.0
max_features = 8  ## max_features
min_decrease = 0.00001

######test#######
n_estimators = 5  #paper:200 :n_estimators
criterion = 'mse'  ##paper:'mse'
max_depth = 5  ##paper:对深度没有要求:-999表示对深度没有要求
min_samples_split = 1000  ##每个节点至少有100个观测才能继续往下分:min_samples_split
sample_size = 1.0
max_features = 8  ## max_features
min_decrease = 0.00001

test = RFCalTargetWgt(x0, n_estimators, criterion, max_depth, min_samples_split,
                      sample_size, max_features, min_decrease)

#gftIO.zdump(test, "C:/wangqiujun/_43_Zimmerman/random_forest/randomforest/randomforest.pkl")


def RFCalTargetWgt(x0, n_estimators, criterion, max_depth, min_samples_split,
                   sample_size, max_features, min_decrease):
    ##step1:strategy creation
    starttime = datetime.datetime.now()
    df_train_data = x0['train']

    ls_test_key = list(np.unique(df_train_data.fitno))
    dict_train_tree = {
        i: random_forest(x0, i, max_depth, min_samples_split, sample_size,
                         n_estimators, max_features, min_decrease)
        for i in ls_test_key
    }
    ls_train_tree = [
        dict_train_tree[i]['targetwgt'] for i in dict_train_tree.keys()
    ]
    df_wgtfinal = pd.concat(ls_train_tree, axis=0)

    ls_train_residual = [
        dict_train_tree[i]['train_Residual'] for i in dict_train_tree.keys()
    ]
    df_train_residual = pd.concat(ls_train_residual, axis=0)

    ls_test_residual = [
        dict_train_tree[i]['test_Residual'] for i in dict_train_tree.keys()
    ]
    df_test_residual = pd.concat(ls_test_residual, axis=0)

    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
    return {
        'targetwgt': df_wgtfinal,
        'train_Residual': df_train_residual,
        'test_Residual': df_test_residual
    }


def calGlobalResi(data, i):
    df_train = data[data.fitno == i]
    sum((df_train.y - np.mean(df_train.y))**2) / data.shape[0]


# Random Forest Algorithm
def random_forest(x0, i, max_depth, min_samples_split, sample_size,
                  n_estimators, max_features, min_decrease):
    df_train_all = x0['train']
    df_train_daily = df_train_all[df_train_all.fitno == i].reset_index()
    df_train = df_train_daily.drop(['snapshot', 'symbol', 'index'], axis=1)
    #train=[df_train.iloc[i,:] for i in range(len(df_train))]
    trees = list()
    ls_x = list(df_train.columns)[:-2]
    max_depth = ((2**31) - 1 if max_depth == -999 else max_depth)
    ##train data
    for n_tree in range(n_estimators):
        sample = df_train.sample(
            n=round(len(df_train) * sample_size), replace=True)
        tree = build_tree(sample, ls_x, max_depth, min_samples_split,
                          max_features, min_decrease)  ##造树
        trees.append(tree)

    ##predict data
    dict_test_data = x0['test']
    df_test_data = dict_test_data[dict_test_data.fitno == i]
    ls_date_test = list(np.unique(list(df_test_data.snapshot)))

    predictions = [
        bagging_predict(trees, df_test_data, i, date) for date in ls_date_test
    ]  ##对test数据集中每个观测进行predict
    df_pre = pd.concat(predictions, axis=0)
    df_pre_max = df_pre.pivot_table(
        values='wgt', index='date', columns='symbol')

    ##calculate precision
    ls_train_date = np.unique(list(df_train_daily.snapshot)).tolist()
    dict_train_precision = {
        date: calprecision(df_train_daily, date, trees)
        for date in ls_train_date
    }
    train_Residual = [
        pd.DataFrame(
            data=[[
                calResidual(dict_train_precision[date])['var_after'],
                calResidual(dict_train_precision[date])['var_before'], date
            ]],
            columns=['var_after', 'var_before', 'date'])
        for date in ls_train_date
    ]
    train_Residual_final = pd.concat(train_Residual, axis=0)

    predictions_test = {
        date: calprecision(df_test_data, date, trees)
        for date in ls_date_test
    }
    test_Residual = [
        pd.DataFrame(
            data=[[
                calResidual(predictions_test[date])['var_after'],
                calResidual(predictions_test[date])['var_before'], date
            ]],
            columns=['var_after', 'var_before', 'date'])
        for date in ls_date_test
    ]
    test_Residual_final = pd.concat(test_Residual, axis=0)

    return {
        'targetwgt': df_pre_max,
        'train_Residual': train_Residual_final,
        'test_Residual': test_Residual_final
    }


# Build a decision tree
def build_tree(train, ls_x, max_depth, min_samples_split, max_features,
               min_decrease):
    root = get_split(train, ls_x, max_features)
    split(root, ls_x, max_depth, min_samples_split, max_features, 1,
          min_decrease)
    return root


# Select the best split point for a dataset
def get_split(train, ls_x, max_features):  ##每次split，都要重新不放回抽样一遍feature
    #class_values = list(set(row[-1] for row in train))
    b_score = 999999
    features = random.sample(ls_x, max_features)

    for index in features:
        for i in list(range(1, 11)):
            groups = test_split(index, i, train)
            mse = mse_index(groups)
            if mse < b_score:
                b_index, b_value, b_score, b_groups = index, i, mse, groups
    return {
        'index': b_index,
        'value': b_value,
        'score': b_score,
        'groups': b_groups
    }


##返回值：用哪个变量分割，threshold是什么，返回split之后的两个节点
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, train):
    left, right = train[train[index] < value], train[train[index] >= value]
    return left, right


# Calculate the mean squared error for a split dataset
def mse_index(groups):
    ls_mse = [
        np.var(groups[i]['y']) * len(groups[i]) for i in range(len(groups))
        if len(groups[i]) != 0
    ]
    mse = sum(ls_mse)
    return mse


# Create child splits for a node or make terminal
def split(node, ls_x, max_depth, min_samples_split, max_features, depth,
          min_decrease):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if len(left) == 0 or len(right) == 0:
        onegroup = pd.concat([left, right])
        node['left'] = node['right'] = to_terminal(onegroup)
        return
# check for max depth

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
# process left child
    if len(left) <= min_samples_split:
        node['left'] = to_terminal(left)
    else:
        temp = get_split(left, ls_x, max_features)
        if node['score'] - temp['score'] <= min_decrease:  ###若mse decrease < 0.00001，那么就stop
            node['left'] = to_terminal(left)
        else:
            node['left'] = temp  ##这里的结果就是dict里面套一个dict
            split(node['left'], ls_x, max_depth, min_samples_split,
                  max_features, depth + 1, min_decrease)

# process right child
    if len(right) <= min_samples_split:
        node['right'] = to_terminal(right)
    else:
        temp = get_split(right, ls_x, max_features)
        if node['score'] - temp['score'] <= min_decrease:
            node['right'] = to_terminal(right)
        else:
            node['right'] = temp
            split(node['right'], ls_x, max_depth, min_samples_split,
                  max_features, depth + 1, min_decrease)


# Create a terminal node value
def to_terminal(group):  ##如果是叶节点，那么返回该叶节点下期stock return的均值作为预测值
    return np.mean(group['y'])


# Make a prediction with a list of bagged trees


def bagging_predict(trees, df_test_data, i, date):
    temp = df_test_data[df_test_data.snapshot == date]
    df_one_test = temp.drop(['snapshot', 'symbol', 'fitno', 'y'], axis=1)
    ser_onetest = df_one_test.apply(lambda x: tree_pre(x, trees), axis=1)
    ser_rank = pd.DataFrame(
        data=gsUtils.cut2bin(ser_onetest, 10, ascending=True), columns=['rank'])
    df_rank = pd.concat([ser_rank, temp], axis=1)
    wgt = 1 / len(df_rank[df_rank['rank'] == 10])
    df_fnl = df_rank[df_rank['rank'] == 10][['symbol']].assign(wgt=wgt)
    return df_fnl.assign(date=date)


def calprecision(data, date, trees):
    temp = data[data.snapshot == date]
    df_one_test = temp.drop(['snapshot', 'symbol', 'fitno', 'y'], axis=1)
    ser_onetest = pd.DataFrame(
        data=df_one_test.apply(lambda x: tree_pre(x, trees), axis=1),
        columns=['yhat'])
    ser_onetest_all = pd.concat(
        [
            ser_onetest,
            temp.reindex(columns=['symbol', 'snapshot', 'fitno', 'y'])
        ],
        axis=1)
    return ser_onetest_all


#def calRsqure(data):
#    ymean=data.y.mean()
#    SST=sum((data.y-ymean)**2)
#    SSE=sum((data.yhat-ymean)**2)
#    return float(SSE/SST)


def calResidual(data):
    var_after = sum((data.y - data.yhat)**2) / data.shape[0]
    var_before = sum((data.y - data.y.mean())**2) / data.shape[0]
    return {'var_after': var_after, 'var_before': var_before}


def tree_pre(x, trees):
    predictions = [predict(tree, x) for tree in trees]
    return np.mean(predictions)


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
