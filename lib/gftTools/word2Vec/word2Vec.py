# -*- coding: utf-8 -*-
import sys
from gensim.models import KeyedVectors
import logging
from . import preprocessing
import pandas as pd
import jieba
from copy import copy

# Martin 2017-12-28 This is a temp file position. Should be changed when organize multiple models are planned.
data_folder = preprocessing.data_folder

jieba.load_userdict(data_folder + "jieba.txt")
category_model_dir = data_folder + 'models/categories_w10_m5/'

# similar words number

# compare the model list from the target word.
categories = [
    u"上市公司", u"期货交易所", u"会计学", u"债券", u"利息", u"基本面", u"基金公司", u"技术分析", u"投资",
    u"策略", u"损益表", u"三板", u"时间序列", u"现金", u"股权投资", u"私人股权投资", u"经济", u"经济学",
    u"股票", u"证券", u"证券投资基金", u"财务会计", u"货币", u"资产", u"资产管理", u"资产负债表", u"金融",
    u"金融公司", u"金融学", u"金融市场", u"风险投资", u"风险投资公司"
]


def pre_load_w2v_model(categories, path):
    """
    Keyword Arguments:
    categories -- categories model list
    dir        -- categories model path
    """
    word2vec_model = {}
    for name in categories:
        # print(name)
        word2vec_model[name] = KeyedVectors.load_word2vec_format(
            path + name + '.w2v_org')
    word2vec_model['whole_wiki'] = KeyedVectors.load_word2vec_format(
        data_folder + 'models/word2vec_org_whole_wiki_corpus_user_dict_m5')
    return word2vec_model


def pre_load_w2v_vocab(categories, path):
    """
    Keyword Arguments:
    categories -- categories model list
    dir        -- categories model path
    """
    word2vec_vocab = {}
    for name in categories:
        # print(name)
        word2vec_vocab[name] = pd.read_csv(
            path + name + '.vocab',
            delim_whitespace=True,
            header=None,
            encoding='utf-8')
    return word2vec_vocab


# load the whole corpus
zh_vocab = pre_load_w2v_vocab(categories, category_model_dir)
zh_model = pre_load_w2v_model(categories, category_model_dir)


def find_model(model, cat_list, token):
    try:
        target_category_string = model.most_similar_to_given(token, cat_list)
    except KeyError:
        return None
    # print("path %s" % vocab_path)
    vocab = zh_vocab[target_category_string]
    if token not in vocab[0].tolist():
        # print("token %s not in %s" % (token, target_category_string))
        cat_list.remove(target_category_string)
        # print(len(cat_list))
        return find_model(model, cat_list, token)
    else:
        target = target_category_string
        # print('use %s' % target)
        return zh_model[target]

def word_2_vec_wiki_pedia(input_text, top_n):
    # preprocessing input, remove punctuation and stopwords
    filters = [preprocessing.strip_punctuation, preprocessing.remove_stopwords]
    filterpunt_text = preprocessing.preprocess_string(input_text, filters)
    tokens_generator = jieba.cut(filterpunt_text)
    tokens = [x for x in tokens_generator if not x.isspace()]

    result = dict()
    for token in tokens:
        model_result = list()

        tmp_list = copy(categories)
        model = find_model(zh_model['whole_wiki'], tmp_list, token)

        if model:
            raw_model_result = model.most_similar(token, topn=top_n)
            for el in raw_model_result:
                model_result.append(el[0])

        result[token] = model_result

    return result
