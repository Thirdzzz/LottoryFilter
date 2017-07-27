#!usr/bin/env python  
#-*- coding: utf-8 -*-  
  
import sys  
import os  
import time 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.lda import LDA

import data

from sklearn import preprocessing

reload(sys)  
sys.setdefaultencoding('utf8')  

def format_data(train_file, test_file):
    train_df = pd.read_csv(train_file, header=0)
    test_df = pd.read_csv(test_file, header=0)
    train_num = train_df.shape[0]

    train_y = train_df['class']
    train_x = train_df.drop(['class'], axis=1)
    test_y = test_df['class']
    test_x = test_df.drop(['class'], axis=1)
    all_x = pd.concat([train_x, test_x])
    all_num = all_x.shape[0]

    all_x = data.add_features(all_x)
    all_x = data.scale01(all_x)
    all_x = data.removeFeatByVar(all_x)
    train_x = all_x.iloc[0:train_num, :]
    test_x = all_x.iloc[train_num:all_num, :]
    test_x = test_x.reset_index(drop=True)
    train_x, test_x = data.selectBestFeat(train_x, train_y, test_x, 100)
    train = pd.concat([train_x, train_y], axis = 1)
    test = pd.concat([test_x, test_y], axis = 1)
    train.to_csv(train_file + "_ed", index=False)
    test.to_csv(test_file + "_ed", index=False)
    return train_x, train_y, test_x, test_y

#train_x, train_y, test_x, test_y = format_data("train.txt", "anchor_url_feat")

'''
train_x = train_df.drop(['class'], axis=1)
train_x = data.scale01(train_x)
train_x = data.removeFeatByVar(train_x)

train_x = data.selectBestFeat(train_x, train_y, 100)


#train_df = data.add_features(train_df)
#train_df.to_csv('train_new.txt')

print train_x.info()
columns = train_x.columns.tolist()
print columns
print len(columns)

#data.display_data('train.txt', True)

single_count_dir = 'single_feat_count'
single_class_dir = 'single_feat_class'
label = 'class'
#for feat in columns:
#    data.show_one_feature_count(train_df, feat, single_count_dir)
    #data.show_one_feature_class(train_df, feat, label, single_class_dir)
#data.show_features_corr(train_x)
#train_df['domain_suf'] = train_df['domain_suf'].apply(lambda x : 'else' if(x not in ['com', 'cn', 'cc', 'net']) else x)
#train_df['url_suf'] = train_df['url_suf'].apply(lambda x : 'else' if(x not in ['html', 'NAN', 'php', 'asp', 'shtml', 'htm']) else x)
#train_df = data.hot_coding(train_df, 'html_level')
#train_df = data.hot_coding(train_df, 'domain_suf')
#train_df = data.hot_coding(train_df, 'url_suf')
#print train_df.head()
'''