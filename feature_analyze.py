#!usr/bin/env python  
#-*- coding: utf-8 -*-  
  
import sys  
import os  
import time 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import data

from sklearn import preprocessing

reload(sys)  
sys.setdefaultencoding('utf8')  

train_df = pd.read_csv('train.txt', header=0)
columns = train_df.columns.tolist()
print columns
#print train_df.info()

#data.display_data('train.txt', True)

single_count_dir = 'single_feat_count'
single_class_dir = 'single_feat_class'
label = 'class'
#for feat in columns:
#    data.show_one_feature_count(train_df, feat, single_count_dir)
    #data.show_one_feature_class(train_df, feat, label, single_class_dir)
data.show_features_corr(train_df)
#train_df['domain_suf'] = train_df['domain_suf'].apply(lambda x : 'else' if(x not in ['com', 'cn', 'cc', 'net']) else x)
#train_df['url_suf'] = train_df['url_suf'].apply(lambda x : 'else' if(x not in ['html', 'NAN', 'php', 'asp', 'shtml', 'htm']) else x)
#train_df = data.hot_coding(train_df, 'html_level')
#train_df = data.hot_coding(train_df, 'domain_suf')
#train_df = data.hot_coding(train_df, 'url_suf')
#print train_df.head()