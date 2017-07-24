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
print train_df.info()
#data.display_data('train.txt', True)
#data.show_one_feature(train_df, 'html_level')
train_df = data.hot_coding(train_df, 'html_level')
print train_df.head()