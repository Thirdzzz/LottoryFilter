#!usr/bin/env python  
#-*- coding: utf-8 -*-  
  
import sys  
import os  

from sklearn import preprocessing
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

  
reload(sys)  
sys.setdefaultencoding('utf8')  
  
def read_data(data_file):  
    data = np.loadtxt(data_file, delimiter=",")
    attr_num = len(data[0]) - 1
    X = data[:, 0:attr_num]  
    Y = data[:, -1]  
    #X = preprocessing.scale(X)
    #Y = preprocessing.scale(Y)
    #print X, Y
    return X, Y  

def scale01(data):
    max_abs_scaler = preprocessing.MinMaxScaler().fit(data)
    data = max_abs_scaler.transform(data)
    return data

def standard(data):
    standarder = preprocessing.StandardScaler().fit(data)
    data = standarder.transform(data)
    return data

#移除低variance的特征
def removeFeatByVar(X, threshold):
    #columns = X.columns.tolist()
    sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    X_array = sel.fit_transform(X)
    return X_array
    #X = pd.DataFrame(X_array, columns=columns)
    #return X

def selectBestFeat(X, Y, k):
    X_new = SelectKBest(chi2, k=k).fit_transform(X, Y)
    rf = RandomForestRegressor(n_estimators=1000, max_depth=20)
    rf.fit(X_new, Y)
    X_res = rf.transform(X_new)
    '''
    clf = Pipeline([
      ('feature_selection', SelectFromModel(LinearSVC(loss='l2', penalty='l1', dual=False))),
      ('classification', RandomForestClassifier())
    ])
    clf.fit(X_new, Y)
    X_res = clf.transform(X_new)
    '''
    print X_res.shape
    return X_res


def display_data(data_file, draw_pic=False):
    df = pd.read_csv(data_file, header=0)
    pd.options.display.float_format = '{:,.3f}'.format # Limit output to 3 decimal places.
    columns = df.columns.tolist()

    feat_mat = df.describe()                #各列特征的一个基本简况
    print feat_mat
    #return 0

    scaler = preprocessing.StandardScaler().fit(df)
    df_array = scaler.transform(df)
    df = pd.DataFrame(df_array, columns=columns)
    # Finding out basic statistical information on your dataset.


    
    if draw_pic:
        if not os.path.exists('feature_images'):
            os.makedirs('feature_images')
        feat_mat = df.describe()                #各列特征的一个基本简况
        print feat_mat
        #feat_mat.T.to_excel("feature_images/feature_material.xlsx")

        corrmat = df.corr(method = 'spearman')  #各特征间的一个相关系数
        corrmat.to_excel("feature_images/corr_map.xlsx")

        f, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(corrmat, vmax=1., square=True)
        plt.xticks(rotation='vertical')
        plt.yticks(rotation='horizontal')               #转热力图保存
        plt.title("Leaky variables correlation map", fontsize=15)
        plt.savefig("feature_images/corr_map.png")
        plt.clf()   

        #plt.show()
        plt.figure(figsize=(12,6))
        for column in columns:
            sns.boxplot(x="class", y=column, data=df)   #各特征在各类型class下的数据分布箱线图
            plt.xlabel('Is lottory', fontsize=12)
            plt.ylabel(column, fontsize=12)
            plt.savefig("feature_images/boxplot_"+column+".png")
            plt.clf()
    

    

if __name__ == '__main__':
    display_data("train.txt_with_header")