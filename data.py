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
    columns = data.columns.tolist()
    max_abs_scaler = preprocessing.MinMaxScaler().fit(data)
    data = max_abs_scaler.transform(data)
    data = pd.DataFrame(data, columns=columns)
    return data

def standard(data):
    standarder = preprocessing.StandardScaler().fit(data)
    data = standarder.transform(data)
    return data

def show_one_feature_count(df, feat, dir):
    feat_df = df[feat].value_counts()
    plt.figure(figsize=(8, 4))
    sns.barplot(feat_df.index, feat_df.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(feat, fontsize=12)
    #plt.show()
    draw_pic(plt, dir, feat)
    plt.clf()
    
def show_one_feature_class(df, feat, label, dir):
    plt.figure(figsize=(8, 4))
    ax = plt.axes()
    data = df.groupby([feat])[[label]].count()
    sns.barplot(x=data.index, y=data[label], alpha=0.8, color='violet', ax=ax)
    data = df.groupby([feat])[[label]].sum()
    sns.barplot(x=data.index, y=data[label], alpha=0.8, color='black', ax=ax)    
    draw_pic(plt, dir, feat)
    plt.clf()

def show_features_corr(df):
    #plt.figure(figsize=(25, 25))
    corr = df.corr()
    f, ax = plt.subplots(figsize=(25, 16))
    '''
    plt.yticks(fontsize=18, rotation='horizontal')
    plt.xticks(fontsize=18, rotation='vertical')
    sns.heatmap(corr, cmap='inferno', linewidths=0.1,vmax=1.0, square=True, annot=True)
    '''
    sns.heatmap(corr, vmax=1., square=True) 
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal') 
    #plt.show()

    draw_pic(plt, 'features_corr', 'features_corr_heat_map')
    plt.clf()
    
def add_features(df):
    df['host_sub_level'] = df['host_level'] - df['domain_level']
    levels = ['host', 'path', 'file', 'domain']
    parts = ['num', 'char', 'flag', 'diff']
    flags = ['_', '-', '#', '@', 'qm', 'pm', 'comma', 'dot']
    df['url_len'] = df['host_len'] +df['path_len'] + df['file_len']
    for part in parts:
        df['url_' + part] = 0
        for level in levels:
            df['url_' + part] = df['url_' + part] + df[level + "_" + part]
        df['url_' + part + "_ratio"] = df['url_' + part] / (df['url_len'] + 1)
    for level in levels:
        for part in parts:
            df[level + "_" + part + "_ratio"] = df[level + "_" + part] / (df[level + "_len"] + 1)
    df['domain_suf'] = df['domain_suf'].apply(lambda x : 'else' if(x not in ['com', 'cn', 'cc', 'net']) else x)
    df = hot_coding(df, 'domain_suf')
    df['url_suf'] = df['url_suf'].apply(lambda x : 'else' if(x not in ['html', 'NAN', 'php', 'asp', 'shtml', 'htm']) else x)
    df = hot_coding(df, 'url_suf') 
    df = hot_coding(df, 'html_level')   
    return df
    
    
def hot_coding(df, feat):
    feat_dummies = pd.get_dummies(df[feat])
    columns = feat_dummies.columns.tolist()
    for i in range(0, len(columns)):
        columns[i] = feat + "_" + columns[i]
    feat_dummies.columns = columns
    df = pd.concat([df, feat_dummies], axis = 1)
    df.drop([feat], axis=1, inplace=True)
    return df

def draw_pic(plt, dir, file_name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + "/" + file_name + ".png")

#移除低variance的特征, 默认去除0方差，即一列全为同一值的特征
def removeFeatByVar(X): 
    columns = X.columns.tolist()
    sel = VarianceThreshold()
    sel.fit(X)
    select_indices = sel.get_support(indices=True)
    print len(select_indices)
    X_select = X.iloc[:, select_indices]
    #print X_select
    return X_select
    #X_array = sel.transform(X)
    #X = pd.DataFrame(X_array)
    #return X

def selectBestFeat(X, Y, test_x, k):
    #X = SelectKBest(chi2, k=k).fit_transform(X, Y)
    #rf = RandomForestRegressor(n_estimators=1000, max_depth=20)
    #rf.fit(X, Y)
    #X = rf.transform(X)
    rf = RandomForestClassifier(n_estimators=1000, max_depth=10)
    rf.fit(X, Y)
    draw_feat_importance(X, rf)
    model = SelectFromModel(rf, prefit=True)
    print model.get_support(indices=True)
    X = model.transform(X)
    X = pd.DataFrame(X)
    test_x = model.transform(test_x)
    test_x = pd.DataFrame(test_x)
    '''
    clf = Pipeline([
      ('feature_selection', SelectFromModel(LinearSVC(loss='l2', penalty='l1', dual=False))),
      ('classification', RandomForestClassifier())
    ])
    clf.fit(X_new, Y)
    X_res = clf.transform(X_new)
    '''
    print X.shape, test_x.shape
    return X, test_x

def draw_feat_importance(X, model):
    plt.figure(figsize=(20,25))
    plt.barh(np.arange(X.columns.shape[0]), model.feature_importances_, 0.5)
    plt.yticks(np.arange(X.columns.shape[0]), X.columns)
    plt.grid()
    plt.xticks(np.arange(0, max(model.feature_importances_), 0.01))
    #plt.show()
    draw_pic(plt, 'feature_importance', 'importance')
    plt.clf()
    


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