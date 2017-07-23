#!usr/bin/env python  
#-*- coding: utf-8 -*-  
  
import sys  
import os  

from sklearn import preprocessing
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
  
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

def read_data_with_head(data_file):
    df = pd.read_csv(data_file, header=0)
    columns = df.columns.tolist()
    max_abs_scaler = preprocessing.MinMaxScaler().fit(df)
    df_array = max_abs_scaler.transform(df)
    df = pd.DataFrame(df_array, columns=columns)
    feat_mat = df.describe()                #各列特征的一个基本简况
    print feat_mat
    #return 0

    scaler = preprocessing.StandardScaler().fit(df)
    df_array = scaler.transform(df)
    df = pd.DataFrame(df_array, columns=columns)
    # Finding out basic statistical information on your dataset.
    pd.options.display.float_format = '{:,.3f}'.format # Limit output to 3 decimal places.
    
    if not os.path.exists('feature_images'):
        os.makedirs('feature_images')

    feat_mat = df.describe()                #各列特征的一个基本简况
    print feat_mat
    return 0
    #feat_mat.T.to_excel("feature_images/feature_material.xlsx")

    corrmat = df.corr(method = 'spearman')  #各特征间的一个相关系数
    corrmat.to_excel("feature_images/corr_map.xlsx")
    #return 0
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
    read_data_with_head("train.txt_with_header")