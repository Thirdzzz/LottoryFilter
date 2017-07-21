#!usr/bin/env python  
#-*- coding: utf-8 -*-  
  
import sys  
import os  

from sklearn import preprocessing
import numpy as np  
  
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

