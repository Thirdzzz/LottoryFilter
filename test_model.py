#!usr/bin/env python  
#-*- coding: utf-8 -*-  
  
import sys  
import os  
import time  
from sklearn import metrics
import numpy as np  
import pickle 
import data
  
reload(sys)  
sys.setdefaultencoding('utf8')  
  
def predict(model, test_x, test_y, is_binary_class):
    predict = model.predict(test_x)  
    F1Score = 0
    #if model_save_file != None:  
    #    model_save[classifier] = model  
    if is_binary_class:  
        precision = metrics.precision_score(test_y, predict)  
        recall = metrics.recall_score(test_y, predict)  
        F1Score = 2 * precision * recall / (precision + recall)
        print 'precision: %.2f%%, recall: %.2f%%, F1Score: %.2f%%' % (100 * precision, 100 * recall, 100 * F1Score)  
    accuracy = metrics.accuracy_score(test_y, predict)  
    print 'accuracy: %.2f%%' % (100 * accuracy)  
    return predict
    

  