<<<<<<< HEAD
#!usr/bin/env python  
#-*- coding: utf-8 -*-  
  
import sys  
import os  
import time  
import numpy as np  
from sklearn import preprocessing
from sklearn import metrics
import pickle 
import data
import train_model
import test_model
import pandas as pd
  
reload(sys)  
sys.setdefaultencoding('utf8')  
  
if __name__ == '__main__':
    train_file = "train.txt_sub_2"
    #test_file = "cc_100000_feat"
    #raw_url_file = "cc_100000"
    test_file = "anchor_url_feat_sub_2"
    raw_url_file = "anchor_url"
    #test_file = "dup_url_feat"
    #raw_url_file = "dup_url.csv"

    #test_file = "test.csv"
    predict_res_file = "predict_lottory"

    train_x, train_y = data.read_data(train_file)
    test_x, test_y = data.read_data(test_file)
    data_nums, col_nums = test_x.shape
    print data_nums, col_nums

    num_train, num_feat = train_x.shape  
    num_test, num_feat = test_x.shape  
    is_binary_class = (len(np.unique(train_y)) == 2) 

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT'] 
    sum_pos = len(test_classifiers)
    #sum_pos = 7
    predicts = [0] * data_nums

    model_save_file = "model.pickle"  
    model_save = {}  
    #sys.exit(0)
    model_obj = train_model.TrainModel()
    for classifier in test_classifiers:  
        print '******************* %s ********************' % classifier  
        start_time = time.time()  
        #model = train_model.train(classifiers[classifier], train_x, train_y) 
        model = model_obj.train(classifier, train_x, train_y)
        print 'training took %fs!' % (time.time() - start_time)  
        predict = test_model.predict(model, test_x, test_y, is_binary_class)
        predicts = predict + predicts
    #print predicts
    all_pos_index = np.where(predicts >= sum_pos)
    urls = pd.read_table(raw_url_file)
    urls = urls.ix[all_pos_index]
    urls.to_csv(predict_res_file)
    
    

=======
#!usr/bin/env python  
#-*- coding: utf-8 -*-  
  
import sys  
import os  
import time  
import numpy as np  
from sklearn import preprocessing
from sklearn import metrics
import pickle 
import data
import train_model
import test_model
import pandas as pd
  
reload(sys)  
sys.setdefaultencoding('utf8')  
  
if __name__ == '__main__':
    train_file = "train.txt_v1"
    #test_file = "cc_100000_feat"
    #raw_url_file = "cc_100000"
    test_file = "anchor_url_feat"
    raw_url_file = "anchor_url"
    #test_file = "test.csv"
    predict_res_file = "predict_lottory"

    train_x, train_y = data.read_data(train_file)
    test_x, test_y = data.read_data(test_file)
    data_nums, col_nums = test_x.shape
    print data_nums, col_nums

    num_train, num_feat = train_x.shape  
    num_test, num_feat = test_x.shape  
    is_binary_class = (len(np.unique(train_y)) == 2) 

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT'] 
    sum_pos = len(test_classifiers)
    #sum_pos = 7
    predicts = [0] * data_nums

    model_save_file = "model.pickle"  
    model_save = {}  
    #sys.exit(0)
    model_obj = train_model.TrainModel()
    for classifier in test_classifiers:  
        print '******************* %s ********************' % classifier  
        start_time = time.time()  
        #model = train_model.train(classifiers[classifier], train_x, train_y) 
        model = model_obj.train(classifier, train_x, train_y)
        print 'training took %fs!' % (time.time() - start_time)  
        predict = test_model.predict(model, test_x, test_y, is_binary_class)
        predicts = predict + predicts
    #print predicts
    all_pos_index = np.where(predicts >= sum_pos)
    urls = pd.read_table(raw_url_file)
    urls = urls.ix[all_pos_index]
    urls.to_csv(predict_res_file)
    
    

>>>>>>> 3db818bbf24d75180002c8646dd428fdc21faa24
  