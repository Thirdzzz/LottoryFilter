#!usr/bin/env python  
#-*- coding: utf-8 -*-  
  
import sys  
import os  
import time
  
import pickle 
import numpy as np  
import pandas as pd
from collections import Counter
import xgboost as xgb

from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split, GridSearchCV

import conf
import data
import feature_analyze
import train_model
import test_model

  
reload(sys)  
sys.setdefaultencoding('utf8')  
  
if __name__ == '__main__':
    '''
    train_file = "train.txt"
    #test_file = "cc_100000_feat"
    #raw_url_file = "cc_100000"
    test_file = "anchor_url_feat"
    raw_url_file = "anchor_url"
    #test_file = "dup_url_feat"
    #raw_url_file = "dup_url.csv"

    #test_file = "test.csv"
    predict_res_file = "predict_lottory"

    train_x, train_y = data.read_data(train_file)
    test_x, test_y = data.read_data(test_file)
    #train_x = data.scale01(train_x)
    #test_x = data.scale01(test_x)
    train_x = data.standard(train_x)
    test_x = data.standard(test_x)

    threshold = 0.8
    sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    sel.fit(train_x)
    train_x = sel.transform(train_x)
    test_x = sel.transform(test_x)
    print train_x.shape
    
    sel = SelectKBest(f_classif, k=40)
    print train_x
    sel.fit(train_x, train_y)
    train_x = sel.transform(train_x)
    test_x = sel.transform(test_x)
    print train_x.shape

    sel = RandomForestRegressor(n_estimators=1000, max_depth=20)
    sel.fit(train_x, train_y)
    train_x = sel.transform(train_x)
    test_x = sel.transform(test_x)    
    print train_x.shape
    '''
    '''
    train_df = pd.read_csv('train.txt', header=0)
    test_df = pd.read_csv('anchor_url_feat', header=0)
    train_num = train_df.shape[0]

    train_y = train_df['class']
    train_x = train_df.drop(['class'], axis=1)
    test_y = test_df['class']
    test_x = test_df.drop(['class'], axis=1)
    all_x = pd.concat([train_x, test_x])
    print all_x.shape
    all_num = all_x.shape[0]

    all_x = data.add_features(all_x)
    all_x = data.scale01(all_x)
    all_x = data.removeFeatByVar(all_x)
    print all_x.shape
    train_x = all_x.iloc[0:train_num, :]
    test_x = all_x.iloc[train_num:all_num, :]
    train_x, test_x = data.selectBestFeat(train_x, train_y, test_x, 100)
    print train_x.columns
    print test_x.columns


    data_nums, col_nums = test_x.shape
    print data_nums, col_nums

    num_train, num_feat = train_x.shape  
    num_test, num_feat = test_x.shape  
    is_binary_class = (len(np.unique(train_y)) == 2) 

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT'] 
    #test_classifiers = ['KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT'] 
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
        print Counter(predict)
        predicts = predict + predicts
        print Counter(predicts)
    
    
    #print predicts
    all_pos_index = np.where(predicts >= sum_pos)
    urls = pd.read_table(raw_url_file)
    urls = urls.ix[all_pos_index]
    urls.to_csv(predict_res_file)

    '''
    train_file = "train.txt"
    test_file = "anchor_url_feat"
    raw_url_file = "anchor_url"
    predict_res_file = "predict_lottory"

    is_new = False
    if is_new == True:
        print 'new'
        train_x, train_y, test_x, test_y = feature_analyze.format_data(train_file, test_file)
    else:
        print 'load'
        train_x, train_y, test_x, test_y = data.load_data(train_file + "_ed", test_file + "_ed")
    #print train_x.columns
    
    #svc = SVC()
    #svc = train_model.optimizeModel(svc, "svc", conf.svc_grid_params, train_x, train_y)
    rf = RandomForestClassifier()
    rf = train_model.optimizeModel(rf, "rf", conf.rf_grid_params, train_x, train_y)
    
    '''
    SEED = 0
    fold_num = 5

    rf = train_model.ModelHelper(clf=RandomForestClassifier, seed=SEED, params=conf.rf_params)
    et = train_model.ModelHelper(clf=ExtraTreesClassifier, seed=SEED, params=conf.et_params)
    ada = train_model.ModelHelper(clf=AdaBoostClassifier, seed=SEED, params=conf.ada_params)
    gb = train_model.ModelHelper(clf=GradientBoostingClassifier, seed=SEED, params=conf.gb_params)
    svc = train_model.ModelHelper(clf=SVC, seed=SEED, params=conf.svc_params)

    # Create our OOF train and test predictions. These base results will be used as new features
    rf_oof_train, rf_oof_test = data.get_oof(rf, train_x, train_y, test_x, fold_num, SEED) # Random Forest
    et_oof_train, et_oof_test = data.get_oof(et,  train_x, train_y, test_x, fold_num, SEED) # Extra Trees
    ada_oof_train, ada_oof_test = data.get_oof(ada, train_x, train_y, test_x, fold_num, SEED) # AdaBoost 
    gb_oof_train, gb_oof_test = data.get_oof(gb, train_x, train_y, test_x, fold_num, SEED) # Gradient Boost
    svc_oof_train, svc_oof_test = data.get_oof(svc, train_x, train_y, test_x, fold_num, SEED) # Support Vector Classifier
    print("Training is complete")
    
    base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
         'ExtraTrees': et_oof_train.ravel(),
         'AdaBoost': ada_oof_train.ravel(),
         'GradientBoost': gb_oof_train.ravel(),
         'SVC': svc_oof_train.ravel(),
        })
    base_predictions_train.head()
    data.show_features_corr(base_predictions_train)
    
    train_x = np.concatenate((rf_oof_train, et_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
    test_x = np.concatenate((rf_oof_test, et_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
    predicts = test_x.sum(axis=1)
    output = open('predict.pkl', 'wb')
    pickle.dump(predicts, output)
    output.close()
    '''
    input = open('predict.pkl', 'rb')
    predicts = pickle.load(input)
    input.close()
    print Counter(predicts)
    '''
    gbm = xgb.XGBClassifier(
         #learning_rate = 0.02,
         n_estimators= 2000,
         max_depth= 4,
         min_child_weight= 2,
         #gamma=1,
         gamma=0.9,                        
         subsample=0.8,
         colsample_bytree=0.8,
         objective= 'binary:logistic',
         nthread= -1,
         scale_pos_weight=1).fit(train_x, train_y)
    predicts = gbm.predict(test_x)
    '''
    
    
    
    
    #print predicts
    all_pos_index = np.where(predicts >= 5)
    urls = pd.read_table(raw_url_file, header=None)
    urls = urls.ix[all_pos_index]
    urls.to_csv(predict_res_file)
    
    

  