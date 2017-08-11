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
    data.display_data("train.txt_with_header")
    #???
    #test_file = "anchor_url_feat"
    #raw_url_file = "anchor_url"
    '''
    train_file = "train.txt"
    test_file = "base_doc_shard_0_16w_feat"
    raw_url_file = "base_doc_shard_0_16w"
    predict_res_file = "predict_lottory"
    '''
    train_file = "train.hub"
    test_file = "test.hub"
    raw_url_file = "test.raw"
    predict_res_file = "predict_hub"
    
    print 'bro?'
    is_new = True
    if is_new == True:
        print 'new'
        train_x, train_y, test_x, test_y = feature_analyze.format_data(train_file, test_file)
    else:
        print 'load'
        train_x, train_y, test_x, test_y = data.load_data(train_file + "_ed", test_file + "_ed")
        
    #print train_x.columns
    print train_x.shape, test_x.shape

    SEED = 0
    fold_num = 5

    rf = train_model.ModelHelper(clf=RandomForestClassifier, seed=SEED, params=train_model.loadParams('rf'))
    et = train_model.ModelHelper(clf=ExtraTreesClassifier, seed=SEED, params=train_model.loadParams('et'))
    ada = train_model.ModelHelper(clf=AdaBoostClassifier, seed=SEED, params=train_model.loadParams('ada'))
    gb = train_model.ModelHelper(clf=GradientBoostingClassifier, seed=SEED, params=train_model.loadParams('gb'))
    svc = train_model.ModelHelper(clf=SVC, seed=SEED, params=train_model.loadParams('svc'))

    
    model_new = True
    if model_new:
        rf = RandomForestClassifier()
        rf = train_model.optimizeModel(rf, "rf", conf.rf_grid_params, train_x, train_y)
        et = ExtraTreesClassifier()
        et = train_model.optimizeModel(et, "et", conf.et_grid_params, train_x, train_y)
        ada = AdaBoostClassifier()
        ada = train_model.optimizeModel(ada, "ada", conf.ada_grid_params, train_x, train_y)
        gb = GradientBoostingClassifier()
        gb = train_model.optimizeModel(gb, "gb", conf.gb_grid_params, train_x, train_y)
        svc = SVC()
        svc = train_model.optimizeModel(svc, "svc", conf.svc_grid_params, train_x, train_y)

    oob_new = True
    if oob_new:
        rf_oof_train, rf_oof_test = data.get_oof(rf, train_x, train_y, test_x, fold_num, SEED, 'rf') # Random Forest
        et_oof_train, et_oof_test = data.get_oof(et,  train_x, train_y, test_x, fold_num, SEED, 'et') # Extra Trees
        ada_oof_train, ada_oof_test = data.get_oof(ada, train_x, train_y, test_x, fold_num, SEED, 'ada') # AdaBoost 
        gb_oof_train, gb_oof_test = data.get_oof(gb, train_x, train_y, test_x, fold_num, SEED, 'gb') # Gradient Boost
        svc_oof_train, svc_oof_test = data.get_oof(svc, train_x, train_y, test_x, fold_num, SEED, 'svc') # Support Vector Classifier
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
        #level_one_output = open("level_one_output.pkl", 'wb')
        level_one_output = open("hub_level_one_output.pkl", 'wb')
        output_list = [train_x, test_x]
        pickle.dump(output_list, level_one_output)
        level_one_output.close()
    else:
        print 'load model'
        #level_one_input = open("level_one_output.pkl", 'r')
        level_one_input = open("hub_level_one_output.pkl", 'r')
        level_output_list = pickle.load(level_one_input)
        train_x = level_output_list[0]
        test_x = level_output_list[1]
        print 'load  level one output'
        
    xg = xgb.XGBClassifier(
                 #learning_rate = 0.02,
                 n_estimators= 2000,
                 max_depth= 4,
                 min_child_weight= 2,
                 #gamma=1,
                 gamma=0.9,                        
                 subsample=0.8,
                 colsample_bytree=0.8,
                 objective= 'binary:logistic',
                 #nthread= -1,
                 scale_pos_weight=1)
    xg_new = True
    if xg_new:
        xg = train_model.optimizeModel(xg, "xg", conf.xg_grid_params, train_x, train_y)
    xg = train_model.ModelHelper(clf=xgb.XGBClassifier, seed=SEED, params=train_model.loadParams('xg')).fit(train_x, train_y)
        
    
    predict_new = True
    predicts = []
    if predict_new:
        predicts = xg.predict(test_x)
        #output = open('predict.pkl', 'wb')
        output = open('hub_predict.pkl', 'wb')
        pickle.dump(predicts, output)
        output.close()        
    else:
        #input = open('predict.pkl', 'rb')
        input = open('hub_predict.pkl', 'rb')
        predicts = pickle.load(input)
        input.close()
        print Counter(predicts)
        
    print predicts
    all_pos_index = np.where(predicts >= 1)
    urls = pd.read_table(raw_url_file, header=None)
    urls = urls.ix[all_pos_index]
    urls.to_csv(predict_res_file)
    

    
    
    

    
    

  