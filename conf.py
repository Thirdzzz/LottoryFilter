#!usr/bin/env python  
#-*- coding: utf-8 -*-  

import numpy as np

rf_params = {
    'n_estimators': 100,
    'max_depth': 8,
    'max_features' : 'auto',
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 2.0
}

rf_grid_params = [
    {
        'n_estimators':[50, 100, 300],
        'max_depth': range(1, 10, 1),
        'max_features': ['auto', 'log2', 'sqrt']
    }
]
et_grid_params = [
    {
        'n_estimators':range(1,1001,200),
        'max_depth': range(1, 10, 1),
        'max_features': ['auto', 'log2', 'sqrt']
    }
]
ada_grid_params = [
    {
        'n_estimators':range(1,100,10),
        'learning_rate':np.linspace(0.1, 1, 10)
    }
]
gb_grid_params = [
    {
        'learning_rate':[0.05, 0.1, 0.15, 0.2],
        'max_depth': range(5, 15, 2),
        'max_features': ['auto', 'log2', 'sqrt']
    }
]

svc_grid_params = [
    {
        'kernel': ['rbf'], 
        'C': np.linspace(1, 3, 3), 
        'gamma': np.linspace(0.001, 0.01, 3)
    },
    {
        'kernel': ['linear'], 
        'C': np.linspace(1, 3, 3)
    }
]

xg_grid_params = [
    {
        'learning_rate': np.linspace(0.01, 0.3, 5),
        'gamma': np.linspace(0.1, 1, 5),
        'max_depth': range(3, 10, 2),
        'min_child_weight': [2],
        'colsample_bytree': [0.8],
        'objective': ['binary:logistic'],
    }
]