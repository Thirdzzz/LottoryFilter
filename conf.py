#!usr/bin/env python  
#-*- coding: utf-8 -*-  

import numpy as np

rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
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
rf_grid_params = [
    {
        'n_estimators':range(1,1000,100),
        'max_depth': range(1, 100, 10),
        'max_features': ['None', 'log2', 'sqrt']
    }
]