from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import sys  
import os  

import numpy as np  
  
reload(sys)  
sys.setdefaultencoding('utf8')


#Load boston housing dataset as an example
data = np.loadtxt("train.txt", delimiter=",")
attr_num = len(data[0]) - 1
X = data[:, 0:attr_num]
Y = data[:, -1]
with open("feature_names.txt", 'r') as f:
    names = f.readlines()

rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
    score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                            cv=ShuffleSplit(len(X), 3, .3))
    scores.append((round(np.mean(score), 3), names[i]))
scores = sorted(scores, reverse=True)
for score in scores:
    print score
