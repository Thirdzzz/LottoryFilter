from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import sys  
import os  
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import data
import numpy as np  
  
reload(sys)  
sys.setdefaultencoding('utf8')


#Load boston housing dataset as an example
datas = np.loadtxt("train.txt", delimiter=",")
attr_num = len(datas[0]) - 1
X = datas[:, 0:attr_num]
Y = datas[:, -1]
with open("feature_names.txt", 'r') as f:
    feature_labels = f.readlines()

rf = RandomForestRegressor(n_estimators=20, max_depth=4)

'''
scores = []
for i in range(X.shape[1]):
    score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                            cv=ShuffleSplit(len(X), 3, .3))
    scores.append((round(np.mean(score), 3), feature_labels[i]))
scores = sorted(scores, reverse=True)
for score in scores:
    print score
    '''
rf.fit(X, Y)
scores = []
for feature in zip(rf.feature_importances_, feature_labels):
    scores.append(feature)
scores = sorted(scores, reverse=True)
for score in scores:
    print score

X = data.removeFeatByVar(X, 0.8)
print X.shape
X = data.selectBestFeat(X, Y, X.shape[1])

    
