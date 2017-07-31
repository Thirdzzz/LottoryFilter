#!usr/bin/env python  
#-*- coding: utf-8 -*-  
  
import sys  
import os  
import time  
from sklearn import preprocessing
import numpy as np  
import pickle 
import data
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import tree  
from sklearn.svm import SVC 

  
reload(sys)  
sys.setdefaultencoding('utf8')  
  
class TrainModel(object):
    def train(self, model_name, train_x, train_y):
        model = self.classifiers[model_name](self, train_x, train_y)
        return model
    
    # Multinomial Naive Bayes Classifier  
    def naive_bayes_classifier(self, train_x, train_y):  
        model = MultinomialNB(alpha=0.01)  
        #train_x = data.scale01(train_x)
        model.fit(train_x, train_y)  
        return model  
      
      
    # KNN Classifier  
    def knn_classifier(self, train_x, train_y):  
        model = KNeighborsClassifier()  
        #train_x = data.standard(train_x)
        model.fit(train_x, train_y) 
        print(model)
        return model  
      
      
    # Logistic Regression Classifier  
    def logistic_regression_classifier(self, train_x, train_y):  
        model = LogisticRegression(penalty='l2')  
        #train_x = data.standard(train_x)
        model.fit(train_x, train_y)  
        return model  
      
      
    # Random Forest Classifier  
    def random_forest_classifier(self, train_x, train_y):  
        model = RandomForestClassifier(n_estimators=8)  
        #train_x = data.standard(train_x)
        model.fit(train_x, train_y)  
        return model  
      
      
    # Decision Tree Classifier  
    def decision_tree_classifier(self, train_x, train_y):  
        model = tree.DecisionTreeClassifier()  
        #train_x = data.standard(train_x)
        model.fit(train_x, train_y)  
        return model  
      
      
    # GBDT(Gradient Boosting Decision Tree) Classifier  
    def gradient_boosting_classifier(self, train_x, train_y):   
        model = GradientBoostingClassifier(n_estimators=200) 
        #train_x = data.standard(train_x)
        model.fit(train_x, train_y)  
        return model  
      
      
    # SVM Classifier  
    def svm_classifier(self, train_x, train_y):  
        model = SVC(kernel='rbf', probability=True)  
        #train_x = data.standard(train_x)
        model.fit(train_x, train_y)  
        return model  
      
    # SVM Classifier using cross validation  
    def svm_cross_validation(self, train_x, train_y):   
        model = SVC(kernel='rbf', probability=True)  
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  
        grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)  
        grid_search.fit(train_x, train_y)  
        best_parameters = grid_search.best_estimator_.get_params()  
        for para, val in best_parameters.items():  
            print para, val  
        model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
        #train_x = data.standard(train_x) 
        model.fit(train_x, train_y)  
        return model  

    classifiers = {'NB':naive_bayes_classifier,   
                  'KNN':knn_classifier,  
                   'LR':logistic_regression_classifier,  
                   'RF':random_forest_classifier,  
                   'DT':decision_tree_classifier,  
                  'SVM':svm_classifier,  
                'SVMCV':svm_cross_validation,  
                 'GBDT':gradient_boosting_classifier  
    }  

class ModelHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self, x, y):
        return self.clf.fit(x, y)
    
    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)

def optimizeModel(clf, name, tuned_parameters, X_train, y_train):
    #scores = ['recall', 'precision']
    scores = ['precision']
    model = GridSearchCV(clf, tuned_parameters, cv=5, scoring='precision_macro')
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        model = GridSearchCV(clf, tuned_parameters, cv=5, scoring='%s_macro' % score)
        model.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(model.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = model.cv_results_['mean_test_score']
        stds = model.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        #y_true, y_pred = y_test, clf.predict(X_test)
        #print(classification_report(y_true, y_pred))
        #print()
        print model
    saveModel(model, name)
    return model
    
def saveModel(clf, name):
    if not os.path.exists("models"):
        os.makedirs("models")
    model_save = open('models/' + name + ".pkl", 'wb')
    pickle.dump(clf, model_save)
    model_save.close()

def loadModel(name):
    model_reader = open('models/' + name + ".pkl", 'r')
    model = pickle.load(model_reader)
    model_reader.close()
    print name, model.get_params()
    return model
      
if __name__ == '__main__':  
    train_file = "train.txt"
    test_file = "test.txt"
    
    
    thresh = 0.5  
    model_save_file = "model.pickle"  
    model_save = {}  
      
    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']  
    #test_classifiers = ['KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT'] 

      
    print 'reading training and testing data...'
    train_x, train_y = data.read_data(train_file)
    test_x, test_y = data.read_data(test_file)
    num_train, num_feat = train_x.shape  
    num_test, num_feat = test_x.shape  
    is_binary_class = (len(np.unique(train_y)) == 2) 
    print is_binary_class
    
    print '******************** Data Info *********************'  
    print '#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat)  
    #sys.exit(0)
    train_model = TrainModel()
    for classifier in test_classifiers:  
        print '******************* %s ********************' % classifier  
        start_time = time.time()  
        #model = train_model.train(classifiers[classifier], train_x, train_y) 
        model = train_model.train(classifier, train_x, train_y)
        print 'training took %fs!' % (time.time() - start_time)  
        predict = model.predict(test_x)  
        F1Score = 0
        if model_save_file != None:  
            model_save[classifier] = model  
        if is_binary_class:  
            precision = metrics.precision_score(test_y, predict)  
            recall = metrics.recall_score(test_y, predict)  
            F1Score = 2 * precision * recall / (precision + recall)
            print 'precision: %.2f%%, recall: %.2f%%, F1Score: %.2f%%' % (100 * precision, 100 * recall, 100 * F1Score)  
        accuracy = metrics.accuracy_score(test_y, predict)  
        print 'accuracy: %.2f%%' % (100 * accuracy)  
  
    if model_save_file != None:  
        pickle.dump(model_save, open(model_save_file, 'wb')) 