# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Practical Statistical Learning                                    #
# File    : \models.py                                                        #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/MCS/                             #
# --------------------------------------------------------------------------- #
# Created       : Wednesday, February 3rd 2021, 10:09:06 pm                   #
# Last Modified : Wednesday, February 3rd 2021, 10:09:06 pm                   #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
from abc import ABC, abstractmethod
import numpy as np
import math
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import linear_model as lm
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import seaborn as sns
import pandas as pd
from data import KFold
# --------------------------------------------------------------------------- #
class Classifier(ABC):
    """Base class for classifiers."""

    def __init__(self, threshold=0.5, random_state=6998):
        self.threshold = threshold
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X, y):
        n = X.shape[0]
        self.scores = OrderedDict()
        # Compute error
        y_hat = self.predict(X)
        y_pred = [1 if i >= self.threshold else 0 for i in y_hat] 
        error =  sum(y_pred != y) / float(n) 
        
        # Compute AUC using sklearn 
        fpr, tpr, thresholds = metrics.roc_curve(y, y_hat, pos_label=1)
        auc = metrics.auc(fpr, tpr)        
        return error, auc


# --------------------------------------------------------------------------- #
class LinearRegression(Classifier):
    """ Trains and evaluates linear regression model"""
    def __init__(self, threshold=0.5, random_state=6998):
        super(LinearRegression, self).__init__(threshold, random_state)
        self.model = None
        self.id = "regression"
        self.name = "Linear Regression"
        self.label = "Linear\nRegression"

    def fit(self, X,y):
        self.model = lm.LinearRegression()
        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict(X)

# --------------------------------------------------------------------------- #
class LogisticRegression(Classifier):
    """ Trains and evaluates logistic regression model"""
    def __init__(self, threshold=0.5, random_state=6998):
        super(LogisticRegression, self).__init__(threshold, random_state)
        self.model = None
        self.id = "logistic"
        self.name = "Logistic Regression"
        self.label = "Logistic\nRegression"

    def fit(self, X,y):
        self.model = lm.LogisticRegression()
        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict(X)
# --------------------------------------------------------------------------- #        
class QuadraticRegression(Classifier):
    """ Trains and evaluates quadratic regression model"""
    def __init__(self, degree=2, threshold=0.5, random_state=6998):
        super(QuadraticRegression, self).__init__(threshold, random_state)
        self.model = None
        self.degree = degree
        self.transformer = None
        self.id = "quadratic"
        self.name = "Quadratic Regression"   
        self.label = "Quadratic\nRegression"   
        

    def fit(self, X,y):
        self.transformer = PolynomialFeatures(degree=self.degree)
        X_quad = self.transformer.fit_transform(X)
        self.model = lm.LinearRegression()
        self.model.fit(X_quad,y)        

    def predict(self, X):
        X_quad = self.transformer.fit_transform(X)
        return self.model.predict(X_quad)

# --------------------------------------------------------------------------- #        
class NaiveBayes(Classifier):
    """ Trains and evaluates Naive Bayes model"""
    def __init__(self, std_X, m0, m1, threshold=0.5, random_state=6998):    
        super(NaiveBayes, self).__init__(threshold, random_state)        
        self.std_X = std_X
        self.m0= m0
        self.m1 = m1
        self.p0 = None
        self.p1 = None
        self.normalizer = np.square(1/np.sqrt(2*np.pi*np.square(self.std_X)))
        self.variance2 = 2 * np.square(std_X)
        self.id = "bayes"
        self.name = "Naive Bayes'"        
        self.label = "Naive\nBayes'"        

    def _compute_densities(self, x):

        p0 = self.normalizer * np.mean(np.exp(-(np.square(x-self.m0)/self.variance2)))
        p1 = self.normalizer * np.mean(np.exp(-(np.square(x-self.m1)/self.variance2)))
            
        return p1>p0

    def fit(self, X,y=None):
        pass
        

    def predict(self, X,y=None):
        return np.apply_along_axis(self._compute_densities,1,X)

# --------------------------------------------------------------------------- #        
class kNNCV(Classifier):
    """ Trains and evaluates kNN via cross-validation and returns the k_values"""
    def __init__(self, n_folds, threshold=0.5, random_state=6998):      
        super(kNNCV, self).__init__(threshold, random_state)          
        self.max_k = 0
        self.n_folds = n_folds
        self.id = "knn"
        self.name = "kNN CV"
        self.label = "kNN CV"

    def _get_best_k(self, scores):
        """Returns the largest value of k within one standard erorr of the lowest error."""
        K = len(scores)
        sd = np.std(scores)
        se = sd / np.sqrt(K)
        min_score = np.min(scores)
        max_score = min_score + se
        candidates = np.where(scores<=max_score)
        return np.max(candidates)+1


    def fit(self, X,y):
        """ Partitions the data into k-folds."""                
        # Split data into n_folds
        self.kFold = KFold(n_folds=self.n_folds, random_state=self.random_state)
        self.kFold.generate_data(X,y)        

        # Determine k_max, the maximum value of k given N and the number of folds.
        N = X.shape[0]
        fold_size = math.floor(N/self.n_folds)
        self.k_max = (fold_size * (self.n_folds-1))-1

        # Iterate over each value of k, storing mean error for each fold
        k_scores = []
        for i in range(1,self.k_max+1):
            model = knn(i)            
            fold_scores = []
            for j in range(self.n_folds):
                data = self.kFold.get_fold_data(j)
                model.fit(data['X_train'], data['y_train'])
                fold_scores.append(model.score(data['X_val'], data['y_val']))
            k_scores.append(np.mean(fold_scores))

        # Obtain best k
        self.best_k = self._get_best_k(k_scores)
        self.model = knn(self.best_k)
        self.model.fit(X, y)


    def predict(self, X):
        """Predicts using best found k_value."""
        return self.model.predict(X)        
