# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Practical Statistical Learning                                    #
# File    : \Regression kNN.py                                                #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/MCS/                             #
# --------------------------------------------------------------------------- #
# Created       : Wednesday, February 3rd 2021, 6:09:52 pm                    #
# Last Modified : Wednesday, February 3rd 2021, 6:16:26 pm                    #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
import numpy as np
import pandas as pd
from collections import OrderedDict
from IPython.display import HTML
# --------------------------------------------------------------------------- #
class DataGen:
    """Generates a training and test samples

    Generates a training set of 200 observations and a test set of
    10,000 observations from a mixture of 10 Guassian distributions.
    """
    def __init__(self, n_gaussians=10, std_centers=1, p=2, random_state=6998):
        self.n_gaussians = n_gaussians
        self.std_centers = std_centers 
        self.random_state = random_state
        self.std_X = np.sqrt(np.square(self.std_centers)/5)
        self.p = p             # number of dimensions
        self.m0s = None
        self.m1s = None        
        self.m0 = None
        self.m1 = None
        self.X = None
        self.y = None
        self.iterations = 0

    def generate_centers(self):
        """Randomly generates Gaussian distribution centers"""
        np.random.seed(self.random_state)        

        self.m0s = np.random.normal(size = (self.n_gaussians, self.p)) * \
            self.std_centers + np.concatenate([np.array([[0, 1]] * self.n_gaussians)])

        self.m1s = np.random.normal(size = (self.n_gaussians, self.p)) * \
            self.std_centers + np.concatenate([np.array([[1, 0]] * self.n_gaussians)])       

    def generate_data(self, N):
        """Generates N observations of Xy data."""
        data = {}

        np.random.seed(self.random_state+self.iterations)
        self.iterations += 1

        # Random variable that randomly assigns a center to each data point. 
        id1 = np.random.randint(self.n_gaussians, size = N)
        id0 = np.random.randint(self.n_gaussians, size = N)        

        # Center assignment
        self.m0 = self.m0s[id0,:]
        self.m1 = self.m1s[id1,:]           

        # X is generated some random distance, scaled by variance, from a randomly selected center.
        self.X = np.random.normal(size = (2 * N, self.p)) * self.std_X + \
                np.concatenate([self.m1, self.m0])
        self.y = np.concatenate(([1]*N, [0]*N))     

        data = {'X': self.X, 'y':self.y}
        return data

    def summary(self):
        unique, counts = np.unique(self.y, return_counts=True)
        d = {"N": self.X.shape[0], "p": self.X.shape[1], "Class 0": counts[0], "Class 1":counts[1]}
        df = pd.DataFrame(data=d, index=[0])
        print("\nSummary of Counts for Generated Data")
        print(df)
        print("\nm0s")
        print(self.m0s)
        df_m00 = pd.DataFrame(data=self.m0s[:,0])        
        df_m01 = pd.DataFrame(data=self.m0s[:,1])        
        print(df_m00.describe().T)
        print(df_m01.describe().T)
        print("\nm1s")
        print(self.m1s)
        df_m10 = pd.DataFrame(data=self.m1s[:,0])        
        df_m11 = pd.DataFrame(data=self.m1s[:,1])        
        print(df_m10.describe().T)        
        print(df_m11.describe().T)        
        
        df_X = pd.DataFrame(data=self.X)
        df_y = pd.DataFrame(data=self.y)
        print("\nDescriptive Statistics for X")
        print(df_X.describe().T)
        print("\nDescriptive Statistics for y")
        print(df_y.describe().T)
        

# --------------------------------------------------------------------------- #
class KFold:
    """Partitions data into k-folds and for a given fold, returns train/val data."""

    def __init__(self, n_folds, shuffle=True, stratified=True, random_state=None):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratified = stratified
        self.X_folds = None
        self.y_folds = None
        self.folds = []
        # Containers for descriptive statistics by fold
        self.X_train_stats = []
        self.y_train_stats = []
        self.X_val_stats = []
        self.y_val_stats = []                

    def _shuffle(self, a):
        '''Randomly shuffles a vector.'''
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(a)
        return a

    def _split(self, X, y):        
        '''Splits data into n_folds without stratification.'''
        self.X_folds = np.split(X, self.n_folds)
        self.y_folds = np.split(y, self.n_folds)

    def _split_stratified(self, X, y):
        '''Splits data into n_folds with stratification.'''
        # number of folds doesn't exceed the number of observations.
        classes, counts = np.unique(y, return_counts=True)        
        if np.all(self.n_folds > counts):
            raise ValueError(f"n_folds = {self.n_folds} must be less than the "
                             f" {counts}, the number of observations in each class.")
                
        # Get the indices for each class
        idx_0 = [key for key, val in enumerate(y) if val == 0]
        idx_1 = [key for key, val in enumerate(y) if val == 1]

        # If requested, shuffle indices within class
        if (self.shuffle):
            idx_0 = self._shuffle(idx_0)
            idx_1 = self._shuffle(idx_1)

        # Separate the data into classes
        X0 = X[idx_0]
        y0 = y[idx_0]
        X1 = X[idx_1]
        y1 = y[idx_1]

        # Split into folds by class
        X0_folds = np.array(np.split(X0,self.n_folds))
        y0_folds = np.array(np.split(y0,self.n_folds))
        X1_folds = np.array(np.split(X1,self.n_folds))
        y1_folds = np.array(np.split(y1,self.n_folds))

        ## Concatenate classes to create X and y folds  
        self.X_folds = np.concatenate((X0_folds, X1_folds), axis=1)
        self.y_folds = np.concatenate((y0_folds, y1_folds), axis=1)

    def _create_data_stats(self, fold, dataset, variable, data):
        return {"Fold": fold, "Dataset": dataset, "Variable": variable,
                "Mean": np.mean(data),
                "Std": np.std(data),
                "Min": np.min(data),
                "25%": np.quantile(data,0.25),
                "50%": np.quantile(data,0.5),
                "75%": np.quantile(data,0.75),
                "Max": np.max(data)}

    def _create_fold_stats(self, fold, X_train, y_train, X_val, y_val):
        '''Appends descriptive statistics for fold 'fold' to container list of stats.'''

        # X_train
        stats = self._create_data_stats(fold, "X_train", "X0", X_train[0])
        self.X_train_stats.append(stats)
        stats = self._create_data_stats(fold, "X_train", "X1", X_train[1])
        self.X_train_stats.append(stats)
        # y_train
        stats = self._create_data_stats(fold, "y_train", "target", y_train[0])
        self.y_train_stats.append(stats)
        
        # X_val
        stats = self._create_data_stats(fold, "X_val", "X0", X_val[0])
        self.X_val_stats.append(stats)
        stats = self._create_data_stats(fold, "X_val", "X1", X_val[1])
        self.X_val_stats.append(stats)
        # y_val
        stats = self._create_data_stats(fold, "y_val", "target", y_val[0])
        self.y_val_stats.append(stats)        

    def _create_fold(self, fold):
        '''Creates the kth fold containing a training and validation set.'''        

        # Extract all but the kth fold into a training set.
        X_train_folds = np.delete(self.X_folds,fold,axis=0)
        y_train_folds = np.delete(self.y_folds,fold,axis=0)

        X_train = np.concatenate(X_train_folds)
        y_train = np.concatenate(y_train_folds)     
        
        # Extract the kth fold for the validation set
        X_val = self.X_folds[fold]
        y_val = self.y_folds[fold]

        # Create and store descriptive statistics
        self._create_fold_stats(fold, X_train, y_train, X_val, y_val)
        
        fold_data = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}

        return fold_data

    def _split(self, X, y):
        '''Splits data into n X_folds and y_folds.'''
        if self.stratified:
            self._split_stratified(X,y)
        else:
            self._split(X,y)

    def _combine(self):
        '''Creates n_folds containing a training and validation set.'''
        self.folds = []
        for i in range(self.n_folds):
            self.folds.append(self._create_fold(i))

    def generate_data(self, X, y):
        self._split(X,y)
        self._combine()
        return self.folds

    def get_fold_data(self, k):
        return self.folds[k]

    def summarize(self):
        '''Prints descriptive statistics by dataset and fold.'''

        print("\n\nDescriptive Statistics for k-Fold Cross-Validation Data")
        print("=========================================================")
        
        ## Render X_train stats
        print(f"\nX_train descriptive statistics")
        df = pd.DataFrame(data=self.X_train_stats)
        HTML(df.to_html(index=False))
        ## Render y_train stats
        print(f"\ny_train descriptive statistics")
        df = pd.DataFrame(data=self.y_train_stats)
        HTML(df.to_html(index=False))        
        
        ## Render X_val stats
        print(f"\nX_val descriptive statistics")
        df = pd.DataFrame(data=self.X_val_stats)
        HTML(df.to_html(index=False))
        ## Render y_val stats
        print(f"\ny_val descriptive statistics")
        df = pd.DataFrame(data=self.y_val_stats)
        HTML(df.to_html(index=False))        
