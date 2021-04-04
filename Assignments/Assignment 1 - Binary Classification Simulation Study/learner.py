# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Practical Statistical Learning                                    #
# File    : \learner.py                                                       #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/practical-statistical-learning/  #
# --------------------------------------------------------------------------- #
# Created       : Sunday, February 14th 2021, 7:54:00 am                      #
# Last Modified : Sunday, February 14th 2021, 7:55:24 am                      #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
import numpy as np
import pandas as pd
from collections import OrderedDict
from IPython.display import HTML

from data import DataGen
from models import LinearRegression, QuadraticRegression, NaiveBayes, kNNCV
from models import LogisticRegression
from visualizations import DataVisualizer, ScoreVisualizer, KNNVisualizer
# --------------------------------------------------------------------------- #
class Simulation:
    """Generates data and evaluates algorithms for a single simulation."""
    def __init__(self, data_gen, n_gaussians=10, n_train=100, n_test = 5000, n_folds=10,
                 p=2, random_state=6998):
        self.data_gen = data_gen
        self.std_centers = data_gen.std_centers        
        self.std_X = np.sqrt(np.square(self.std_centers)/5)  
        self.n_gaussians = n_gaussians
        self.n_train = n_train
        self.n_test = n_test
        self.n_folds = n_folds
        self.p = p
        self.random_state = random_state    

        self.train_data = {}
        self.test_data = {}
        self.scores = pd.DataFrame()
        self.best_k = pd.DataFrame()
    
    def run(self, models):        
        self.train_data = self.data_gen.generate_data(self.n_train)
        self.test_data =  self.data_gen.generate_data(self.n_test)        
        for model in models.values():
            model.fit(self.train_data['X'], self.train_data['y'])
            train_error, train_auc = model.score(self.train_data['X'], self.train_data['y'])
            test_error, test_auc = model.score(self.test_data['X'], self.test_data['y'])
            d = {"Model Id": model.id, "Model": model.name, "Model Label": model.label, 
                 "Train Error": train_error,"Train AUC": train_auc,
                 "Test Error": test_error, "Test AUC": test_auc}
            df = pd.DataFrame(data=d, index=[0])
            self.scores = pd.concat((self.scores,df),axis=0)            
            if model.id == "knn":
                d = {"Best k": model.best_k}
                self.best_k = pd.DataFrame(data=d, index=[0])

# --------------------------------------------------------------------------- #
class Learner:
    """Learning algorithm: Primary driver for the simulation study."""
    def __init__(self, std_centers_list=[], n_gaussians=10, n_simulations=20, n_folds=10, 
                 n_train=100, n_test = 5000, p=2, 
                 random_state=6998):
        self.std_centers_list = std_centers_list
        self.n_gaussians = n_gaussians
        self.n_simulations = n_simulations
        self.n_folds = n_folds
        self.n_train = n_train # per class
        self.n_test = n_test # per class
        self.p = p        
        self.random_state = random_state
        # Bundle parameters for visualization
        self.params = {"std_centers_list": std_centers_list, 
                       "n_gaussians": n_gaussians, "n_simulations": n_simulations, 
                       "n_folds": n_folds, "n_train": n_train, "n_test": n_test, 
                       "p": p, "random_state": random_state}
        # Output data
        self.centers = pd.DataFrame()
        self.data = pd.DataFrame()
        self.scores = pd.DataFrame()
        self.k_values = pd.DataFrame()

    def _initialize_models(self, data_generator):
        """Initializes models prior to training."""
        models = {"Linear Regression": LinearRegression(),
                "Logistic Regression": LogisticRegression(),
                "Quadratic Regression": QuadraticRegression(),
                "Naive Bayes'": NaiveBayes(std_X=data_generator.std_X, m0=data_generator.m0s, m1=data_generator.m1s),
                "kNN CV": kNNCV(n_folds=self.n_folds)}                   
        return models


    def _save_centers(self, trial, trial_std, data_generator):
        """Stores centers for retrieval and plotting."""

        m0 = data_generator.m0s
        m1 = data_generator.m1s
        m = np.concatenate((m0,m1), axis=0)
        y = np.concatenate((np.repeat(0,self.n_gaussians), np.repeat(1,  self.n_gaussians)),axis=0).reshape(-1,1)
        m = np.concatenate((m,y),axis=1)
        df = pd.DataFrame(data=m)        
        df.columns = ["x", "y", "Class"]        
        df["Set"] = trial
        df["Set Centers Standard Deviation"] = trial_std
        self.centers = pd.concat((self.centers, df), axis=0)        

    def _save_data(self, trial, trial_std, simulation_id, simulation):
        """Stores data for retrieval and plotting"""
        df = pd.DataFrame()
        df["x"] = simulation.train_data["X"][:,0]
        df["y"] = simulation.train_data["X"][:,1]
        df["Class"] = simulation.train_data["y"]
        df["Set"] = trial
        df["Set Centers Standard Deviation"] = trial_std
        df['Simulation'] = simulation_id        
        assert(df.shape == (2*self.n_train,6))
        self.data = pd.concat((self.data,df), axis=0)        

    def _save_scores(self, trial, trial_std, simulation_id, simulation):
        """Stores scores for retrieval and plotting."""        
        df = simulation.scores
        df["Set"] = trial
        df["Set Centers Standard Deviation"] = trial_std
        df['Simulation'] = simulation_id        
        assert(df.shape == (5,10))
        self.scores = pd.concat((self.scores, df), axis=0)        

    def _save_k_values(self, trial, trial_std, simulation_id, simulation):
        """Stores scores for retrieval and plotting."""   
        df = simulation.best_k
        df["Set"] = trial
        df["Set Centers Standard Deviation"] = trial_std
        df['Simulation'] = simulation_id        
        assert(df.shape == (1,4))
        self.k_values = pd.concat((self.k_values , df), axis=0)                

    def run(self):
        """Runs the analysis via sequence of simulations."""        
        for i in range(len(self.std_centers_list)):

            # Compute centers
            dg = DataGen(std_centers=self.std_centers_list[i], n_gaussians=self.n_gaussians,
                        p=self.p, random_state=self.random_state)
            dg.generate_centers()

            # Save Centers for the set
            context = {"Set": i, "Set Center Standard Deviation": self.std_centers_list[i]}            
            self._save_centers(trial=i, trial_std=self.std_centers_list[i], data_generator=dg)

            models = self._initialize_models(dg)

            for j in range(self.n_simulations): 
                simulation = Simulation(data_gen=dg, n_gaussians=self.n_gaussians,
                                n_train=self.n_train, n_test=self.n_test, n_folds=self.n_folds,
                                p=self.p, random_state=self.random_state)
                simulation.run(models)

                d = {"Set": i, "Set Center Standard Deviation": self.std_centers_list[i],
                     "Simulation": j}
                self._save_data(trial=i, trial_std=self.std_centers_list[i], simulation_id=j, simulation=simulation)
                self._save_scores(trial=i, trial_std=self.std_centers_list[i], simulation_id=j, simulation=simulation)
                self._save_k_values(trial=i, trial_std=self.std_centers_list[i], simulation_id=j, simulation=simulation)
    

    def report_data(self):
        """ Renders data plots."""
        data_visualizer = DataVisualizer(params=self.params)
        for i in range(len(self.std_centers_list)):
            centers = self.centers[self.centers["Set"] == i]
            data = self.data[self.data["Set"] == i]            
            data_visualizer.fit(centers, data)
            data_visualizer.plot()            

    def report_scores(self):
        """Renders reports of scores."""
        score_visualizer = ScoreVisualizer(params=self.params)
        for i in range(len(self.std_centers_list)):
            # Plot Score line plots and boxplots
            scores = self.scores[self.scores["Set"] == i]
            score_visualizer.fit(scores)
            score_visualizer.plot("line")
            score_visualizer.plot("box")            


    def report_k_values(self):
        knn_visualizer = KNNVisualizer(params=self.params)
        for i in range(len(self.std_centers_list)):        
            # Plot boxplot of k values.
            k_values = self.k_values[self.k_values["Set"] == i]
            knn_visualizer.fit(k_values)
            knn_visualizer.plot()                  
              
    def report(self):
        """Renders visualizations of performance by set of Guassians."""
        data_visualizer = DataVisualizer(params=self.params)
        score_visualizer = ScoreVisualizer(params=self.params)
        knn_visualizer = KNNVisualizer(params=self.params)
        for i in range(len(self.std_centers_list)):
            # Plot centers and data
            centers = self.centers[self.centers["Set"] == i]
            data = self.data[self.data["Set"] == i]            
            data_visualizer.fit(centers, data)
            data_visualizer.plot()

            # Plot Score line plots and boxplots
            scores = self.scores[self.scores["Set"] == i]
            score_visualizer.fit(scores)
            score_visualizer.plot("line")
            score_visualizer.plot("box")

            # Plot boxplot of k values.
            k_values = self.k_values[self.k_values["Set"] == i]
            knn_visualizer.fit(k_values)
            knn_visualizer.plot()            
