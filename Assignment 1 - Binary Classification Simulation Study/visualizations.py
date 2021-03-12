# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Practical Statistical Learning                                    #
# File    : \Visualizations.py                                                #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/MCS/                             #
# --------------------------------------------------------------------------- #
# Created       : Wednesday, February 3rd 2021, 7:21:22 pm                    #
# Last Modified : Wednesday, February 3rd 2021, 7:23:17 pm                    #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
from abc import ABC, abstractmethod
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from IPython.display import display, HTML, Markdown
# --------------------------------------------------------------------------- #
class Visuals(ABC):
    """Base class for plots."""    
    def __init__(self, params, palette="colorblind", n_colors=4, style="whitegrid", random_state=6998):
        self.params = params
        self.palette = palette
        self.n_colors = n_colors
        self.style = style
        self.data = None
        self.performance = None
        self.random_state = random_state

    def df_to_markdown(self, df, heading, space_before=1, space_after=0, underline=True):
        newline = "\n"
        uline = "_"
        linewidth  = 80
        heading == "**" + heading + "**"
        if space_before > 0:
            print(newline*space_before)
        display(Markdown(heading))
        if underline:
            print(uline*linewidth)
        if space_after > 0:
            print(newline*space_after)
        display(HTML(df.to_html()))


    def _get_score_stats(self, data, metric, confidence=0.95):                
        stats = data.groupby(by="Model")[metric].describe(percentiles=[0.5])
        stats["Lower 95% CI"] = stats['mean'] - 1.96 * stats['std'] / np.sqrt(stats['count'])
        stats["Upper 95% CI"] = stats['mean'] + 1.96 * stats['std'] / np.sqrt(stats['count'])        
        return stats        

    def _get_stats(self, data, var, confidence=0.95):
        stats = data[var].describe(percentiles=[0.5])
        stats["Lower 95% CI"] = stats['mean'] - 1.96 * stats['std'] / np.sqrt(stats['count'])
        stats["Upper 95% CI"] = stats['mean'] + 1.96 * stats['std'] / np.sqrt(stats['count'])        
        return stats                
    
    @abstractmethod
    def fit(self, results):
        pass

    @abstractmethod
    def plot(self, title):
        pass
# --------------------------------------------------------------------------- #
class DataVisualizer(Visuals):
    """Plots two gaussian distributions at random"""     
    def __init__(self, params, palette="colorblind", n_colors=2, style="whitegrid", random_state=6998):
        super(DataVisualizer, self).__init__(params, palette, n_colors, style, random_state)
        self.centers = None
        self.data = None
        self.gaussian_id = None
        self.gaussian_std = None

    def fit(self, centers, data):
        """Randomly selects two distributions then, extracts and formats data for plotting."""
        np.random.seed(self.random_state)       
        self.centers = centers
        self.data = data

        self.gaussian_id = centers['Set'].iloc[0]
        self.gaussian_std = np.round(centers['Set Centers Standard Deviation'].iloc[0],3)        
        
        # Randomly select 2 guassian distributions for plotting
        gds = np.array([0,0])
        while (gds[0] == gds[1]):
            gds = np.random.randint(0,self.params['n_simulations'],2)
        
        # Extract Data
        self.X0 = data[data["Simulation"] == gds[0]]        
        self.X1 = data[data["Simulation"] == gds[1]]
        
    def plot(self, title=None):
        """Plots centers and 2 distributions"""
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,5))        

        sns.set_palette(self.palette, n_colors=self.n_colors)
        sns.axes_style(self.style)        

        title = f"Randomly Selected Gaussian Distributions\nGaussian Mixture {self.gaussian_id} $\mathcal{{N}}(0,\sigma={{{self.gaussian_std}}})$"        

        # 1st Gaussian Selected
        sns.scatterplot(data=self.X0, x="x", y="y", hue="Class", alpha=0.5, ax=axs[0])
        sns.scatterplot(data=self.centers, x="x", y="y", hue="Class", 
                        marker="+", alpha=1.0, ax=axs[0], legend=False)        

        # 2nd Gaussian Selected
        sns.scatterplot(data=self.X1, x="x", y="y", hue="Class", alpha=0.5, ax=axs[1])
        sns.scatterplot(data=self.centers, x="x", y="y", hue="Class", 
                        marker="+", alpha=1.0, ax=axs[1], legend=False)        

        fig.suptitle(t=title, weight='bold')        
        fig.tight_layout()                                                           

# --------------------------------------------------------------------------- #
class ScoreVisualizer(Visuals):
    """Plots training and test errors for all models and simulations"""     
    def __init__(self, params, palette="colorblind", n_colors=4, style="whitegrid", random_state=6998):
        super(ScoreVisualizer, self).__init__(params, palette, n_colors, style, random_state)
        self.scores = None
        self.gaussian_id = None
        self.gaussian_std = None

    def fit(self, scores):
        """Formats data for plotting."""       
        
        self.gaussian_id = scores['Set'].iloc[0]
        self.gaussian_std = np.round(scores['Set Centers Standard Deviation'].iloc[0],3)

        # Reformat Error into long format and split into training and test
        self.scores = scores
        self.errors = scores[["Simulation", "Model", "Model Label", "Train Error", "Test Error" ]]
        self.errors.columns = ["Simulation", "Model", "Model Label","Training", "Test" ]
        self.errors = pd.melt(self.errors, id_vars=["Simulation", "Model", "Model Label"],  
                               var_name="Dataset",
                               value_vars=["Training","Test"],
                               value_name="Error")
        self.training_ERROR = self.errors[self.errors["Dataset"] == "Training"]
        self.test_ERROR = self.errors[self.errors["Dataset"] == "Test"]


        # Reformat AUC into long format and split into training and test
        self.auc = scores[["Simulation", "Model","Model Label", "Train AUC", "Test AUC" ]]
        self.auc.columns = ["Simulation", "Model", "Model Label", "Training", "Test" ]
        self.auc = pd.melt(self.auc, id_vars=["Simulation", "Model", "Model Label"],  
                               var_name="Dataset",
                               value_vars=["Training","Test"],
                               value_name="AUC")
        self.training_AUC = self.auc[self.auc["Dataset"] == "Training"]
        self.test_AUC = self.auc[self.auc["Dataset"] == "Test"]       

    def _plot_boxes(self, title=None):
        
        sns.set_palette(self.palette, self.n_colors)
        sns.set_style(self.style)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,5))   

        title = "Training and Test Errors"
        sns.boxplot(x='Model Label', y="Error", hue="Dataset", data=self.errors, ax=axs[0]).set_title(title)
        
        title = "Training and Test AUC"
        sns.boxplot(x='Model Label', y="AUC", hue="Dataset", data=self.auc, ax=axs[1]).set_title(title)        
        
        title = f"Performance Scores\nGaussian Mixture {self.gaussian_id} $\mathcal{{N}}(0,\sigma={{{self.gaussian_std}}})$"
        fig.suptitle(t=title, weight='bold')     
        fig.tight_layout()        

    def _plot_lines(self, metric="Error"):
        
        sns.set_palette(self.palette, self.n_colors)
        sns.set_style(self.style)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

        if (metric == "Error"):
            data = self.errors
            y = "Error"
        else:
            data = self.auc
            y = "AUC"

        n_train = self.params["n_train"]        
        subtitle = f"Training Set\nN={n_train*2} Observations"
        sns.lineplot(x="Simulation", y=y, hue="Model", data=data, ax=axs[0]).set_title(subtitle)

        n_test = self.params["n_test"]                
        subtitle = f"Test Set\nN={n_test*2} Observations"        
        sns.lineplot(x='Simulation', y=y, hue="Model", data=data, legend=False, ax=axs[1]).set_title(subtitle)        

        title = f"{metric} Scores\nGaussian Mixture {self.gaussian_id} $\mathcal{{N}}(0,\sigma={{{self.gaussian_std}}})$"
        fig.suptitle(t=title, weight='bold')        
        fig.tight_layout()        

    def _plot_stats(self, metric):
        """Renders descriptive statistics for training / test error and AUC."""        
        train_stats = self._get_score_stats(data=self.scores, metric="Train "+ metric)
        heading = "Training " + metric
        self.df_to_markdown(train_stats, heading)

        test_stats = self._get_score_stats(data=self.scores, metric="Test "+ metric)
        heading = "Test " + metric
        self.df_to_markdown(test_stats, heading)        
        
        
    def plot(self, kind="line"):
        if kind == "line":
            self._plot_lines(metric="Error")
            self._plot_stats(metric="Error")
            self._plot_lines(metric="AUC")
            self._plot_stats(metric="AUC")
        else:
            self._plot_boxes()                           

# --------------------------------------------------------------------------- #
class KNNVisualizer(Visuals):
    """Plots training and test errors for all models and simulations"""     
    def __init__(self, params, palette="colorblind", n_colors=4, style="whitegrid", random_state=6998):
        super(KNNVisualizer, self).__init__(params, palette, n_colors, style, random_state)
        self.stats = None

    def fit(self, k_values):
        """Formats data for plotting."""     
        self.k_values = k_values["Best k"]
        self.stats = k_values['Best k'].describe(percentiles=[0.5]).T
        self.stats['ste'] = k_values['Best k'].sem()   

    def _boxplot(self):
        """Renders boxplot of the distribution of k-values chosen by 10-fold cross validation."""

        title = f"k Nearest Neighbors\nDistribution of K values chosen by 10-fold CV"        
        
        sns.set_palette(self.palette, self.n_colors)
        sns.set_style(self.style)

        fig, axs = plt.subplots(figsize=(12,5))
        
        sns.boxplot(x=self.k_values, ax=axs).set_title(title, weight="bold")        
        fig.tight_layout()                

    def _histogram(self):
        """Renders histogram of k-values chosen by 10-fold cross validation."""

        title = f"k Nearest Neighbors\nK values chosen by 10-fold CV"        
        
        sns.set_palette(self.palette, self.n_colors)
        sns.set_style(self.style)

        fig, axs = plt.subplots(figsize=(12,5))        
        
        sns.histplot(x=self.k_values, ax=axs).set_title(title, weight="bold")        
        fig.tight_layout()          

    def _print_stats(self):
        """Renders a dataframe containing descriptive statistics for k-values."""
        df = pd.DataFrame(data=self.k_values)
        stats = self._get_stats(df, var="Best k")    
        heading = "Descriptive Statistics for K Values Chosen by 10-Fold CV"
        stats = stats.to_frame()
        self.df_to_markdown(stats, heading)

    def plot(self, title=None):
        self._histogram()
        self._boxplot()
        self._print_stats()


