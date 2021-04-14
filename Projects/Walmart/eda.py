# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Walmart Sales Prediction Model                                    #
# File    : \eda.py                                                           #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Course  : Practical Statistical Learning (Spring '21)                       #
# Email   : jtjames2@illinois.edu                                             #
# URL     : https://github.com/john-james-sf/Practical-Statistical-Learning   #
# --------------------------------------------------------------------------- #
# Created       : Tuesday, April 13th 2021, 11:12:41 am                       #
# Last Modified : Wednesday, April 14th 2021, 6:30:30 am                      #
# Modified By   : John James (jtjames2@illinois.edu)                          #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
import os
from collections import OrderedDict
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
import dateutil.parser as dparser
sns.set(style="whitegrid")
sns.set_palette("dark")
# --------------------------------------------------------------------------- #
def get_data():
    train_filename = "./data/train_ini.csv"
    all_filename = "./data/train.csv"
    train = pd.read_csv(train_filename)
    full = pd.read_csv(all_filename)
    return train, full
# --------------------------------------------------------------------------- #
def preprocess(df):
    # Add Year, Month and Week to dataframe
    df["DateTime"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Year"] = df["DateTime"].dt.isocalendar().year
    df["Month"] = df["DateTime"].dt.month
    df["Month_Name"] = df["Month"].apply(lambda x: calendar.month_abbr[x])
    df["Year_Month"] = df["Month_Name"] + "-" + df["Year"].astype(str)
    df["Week"] = df["DateTime"].dt.isocalendar().week
    # Add IsHolidayMonth to DataFrame
    df["IsHolidayMonth"] = False
    df.loc[df["Month"].isin([2,9,11,12]), "IsHolidayMonth"] = True

    # Change Department and Store as String for Plotting
    df["Dept"] = df["Dept"].apply(str)
    df["Store"] = df["Store"].apply(str)
    # Add Location as Department / Store combination
    df["Location"] = df["Dept"].astype(str) + "-" + df["Store"].astype(str)
    # Holidays
    df.loc[(df["IsHoliday"]== True) & (df["Date"]=="2010-02-12"), "Holiday"] = "Super Bowl 2010"
    df.loc[(df["IsHoliday"]== True) & (df["Date"]=="2011-02-11"), "Holiday"] = "Super Bowl 2011"
    df.loc[(df["IsHoliday"]== True) & (df["Date"]=="2010-09-10"), "Holiday"] = "Labor Day"
    df.loc[(df["IsHoliday"]== True) & (df["Date"]=="2010-11-26"), "Holiday"] = "Thanksgiving"
    df.loc[(df["IsHoliday"]== True) & (df["Date"]=="2010-12-31"), "Holiday"] = "Christmas"
    df["Holiday"].fillna("Non-Holiday Week", inplace=True)
    # Extract data from 2010-2 thru 2011-01 (to avoid double counting February)
    df = df[~((df["Year"] == 2011) & (df["Month"]==2))]
    return df
    
def get_departments(df):
    departments = df.groupby(by=["Dept","DateTime"]).mean() 
    departments.reset_index(inplace=True)
    departments.set_index('DateTime', inplace=True)
    departments.drop(columns=['Store'], inplace=True)    
    return departments

def get_stores(df):
    stores = df.groupby(by=["Store","DateTime"]).mean() 
    stores.reset_index(inplace=True)
    stores.set_index('DateTime', inplace=True)
    stores.drop(columns=['Dept'], inplace=True)    
    return stores

def show_sales(df, level=None, hue=None, start="2010-02", end="2011-02"):

    if level:
        title = f"Walmart Sales Analysis\nWeekly Average Sales by {level}\n{start} thru {end}"
    else:
        title = f"Walmart Sales Analysis\nWeekly Average Sales {start} thru {end}"

    fig, ax = plt.subplots(figsize=(8,8))    
    sns.lineplot(x="DateTime", y="Weekly_Sales",hue=hue, data=df, ax=ax)
    ax.set(xlabel="Date", ylabel="Weekly Sales", title=title)
    ax.title.set_size(12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()   

def show_forecasts(results, hue=None, title=None):
    title = f"Walmart Sales Analysis\nActual vs Forecast Sales\n2011-03 thru 2012-10"
    snaive = results[results["Model"].isin(["ACTUAL", "SNAIVE"])]
    stlf = results[results["Model"].isin(["ACTUAL", "STLF"])]
    tslm = results[results["Model"].isin(["ACTUAL", "TSLM"])]
    tslm_svd = results[results["Model"].isin(["ACTUAL", "TSLM.SVD"])]

    
    fig, axes = plt.subplots(nrows=2, ncols=2,  figsize=(12,12))
    
    sns.lineplot(x="DateTime", y="Weekly_Sales", data=snaive, hue=hue, ax=axes[0,0])
    axes[0,0].set(xlabel="Date", ylabel="Weekly Sales", title="Seasonal Naive")

    sns.lineplot(x="DateTime", y="Weekly_Sales", data=stlf, hue=hue, ax=axes[0,1])
    axes[0,1].set(xlabel="Date", ylabel="Weekly Sales", title="Seasonal & Trend Decomposition w/ Loess")    

    sns.lineplot(x="DateTime", y="Weekly_Sales", data=tslm, hue=hue, ax=axes[1,0])
    axes[1,0].set(xlabel="Date", ylabel="Weekly Sales", title="Linear Time Series")        

    sns.lineplot(x="DateTime", y="Weekly_Sales", data=tslm_svd, hue=hue, ax=axes[1,1])
    axes[1,1].set(xlabel="Date", ylabel="Weekly Sales", title="Time Series w/ Rank Reduction")        

    plt.suptitle(f"Walmart Sales Analysis\nActual vs Forecast\nMarch 2011 thru October 2012", fontsize=12)    
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()       


def show_decomposition(d, hue=None, level=None):
    legend = True if hue else False    
    ## Figure
    fig, axes = plt.subplots(nrows=4,ncols=1,figsize=(12,12), sharex=True)

    sns.lineplot(x="DateTime", y="Weekly_Sales", data=d["Weekly_Sales"], ax=axes[0], hue=hue, legend=legend, color="steelblue")
    sns.lineplot(x="DateTime", y="trend", data=d["trend"], ax=axes[1],legend=legend, color="steelblue")
    sns.lineplot(x="DateTime", y="season", data=d["season"], ax=axes[2], legend=legend, color="steelblue")
    sns.lineplot(x="DateTime", y="resid", data=d["resid"], ax=axes[3], legend=legend,
    color="steelblue")

    # Axis Labels
    axes[0].set(xlabel=None)
    axes[1].set(xlabel=None)
    axes[2].set(xlabel=None)
    axes[3].set(xlabel=None)

    axes[0].set_title("Observed")
    axes[1].set_title("Trend")
    axes[2].set_title("Seasonal")
    axes[3].set_title("Residual")
    if level:
        title = f"Walmart Sales Analysis\nWeekly Sales Decomposition by {level}\n2010-02 thru 2011-02"
    else:
        title = f"Walmart Sales Analysis\nWeekly Sales Decomposition\n2010-02 thru 2011-02"

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

def decompose(df, period=52):
    d = pd.DataFrame()
    stl = STL(df["Weekly_Sales"], period=period)
    tsd = stl.fit()    
    
    resid = tsd.resid.to_frame().reset_index()    
    resid["DateTime"] = df["DateTime"]
    
    trend = tsd.trend.to_frame().reset_index()    
    trend["DateTime"] = df["DateTime"]
    
    season = tsd.seasonal.to_frame().reset_index()    
    season["DateTime"] = df["DateTime"]
    d = df    
    d.loc[:,"season"] = season["season"]
    d.loc[:,"trend"] = trend["trend"]
    d.loc[:,"resid"] = resid["resid"]
    return d

def get_seasonality(d):
    """Measures seasonality by department"""
    
    d["strength"] = max(0,1-(np.var(d["resid"]) / np.var(d["season"]+d["resid"]) ))    
    return d

def analyze_sales(df, level="Dept", start=11.0, end=12.0):
    df = df[df["Month"].isin([start, end])]
    g = df.groupby(by=level).agg({"Weekly_Sales": ['mean','std']})
    g.columns = g.columns.droplevel()
    print(g.head())
    top10_mean = g.nlargest(n=10, columns=['mean'])
    top10_std = g.nlargest(n=10, columns=["std"])
    top10_mean_df = df[df["Dept"].isin(top10_mean.index)]
    top10_std_df = df[df["Dept"].isin(top10_std.index)]
    show_sales(top10_mean_df, level="Department", hue="Dept", start="2010-11", end="2010-12")
    show_sales(top10_std_df, level="Department", hue="Dept", start="2010-11", end="2010-12")

def get_pred(model):
    directory = "./results/" + model
    files = os.listdir(directory)
    pred = pd.DataFrame()
    for f in files:
        filename = directory + "/" + f
        df = pd.read_csv(filename)
        pred = pd.concat([pred, df], axis=0)    
    pred["DateTime"] = pd.to_datetime(pred["Date"], format="%Y-%m-%d")
    pred["Model"] = model
    return pred
        

def forecasts(full):
    models = ["snaive", "snaive_hs", "stlf_hs_ets", "stlf.svd_hs_ets_12_pc", "tslm_hs", "tslm.svd_hs_12_pc"]
    pred = pd.DataFrame()
    for model in models:
        p = get_pred(model)
        pred = pd.concat([pred,p], axis=0)
    actual_2011 = full[(full["Year"]==2011) & (full["Month"]>2)]
    actual_2012 = full.loc[full["Year"]==2012]
    actual = pd.concat([actual_2011, actual_2012], axis=0)
    actual["Model"] = "ACTUAL"
    pred["Weekly_Sales"] = pred["Weekly_Pred"]
    results = pd.concat([actual, pred], axis=0)
    results.sort_values(by=["Model","Date"], inplace=True)
    print(results.head())
    pred.sort_values(by=["Model","Date"], inplace=True)
    actual.sort_values(by="Date", inplace=True)    
    show_forecasts(results,hue='Model')    



def analyze(df):
    # Analyze department seasonality
    analyze_sales(df, level="Dept", start=11, end=12)
    
def main():
    df, full = get_data()
    df = preprocess(df) 
    d = decompose(df)
    show_decomposition(d)
    #full = preprocess(full)   
    #forecasts(full)    

if __name__ == "__main__":
    main()

#%%


