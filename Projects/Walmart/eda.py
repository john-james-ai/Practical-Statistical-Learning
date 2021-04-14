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
# Last Modified : Tuesday, April 13th 2021, 8:17:20 pm                        #
# Modified By   : John James (jtjames2@illinois.edu)                          #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
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
    filename = "./data/train_ini.csv"
    train = pd.read_csv(filename)
    return train
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

    # Add Department and Store as String for Plotting
    df["Dept_s"] = df["Dept"].apply(str)
    df["Store_s"] = df["Store"].apply(str)
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

def show_sales(df, level=None, start="2010-02", end="2011-02"):

    if level:
        title = f"Walmart Sales Analysis\nWeekly Average Sales by {level}\{start} thru {end}"
    else:
        title = f"Walmart Sales Analysis\nWeekly Average Sales {start} thru {end}"

    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(x="DateTime", y="Weekly_Sales", data=df, ax=ax)
    ax.set(xlabel="Date", ylabel="Weekly Sales", title=title)
    ax.title.set_size(12)
    plt.tight_layout()
    plt.show()    

def analyze_sales(df, level="Dept"):
    g = df.groupby(by=level).mean()
    print(g.describe(percentiles=[.2,.5,.8]).T)
    print(g.nlargest(n=10, columns=["Weekly_Sales"]))



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

def analyze_seasonality(df, level="Dept"):
    df = df.groupby(by=[level,"DateTime"]).mean()
    df.reset_index(inplace=True)   
    d = decompose(df)     
    print(d.head())
    s = get_seasonality(d)
    print(s.head())
    # groups = np.sort(df[level].unique())
    # strength = []
    # season = []
    # for group in groups:
    #     d = decompose(df[df[level]==group])  
    #     print(d.head())  
        
    #     s = get_seasonality(d)
    #     strength.append(s["strength"])
    #     season.append(s["season"].mean())
    # d = {level: groups, "Seasonality": season, "Strength": strength}        
    return s

def analyze(df):
    # Analyze department seasonality
    s = analyze_seasonality(df, level="Dept")
    print(s.head())
    print(s.describe(percentiles=[.05, .2,  .5, .8, .95]).T)
    print(s.sort_values(by="season", ascending=False))
    top5dept = [77,39,43,99,78]
    top5dept = df[df["Dept"].isin(top5dept)] 
    d = decompose(top5dept)
    show_decomposition(d)
    
def main():
    df = get_data()
    df = preprocess(df)    
    analyze(df)

if __name__ == "__main__":
    main()

#%%    



