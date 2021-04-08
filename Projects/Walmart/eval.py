# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Walmart Sales Prediction Model                                    #
# File    : \eval.py                                                          #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/Practical-Statistical-Learning   #
#           /Projects/Walmart/                                                #
# --------------------------------------------------------------------------- #
# Created       : Saturday, April 3rd 2021, 5:41:24 pm                        #
# Last Modified : Saturday, April 3rd 2021, 5:41:24 pm                        #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
import numpy as np
import pandas as pd

from mymain import mypredict

train = pd.read_csv('train_ini.csv', parse_dates=['Date'])
test = pd.read_csv('test.csv', parse_dates=['Date'])

# save weighed mean absolute error WMAE
n_folds = 10
next_fold = None
wae = []

# time-series CV
for t in range(1, n_folds+1):
    print(f'Fold{t}...')

    # *** THIS IS YOUR PREDICTION FUNCTION ***
    train, test_pred = mypredict(train, test, next_fold, t)

    # Load fold file
    # You should add this to your training data in the next call to mypredict()
    fold_file = 'fold_{t}.csv'.format(t=t)
    next_fold = pd.read_csv(fold_file, parse_dates=['Date'])

    # extract predictions matching up to the current fold
    scoring_df = next_fold.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')

    # extract weights and convert to numpy arrays for wae calculation
    weights = scoring_df['IsHoliday_x'].apply(lambda is_holiday:5 if is_holiday else 1).to_numpy()
    actuals = scoring_df['Weekly_Sales'].to_numpy()
    preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()

    wae.append((np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())

print(wae)
print(sum(wae)/len(wae))