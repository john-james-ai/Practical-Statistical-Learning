# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Walmart Sales Prediction Model                                    #
# File    : \mymain.R                                                         #
# R       : 3.6.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Course  : Practical Statistical Learning (Spring '21)                       #
# Email   : jtjames2@illinois.edu                                             #
# URL     : https://github.com/john-james-sf/Practical-Statistical-Learning   #
# --------------------------------------------------------------------------- #
# Created       : Saturday, April 10th 2021, 7:46:00 am                       #
# Last Modified : Monday, April 12th 2021, 10:13:25 am                        #
# Modified By   : John James (jtjames2@illinois.edu)                          #
# =========================================================================== #
# Acknowledgment: Parts of this module borrowed liberally from:               #
#                                                                             #
#   1. Author: David Thaler                                                   #
#   2. Date: May 13, 2014                                                     #
#   3. Title: Walmart_competition_code                                        #
#   4. Link: https://github.com/davidthaler/Walmart_competition_code/         #
# =========================================================================== #
library(lubridate)
library(tidyverse)
library(forecast)
library(dplyr)
library(reshape)
library(plyr)
source("utils.R")
# =========================================================================== #
#                       TIME SERIES FORECASTING MODELS                        #
# =========================================================================== #
regression <- function(train, test) {
  # Computes forecast using linear regression
  #
  # Parameters:
  # ----------
  # train   : A data.table of Weekly_Sales values from the training set of 
  #           dimension (weeks in training data) x (number of stores)
  # test    : An all-zeros data.table of dimension:
  #           (number of weeks in training data) x (number of stores)
  #           The forecasts are written in place of the zeros.
  #
  # Returns:
  # --------
  # forecast : A data table containing the forecasts 
  # ------------------------------------------------------------------------- #

  coef <- lm.fit(as.matrix(train[, -(2:4)]), tmp_train$Weekly_Sales)$coefficients
  coef[is.na(mycoef)] <- 0
  pred <- coef[1] + as.matrix(test[, 4:55]) %*% mycoef[-1]

}



mypredict = function(){

  # Prepare data for regression or time series modeling.
  d <- prep_ts_data() 
  train.dt <- d$train
  test.dt <- d$test
  test.current <- d$current

  # Pre-allocate a list to store forecasts
  forecasts <- vector(mode = "list", length = length(train.dt))  

  # Process data by department 
  for (i in 1:length(train.dt)) {
    train.dept.dt <- as.data.table(train.dt[[i]])
    test.dept.dt <- as.data.table(test.dt[[i]])    
    
    # Fit the model and render forecast
    pred <- tslm.svd(train.dept.dt, test.dept.dt, model.type="ets", n.comp=12)
    
    # Shifts forecast forward to account for the floating
    # Christmas Holiday. 
    if ((t==5) & (adjust == TRUE)) {      
      pred <- shift(pred)
    }    

    # Reshape into original long form and convert Store back to integer. 
    pred <- melt(pred, id.vars=c("Date","Dept"), variable.name = "Store",
                  value.name = "Weekly_Pred", variable.factor=FALSE)
    pred$Store <- as.integer(pred$Store)    
          
    forecasts[[i]] <- pred
  }
  
  forecasts <- bind_rows(forecasts)
  setDT(forecasts)

  # Merge forecasts with test data to create submission  
  submission <- test.current %>% left_join(forecasts, by=c("Date","Dept","Store"))
    
  save_results(submission, mname,t)    
  
  return(submission)
}

tslm.baseline <- function(train, test){
  # Computes a forecast using time series linear regression 
  
  train <- as.matrix(train[,":="(Date=NULL,Dept=NULL)])
  train[is.na(train)] <- 0
  horizon <- nrow(test)
  
  for (j in 1:ncol(train)) {
    s <- ts(train[, j], frequency=52)
    fit <- tslm(s ~ trend + season)
    fc <- forecast(fit, h=horizon)
    test[, j+2] <- as.numeric(fc$mean)  
  }
  return(test)
}

tslm.svd <- function(train, test, model.type, n.comp){
  # Computes a forecast using time series linear regression 
  train <- preprocess.svd(train, n.comp) 
  train[is.na(train)] <- 0
  horizon <- nrow(test)
  
  for (j in 1:ncol(train)) {
    s <- ts(train[, j], frequency=52)
    fit <- tslm(s ~ trend + season)
    fc <- forecast(fit, h=horizon)
    test[, j+2] <- as.numeric(fc$mean)  
  }
  return(test)
}
seasonal.naive <- function(train, test){
  # Computes seasonal naive forecasts
  #
  # args:
  # train - A matrix of Weekly_Sales values from the training set of dimension
  #         (number of weeeks in training data) x (number of stores)
  # test - An all-zeros matrix of dimension:
  #       (number of weeeks in training data) x (number of stores)
  #       The forecasts are written in place of the zeros.
  #
  # returns:
  #  the test(forecast) data frame with the forecasts filled in 
  # 

  train <- as.matrix(train[,":="(Date=NULL,Dept=NULL)])
  train[is.na(train)] <- 0

  h <- nrow(test)
  tr <- train[nrow(train) - (52:1) + 1,]
  tr[is.na(tr)] <- 0
  test[,2:ncol(test)]  <- tr[1:h,2:ncol(test)]
  
  return(test)
}

shift <- function(test, threshold=1.1, shift=1.5){
  # This function executes a shift of the sales forecasts in the Christmas
  # period to reflect that the models are weekly, and that the day of the week
  # that Christmas occurs on shifts later into the week containing the holiday.
  #
  # NB: Train is actually not used here. Previously, there were other post-
  #     adjustments which did use it, and it is taken in here to preserve a 
  #     calling signature.
  #
  # args:
  # train - this is an n_weeks x n_stores data.table of values of Weekly_Sales
  #         for the training set within department, across all the stores
  # test - this is a (forecast horizon) x n_stores data.table of Weekly_Sales
  #        for the training set within department, across all the stores
  # threshold - the shift is executed if the mean of Weekly_Sales for weeks
  #          49-51 is greater than that for weeks 48 and 52 by at least
  #          a ratio of threshold
  # shift - The number of days to shift sales around Christmas.
  #         Should be 2 if the model is based on the last year only,
  #         or 2.5 if it uses both years
  #
  s <- ts(rep(0,39), frequency=52, start=c(2012,44))
  idx <- cycle(s) %in% 48:52
  holiday <- test[5:9,3:ncol(test)]
  baseline <- mean(rowMeans(holiday[c(1, 5),], na.rm=TRUE))
  surge <- mean(rowMeans(holiday[2:4,], na.rm=TRUE))
  holiday[is.na(holiday)] <- 0
  if(is.finite(surge/baseline) & surge/baseline > threshold){
      shifted.sales <- ((7-shift)/7) * holiday
      shifted.sales[2:5] <- shifted.sales[2:5] + (shift/7) * holiday[1:4]
      shifted.sales[1] <- holiday[1]
      test[5:9,3:ncol(test)] <- shifted.sales
  }
  test
}

stlf.svd <- function(train, test, model.type, n.comp){
  # Replaces the training data with a rank-reduced approximation of itself,
  # then forecasts each store using stlf() from the forecast package.
  # That function performs an STL decomposition on each series, seasonally
  # adjusts the data, non-seasonally forecasts the seasonally adjusted data,
  # and then adds in the naively extended seasonal component to get the
  # final forecast.
  #
  
  
  horizon <- nrow(test)
  train <- preprocess.svd(train, n.comp) 
  for(j in 1:ncol(train)){
    s <- ts(train[, j], frequency=52)
    if(model.type == 'ets'){
      fc <- stlf(s, 
                 h=horizon, 
                 s.window=3, 
                 method='ets',
                 ic='bic', 
                 opt.crit='mae')
    }else if(model.type == 'arima'){
      fc <- stlf(s, 
                 h=horizon, 
                 s.window=3, 
                 method='arima',
                 ic='bic')
    }else{
      stop('Model type must be one of ets or arima.')
    }
    pred <- as.numeric(fc$mean)
    test[, j+2] <- pred
  }
  test
}
preprocess.svd <- function(train, n.comp){
  # Replaces the training data with a rank-reduced approximation of itself.
  # This is for noise reduction. The intuition is that characteristics
  # that are common across stores (within the same department) are probably
  # signal, while those that are unique to one store may be noise.
  #
  # args:
  # train - A matrix of Weekly_Sales values from the training set of dimension
  #         (number of weeeks in training data) x (number of stores)
  # n.comp - the number of components to keep in the singular value
  #         decomposition
  #
  # returns:
  #  the rank-reduced approximation of the training data
  train <- as.matrix(train[,":="(Date=NULL,Dept=NULL)])
  train[is.na(train)] <- 0
  z <- svd(train[, 2:ncol(train)], nu=n.comp, nv=n.comp)
  s <- diag(z$d[1:n.comp])
  train[, 2:ncol(train)] <- z$u %*% s %*% t(z$v)
  train
}
prep_ts_data <- function() {
  # This function prepares the training data for time series forecasting.
  # 
  # Iterations 2:10 will feed next periods sales
  if (t > 1){
    train <<- train %>% add_row(new_train)
  }  

  # Extract current fold data from test
  start_date <- ymd("2011-03-01") %m+% months(2 * (t - 1))
  end_date <- ymd("2011-05-01") %m+% months(2 * (t - 1))
  test.current <- test %>% filter(Date >= start_date & Date < end_date) %>% 
    select(-IsHoliday) 

  # Find the unique pairs of department and stores that are appear in 
  # both training and test set.
  train.pairs <- train[, 1:2] %>% dplyr::count(Store, Dept) %>% filter(n != 0)
  test.pairs <- test.current[, 1:2] %>% dplyr::count(Store, Dept) %>% filter(n != 0)
  dept.store.xref <- intersect(train.pairs[, 1:2], test.pairs[, 1:2])    

  # Expand by store, dept, and date to create a cubic data set, filter
  # by unique stores and depts in the test set, add a numeric 'Week' variable,
  # convert to data.table, reshape to wide format, then group by dept.
  keys <- train %>% tidyr::expand(Dept, Store, Date)
  suppressMessages(train.dt <- train %>% dplyr::right_join(keys) %>% 
    dplyr::right_join(dept.store.xref) %>%
    select(!c(IsHoliday)) %>% as.data.table(key="Date") %>%
    dcast(Date + Dept ~ Store, value.var = "Weekly_Sales") %>%
    group_split(Dept)) 

  # Same for test
  keys <- test.current %>% tidyr::expand(Dept, Store, Date)
  suppressMessages(test.dt <- test.current %>% dplyr::right_join(keys) %>% 
    dplyr::right_join(dept.store.xref) %>%
    mutate(Weekly_Pred=0) %>%
    as.data.table(key="Date") %>%
    dcast(Date + Dept ~ Store, value.var = "Weekly_Pred") %>%
    group_split(Dept)) 

  package <- list(train=train.dt, test=test.dt, current=test.current)  
   
  return(package)
}