library(lubridate)
library(tidyverse)
library(forecast)
library(dplyr)
library(reshape)
library(plyr)
source("utils.R")

model_1 = function(){
  
  # as usual
  start_date <- ymd("2011-03-01") %m+% months(2 * (t - 1))
  end_date <- ymd("2011-05-01") %m+% months(2 * (t - 1))
  test_current <- test %>% filter(Date >= start_date & Date < end_date) %>% select(-IsHoliday) %>% mutate(Wk = week(Date))
  
  if (t > 1){
    train <<- train %>% add_row(new_train)
  }
  
  # find the unique pairs of (Store, Dept) combo that appeared in both training and test sets
  train_pairs <- train[, 1:2] %>% dplyr::count(Store, Dept) %>% filter(n != 0)
  test_pairs <- test_current[, 1:2] %>% dplyr::count(Store, Dept) %>% filter(n != 0)
  unique_pairs <- intersect(train_pairs[, 1:2], test_pairs[, 1:2])

  # Get unique train and test dates
  test_depts <- unique(test_current$Dept)
  train_dates <- unique(train$Date)
  n_train_dates <- length(train_dates)
  test_dates <- unique(test_current$Date)
  n_test_dates <- length(test_dates)

  # pick out the needed training samples, convert to dummy coding, then put them into a list
  train_split <- unique_pairs %>% 
    dplyr::left_join(train, by = c('Store', 'Dept'))  %>% 
    group_split(Store, Dept) 
  
  # do the same for the test set
  test_split <- unique_pairs %>% 
    dplyr::left_join(test_current, by = c('Store', 'Dept'))  %>% 
    group_split(Store, Dept) 
  
  # pre-allocate a list to store the predictions
  test_forecast <- vector(mode = "list", length = nrow(unique_pairs))
  
  # perform regression for each split, note we used lm.fit instead of lm
  for (i in 1:nrow(unique_pairs)) {
    
    tmp_train <- as.data.frame(train_split[[i]])
    tmp_test <- as.data.frame(test_split[[i]])    

    #Initialize training and test dataframes with dates
    train.df <- data.frame(Date=train_dates)
    test.df  <- data.frame(Date=test_dates)

    # Merge sales information 
    train.df <- merge(x=train.df, y=tmp_train, by="Date", all=TRUE)
    test.df  <- merge(x=test.df, y=tmp_test, by="Date", all =TRUE)
    train.df[is.na(train.df)] <- 0

    # Fit the model and render forecast
    pred <- tslm.basic(train.df, test.df)    

    # Holiday Shift
    if (t==5) {
      pred = shift(pred)
    }      
    test_forecast[[i]] <- pred
  }


  
  # turn the list into a table at once, this is much more efficient then keep concatenating small tables
  test_forecast <- bind_rows(test_forecast)
  
  #save_results(test_forecast, mname,t)    
  
  return(test_forecast)
}
tslm.basic <- function(train, test){
  # Computes a forecast using time series linear regression 
  # 
  # This function was adapted from the following source:
  #   1. Author: David Thaler
  #   2. Date: May 13, 2014
  #   3. Title: Walmart_competition_code
  #   4. Link: https://github.com/davidthaler/Walmart_competition_code/grouped.forecast.R
  
  horizon <- nrow(test)
  s <- ts(train$Weekly_Sales, frequency=52)
  fit <- tslm(s ~ trend + season)
  fc <- forecast(fit, h=horizon)
  test$Weekly_Pred <- as.numeric(fc$mean)  
  test
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
  # train - this is an n_weeks x n_stores matrix of values of Weekly_Sales
  #         for the training set within department, across all the stores
  # test - this is a (forecast horizon) x n_stores matrix of Weekly_Sales
  #        for the training set within department, across all the stores
  # threshold - the shift is executed if the mean of Weekly_Sales for weeks
  #          49-51 is greater than that for weeks 48 and 52 by at least
  #          a ratio of threshold
  # shift - The number of days to shift sales around Christmas.
  #         Should be 2 if the model is based on the last year only,
  #         or 2.5 if it uses both years
  #
  # This function was adapted from the following source:
  #   1. Author: David Thaler
  #   2. Date: May 13, 2014
  #   3. Title: Walmart_competition_code
  #   4. Link: https://github.com/davidthaler/Walmart_competition_code/grouped.forecast.R
  s <- ts(rep(0,39), frequency=52, start=c(2012,44))
  idx <- cycle(s) %in% 48:52
  holiday <- test$Weekly_Pred[2:46]
  baseline <- mean(rowMeans(holiday[c(1, 5)], na.rm=TRUE))
  surge <- mean(rowMeans(holiday[2:4], na.rm=TRUE))
  holiday[is.na(holiday)] <- 0
  if(is.finite(surge/baseline) & surge/baseline > threshold){
      shifted.sales <- ((7-shift)/7) * holiday
      shifted.sales[2:5] <- shifted.sales[2:5] + (shift/7) * holiday[1:4]
      shifted.sales[1] <- holiday[1]
      test$Weekly_Pred[2:46] <- shifted.sales
  }
  test
}
