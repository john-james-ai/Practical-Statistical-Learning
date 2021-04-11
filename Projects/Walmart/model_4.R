library(lubridate)
library(forecast)
library(tidyverse)
library(plyr)
library(reshape2)
source("utils.R")

model_4 = function() {
  # Obtain the model being evaluated.
  model <- tslm.svd

  # as usual
  start_date <- ymd("2011-03-01") %m+% months(2 * (t - 1))
  end_date <- ymd("2011-05-01") %m+% months(2 * (t - 1))
  test_current <- test %>% filter(Date >= start_date & Date < end_date) %>% select(-IsHoliday) %>% mutate(Wk = week(Date))
  
  if (t > 1){
    train <<- train %>% add_row(new_train)
  }
  
  # Get unique stores, dates, and departments
  train.dates <- unique(train$Date)
  num.train.dates <- length(train.dates)
  test.dates <- unique(test_current$Date)
  num.test.dates <- length(test.dates)
  all.stores <- unique(test_current$Store)
  num.stores <- length(all.stores)
  test.depts <- unique(test_current$Dept)  
  
  # Add week variable to train and test. Subtract 1 from weeks for 2010, aligns them with other years.
  train$Wk = ifelse(year(train$Date) == 2010, week(train$Date)-1, week(train$Date))
  test_current$Wk = ifelse(year(test_current$Date) == 2010, week(test_current$Date)-1, week(test_current$Date))
  
  # Initialize training and test dataframes with Date and Store information 
  train.df <- data.frame(Date=rep(train.dates, num.stores),
                            Store=rep(all.stores, each=num.train.dates)) 
  test.df <- data.frame(Date=rep(test.dates, num.stores),
                           Store=rep(all.stores, each=num.test.dates))
  print(head(train.df))
  print(dim(train.df))
  predictions <- test_current
  predictions$Weekly_Pred <- 0

  for (dept in test.depts) {
    print(paste("Dept: ", dept))
    tmp_train.dept <- train.df
    tmp_test.dept <- test.df
    # This joins in Weekly_Sales but generates NA's. NA's are resolved
    # by each model differently.
    tmp_train.dept <- join(tmp_train.dept, train[train$Dept==dept, c('Store','Date','Weekly_Sales')])
    tmp_train.dept <- reshape::cast(tmp_train.dept, Date ~ Store, value.var="Weekly_Sales")    
    print("Loop join/dcast 1")
    
    tmp_test.dept$Weekly_Sales <- 0
    tmp_test.dept <- reshape::cast(tmp_test.dept, Date ~ Store, value.var="Weekly_Sales")
    print("Loop join/dcast 2")
    print(head(tmp_train.dept))

    # Train model and produce forecasts
    result <- model(tmp_train.dept, tmp_test.dept)
    print("Model fit")
    # Here, we shift the forecast for departments that experience a holiday surge
    if (t==5) {
      result = shift(result)
    }
    # This has all Stores/Dates for this dept, but may have some that
    # don't go into the submission.    
    print("About to melt")
    print(head(result))
    result <- reshape2::melt(result, id=c("Date"), value.name="Weekly_Sales")    
    print("Melted")
    print(head(result))
    print(paste("Result dim",dim(result)))
    
    predictions.dept.idx <- predictions$Dept==dept
    #These are the Store-Date pairs in the submission for this dept
    predictions.dept <- predictions[predictions.dept.idx, c('Store', 'Date')]
    print(head(predictions.dept))
    predictions.dept <- join(predictions.dept, result)
    print(head(predictions.dept))
    print("Checking match")
    print(length(predictions.dept$Weekly_Sales))
    print(length(predictions.dept.idx))
    predictions$Weekly_Pred[predictions.dept.idx] <- predictions.dept$Weekly_Sales    
    
    # Save Predictions  ** DELETE BEFORE SUBMITTING ****
    #save_results(predictions, mname, t)
  }
  predictions
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
  train[is.na(train)] <- 0
  for(j in 2:ncol(train)){
    s <- ts(train[, j], frequency=52)
    fit <- tslm(s ~ trend + season)
    fc <- forecast(fit, h=horizon)
    test[, j] <- as.numeric(fc$mean)
  }
  test
}

tslm.svd <- function(train, test, n.comp=12){
  # Computes a forecast using time series linear regression with dimension reduction
  # 
  # This function was adapted from the following source:
  #   1. Author: David Thaler
  #   2. Date: May 13, 2014
  #   3. Title: Walmart_competition_code
  #   4. Link: https://github.com/davidthaler/Walmart_competition_code/grouped.forecast.R

  horizon <- nrow(test)
  train[is.na(train)] <- 0
  print("About to svd")
  print(dim(train))
  train <- as.data.frame(train)
  train <- preprocess.svd(train, n.comp) 
  print("Finished svd")
  print(dim(train))
  
  for(j in 2:ncol(train)){
    s <- ts(train[, j], frequency=52)
    fit <- tslm(s ~ trend + season)
    fc <- forecast(fit, h=horizon)
    test[, j] <- as.numeric(fc$mean)
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
  # 
  # This function was adapted from the following source:
  #   1. Author: David Thaler
  #   2. Date: May 13, 2014
  #   3. Title: Walmart_competition_code
  #   4. Link: https://github.com/davidthaler/Walmart_competition_code/grouped.forecast.R
  train[is.na(train)] <- 0
  z <- svd(train[, 2:ncol(train)], nu=n.comp, nv=n.comp)
  s <- diag(z$d[1:n.comp])
  train[, 2:ncol(train)] <- z$u %*% s %*% t(z$v)
  train
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
  holiday <- test[idx, 2:46]
  baseline <- mean(rowMeans(holiday[c(1, 5), ], na.rm=TRUE))
  surge <- mean(rowMeans(holiday[2:4, ], na.rm=TRUE))
  holiday[is.na(holiday)] <- 0
  if(is.finite(surge/baseline) & surge/baseline > threshold){
      shifted.sales <- ((7-shift)/7) * holiday
      shifted.sales[2:5, ] <- shifted.sales[2:5, ] + (shift/7) * holiday[1:4, ]
      shifted.sales[1, ] <- holiday[1, ]
      test[idx, 2:46] <- shifted.sales
  }
  test
}