library(lubridate)
library(forecast)
library(tidyverse)
library(reshape)
library(plyr)

model_3 = function() {
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
  predictions <- test_current
  predictions$Weekly_Pred <- 0

  for (dept in test.depts) {
    tmp_train.dept <- train.df
    tmp_test.dept <- test.df
    # This joins in Weekly_Sales but generates NA's. NA's are resolved
    # by each model differently.
    tmp_train.dept <- suppressMessages(join(tmp_train.dept, train[train$Dept==dept, c('Store','Date','Weekly_Sales')]))
    tmp_train.dept <- suppressMessages(cast(tmp_train.dept, Date ~ Store))    
    
    tmp_test.dept$Weekly_Sales <- 0
    tmp_test.dept <- suppressMessages(cast(tmp_test.dept, Date ~ Store))
    result <- model(tmp_train.dept, tmp_test.dept)
    # This has all Stores/Dates for this dept, but may have some that
    # don't go into the submission.    
    result <- melt(result, id=c("Date"))
    
    predictions.dept.idx <- predictions$Dept==dept
    #These are the Store-Date pairs in the submission for this dept
    predictions.dept <- predictions[predictions.dept.idx, c('Store', 'Date')]
    predictions.dept <- suppressMessages(join(predictions.dept, result))
    predictions$Weekly_Pred[predictions.dept.idx] <- predictions.dept$value    
    
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

tslm.svd <- function(train, test, n.comp=8){
  # Computes a forecast using time series linear regression with dimension reduction
  # 
  # This function was adapted from the following source:
  #   1. Author: David Thaler
  #   2. Date: May 13, 2014
  #   3. Title: Walmart_competition_code
  #   4. Link: https://github.com/davidthaler/Walmart_competition_code/grouped.forecast.R

  horizon <- nrow(test)
  train[is.na(train)] <- 0
  train <- preprocess.svd(train, n.comp) 
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