library(glmnet)
library(forecast)
library(tidyverse)
library(lubridate)
library(reshape)
library(data.table)
library(plyr)
library(uroot)

mypredict = function(model) {  
  model = model[[1]]
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
  train.dt <- data.table(Date=rep(train.dates, num.stores),
                            Store=rep(all.stores, each=num.train.dates)) 
  test.dt <- data.table(Date=rep(test.dates, num.stores),
                           Store=rep(all.stores, each=num.test.dates))
  
  predictions <- test_current
  predictions$Weekly_Pred <- 0

  for (dept in test.depts) {
    tmp_train.dept <- train.dt
    tmp_test.dept <- test.dt
    # This joins in Weekly_Sales but generates NA's. NA's are resolved
    # by each model differently.
    tmp_train.dept <- suppressMessages(join(tmp_train.dept, train[train$Dept==dept, c('Store','Date','Weekly_Sales')]))
    tmp_train.dept <- suppressMessages(dcast(tmp_train.dept, Date ~ Store))    
    
    tmp_test.dept$Weekly_Sales <- 0
    tmp_test.dept <- suppressMessages(dcast(tmp_test.dept, Date ~ Store))
    result <- model(tmp_train.dept, tmp_test.dept)
    # This has all Stores/Dates for this dept, but may have some that
    # don't go into the submission.
    result <- melt(result)        
    predictions.dept.idx <- predictions$Dept==dept
    #These are the Store-Date pairs in the submission for this dept
    predictions.dept <- predictions[predictions.dept.idx, c('Store', 'Date')]
    predictions.dept <- suppressMessages(join(predictions.dept, result))
    predictions$Weekly_Pred[predictions.dept.idx] <- predictions.dept$value    
    
  }
  predictions
}

tslm.basic <- function(train, test){
  # Computes a forecast using linear regression and seasonal dummy variables
  #
  # args:
  # train - A matrix of Weekly_Sales values from the training set of dimension
  #         (number of weeks in training data) x (number of stores)
  # test - An all-zeros matrix of dimension:
  #       (number of weeks in training data) x (number of stores)
  #       The forecasts are written in place of the zeros.
  #
  # returns:
  #  the test(forecast) data frame with the forecasts filled in 
  horizon <- nrow(test)
  train[is.na(train)] <- 0  
  train_df <- todf(train)
  test_df <- todf(test)
  for(j in 2:ncol(train_df)){
    s <- ts(train_df[, j], frequency=52)    
    fit <- tslm(s ~ trend + season)
    fc <- forecast(fit, h=horizon)
    test_df[, j] <- as.numeric(fc$mean)
  }
  test <- todt(test_df)  
  return(test)
}

stlf.svd <- function(train, test, model.type, n.comp){
  # Replaces the training data with a rank-reduced approximation of itself,
  # then forecasts each store using stlf() from the forecast package.
  # That function performs an STL decomposition on each series, seasonally
  # adjusts the data, non-seasonally forecasts the seasonally adjusted data,
  # and then adds in the naively extended seasonal component to get the
  # final forecast.
  #
  # args:
  # train - A matrix of Weekly_Sales values from the training set of dimension
  #         (number of weeeks in training data) x (number of stores)
  # test - An all-zeros matrix of dimension:
  #       (number of weeeks in training data) x (number of stores)
  #       The forecasts are written in place of the zeros.
  # model.type - one of 'ets' or 'arima', specifies which type of model to
  #        use for the non-seasonal forecast
  # n.comp - the number of components to keep in the singular value
  #         decomposition that is performed for preprocessing
  #
  # returns:
  #  the test(forecast) data frame with the forecasts filled in 
  train_df = todf(train)
  test_df = todf(test)

  horizon <- nrow(test_df)
  train <- preprocess.svd(train_df, n.comp) 
  for(j in 2:ncol(train)){
    s <- ts(train_df[, j], frequency=52)
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
    test_df[, j] <- pred
  }
  test <- todt(test_df)
  return(test)
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
  train[is.na(train)] <- 0
  z <- svd(train[, 2:ncol(train)], nu=n.comp, nv=n.comp)
  s <- diag(z$d[1:n.comp])
  train[, 2:ncol(train)] <- z$u %*% s %*% t(z$v)
  train
}
fallback <- function(train, horizon){
  # This method is a fallback forecasting method in the case that there are
  # enough NA's to possibly crash arima models. It takes one seasonal 
  # difference, forecasts with a level-only exponential model, and then
  # inverts the seasonal difference.
  # 
  # args:
  # train - a vector of training data for one store
  # horizon - the forecast horizon in weeks
  #
  # returns:
  #  a vector of forecast values
  s <- ts(train, frequency=52)
  s[is.na(s)] <- 0
  fc <- ses(diff(s, 52), h=horizon)
  result <- diffinv(fc$mean, lag=52, xi=s[length(s) - 51:0])
  result[length(result) - horizon:1 + 1]
}

seasonal.arima.svd <- function(train, test, n.comp){
  # Replaces the training data with a rank-reduced approximation of itself
  # and then produces seasonal arima forecasts for each store.
  #
  # args:
  # train - A matrix of Weekly_Sales values from the training set of dimension
  #         (number of weeeks in training data) x (number of stores)
  # test - An all-zeros matrix of dimension:
  #       (number of weeeks in training data) x (number of stores)
  #       The forecasts are written in place of the zeros.
  # n.comp - the number of components to keep in the singular value
  #         decomposition that is performed for preprocessing
  #
  # returns:
  #  the test(forecast) data frame with the forecasts filled in 
  train_df = todf(train)
  test_df = todf(test)
  horizon <- nrow(test_df)
  tr <- preprocess.svd(train_df, n.comp)
  for(j in 2:ncol(tr)){
    if(sum(is.na(train_df[, j])) > nrow(train_df)/3){
      # Use DE model as fallback
      test[, j] <- fallback(tr[,j], horizon)
      store.num <- names(train_df)[j]
      print(paste('Fallback on store:', store.num))
    }else{
      # fit arima model
      s <- ts(tr[, j], frequency=52)
      model <- auto.arima(s, ic='bic', seasonal.test='ch')
      fc <- forecast(model, h=horizon)
      test_df[, j] <- as.numeric(fc$mean)
    }
  }
  test <- todt(test_df)
  test
}

todt <- function(x) {
  dt = x
  setDT(dt)
  return(dt)
}

todf <- function(x) {
  df = x
  setDF(df)
  return(df)
}