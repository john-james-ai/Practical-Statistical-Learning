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
# Last Modified : Tuesday, April 13th 2021, 11:34:16 pm                       #
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
library(plyr)
library(tidyverse)
library(forecast)
library(data.table)

verbose = FALSE
default.params <- list(model="tslm.svd", data='t', adjust=TRUE, model.type="ets", n.comp=12)
# =========================================================================== #
#                               PREDICT                                       #
# =========================================================================== #
mypredict = function(params=NULL){
  # Forecasts sales for 8 weeks, given training data of the previous 12+ months.
  # The params variable is used during development and specifies the model
  # and parameters to be predicted.
  
  # Unpack params if present
  if (length(params) == 0) {
    params <- default.params
  }
  
  # Preprocesses the data in the format required of the designated model
  d <- preprocess_data(params)
  train.dt <- d$train
  test.dt <- d$test
  test.fold <- d$current

  # Obtains the model designated in params. If params is NULL, return best model.
  model <- get.model(params)  

  # Pre-allocate a list to store forecasts
  forecasts <- vector(mode = "list", length = length(train.dt))  

  # Process data by department 
  for (i in 1:length(train.dt)) {
    train.dept.dt <- as.data.table(train.dt[[i]])
    test.dept.dt <- as.data.table(test.dt[[i]])    
    
    # Fit the model and render forecast
    pred <- model(train.dept.dt, test.dept.dt, model.type=params$model.type, n.comp=params$n.comp)
    
    # Shifts forecast forward to account for the floating
    # Christmas Holiday. 
    if ((t==5) & (params$adjust == TRUE)) {      
      pred <- shift(pred)
    }    

    # Post process data
    pred <- postprocess_data(test.current=test.dept.dt, pred.current=pred, params=params)
    forecasts[[i]] <- pred
  }
  
  forecasts <- bind_rows(forecasts)
  setDT(forecasts)

  # Merge forecasts with test data to create submission  
  submission <- test.fold %>% left_join(forecasts, by=c("Date","Dept","Store"))
    
  save_results(submission, params,t)    
  
  return(submission)
}
# =========================================================================== #
#                       TIME SERIES FORECASTING MODELS                        #
# =========================================================================== #
get.model <- function(params=NULL) {
  # Selects the forecast model based on params. If params is null, returns
  # the best performing model tslm.svd

  announce()

  if (is.null(params)) {
    m <- tslm.svd
  } else if (params$model == 'regression') {
    m <- regression
  } else if (params$model == 'snaive') {
    m <- snaive.baseline
  } else if (params$model == 'snaive.custom') {
    m <- snaive.custom
  } else if (params$model == 'tslm') {
    m <- tslm.baseline
  } else if (params$model == 'tslm.svd') {
    m <- tslm.svd
  } else if (params$model == 'stlf') {
    m <- stlf.baseline
  } else if (params$model == 'stlf.svd') {
    m <- stlf.svd 
  } else {
    print(paste(params$model, " is not a valid model"))
  }
  return(m)
}
# =========================================================================== #
regression <- function(train, test, model.type=NULL, n.comp=NULL) {
  # Computes forecast using linear regression
  announce()

  coef <- lm.fit(as.matrix(train[, -(2:4)]), train$Weekly_Sales)$coefficients
  coef[is.na(coef)] <- 0
  pred <- coef[1] + as.matrix(test[, 4:55]) %*% coef[-1]
  return(pred)
}
# --------------------------------------------------------------------------- #
snaive.baseline<- function(train, test, model.type=NULL, n.comp=NULL) {
  # Sets forecast equal to last observed value from the same season of the prior year.
  announce()

  train <- as.matrix(train[,":="(Date=NULL,Dept=NULL)])
  train[is.na(train)] <- 0
  horizon <- nrow(test)
  
  for(j in 1:ncol(train)){
    s <- ts(train[, j], frequency=52) 
    fc <- snaive(s, horizon)
    test[, j+2] <-  as.numeric(fc$mean)  
  }
  return(test) 
}
# --------------------------------------------------------------------------- #
snaive.custom <- function(train, test, model.type=NULL, n.comp=NULL){
  # Manually computes seasonal naive forecasts. Does not use forecast packaage.
  announce()
  h <- nrow(test)
  tr <- train[nrow(train) - (52:1) + 1,]
  tr[is.na(tr)] <- 0
  test[,2:ncol(test)]  <- tr[1:h,2:ncol(test)]
  test
}
# --------------------------------------------------------------------------- #
tslm.baseline <- function(train, test, model.type=NULL, n.comp=NULL){
  # Computes a forecast using time series linear regression 
  announce()
  
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
# --------------------------------------------------------------------------- #
tslm.svd <- function(train, test, model.type, n.comp=12){
  # Computes a forecast using time series linear regression 
  announce()
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
# --------------------------------------------------------------------------- #
stlf.baseline <- function(train, test, model.type, n.comp) {
  # Performs STL Decomposition on a rank-reduced approximation of the data
  announce()
  horizon <- nrow(test)
  train <- as.matrix(train[,":="(Date=NULL,Dept=NULL)]) 
  train[is.na(train)] <- 0
  for(j in 1:ncol(train)){
    s <- ts(train[, j], frequency=4)
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
# --------------------------------------------------------------------------- #
stlf.svd <- function(train, test, model.type, n.comp) {
  # Performs STL Decomposition on a rank-reduced approximation of the data
  announce()
  horizon <- nrow(test)
  train <- preprocess.svd(train, n.comp) 
  for(j in 1:ncol(train)){
    s <- ts(train[, j], frequency=4)
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




# =========================================================================== #
#                              DATA PROCESSING                                #
# =========================================================================== #
preprocess_data <- function(params=NULL) {
  # Preprocesses data according to the data parameter in the params list.
  # If params is NULL, it processes data for time series analysis.

  announce()

  if (length(params)==0) {
    d <- prep_ts_data()
  } else if(params$data == 'r') {
    d <- prep_regression_data()    
  } else {
    d <- prep_ts_data()  }
  return(d)
}
# --------------------------------------------------------------------------- #
#                            TIME SERIES FORMAT                               #
# --------------------------------------------------------------------------- #
prep_ts_data <- function() {
  # This function prepares the training data for time series forecasting.
  # 
  # Iterations 2:10 will feed next periods sales

  announce()
  if (t > 1){
    train <<- train %>% add_row(new_train)
  }  

  # Extract current fold data from test
  start_date <- ymd("2011-03-01") %m+% months(2 * (t - 1))
  end_date <- ymd("2011-05-01") %m+% months(2 * (t - 1))
  test.fold <- test %>% filter(Date >= start_date & Date < end_date) %>% 
    select(-IsHoliday) 

  # Find the unique pairs of department and stores that are appear in 
  # both training and test set.
  train.pairs <- train[, 1:2] %>% dplyr::count(Store, Dept) %>% filter(n != 0)
  test.pairs <- test.fold[, 1:2] %>% dplyr::count(Store, Dept) %>% filter(n != 0)
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
  keys <- test.fold %>% tidyr::expand(Dept, Store, Date)
  suppressMessages(test.dt <- test.fold %>% dplyr::right_join(keys) %>% 
    dplyr::right_join(dept.store.xref) %>%
    mutate(Weekly_Pred=0) %>%
    as.data.table(key="Date") %>%
    dcast(Date + Dept ~ Store, value.var = "Weekly_Pred") %>%
    group_split(Dept)) 

  package <- list(train=train.dt, test=test.dt, current=test.fold)  
   
  return(package)
}
# --------------------------------------------------------------------------- #
#                             REGRESSION FORMAT                               #
# --------------------------------------------------------------------------- #
prep_regression_data <- function() {

  
  
  announce()
  if (t > 1){
    train <<- train %>% add_row(new_train)
  }  
  # Extract test data for the current fold.
  start_date <- ymd("2011-03-01") %m+% months(2 * (t - 1))
  end_date <- ymd("2011-05-01") %m+% months(2 * (t - 1))
  test_current <- test %>% filter(Date >= start_date & Date < end_date) %>% select(-IsHoliday) %>% mutate(Wk = week(Date))
  
  # find the unique pairs of (Store, Dept) combo that appeared in both training and test sets
  train_pairs <- train[, 1:2] %>% dplyr::count(Store, Dept) %>% filter(n != 0)
  test_pairs <- test_current[, 1:2] %>% dplyr::count(Store, Dept) %>% filter(n != 0)
  unique_pairs <- intersect(train_pairs[, 1:2], test_pairs[, 1:2])

  # pick out the needed training samples, convert to dummy coding, then put them into a list
  train_split <- unique_pairs %>% 
    left_join(train, by = c('Store', 'Dept')) %>% 
    mutate(Wk = factor(ifelse(year(Date) == 2010, week(Date) - 1, week(Date)), levels = 1:52)) %>% 
    mutate(Yr = year(Date))
  train_split = as_tibble(model.matrix(~ Weekly_Sales + Store + Dept + Yr + Wk, train_split)) %>% group_split(Store, Dept)
    
  # do the same for the test set
  test_split <- unique_pairs %>% 
    left_join(test_current, by = c('Store', 'Dept')) %>%    
    mutate(Date=Date) %>% 
    mutate(Wk = factor(ifelse(year(Date) == 2010, week(Date) - 1, week(Date)), levels = 1:52)) %>% 
    mutate(Yr = year(Date))
  if (t==5) {
    inspect(test_current, "Checking test current")
    inspect(test_split, "Checking fold 5 test_split")
    mm <- model.matrix(~ Store + Dept + Yr + Wk, test_split)
    inspect(mm, "Checking Model matrix")
  }

  test_split = as_tibble(model.matrix(~ Store + Dept + Yr + Wk, test_split)) %>% mutate(Date = test_split$Date) %>% group_split(Store, Dept)  

  package <- list(train=train_split, test=test_split, current=test_current)

  return(package)
}
# --------------------------------------------------------------------------- #
preprocess.svd <- function(train, n.comp){
  # Replaces the training data with a rank-reduced approximation of itself.
  announce()

  train <- as.matrix(train[,":="(Date=NULL,Dept=NULL)])
  train[is.na(train)] <- 0
  z <- svd(train[, 2:ncol(train)], nu=n.comp, nv=n.comp)
  s <- diag(z$d[1:n.comp])
  train[, 2:ncol(train)] <- z$u %*% s %*% t(z$v)
  return(train)
}
# --------------------------------------------------------------------------- #
postprocess_data <- function(test.current, pred.current, params=NULL) {
  # Preprocesses data according to the data parameter in the params list.
  # If params is NULL, it processes data for time series analysis.

  announce()

  if (length(params)==0) {
    d <- post_ts_data(test.current=test.current, pred.current=pred.current)
  } else if(params$data == 'r') {
    d <- post_regression_data(test.current=test.current, pred.current=pred.current)    
  } else {
    d <- post_ts_data(test.current=test.current, pred.current=pred.current)  }
  return(d)
}
# --------------------------------------------------------------------------- #
post_ts_data <- function(test.current, pred.current) {
  # Reshape into original long form and convert Store back to integer. 
  
  pred.current <- melt(pred.current, id.vars=c("Date","Dept"), variable.name = "Store",
                value.name = "Weekly_Pred", variable.factor=FALSE)
  pred.current$Store <- as.integer(pred.current$Store)    
  return(pred.current)  
} 
# --------------------------------------------------------------------------- #
post_regression_data <- function(test.current, pred.current) {
  # Reformat to that expected by the grader.
  test.current <- cbind(test.current[,2:3], Date=test.current$Date, Weekly_Pred=pred.current[,1])
  return(test.current)
} 

# =========================================================================== #
#                                  SHIFT                                      #
# =========================================================================== #
shift <- function(test, threshold=1.1, shift=1.5){.
  # Shifts forecasts for weeks 48-51 to account for shift of 12/25.  
  announce()
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
  return(test)
}
# =========================================================================== #
#                             UTILITY FUNCTIONS                               #
# =========================================================================== #
save_results <- function(x,params, fold) {
  # Saves predictions for model 
  announce()
  maindir <- "results/"
  subdir <- get_directory(params)
  path <- paste0(maindir, subdir)
  
  dir.create(path, showWarnings = FALSE)  
  filename <- paste0("/fold_",fold,"_predicted.csv")
  
  filepath <- paste0(path,filename)
  write.csv(x, filepath, row.names=FALSE)
}
# --------------------------------------------------------------------------- #
get_directory <- function(params=NULL) {
  announce()
  if (params$adjust == TRUE) {
    shift = "_hs"
  } else {
    shift = ""
  }
  if (is.null(params)) {
    f <- 'tslm_hs_svd_12_pc'
  } else if (params$model == 'regression') {
    f <- paste0(params$model,shift) 
  } else if (params$model == 'snaive') {
    f <- paste0(params$model,shift) 
  } else if (params$model == 'snaive.custom') {
    f <- paste0(params$model,shift) 
  } else if (params$model == 'tslm') {
    f <- paste0(params$model,shift) 
  } else if (params$model == 'tslm.svd') {
    f <- paste0(params$model,shift,"_", params$n.comp, "_pc")
  } else if (params$model == 'stlf') {
    f <- paste0(params$model,shift,"_",params$model.type)
  } else if (params$model == 'stlf.svd') {
    f <- paste0(params$model,shift,"_",params$model.type, "_", params$n.comp, "_pc")
  }  else {
    print("Model type invalid")
  }
  return(f)    
}
# --------------------------------------------------------------------------- #
gather_results <- function(model) {
  # Obtains predictions for model
  announce()
  maindir <- "results/"
  subdir <- model
  path <- paste0(maindir, subdir)
  actual_list <- list()
  pred_list <- list()
  
  
  for (i in 1:10) {
      filename <- paste0("/fold_",i,"_actual.csv")
      filepath <- paste0(path, filename)
      actual_list[[i]] <- read.csv(filepath)        
  }

  for (i in 1:10) {
      filename <- paste0("/fold_",i,"_predicted.csv")
      filepath <- paste0(path, filename)
      pred_list[[i]] <- read.csv(filepath)        
  }

  actual <- data.table::rbindlist(actual_list)
  predicted <- data.table::rbindlist(pred_list)
  results <- list("actual"=actual, "predicted"=predicted)

  return(results)
}
# --------------------------------------------------------------------------- #
get_filename <- function(params=NULL) {
  announce()
  if (params$adjust == TRUE) {
    shift = "_hs"
  } else {
    shift = ""
  }
  if (is.null(params)) {
    f <- 'tslm_hs_svd_12_pc.csv'
  } else if (params$model == 'regression') {
    f <- paste0(params$model,shift,".csv") 
  } else if (params$model == 'snaive') {
    f <- paste0(params$model,shift,".csv") 
  } else if (params$model == 'snaive.custom') {
    f <- paste0(params$model,shift,".csv") 
  } else if (params$model == 'tslm') {
    f <- paste0(params$model,shift,".csv") 
  } else if (params$model == 'tslm.svd') {
    f <- paste0(params$model,shift,"_", params$n.comp, "_pc.csv")
  } else if (params$model == 'stlf') {
    f <- paste0(params$model,shift,"_",params$model.type, ".csv")
  } else if (params$model == 'stlf.svd') {
    f <- paste0(params$model,shift,"_",params$model.type, "_", params$n.comp, "_pc.csv")
  }  else {
    print("Model type invalid")
  }
  return(f)    
}
# --------------------------------------------------------------------------- #
save_stats <- function(stats, params=NULL) {
  announce()
  directory <- "results/stats/"
  filename <- get_filename(params)
  filepath <- paste0(directory,filename)

  df <- as.data.frame(matrix(unlist(stats), nrow=length(unlist(stats[1]))))
  names(df) <- c("Fold", "WAE", "Time Elapsed")  
  dir.create(directory, showWarnings = FALSE)
  write.csv(df, filepath, row.names=FALSE)
}
# --------------------------------------------------------------------------- #
save_summary_stats <- function(stats, params=NULL) {
  announce()
  directory <- "analysis/"
  filename <- 'summary.csv'
  filepath <- paste0(directory,filename)

  result <- list(Model=toupper(params$model), Score=stats$score, Time=stats$time)
  df <- data.frame(Model=c(), Score=c(), Time=c())

  if (file.exists(filepath)) {
    df <- read.csv(filepath)
    df <- rbind(df,result)
  } else {
    df <- rbind(df,result)
  }
  dir.create(directory, showWarnings = FALSE)
  write.csv(df, filepath, row.names=FALSE)
}
# --------------------------------------------------------------------------- #
inspect <- function(x, msg=FALSE, summary=FALSE) {  

    cat("\n")
    print("===========================================")
    if (msg != FALSE) {
        print(msg)        
    }
    print("-------------------------------------------")
    
    print(dim(x))
    print(head(x))    
    if (summary == TRUE) {
        print(summary(x))
    }
}
announce <- function() {
  if (verbose==TRUE)
    print(sys.calls()[[sys.nframe()-1]])
}