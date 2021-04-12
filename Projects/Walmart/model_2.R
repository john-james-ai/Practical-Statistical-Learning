library(lubridate)
library(tidyverse)
library(forecast)
library(dplyr)
library(reshape)
library(plyr)
library(testit)
source("utils.R")

model_2 = function(){

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
    pred <- tslm.baseline(train.dept.dt, test.dept.dt)  

    # Reshape into original long form and convert Store back to integer. 
    pred <- melt(pred, id.vars=c("Date","Dept"), variable.name = "Store",
                  value.name = "Weekly_Pred", variable.factor=FALSE)
    pred$Store <- as.integer(pred$Store)

    #Put this in module that does forecast
    # # Holiday Shift
    # if (t==5) {
    #   pred = shift(pred)
    # }      
    forecasts[[i]] <- pred
  }
  
  # turn the list into a data.table at once, this is much more efficient then keep concatenating small tables  
  forecasts <- bind_rows(forecasts)
  setDT(forecasts)
  # Merge forecasts with submission
  inspect(forecasts,"Combined forecasts prior to merge")
  
  submission <- test.current %>% left_join(forecasts, by=c("Date","Dept","Store"))
  
  inspect(submission, "Check submission")
  
  save_results(submission, mname,t)    
  
  return(submission)
}
tslm.baseline <- function(train, test){
  # Computes a forecast using time series linear regression 
  # 
  # This function was adapted from the following source:
  #   1. Author: David Thaler
  #   2. Date: May 13, 2014
  #   3. Title: Walmart_competition_code
  #   4. Link: https://github.com/davidthaler/Walmart_competition_code/grouped.forecast.R
  
  train <- as.matrix(train[,":="(Date=NULL,Dept=NULL)])
  horizon <- nrow(test)
  train[is.na(train)] <- 0

  for (j in 1:ncol(train)) {
    s <- ts(train[, j], frequency=52)
    fit <- tslm(s ~ trend + season)
    fc <- forecast(fit, h=horizon)
    test[, j+2] <- as.numeric(fc$mean)  
  }
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

prep_ts_data <- function() {
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
  train.dt <- train %>% dplyr::right_join(keys) %>% 
    dplyr::right_join(dept.store.xref) %>%
    select(!c(IsHoliday)) %>% as.data.table(key="Date") %>%
    dcast(Date + Dept ~ Store, value.var = "Weekly_Sales") %>%
    group_split(Dept) 

  # Same for test
  keys <- test.current %>% tidyr::expand(Dept, Store, Date)
  test.dt <- test.current %>% dplyr::right_join(keys) %>% 
    dplyr::right_join(dept.store.xref) %>%
    mutate(Weekly_Pred=0) %>%
    as.data.table(key="Date") %>%
    dcast(Date + Dept ~ Store, value.var = "Weekly_Pred") %>%
    group_split(Dept) 

  package <- list(train=train.dt, test=test.dt, current=test.current)  
   
  return(package)
}