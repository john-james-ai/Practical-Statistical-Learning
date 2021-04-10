library(glmnet)
library(forecast)
library(tidyverse)
library(lubridate)
library(reshape)
library(data.table)
library(dplyr)

source("visual.R")

# read in train / test dataframes
train <- readr::read_csv('data/train_ini.csv')
test <- readr::read_csv('data/test.csv')

# save weighted mean absolute error WMAE
num_folds <- 10

for (t in 1:num_folds) {
  # as usual
  start_date <- ymd("2011-03-01") %m+% months(2 * (t - 1))
  end_date <- ymd("2011-05-01") %m+% months(2 * (t - 1))
  test_current <- test %>% filter(Date >= start_date & Date < end_date) %>% select(-IsHoliday) %>% mutate(Wk = week(Date))
  
  if (t > 1){
    train <<- train %>% add_row(new_train)
  }  
  # train_dt <- train
  # test_dt <- test_current
  # setDT(train_dt)
  # setDT(test_dt)
  # find the unique pairs of (Store, Dept) combo that appeared in both training and test sets
  train_pairs <- train[, 1:2] %>% dplyr::count(Store, Dept) %>% filter(n != 0)
  test_pairs <- test[, 1:2] %>% dplyr::count(Store, Dept) %>% filter(n != 0)
  unique_pairs <- intersect(train_pairs[, 1:2], test_pairs[, 1:2])
  
  # train_dt <- dcast(train_dt, Date + Dept ~ Store, value.var="Weekly_Sales")
  # train_split <- split(train_dt, by="Dept")
  # print("_______________________________________")
  # print(paste("Fold:",t))
  # print("_______________________________________")
  # if (t>1) {
  #   print(head(train_split))
  # }
  
  # pick out the needed training samples, convert to dummy coding, then put them into a list
  train_split <- unique_pairs %>% 
    left_join(train, by = c('Store', 'Dept')) %>% 
    mutate(Wk = ifelse(year(Date) == 2010, week(Date) - 1, week(Date))) %>% 
    mutate(Yr = year(Date))
  #train_split = as_tibble(model.matrix(~ Weekly_Sales + Store + Dept + Yr + Wk, train_split)) %>% group_split(Store, Dept)
  train_split = as_tibble(model.matrix(~ Weekly_Sales + Store + Dept + Wk, train_split)) %>% group_split(Store, Dept)
  print(head(train_split))
  
  # # do the same for the test set
  # test_split <- unique_pairs %>% 
  #   left_join(test_current, by = c('Store', 'Dept')) %>% 
  #   mutate(Wk = ifelse(year(Date) == 2010, week(Date) - 1, week(Date))) %>% 
  #   mutate(Yr = year(Date))
  # #test_split = as_tibble(model.matrix(~ Store + Dept + Yr + Wk, test_split)) %>% mutate(Date = test_split$Date) %>% group_split(Store, Dept)  
  # test_split = as_tibble(model.matrix(~ Store + Dept + Wk, test_split)) %>% group_split(Store, Dept)  
  # print(head(test_split))

  # # perform regression for each split, note we used lm.fit instead of lm
  # for (i in 1:nrow(unique_pairs)) {
  #   tmp_train <- train_split[[i]]
  #   tmp_test <- test_split[[i]]
    
  #   mycoef <- lm.fit(as.matrix(tmp_train[, -(2:4)]), tmp_train$Weekly_Sales)$coefficients
  #   mycoef[is.na(mycoef)] <- 0
  #   tmp_pred <- mycoef[1] + as.matrix(tmp_test[, 4:55]) %*% mycoef[-1]
    
  #   test_pred[[i]] <- cbind(tmp_test[, 2:3], Date = tmp_test$Date, Weekly_Pred = tmp_pred[,1])
  # }
  
  # # turn the list into a table at once, this is much more efficient then keep concatenating small tables
  # test_pred <- bind_rows(test_pred)
  
  # return(test_pred)
}