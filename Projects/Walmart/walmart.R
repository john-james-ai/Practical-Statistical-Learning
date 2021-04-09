library(lubridate)
library(tidyverse)
library(glmnet)

# lasso_model = function(train_reg, test_reg) {
#     X = as.matrix(train_reg[,-(2:4)])
#     y = as.matrix(train_reg$Weekly_Sales)
#     X_test = as.matrix(test_reg)
#     # Lambdas to attempt
#     lambdas <- 10^seq(1,-3,by=-.1)

#     fit = glmnet(X,y, alpha=0, nlambdas=20)
#     y_pred = predict(fit, newx=X_test)

#     mycoef <- lm.fit(as.matrix(tmp_train[, -(2:4)]), tmp_train$Weekly_Sales)$coefficients
#     mycoef[is.na(mycoef)] <- 0
#     tmp_pred <- mycoef[1] + as.matrix(tmp_test[, 4:55]) %*% mycoef[-1]
#}

ridge = function(tmp_train, tmp_test) {
  meta_data = as.matrix(tmp_train[,(2:4)])
  X = as.matrix(tmp_train[, -(2:4)])
  # print(dim(X))
  # print(head(X))
  y = as.matrix(tmp_train$Weekly_Sales)
  #print(nrow(y))
  X_test = as.matrix(tmp_test[, 4:55])    
  icept = rep(1,nrow(X_test))
  X_test = cbind(icept,X_test)
  # print(dim(X_test))
  # print(head(X_test))  
  if (nrow(y)!= 1 && length(unique(y))>1) {
    fit = glmnet(X, y, alpha=1, nlambda=20,standardize=FALSE, family='gaussian')
    y_pred = predict(fit, newx=X_test, type='response', s=0.5)
  } else {
    print(nrow(X_test))
    print(y)
    y_pred = rep(y,nrow(X_test))
  }
  
  return(y_pred)
}
regression = function(tmp_train, tmp_test) {
  mycoef <- lm.fit(as.matrix(tmp_train[, -(2:4)]), tmp_train$Weekly_Sales)$coefficients
  mycoef[is.na(mycoef)] <- 0
  tmp_pred <- mycoef[1] + as.matrix(tmp_test[, 4:55]) %*% mycoef[-1]  
  return(tmp_pred)
}

predictor <- ridge

mypredict = function(){
  # as usual
  start_date <- ymd("2011-03-01") %m+% months(2 * (t - 1))
  end_date <- ymd("2011-05-01") %m+% months(2 * (t - 1))
  test_current <- test %>% filter(Date >= start_date & Date < end_date) %>% select(-IsHoliday) %>% mutate(Wk = week(Date))
  
  if (t > 1){
    train <<- train %>% add_row(new_train)
  }
  
  # find the unique pairs of (Store, Dept) combo that appeared in both training and test sets
  train_pairs <- train[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  test_pairs <- test_current[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  unique_pairs <- intersect(train_pairs[, 1:2], test_pairs[, 1:2])

  print("There are ")
  
  # pick out the needed training samples, convert to dummy coding, then put them into a list
  train_split <- unique_pairs %>% 
    left_join(train, by = c('Store', 'Dept')) %>% 
    mutate(Wk = factor(ifelse(year(Date) == 2010, week(Date) - 1, week(Date)), levels = 1:52)) %>% 
    mutate(Yr = year(Date))
  train_split = as_tibble(model.matrix(~ Weekly_Sales + Store + Dept + Yr + Wk, train_split)) %>% group_split(Store, Dept)
  
  # do the same for the test set
  test_split <- unique_pairs %>% 
    left_join(test_current, by = c('Store', 'Dept')) %>% 
    mutate(Wk = factor(ifelse(year(Date) == 2010, week(Date) - 1, week(Date)), levels = 1:52)) %>% 
    mutate(Yr = year(Date))
  test_split = as_tibble(model.matrix(~ Store + Dept + Yr + Wk, test_split)) %>% mutate(Date = test_split$Date) %>% group_split(Store, Dept)

  # pre-allocate a list to store the predictions
  test_pred <- vector(mode = "list", length = nrow(unique_pairs))
  
  # perform regression for each split, note we used lm.fit instead of lm
  for (i in 1:nrow(unique_pairs)) {
    tmp_train <- train_split[[i]]
    tmp_test <- test_split[[i]]
    
    # mycoef <- lm.fit(as.matrix(tmp_train[, -(2:4)]), tmp_train$Weekly_Sales)$coefficients
    # mycoef[is.na(mycoef)] <- 0
    # tmp_pred <- mycoef[1] + as.matrix(tmp_test[, 4:55]) %*% mycoef[-1]
    tmp_pred = predictor(tmp_train, tmp_test)
    test_pred[[i]] <- cbind(tmp_test[, 2:3], Date = tmp_test$Date, Weekly_Pred = tmp_pred[,1])
  }
  
  # turn the list into a table at once, this is much more efficient then keep concatenating small tables
  test_pred <- bind_rows(test_pred)
  
  return(test_pred)
}