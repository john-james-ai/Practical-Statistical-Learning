source("walmart.R")
library(lubridate)

# read in train / test dataframes
train <- readr::read_csv('data/train_ini.csv')
test <- readr::read_csv('data/test.csv')

# save weighted mean absolute error WMAE
num_folds <- 10
wae <- rep(0, num_folds)
models <- list(tslm.basic=tslm.basic)
mnames <- c("tslm.basic")
detail_length <- num_folds*length(models)
summary_length <- length(models)
detail <- list()
summary <- list()

for (m in 1:length(models)) {
  model_start = Sys.time()

  for (t in 1:num_folds) {
    fold_start = Sys.time()    
    # *** THIS IS YOUR PREDICTION FUNCTION ***
    test_pred <- mypredict(models[m])
    
    # load fold file 
    fold_file <- paste0('data/fold_', t, '.csv')
    new_train <- readr::read_csv(fold_file, 
                                col_types = cols())

    # extract predictions matching up to the current fold
    scoring_tbl <- new_train %>% 
        left_join(test_pred, by = c('Date', 'Store', 'Dept'))
    
    # compute WMAE
    actuals <- scoring_tbl$Weekly_Sales
    preds <- scoring_tbl$Weekly_Pred
    preds[is.na(preds)] <- 0
    weights <- if_else(scoring_tbl$IsHoliday, 5, 1)
    wae[t] <- sum(weights * abs(actuals - preds)) / sum(weights)
 
    # Store process metadata for reporting
    fold_end = Sys.time()
    fold_time = fold_end - fold_start
    fold_time = round(fold_time, digits=2)

    idx = m * t
    detail[idx] = list("Date" = now(), "Model"= mnames[m], "Fold"=t, "WAE"=wae[t], "Elapsed"=fold_time)    

    x <- sprintf("Model %s: Fold: %d  WAE: %f  Elapsed: %f", mnames[m], t, wae[t], fold_time)
    print(x)
  }  
  model_end = Sys.time()
  model_time = model_end - model_start
  model_time = round(model_time,digits=2)
  summary[m] = list("Date" = now(), "Model"= mnames[m], "Mean WAE"=mean(wae), "Elapsed"=model_time)    
  x <- sprintf("Model %s: Mean WAE: %f  Elapsed: %f", mnames[m], mean(wae), model_time)
  print(x)
}
write.csv(detail)
setDF(detail)
setDF(summary)
detail_filename = paste0("reports/",now(),"_detail.csv")
summary_filename = paste0("reports/",now(),"_summary.csv")
write.csv(detail,detail_filename)
write.csv(detail,summary_filename)
