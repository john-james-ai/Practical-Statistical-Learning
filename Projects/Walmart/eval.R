source("mymain.R")
library(lubridate)

# Parameters for models.
m0 <- list(model="regression", data='r', adjust=FALSE, model.type=NULL, n.comp=NULL)
m1 <- list(model="regression", data='r', adjust=TRUE, model.type=NULL, n.comp=NULL)
m2 <- list(model="snaive", data='t', adjust=TRUE, model.type=NULL, n.comp=NULL)
m3 <- list(model="snaive.custom", data='t', adjust=TRUE, model.type=NULL, n.comp=NULL)
m4 <- list(model="stlf", data='t', adjust=TRUE, model.type="ets", n.comp=NULL)
m5 <- list(model="stlf", data='t', adjust=TRUE, model.type="arima", n.comp=NULL)
m6 <- list(model="stlf.svd", data='t', adjust=TRUE, model.type="ets", n.comp=12)
m7 <- list(model="stlf.svd", data='t', adjust=TRUE, model.type="arima", n.comp=12)
m8 <- list(model="tslm", data='t', adjust=TRUE, model.type=NULL, n.comp=NULL)
m9 <- list(model="tslm.svd", data='t', adjust=TRUE, model.type="ets", n.comp=12)
m10 <- list(model="tslm.svd", data='t', adjust=TRUE, model.type="arima", n.comp=12)

params=NULL
params <- m0
# read in train / test dataframes
train <- readr::read_csv('data/train_ini.csv')
test <- readr::read_csv('data/test.csv')

# save weighted mean absolute error WMAE
num_folds <- 10
wae <- rep(0, num_folds)

# Create lists for metadata
folds <- list()
scores <- list()
times <- list()

model_start <- Sys.time()

for (t in 1:num_folds) {
  fold_start <- Sys.time()    
  # *** THIS IS YOUR PREDICTION FUNCTION ***  
  test_pred <- mypredict(params)
  
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
  fold_end <- Sys.time()
  fold_time <- fold_end - fold_start
  fold_time <- round(fold_time, digits=2)
  # Create metadata
  folds[[t]] <- t
  scores[[t]] <- wae[t]
  times[[t]] <- fold_time    

  x <- sprintf("Model %s: Fold: %d  WAE: %f  Elapsed: %f", toupper(params$model), t, wae[t], fold_time)
  print(x)
}
# Save Model Statistics
stats <- list("Fold"=folds, "WAE"=scores, "Time Elapsed"=times)
save_stats(stats, params)
# Report model summary
model_end <- Sys.time()
model_time <- model_end - model_start
model_time <- round(model_time,digits=2)
summary.stats <- list(params=params, score=mean(wae), time=model_time)
save_summary_stats(summary.stats, params)
x <- sprintf("Model %s: Mean WAE: %f  Elapsed: %f", toupper(params$model), mean(wae), model_time)
print(x)


