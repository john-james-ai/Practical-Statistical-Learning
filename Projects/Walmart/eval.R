source("model_2.R")
library(lubridate)

# Model information
mname <- 'InConstruction'
mypredict <- model_2

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
  test_pred <- mypredict()
  
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

  x <- sprintf("Model %s: Fold: %d  WAE: %f  Elapsed: %f", mname, t, wae[t], fold_time)
  print(x)
}
# Save Model Statistics
stats <- list("Fold"=folds, "WAE"=scores, "Time Elapsed"=times)
df <- as.data.frame(matrix(unlist(stats), nrow=length(unlist(stats[1]))))
names(df) <- c("Fold", "WAE", "Time Elapsed")
detail_filename <- paste0("reports/",mname,"_detail.csv")
write.csv(df,detail_filename, row.names=FALSE)
# Report model summary
model_end <- Sys.time()
model_time <- model_end - model_start
model_time <- round(model_time,digits=2)
x <- sprintf("Model %s: Mean WAE: %f  Elapsed: %f", mname, mean(wae), model_time)
print(x)


