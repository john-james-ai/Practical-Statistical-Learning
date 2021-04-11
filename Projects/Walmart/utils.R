library(data.table)

save_results <- function(x,model,fold, actual=FALSE) {
    maindir <- "results/"
    subdir <- model
    path <- paste0(maindir, subdir)
    dir.create(path, showWarnings = FALSE)
    if (actual==TRUE) {
        filename <- paste0("/fold_",fold,"_actual.csv")
    } else {
        filename <- paste0("/fold_",fold,"_predicted.csv")
    }
    filepath <- paste0(path,filename)
    write.csv(x, filepath, row.names=FALSE)
}

gather_results <- function(model) {
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

inspect <- function(x, summary=FALSE) {
    print(dim(x))
    print(head(x))
    if (summary == TRUE) {
        print(summary(x))
    }
}