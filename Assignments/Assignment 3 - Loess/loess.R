lo.lev <- function(x1, sp){
  
  ## YOUR CODE: compute the diagonal entries of 
  ##            the smoother matrix S
  
}

onestep_CV <- function(x1, y1, sp){
  
  ## YOUR CODE: 
  ## 1) fit a loess model y1 ~ x1 with span = sp, and extract 
  ##    the corresponding residual vector
  ## 2) call lo.lev to obtain the diagonal entries of S
  ## 3) compute LOO-CV and GCV
  
  return(list(cv = cv, gcv = gcv))
}

myCV <- function(x1, y1, span){
  ## x1, y1: two vectors
  ## span: a sequence of values for "span"
  
  m = length(span)
  cv = rep(0, m)
  gcv = rep(0, m)
  for(i in 1:m){
    tmp = onestep_CV(x1, y1, span[i])
    cv[i] = tmp$cv
    gcv[i] = tmp$gcv
  }
  return(list(cv = cv, gcv = gcv))
}
