library(mclust)
options(digits=8)
options()$digits
 
compute_gamma <- function(data, G, para) {
    prob <- para$prob
    mean <- para$mean
    Sigma <- para$Sigma
    Sigma_inv <-para$Sigma_inv

    # Compute ak

    ak <- rep(0,G)
    for (i in 1:G) {
        ak[i] <- log(prob[i]/prob[1]) + 
                    0.5 * t(data-mean[,1]) %*% Sigma_inv %*% (data-mean[,1]) - 
                    0.5 * t(data-mean[,i]) %*% Sigma_inv %*% (data-mean[,i])            
        }

    # Shift for numerical stability
    ak_new = ak-max(ak)    

    # Compute gamma_k, the posterior probability of Z_i = k
    gamma_k = exp(ak_new) / sum(exp(ak_new))    
    return(gamma_k)
}

Estep <- function(data, G, para) {    
    gamma_k <- t(apply(data, 1, compute_gamma, G, para))            
    return(gamma_k)
}

Mstep <- function(data, G, para, prob) {    
    n <- nrow(data)
    p <- ncol(data)
    # Update prob
    prob_new = colMeans(prob)
    # Update mean
    mu_new <- t(data) %*% prob %*% diag(1/colSums(prob))
    # Update Sigma
    Sigma_new <- matrix(0,p,p)
    for (i in 1:G) {
        y <- t(data) - mu_new[,i]
        Sigma_new <- Sigma_new + y %*% diag(prob[,i]) %*% t(y)
    }
    Sigma_new <- Sigma_new / n
    Sigma_inv <- solve(Sigma_new)

    para_new <- list(prob = prob_new,
                     mean = mu_new,
                     Sigma = Sigma_new,
                     Sigma_inv = Sigma_inv)    
    return(para_new)
}

myEM <- function(data, itmax, G, para) {
    
    para$Sigma_inv = solve(para$Sigma)    
    for (i in 1:itmax) {
        prob <- Estep(data, G, para)
        para <- Mstep(data, G, para, prob)
    }
    list.remove(para, "Sigma_inv")
    return(para)
}

# Obtain data
dim(faithful)


# Initialize
K <- 2
n <- nrow(faithful)
set.seed(234)  # replace 234 by the last 4-dig of your University ID
gID <- sample(1:K, n, replace = TRUE)
Z <- matrix(0, n, K)
for(k in 1:K)
  Z[gID == k, k] <- 1 
ini0 <- mstep(modelName="EEE", faithful , Z)$parameters
Sigma = ini0$variance$Sigma
Sigma_inv = solve(Sigma)

# Initial parameters
para0 <- list(prob = ini0$pro, 
              mean = ini0$mean, 
              Sigma = ini0$variance$Sigma,
              Sigma_inv = Sigma_inv)
print(para0)

# Compare
para = myEM(data=faithful, itmax=20, G=K, para=para0)
print(para)

Rout <- em(modelName = "EEE", data = faithful,
           control = emControl(eps=0, tol=0, itmax = 20), 
           parameters = ini0)$parameters
list(Rout$pro, Rout$mean, Rout$variance$Sigma)


# # Obtain data
dim(faithful)
K <- 3
set.seed(234)  # replace 234 by the last 4-dig of your University ID
gID <- sample(1:K, n, replace = TRUE)
Z <- matrix(0, n, K)
for(k in 1:K)
  Z[gID == k, k] <- 1 
ini0 <- mstep(modelName="EEE", faithful , Z)$parameters
Sigma = ini0$variance$Sigma
Sigma_inv = solve(Sigma)

para0 <- list(prob = ini0$pro, 
              mean = ini0$mean, 
              Sigma = ini0$variance$Sigma,
              Sigma_inv = Sigma_inv)
print(para0)

para = myEM(data=faithful, itmax=20, G=K, para=para0)
print(para)

Rout <- em(modelName = "EEE", data = faithful,
           control = emControl(eps=0, tol=0, itmax = 20), 
           parameters = ini0)$parameters
print(list(Rout$pro, Rout$mean, Rout$variance$Sigma))