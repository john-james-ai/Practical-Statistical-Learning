library(mclust)
library(reticulate)
use_virtualenv("r-reticulate")
source_python("Assignment_4_6998_netID.py")

# Basic exploration
dim(faithful)
head(faithful)
n <- nrow(faithful)

# Two Clusters
K <- 2
set.seed(6998)
gID <- sample(1:K, n, replace = TRUE)
Z <- matrix(0,n,K)
for (k in 1:K)
    Z[gID == k, k] <- 1
ini0 <- mstep(modelName="EEE", faithful, Z)$parameters

# Set initial values of parametesr
para0 <- list(prob = ini0$pro,
              mean = ini0$mean,
              Sigma = ini0$variance$Sigma
)
print(para0)
