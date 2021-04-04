# Coordinate Descent with Lasso Regression

## Lasso Penalized Regression

Lasso penalized regression is a model selection technique, often used when the number of predictors $p$ exceeds the number of observations $n$. By suppressing the magnitudes of irrelevant predictors to zero, lasso enforces sparsity, effectively reducing the number of predictors.

Let $y_i$ be the target for the $i^{th}$ observation, $x_{i,j}$ be the $j^{th}$ feature from the $i^{th}$ observation and $\beta_j$ be the coefficient corresponding to  $x_{i,j}$ . Ignoring the intercept $\beta_0$, the lasso penalized regression can be expressed as minimizing the following objective function:

$$\underset{\beta_j}{min}\displaystyle\sum_{i=1}^n (y_i-\displaystyle\sum_{k\ne j} x_{ik}\hat{\beta_k}-x_{ij}\beta_j)^2+\lambda\displaystyle\sum_{k \ne j}|\hat{\beta_k}|+\lambda|\beta_j|,$$

where we seek to find a $\hat{\beta_j}\approx\beta_j\forall j\in\{1..p\}$ and $\beta_j$ is the unknown true model parameter. Since we are minimizing with respect to $\beta_j$, we can ignore the $\lambda\displaystyle\sum_{k \ne j}|\hat{\beta_k}|$ express. Further, we can let:

$$r_i = y_i-\displaystyle\sum_{k\ne j} x_{ik}\hat{\beta_k}.$$

Now the objective function minimizes to:

$$\underset{\beta_j}{min}\displaystyle\sum_{i=1}^n(r_i-x_{ij}\beta_j)^2+\lambda|\beta_j|.$$

## Coordinate Descent

The basic coordinate descent (CD) framework updates and adjusts a single parameter while holding all other parameters fixed and repeats until a termination condition is met. More precisely:

### Coordinate Descent Algorithm

1. Set $k$ = 0
2. repeat
   1. Select an index $i_k\in \{1,2,...,n\};
   2. Update x_i_k to x^k_i_k b depending upon x^{k-1};
   3. Keep x_j unchanged, i.e., x_j^k=x_j^{k-1}, $\forall j\ne k$;
   4. Let $k=k+1$
3. until termination condition is satisfied.

Intuitively, CD methods can be visualized (in the 2 dimensional case) as a moving along a grid, one component per iteration, down the contours of the objective function.



