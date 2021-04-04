## Introduction

This simulation study is an extension of the Nearest Neighbor Methods simulation study from chapter 2 of the Elements of Statistical Learning. The objective of this study is to evaluate the performance of the following algorithms in a binary classification context.

- Linear Regression
- Quadratic Regression
- k-Nearest Neighbors
- Bayes Rule
- Logistic Regression

The logistic regression algorithm serves as a baseline by which the above methods will be evaluated. 

### Data

The training data $X∈R2$ and $Y=(0,1)$ was generated from a mixture of 10 bivariate Gaussian distributions, the density function given by:

$$\frac{1}{10}\displaystyle\sum_{i=1}^{10}\bigg(\frac{1}{\sqrt{2\pi s^2}}\bigg)^2 e^{-\|x-m_{kl}\|^2/(2s^2)}$$

Components are uncorrelated with different means i.e.

$$X|Y=k,Z=l∼N(m_{kl},s^2I_2),X|Y=k,Z=l∼N(m_{kl},s^2I_2),$$

- where $k=0,1,l=1:10,P(Y=k)=1/2,\text{and }P(Z=1)=1/10.$

The two-dimensional Gaussian centers for $l=1,...,10$ are distributed as follows:

$$m_{0l}\text{ i.i.d.} \sim \mathcal{N}((0,1)^T,\sigma^2 I_2)$$ 

$$m_{1l}\text{ i.i.d.} \sim \mathcal{N}((1,0)^T,\sigma^2 I_2)$$ 

