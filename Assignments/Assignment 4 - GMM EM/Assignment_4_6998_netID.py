# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Walmart Sales Prediction Model                                    #
# File    : \Assignment_4_6998_netID.py                                       #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Course  : Practical Statistical Learning (Spring '21)                       #
# Email   : jtjames2@illinois.edu                                             #
# URL     : https://github.com/john-james-sf/Practical-Statistical-Learning   #
# --------------------------------------------------------------------------- #
# Created       : Sunday, April 4th 2021, 4:45:13 pm                          #
# Last Modified : Sunday, April 4th 2021, 4:45:13 pm                          #
# Modified By   : John James (jtjames2@illinois.edu)                          #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
import numpy as np
from numpy.random import default_rng
import pandas as pd
# --------------------------------------------------------------------------- #
def compute_Ak(data, G, para):
    """Computes the log probability of the components"""            
    print("Computing_AK")
    pk = para["p"]
    mu = para["mu"]
    Sigma_inv = para["Sigma_inv"] 
    print(f"     Data Shape is {data.shape}")
    d = {"pk":pk, "mu":mu, "Signa_inv": Sigma_inv}
    print(f"     Parameters in ComputeAK: {d}")

    ak = np.zeros(G)

    for i in range(G):
        ak[i] = np.log(pk[i] / pk[0]) +\
            0.5 * ((data-mu[:,0]).T.dot(Sigma_inv)).dot(data-mu[:,0]) - \
            0.5 * ((data-mu[:,i]).T.dot(Sigma_inv)).dot(data-mu[:,i]) 
    print(f"     ak: {ak}")
    return ak
# --------------------------------------------------------------------------- #
def compute_Bk(ak):
    ak_new = ak-np.amax(ak,axis=0)
    bk = np.exp(ak_new) / np.sum(np.exp(ak_new))
    return bk

# --------------------------------------------------------------------------- #
def Estep(data, G, para):
    """Computes the Gaussian probability density, given mean and covariance."""
    print(f"  E-Step")
    print(f"    Data shape: {data.shape}")
    print(f"    parameters: {para}")
    # Compute inverse of shared covariance 
    
    # Compute Ak, the log probability of the components    
    ak = data.apply(compute_Ak, args=(G,para), axis=1, result_type="broadcast")
    print(f"    ak shape: {ak.shape}")
    # Compute Bk, the posterior probabilities p

    bk = np.apply_along_axis(compute_Bk,1,ak)
    print(f"    bk shape: {bk.shape}")
    return bk
# --------------------------------------------------------------------------- #
def Mstep(data, G, para, post_prob):
    """Computes the updated parameters."""
    print(f"  M-Step")
    print(f"     Data shape: {data.shape}")
    print(f"      Parameters: {para}")
    n = data.shape[0]
    p = data.shape[1]
    
    # New posterior probabilities
    new_prob = np.mean(post_prob, axis=0)
    print(f"              New P: {new_prob}")
    print(f"    Post Prob shape: {post_prob.shape}")
    print(f"          Post Prob: {post_prob[0:5,:]}")
    # New means
    new_mean = data.T.dot(post_prob) / np.sum(post_prob,axis=0)
    print(f"\n          New Mean: {new_mean}")
    exit()
    # New shared sigma
    new_Sigma = np.zeros((p,p))
    for i in range(G):
        print(new_mean[i])
        y = data.T - new_mean[i][0]
        print(f"    y shape: {y.shape}")
        print(f"           {np.diag(post_prob[:,i])}")
        new_Sigma = new_Sigma + y.dot(np.diag(post_prob[:,i])).dot(y.T)
    new_Sigma /= n    
    Sigma_inv = np.linalg.inv(new_Sigma)    
    # Pack up parameters
    para["p"] = new_prob
    para["mu"] = new_mean
    para["Sigma"] = new_Sigma
    para["Sigma_inv"] = Sigma_inv

    return para
# --------------------------------------------------------------------------- #
def myEM(data, itmax, G, para):
    """Computes the maximum likelihood parameters given incomplete data."""
    for i in range(itmax):
        print(f"Iteration: {i}")
        post_prob = Estep(data,G,para)
        para = Mstep(data, G,para, post_prob)
        print(f"      New para: {para}")

    return para

# --------------------------------------------------------------------------- #
def main():
    rng = default_rng(6998)
    data = pd.read_csv("faithful.csv")
    n = data.shape[0]
    p = data.shape[1]
    K = 2
    ks = np.arange(K)     
    print(ks)
    print(data.head())
    gID = rng.choice(ks,n,replace=True)
    # Create latent variable
    Z = np.zeros((n,K))
    for k in range(K):
        Z[gID==k, k] = 1
    # Run single iteration
    para0 = {}
    para0["p"] = np.array([0.5,0.5])
    para0["mu"] = np.array([[3.467750,3.5078162], [70.132353,71.6617647]])
    para0["Sigma"] = np.array([[1.2975376, 13.911099], [13.9110994, 183.559040]])    
    para0["Sigma_inv"] = np.linalg.inv(para0["Sigma"])    
    print(para0)
    print(f"Shape of mu is {para0['mu'].shape}")
    print(myEM(data, itmax=20, G=K, para=para0))

if __name__ == "__main__":
    main()
#%%
