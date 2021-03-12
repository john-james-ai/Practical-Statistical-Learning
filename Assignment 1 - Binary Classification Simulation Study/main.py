# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Practical Statistical Learning                                    #
# File    : \main.py                                                          #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/MCS/                             #
# --------------------------------------------------------------------------- #
# Created       : Wednesday, February 3rd 2021, 9:04:28 pm                    #
# Last Modified : Wednesday, February 3rd 2021, 9:04:28 pm                    #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
import numpy as np

from learner import Learner
# --------------------------------------------------------------------------- #
def main():
    """ Main driver"""    
    std_centers_list = [0.5,1,2,4,8]        
    n_gaussians = 4
    n_simulations = 20
    n_folds = 5
    n_train=10
    n_test=500
    #-----------------------------#
    learner = Learner(std_centers_list = std_centers_list, n_gaussians=n_gaussians,
                        n_simulations=n_simulations, n_folds=n_folds,
                        n_train=n_train, n_test=n_test)
    learner.run()
    learner.report()
     
if __name__ == "__main__":
    main()

#%%