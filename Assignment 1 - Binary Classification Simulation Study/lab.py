# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Practical Statistical Learning                                    #
# File    : \lab.py                                                           #
# Python  : 3.9.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : john.james@nov8.ai                                                #
# URL     : https://github.com/john-james-sf/MCS/                             #
# --------------------------------------------------------------------------- #
# Created       : Thursday, February 4th 2021, 3:05:51 am                     #
# Last Modified : Thursday, February 4th 2021, 3:05:51 am                     #
# Modified By   : John James (john.james@nov8.ai)                             #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2021 nov8.ai                                                  #
# =========================================================================== #
#%%
from collections import OrderedDict
d = OrderedDict()
d[1] = {'a':1, 'b':2}
d[2] = {'a':5, 'b':3}
d[3] = {'a':3, 'b':2}
print(d)

d = OrderedDict(sorted(d.items(), key=lambda key_value_pair: key_value_pair[1]['b']))
print(d)
