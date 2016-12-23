# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:18:38 2016

@author: Andy
"""

import pandas as pd
import numpy as np 

reg = 10

train = pd.read_csv("D:\Projects\Kaggle\Brain\clicks_train.csv")

cnt = train[train.clicked==1].ad_id.value_counts()
cntall = train.ad_id.value_counts()

del train

def get_prob(k):
    if k not in cnt:
        return 0
        #Maybe if we add another parameter we can add a little extra weight to adds that appear more
        #Or rather, a weight that emphasises adds that appear more than once
    #return cnt[k] + np.log(cnt[k])**2 / (float(cntall[k]) + reg)
    return cnt[k]/(float(cntall[k]) + reg)

def srt(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
    return " ".join(map(str,ad_ids))


subm = pd.read_csv("D:\Projects\Kaggle\Brain\sample_submission.csv") 
#subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
#subm.to_csv("subm_mod_prob_reg10.csv", index=False)