# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:18:38 2016

@author: Andy
"""

import pandas as pd
import numpy as np 

reg = 10

#Load training data
train = pd.read_csv("D:\Projects\Kaggle\Brain\clicks_train.csv")

#Get count of all adds that have been clicked
cnt = train[train.clicked==1].ad_id.value_counts()
#Get count of all adds that appear
cntall = train.ad_id.value_counts()

#del train

#If an add has been clicked a nonzeno number of times, find the probability it will was clicked, with an added 'fudge factor'
#to de-emphasise adds that appear a very small number of times
def get_prob(k):
    if k not in cnt:
        return 0
        #Maybe if we add another parameter we can add a little extra weight to adds that appear more
        #Or rather, a weight that emphasises adds that appear more than once
    #return cnt[k] + np.log(cnt[k])**2 / (float(cntall[k]) + reg)
    return cnt[k]/(float(cntall[k]) + reg)

#Actually create a dataframe with the ad probabilities, note that the index is the ad_id
ad_prob = pd.DataFrame(cntall.apply(lambda x: get_prob(x)))
#Switch things around so that the ad_prob is actually the probability, right now it is still called ad_id
ad_prob['ad_prob'] = ad_prob['ad_id']
ad_prob['ad_id'] = ad_prob.index

#Merge ad_prob with training data so we now have all probabilities on tap for training
train = pd.merge(train, ad_prob, left_on=['ad_id'], right_on=['ad_id'], how = 'inner')

#Lets recast datatypes so that we can save some memory
train['display_id'] = train['display_id'].astype(np.int32)
train['ad_id'] = train['ad_id'].astype(np.int32)
train['clicked'] = train['clicked'].astype(np.int8)
train['ad_prob'] = train['ad_prob'].astype(np.float16)

#Lets save this badboy so we dont have to keep doing this
train.to_csv("train_w_prob.csv", index = True)
#def srt(x):
#    ad_ids = map(int, x.split())
#    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
#    return " ".join(map(str,ad_ids))
#
#
#subm = pd.read_csv("D:\Projects\Kaggle\Brain\sample_submission.csv") 
#subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
#subm.to_csv("subm_mod_prob_reg10.csv", index=False)