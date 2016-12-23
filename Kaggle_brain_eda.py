# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 23:29:38 2016

@author: Andy
"""

import pandas as pd
import numpy as np

#clicks = pd.read_csv("D:\Projects\Kaggle\Brain\clicks_train.csv")
#events = pd.read_csv("D:\Projects\Kaggle\Brain\events.csv")
#
#events = events.drop('geo_location', 1)
#events = events.drop('uuid', 1)
#events = events.drop('timestamp', 1)
#
#events.to_csv("events_platform.csv", index=False)
#
#clicks = pd.read_csv("D:\Projects\Kaggle\Brain\clicks_train.csv")
#events = pd.read_csv("D:\Projects\Kaggle\Brain\Modified_csvs\events_platform.csv")
#
#merged = pd.merge(clicks, events, on='display_id')
#
#merged.clicked = merged.clicked.astype(bool)
#
#merged.to_csv("D:\Projects\Kaggle\Brain\Modified_csvs\platform_merged.csv", index=False)

merged = pd.read_csv("D:\Projects\Kaggle\Brain\Modified_csvs\platform_merged.csv")

#Value to make sure we un-emphasise ads that have little info
reg = 10

train = pd.read_csv("D:\Projects\Kaggle\Brain\clicks_train.csv")

cnt = train[train.clicked==1].ad_id.value_counts()
cntall = train.ad_id.value_counts()

#My computer only has so much memory
del train

#Gets a probability for each add simply based off how many times it is clicked/how many times it appears (plus extra penatly)
def get_prob(k):
    if k not in cnt:
        return 0
        #Maybe if we add another parameter we can add a little extra weight to adds that appear more
        #Or rather, a weight that emphasises adds that appear more than once
    #return cnt[k] + np.log(cnt[k])**2 / (float(cntall[k]) + reg)
    return cnt[k]/(float(cntall[k]) + reg)
    
#Lets try replacing ad_id with the actual probabilities to build a model
merged['add_id'] = merged.add_id.apply()

merged.to_csv("D:\Projects\Kaggle\Brain\Modified_csvs\merged_ad_id_2_prob.csv")

merged = pd.read_csv("D:\Projects\Kaggle\Brain\Modified_csvs\merged_ad_id_2_prob.csv")