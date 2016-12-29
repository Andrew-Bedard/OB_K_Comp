# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:08:25 2016

@author: Andy
"""

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#If submission == True, then outputs csv for submission
submission = False

#Read in datasets as dataframes
train = pd.read_csv('D:\Projects\Kaggle\Brain\clicks_train.csv', dtype = np.int32)

events = pd.read_csv('D:\Projects\Kaggle\Brain\events.csv')

#not going to train on uuid so drop that nonsense
events = events.drop('uuid', 1)

#EDA shows there is some funny stuff happening with platform (events.platform.unique()) so lets make sure we only have
#proper values
events.loc[events['platform'] == '3'] = 3
events.loc[events['platform'] == '2'] = 2
events.loc[events['platform'] == '1'] = 1
events.loc[events['platform'] == '\\N'] = np.NaN

events = events.dropna(axis = 0)

#save some memory
events.display_id = events.display_id.astype(np.int32)
events.document_id = events.document_id.astype(np.int32)

#Get rid of info excluding country from geo_location
events.geo_location = events.geo_location.str[:2]

#In order to use the country info in the model, lets replace everything outside the top ten to OT (as in other) maybe in the future
#It would make sense to set them equal to continent
largest_list = events.geo_location.value_counts()[:10]
events.loc[~events['geo_location'].isin(largest_list.index), 'geo_location'] = 'OT'

#Convert timestamp to hour and day of the week
events["hour"] = (events.timestamp // (3600 * 1000)).astype(np.int32) % 24
events["day"] = (events.timestamp // (3600 * 24 * 1000)).astype(np.int32) % 7

#We can now drop timestamp
events = events.drop('timestamp', 1)

#Lets merge events and train
train = train.merge(events, on='display_id')

#Dont need events dataframe anymore
del events

#There is no way that I'm going to be able to train on such a large set on my computer, lets sample out some stuff
sampled_train = train.sample(frac = 0.05, replace = True)
sampled_train2 = train.sample(frac = 0.05, replace = True)

del train

#!!!!!!!!!!!!!!!!!!!!!!!SANITY CHECK:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
#make sure the distributions of things are the same

#plt.figure(figsize=(12,4))
#sampled_train.hour.hist(bins=np.linspace(-0.5, 23.5, 25), label="sampled_train", alpha=0.7, normed=True)
#train.hour.hist(bins=np.linspace(-0.5, 23.5, 25), label="train", alpha=0.5, normed=True)
#plt.xlim(-0.5, 23.5)
#plt.legend(loc="best")
#plt.xlabel("Hour of Day")
#plt.ylabel("Fraction of Events")

#sklearn doesn't like categories from pandas, so instead convert geo_location to integers based on country
sampled_train.geo_location = sampled_train.geo_location.astype('category')
sampled_train.geo_location = sampled_train['geo_location'].cat.codes

sampled_train2.geo_location = sampled_train2.geo_location.astype('category')
sampled_train2.geo_location = sampled_train2['geo_location'].cat.codes

#Lets try a random forest on this guy
log_reg = sk.linear_model.LogisticRegression()
#forest = RandomForestClassifier(n_estimators = 100)

col_obvs = ['ad_id', 'document_id', 'platform', 'geo_location', 'hour', 'day']
col_res = ['clicked']

log_reg = log_reg.fit(sampled_train.as_matrix(col_obvs), sampled_train.as_matrix(col_res))
#forest = forest.fit(sampled_train.as_matrix(col_obvs), sampled_train.as_matrix(col_res))

in_score = log_reg.score(sampled_train.as_matrix(col_obvs), sampled_train.as_matrix(col_res))
#in_score = forest.score(sampled_train.as_matrix(col_obvs), sampled_train.as_matrix(col_res))

del sampled_train

out_score = log_reg.score(sampled_train2.as_matrix(col_obvs), sampled_train2.as_matrix(col_res))

if submission == True:
    subm = pd.read_csv("D:\Projects\Kaggle\Brain\sample_submission.csv")
    subm.to_csv("subm_mod_prob_reg10.csv", index=False)