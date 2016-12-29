# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 14:09:05 2016

@author: Andy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
from datetime import datetime, timedelta
from pytz import timezone

#Load events
events = pd.read_csv("D:\Projects\Kaggle\Brain\events.csv", index_col=0)

#Change datatype to save some memory
events.document_id = events.document_id.astype(np.int32)

#Import training clicks data and merge with events
train = pd.merge(pd.read_csv("D:\Projects\Kaggle\Brain\clicks_train.csv", dtype=np.int32, index_col=0),
                 events, left_index=True, right_index=True)

#Now that we have merged we can get rid of events to save on some memory
del events

#Lets check out the hour and the day people be doing things
train["hour"] = (train.timestamp // (3600 * 1000)) % 24
train["day"] = train.timestamp // (3600 * 24 * 1000)

#Drop geo_location info except for country code
train.geo_location = train.geo_location.str[:2]

#Get the names of the top 5 countries from geo_location
cntrys = train.geo_location.value_counts()[:5]

#Remove all entries not in the top 5 most common countries
train = train.loc[train['geo_location'].isin(cntrys.index)]
                  
plt.figure(figsize=(12,4))
train.loc[train['platform'].isin([1])].geo_location.value_counts().hist(bins = 5, label="Desktop", alpha = 0.7, normed = True)
train.loc[train['platform'].isin([2])].geo_location.value_counts().hist(bins = 5, label="Phone", alpha = 0.7, normed = True)
train.loc[train['platform'].isin([3])].geo_location.value_counts().hist(bins = 5, label="Tablet", alpha = 0.7, normed = True)
plt.xlim(1, 5)
plt.legend(loc="best")
plt.xlabel("Platform")
plt.ylabel("Fraction of users")

plt.figure(figsize=(12,4))
train.loc[train['platform'].isin([1])].geo_location.value_counts().plot(kind = 'bar', label="Dekstop", alpha = 0.6, normed=True)
train.loc[train['platform'].isin([2])].geo_location.value_counts().plot(kind = 'bar', label="Dekstop", alpha = 0.6, normed=True)
train.loc[train['platform'].isin([3])].geo_location.value_counts().plot(kind = 'bar', label="Dekstop", alpha = 0.6, normed=True)
plt.legend(loc="best")
plt.xlabel("Platform")
plt.ylabel("Fraction of users")