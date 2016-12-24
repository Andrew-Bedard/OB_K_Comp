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

plt.figure(figsize=(12,4))
train.hour.hist(bins=np.linspace(-0.5, 23.5, 25), label="train", alpha=0.7, normed=True)
plt.xlim(-0.5, 23.5)
plt.legend(loc="best")
plt.xlabel("Hour of Day")
plt.ylabel("Fraction of Events")

#Drop geo_location info except for country code
train.geo_location = train.geo_location.str[:2]

#Get the names of the top 5 countries from geo_location
cntrys = train.geo_location.value_counts()[:5]

#Remove all entries not in the top 5 most common countries
train = train.loc[train['geo_location'].isin(cntrys.index)]