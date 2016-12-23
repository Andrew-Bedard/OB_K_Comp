# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 20:50:10 2016

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

#Lets see if we can take the look at what time people check things in different time zones
canada = train.loc[train['geo_location'].str[:2].isin(['CA'])]
provinces = canada.geo_location.str[3:].unique()

plt.figure(figsize=(12,4))
canada.hour.hist(bins=np.linspace(-0.5, 23.5, 25), label="Non-corrected", alpha=0.7, normed=True)
plt.xlim(-0.5, 23.5)
plt.legend(loc="best")
plt.xlabel("Hour of Day")
plt.ylabel("Fraction of Events")

#Create column of provinces
canada["province"] = canada.geo_location.str[3:5]
#Get rid of geo_location because we have extracted all the relevant info
canada = canada.drop('geo_location',1)

#Dict that contains info on how many hours we have to offset UTC to compare behavior at local time
timezone_correction = {'BC': -8, 'AB': -7, 'SK': -6, 'MB':-6, 'ON': -5, 'QC':-5, 'NS': -4, 'NB':-4, 'PE':-4, 'YT':-8, 'NL':-4, 'NT':-7, 'NU':-5}

#Create column with corrected time, this can probably be changed to single line, but I don't know how
canada['timed'] = canada['province'].map(timezone_correction)

#Correct hour
canada['hour'] = (canada['hour'] + canada['timed']) % 24

#Drop column that I used to calculate corrected hour
canada = canada.drop('timed',1)

#lets take a look at what we got
plt.figure(figsize=(12,4))
canada.hour.hist(bins=np.linspace(-0.5, 23.5, 25), label="Time-zone corrected", alpha=0.7, normed=True)
plt.xlim(-0.5, 23.5)
plt.legend(loc="best")
plt.xlabel("Hour of Day")
plt.ylabel("Fraction of Events")

#Take a look at platform use by hour
plt.figure(figsize=(12,4))
canada.loc[canada.platform == 1].hour.hist(bins=np.linspace(-0.5, 23.5, 25), label="Desktop", alpha=0.7, normed=True)
canada.loc[canada.platform == 2].hour.hist(bins=np.linspace(-0.5, 23.5, 25), label="Mobile", alpha=0.5, normed=True)
canada.loc[canada.platform == 3].hour.hist(bins=np.linspace(-0.5, 23.5, 25), label="Tablet", alpha=0.4, normed=True)
plt.xlim(-0.5, 23.5)
plt.legend(loc="best")
plt.xlabel("Hour of Day")
plt.ylabel("Fraction of Events")

#Last plot included every part of event, thus we should remove duplicates from display_id so we only count every unique event
canada['display_id'] = canada.index
canada = canada.drop_duplicates(['display_id'])
canada = canada.drop('display_id', 1)

#Take a look at platform use by hour, but now only for unique events 
#(I dont think this changes anything, because if 8 adds are always shown, then the proportions remain the same)
plt.figure(figsize=(12,4))
canada.loc[canada.platform == 1].hour.hist(bins=np.linspace(-0.5, 23.5, 25), label="Desktop", alpha=0.7, normed=True)
canada.loc[canada.platform == 2].hour.hist(bins=np.linspace(-0.5, 23.5, 25), label="Mobile", alpha=0.5, normed=True)
canada.loc[canada.platform == 3].hour.hist(bins=np.linspace(-0.5, 23.5, 25), label="Tablet", alpha=0.4, normed=True)
plt.xlim(-0.5, 23.5)
plt.legend(loc="best")
plt.xlabel("Hour of Day")
plt.ylabel("Fraction of Events")

