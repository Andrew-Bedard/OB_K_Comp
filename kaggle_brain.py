# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:08:25 2016

@author: Andy
"""

import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns

p = sns.color_palette()

#Read in datasets as dataframes
#df_train = pd.read_csv('D:\Projects\Kaggle\Brain\clicks_train.csv')

events = pd.read_csv('D:\Projects\Kaggle\Brain\events.csv')

#Look at some stuff
events.head()

#Check out platforms
plat = events.platform.value_counts()
print(plat)
print('\nUnique values of platform:', events.platform.unique())

#Replace weird \\N values with NAN for dropping rows
events = events.platform.replace('\\N', np.nan, regex = True)

#Drop Rows with NAN
#events = events.dropna()