# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 20:07:41 2016

@author: Andy
"""

import pandas as pd
import warnings

warnings.filterwarnings('ignore')

page_views_sample_df = pd.read_csv("D:\Projects\Kaggle\Brain\page_views_sample.csv", usecols=['uuid', 'geo_location'])
# Drop NAs
page_views_sample_df.dropna(inplace=True)
# Drop EU code
page_views_sample_df = page_views_sample_df.loc[~page_views_sample_df.geo_location.isin(['EU', '--']), :]
# Drop duplicates
page_views_sample_df = page_views_sample_df.drop_duplicates('uuid', keep='first')

country = page_views_sample_df.copy()
country.columns = ['uuid', 'Country']
country.Country = country.Country.str[:2]
country.loc[:, 'UserCount'] = country.groupby('Country')['Country'].transform('count')
country = country.loc[:, ['Country', 'UserCount']].drop_duplicates('Country', keep='first')
country.sort_values('UserCount', ascending=False, inplace=True)
country.head(10)