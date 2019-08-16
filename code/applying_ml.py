# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 19:04:48 2019

@author: jwulz
"""

import pandas as pd
import numpy as np
import os
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


import statsmodels.api as sm
from statsmodels.formula.api import ols

os.chdir('C:\ML\Springboard\music-reviews-master')
full_data = pd.read_csv('full_data.csv').dropna()


X = full_data.drop(['review_id', 'album_uri_to_use', 'score', 'artist_name', 'album_name', 'publication_year', 'release_year', 'missing'], axis=1)
regression_columns = ['danceability_mean', 'energy_mean', 'loudness_mean', 'speechiness_mean', 'acousticness_mean', 
                      'instrumentalness_mean', 'liveness_mean', 'valence_mean', 'tempo_mean', 'duration_ms_mean', 'key_max_pct']
X = full_data[regression_columns].dropna()
X2 = StandardScaler().fit_transform(X)
y = full_data.score


#reg = LinearRegression().fit(X, y)

reg = sm.OLS(y, X).fit()
reg.summary()


data = {'fitted_values':reg.fittedvalues, 'true_values':full_data.score}
residuals = pd.DataFrame(data)
subset = residuals.sample(250)
plt.scatter(subset.fitted_values, subset.true_values)


