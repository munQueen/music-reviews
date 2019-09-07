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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import category_encoders as ce

import statsmodels.api as sm
from statsmodels.formula.api import ols

os.chdir('C:\Learning\ML\spotify')
full_data = pd.read_csv('full_data.csv').dropna()

#create subset1 and subset2


X = full_data.drop(['review_id', 'album_uri_to_use', 'score', 'artist_name', 'album_name', 'publication_year', 'release_year', 'missing'], axis=1)
regression_columns = ['danceability_mean', 'energy_mean', 'loudness_mean', 'speechiness_mean', 'acousticness_mean', 
                      'instrumentalness_mean', 'liveness_mean', 'valence_mean', 'tempo_mean', 'duration_ms_mean', 'key_max_pct']
full_data['primary_time'] = full_data['primary_time'].apply(str)
ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
categorical_columns = full_data[['primary_key', 'primary_mode', 'primary_time']]
categorical_data = ohe.fit_transform(categorical_columns)





numeric_features = full_data[regression_columns].dropna()
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)
numeric_scaled = pd.DataFrame(numeric_scaled)
numeric_scaled.columns = regression_columns

X = pd.concat([numeric_scaled.reset_index(), categorical_data.reset_index()], axis=1)
y = list(full_data.score)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)

#reg = LinearRegression().fit(X, y)

X_train = sm.add_constant(X_train)
reg = sm.OLS(y_train, X_train).fit()
reg.summary()




data = {'fitted_values':reg.fittedvalues, 'residuals':reg.resid}
residuals = pd.DataFrame(data)
subset = residuals.sample(250)
plt.scatter(subset.fitted_values, subset.residuals)


#testing for multicol:
X3 = X.drop(['loudness_mean'], axis=1)
X3 = X3.drop(['valence_mean'], axis=1)
reg = sm.OLS(y, X3).fit()
reg.summary()





reg = sm.OLS(y, X2).fit()
reg.summary()


data = {'fitted_values':reg.fittedvalues, 'true_values':full_data.score}
residuals = pd.DataFrame(data)
subset = residuals.sample(250)
plt.scatter(subset.fitted_values, subset.true_values)