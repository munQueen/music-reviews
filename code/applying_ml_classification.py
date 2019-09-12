# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 00:47:43 2019

@author: jwulz
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, classification_report 


os.chdir('C:\Learning\ML\spotify')
full_data = pd.read_csv('full_data.csv').dropna()

#remove all albums which are given multiple genres 
single_genre_albums = full_data['rap'].astype(int) + full_data['electronic'].astype(int) +\
   full_data['experimental'].astype(int) + full_data['folk/country'].astype(int) +\
   full_data['global'].astype(int) + full_data['jazz'].astype(int) +\
   full_data['metal'].astype(int) + full_data['pop/r&b'].astype(int) +\
   full_data['rock'].astype(int) == 1

data = full_data[single_genre_albums]
data['genre'] = ''
data.loc[data['rap']== True, 'genre'] = 'rap'
data.loc[data['electronic']== True, 'genre'] = 'electronic'
data.loc[data['experimental']== True, 'genre'] = 'experimental'
data.loc[data['folk/country']== True, 'genre'] = 'folk/country'
data.loc[data['global']== True, 'genre'] = 'global'
data.loc[data['jazz']== True, 'genre'] = 'jazz'
data.loc[data['metal']== True, 'genre'] = 'metal'
data.loc[data['pop/r&b']== True, 'genre'] = 'pop/r&b'
data.loc[data['rock']== True, 'genre'] = 'rock'
data.genre.value_counts()


#set up all of the variables 
regression_columns = ['danceability_mean', 'energy_mean', 'loudness_mean', 'speechiness_mean', 'acousticness_mean', 
                      'instrumentalness_mean', 'liveness_mean', 'valence_mean', 'tempo_mean', 
                      'danceability_std', 'energy_std', 'loudness_std', 'speechiness_std', 'acousticness_std', 
                      'instrumentalness_std', 'liveness_std', 'valence_std', 'tempo_std', 'duration_ms_mean', 'key_max_pct']


data['primary_time'] = data['primary_time'].apply(str)
ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
categorical_columns = data[['primary_key', 'primary_mode', 'primary_time']]
#prepare the categorical variables
ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)

categorical_data = ohe.fit_transform(categorical_columns)
#for linear regression: drop 1 of each category
categorical_data = categorical_data.drop(columns=['primary_key_C', 'primary_mode_Major', 'primary_time_4.0'])

#prepare the numeric variables
numeric_features = data[regression_columns].dropna()
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)
numeric_scaled = pd.DataFrame(numeric_scaled)
numeric_scaled.columns = regression_columns
X = pd.concat([numeric_scaled.reset_index(), categorical_data.reset_index()], axis=1).drop(columns=['index'])
y = data.genre


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)
classifier = LinearSVC(random_state=0, tol=1e-4, max_iter=8000)
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)

clf = SVC(random_state=0, kernel='linear')
clf.fit(X_train, y_train)

params = [
      {'C':[1, 10, 100, 1000], 'kernel':['linear']}
      ]

model = GridSearchCV(estimator=SVC(),
                     param_grid=params,
                     cv=5)

model.fit(X_train, y_train)