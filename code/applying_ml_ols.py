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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import category_encoders as ce

import statsmodels.api as sm
from statsmodels.formula.api import ols

os.chdir('C:\Learning\ML\spotify')
full_data = pd.read_csv('full_data.csv').dropna()


#how to transform the score variable?
score_sq = full_data.score ** 2


X = full_data.drop(['review_id', 'album_uri_to_use', 'score', 'artist_name', 'album_name', 'publication_year', 'release_year', 'missing'], axis=1)
#set up all of the variables 
regression_columns = ['danceability_mean', 'energy_mean', 'loudness_mean', 'speechiness_mean', 'acousticness_mean', 
                      'instrumentalness_mean', 'liveness_mean', 'valence_mean', 'tempo_mean', 
                      'danceability_std', 'energy_std', 'loudness_std', 'speechiness_std', 'acousticness_std', 
                      'instrumentalness_std', 'liveness_std', 'valence_std', 'tempo_std', 'duration_ms_mean', 'key_max_pct']


full_data['primary_time'] = full_data['primary_time'].apply(str)
ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)

#set up a function to run this model on each genre of music separately 
def genre_linreg(gname, visualize_residuals=False):
   subset_df = full_data[full_data[gname] == True]
   categorical_columns = subset_df[['primary_key', 'primary_mode', 'primary_time']]
   #prepare the categorical variables
   ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
   
   categorical_data = ohe.fit_transform(categorical_columns)
   #for linear regression: drop 1 of each category
   categorical_data = categorical_data.drop(columns=['primary_key_C', 'primary_mode_Major', 'primary_time_4.0'])
   
   #prepare the numeric variables
   numeric_features = subset_df[regression_columns].dropna()
   scaler = StandardScaler()
   numeric_scaled = scaler.fit_transform(numeric_features)
   numeric_scaled = pd.DataFrame(numeric_scaled)
   numeric_scaled.columns = regression_columns
   X = pd.concat([numeric_scaled.reset_index(), categorical_data.reset_index()], axis=1).drop(columns=['index'])
   y = list(subset_df.score ** 2)
   X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)
   X_train = sm.add_constant(X_train)
   genre_linreg = sm.OLS(y_train, X_train).fit()
   print("Linear regression for genre:", gname)
   print(genre_linreg.summary())
   
   data = {'fitted_values':genre_linreg.fittedvalues, 'residuals':genre_linreg.resid}
   residuals = pd.DataFrame(data)
   np.random.seed(42)
   
   if visualize_residuals == True:
      if residuals.fitted_values.shape[0] < 250:
         subset = residuals
      else:
         subset = residuals.sample(250)
      plt.figure()
      plt.scatter(subset.fitted_values, subset.residuals)
      plt.title(gname)
      plt.show()
      
   return(X_train, X_test, y_train, y_test, genre_linreg)

rap_X_train, rap_X_test, rap_y_train, rap_y_test, rap_linreg = genre_linreg('rap')
electronic_X_train, electronic_X_test, electronic_y_train, electronic_y_test, electronic_linreg = genre_linreg('electronic')
experimental_X_train, experimental_X_test, experimental_y_train, experimental_y_test, experimental_linreg = genre_linreg('experimental')
folk_X_train, folk_X_test, folk_y_train, folk_y_test, folk_linreg = genre_linreg('folk/country')
global_X_train, global_X_test, global_y_train, global_y_test, global_linreg = genre_linreg('global')
jazz_X_train, jazz_X_test, jazz_y_train, jazz_y_test, jazz_linreg = genre_linreg('jazz')
metal_X_train, metal_X_test, metal_y_train, metal_y_test, metal_linreg = genre_linreg('metal')
pop_X_train, pop_X_test, pop_y_train, pop_y_test, pop_linreg = genre_linreg('pop/r&b')
rock_X_train, rock_X_test, rock_y_train, rock_y_test, rock_linreg = genre_linreg('rock')


#best R2: metal (.11)
#jazz (.10)
#electronic
#rock 


def rfr_model(X, y):
   #grid search
   gsc = GridSearchCV(estimator=RandomForestRegressor(), 
                      param_grid={
                            'max_depth':range(3,7),
                            'n_estimators':(10,50,100,1000), 
                            }, 
                            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
   
   grid_result = gsc.fit(X, y)
   best_params = grid_result.best_params_
   
   rfr = RandomForestRegressor(max_depth=best_params["max_depth"],
                               n_estimators=best_params["n_estimators"], 
                               random_state=False, verbose=False)
   
   #perform K-fold CV
   scores = cross_val_score(rfr, X, y, cv=10, scoring='neg_mean_absolute_error')
   predictions = cross_val_predict(rfr, X, y, cv=10)
   rfr2 = RandomForestRegressor(max_depth=best_params["max_depth"],
                               n_estimators=best_params["n_estimators"], 
                               random_state=False, verbose=False)
   rfr2.fit(X, y)
   return scores, predictions, rfr2


metal_scores, metal_predictions, metal_rfr = rfr_model(metal_X_train, metal_y_train)

metal_rfr.feature_importances_
metal_mse = sum((metal_predictions - metal_y_train) ** 2) / len(metal_predictions)
