# -*- coding: utf-8 -*-

"""
Created on Thu Aug 15 19:04:48 2019
@author: jwulz
"""
# import all of the packages needed
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import category_encoders as ce
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


os.chdir('C:\Learning\ML\spotify')
full_data = pd.read_csv('full_data.csv').dropna()


################X = full_data.drop(['review_id', 'album_uri_to_use', 'score', 'artist_name', 'album_name', 'publication_year', 'release_year', 'missing'], axis=1)
# list the numeric variables here
regression_columns = ['danceability_mean', 'energy_mean', 'loudness_mean', 'speechiness_mean', 'acousticness_mean', 
                      'instrumentalness_mean', 'liveness_mean', 'valence_mean', 'tempo_mean', 
                      'danceability_std', 'energy_std', 'loudness_std', 'speechiness_std', 'acousticness_std', 
                      'instrumentalness_std', 'liveness_std', 'valence_std', 'tempo_std', 'duration_ms_mean', 'key_max_pct']


full_data['primary_time'] = full_data['primary_time'].apply(str)
ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)

#set up a function to run this model on each genre of music separately 
def genre_linreg(gname, visualize_residuals=False):
   # keep only data pertaining to the given genre
   subset_df = full_data[full_data[gname] == True]
   categorical_columns = subset_df[['primary_key', 'primary_mode', 'primary_time']]
   
   # prepare the categorical variables
   ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
   categorical_data = ohe.fit_transform(categorical_columns)
   # for linear regression dummy variables: drop 1 of each category
   categorical_data = categorical_data.drop(columns=['primary_key_C', 'primary_mode_Major', 'primary_time_4.0'])
   
   # prepare the numeric variables
   numeric_features = subset_df[regression_columns].dropna()
   scaler = StandardScaler()
   numeric_scaled = scaler.fit_transform(numeric_features)
   numeric_scaled = pd.DataFrame(numeric_scaled)
   numeric_scaled.columns = regression_columns
   X = pd.concat([numeric_scaled.reset_index(), categorical_data.reset_index()], axis=1).drop(columns=['index'])
   X = sm.add_constant(X)
   y = list(subset_df.score ** 2)
   
   # split X and y into test and train sets
   X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.4)
   
   # fit OLS
   genre_linreg = sm.OLS(y_train, X_train).fit()
   print("Linear regression for genre:", gname)
   print(genre_linreg.summary())
   
   # code to optionally visualize the residuals
   if visualize_residuals == True:
   
      data = {'fitted_values':genre_linreg.fittedvalues, 'residuals':genre_linreg.resid}
      residuals = pd.DataFrame(data)
      np.random.seed(42)
   
      if residuals.fitted_values.shape[0] < 250:
         subset = residuals
      else:
         subset = residuals.sample(250)
      plt.figure()
      plt.scatter(subset.fitted_values, subset.residuals)
      plt.title(gname)
      plt.show()
      
   return(X_train, X_test, y_train, y_test, genre_linreg)

# run the function for each genre
rap_X_train, rap_X_test, rap_y_train, rap_y_test, rap_linreg = genre_linreg('rap')
electronic_X_train, electronic_X_test, electronic_y_train, electronic_y_test, electronic_linreg = genre_linreg('electronic')
experimental_X_train, experimental_X_test, experimental_y_train, experimental_y_test, experimental_linreg = genre_linreg('experimental')
folk_X_train, folk_X_test, folk_y_train, folk_y_test, folk_linreg = genre_linreg('folk/country')
global_X_train, global_X_test, global_y_train, global_y_test, global_linreg = genre_linreg('global')
jazz_X_train, jazz_X_test, jazz_y_train, jazz_y_test, jazz_linreg = genre_linreg('jazz')
metal_X_train, metal_X_test, metal_y_train, metal_y_test, metal_linreg = genre_linreg('metal')
pop_X_train, pop_X_test, pop_y_train, pop_y_test, pop_linreg = genre_linreg('pop/r&b')
rock_X_train, rock_X_test, rock_y_train, rock_y_test, rock_linreg = genre_linreg('rock')

# calculate the MSE
ols_rap_train_mse = mean_squared_error(rap_y_train, rap_linreg.predict(rap_X_train))
ols_rap_test_mse = mean_squared_error(rap_y_test, rap_linreg.predict(rap_X_test))
ols_rock_train_mse = mean_squared_error(rock_y_train, rock_linreg.predict(rock_X_train))
ols_rock_test_mse = mean_squared_error(rock_y_test, rock_linreg.predict(rock_X_test))
ols_electronic_train_mse = mean_squared_error(electronic_y_train, electronic_linreg.predict(electronic_X_train))
ols_electronic_test_mse = mean_squared_error(electronic_y_test, electronic_linreg.predict(electronic_X_test))
ols_pop_train_mse = mean_squared_error(pop_y_train, pop_linreg.predict(pop_X_train))
ols_pop_test_mse = mean_squared_error(pop_y_test, pop_linreg.predict(pop_X_test))







from sklearn.model_selection import RandomizedSearchCV
# set up the combinations of parameters
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#run the grid search random forest regressors
rf = RandomForestRegressor(random_state=42)
rfr_rap = RandomizedSearchCV(estimator=rf, param_distributions = random_grid, n_iter=50, cv=5, verbose=10)
rfr_rap.fit(rap_X_train, rap_y_train)


rfr_rock = RandomizedSearchCV(estimator=rf, param_distributions = random_grid, n_iter=50, cv=5, verbose=10)
rfr_rock.fit(rock_X_train, rock_y_train)
#started at 10:25pm

rfr_electronic = RandomizedSearchCV(estimator=rf, param_distributions = random_grid, n_iter=50, cv=5, verbose=10)
rfr_electronic.fit(electronic_X_train, electronic_y_train)


rfr_pop = RandomizedSearchCV(estimator=rf, param_distributions = random_grid, n_iter=50, cv=5, verbose=10)
rfr_pop.fit(pop_X_train, pop_y_train)


rfr_rap_train_mse = mean_squared_error(rap_y_train, rfr_rap.predict(rap_X_train))
rfr_rap_test_mse = mean_squared_error(rap_y_test, rfr_rap.predict(rap_X_test))
rfr_rock_train_mse = mean_squared_error(rock_y_train, rfr_rock.predict(rock_X_train))
rfr_rock_test_mse = mean_squared_error(rock_y_test, rfr_rock.predict(rock_X_test))
rfr_electronic_train_mse = mean_squared_error(electronic_y_train, rfr_electronic.predict(electronic_X_train))
rfr_electronic_test_mse = mean_squared_error(electronic_y_test, rfr_electronic.predict(electronic_X_test))
rfr_pop_train_mse = mean_squared_error(pop_y_train, rfr_pop.predict(pop_X_train))
rfr_pop_test_mse = mean_squared_error(pop_y_test, rfr_pop.predict(pop_X_test))

#create a dataframe to compare the different MSEs for the four genres


ols_tests = [ols_rap_test_mse, ols_rock_test_mse, ols_electronic_test_mse, ols_pop_test_mse]
ols_train = [ols_rap_train_mse, ols_rock_train_mse, ols_electronic_train_mse, ols_pop_train_mse]
rfr_tests = [rfr_rap_test_mse, rfr_rock_test_mse, rfr_electronic_test_mse, rfr_pop_test_mse]
rfr_train = [rfr_rap_train_mse, rfr_rock_train_mse, rfr_electronic_train_mse, rfr_pop_train_mse]

genre_col = ['rap'] * 4 + ['rock'] * 4 + ['electronic'] * 4 + ['pop'] * 4
model_col = ['OLS Test', 'OLS Train', 'RFR Test', 'RFR Train'] * 4
mse_col = ols_tests + ols_train + rfr_tests + rfr_train
data = {'Genre':genre_col, 'Model':model_col, 'MSE':mse_col}
model_comparison = pd.DataFrame(data)

plt.figure()
sns.barplot(hue='Model', x='Genre', y='MSE', data=model_comparison.loc[(model_comparison.Model == 'OLS Test') | (model_comparison.Model == 'RFR Test')])
plt.title("Comparison of OLS vs RFR - Testing data")
plt.xlabel("Genre")
plt.ylabel("MSE (smaller is better)")
plt.show()

plt.figure()
sns.barplot(hue='Model', x='Genre', y='MSE', data=model_comparison.loc[(model_comparison.Model == 'OLS Train') | (model_comparison.Model == 'RFR Train')])
plt.title("Comparison of OLS vs RFR - Training data")
plt.xlabel("Genre")
plt.ylabel("MSE (smaller is better)")
plt.show()