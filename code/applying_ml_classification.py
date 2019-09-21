import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

os.chdir('C:\Learning\ML\spotify')
full_data = pd.read_csv('full_data.csv').dropna()

# remove all albums which are given multiple genres 
single_genre_albums = full_data['rap'].astype(int) + full_data['electronic'].astype(int) +\
   full_data['experimental'].astype(int) + full_data['folk/country'].astype(int) +\
   full_data['global'].astype(int) + full_data['jazz'].astype(int) +\
   full_data['metal'].astype(int) + full_data['pop/r&b'].astype(int) +\
   full_data['rock'].astype(int) == 1

# create a 'genre' column 
data = full_data[single_genre_albums].dropna()
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


#transformations:
#drop the bottom genres, resample from the top genre for balanced classes
print(data.genre.value_counts() )

# best options appear to be either keeping rock/electronic and resampling to 1000, or rock/electronic/rap/pop and resampling to 500
#resampling top 2 to 1000
rock_sample = data[data.genre == 'rock'].sample(n=1000, random_state=42)
electronic_sample = data[data.genre == 'electronic'].sample(n=1000, random_state=42)
resampled_1000 = data[(data.genre == 'rap') | (data.genre == 'pop/r&b')].append(rock_sample).append(electronic_sample)

#resampling all 4 to 500
rock_sample = data[data.genre == 'rock'].sample(n=500, random_state=42)
electronic_sample = data[data.genre == 'electronic'].sample(n=500, random_state=42)
rap_sample = data[data.genre == 'rap'].sample(n=500, random_state=42)
pop_sample = data[data.genre == 'pop/r&b'].sample(n=500, random_state=42)

resampled_500 = rock_sample.append(electronic_sample).append(rap_sample).append(pop_sample)


# using the resampled to 500 data
data = resampled_500

#set up all of the variables 
regression_columns = ['danceability_mean', 'energy_mean', 'loudness_mean', 'speechiness_mean', 'acousticness_mean', 
                      'instrumentalness_mean', 'liveness_mean', 'valence_mean', 'tempo_mean', 
                      'danceability_std', 'energy_std', 'loudness_std', 'speechiness_std', 'acousticness_std', 
                      'instrumentalness_std', 'liveness_std', 'valence_std', 'tempo_std', 'duration_ms_mean', 'key_max_pct']

# prepare the categorical variables
data['primary_time'] = data['primary_time'].apply(str)
ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
categorical_columns = data[['primary_key', 'primary_mode', 'primary_time']]
categorical_data = ohe.fit_transform(categorical_columns)
#for drop 1 of each category
categorical_data = categorical_data.drop(columns=['primary_key_C', 'primary_mode_Major', 'primary_time_4.0'])


#prepare the numeric variables
numeric_features = data[regression_columns]
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)
numeric_scaled = pd.DataFrame(numeric_scaled)
numeric_scaled.columns = regression_columns
X = pd.concat([numeric_scaled.reset_index(), categorical_data.reset_index()], axis=1).drop(columns=['index'])
y = data.genre


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.4)

def metric_aggregation(model, X, y):
   # this function will be used to take the various models, and store output from classification_report() 
   # after running this for our three models, we can compare their performance in a variety of ways
   true_data = y
   predicted_data = model.predict(X)
   cr = classification_report(true_data, predicted_data, output_dict=True)
   avg_f1 = cr['weighted avg']['f1-score']
   avg_precision = cr['weighted avg']['precision']
   avg_recall = cr['weighted avg']['recall']
   return([avg_f1, avg_precision, avg_recall])
   
   
#Start running the three models
params = [
      {'C':[0.01, 0.1, 1, 10, 100], 
       'kernel':['rbf'], 
       'gamma':[0.001, 0.01, 0.1, 1]
      }]

svc_model = GridSearchCV(estimator=SVC(random_state=42),
                     param_grid=params,
                     cv=5
                     )

svc_model.fit(X_train, y_train)
print(svc_model.best_params_)


svc_train_metrics = metric_aggregation(svc_model, X_train, y_train)
svc_test_metrics = metric_aggregation(svc_model, X_test, y_test)



#running KNN 
params = [
      {'n_neighbors':[1, 2, 3, 4, 5, 6, 7], 
       'p':[1, 2, 3]
      }]
knn_model = GridSearchCV(estimator=KNeighborsClassifier(),
                     param_grid=params,
                     cv=5)
knn_model.fit(X_train, y_train)

knn_train_metrics = metric_aggregation(knn_model, X_train, y_train)
knn_test_metrics = metric_aggregation(knn_model, X_test, y_test)




# calibrating the random forest classifier
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)]
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


rf = RandomForestClassifier(random_state=42)
rfc_model = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               n_iter=50,
                               cv=10,
                               verbose=10)
rfc_model.fit(X_train, y_train)
y_pred = rfc_model.predict(X_test)

rfc_test_metrics = metric_aggregation(rfc_model, X_test, y_test)   
rfc_train_metrics = metric_aggregation(rfc_model, X_train, y_train) 

rfc_params = rfc_model.best_params_


# set up the output 
model_col = ['SVC Test'] * 3  + ['SVC Train'] * 3 +  ['KNN Test'] * 3 + ['KNN Train'] * 3  + ['RFC Test'] * 3 + ['RFC Train'] * 3
metric_col = ['F1', 'Precision', 'Recall'] * 6
data = svc_test_metrics + svc_train_metrics + knn_test_metrics + knn_train_metrics + rfc_test_metrics + rfc_train_metrics

metric_dict = {'Model':model_col, 'Metric':metric_col, 'Data':data}
metric_df = pd.DataFrame(metric_dict)


plt.figure()
sns.barplot(hue='Metric', x='Model', y='Data', data=metric_df.loc[(metric_df.Model == 'SVC Test') | (metric_df.Model == 'KNN Test')| (metric_df.Model == 'RFC Test')])
plt.title("Comparison of classification models - Testing data")
plt.xlabel("Model")
plt.ylabel("Value of metrics")
plt.show()

plt.figure()
sns.barplot(hue='Metric', x='Model', y='Data', data=metric_df.loc[(metric_df.Model == 'SVC Train') | (metric_df.Model == 'KNN Train')| (metric_df.Model == 'RFC Train')])
plt.title("Comparison of classification models - Training data")
plt.xlabel("Model")
plt.ylabel("Value of metrics")
plt.show()

#look at the confusion matrix for SVC and RFC
cm = confusion_matrix(y_test, rfc_model.predict(X_test))
cm_df = pd.DataFrame(cm)
cm_df.columns = ['Predicted Electronic', 'Predicted Pop/R&B', 'Predicted Rap', 'Predicted Rock']
cm_df.index = ['True Electronic', 'True Pop/R&B', 'True Rap', 'True Rock']