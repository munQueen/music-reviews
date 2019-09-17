import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
#import category_encoders as ce
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, classification_report 


#os.chdir('C:\Learning\ML\spotify')
os.chdir('C:\ML\Springboard\music-reviews-master')
full_data = pd.read_csv('full_data.csv').dropna()

#remove all albums which are given multiple genres 
single_genre_albums = full_data['rap'].astype(int) + full_data['electronic'].astype(int) +\
   full_data['experimental'].astype(int) + full_data['folk/country'].astype(int) +\
   full_data['global'].astype(int) + full_data['jazz'].astype(int) +\
   full_data['metal'].astype(int) + full_data['pop/r&b'].astype(int) +\
   full_data['rock'].astype(int) == 1

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
data.genre.value_counts() #/ len(data.genre)

#completely drop jazz and global
#consider dropping metal

#definitely resample rock
#consider resampling electronic, rap

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


#set up all of the variables 
regression_columns = ['danceability_mean', 'energy_mean', 'loudness_mean', 'speechiness_mean', 'acousticness_mean', 
                      'instrumentalness_mean', 'liveness_mean', 'valence_mean', 'tempo_mean', 
                      'danceability_std', 'energy_std', 'loudness_std', 'speechiness_std', 'acousticness_std', 
                      'instrumentalness_std', 'liveness_std', 'valence_std', 'tempo_std', 'duration_ms_mean', 'key_max_pct']


data = resampled_1000

def airplane_stuff():
    data['primary_time'] = data['primary_time'].apply(str)
    ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
    categorical_columns = data[['primary_key', 'primary_mode', 'primary_time']]
    #prepare the categorical variables
    ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
    
    categorical_data = ohe.fit_transform(categorical_columns)
    #for linear regression: drop 1 of each category
    categorical_data = categorical_data.drop(columns=['primary_key_C', 'primary_mode_Major', 'primary_time_4.0'])
    return()

#prepare the numeric variables
numeric_features = data[regression_columns]
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)
numeric_scaled = pd.DataFrame(numeric_scaled)
numeric_scaled.columns = regression_columns
#X = pd.concat([numeric_scaled.reset_index(), categorical_data.reset_index()], axis=1).drop(columns=['index'])
X = numeric_scaled
y = data.genre


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)
#classifier = SVC(random_state=0, tol=1e-4, max_iter=8000)
#classifier.fit(X_train, y_train)
#classifier.score(X_train, y_train)

clf = SVC(random_state=0, kernel='rbf', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cr = classification_report(y_test, y_pred, output_dict=True)
print("Accuracy on training data:", clf.score(X_train, y_train))
print("Accuracy on testing data:", clf.score(X_test, y_test))
print('Micro avg score on test data:', cr['micro avg']['f1-score'])

#to look up: 
#ROC curve w/multiple classes? 


params = [
      {'C':[1, 10, 100], 'kernel':['linear', 'rbf', 'poly'], 
       'gamma':[0.001, 0.0001, 0.00001], 'degree':[2, 3, 4, 5]}
      ]

model = GridSearchCV(estimator=SVC(),
                     param_grid=params,
                     cv=5)

model.fit(X_train, y_train)

clf = SVC(random_state=42, kernel='rbf', C=100, gamma=0.001)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cr = classification_report(y_test, y_pred, output_dict=True)
print("Accuracy on training data:", clf.score(X_train, y_train))
print("Accuracy on testing data:", clf.score(X_test, y_test))
print('Micro avg score on test data:', cr['micro avg']['f1-score'])
cr2 = classification_report(y_test, y_pred)

