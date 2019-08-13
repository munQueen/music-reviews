# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:28:41 2019

@author: jwulz
"""
import pandas as pd


spotify_albums = pd.read_csv('C:/Learning/ML/spotify/spotify_fetch.csv')
spotify_albums = spotify_albums[['artist_name', 'album_name_pf', 'album_uri_to_use']]
audio_features = pd.read_csv('C:/Learning/ML/spotify/audio_features.csv')
pitchfork = pd.read_csv('C:/Learning/ML/spotify/pitchfork_data.csv')

# bring in the album URIs 
albums = pd.merge(left=pitchfork, right=spotify_albums, how='left', left_on=['artist_name', 'album_name'], right_on=['artist_name', 'album_name_pf']).drop('album_name_pf', axis=1)
albums = albums.dropna()
albums = albums[albums.album_uri_to_use.str.contains('album')]

# prepare the audio features to create aggregation variables
audio_features['key'] = audio_features['key'].astype('category')
audio_features['mode'] = audio_features['mode'].astype('category')
audio_features['time_signature'] = audio_features['time_signature'].astype('category')

# setting up the album-level audio features
album_features = audio_features.groupby('album_uri')
album_means = album_features.mean()
album_stds = album_features.std()
album_stats = album_means.merge(album_stds, left_index=True, right_index=True, suffixes=['_mean', '_std'])

# define the aggregation functions we will use for the categorial variables
def mode_fn(values):
   # return the value that has the most occurances
   # if there is a tie, arbitrarily choose the first one
   return(pd.Series.mode(values)[0])
   
primary_key = album_features['key'].agg(mode_fn)
primary_mode = album_features['mode'].agg(mode_fn)
primary_time = album_features['time_signature'].agg(mode_fn)
   
def group_max_value_pct(values):
   # for the most numerous value - count what % this value constitutes of the entire group
   return(pd.Series(values.value_counts() / values.count()).max())
   
key_max_value_pct = album_features['key'].agg(group_max_value_pct)
mode_max_value_pct = album_features['mode'].agg(group_max_value_pct)
time_max_value_pct = album_features['time_signature'].agg(group_max_value_pct)

album_stats['key_max_pct'] = key_max_value_pct
album_stats['primary_key'] = primary_key
album_stats['mode_max_pct'] = mode_max_value_pct
album_stats['primary_mode'] = primary_mode
album_stats['time_max_value_pct'] = time_max_value_pct
album_stats['primary_time'] = primary_time

full_data = albums.merge(album_stats, how='left', left_on='album_uri_to_use', right_index=True)

# clean up the release year variable, remove the one entry listed for 2017 
full_data['release_year'] = full_data['release_year'].astype('int64')
full_data = full_data.loc[full_data['release_year'] != 2017]
full_data = full_data.loc[full_data['publication_year'] != 2017]


full_data.to_csv(path_or_buf='C:/Learning/ML/spotify/full_data.csv', index=False)
