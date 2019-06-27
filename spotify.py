# -*- coding: utf-8 -*-
"""
Created on Mon May 13 00:40:36 2019

@author: jwulz
"""

import config
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import sqlite3
import pandas as pd
import re

# Flag to determine if data should be gotten from the API or from local files
use_spotify_flag = True

client_credentials_manager = SpotifyClientCredentials(client_id=config.client_id, client_secret=config.client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


#code to bring in the Pitchfork data
pitchfork_file = 'C:/Learning/ML/spotify/pitchfork-data/database.sqlite'
conn = sqlite3.connect(pitchfork_file)
c = conn.cursor()
c.execute("select reviewid, title, artist from reviews")
reviews = c.fetchall()

# turn this data into a DataFrame
reviews_df = pd.DataFrame(reviews)
reviews_df.columns = ['review_id', 'album_name_pf', 'artist_name']
reviews_df.info()

# use regular expressions to remove the most troublesome phrases 
ep_regex = "\sEP$"
ost_regex = "\sOST$"
for index, row in reviews_df.iterrows():
   row['album_name_pf'] = (re.sub(ep_regex, "", row['album_name_pf'])).strip()
   row['album_name_pf'] = (re.sub(ost_regex, "", row['album_name_pf'])).strip()



# search Spotify using the album name
# use the naming convention that 'sp' indicates spotify data, 'pf' indicates pitchfork data 
def album_search_by_album(album_name, artist_name):
    sp_search = sp.search(q=album_name, limit=50, type='album')
    # check if any results were found - if no results from the search, end the function
    if sp_search['albums']['total'] == 0:
       return('artist_name', album_name, 'zero result search', 'zero result search')
    
    else:
       #iterate through the spotify results to find the first artist that matches
       for element in sp_search['albums']['items']:
          if element['artists'][0]['name'].upper() == artist_name.upper():
             artist_name_sp = element['artists'][0]['name']
             album_name_sp = element['name']
             album_uri = element['uri']
             return(artist_name_sp, album_name, album_name_sp, album_uri)
   
       return(artist_name, album_name, 'no spotify matches', 'no spotify matches')

# for every row in the Pitchfork data, search Spotify and store the results in 'spotify_fetch' 
if use_spotify_flag == True:
   spotify_fetch = {'artist_name': [], 'album_name_pf': [], 'album_name_sp': [], 'album_uri': []}
   for counter, row in reviews_df.iterrows():
      if counter % 100 == 0:
         print("Running the ", counter, "th iteration")
   
      # some of the names raise error message, remove them 
      if (row['album_name_pf'] not in('', '*')) and (row['artist_name'] not in ('', '*')):
          artist_name_sp, album_name_pf, album_name_sp, album_uri = album_search_by_album(row['album_name_pf'], row['artist_name'])
          spotify_fetch['artist_name'].append(row['artist_name'])
          spotify_fetch['album_name_pf'].append(row['album_name_pf'])
          spotify_fetch['album_name_sp'].append(album_name_sp)
          spotify_fetch['album_uri'].append(album_uri)
          
   # turn results into a DataFrame 
   spotify_df = pd.DataFrame(spotify_fetch)
   spotify_df['album_name_sp'] = spotify_df['album_name_sp'].str.upper()
   spotify_df['album_name_pf'] = spotify_df['album_name_pf'].str.upper()
   spotify_df['artist_name'] = spotify_df['artist_name'].str.upper()
   
   spotify_df.to_csv(path_or_buf='C:/Learning/ML/spotify/spotify_df_v2.csv', index=False)
if use_spotify_flag == False:
   spotify_df = pd.read_csv('C:/Learning/ML/spotify/spotify_df_v2.csv')


spotify_df = spotify_df.dropna()


# second method of searching requires two parts:
# first, search for the artist 
def artist_search(artist_name):
   sp_search = sp.search(q=artist_name, limit=10, type='artist')
   # if there are no artists, 'total' will be 0 
   if sp_search['artists']['total'] > 0:
      for element in sp_search['artists']['items']:
         if artist_name.upper() == element['name'].upper():
            return(element['uri'])
      #return blanks in case 
      return('')
   return('')   

# second part - search the artist's discography for the desired album
def album_search_by_artist(artist_name, album_name):
   artist_uri = artist_search(artist_name)
   if artist_uri == '':
      return('artist not found', 'artist not found')
   sp_search = sp.artist_albums(artist_uri)
   sp_album_uri = ''
   sp_single_uri = ''
   if sp_search['total'] > 0:
      for element in sp_search['items']:
         if element['name'].upper() == album_name.upper():
            #logic to handle if the album is considered an album or an ep
            if element['album_group'] == 'single':
               sp_single_uri = element['uri']
               sp_single_name = element['name']
            elif element['album_group'] == 'album':
               sp_album_uri = element['uri']
               sp_album_name = element['name']
      if sp_album_uri != '':
         return(sp_album_name, sp_album_uri)
      elif sp_single_uri != '':
         return(sp_single_name, sp_single_uri)
      else:
         return('no match found', 'no match found')
         
#search_uri = artist_search('fake boyfriend')
#album_name, album_uri = album_search_by_artist('massive attack', 'mezzanine')
#print(album_name, album_uri)
#album_name, album_uri = album_search_by_artist('adr', 'throat')
#print(album_name, album_uri)

if use_spotify_flag == True:
   spotify_fetch = {'artist_name': [], 'album_name_pf': [], 'album_name_sp': [], 'album_uri': []}
   for counter, row in reviews_df.iterrows():
       print("running the ", counter, "th iteration")
       if (row['album_name_pf'] not in('', '*')) and (row['artist_name'] not in ('', '*')):
          album_name_sp, album_uri = album_search_by_artist(row['artist_name'], row['album_name_pf'])
          spotify_fetch['artist_name'].append(row['artist_name'])
          spotify_fetch['album_name_pf'].append(row['album_name_pf'])
          spotify_fetch['album_name_sp'].append(album_name_sp)
          spotify_fetch['album_uri'].append(album_uri)
          
   spotify_df = pd.DataFrame(spotify_fetch)
   spotify_df['album_name_sp'] = spotify_df['album_name_sp'].str.upper()
   spotify_df['album_name_pf'] = spotify_df['album_name_pf'].str.upper()
   spotify_df['artist_name'] = spotify_df['artist_name'].str.upper()
   
   spotify_df.to_csv(path_or_buf='C:/Learning/ML/spotify/spotify_df_by_band.csv', index=False)
   spotify_df_by_band = spotify_df

if use_spotify_flag == False:
   spotify_df_by_band = pd.read_csv('C:/Learning/ML/spotify/spotify_df_by_band.csv')


# join spotify_df on spotify_df_by_band
combined = spotify_df.merge(spotify_df_by_band, how='left', on=['artist_name', 'album_name_pf'], suffixes=['_by_album', '_by_artist'])
combined['by_album_success'] = combined['album_uri_by_album'].str.contains('album')
combined['by_artist_success'] = combined['album_uri_by_artist'].str.contains('album')

#get rid of the NAs
combined.dropna()

# find which URI to use - 
# use the 'by band' method, if that doesn't work then use the 'by album' method
combined['album_uri_to_use'] = combined['album_uri_by_artist']
combined.loc[combined['album_uri_to_use'].str.contains('album') == False, 'album_uri_to_use'] = combined.loc[combined['album_uri_to_use'].str.contains('album') == False, 'album_uri_by_album']
combined.loc[combined['album_uri_by_artist'].str.contains('album') == False]

# save the data
combined.to_csv(path_or_buf='C:/Learning/ML/spotify/spotify_fetch.csv', index=False)

missing_albums = combined.loc[combined['album_uri_by_artist'].str.contains('album') == False]
missing_albums.loc[missing_albums['artist_name'].str.contains(',')].shape
combined.loc[(combined['album_uri_by_artist'].str.contains('album')) & combined['artist_name'].str.contains(',')].shape

album_uris = combined.loc[combined['album_uri_to_use'].str.contains('album'), 'album_uri_to_use'].values

if use_spotify_flag == True:
   template = {
      'album_uri':[],
      'danceability':[],
      'energy':[],
      'key':[], 
      'loudness':[],
      'mode':[],
      'speechiness':[],
      'acousticness':[],
      'instrumentalness':[],
      'liveness':[],
      'valence':[],
      'tempo':[],
      'type':[],
      'id':[],
      'uri':[],
      'track_href':[],
      'analysis_url':[],
      'duration_ms':[],
      'time_signature':[]
      }

   for counter, album in enumerate(album_uris[9464:9466]):
      if counter % 10 == 0:
         print("On interation: ", counter)
      #print(album)
      album_object = sp.album(album)
      for track in album_object['tracks']['items']:
         template['album_uri'].append(album)
         audio_features = sp.audio_features(track['uri'])[0]
         for feature in audio_features:
            template[feature].append(audio_features[feature])
         
   audio_features = pd.DataFrame(template)
   audio_features.to_csv(path_or_buf='C:/Learning/ML/spotify/audio_features.csv', index=False)

else:
   audio_features = pd.read_csv('C:/Learning/ML/spotify/audio_features.csv')










