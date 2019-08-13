# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:17:33 2019

@author: jwulz
"""

import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns


pitchfork_file = 'C:/Learning/ML/spotify/pitchfork-data/database.sqlite'
conn = sqlite3.connect(pitchfork_file)
c = conn.cursor()

c.execute("""
          SELECT 
             reviews.reviewid, 
             reviews.score, 
             reviews.artist, 
             reviews.title, 
             reviews.pub_year, 
             years.year as release_year 
          from reviews
          left join years 
          on reviews.reviewid = years.reviewid""")
pf = c.fetchall()
pf = pd.DataFrame(pf)
pf.columns = ['review_id', 'score', 'artist_name', 'album_name', 'publication_year', 'release_year']

# handle duplicates from the 'years' dataset
# many albums that have re-releases have two entries, and we want to remove both of them
duplicate_ids = pf.review_id.value_counts() > 1
duplicate_ids = duplicate_ids.reset_index()
duplicate_ids.columns = ['index', 'duplicate_flag']
t = pd.merge(pf, duplicate_ids, left_on='review_id', right_on='index').drop('index', axis=1)
duplicates = t[t.duplicate_flag]
non_duplicates = t[t.duplicate_flag == False]

# remove cases where the publication year is more than a year later than album release
non_duplicates = non_duplicates.loc[non_duplicates['publication_year'] - non_duplicates['release_year'] < 2].drop('duplicate_flag', axis=1)


print("Re-release average score:", duplicates.score.mean())
print("Normal average score:", non_duplicates.score.mean())
non_duplicates['artist_name'] = non_duplicates.artist_name.str.upper()
non_duplicates['album_name'] = non_duplicates.album_name.str.upper()


# we will go with the non_duplicate data as the records of interest
time_trend = non_duplicates.groupby('publication_year').mean()
time_trend.score.plot()
plt.show()


#c.execute("""
#          SELECT genres.*, reviews.score
#          from genres
#          left join reviews
#          on genres.reviewid = reviews.reviewid
#          """)


# fetch the genre data
c.execute("""
          SELECT *
          from genres         
          """)
genres = pd.DataFrame(c.fetchall())
genres.columns = ['review_id', 'genre']
genres = genres.drop_duplicates()

# create a 'flag' variable so that when we pivot, the new variables are populated correctly 
genres['flag'] = True
# turn the Nas into 'missing' so that the pivot keeps their data
genres.loc[genres['genre'].isna() == True, 'genre'] ='missing'
genre_tags = genres.pivot(index='review_id', columns='genre', values='flag')
genre_tags = genre_tags.fillna(False)

genre_tags.to_csv(path_or_buf='C:/Learning/ML/spotify/genre_tags.csv', index=False)

# combine our previous pf dataset with the genre tags
pitchfork_data = non_duplicates.merge(genre_tags, how='left', on='review_id')
pitchfork_data.to_csv(path_or_buf='C:/Learning/ML/spotify/pitchfork_data.csv', index=False)

# what genres are frequently used together?
# to explore this, plot the correlation of each genre with all other genres to see which ones have large positive counts

sns.set()
genre_corr = genre_tags.corr()

columns = list(genre_corr)
for i in columns:
   plot_data = genre_corr.loc[genre_corr[i] != 1, i]   
   plot_data.plot(kind='bar')
   plt.title(i)
   plt.show()

full_data = non_duplicates.merge(genre_tags, how='left', on='review_id')
# in examining the full data - 
# what has been the average score, by genre tag, every year
# some genres have generally faired better than others, but not massively so 
genres_full = genres.merge(non_duplicates, how='left', on='review_id')
genre_trends = genres_full.groupby(['publication_year', 'genre']).mean()['score']
genre_trends = genre_trends.unstack()
genre_trends.plot()



def combine_columns(df, col1, col2):
   combined_label = str(col1 + '_' + col2)
   df[combined_label] = (df[col1] & t[col2])
combine_columns(genre_tags, 'electronic', 'jazz')
combine_columns(genre_tags, 'experimental', 'jazz')
combine_columns(genre_tags, 'experimental', 'rock')
combine_columns(genre_tags, 'rock', 'metal')
combine_columns(genre_tags, 'pop/r&b', 'global')



