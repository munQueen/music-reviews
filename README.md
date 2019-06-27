# music-reviews
Use of Spotify's audio feature data to predict Pitchfork review scores

The programs are organized as follows:
1. spotify.py is used to retrieve and prepare data from the spotify API
2. pitchfork.py is used to retrieve and prepare data from the pitchfork kaggle dataset
3. combining.py  merges the resulting datasets from spotify and pitchfork into full_data.csv

One dataset - audio_features.csv - is too large for GitHub and is not available in this repo.
However, it is not required, as full_data.csv contains all information needed 
