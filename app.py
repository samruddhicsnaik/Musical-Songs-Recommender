# #***********************IMPORT MODULES**********************#

from flask import Flask, render_template, request
from decimal import Decimal

import requests
import datetime
from urllib.parse import urlencode
import base64

import pandas as pd
import re
# from datetime import datetime
import itertools

from skimage import io
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util


# #***********************CREDENTIALS**********************#

client_id = "5a3253bc364d4ba09a7391008566a4fb"
client_secret = "e96d1058f2fb47e8976d2bae05589c77"
user_id = "31yqmt5o4fmugzzw7buxuhpzi2ha"


# #***********************SPOTIFY API**********************#

class SpotifyAPI(object):
    access_token = None
    access_token_expires = datetime.datetime.now()
    access_token_did_expire = True
    client_id = None
    client_secret = None
    token_url = "https://accounts.spotify.com/api/token"
    
    def __init__(self, client_id, client_secret, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id = client_id
        self.client_secret = client_secret

    def get_client_credentials(self):
        """
        Returns a base64 encoded string
        """
        client_id = self.client_id
        client_secret = self.client_secret
        if client_secret == None or client_id == None:
            raise Exception("You must set client_id and client_secret")
        client_creds = f"{client_id}:{client_secret}"
        client_creds_b64 = base64.b64encode(client_creds.encode())
        return client_creds_b64.decode()
    
    def get_token_headers(self):
        client_creds_b64 = self.get_client_credentials()
        return {
            "Authorization": f"Basic {client_creds_b64}"
        }
    
    def get_token_data(self):
        return {
            "grant_type": "client_credentials"
        } 
    
    def perform_auth(self):
        token_url = self.token_url
        token_data = self.get_token_data()
        token_headers = self.get_token_headers()
        r = requests.post(token_url, data=token_data, headers=token_headers)
        if r.status_code not in range(200, 299):
            raise Exception("Could not authenticate client.")
            # return False
        data = r.json()
        now = datetime.datetime.now()
        access_token = data['access_token']
        expires_in = data['expires_in'] # seconds
        expires = now + datetime.timedelta(seconds=expires_in)
        self.access_token = access_token
        self.access_token_expires = expires
        self.access_token_did_expire = expires < now
        return True
    
    def get_access_token(self):
        token = self.access_token
        expires = self.access_token_expires
        now = datetime.datetime.now()
        if expires < now:
            self.perform_auth()
            return self.get_access_token()
        elif token == None:
            self.perform_auth()
            return self.get_access_token() 
        return token
    
    def get_resource_header(self):
        access_token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        return headers
        
        
    def get_resource(self, lookup_id, resource_type='albums', version='v1'):
        endpoint = f"https://api.spotify.com/{version}/{resource_type}/{lookup_id}"
        headers = self.get_resource_header()
        r = requests.get(endpoint, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()
    
    def get_album(self, _id):
        return self.get_resource(_id, resource_type='albums')
    
    def get_artist(self, _id):
        return self.get_resource(_id, resource_type='artists')
    
    def get_track(self, _id):
        return self.get_resource(_id, resource_type='tracks')
    
    def get_playlist(self, _id):
        return self.get_resource(_id, resource_type='playlists')
    
    def get_audio_features(self, _id):
        return self.get_resource(_id, resource_type='audio-features')
    
    def get_artist_albums(self, lookup_id, resource_type='artists', version='v1'):
        endpoint = f"https://api.spotify.com/v1/artists/{lookup_id}/albums"
        headers = self.get_resource_header()
        r = requests.get(endpoint, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()
    
    def get_album_tracks(self, lookup_id, resource_type='albums', version='v1'):
        endpoint = f"https://api.spotify.com/v1/albums/{lookup_id}/tracks"
        headers = self.get_resource_header()
        r = requests.get(endpoint, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()
    
    def get_current_user_playlists(self, lookup_id, resource_type='users', version='v1'):
        endpoint = f"https://api.spotify.com/{version}/{resource_type}/{lookup_id}/playlists"
        headers = self.get_resource_header()
        r = requests.get(endpoint, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()
    
    def base_search(self, query_params): # type
        headers = self.get_resource_header()
        endpoint = "https://api.spotify.com/v1/search"
        lookup_url = f"{endpoint}?{query_params}"
        r = requests.get(lookup_url, headers=headers)
        if r.status_code not in range(200, 299):  
            return {}
        return r.json()
    
    def search(self, query=None, operator=None, operator_query=None, search_type='artist' ):
        if query == None:
            raise Exception("A query is required")
        if isinstance(query, dict):
            query = " ".join([f"{k}:{v}" for k,v in query.items()])
        if operator != None and operator_query != None:
            if operator.lower() == "or" or operator.lower() == "not":
                operator = operator.upper()
                if isinstance(operator_query, str):
                    query = f"{query} {operator} {operator_query}"
        query_params = urlencode({"q": query, "type": search_type.lower()})
        print(query_params)
        return self.base_search(query_params)

spotify = SpotifyAPI(client_id, client_secret)


# #***********************DATA**********************#

tracks_df = pd.read_csv('tracks.csv')


# #***********************RECOMMENDATION**********************#

class recommendation:
    
  def prepare_data(self):
      tracks_df['artists_name'] = tracks_df['artists_name'].apply(lambda x: re.findall(r"'([^'\"]*)'", x))
      tracks_df['date']= pd.to_datetime(tracks_df['date'])
      tracks_df['date'] = tracks_df['date'].dt.strftime('%Y-%m-%d')
      tracks_df['year'] = tracks_df['date'].apply(lambda x: x.split('-')[0])
      tracks_df['popularity_red'] = tracks_df['popularity'].apply(lambda x: int(x/5))
      tracks_df['tracks_genres'] = tracks_df['tracks_genres'].apply(lambda x: re.findall(r"'([^']*)'", x))
  #     tracks_df.head()
      return tracks_df

  def ohe_prep(self, df, column, new_name):
      """ 
      Create One Hot Encoded features of a specific column

      Parameters: 
          df (pandas dataframe): Spotify Dataframe
          column (str): Column to be processed
          new_name (str): new column name to be used

      Returns: 
          tf_df: One hot encoded features 
      """

      tf_df = pd.get_dummies(df[column])
      feature_names = tf_df.columns
      tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
      tf_df.reset_index(drop = True, inplace = True)    
      return tf_df
  
  
  # function to build entire feature set
  def create_feature_set(self, df, float_cols):
      """ 
      Process spotify df to create a final set of features that will be used to generate recommendations

      Parameters: 
          df (pandas dataframe): Spotify Dataframe
          float_cols (list(str)): List of float columns that will be scaled 

      Returns: 
          final: final set of features 
      """

      #tfidf genre lists
      tfidf = TfidfVectorizer()
      tfidf_matrix =  tfidf.fit_transform(df['tracks_genres'].apply(lambda x: " ".join(x)))
      genre_df = pd.DataFrame(tfidf_matrix.toarray())
      genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
      genre_df.reset_index(drop = True, inplace=True)

      #explicity_ohe = ohe_prep(df, 'explicit','exp')    
      year_ohe = self.ohe_prep(df, 'year','year') * 0.5
      popularity_ohe = self.ohe_prep(df, 'popularity_red','pop') * 0.15

      #scale float columns
      floats = df[float_cols].reset_index(drop = True)
      scaler = MinMaxScaler()
      floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

      #concanenate all features
      final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis = 1)

      #add song id
      final['track_id']=df['track_id'].values

      return final

  def get_user_playlists(self):
      id_name = {}
      list_photo = {}
      for i in spotify.get_current_user_playlists(user_id)['items']:

          id_name[i['name']] = i['uri'].split(':')[2]
          list_photo[i['uri'].split(':')[2]] = i['images'][0]['url']
      return id_name, list_photo
  
  def create_necessary_outputs(self, playlist_name, id_dic, df):
      """ 
      Pull songs from a specific playlist.

      Parameters: 
          playlist_name (str): name of the playlist you'd like to pull from the spotify API
          id_dic (dic): dictionary that maps playlist_name to playlist_id
          df (pandas dataframe): spotify dataframe

      Returns: 
          playlist: all songs in the playlist THAT ARE AVAILABLE IN THE KAGGLE DATASET
      """

      #generate playlist dataframe
      playlist = pd.DataFrame()
      playlist_name = playlist_name

      for ix, i in enumerate(spotify.get_playlist(id_dic[playlist_name])['tracks']['items']):
          #print(i['track']['artists'][0]['name'])
          playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
          playlist.loc[ix, 'name'] = i['track']['name']
          playlist.loc[ix, 'track_id'] = i['track']['id'] # ['uri'].split(':')[2]
          playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
          playlist.loc[ix, 'date_added'] = i['added_at']

      playlist['date_added'] = pd.to_datetime(playlist['date_added'])  

      playlist = playlist[playlist['track_id'].isin(df['track_id'].values)].sort_values('date_added',ascending = False)

      return playlist
  
  def visualize_songs(self, df):
      """ 
      Visualize cover art of the songs in the inputted dataframe

      Parameters: 
          df (pandas dataframe): Playlist Dataframe
      """

      temp = df['url'].values
      plt.figure(figsize=(15,int(0.625 * len(temp))))
      columns = 5

      for i, url in enumerate(temp):
          plt.subplot(len(temp) / columns + 1, columns, i + 1)

          image = io.imread(url)
          plt.imshow(image)
          plt.xticks(color = 'w', fontsize = 0.1)
          plt.yticks(color = 'w', fontsize = 0.1)
          plt.xlabel(df['name'].values[i], fontsize = 12)
          plt.tight_layout(h_pad=0.4, w_pad=0)
          plt.subplots_adjust(wspace=None, hspace=None)

      plt.show()
      
  def generate_playlist_feature(self, complete_feature_set, playlist_df, weight_factor):
      """ 
      Summarize a user's playlist into a single vector

      Parameters: 
          complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
          playlist_df (pandas dataframe): playlist dataframe
          weight_factor (float): float value that represents the recency bias. The larger the recency bias, the most priority recent songs get. Value should be close to 1. 

      Returns: 
          playlist_feature_set_weighted_final (pandas series): single feature that summarizes the playlist
          complete_feature_set_nonplaylist (pandas dataframe): 
      """

      complete_feature_set_playlist = complete_feature_set[complete_feature_set['track_id'].isin(playlist_df['track_id'].values)]#.drop('id', axis = 1).mean(axis =0)
      complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['track_id','date_added']], on = 'track_id', how = 'inner')
      complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['track_id'].isin(playlist_df['track_id'].values)]#.drop('id', axis = 1)

      playlist_feature_set = complete_feature_set_playlist.sort_values('date_added',ascending=False)

      most_recent_date = playlist_feature_set.iloc[0,-1]

      for ix, row in playlist_feature_set.iterrows():
          playlist_feature_set.loc[ix,'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)

      playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))

      playlist_feature_set_weighted = playlist_feature_set.copy()
      #print(playlist_feature_set_weighted.iloc[:,:-4].columns)
      playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-4].mul(playlist_feature_set_weighted.weight,0))
      playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
      #playlist_feature_set_weighted_final['id'] = playlist_feature_set['id']

      return playlist_feature_set_weighted_final.sum(axis = 0), complete_feature_set_nonplaylist
  
  def generate_playlist_recos(self, df, features, nonplaylist_features):
      """ 
      Pull songs from a specific playlist.

      Parameters: 
          df (pandas dataframe): spotify dataframe
          features (pandas series): summarized playlist feature
          nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist

      Returns: 
          non_playlist_df_top_40: Top 40 recommendations for that playlist
      """

      non_playlist_df = df[df['track_id'].isin(nonplaylist_features['track_id'].values)]
      non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('track_id', axis = 1).values, features.values.reshape(1, -1))[:,0]
      non_playlist_df_top_40 = non_playlist_df.sort_values('sim',ascending = False).head(10)
      non_playlist_df_top_40['url'] = non_playlist_df_top_40['track_id'].apply(lambda x: spotify.get_track(x)['album']['images'][1]['url'])

      return non_playlist_df_top_40


# #***********************CURRENT USER PLAYLISTS**********************#

user_playlists = spotify.get_current_user_playlists(user_id)

playlists_details = {}
for i in range(len(user_playlists['items'])):
  playlist_details = {}
  playlist_details['name'] = user_playlists['items'][i]['name']
  playlist_details['photo'] = user_playlists['items'][i]['images'][0]['url']
  playlists_details[user_playlists['items'][i]['id']] = playlist_details

for playlist in playlists_details:
  playlist_details = spotify.get_playlist(playlist)
  playlist_tracks_details = []
  for i in range(len(playlist_details['tracks']['items'])):
      playlist_track_details = {}
      playlist_track_details['sr'] = i + 1
      playlist_track_details['id'] = playlist_details['tracks']['items'][i]['track']['id']
      playlist_track_details['name'] = playlist_details['tracks']['items'][i]['track']['name']
      playlist_track_details['album'] = playlist_details['tracks']['items'][i]['track']['album']['name']
      playlist_track_details['duration'] = round((playlist_details['tracks']['items'][i]['track']['duration_ms']/(1000*60))%60, 2)
      playlist_track_details['photo'] = playlist_details['tracks']['items'][i]['track']['album']['images'][2]['url']
      playlist_tracks_details.append(playlist_track_details)
  playlists_details[playlist]['tracks'] = playlist_tracks_details


rec = recommendation()
tracks_df = rec.prepare_data()

float_cols = tracks_df.dtypes[tracks_df.dtypes == 'float64'].index.values
complete_feature_set = rec.create_feature_set(tracks_df, float_cols=float_cols)


# #***********************GET USER PLAYLISTS**********************#

id_name, list_photo = rec.get_user_playlists()


# #***************************ALBUMS******************************#

albums_df = pd.read_csv('albums.csv')
# albums_df['date']= pd.to_datetime(albums_df['date'])
# albums_df['date'] = albums_df['date'].dt.strftime('%Y-%m-%d')
# albums_df['year'] = albums_df['date'].apply(lambda x: x.split('-')[0])


class similarAlbums:
    
    def prepare_data(self):
        # albums_df = pd.read_csv('albums.csv')
        albums_df['date']= pd.to_datetime(albums_df['date'])
        albums_df['date'] = albums_df['date'].dt.strftime('%Y-%m-%d')
        albums_df['year'] = albums_df['date'].apply(lambda x: x.split('-')[0])
        albums_df['popularity_red'] = albums_df['album_popularity'].apply(lambda x: int(x/5))
        albums_df['album_genre'] = albums_df['album_genre'].apply(lambda x: re.findall(r"'([^']*)'", x))
        
    def ohe_prep(self, df, column, new_name):
        """ 
        Create One Hot Encoded features of a specific column

        Parameters: 
            df (pandas dataframe): Spotify Dataframe
            column (str): Column to be processed
            new_name (str): new column name to be used

        Returns: 
            tf_df: One hot encoded features 
        """

        tf_df = pd.get_dummies(df[column])
        feature_names = tf_df.columns
        tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
        tf_df.reset_index(drop = True, inplace = True)    
        return tf_df

    # function to build entire feature set
    def create_feature_set(self, df):
        """ 
        Process spotify df to create a final set of features that will be used to generate recommendations

        Parameters: 
            df (pandas dataframe): Spotify Dataframe
            float_cols (list(str)): List of float columns that will be scaled 

        Returns: 
            final: final set of features 
        """

        #tfidf genre lists
        tfidf = TfidfVectorizer()
        tfidf_label = TfidfVectorizer()
        tfidf_matrix =  tfidf.fit_transform(df['album_genre'].apply(lambda x: " ".join(x)))
        tfidf_label_matrix = tfidf_label.fit_transform(df['album_label'])
        genre_df = pd.DataFrame(tfidf_matrix.toarray())
        label_df = pd.DataFrame(tfidf_label_matrix.toarray())
        genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
        label_df.columns = ['label' + "|" + i for i in tfidf_label.get_feature_names_out()]
        genre_df.reset_index(drop = True, inplace=True)
        label_df.reset_index(drop = True, inplace=True)

        #explicity_ohe = ohe_prep(df, 'explicit','exp')    
        year_ohe = self.ohe_prep(df, 'year','year') * 0.5
        popularity_ohe = self.ohe_prep(df, 'popularity_red','pop') * 0.15

        #concanenate all features
        final = pd.concat([genre_df, label_df, popularity_ohe, year_ohe], axis = 1)

        #add song id
        final['album_id']=df['album_id'].values

        return final
    
    def generate_album_recos(self, df, features, nonplaylist_features):
        """ 
        Pull songs from a specific playlist.

        Parameters: 
            df (pandas dataframe): spotify dataframe
            features (pandas series): summarized playlist feature
            nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist

        Returns: 
            non_playlist_df_top_40: Top 40 recommendations for that playlist
        """

        non_playlist_df = df[df['album_id'].isin(nonplaylist_features['album_id'].values)]
        non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('album_id', axis = 1).values, features.drop('album_id', axis = 1).values.reshape(1, -1))[:,0]
        non_playlist_df_top_40 = non_playlist_df.sort_values('sim',ascending = False).head(10)

        return non_playlist_df_top_40


sim_albums = similarAlbums()
sim_albums.prepare_data()
complete_album_feature_set = sim_albums.create_feature_set(albums_df)






start, end = 2000, 2020
albums = {}
albums_id = []
for index, row in albums_df.iterrows():
    if row['year'] in [str(x+1) for x in range(start, end)]:
        album = spotify.get_album(row['album_id'])
        album_details = {}
        album_tracks = []
        album_details['name'] = album['name']
        album_details['photo'] = album['images'][0]['url']
        album_details['total_tracks'] = album['total_tracks']
        album_details['artists_name'] = re.findall(r"'([^'\"]*)'", albums_df[albums_df['album_id']==row['album_id']]['artists_name'].item())
        artists_name_str = ''
        for i in range(len(album_details['artists_name'])):
            if i != len(album_details['artists_name']) - 1:
                artists_name_str = artists_name_str + album_details['artists_name'][i] + ', '
            else:
                artists_name_str = artists_name_str + album_details['artists_name'][i]
        album_details['artists_name'] = artists_name_str
        for item in album['tracks']['items']:
            album_track = {}
            album_track['id'] = item['id']
            album_track['name'] = item['name']
            album_track['track_number'] = item['track_number']
            album_track['duration'] = round((item['duration_ms']/(1000*60))%60, 2)
            album_track['artists'] = tracks_df[tracks_df['track_id']==item['id']]['artists_name'].item()
            artists_name_str = ''
            for i in range(len(album_track['artists'])):
                if i != len(album_track['artists']) - 1:
                    artists_name_str = artists_name_str + album_track['artists'][i] + ', '
                else:
                    artists_name_str = artists_name_str + album_track['artists'][i]
            album_track['artists'] = artists_name_str
            album_tracks.append(album_track) 
        album_details['tracks'] = album_tracks
        albums[album['id']] = album_details
        albums_id.append(album['id'])

albums_details = [albums_id, albums]


popular_albums_df = albums_df[albums_df['album_popularity'] >=50].sort_values(by=["album_popularity", "date"], ascending=False).head(10)
popular_albums_id = popular_albums_df['album_id'].tolist()

new_albums_df = albums_df.sort_values(by=['date'], ascending=False).head(10)
new_albums_id = new_albums_df['album_id'].tolist()




#************************ARTISTS***********************#



artists_df = pd.read_csv('artists.csv')


class similarArtists:
    
    def prepare_data(self):
        # artists_df = pd.read_csv('artists.csv')
        artists_df['popularity_red'] = artists_df['popularity'].apply(lambda x: int(x/5))
        artists_df['genres'] = artists_df['genres'].apply(lambda x: re.findall(r"'([^']*)'", x))
        
    def ohe_prep(self, df, column, new_name):
        """ 
        Create One Hot Encoded features of a specific column

        Parameters: 
            df (pandas dataframe): Spotify Dataframe
            column (str): Column to be processed
            new_name (str): new column name to be used

        Returns: 
            tf_df: One hot encoded features 
        """

        tf_df = pd.get_dummies(df[column])
        feature_names = tf_df.columns
        tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
        tf_df.reset_index(drop = True, inplace = True)    
        return tf_df

    # function to build entire feature set
    def create_feature_set(self, df):
        """ 
        Process spotify df to create a final set of features that will be used to generate recommendations

        Parameters: 
            df (pandas dataframe): Spotify Dataframe
            float_cols (list(str)): List of float columns that will be scaled 

        Returns: 
            final: final set of features 
        """

        #tfidf genre lists
        tfidf = TfidfVectorizer()
        tfidf_matrix =  tfidf.fit_transform(df['genres'].apply(lambda x: " ".join(x)))
        genre_df = pd.DataFrame(tfidf_matrix.toarray())
        genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
        genre_df.reset_index(drop = True, inplace=True)

        #explicity_ohe = ohe_prep(df, 'explicit','exp')    
        popularity_ohe = self.ohe_prep(df, 'popularity_red','pop') * 0.15
        
        #concanenate all features
        final = pd.concat([genre_df, popularity_ohe], axis = 1)

        #add song id
        final['artist_id']=df['artist_id'].values

        return final
    
    def generate_artist_recos(self, df, features, nonplaylist_features):
        """ 
        Pull songs from a specific playlist.

        Parameters: 
            df (pandas dataframe): spotify dataframe
            features (pandas series): summarized playlist feature
            nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist

        Returns: 
            non_playlist_df_top_40: Top 40 recommendations for that playlist
        """

        non_playlist_df = df[df['artist_id'].isin(nonplaylist_features['artist_id'].values)]
        non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('artist_id', axis = 1).values, features.drop('artist_id', axis = 1).values.reshape(1, -1))[:,0]
#         non_playlist_df_top_40 = non_playlist_df.sort_values('sim', ascending = False).head(10)
        non_playlist_df_top_40 = non_playlist_df.sort_values(by = ['sim', 'followers'], ascending = False).head(10)

        return non_playlist_df_top_40


sim_artists = similarArtists()
sim_artists.prepare_data()
complete_artist_feature_set = sim_artists.create_feature_set(artists_df)




popular_artists = artists_df.sort_values(by=['followers', 'popularity'], ascending=False).head(10)

popular_artists_id = popular_artists['artist_id'].tolist()
popular_artists_details = {}
for index, artist in popular_artists.iterrows():
    popular_artist_details = {}
    popular_artist_details['name'] = artist['artist_name']
    popular_artist_details['popularity'] = artist['popularity']
    popular_artist_details['followers'] = artist['followers']
    popular_artist_details['photo'] = artist['photo']
    popular_artists_details[artist['artist_id']] = popular_artist_details

popular_artists = [popular_artists_id, popular_artists_details, popular_albums_id,  new_albums_id, albums_details[1]]
# popular_artists = [popular_artists_id, popular_artists_details]



#***********************FLASK APP**********************#
app = Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/home', methods=['GET', "POST"])
def home():
    return render_template('home.html', artists_details = popular_artists)

@app.route('/playlists', methods=['GET', 'POST'])
def playlists():
    return render_template('playlists.html', recommended_songs = playlists_details)

@app.route('/albums', methods=['GET', 'POST'])
def albums():
    return render_template('albums.html', recommended_songs = albums_details)

@app.route('/album', methods=['GET', 'POST'])
def album():
    a_id = request.form['album_id']
    a_details = albums_details[1][a_id]
    
    return render_template('album.html', album_songs = a_details)

@app.route('/artist_albums', methods=['GET', 'POST'])
def artist_albums():
    art_id = request.form['artist_id']
    # art_details = popular_artists[1][art_id]
    artist_albums = spotify.get_artist_albums(art_id)

    global artist_albums_details
    artist_albums_details = {}
    artist_albums_id = []
    for album in artist_albums['items']:
        artist_album_details = {}
        artist_albums_id.append(album['id'])
        artist_album_details['name'] = album['name']
        artist_album_details['photo'] = album['images'][0]['url']
            
        album_details = spotify.get_album_tracks(album['id'])
        album_tracks_details = []
        for track in album_details['items']:
            album_track_details = {}
            album_track_details['id'] = track['id']
            album_track_details['name'] = track['name']
            # round((playlist_details['tracks']['items'][i]['track']['duration_ms']/(1000*60))%60, 2)
            album_track_details['duration'] = round((track['duration_ms']/(1000*60))%60, 2)
            album_track_details['track_number'] = track['track_number']
            album_tracks_details.append(album_track_details)
        artist_album_details['tracks'] = album_tracks_details
        artist_albums_details[album['id']] = artist_album_details
    albums_details = [artist_albums_id, artist_albums_details, popular_artists[1][art_id]['name']]
    return render_template('artist_albums.html', artists_details = albums_details)

@app.route('/artist_album', methods=['GET', 'POST'])
def artist_album():
    alb_id = request.form['album_id']

    album_details = spotify.get_album_tracks(alb_id)
    artist_album_details = {}
    artist_album_details['name'] = artist_albums_details[alb_id]['name']
    artist_album_details['photo'] = artist_albums_details[alb_id]['photo']
    album_tracks_details = []
    for track in album_details['items']:
        album_track_details = {}
        album_track_details['id'] = track['id']
        album_track_details['name'] = track['name']
        album_track_details['duration'] = round((track['duration_ms']/(1000*60))%60, 2)
        album_track_details['track_number'] = track['track_number']
        album_tracks_details.append(album_track_details)
    artist_album_details['tracks'] = album_tracks_details
    return render_template('/artist_album.html', artist_alb = artist_album_details)


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    p_id = request.form['playlist_id']
    playlist = rec.create_necessary_outputs(playlists_details[p_id]['name'], id_name, tracks_df)


  # #***********************CREATE PLAYLIST VECTOR**********************#
    complete_feature_set_playlist_vector, complete_feature_set_nonplaylist_vector = rec.generate_playlist_feature(complete_feature_set, playlist, 1.09)


  # #***********************RECOMMEND SONGS**********************#
    rec_songs = rec.generate_playlist_recos(tracks_df, complete_feature_set_playlist_vector, complete_feature_set_nonplaylist_vector)

    rec_songs_details = []
    for index, row in rec_songs.iterrows():
        rec_songs_details.append(row.to_dict())

    # for song in rec_songs_details:
    #     song['duration_ms'] = round((song['duration_ms']/(1000*60))%60, 2)

    artists_name = []
    index = 0
    for song in rec_songs_details:
        index += 1
        song['sr'] = index
        song['duration_ms'] = round((song['duration_ms']/(1000*60))%60, 2)
        artists_name_str = ''
        for i in range(len(song['artists_name'])):
            if i != len(song['artists_name']) - 1:
                artists_name_str = artists_name_str + song['artists_name'][i] + ', '
            else:
                artists_name_str = artists_name_str + song['artists_name'][i]
        song['artists_name'] = artists_name_str
        artists_name.append(artists_name_str)

    details = [playlists_details[p_id], rec_songs_details]

    return render_template('songs.html', recommended_songs = details)

if __name__=="__main__":
  app.run(debug=True)


