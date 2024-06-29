import os
from dotenv import load_dotenv
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import pickle
from genre_classifier_functions import extract_mfcc_from_single_file, plot_song_genres, download_and_convert_to_wav
import numpy as np
import json
from tensorflow import keras

# Spotify API credentials
load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# Get the current directory of the script
download_directory = os.path.dirname(__file__)
OUTPUT_JSON = "single_song_mfcc.json"
figure = "images/genre_percentage_plot.png"
start_time = 70  # Start time in seconds (1:10)
end_time = 100  # End time in seconds (1:40)

# Path to the JSON file of the single song to be classified
json_path = "single_song_mfcc.json"

# Path of the genre classifier model
model_path = "models/genre_classifier.keras"
# Load the saved keras model
model = keras.models.load_model(model_path)

def genre_predictor():
    '''Function to predict the genres of a song'''

    genre_names = ['Blues', 'Classical', 'Country', 'Disco', 'Electronic', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    # Load data from JSON file
    def load_data(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract features and labels
        X_song = np.array(data['mfcc_features'])
        
        return X_song

    # Load the data
    X_song = load_data(json_path)

    # Expand dimensions to match CNN input shape requirements
    X_song = np.expand_dims(X_song, axis=-1)

    # Make a prediction
    predictions = model.predict(X_song)

    # Assuming predictions is a 2D array (batch_size, num_classes)
    # and we are interested in the first prediction
    predicted_probabilities = predictions[0]

    # Convert the NumPy array to a list
    predicted_probabilities_list = predicted_probabilities.tolist()

    # Ensure genre_names has the same length as the number of classes
    assert len(genre_names) == len(predicted_probabilities_list), "Number of genre names does not match number of classes"

    # Convert probabilities to percentages and filter out zeros
    percentages = [prob * 100 for prob in predicted_probabilities_list]

    # Get the indices of the top 5 probabilities
    top_5_indices = np.argsort(percentages)[-5:]

    # Get the top 5 probabilities and corresponding genre names
    top_5_percentages = [percentages[i] for i in top_5_indices]
    top_5_genres = [genre_names[i] for i in top_5_indices]

    return top_5_genres, top_5_percentages


# Authenticate with Spotify
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load hit-predictor model
load_predictors = pickle.load(open('models/prehitctors.pkl', 'rb'))

# # cCreate the app itself. Custom CSS for styling
custom_css = """
<style>
body {
    font-family: 'Helvetica Neue', sans-serif;
}
.sidebar .sidebar-content .header {
    background-color: #333;
    color: white;
    padding: 10px;
    margin-bottom: 10px;
}
.centered {
    text-align: center;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Logo in sidebar
st.sidebar.image('images/Logohit.jpeg', use_column_width=True)

# Streamlit sidebar header
st.sidebar.header('Search Engine')

# Main content
st.title('Song hit predictor based on past hits')

# Fields for artist and song input
artist_name = st.sidebar.text_input('Artist Name', '')
song_title = st.sidebar.text_input('Song Title', '')

if artist_name and song_title:
    # Search for track
    results = sp.search(q=f'artist:{artist_name} track:{song_title}', type='track', limit=1)
    
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        spotify_song_link = track['external_urls']['spotify'] 
        
        # Adjusted column widths
        col1, spacer, col2 = st.columns([2.8, 0.3, 3.2])  # Adjust the widths as needed
        
        with col1:
            st.subheader('Track Found:')
            st.write(f"**{track['name']}** by **{track['artists'][0]['name']}**")
            st.write(f"Album: {track['album']['name']}")
            st.image(track['album']['images'][0]['url'], caption='Album Cover', width=200)
            st.write(f"Release Year: {track['album']['release_date'][:4]}")
        
        with col2:
            # Predict if the song is a hit
            audio_features = sp.audio_features(track['id'])
            if audio_features:
                feature_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                 'speechiness', 'acousticness', 'instrumentalness', 
                                 'liveness', 'valence', 'tempo', 'duration_ms', 
                                 'time_signature']
                
                feature_values = [audio_features[0][feature] for feature in feature_names]
                
                features_dict = {feature: value for feature, value in zip(feature_names, feature_values)}
                
                #create colum "decade" in the track dictionary
                release_year = int(track['album']['release_date'][:4])
                if 1960 <= release_year <= 1969:
                    features_dict['decade'] = 1960
                elif 1970 <= release_year <= 1979:
                    features_dict['decade'] = 1970
                elif 1980 <= release_year <= 1989:
                    features_dict['decade'] = 1980
                elif 1990 <= release_year <= 1999:
                    features_dict['decade'] = 1990
                elif 2000 <= release_year <= 2009:
                    features_dict['decade'] = 2000
                elif release_year >= 2010:
                    features_dict['decade'] = 2010
                
                #transform to df
                feature_df = pd.DataFrame([features_dict])
                
                features_input = feature_df.copy()  # Ensure to keep all columns needed for prediction
                
                hit_prediction = load_predictors.predict(features_input)
                hit_prediction_proba = load_predictors.predict_proba(features_input)

                st.subheader('Prediction:')

                # Display prediction results
                if hit_prediction[0] == 1:
                    st.markdown("<h1 style='color: green; text-align: center;'><span>★</span> HIT <span>★</span></h1>", unsafe_allow_html=True)
                    st.markdown(f"<h3>The song has a <span style='color:green;'>{hit_prediction_proba[0][1] * 100:.2f}%</span> probability of <span style='color:green;'>becoming a hit</span></h3>", unsafe_allow_html=True)

                else:
                    st.markdown("<h1 style='color: red; text-align: center;'>💔 FLOP 💔</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h3>The song has a <span style='color:red;'>{100 - hit_prediction_proba[0][0] * 100:.2f}%</span> probability of <span style='color:red;'>becoming a hit</span></h3>", unsafe_allow_html=True)

                artist_ = track['artists'][0]['name']
                song_ =  track['name']
                song_name = f"{song_} - {artist_}"
                print(f"{song_name}")
                song_path = download_and_convert_to_wav(song_name=song_name, output_dir=download_directory)

                if song_path != None: # This means the song exits and was downloaded from YouTube
                    # Get the MFCCs of the song to be calssified and plot the results
                    extract_mfcc_from_single_file(song_path, OUTPUT_JSON)
                    top_5_genres, top_5_percentages = genre_predictor()
                    plot_song_genres(top_5_genres, top_5_percentages, figure)
                else:
                    error_message = "Error. Not possible access to the song audio. Try another song"                
                st.image(figure, use_column_width=True)  

    else:
        st.subheader('Track Not Found')

st.markdown("---")
