import os
from pydub import AudioSegment
import json
import math
import librosa
import matplotlib.pyplot as plt
import yt_dlp as youtube_dl
import numpy as np


# Constants for audio properties
SR = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SR * DURATION
# Get the current directory of the script
download_directory = os.path.dirname(__file__)


def extract_mfcc_from_single_file(wav_file_path, output_file, n_mfcc=13, fft_size=2048, hop_size=512, segments=10):
    """
    Extracts MFCC features from a single audio file and saves them in a json file.

    :param wav_file_path (str): Path to the WAV files
    :param output_file (str): Path to the JSON file to save extracted MFCCs
    :param n_mfcc (int): Number of MFCC features to extract
    :param fft_size (int): Number of samples per FFT
    :param hop_size (int): Number of samples between successive frames
    :param segments (int): Number of segments per track
    :return: None
    """

    # Initialize dictionary to hold data
    mfcc_data = {
        "mfcc_features": []
    }

    segment_samples = int(SAMPLES_PER_TRACK / segments)
    expected_mfcc_vectors_per_segment = math.ceil(segment_samples / hop_size)

    # Load audio file
    signal, sample_rate = librosa.load(wav_file_path, sr=SR)

    # Divide audio file into segments and extract MFCCs from each segment
    for segment in range(segments):
        start_sample = segment_samples * segment
        end_sample = start_sample + segment_samples

        mfccs = librosa.feature.mfcc(y=signal[start_sample:end_sample], sr=sample_rate, n_mfcc=n_mfcc, n_fft=fft_size, hop_length=hop_size)
        mfccs = mfccs.T

        # Only store MFCCs if they have the expected number of vectors
        if len(mfccs) == expected_mfcc_vectors_per_segment:
            mfcc_data["mfcc_features"].append(mfccs.tolist())
            print(f"{wav_file_path}, segment: {segment + 1}")

    # Write extracted MFCC data to JSON file
    with open(output_file, "w") as json_file:
        json.dump(mfcc_data, json_file, indent=4)


def plot_song_genres(top_genres, top_percentages, filename):
    '''Function to plot the 3 main predicted genres of the song and group the rest as "Others"'''

    # Ensure there are more than 3 genres to group the rest as "Others"
    if len(top_genres) > 3:
        # Sort genres and percentages in descending order of percentages
        sorted_indices = np.argsort(top_percentages)[::-1]
        top_genres = np.array(top_genres)[sorted_indices].tolist()
        top_percentages = np.array(top_percentages)[sorted_indices].tolist()
        
        # Calculate the sum of percentages for the rest of the genres
        others_percentage = sum(top_percentages[3:])
        
        # Only keep the top 3 genres and their percentages
        top_genres = top_genres[:3]
        top_percentages = top_percentages[:3]
        
        # Append "Others" to the genres and its percentage
        top_genres.append("Others")
        top_percentages.append(others_percentage)

    plt.figure(figsize=(10, 10))  # Increased figure size for better visibility

    # Normalize probabilities to range between 0 and 1 for color intensity
    norm_percentages = np.array(top_percentages) / 100.0

    # Create a colormap
    cmap = plt.get_cmap('Blues')

    # Plot the bars with colors based on the probabilities
    bars = plt.barh(top_genres[::-1], top_percentages[::-1], color=cmap(norm_percentages[::-1]))

    # Remove the plot border
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    # Remove the x-axis
    plt.gca().axes.get_xaxis().set_visible(False)

    # Set title with increased font size
    plt.title('Genre Analysis', fontsize=48)#, fontweight='bold')

    # Annotate the bars with the percentage values with increased font size and horizontal offset
    for index, value in enumerate(top_percentages[::-1]):
        plt.text(value + 2, index, f'{value:.2f}%', va='center', ha='left', fontsize=40, color='black')  # Increased font size to 40 and added offset

    # Increase the size of the y-axis labels and add some padding
    plt.gca().set_yticklabels(top_genres[::-1], fontsize=40, ha='right')
    plt.gca().yaxis.set_tick_params(pad=20)  # Add padding to separate the genre names from the bars

    # Save the plot as a PNG file
    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.close()  # Close the plot


def download_and_convert_to_wav(song_name, output_dir):
    '''Function to download and convert YouTube video to MP3'''
    search_query = f"ytsearch:{song_name}"
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(search_query, download=True)
        video_id = info_dict['entries'][0]['id']
        temp_filename = os.path.join(output_dir, f"{video_id}.mp3")
        final_filename = os.path.join(output_dir, "song.mp3")
        os.rename(temp_filename, final_filename)
        # Convert MP3 to WAV and extract the segment from 1:10 to 1:40
        wav_file_path = os.path.join(download_directory, "song.wav")
        audio = AudioSegment.from_mp3(final_filename)
        segment = audio[70 * 1000: 100 * 1000]
        segment.export(wav_file_path, format="wav")
        print(f"Converted {final_filename} to {wav_file_path}")

        # Delete the original MP3 file
        os.remove(final_filename)
        print(f"Deleted original MP3 file {final_filename}")

        return wav_file_path