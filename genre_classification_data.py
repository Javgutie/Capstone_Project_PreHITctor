import librosa
import json
import math
import os


# Paths and audio configuration constants
SCRIPT_DIRECTORY = os.path.dirname(__file__) # Get the current directory of the script
AUDIO_DATA_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "music_dataset")
TRACK_DURATION_SECONDS = 30  
AUDIO_SAMPLE_RATE = 22050
TOTAL_SAMPLES_PER_TRACK = AUDIO_SAMPLE_RATE * TRACK_DURATION_SECONDS
JSON_OUTPUT_FILE = "genre_classifier.json"


def process_and_save_mfcc(dataset_directory, json_output_file, num_mfcc=13, fft_points=2048, hop_size=512, num_segments=10):
    """
    Processes audio files to extract MFCC features and saves them along with genre labels in a JSON file.

    :param dataset_directory (str): Directory path containing the dataset
    :param json_output_file (str): File path for the output JSON file
    :param num_mfcc (int): Number of MFCC features to compute
    :param fft_points (int): Number of points for FFT computation
    :param hop_size (int): Number of samples between successive frames
    :param num_segments (int): Number of segments to divide each track into
    :return: None
    """

    # Initialize dictionary to store MFCC data
    mfcc_dataset = {
        "genres": [],
        "labels": [],
        "mfcc_features": []
    }

    samples_per_segment = int(TOTAL_SAMPLES_PER_TRACK / num_segments)
    expected_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_size)

    # Get the list of subdirectories (subgenres) and sort them alphabetically
    genre_directories = sorted([os.path.join(dataset_directory, dir_name) for dir_name in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, dir_name))])

    # Process each genre directory
    for genre_index, genre_dir in enumerate(genre_directories):

        # Extract genre name from directory path and add to genre list
        genre_name = os.path.basename(genre_dir)
        mfcc_dataset["genres"].append(genre_name)

        # Process each audio file in the genre directory
        for audio_file in os.listdir(genre_dir):
            if audio_file.endswith(".wav"):  # Process only .wav files
                audio_file_path = os.path.join(genre_dir, audio_file)
                audio_signal, sample_rate = librosa.load(audio_file_path, sr=AUDIO_SAMPLE_RATE)

                # Segment the audio file and extract MFCCs from each segment
                for segment in range(num_segments):
                    start_sample = samples_per_segment * segment
                    end_sample = start_sample + samples_per_segment

                    mfcc = librosa.feature.mfcc(y=audio_signal[start_sample:end_sample], sr=sample_rate, n_mfcc=num_mfcc, n_fft=fft_points, hop_length=hop_size)
                    mfcc = mfcc.T

                    # Store MFCCs only if they have the expected length
                    if len(mfcc) == expected_mfcc_vectors_per_segment:
                        mfcc_dataset["mfcc_features"].append(mfcc.tolist())
                        mfcc_dataset["labels"].append(genre_index)
                        print(f"{audio_file_path}, segment: {segment + 1}")

    # Write MFCC data to JSON file
    with open(json_output_file, "w") as json_file:
        json.dump(mfcc_dataset, json_file, indent=4)

process_and_save_mfcc(AUDIO_DATA_DIRECTORY, JSON_OUTPUT_FILE)
