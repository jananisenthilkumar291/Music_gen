import librosa
import numpy as np
import os
import pandas as pd

def extract_chroma_features(audio_file, sr=22050):
    y, sr = librosa.load(audio_file, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma_stft

def chroma_to_numerical(chroma_features):
    chroma_features = chroma_features.T
    chroma_features = (chroma_features - chroma_features.min()) / (chroma_features.max() - chroma_features.min())
    return chroma_features[:1500,:]

# Directory containing the .wav files
wav_dir = './wav_tracks/'

# List to hold all the numerical data
all_numerical_data = []

# Process each .wav file in the directory
for file_name in os.listdir(wav_dir):
    if file_name.endswith('.wav'):
        audio_file = os.path.join(wav_dir, file_name)
        chroma_features = extract_chroma_features(audio_file)
        numerical_data = chroma_to_numerical(chroma_features)
        print(len(numerical_data),' - ',file_name)
        all_numerical_data.append(numerical_data)

# Combine all numerical data into a single numpy array
combined_data = np.vstack(all_numerical_data)

# Convert the numpy array to a pandas DataFrame
df = pd.DataFrame(combined_data)

# Save the DataFrame to a CSV file
df.to_csv('chroma_features.csv', index=False)

print("Chroma features have been extracted and saved to chroma_features.csv")

