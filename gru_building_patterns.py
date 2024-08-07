from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, BatchNormalization, Dropout, Dense, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np

# Load data
numerical_data = pd.read_csv('chroma_features.csv')

# Ensure the data has three dimensions: (num_samples, timesteps, features)
timesteps = 1500  # or whatever number of timesteps you expect
features = 12

# Reshape data
num_samples = len(numerical_data) // timesteps
X = numerical_data.values[:num_samples * timesteps].reshape((num_samples, timesteps, features))

# Split data
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# Normalize data
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_val = (X_val - X_val.min()) / (X_val.max() - X_val.min())

model_gru = Sequential()
model_gru.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model_gru.add(GRU(128, return_sequences=True, kernel_regularizer=l2(0.001)))
model_gru.add(BatchNormalization())
model_gru.add(Dropout(0.3))
model_gru.add(GRU(128, return_sequences=True, kernel_regularizer=l2(0.001)))
model_gru.add(BatchNormalization())
model_gru.add(Dropout(0.2))
model_gru.add(Dense(features, activation='softmax'))

model_gru.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
y_train = X_train
y_val = X_val
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_gru.keras', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

model_gru.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping, model_checkpoint, reduce_lr])
model_gru.load_weights('best_model_gru.keras')
def generate_music(model, seed, length):
    generated = []
    input_seq = seed

    for _ in range(length):
        prediction = model.predict(input_seq, verbose=0) # , verbose=0
        generated.append(prediction[0, -1, :]) # appending last timestep
        input_seq = np.roll(input_seq, -1)
        input_seq[0, -1, :] = prediction[0, -1, :] # [0, -1, :]

    return np.array(generated)

seed = X_val[0].reshape(1, -1, 12)
generated_music = generate_music(model_gru, seed, 1500)

# Convert numerical data to MIDI
from midiutil import MIDIFile

def numerical_to_midi(numerical_data, output_file):
    midi = MIDIFile(1)
    track = 0
    time = 0
    channel = 0
    duration = 1
    volume = 100

    midi.addTempo(track, time, 120)

    for i, note in enumerate(numerical_data):
        pitch = np.argmax(note) % 12
        midi.addNote(track, channel, pitch + 60, time + i, duration, volume)

    with open(output_file, "wb") as output:
        midi.writeFile(output)

numerical_to_midi(generated_music, 'generated_music.mid')
#!fluidsynth -ni './GeneralUser GS 1.471/GeneralUser GS v1.471.sf2' generated_music.mid -F generated_music.wav -r 44100^C
#!ffmpeg -i generated_music.wav generated_music.mp3

# print('Run the following commands to convert mid file to mp3 on successfully training the model')
# print("\n\nfluidsynth -ni './GeneralUser GS 1.471/GeneralUser GS v1.471.sf2' generated_music.mid -F generated_music.wav -r 44100")
# print("\nffmpeg -i generated_music.wav generated_music.mp3")

print('\n\nOr just use\n\n')
print("fluidsynth -ni './GeneralUser GS 1.471/GeneralUser GS v1.471.sf2' generated_music.mid -F generated_music.wav -r 44100 ; ffmpeg -i generated_music.wav generated_music.mp3\n")
