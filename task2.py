# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from python_speech_features import mfcc
import librosa
import os

# Function to extract features from audio files
def extract_features(audio_path):
    try:
        audio, _ = librosa.load(audio_path, res_type='kaiser_fast')
        mfccs = mfcc(audio, samplerate=16000, nfft=1200, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512)
        return mfccs
    except Exception as e:
        print("Error encountered while parsing audio file:", audio_path)
        return None

# Load dataset and preprocess
def load_and_preprocess_data(data_dir):
    emotions = {'angry': 0, 'happy': 1, 'neutral': 2, 'sad': 3}  # Add more emotions as needed
    X, y = [], []

    for emotion, label in emotions.items():
        emotion_folder = os.path.join(data_dir, emotion)
        for filename in os.listdir(emotion_folder):
            audio_path = os.path.join(emotion_folder, filename)
            features = extract_features(audio_path)
            if features is not None:
                X.append(features)
                y.append(label)

    X = np.array(X)
    y = to_categorical(y)

    return X, y

# Define the model
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load and preprocess data
data_directory = '/path/to/dataset'
X, y = load_and_preprocess_data(data_directory)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]
model = build_model(input_shape, num_classes)

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
