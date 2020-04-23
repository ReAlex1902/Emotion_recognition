import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import datetime

# Extract features from a sound file
# mfcc, chroma, mel
def extract_feature(file_name):
    # with soundfile.SoundFile(file_name) as sound_file:
    # X = sound_file.read(dtype="float32")
    # sample_rate = sound_file.samplerate
    y, sr = librosa.load(file_name)
    hop_length = 512

    result = []

    # get standard deviation and mean of chroma
    stft = np.abs(librosa.stft(y))
    chroma_std = np.std(librosa.feature.chroma_stft(S=stft, sr=sr))
    result.append(chroma_std)
    chroma_mean = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr))
    result.append(chroma_mean)

    # get standard deviation and mean of mel
    mel_std = np.std(librosa.feature.melspectrogram(y, sr=sr))
    result.append(mel_std)
    mel_mean = np.mean(librosa.feature.melspectrogram(y, sr=sr))
    result.append(mel_mean)

    # Spectral Centroid Feature ('cent' is an array of values, 'cent_mean' is the mean of the array)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = np.mean(cent)
    result.append(cent_mean)

    # MFCC, delta, and delta second order
    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

    # get standard deviation and mean of mfcc
    mfcc_std = np.std(mfcc)
    result.append(mfcc_std)

    mfcc_mean = np.mean(mfcc)            
    result.append(mfcc_mean)

    # And the first-order differences 
    mfcc_delta = np.mean(librosa.feature.delta(mfcc))
    result.append(mfcc_delta)

    # And second-order differences
    mfcc_delta2 = np.mean(librosa.feature.delta(mfcc, order=2))
    result.append(mfcc_delta2)

    # RMS feature and mean
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    result.append(rms_mean)

    # Spectral Rolloff Mean
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.1)
    rolloff_mean = np.mean(rolloff)
    result.append(rolloff_mean)

    #Spectral Rolloff Range
    rolloff_range = np.ptp(rolloff)
    result.append(rolloff_range)

    # Zero Crossing Rate
    ZCR = librosa.feature.zero_crossing_rate(y)
    result.append(rolloff_mean)

    return result

# possible emotions
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised',

    'W': 'angry',
    'L': 'boredom',
    'E': 'disgust',
    'A': 'fearful',
    'F': 'happy',
    'T': 'sad',
    'N': 'neutral',
    
    'a': 'angry',
    'd': 'disgust',
    'f': 'fearful',
    'h': 'happy',
    'n': 'neutral',
    'sa': 'sad',
    'su': 'surprised',

    'ANG': 'angry',
    'FEA': 'fearful',
    'DIS': 'disgust',
    'HAP': 'happy',
    'SAD': 'sad',
    'NEU': 'neutral'
}

# Load the data and extract features for each sound file
def load_data():
    num_files = 0
    file_dics = []
    # get data from DataFlair dataset
    for file in glob.glob("C:\\Users\\Zack\\Desktop\\Emotion Data\\English Data\\AudioWAV\\Actor_*\\*.wav"):
        dir_name = os.path.dirname(file)
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        results = extract_feature(dir_name + '\\' + file_name)
        data = {'File Name': file_name, 'Emotion': emotion, 
            'Chroma Standard Deviation': results[0], 'Croma Mean': results[1],  
            'MEL Standard Deviation': results[2], 'MEL Mean': results[3],
            'Spectral Centroid Feature': results[4], 'MFCC Standard Deviation': results[5],
            'MFCC Mean': results[6], 'MFCC 1st Derivative': results[7], 'MFCC 2nd Derivative': results[8],
            'Root Mean Square Mean': results[9], 'Spectral Rolloff Mean': results[10], 
            'Spectral Rolloff Range': results[11], 'Zero Crossing Rate': results[12]
        }
        file_dics.append(data)
        num_files = num_files + 1
        print(num_files)

    # get data from German dataset
    for file in glob.glob("C:\\Users\\Zack\\Desktop\\Emotion Data\\German Data\\wav\\*.wav"):
        dir_name = os.path.dirname(file)
        file_name = os.path.basename(file)
        emotion = emotions[file_name[5]]
        results = extract_feature(dir_name + '\\' + file_name)
        data = {'File Name': file_name, 'Emotion': emotion, 
            'Chroma Standard Deviation': results[0], 'Croma Mean': results[1],  
            'MEL Standard Deviation': results[2], 'MEL Mean': results[3],
            'Spectral Centroid Feature': results[4], 'MFCC Standard Deviation': results[5],
            'MFCC Mean': results[6], 'MFCC 1st Derivative': results[7], 'MFCC 2nd Derivative': results[8],
            'Root Mean Square Mean': results[9], 'Spectral Rolloff Mean': results[10], 
            'Spectral Rolloff Range': results[11], 'Zero Crossing Rate': results[12]
        }
        file_dics.append(data)
        num_files = num_files + 1
        print(num_files)

    # get data from British dataset
    for file in glob.glob("C:\\Users\\Zack\\Desktop\\Emotion Data\\British Data\\*\\*.wav"):
        dir_name = os.path.dirname(file)
        file_name = os.path.basename(file)
        e = file_name[0]
        if e == 's':
            e+= file_name[1]
        emotion = emotions[e]
        results = extract_feature(dir_name + '\\' + file_name)
        data = {'File Name': file_name, 'Emotion': emotion, 
            'Chroma Standard Deviation': results[0], 'Croma Mean': results[1],  
            'MEL Standard Deviation': results[2], 'MEL Mean': results[3],
            'Spectral Centroid Feature': results[4], 'MFCC Standard Deviation': results[5],
            'MFCC Mean': results[6], 'MFCC 1st Derivative': results[7], 'MFCC 2nd Derivative': results[8],
            'Root Mean Square Mean': results[9], 'Spectral Rolloff Mean': results[10], 
            'Spectral Rolloff Range': results[11], 'Zero Crossing Rate': results[12]
        }
        file_dics.append(data)
        num_files = num_files + 1
        print(num_files)

        # get data from CREMA-D dataset
    for file in glob.glob("C:\\Users\\Zack\\Desktop\\Emotion Data\\AudioWAV\\*.wav"):
        dir_name = os.path.dirname(file)
        file_name = os.path.basename(file)
        split = file_name.split('_')
        emotion = emotions[split[2]]
        results = extract_feature(dir_name + '\\' + file_name)
        data = {'File Name': file_name, 'Emotion': emotion, 
            'Chroma Standard Deviation': results[0], 'Croma Mean': results[1],  
            'MEL Standard Deviation': results[2], 'MEL Mean': results[3],
            'Spectral Centroid Feature': results[4], 'MFCC Standard Deviation': results[5],
            'MFCC Mean': results[6], 'MFCC 1st Derivative': results[7], 'MFCC 2nd Derivative': results[8],
            'Root Mean Square Mean': results[9], 'Spectral Rolloff Mean': results[10], 
            'Spectral Rolloff Range': results[11], 'Zero Crossing Rate': results[12]
        }
        file_dics.append(data)
        num_files = num_files + 1
        print(num_files)

    return file_dics

start_time = datetime.datetime.now()

file_dics = load_data()
data_set = pd.DataFrame(file_dics)

data_set.to_csv ('C:\\Users\\Zack\\Desktop\\export_dataframe.csv', index = False, header=True)

end_time = datetime.datetime.now()
run_time = end_time - start_time
print("run time:", run_time)
