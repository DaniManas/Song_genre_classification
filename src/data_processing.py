import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    features = np.concatenate((np.mean(mfcc.T, axis=0), 
                               np.mean(chroma.T, axis=0), 
                               np.mean(spectral_contrast.T, axis=0)))
    return features
