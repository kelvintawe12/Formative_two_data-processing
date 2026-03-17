import joblib
import librosa
import numpy as np
from pathlib import Path

class VoiceModel:
    def __init__(self):
        self.model = joblib.load('../../models/voice_model.joblib')
        self.scaler = joblib.load('../../models/voice_scaler.joblib')
        self.encoder = joblib.load('../../models/voice_label_encoder.joblib')

    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=22050)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y+0.0001, sr=sr)[0])
        rms = np.mean(librosa.feature.rms(y=y))
        features = np.hstack([mfccs, rolloff, rms]).reshape(1, -1)
        return self.scaler.transform(features)

    def verify(self, audio_path, threshold=0.7):
        features = self.extract_features(audio_path)
        pred = self.model.predict(features)[0]
        prob = self.model.predict_proba(features)[0].max()
        is_approved = self.encoder.inverse_transform([pred])[0] == 'approve' and prob > threshold
        return is_approved, prob, self.encoder.inverse_transform([pred])[0]

