#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import joblib
import librosa
import soundfile as sf
# from deepface import DeepFace  # Not needed; using MobileNet
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
KNOWN_KELVIN_EMB_PATH = DATA_DIR / 'image_features.csv'  # kelvin embeddings

class SecureRecommenderDemo:
    def __init__(self):
        self.face_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        # self.face_model = joblib.load(MODELS_DIR / 'face_model.joblib')  # No classifier trained; similarity used
        self.voice_model = joblib.load(MODELS_DIR / 'voice_model.joblib')
        self.recommender_model = joblib.load(MODELS_DIR / 'recommender_model.joblib')
        self.voice_scaler = joblib.load(MODELS_DIR / 'voice_scaler.joblib')
        self.voice_encoder = joblib.load(MODELS_DIR / 'voice_label_encoder.joblib')
        self.df_merged = pd.read_csv(DATA_DIR / 'merged_features.csv')
        self.image_features = pd.read_csv(KNOWN_KELVIN_EMB_PATH)
        self.kelvin_embs = self.image_features[self.image_features['person_id'] == 'kelvin'][['emb_0', 'emb_1', 'emb_2'] + [f'emb_{i}' for i in range(3,1280)]].mean().values.reshape(1,-1)  # Avg kelvin

    def extract_face_features(self, img_path):
        """MobileNet embedding (224x224) - matches notebook."""
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        emb = self.face_extractor.predict(img_array, verbose=0)[0]
        return emb.reshape(1,-1)

    def authenticate_face(self, img_path):
        """Cosine sim > 0.8 = kelvin."""
        emb = self.extract_face_features(img_path)
        sim = cosine_similarity(emb, self.kelvin_embs)[0][0]
        print(f'Face similarity: {sim:.3f}')
        return sim > 0.8, sim

    def extract_voice_features(self, audio_path):
        """MFCC + spectral (matches audio notebook)."""
        audio, sr = librosa.load(audio_path, sr=22050)
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0])
        energy = np.mean(librosa.feature.rms(y=audio))
        features = np.hstack([mfcc, spectral_rolloff, energy]).reshape(1,-1)
        return self.voice_scaler.transform(features)

    def verify_voice(self, audio_path):
        """Predict approved?"""
        features = self.extract_voice_features(audio_path)
        pred = self.voice_model.predict(features)[0]
        prob = self.voice_model.predict_proba(features)[0].max()
        label = self.voice_encoder.inverse_transform([pred])[0]
        print(f'Voice prediction: {label} (conf: {prob:.3f})')
        return label == 'kelvin' and prob > 0.7, prob

    def recommend_product(self, customer_id=None):
        """Sample rec from merged data."""
        if customer_id is None:
            customer_id = self.df_merged['customer_id'].sample(1).iloc[0]
        sample = self.df_merged[self.df_merged['customer_id'] == customer_id].drop(columns=['product_category']).fillna(0).iloc[0:1]
        pred = self.recommender_model.predict(sample)[0]
        proba = self.recommender_model.predict_proba(sample).max()
        self.le = LabelEncoder().fit(self.df_merged['product_category']) if not hasattr(self, 'le') else self.le
        product = self.le.inverse_transform([pred])[0]
        return customer_id, product, proba

    def run_flow(self, img_path, audio_path):
        print('🔐 Secure Flow Starting...')
        face_ok, face_score = self.authenticate_face(img_path)
        if not face_ok:
            print('❌ FACE DENIED')
            return
        print('✅ Face Authorized')
        voice_ok, voice_conf = self.verify_voice(audio_path)
        if not voice_ok:
            print('❌ VOICE DENIED')
            return
        print('✅ Voice Verified')
        cust, product, conf = self.recommend_product()
        print(f'🎉 APPROVED! Rec for {cust}: {product} (conf {conf:.3f})')

def main():
    parser = argparse.ArgumentParser(description='Multimodal Secure Demo')
    parser.add_argument('--image', default='data/images/kelvin_neutral.png', help='Image path')
    parser.add_argument('--audio', default='data/audio/kelvin_yes_approve.m4a', help='Audio path')
    parser.add_argument('--unauthorized', action='store_true', help='Test unauthorized (nick/wrong)')
    args = parser.parse_args()

    if args.unauthorized:
        args.image = 'data/images/kelvin_surprised.png'  # Sim wrong
        args.audio = 'data/audio/nick_confirm_tx.m4a'
        print('🧪 Running UNAUTHORIZED sim...')

    demo = SecureRecommenderDemo()
    demo.run_flow(args.image, args.audio)

if __name__ == '__main__':
    main()

