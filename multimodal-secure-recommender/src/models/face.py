import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import warnings
warnings.filterwarnings('ignore')

class FaceModel:
    def __init__(self):
        self.extractor = load_model('../../models/face_feature_extractor.h5')
        self.classifier = joblib.load('../../models/face_model.joblib')
        # Load known kelvin avg embedding
        import pandas as pd
        df = pd.read_csv('../../data/processed/image_features.csv')
        kelvin_cols = [col for col in df.columns if col.startswith('emb_')]
        self.kelvin_emb = df[df['person_id'] == 'kelvin'][kelvin_cols].mean().values.reshape(1,-1)

    def extract_embedding(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return self.extractor.predict(x, verbose=0)[0].reshape(1,-1)

    def authenticate(self, img_path, threshold=0.8):
        emb = self.extract_embedding(img_path)
        sim = cosine_similarity(emb, self.kelvin_emb)[0][0]
        authorized = sim > threshold
        return authorized, sim

