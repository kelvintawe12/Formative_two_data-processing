import joblib
import pandas as pd
from pathlib import Path

class RecommenderModel:
    def __init__(self, model_path='../../models/recommender_model.joblib'):
        self.model = joblib.load(model_path)
        # Assume label encoder saved or fit from data
        self.df = pd.read_csv('../../data/processed/merged_features.csv')
        from sklearn.preprocessing import LabelEncoder
        self.le = LabelEncoder()
        self.le.fit(self.df['product_category'])  # Replace with actual target col

    def predict(self, features_df):
        pred = self.model.predict(features_df)[0]
        proba = self.model.predict_proba(features_df).max()
        product = self.le.inverse_transform([pred])[0]
        return product, proba

    def sample_recommendation(self, customer_id=None):
        if customer_id is None:
            customer_id = self.df['customer_id'].sample(1).iloc[0]
        sample = self.df[self.df['customer_id'] == customer_id].drop(columns=['product_category', 'customer_id']).fillna(0).iloc[0:1]
        return self.predict(sample)

