# ============================================================================
# ULTIMATE DIAGNOSTIC 
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*70)
print(" ULTIMATE DIAGNOSTIC TOOL")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')

# ============================================================================
# 1. EXAMINE IMAGE FEATURES
# ============================================================================

print("\n IMAGE FEATURES ANALYSIS")
print("-" * 50)

image_features_path = os.path.join(PROCESSED_DIR, 'image_features.csv')
if os.path.exists(image_features_path):
    image_df = pd.read_csv(image_features_path)
    print(f" Loaded image_features.csv")
    print(f"   Shape: {image_df.shape}")
    print(f"   Columns: {list(image_df.columns)}")
    
    # Check person_id distribution
    print(f"\n   Person ID distribution:")
    print(image_df['person_id'].value_counts())
    
    # Check if there are any patterns
    print(f"\n   First 5 rows of person_id:")
    print(image_df['person_id'].head())
    
    # Get feature columns (all except metadata)
    metadata_cols = ['person_id', 'expression', 'original_path', 'augmentation']
    feature_cols = [col for col in image_df.columns if col not in metadata_cols]
    print(f"\n   Number of feature columns: {len(feature_cols)}")
    
    # Show sample feature values for each person
    print(f"\n   Sample feature values for each person (first 10 features):")
    for person in ['Cynthia', 'kelvin', 'nick']:
        person_data = image_df[image_df['person_id'] == person]
        if len(person_data) > 0:
            sample = person_data.iloc[0]
            print(f"\n   {person}:")
            for i, col in enumerate(feature_cols[:10]):
                print(f"      {col}: {sample[col]:.4f}")
    
    # Check if features are all the same
    print(f"\n   Checking if features are identical for all samples:")
    first_row = image_df[feature_cols].iloc[0].values
    all_same = True
    for i in range(1, len(image_df)):
        if not np.array_equal(first_row, image_df[feature_cols].iloc[i].values):
            all_same = False
            break
    print(f"   All features identical? {all_same}")
    
    # Check feature statistics
    print(f"\n   Feature statistics across all samples:")
    all_features = image_df[feature_cols].values
    print(f"   Mean: {np.mean(all_features):.4f}")
    print(f"   Std: {np.std(all_features):.4f}")
    print(f"   Min: {np.min(all_features):.4f}")
    print(f"   Max: {np.max(all_features):.4f}")
    
    # Check if any features are constant
    constant_features = []
    for col in feature_cols:
        if image_df[col].nunique() == 1:
            constant_features.append(col)
    print(f"\n   Constant features (all same value): {len(constant_features)}")
    if len(constant_features) > 0:
        print(f"   First 5 constant features: {constant_features[:5]}")
    
else:
    print(f" Image features file not found")

# ============================================================================
# 2. EXAMINE AUDIO FEATURES
# ============================================================================

print("\n\n AUDIO FEATURES ANALYSIS")
print("-" * 50)

audio_features_path = os.path.join(PROCESSED_DIR, 'audio_features.csv')
if os.path.exists(audio_features_path):
    audio_df = pd.read_csv(audio_features_path)
    print(f" Loaded audio_features.csv")
    print(f"   Shape: {audio_df.shape}")
    print(f"   Columns: {list(audio_df.columns)}")
    
    # Check person_id distribution
    print(f"\n   Person ID distribution:")
    print(audio_df['person_id'].value_counts())
    
    # Check phrases
    print(f"\n   Phrase distribution:")
    print(audio_df['phrase'].value_counts())
    
    # Get feature columns
    metadata_cols = ['person_id', 'phrase', 'original_index', 'augmentation', 'original_path']
    feature_cols = [col for col in audio_df.columns if col not in metadata_cols]
    print(f"\n   Number of feature columns: {len(feature_cols)}")
    
    # Show sample feature values for each person
    print(f"\n   Sample feature values for each person (first 10 features):")
    for person in ['Cynthia', 'kelvin', 'nick']:
        person_data = audio_df[audio_df['person_id'] == person]
        if len(person_data) > 0:
            sample = person_data.iloc[0]
            print(f"\n   {person}:")
            for i, col in enumerate(feature_cols[:10]):
                print(f"      {col}: {sample[col]:.4f}")
    
    # Check if features are all the same
    print(f"\n   Checking if features are identical for all samples:")
    first_row = audio_df[feature_cols].iloc[0].values
    all_same = True
    for i in range(1, len(audio_df)):
        if not np.array_equal(first_row, audio_df[feature_cols].iloc[i].values):
            all_same = False
            break
    print(f"   All features identical? {all_same}")
    
    # Check feature statistics
    print(f"\n   Feature statistics across all samples:")
    all_features = audio_df[feature_cols].values
    print(f"   Mean: {np.mean(all_features):.4f}")
    print(f"   Std: {np.std(all_features):.4f}")
    print(f"   Min: {np.min(all_features):.4f}")
    print(f"   Max: {np.max(all_features):.4f}")
    
    # Check if any features are constant
    constant_features = []
    for col in feature_cols:
        if audio_df[col].nunique() == 1:
            constant_features.append(col)
    print(f"\n   Constant features (all same value): {len(constant_features)}")
    if len(constant_features) > 0:
        print(f"   First 5 constant features: {constant_features[:5]}")
    
    # Check if features are different between persons
    print(f"\n   Feature differences between persons (first 5 features):")
    for col in feature_cols[:5]:
        print(f"\n   {col}:")
        for person in ['Cynthia', 'kelvin', 'nick']:
            person_data = audio_df[audio_df['person_id'] == person]
            if len(person_data) > 0:
                values = person_data[col].values
                print(f"      {person}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

# ============================================================================
# 3. EXAMINE THE MODELS THEMSELVES
# ============================================================================

print("\n\ MODEL ANALYSIS")
print("-" * 50)

# Load face model
face_model_path = os.path.join(MODELS_DIR, 'face_classifier_fixed.joblib')
if os.path.exists(face_model_path):
    face_model = joblib.load(face_model_path)
    print(f"\n Face Model: {face_model_path}")
    print(f"   Type: {type(face_model).__name__}")
    print(f"   Classes: {face_model.classes_}")
    
    # Check feature importances
    if hasattr(face_model, 'feature_importances_'):
        importances = face_model.feature_importances_
        print(f"   Feature importances - min: {np.min(importances):.6f}, max: {np.max(importances):.6f}, mean: {np.mean(importances):.6f}")
        
        # Check if any features are extremely important
        high_importance = np.sum(importances > 0.01)
        print(f"   Features with importance > 0.01: {high_importance}")

# Load voice model
voice_model_path = os.path.join(MODELS_DIR, 'voice_model_fixed.joblib')
if os.path.exists(voice_model_path):
    voice_model = joblib.load(voice_model_path)
    print(f"\n Voice Model: {voice_model_path}")
    print(f"   Type: {type(voice_model).__name__}")
    print(f"   Classes: {voice_model.classes_}")
    
    # Check feature importances
    if hasattr(voice_model, 'feature_importances_'):
        importances = voice_model.feature_importances_
        print(f"   Feature importances - min: {np.min(importances):.6f}, max: {np.max(importances):.6f}, mean: {np.mean(importances):.6f}")
        
        # Check if any features are extremely important
        high_importance = np.sum(importances > 0.01)
        print(f"   Features with importance > 0.01: {high_importance}")

# ============================================================================
# 4. TEST WITH YOUR ACTUAL FILES
# ============================================================================

print("\n\n TESTING WITH YOUR ACTUAL FILES")
print("-" * 50)

# Test Cynthia's files
print(f"\n Testing Cynthia's face image:")
# We need to simulate since we don't have the actual feature extractor
print(f"   This would require the actual feature extraction pipeline")
print(f"   The issue is likely in how features are being extracted in demo.py")

print("\n" + "="*70)
print(" DIAGNOSTIC COMPLETE")
print("="*70)
print("\nPlease share the output of this script so I can see what's wrong with your data!")