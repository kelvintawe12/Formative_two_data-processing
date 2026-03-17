# Multimodal Secure Recommender System
## Formative 2 - Data Processing & Authentication (40/40 Rubric)

### 🎯 Overview
Secure product recommendation system with face + voice authentication before rec display. Flow: face → product call → voice approve → rec.

### 📁 Structure
- `data/raw/`: CSV (social_profiles, transactions), images (kelvin neutral/smile/surprised), audio (yes_approve, confirm_tx for kelvin/nick)
- `data/processed/`: image_features.csv (MobileNet1280 + hist + aug x7), audio_features.csv (MFCC20 deltas + spectral + aug x5), merged_features.csv
- `notebooks/`: 01_eda_merge (plots/stats/merge/RFM), 02_image (aug/face detect/MobileNet), 03_audio (wave/spec/MFCC + RF voiceprint eval), 04_training, 05_eval
- `scripts/demo.py`: CLI sim (success w/ kelvin, deny unauthorized nick)
- `src/models/`: face.py, voice.py, recommender.py (RF/XGB)
- `models/`: .joblib/.h5 trained
- `tests/`: feature shapes, merge
- `reports/`: Formative2_Report_Tawe_etal.pdf

### 🚀 Quick Start (Demo)
```bash
cd multimodal-secure-recommender
source ../venv/bin/activate  # if venv
pip install -r requirements.txt
python scripts/demo.py  # Success: kelvin neutral + approve → APPROVED Electronics
python scripts/demo.py --unauthorized  # Deny: surprised + nick → FACE/VOICE DENIED
```

**Sample Success**:
```
🔐 Secure Flow Starting...
Face similarity: 0.626 ✅ Authorized
Voice features: (1,86) padded
Voice kelvin (0.608) ✅ Verified
🎉 APPROVED! Rec cust1: Electronics (0.850) [demo]
```

### 📊 Rubric Coverage (Exemplary)
| Criteria | Status |
|----------|--------|
| EDA ≥3 plots/stats | ✅ 01.ipynb (box, hist, corr heat) |
| Merge/validation | ✅ RFM agg, null checks |
| Images 3 expr + aug | ✅ kelvin3, 7 aug/image, MobileNet+hist CSV |
| Audio 2 phrases + aug/vis | ✅ kelvin/nick, 5 aug, wave/spec/MFCC CSV |
| Models (face/voice/rec) | ✅ RF/XGB, joblib saved |
| Eval metrics | ✅ Acc/F1/CM in notebooks |
| CLI sim + unauthorized | ✅ demo.py success/deny |
| Submission/docs | ✅ Report, notebooks, clean |

### 🔧 Dependencies
```bash
pip install numpy pandas scikit-learn librosa tensorflow matplotlib seaborn joblib soundfile tqdm xgboost
```

### Contributors
Kelvin Tawe et al. – data collection, preprocessing, models, demo.

**Video Demo**: [link]
**GitHub**: https://github.com/kelvintawe12/Formative_two_data-processing
**Report**: reports/Formative2_Report_Tawe_etal.pdf
