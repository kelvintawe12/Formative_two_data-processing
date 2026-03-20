# Multimodal Secure Recommender System
## Formative 2 - Data Processing & Authentication 

### Overview
This is a secure product recommendation system that uses **multimodal biometric authentication** combining face recognition and voice verification. To receive personalized product recommendation, users must pass both authentication layers.

### Features
- **Face Recognition**: Random Forest classifier trained on 63 facial images (3 expressions × 3 users × 7 augmentations)
- **Voice Verification**: Random Forest classifier using MFCC features from 36 audio samples (2 phrases × 3 users × 6 augmentations)
- **Product Recommendation**: XGBoost model trained on customer transaction data with personalized recommendations
- **Security**: Unauthorized face or voice attempts are immediately rejected

### Project Structure


### Quick Start

#### Installation
```bash
# Clone the repository
git clone https://github.com/kelvintawe12/Formative_two_data-processing.git
cd Formative_two_data-processing/multimodal-secure-recommender

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### Run the Demo
```bash
cd scripts
python demo.py
```

### Demo menu options
┌──────────────────────────────────────────────────┐
│  1. Run ALL scenarios (full demo)                │
│  2. Authorized user - Cynthia                    │
│  3. Authorized user - Kelvin                     │
│  4. Authorized user - Nick                       │
│  5. Unauthorized face attempt                    │
│  6. Unauthorized voice attempt                   │
│  7. Exit                                         │
└──────────────────────────────────────────────────┘

### Sample output
======================================================================
  SCENARIO: AUTHORIZED USER - Cynthia
======================================================================

   STEP 1: FACE RECOGNITION
  ----------------------------------------
  • Analyzing face: Cynthia_neutral.jpeg
   Face recognized as: Cynthia
  • Confidence: 86.06%

   STEP 2: VOICE VERIFICATION
  ----------------------------------------
  • Verifying voice: Cynthia_yes_approve.m4a
   Voice verified as: Cynthia
  • Confidence: 69.14%

   STEP 3: PRODUCT RECOMMENDATION
  ----------------------------------------
  • Generating recommendations for: Cynthia

   ALL CHECKS PASSED - ACCESS GRANTED

   TOP RECOMMENDATIONS:
     1. Books        [█████████░░░░░░░░░░░] 45.0%
     2. Clothing     [█████░░░░░░░░░░░░░░░] 25.0%
     3. Electronics  [███░░░░░░░░░░░░░░░░░] 15.0%

  User: Cynthia | Face: 86.06% | Voice: 69.14%
  Top Pick: Books

### Unauthorized attempt

ACCESS DENIED: Face not recognized

#### Unauthorized Voice (Cynthia's face + Kelvin's voice):

Face recognized as: Cynthia
VOICE MISMATCH! Expected Cynthia, got kelvin
ACCESS DENIED: Voice verification failed

### Data Collection & Processing

**Image Data**

* 3 users: Cynthia, kelvin, nick

* 3 expressions per user: neutral, smile, surprised

* 7 augmentations per image: rotation, flipping, brightness, contrast, grayscale, shear, zoom

* Feature extraction: MobileNet embeddings (1280) + histogram features (768) + statistics (4)

* Total: 63 samples × 2052 features

**Audio Data**

* 2 phrases per user: "Yes approve", "Confirm transaction"

* 5 augmentations per sample: pitch shift (up/down), time stretch (faster/slower), background noise

* Feature extraction: 20 MFCC coefficients + deltas + delta-deltas + spectral features

* Total: 36 samples × 126 features

### Dependencies 
```bash
pip install -r requirements.txt
```

### Contributors

1. Kelvin Tawe – Data collection, preprocessing, model development, demo implementation

2. Cynthia Mutie – Audio collection, feature extraction, evaluation

3. Nick-Lemy – Image collection, augmentation, documentation

### Links

* GitHub Repository: https://github.com/kelvintawe12/Formative_two_data-processing

* Video Demo: https://youtu.be/qANhWk67rqk

* Final Report: 