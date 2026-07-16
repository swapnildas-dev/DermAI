# DermAI

![Python](https://img.shields.io/badge/Python-3.11-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red) ![License](https://img.shields.io/badge/license-MIT-green)

AI-powered skin lesion classifier for early detection of skin conditions.

**[Live Demo](https://dermai-7r4wobn2yv34nco2jay3jw.streamlit.app)**

## Why this exists

Skin cancer is the most common cancer in the United States, with over 5 million cases diagnosed each year. Melanoma alone accounts for the majority of skin cancer deaths — yet when caught early, the five-year survival rate is 99%. If it spreads, that drops to 35%.

The problem is access. Most people don't see a dermatologist regularly, and by the time a concerning lesion gets checked, it can be too late. DermAI was built to lower that barrier — giving anyone a way to get an instant preliminary assessment of a skin lesion before deciding whether to seek professional care. It won't replace a dermatologist, but it can be the nudge that gets someone in the door early enough to matter.

## What it does

DermAI analyzes close-up photos of skin lesions and classifies them into 7 conditions, trained on the HAM10000 dataset (2018) — a collection of 10,000+ clinical dermoscopy images published by researchers from the Medical University of Vienna and ViDIR Group, and featured in Nature Scientific Data. It combines image analysis with a symptom questionnaire to provide a risk assessment and early detection screening.

**Conditions detected:**
- Melanoma
- Basal Cell Carcinoma
- Actinic Keratosis
- Benign Keratosis-like Lesion
- Dermatofibroma
- Nevus
- Vascular Lesion

## Features

- Upload a photo of a skin lesion for instant AI analysis
- Symptom questionnaire to improve assessment accuracy
- Risk level output with condition explanation
- Condition glossary and skin health tips

## Tech Stack

- **Model:** TensorFlow / Keras (CNN with fine-tuned backbone)
- **Frontend:** Streamlit
- **Data processing:** NumPy, scikit-learn, Pillow
- **Deployment:** Streamlit Cloud

## Run locally

```bash
git clone https://github.com/swapnildas-dev/DermAI.git
cd DermAI
pip install -r requirements.txt
streamlit run app.py
```

## Current Limitations

DermAI is currently trained on professional dermoscopy images — the kind taken with specialized cameras at a dermatologist's office. Consumer phone photos have different lighting, angle, and resolution, which affects accuracy. This is a known limitation being addressed in V2.

## Roadmap — V2

- **Phone photo support** — retrain and adapt the model to work with everyday smartphone photos so anyone can use it without clinical equipment
- **Dermatologist matching** — use the classification result alongside the user's location and health insurance to surface nearby in-network dermatologists, making it easier to go from "something looks off" to an actual appointment

## Disclaimer

DermAI is not a medical diagnostic tool. Always consult a board-certified dermatologist for any skin concerns.
