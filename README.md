# Smart Crop Recommendation System 🌱

A machine learning web app that recommends the most suitable crop based on soil nutrients and weather conditions.

**Live demo:** https://smart-crop-recommendation-systemgit-dqetjw4m5okw7hgjk93pok.streamlit.app/
**Project page:** https://foadshahrokh.online/projects/1

## Overview

The user enters seven values: nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, soil pH and rainfall. A Random Forest classifier predicts the best crop out of 22 classes and shows:

- the recommended crop and the model's confidence,
- the top 3 suggestions with their probabilities,
- the feature importance of the model.

## Dataset

`data/Crop_recommendation.csv`: 2,200 samples, 22 crops (100 samples per crop), 7 numeric features.
Source: [Crop Recommendation Dataset (Kaggle)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

## Models

The data was split 80/20 (`random_state=42`) and three classifiers were compared:

| Model | Test accuracy |
|---|---|
| Logistic Regression | 94.5% |
| Decision Tree | 98.6% |
| Random Forest | 99.3% |

The Random Forest model is used in the app (`models/random_forest_model.pkl`).
Training and evaluation are in the notebook in `notebooks/`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Streamlit · joblib

## Author

Foad Nejad Shahrokh Abadi · [foadshahrokh.online](https://foadshahrokh.online) · [LinkedIn](https://www.linkedin.com/in/foad-nejad-shahrokh-abadi-628534342/)
