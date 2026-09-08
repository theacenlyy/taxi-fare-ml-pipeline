# 🚕 Production-Grade Fare Prediction ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/Architecture-Modular%20Pipeline-green.svg)]()

## 📌 Architectural Overview
This repository engineered for **Cellula Technologies (ML Track Tasks 1 & 2)** delivers an end-to-end Machine Learning pipeline designed to predict ride fares under high variance and missing data conditions. 

The core focus of this implementation is **strict leakage prevention, mathematical scalability, and deployment readiness**.
import joblib
import pandas as pd

# Load inference artifact
pipeline = joblib.load('final_fare_prediction_pipeline.pkl')

# Real-time Prediction Engine Pipeline
sample_data = pd.DataFrame([{
    'Distance': 12.5,
    'JFK_Dist': 3.2,
    'Car_Condition': 'Good',
    'Traffic_Conditions': 'High',
    'Hour': 18,
    'Weather': 'Rainy'
}])

predicted_fare = pipeline.predict(sample_data)
print(f"Predicted Fare Output: ${predicted_fare[0]:.2f}")
