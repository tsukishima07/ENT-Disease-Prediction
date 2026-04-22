# 🩺 AI-Based ENT Disease Prediction System

## 📌 Overview
This project is an AI-powered system that predicts possible ENT (Ear, Nose, Throat) diseases based on user-input symptoms. It helps users get early insights into potential health conditions and suggests precautions and medicines.

The system uses machine learning models trained on symptom datasets to provide accurate predictions.

---

## 🚀 Features
- Predicts ENT diseases based on symptoms
- User-friendly interface (Python UI)
- Displays:
  - Disease description
  - Suggested precautions
  - Recommended medicines
- Pre-trained ML model for fast predictions

---

## 🛠️ Tech Stack
- Python
- Machine Learning (Scikit-learn / Pickle model)
- Pandas & NumPy
- GUI (Custom Python UI)
- 
 📂 Project Structure
ENT/
│── app_ui.py # Main UI application
│── model.pkl # Trained ML model
│── symptom_list.pkl # List of symptoms
│── cleaned_dataset.csv # Training dataset
│── disease_description.csv # Disease info
│── symptom_precaution.csv # Precautions
│── medicine.csv # Medicines data
│── Symptom-severity.csv # Symptom weights
│── feature_importance.png # Model insights
│── t.py # Additional script
