import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import base64
import os
# Load datasets
desc_df = pd.read_csv("disease_description.csv")
prec_df = pd.read_csv("symptom_precaution.csv")
medicine_df = pd.read_csv("medicine.csv")

# Load trained model and symptom list
model = joblib.load("model.pkl")
symptom_list = joblib.load("symptom_list.pkl")

# Function to set background and styling
def set_background():
    # Add custom CSS for a professional look
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
        }
        .stButton button {
            background: linear-gradient(90deg, #4e73df 0%, #3a66d6 100%);
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 0.6rem 1.2rem;
            border: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #3a66d6 0%, #2e59d9 100%);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }
        h1, h2, h3 {
            color: #2c3e50;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stSuccess {
            background: linear-gradient(90deg, #1cc88a 0%, #13a673 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(28, 200, 138, 0.2);
        }
        .stInfo {
            background: linear-gradient(90deg, #36b9cc 0%, #2a91a2 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(54, 185, 204, 0.2);
        }
        .stWarning {
            background: linear-gradient(90deg, #f6c23e 0%, #e0b137 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(246, 194, 62, 0.2);
        }
        .stMultiSelect {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .css-1d391kg, .css-12oz5g7 {
            padding: 2rem;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 2px 12px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        .stMarkdown p {
            font-size: 16px;
            line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply background and styling
set_background()

# App header with professional styling
st.markdown("""
<div style='text-align: center; padding: 1.5rem; background: linear-gradient(90deg, #4e73df11 0%, #4e73df22 100%); border-radius: 10px; margin-bottom: 1.5rem;'>
    <h1 style='color: #4e73df; font-weight: 600; margin-bottom: 0.5rem;'>🩺 Clinical Diagnostic Assistant</h1>
    <h3 style='color: #5a5c69; font-weight: 400; margin-top: 0;'>Evidence-Based Differential Diagnosis System</h3>
</div>
""", unsafe_allow_html=True)

# Create a container with a professional card-like appearance
st.markdown("""
<div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 2rem;'>
    <p style='color: #5a5c69; font-size: 16px;'>This clinical decision support system utilizes machine learning algorithms to analyze presenting symptoms and generate potential differential diagnoses. For optimal diagnostic accuracy, please select all clinical manifestations you are currently experiencing.</p>
    <p style='color: #5a5c69; font-size: 14px; font-style: italic;'>Note: This tool is designed to assist healthcare professionals and should not replace proper medical consultation. Always consult with a qualified healthcare provider for definitive diagnosis and treatment.</p>
</div>
""", unsafe_allow_html=True)

# Create three columns for better layout
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown("<h4 style='color: #4e73df; font-weight: 500;'>Clinical Presentation</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color: #5a5c69;'>Select all presenting signs and symptoms:</p>", unsafe_allow_html=True)

with col2:
    # App content - centered image with shadow and border
    st.markdown("""
    <div style='display: flex; justify-content: center;'>
        <img src='https://img.freepik.com/free-vector/doctor-character-background_1270-84.jpg' style='width: 300px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
    </div>
    """, unsafe_allow_html=True)

# Symptom selection with a professional container
st.markdown("<div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
selected_symptoms = st.multiselect("Select all clinical manifestations", options=symptom_list)
st.markdown("</div>", unsafe_allow_html=True)

# Predict button with professional styling
st.markdown("<div style='padding: 1rem 0;'></div>", unsafe_allow_html=True)
if st.button("🔬 Generate Differential Diagnosis"):
    if selected_symptoms:
        # 1. Input vector
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]

        # 2. Prediction
        prediction = model.predict([input_vector])[0]
        
        # Create a professional clinical report container
        st.markdown("""<div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-top: 1rem;'>
            <h3 style='color: #2c3e50; border-bottom: 1px solid #eaeaea; padding-bottom: 0.5rem;'>Clinical Assessment Report</h3>
        </div>""", unsafe_allow_html=True)
        
        # Primary diagnosis
        st.markdown(f"""<div style='background-color: #f8f9fc; padding: 1rem; border-left: 4px solid #4e73df; border-radius: 5px; margin: 1rem 0;'>
            <h4 style='color: #4e73df; margin: 0;'>Primary Differential Diagnosis</h4>
            <p style='font-size: 18px; font-weight: 500; margin-top: 0.5rem;'>{prediction}</p>
        </div>""", unsafe_allow_html=True)

        # 3. Clinical Description
        desc_row = desc_df[desc_df["Disease"].str.lower() == prediction.lower()]
        if not desc_row.empty:
            description = desc_row["Description"].values[0]
            st.markdown(f"""<div style='background-color: #f8f9fc; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
                <h4 style='color: #2c3e50; margin: 0;'>Clinical Description</h4>
                <p style='margin-top: 0.5rem;'>{description}</p>
            </div>""", unsafe_allow_html=True)

        # 4. Clinical Management
        prec_row = prec_df[prec_df["Disease"].str.lower() == prediction.lower()]
        if not prec_row.empty:
            precautions = [prec_row[f"Precaution_{i}"].values[0] for i in range(1, 5) if prec_row[f"Precaution_{i}"].values[0]]
            
            st.markdown(f"""<div style='background-color: #f8f9fc; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
                <h4 style='color: #2c3e50; margin: 0;'>Clinical Management Recommendations</h4>
                <ul style='margin-top: 0.5rem;'>
                    {''.join([f'<li>{p}</li>' for p in precautions if p])}
                </ul>
            </div>""", unsafe_allow_html=True)

        # 5. Pharmacological Interventions
        med_row = medicine_df[medicine_df["Disease"].str.lower() == prediction.lower()]
        if not med_row.empty:
            meds = med_row["Medicine"].values[0].split(";")
            
            st.markdown(f"""<div style='background-color: #f8f9fc; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
                <h4 style='color: #2c3e50; margin: 0;'>Pharmacological Interventions</h4>
                <ul style='margin-top: 0.5rem;'>
                    {''.join([f'<li>{med.strip()}</li>' for med in meds if med.strip()])}
                </ul>
            </div>""", unsafe_allow_html=True)
            
        # Disclaimer
        st.markdown("""<div style='background-color: #fff3cd; padding: 0.75rem; border-radius: 5px; margin: 1rem 0; border-left: 4px solid #ffc107;'>
            <p style='margin: 0; font-size: 14px;'><strong>Clinical Note:</strong> This differential diagnosis is generated based on the provided symptoms and should be confirmed by a qualified healthcare professional. Treatment should only be initiated after proper clinical evaluation.</p>
        </div>""", unsafe_allow_html=True)