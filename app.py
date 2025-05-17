import streamlit as st
import joblib

# Load the trained model
model = joblib.load("cancer_risk_predictor.pkl")

# App Title
st.title("Cervical Cancer Risk Predictor")

# User Inputs
age = st.number_input("Enter Age:", min_value=18, max_value=100, value=25)
smokes = st.selectbox("Do you smoke?", ["No", "Yes"])

# Format Inputs for Model
input_data = [[age, 1 if smokes == "Yes" else 0]]


import pandas as pd

# Load the trained model
model = joblib.load("cancer_risk_predictor.pkl")

# UI for user input
st.title("Cervical Cancer Risk Predictor")

age = st.number_input("Age", min_value=10, max_value=100)
num_partners = st.number_input("Number of sexual partners", min_value=0, max_value=20)
first_sex = st.number_input("Age at first sexual intercourse", min_value=10, max_value=50)
# Make sure "Smokes" input is correctly assigned
smokes = st.selectbox("Do you smoke?", ["Yes", "No"])  # User selects "Yes" or "No"

# Convert "Yes"/"No" to numerical format
smokes_numeric = 1 if smokes == "Yes" else 0

# Create input DataFrame with correct format
input_data = pd.DataFrame([[age, num_partners, first_sex, smokes_numeric] + [0] * (len(model.feature_names_in_) - 4)], 
                          columns=model.feature_names_in_)

# Make prediction
prediction = model.predict(input_data)

# Show result
st.write(f"Risk Prediction: {'High' if prediction[0] == 1 else 'Low'}")

print("Features seen during training:", model.feature_names_in_)
print("Features provided during prediction:", input_data.columns)
print("Expected Features:", model.feature_names_in_)
print(f"Raw Model Prediction: {model.predict(input_data)}")

import joblib

# Load trained model
model = joblib.load("cancer_risk_predictor.pkl")