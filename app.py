import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('random_forest_diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("🩺 Diabetes Prediction App")

st.write("Enter the following medical details to predict the likelihood of diabetes.")

# Input fields
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction
if st.button("Predict"):
    user_data = np.array([[ glucose, blood_pressure, skin_thickness,
                           insulin, bmi, diabetes_pedigree, age]])
    
    # Scale the input
    user_data_scaled = scaler.transform(user_data)

    # Make prediction
    prediction = model.predict(user_data_scaled)[0]

    if prediction == 1:
        st.error("🔴 The person is likely to have diabetes.")
    else:
        st.success("🟢 The person is unlikely to have diabetes.")
