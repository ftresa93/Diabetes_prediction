#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import joblib
import numpy as np
# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
st.title("🩺 Diabetes Prediction")
st.write("Enter the feature values below to predict the diagnosis.")
# User Inputs
Glucose = st.number_input("Glucose", min_value=0.0, format="%.4f")
Body_mass_index = st.number_input("BMI", min_value=0.0, format="%.4f")
Age = st.number_input("Age", min_value=0, format="%d")
Insulin = st.number_input("Insulin", min_value=0.0, format="%.4f")
SkinThickness = st.number_input("SkinThickness", min_value=0.0, format="%.4f")
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, format="%.4f")
Pregnancies = st.number_input("Pregnancies", min_value=0, format="%d")

if st.button("Predict Diagnosis"):
    input_data = np.array([[
        Glucose,
        Body_mass_index,
        Age,
        Insulin,
        SkinThickness,
        DiabetesPedigreeFunction,
        Pregnancies,
    ]])
    # Scale the input
    input_scaled = scaler.transform(input_data)
    # Predict
    prediction = model.predict(input_scaled)
    # Display Result
    if  prediction[1] == 1:
        st.error("🔴 Diagnosis: The patient is Diabetic")
    else:
        st.success("🟢 Diagnosis: The patient is not Diabetic")


# In[ ]:




