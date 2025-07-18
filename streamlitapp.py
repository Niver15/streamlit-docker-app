#STREAMLIT APP
import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('model/my_model.pkl')

st.title("Disease Case Reduction Predictor (SDG 3)")

npi1 = st.slider('School closing (S1)', 0, 3, 1)
npi2 = st.slider('Workplace closing (S2)', 0, 3, 1)
npi3 = st.slider('Cancel public events (S3)', 0, 2, 1)
npi4 = st.slider('Restrictions on gatherings (S4)', 0, 4, 1)
npi5 = st.slider('Public information campaigns (S5)', 0, 2, 1)

features = np.array([[npi1, npi2, npi3, npi4, npi5]])

if st.button("Predict"):
   prediction = model.predict(features)[0]
   st.write(f"Predicted reduction in new cases next week: {prediction:.2f}")

import os

if not os.path.exists('model/my_model.pkl'):
    st.error("Model file not found at 'model/my_model.pkl'. Please train and save the model first.")
    st.stop()
model = joblib.load('model/my_model.pkl')
st.write("Model loaded successfully!")
