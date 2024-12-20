import streamlit as st
import requests
import numpy as np

# Streamlit App Title
st.title("Real-Time Classification with Cloud-Deployed Model")

# Instructions
st.write("Enter the feature values below as a list and click 'Classify' to get predictions.")

# Input for features as a list
features_input = st.text_area(
    "Enter features as a comma-separated list (e.g., -1.132, -0.743, -1.167, -0.564, -0.200, -0.369, -0.293)",
    "-1.132, -0.743, -1.167, -0.564, -0.200, -0.369, -0.293"
)

# API Endpoint URL
api_url = "https://fastapi-app-mlproject-bhumika.onrender.com/predict"  # Replace with your cloud FastAPI endpoint

# Button to classify
if st.button("Classify"):
    try:
        # Convert the input string into a list of floats
        features = [float(x.strip()) for x in features_input.split(",")]

        # Prepare data payload
        payload = {"features": features}

        # Send POST request to FastAPI endpoint
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Probability: {result['probability']}")
        else:
            st.error(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        st.error(f"Failed to process input or connect to API. Error: {e}")
