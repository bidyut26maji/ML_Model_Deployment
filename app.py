import streamlit as st
import pandas as pd
import pickle  # Changed from joblib to pickle
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

# --- LOAD THE TRAINED MODEL ---
@st.cache_resource
def load_model():
    # Update the path to look for the .pkl file
    model_path = "iris_model.pkl" 
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    else:
        st.error(f"Model file '{model_path}' not found!")
        return None

model = load_model()

# --- UI INTERFACE ---
st.title("🌸 Iris Species Predictor")
st.markdown("""
This app uses a **Logistic Regression** model to predict the species of an Iris flower 
based on its physical measurements.
""")

# Sidebar for user inputs
st.sidebar.header("Input Floral Features")

def get_user_input():
    # Ranges aligned with standard Iris dataset measurements
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)
    
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return features

input_data = get_user_input()

# --- PREDICTION LOGIC ---
if model is not None:
    # Standard labels for the Iris dataset
    target_names = ['Setosa', 'Versicolor', 'Virginica']
    
    if st.button("Predict Species"):
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        species = target_names[prediction[0]]
        
        # Display Results
        st.success(f"### Result: {species}")
        
        # Visualizing probabilities
        st.write("#### Prediction Probabilities:")
        prob_df = pd.DataFrame(probability, columns=target_names)
        st.bar_chart(prob_df.T)
else:
    st.warning("Please upload 'iris_model.pkl' to the directory to enable predictions.")