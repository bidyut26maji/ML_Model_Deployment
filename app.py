import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = "iris_model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error("❌ 'iris_model.pkl' not found. Run train_model.py first.")
        return None

model = load_model()

# --- TITLE ---
st.title("🌸 Iris Flower Classifier")
st.markdown("Predict Iris species using ML (Logistic Regression + Scaling).")

# --- SIDEBAR INPUT ---
st.sidebar.header("Input Features")

def get_input():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return data

input_data = get_input()

# --- SHOW INPUT ---
st.write("### 📥 Input Data")
st.write(pd.DataFrame(input_data, columns=[
    "Sepal Length", "Sepal Width", "Petal Length", "Petal Width"
]))

# --- PREDICTION ---
if model is not None:
    target_names = ["Setosa", "Versicolor", "Virginica"]

    if st.button("🔍 Predict"):
        try:
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)

            species = target_names[prediction[0]]
            confidence = np.max(probability)

            # RESULT
            st.success(f"🌼 Predicted Species: **{species}**")
            st.info(f"📊 Confidence: {confidence:.2f}")

            # PROBABILITY CHART
            st.write("### 📊 Prediction Probabilities")
            prob_df = pd.DataFrame(probability, columns=target_names)
            st.bar_chart(prob_df.T)

        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.warning("⚠️ Model not loaded. Please check your model file.")
