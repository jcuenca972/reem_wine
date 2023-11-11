import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load data and models (ensure caching for efficiency)
@st.cache_data
def load_data():
    data = pd.read_csv('wine_fraud.csv')
    return data

df= load_data()


model = joblib.load("winefinalmodel.joblib")

label_encoder = joblib.load('label_encoder.joblib') 

# Prediction function
def predict_wine_type(features):
    features_df = pd.DataFrame(features, index=[0])
    prediction = model.predict(features_df)
    probability = model.predict_proba(features_df).max()
    return prediction, probability

# Streamlit app layout
st.title("Wine Fraud Prediction App")
st.write("This app predicts the **wine type** using **wine features** input via the **side panel**.")

# Display wine image
image = Image.open('wine_image.png')
st.image(image, use_column_width=True)

# Sidebar for user input features
with st.sidebar:
    st.header("Wine Features Input")
    
    fixed_acidity = st.slider('Fixed Acidity', min_value=0.0, max_value=15.0, value=7.0, format="%.2f")
    volatile_acidity = st.slider('Volatile Acidity', min_value=0.0, max_value=2.0, value=0.5, format="%.2f")
    citric_acid = st.slider('Citric Acid', min_value=0.0, max_value=1.0, value=0.25, format="%.2f")
    residual_sugar = st.slider('Residual Sugar', min_value=0.0, max_value=20.0, value=2.0, format="%.2f")
    chlorides = st.slider('Chlorides', min_value=0.0, max_value=0.2, value=0.05, format="%.2f")
    free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', min_value=0, max_value=200, value=30, format="%d")
    total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', min_value=0, max_value=400, value=120, format="%d")
    density = st.slider('Density', min_value=0.0, value=0.99, step=0.0001, format="%.4f")
    pH = st.slider('pH', min_value=0.0, max_value=4.0, value=3.15, format="%.2f")
    sulphates = st.slider('Sulphates', min_value=0.0, max_value=2.0, value=0.5, format="%.2f")
    alcohol = st.slider('Alcohol', min_value=0.0, max_value=20.0, value=10.0, format="%.1f")


    quality_options = label_encoder.classes_
    quality = st.selectbox('Quality', options=quality_options)   

# Button to make prediction
if st.button('Predict Wine Type'):

    encoded_quality = label_encoder.transform([quality])[0]

    features = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol,
        'quality': quality
    }

    prediction, probability = predict_wine_type(features)
    st.write(f'Predicted Wine Type: {"Red" if prediction[0] == 0 else "White"} with a probability of {probability:.2f}')
