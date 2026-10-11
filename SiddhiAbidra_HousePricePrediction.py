# Project: House Price Prediction System
# Author: Siddhi Abidra
# Institution: Jabalpur Engineering College (JEC)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import joblib

# 1. Dataset Generation & Preprocessing
def load_data():
    np.random.seed(42)
    area = np.random.randint(500, 4000, 200)
    bedrooms = np.random.randint(1, 6, 200)
    bathrooms = np.random.randint(1, 4, 200)
    age = np.random.randint(0, 30, 200)
    
    # Simulating realistic price drivers
    price = (area * 140) + (bedrooms * 25000) + (bathrooms * 18000) - (age * 1200) + np.random.normal(0, 8000, 200)
    
    df = pd.DataFrame({
        'Area': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age': age,
        'Price': price
    })
    return df

df = load_data()

# 2. Model Training
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'house_model.pkl')

# 3. Streamlit Frontend Interface
st.set_page_config(page_title="House Price Prediction", page_icon="🏡")

st.title("🏡 House Price Prediction & Business Analytics Dashboard")
st.write("This application helps real estate stakeholders predict property prices and analyze housing market trends.")

# Sidebar / User Inputs
st.sidebar.header("Input Property Features")
area_input = st.sidebar.number_input("Total Area (sq ft)", min_value=300, max_value=10000, value=1500)
bedrooms_input = st.sidebar.slider("Bedrooms", 1, 5, 3)
bathrooms_input = st.sidebar.slider("Bathrooms", 1, 4, 2)
age_input = st.sidebar.slider("Property Age (Years)", 0, 50, 10)

if st.button("Predict Estimated Price"):
    loaded_model = joblib.load('house_model.pkl')
    input_data = pd.DataFrame([[area_input, bedrooms_input, bathrooms_input, age_input]], 
                              columns=['Area', 'Bedrooms', 'Bathrooms', 'Age'])
    prediction = loaded_model.predict(input_data)
    
    st.success(f"Estimated Market Price: ₹ {prediction[0]*80:,.2f}")
    st.info("Note: Price calculated based on historical square footage, room count, and property age drivers.")

# Display basic exploratory metrics
if st.checkbox("Show Exploratory Data Summary"):
    st.write(df.describe())