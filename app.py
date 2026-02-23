import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction by Shajid")
st.markdown("Provide the following details to check your heart disease risk:")

# Collect user input
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# When Predict is clicked
if st.button("Predict"):

    # Create a raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
    }

    # Handle categorical one-hot encoding for all expected columns
    for col in expected_columns:
        if col.startswith('Sex_'):
            raw_input[col] = 1 if col == f'Sex_{sex}' else 0
        elif col.startswith('ChestPainType_'):
            raw_input[col] = 1 if col == f'ChestPainType_{chest_pain}' else 0
        elif col.startswith('RestingECG_'):
            raw_input[col] = 1 if col == f'RestingECG_{resting_ecg}' else 0
        elif col.startswith('ExerciseAngina_'):
            raw_input[col] = 1 if col == f'ExerciseAngina_{exercise_angina}' else 0
        elif col.startswith('ST_Slope_'):
            raw_input[col] = 1 if col == f'ST_Slope_{st_slope}' else 0

    # Create input dataframe with correct columns
    input_df = pd.DataFrame([raw_input])

    # Ensure correct column order
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")