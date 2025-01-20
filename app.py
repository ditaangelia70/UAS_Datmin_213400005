import streamlit as st
import pickle
import pandas as pd

# Load the trained Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Streamlit app title
st.title("Gender Prediction App")
st.write("This app predicts gender based on facial features using a trained Random Forest model.")

# Input features from the user
forehead_width_cm = st.number_input("Enter forehead width (cm):", min_value=0.0, format="%.2f")
forehead_height_cm = st.number_input("Enter forehead height (cm):", min_value=0.0, format="%.2f")
nose_wide = st.selectbox("Is the nose wide?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
nose_long = st.selectbox("Is the nose long?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
lips_thin = st.selectbox("Are the lips thin?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
distance_nose_to_lip_long = st.selectbox("Is the distance from nose to lip long?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
long_hair = st.selectbox("Does the individual have long hair?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Prediction button
if st.button("Predict Gender"):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'forehead_width_cm': [forehead_width_cm],
        'forehead_height_cm': [forehead_height_cm],
        'nose_wide': [nose_wide],
        'nose_long': [nose_long],
        'lips_thin': [lips_thin],
        'distance_nose_to_lip_long': [distance_nose_to_lip_long],
        'long_hair': [long_hair]
    }, index=[0])

    # Get the feature names from the trained model
    feature_names = rf_model.feature_names_in_

    # Reorder the columns of input_data to match the training data
    input_data = input_data[feature_names]

    # Make the prediction
    prediction = rf_model.predict(input_data)

    # Display the prediction
    if prediction[0] == 1:
        st.success("Predicted Gender: Male")
    else:
        st.success("Predicted Gender: Female")
