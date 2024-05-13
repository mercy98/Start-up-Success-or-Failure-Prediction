import streamlit as st
import numpy as np
import pandas as pd  # Add this line to import pandas
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Load your trained model
# If you have already saved your model with joblib, you can load it directly
# model = joblib.load('svm_pipeline.joblib')

# Alternatively, define the model in the app if the model is not saved
def create_model():
    return make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))

model = create_model()

# Sample training data for four features
# You would replace this with your actual training data
X_train = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
Y_train = np.array([0, 1])
model.fit(X_train, Y_train)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        color: #333333;
        font-size: 16px;
    }
    .reportview-container .main .block-container {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description of the app
st.title("SVM Model Deployment for Success or Failure Prediction")
st.write("""
Welcome to the Startup Success Predictor!

This tool helps predict whether a startup is likely to succeed or fail based on certain factors.
          You can enter the details manually or upload a CSV file with the information.
""")

# Collect inputs
st.sidebar.header("Please input your features below")
# Example features (adjust according to your model's features)
total_investment = st.sidebar.number_input("Total Investment", value=0.0)
round_A = st.sidebar.number_input("Round A", value=0.0)
round_B = st.sidebar.number_input("Round B", value=0.0)
industry_group = st.sidebar.number_input("Industry Group", value=0.0)

# Prepare input data for prediction
input_data = np.array([[total_investment, round_A, round_B, industry_group]])
input_data_scaled = model.named_steps['standardscaler'].transform(input_data)

# Predict button
if st.sidebar.button("Predict", key="predict_button"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.decision_function(input_data_scaled)

    # Displaying the prediction
    if prediction[0] == 0:
        st.success("The prediction is: Failure")
        st.write(f"Confidence score: {prediction_proba[0]:.2f}")
    else:
        st.success("The prediction is: Success")
        st.write(f"Confidence score: {prediction_proba[0]:.2f}")

# File uploader for CSV files
st.sidebar.header("Upload CSV File for Prediction")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    # Predict button for uploaded CSV
    if st.sidebar.button("Predict from CSV", key="predict_csv_button"):
        input_data_scaled = model.named_steps['standardscaler'].transform(df)
        predictions = model.predict(input_data_scaled)
        prediction_probas = model.decision_function(input_data_scaled)

        # Displaying the predictions
        st.write("Predictions:")
        for i in range(len(predictions)):
            if predictions[i] == 0:
                st.write(f"Row {i+1}: Failure, Confidence score: {prediction_probas[i]:.2f}")
            else:
                st.write(f"Row {i+1}: Success, Confidence score: {prediction_probas[i]:.2f}")

# You can optionally save your trained model using joblib if needed
# joblib.dump(model, 'svm_pipeline.joblib')
