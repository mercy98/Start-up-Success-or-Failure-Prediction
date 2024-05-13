import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# Load the trained SVM model
svm_model = load('svm_model.joblib')

# Function to make predictions (modify according to your model's input requirements)
def make_prediction(input_data):
    # Ensure input_data is correctly preprocessed for your model
    # This may include scaling, reshaping, etc.
    prediction = svm_model.predict(input_data)
    return prediction

def main():
    st.title('Startup Success Prediction')
    st.write('Upload a CSV file containing startup data to predict success:')
    
    # Input parameters
    cat_Industry_Group = str(st.number_input('cat_Industry_Group', min_value=0, step=1))
    cat_funding_rounds = str(st.number_input('cat_funding_rounds', min_value=0, step=10000))
    cat_total_investment = str(st.number_input('cat_total_investment', min_value=0, step=10000))
    cat_round_A = str(st.number_input('cat_round_A', min_value=0, step=1))
    cat_round_B = str(st.number_input('cat_round_B', min_value=0, step=10000))
    cat_round_C = str(st.number_input('cat_round_C', min_value=0, step=10000))

    # Prediction button
    if st.button('Predict'):
        # Make predictions
        input_data = [[cat_Industry_Group, cat_funding_rounds, cat_total_investment, cat_round_A, cat_round_B, cat_round_C]]
        prediction = make_prediction(input_data)
        
        # Convert prediction to "success" or "failure"
        prediction_label = "success" if prediction == 1 else "failure"
        
        # Display prediction
        st.write('Prediction:', prediction_label)

if __name__ == "__main__":
    main()
