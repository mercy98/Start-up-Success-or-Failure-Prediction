import streamlit as st
import numpy as np
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

# Title of the app
st.title("SVM Model Deployment for Success or Failure Prediction")

# Collect inputs
st.sidebar.header("Input Features")
# Example features (adjust according to your model's features)
total_investment = st.sidebar.number_input("Total Investment", value=0.0)
round_A = st.sidebar.number_input("Round A", value=0.0)
round_B = st.sidebar.number_input("Round B", value=0.0)
industry_group = st.sidebar.number_input("Industry Group", value=0.0)

# Prepare input data for prediction
input_data = np.array([[total_investment, round_A, round_B, industry_group]])
input_data_scaled = model.named_steps['standardscaler'].transform(input_data)


# Prepare input data for prediction
input_data = vectorizer.transform([comment])

# Predict button
if st.sidebar.button("Submit"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Displaying the prediction
    st.success(f"The predicted class is: {Topicnames_target[prediction[0]]}")
    st.write("Class probabilities:")
    for i, prob in enumerate(prediction_proba[0]):
        st.write(f"{Topicnames_target[i]}: {prob:.2f}")

# # Predict button
# if st.sidebar.button("Predict"):
#     prediction = model.predict(input_data_scaled)
#     prediction_proba = model.decision_function(input_data_scaled)

#     # Displaying the prediction
#     if prediction[0] == 0:
#         st.success("The prediction is: Failure")
#         st.write(f"Confidence score: {prediction_proba[0]:.2f}")
#     else:
#         st.success("The prediction is: Success")
#         st.write(f"Confidence score: {prediction_proba[0]:.2f}")

# # You can optionally save your trained model using joblib if needed
# # joblib.dump(model, 'svm_pipeline.joblib')