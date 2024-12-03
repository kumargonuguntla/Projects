import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('best_model.pkl')

# App title
st.title("Attorney Involvement Prediction")

# Sidebar for user input
st.sidebar.header("Input Features")

# Function to take user inputs
def user_input_features():
    CASENUM = st.sidebar.number_input("Case Number", min_value=1, step=1)
    CLMSEX = st.sidebar.selectbox("Claimant Gender (1=Male, 0=Female)", [1, 0])
    CLMINSUR = st.sidebar.selectbox("Claimant Insured (1=Yes, 0=No)", [1, 0])
    SEATBELT = st.sidebar.selectbox("Seatbelt Used (1=Yes, 0=No)", [1, 0])
    CLMAGE = st.sidebar.number_input("Claimant Age", min_value=18, step=1)
    LOSS = st.sidebar.number_input("Financial Loss", min_value=0.0, step=0.01)
    Accident_Severity = st.sidebar.selectbox("Accident Severity", ["Minor", "Moderate", "Severe"])
    Claim_Amount_Requested = st.sidebar.number_input("Claim Amount Requested", min_value=0.0, step=0.01)
    Claim_Approval_Status = st.sidebar.selectbox("Claim Approved (1=Yes, 0=No)", [1, 0])
    Settlement_Amount = st.sidebar.number_input("Settlement Amount", min_value=0.0, step=0.01)
    Policy_Type = st.sidebar.selectbox("Policy Type", ["Comprehensive", "Third-Party"])
    Driving_Record = st.sidebar.selectbox("Driving Record", ["Clean", "Minor Offenses", "Major Offenses"])

    # Convert categorical inputs to numerical
    Accident_Severity_map = {"Minor": 0, "Moderate": 1, "Severe": 2}
    Policy_Type_map = {"Comprehensive": 0, "Third-Party": 1}
    Driving_Record_map = {"Clean": 0, "Minor Offenses": 1, "Major Offenses": 2}

    data = {
        "CLMSEX": CLMSEX,
        "CLMINSUR": CLMINSUR,
        "SEATBELT": SEATBELT,
        "CLMAGE": CLMAGE,
        "LOSS": LOSS,
        "Accident_Severity": Accident_Severity_map[Accident_Severity],
        "Claim_Amount_Requested": Claim_Amount_Requested,
        "Claim_Approval_Status": Claim_Approval_Status,
        "Settlement_Amount": Settlement_Amount,
        "Policy_Type": Policy_Type_map[Policy_Type],
        "Driving_Record": Driving_Record_map[Driving_Record]
    }

    return pd.DataFrame([data])

# Collect user inputs
input_df = user_input_features()

# Get the feature names the model was trained on
training_features = model.get_booster().feature_names

# Ensure input_df has the same columns as the training data
input_df = input_df.reindex(columns=training_features, fill_value=0)

# Main panel
st.subheader("User Input Features")
st.write(input_df)

# Scale the input features if required
# Load the scaler if it was saved; otherwise, use the original input_df
try:
    scaler = joblib.load('scaler.pkl')
    scaled_input = scaler.transform(input_df)
except FileNotFoundError:
    scaled_input = input_df

## Make predictions
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    prediction_prob = model.predict_proba(scaled_input)[:, 1]

    st.subheader("Prediction")
    st.write("Attorney Involved" if prediction[0] == 1 else "No Attorney Involved")

    st.subheader("Prediction Probability")
    st.write(f"Probability of Attorney Involvement: {prediction_prob[0]:.2f}")
    st.write("Attorney Involved" if prediction_prob[0] > 0.5 else "No Attorney Involved")